"""Deterministic synthetic data generator for the reconciliation pipeline.

Writes two clearly synthetic Excel exports (orders and payments) plus a JSON
manifest describing every anomaly that was injected. The manifest exists so
tests and reviewers can verify that the pipeline detects exactly what the
generator produced, including the documented cascades (for example, a payment
whose order row failed input validation cannot be reconciled at all).

Run from the project root:

    python scripts/generate_sample_data.py --order-count 7500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow direct execution: python scripts/generate_sample_data.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recon.config import (  # noqa: E402
    DEFAULT_ORDER_COUNT,
    DEFAULT_ORDERS_PATH,
    DEFAULT_PAYMENTS_PATH,
    DEFAULT_RANDOM_SEED,
    EXPECTED_CURRENCIES,
    EXPECTED_ORDER_STATUSES,
    EXPECTED_PAYMENT_METHODS,
    EXPECTED_PAYMENT_STATUSES,
    EXPECTED_SALES_CHANNELS,
    ORDER_COLUMNS,
    PAYMENT_COLUMNS,
    PROJECT_ROOT,
    resolve_path,
)
from recon.logging_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

ORDERS_SHEET_NAME = "orders"
PAYMENTS_SHEET_NAME = "payments"
MANIFEST_FILE_NAME = "generation_manifest.json"

GENERATION_START = "2024-04-01"
GENERATION_END = "2025-09-30"

CURRENCY_RATE_TO_EUR = {"EUR": 1.0, "USD": 1.08, "GBP": 0.86, "CHF": 0.95}
CURRENCY_WEIGHTS = {"EUR": 0.55, "USD": 0.25, "GBP": 0.12, "CHF": 0.08}
ORDER_STATUS_WEIGHTS = {
    "completed": 0.55,
    "shipped": 0.25,
    "pending": 0.08,
    "cancelled": 0.07,
    "refunded": 0.05,
}
SALES_CHANNEL_WEIGHTS = {
    "web": 0.45,
    "mobile": 0.25,
    "store": 0.15,
    "phone": 0.05,
    "partner": 0.10,
}
PAYMENT_METHOD_WEIGHTS = {
    "card": 0.55,
    "bank_transfer": 0.20,
    "paypal": 0.15,
    "direct_debit": 0.10,
}
# Clean base data: the payment status that a clean order status implies.
PAYMENT_STATUS_BY_ORDER_STATUS = {
    "completed": "successful",
    "shipped": "successful",
    "pending": "pending",
    "cancelled": "failed",
    "refunded": "refunded",
}
SPLIT_PAYMENT_SHARE = 0.06
TRIPLE_PAYMENT_SHARE = 0.01
SPLITTABLE_ORDER_STATUSES = ("completed", "shipped", "pending")

BASE_AMOUNT_LOG_MEAN = 4.6
BASE_AMOUNT_LOG_SIGMA = 0.85
MIN_ORDER_AMOUNT = 5.0
MAX_ORDER_AMOUNT = 25_000.0

# Deliberately wrong values used for the input data-quality injections.
INVALID_DATE_VALUES = ("not-a-date", "2025-13-45", "31/02/2025")
INVALID_NUMERIC_VALUES = ("N/A", "TBD", "12.3.4", "-", "1.2.3")
MALFORMED_ORDER_ID_VALUES = (
    "ORD-12",
    "ORD-1234567",
    "ORD-12A456",
    "ORD000123",
    "ORD-00012Z",
)
MALFORMED_PAYMENT_ID_VALUES = (
    "PAY-123",
    "PAY-12345678",
    "PAY-12A4567",
    "PAY0001234",
    "PAY-00012Z7",
)
# Every value below must stay outside the expected sets even after the pipeline
# normalizes it (trim + upper case), otherwise the row is not actually invalid.
UNEXPECTED_CURRENCIES = ("XYZ", "EURO", "DOLLAR", "PESO", "BTC")
UNEXPECTED_ORDER_STATUSES = ("unknown", "canceled", "COMPLETE", "refund")
UNEXPECTED_SALES_CHANNELS = ("banana", "affiliate", "unknown_channel")
UNEXPECTED_PAYMENT_STATUSES = ("unknown", "success", "cancel", "done")
UNEXPECTED_PAYMENT_METHODS = ("crypto", "cash", "banana", "unknown_method")

# Relative mix of the data-quality problems. With the default 40 rows per file
# these ratios reproduce exactly 10 / 8 / 5 / 5 / 12 (orders) and
# 10 / 7 / 5 / 5 / 13 (payments); smaller runs stay deterministic but rounded.
ORDER_QA_REASON_RATIOS = (10, 8, 5, 5, 12)
PAYMENT_QA_REASON_RATIOS = (10, 7, 5, 5, 13)
MISSING_VALUE_RATIOS = (4, 3, 3)
UNEXPECTED_CATEGORY_RATIOS = (5, 4, 3)
PAYMENT_UNEXPECTED_CATEGORY_RATIOS = (5, 4, 4)


@dataclass(frozen=True, slots=True)
class InjectionCounts:
    """How many anomalies of each type to inject (kept in one auditable place)."""

    missing_payment: int = 120
    missing_order: int = 60
    amount_mismatch: int = 90
    currency_mismatch: int = 50
    status_cancelled_with_successful_payment: int = 25
    status_completed_with_failed_payment: int = 20
    duplicate_order: int = 25
    duplicate_payment_same_id: int = 25
    duplicate_payment_new_id: int = 15
    qa_order_rows: int = 40
    qa_payment_rows: int = 40

    @property
    def quarantined_rows(self) -> int:
        return self.qa_order_rows + self.qa_payment_rows


@dataclass(slots=True)
class InjectionTargets:
    """The exact records selected for each anomaly family (all disjoint)."""

    missing_payment: list[str] = field(default_factory=list)
    amount_mismatch: list[str] = field(default_factory=list)
    currency_mismatch: list[str] = field(default_factory=list)
    status_cancelled: list[str] = field(default_factory=list)
    status_completed: list[str] = field(default_factory=list)
    duplicate_order: list[str] = field(default_factory=list)
    duplicate_payment_same_id: list[str] = field(default_factory=list)
    duplicate_payment_new_id: list[str] = field(default_factory=list)
    qa_orders: list[str] = field(default_factory=list)
    qa_payments: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Paths written plus the manifest that was persisted."""

    orders_path: Path
    payments_path: Path
    manifest_path: Path
    manifest: dict


class _PaymentIdGenerator:
    """Hands out well-formed payment ids that are not already in use."""

    def __init__(self, start: int, used: set[str]) -> None:
        self._next = start
        self._used = used

    def next(self) -> str:
        while True:
            candidate = f"PAY-{self._next:07d}"
            self._next += 1
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate


def _probabilities(weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
    labels = list(weights)
    values = np.array([weights[label] for label in labels], dtype=float)
    return labels, values / values.sum()


def _shuffled(values: Sequence[str], rng: np.random.Generator) -> list[str]:
    order = rng.permutation(len(values))
    return [values[int(position)] for position in order]


def _chunks(values: Sequence[str], sizes: Sequence[int]) -> list[list[str]]:
    """Split ``values`` into consecutive groups of the given sizes."""

    if sum(sizes) != len(values):
        raise ValueError(f"chunk sizes {sizes} do not cover {len(values)} values")
    groups: list[list[str]] = []
    start = 0
    for size in sizes:
        groups.append(list(values[start : start + size]))
        start += size
    return groups


def _cycle(values: Sequence[str], count: int) -> list[str]:
    return [values[index % len(values)] for index in range(count)]


def _proportional_split(total: int, ratios: Sequence[float]) -> list[int]:
    """Split ``total`` across ``ratios`` using largest-remainder rounding."""

    if total <= 0:
        return [0] * len(ratios)
    exact = np.array(ratios, dtype=float)
    exact = exact / exact.sum() * total
    sizes = np.floor(exact).astype(int)
    remainder = total - int(sizes.sum())
    for index in np.argsort(-(exact - sizes), kind="stable")[:remainder]:
        sizes[int(index)] += 1
    return [int(size) for size in sizes]


def _business_day_pool() -> pd.DatetimeIndex:
    """Candidate order dates: Sundays are dropped to look like a real order pattern."""

    days = pd.date_range(GENERATION_START, GENERATION_END, freq="D")
    return days[days.dayofweek != 6]


def _cast_to_object(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Allow mixed values (numbers, text, blanks) in a column, as real exports have."""

    for column in columns:
        frame[column] = frame[column].astype(object)


def _order_ids_of_order_rows(order_count: int) -> list[str]:
    return [f"ORD-{number:06d}" for number in range(1, order_count + 1)]


def _check_value_ranges() -> None:
    """Fail fast if the generator and ``recon.config`` disagree about valid values.

    This guards the subtle bug where an "invalid" value (for example ``"Card"``)
    is actually accepted once the pipeline normalizes it to upper case.
    """

    def normalize(values) -> set[str]:
        return {str(value).upper() for value in values}

    def require_subset(values, expected, label: str) -> None:
        extra = normalize(values) - normalize(expected)
        if extra:
            raise ValueError(f"generator emits {label} values outside the config: {sorted(extra)}")

    require_subset(CURRENCY_WEIGHTS, EXPECTED_CURRENCIES, "currency")
    require_subset(ORDER_STATUS_WEIGHTS, EXPECTED_ORDER_STATUSES, "order status")
    require_subset(SALES_CHANNEL_WEIGHTS, EXPECTED_SALES_CHANNELS, "sales channel")
    require_subset(PAYMENT_METHOD_WEIGHTS, EXPECTED_PAYMENT_METHODS, "payment method")
    require_subset(
        PAYMENT_STATUS_BY_ORDER_STATUS, EXPECTED_ORDER_STATUSES, "order status (payment mapping)"
    )
    require_subset(
        PAYMENT_STATUS_BY_ORDER_STATUS.values(), EXPECTED_PAYMENT_STATUSES, "payment status"
    )

    unexpected = (
        (UNEXPECTED_CURRENCIES, EXPECTED_CURRENCIES, "currency"),
        (UNEXPECTED_ORDER_STATUSES, EXPECTED_ORDER_STATUSES, "order status"),
        (UNEXPECTED_SALES_CHANNELS, EXPECTED_SALES_CHANNELS, "sales channel"),
        (UNEXPECTED_PAYMENT_STATUSES, EXPECTED_PAYMENT_STATUSES, "payment status"),
        (UNEXPECTED_PAYMENT_METHODS, EXPECTED_PAYMENT_METHODS, "payment method"),
    )
    for values, expected, label in unexpected:
        overlap = normalize(values) & normalize(expected)
        if overlap:
            raise ValueError(
                f"injected unexpected {label} values are valid after normalization: {sorted(overlap)}"
            )


# --------------------------------------------------------------------------
# Base data
# --------------------------------------------------------------------------


def _generate_orders(order_count: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build a clean, realistic-looking orders export."""

    customer_pool = max(50, order_count // 5)
    ranks = np.arange(1, customer_pool + 1, dtype=float)
    customer_weights = ranks**-0.9
    customer_weights /= customer_weights.sum()
    customer_numbers = rng.choice(customer_pool, size=order_count, p=customer_weights)

    date_pool = _business_day_pool()
    order_dates = date_pool.take(rng.integers(0, len(date_pool), size=order_count))

    currency_labels, currency_probabilities = _probabilities(CURRENCY_WEIGHTS)
    currencies = rng.choice(currency_labels, size=order_count, p=currency_probabilities)
    rates = np.array([CURRENCY_RATE_TO_EUR[currency] for currency in currencies])
    amounts = np.round(
        np.clip(
            rng.lognormal(BASE_AMOUNT_LOG_MEAN, BASE_AMOUNT_LOG_SIGMA, size=order_count)
            * rates,
            MIN_ORDER_AMOUNT,
            MAX_ORDER_AMOUNT,
        ),
        2,
    )

    status_labels, status_probabilities = _probabilities(ORDER_STATUS_WEIGHTS)
    statuses = rng.choice(status_labels, size=order_count, p=status_probabilities)

    channel_labels, channel_probabilities = _probabilities(SALES_CHANNEL_WEIGHTS)
    channels = rng.choice(channel_labels, size=order_count, p=channel_probabilities)

    return pd.DataFrame(
        {
            "order_id": _order_ids_of_order_rows(order_count),
            "customer_id": [f"CUST-{number + 1:05d}" for number in customer_numbers],
            "order_date": order_dates,
            "currency": currencies,
            "order_amount": amounts,
            "order_status": statuses,
            "sales_channel": channels,
        },
        columns=list(ORDER_COLUMNS),
    )


def _split_amount(total: float, parts: int, rng: np.random.Generator) -> list[float]:
    """Split an order total into ``parts`` amounts that add back up to the total."""

    if parts == 1:
        return [round(total, 2)]
    weights = rng.dirichlet(np.ones(parts))
    amounts = [round(total * weight, 2) for weight in weights[:-1]]
    remainder = round(total - sum(amounts), 2)
    if remainder <= 0:
        return [round(total, 2)]
    amounts.append(remainder)
    return amounts


def _generate_payments(orders: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, int]:
    """Build a clean payments export; most orders are paid once, some in instalments."""

    order_count = len(orders)
    split_draw = rng.random(order_count)
    triple_draw = rng.random(order_count)
    statuses = orders["order_status"].to_numpy()
    splittable = np.isin(statuses, SPLITTABLE_ORDER_STATUSES)

    parts = np.ones(order_count, dtype=int)
    parts[splittable & (split_draw < SPLIT_PAYMENT_SHARE)] = 2
    parts[splittable & (triple_draw < TRIPLE_PAYMENT_SHARE)] = 3

    lag_days = rng.integers(0, 6, size=order_count)
    late_payments = rng.random(order_count) < 0.04
    lag_days[late_payments] = rng.integers(7, 22, size=int(late_payments.sum()))

    method_labels, method_probabilities = _probabilities(PAYMENT_METHOD_WEIGHTS)

    records: list[tuple[object, ...]] = []
    payment_sequence = 0
    for position, order in enumerate(orders.itertuples(index=False)):
        part_count = int(parts[position])
        method = str(rng.choice(method_labels, p=method_probabilities))
        payment_date = order.order_date + pd.Timedelta(days=int(lag_days[position]))
        for amount in _split_amount(float(order.order_amount), part_count, rng):
            payment_sequence += 1
            records.append(
                (
                    f"PAY-{payment_sequence:07d}",
                    order.order_id,
                    payment_date,
                    order.currency,
                    amount,
                    PAYMENT_STATUS_BY_ORDER_STATUS[order.order_status],
                    method,
                )
            )

    payments = pd.DataFrame(records, columns=list(PAYMENT_COLUMNS))
    return payments, payment_sequence + 1


# --------------------------------------------------------------------------
# Anomaly injection
# --------------------------------------------------------------------------


def _pull(pool: Sequence[str], count: int, used: set[str], label: str) -> list[str]:
    """Take ``count`` unused records from the front of a pre-shuffled pool."""

    picked: list[str] = []
    for candidate in pool:
        if len(picked) == count:
            break
        if candidate in used:
            continue
        picked.append(candidate)
        used.add(candidate)
    if len(picked) != count:
        raise ValueError(
            f"not enough eligible orders for {label}: needed {count}, found {len(picked)}. "
            "Increase --order-count (the injected anomaly counts are fixed)."
        )
    return picked


def _allocate_targets(
    orders: pd.DataFrame,
    payments: pd.DataFrame,
    counts: InjectionCounts,
    rng: np.random.Generator,
) -> InjectionTargets:
    """Choose disjoint target records for every anomaly family."""

    order_ids = [str(value) for value in orders["order_id"]]
    status_by_order_id = {
        str(order_id): str(status)
        for order_id, status in zip(orders["order_id"], orders["order_status"])
    }
    payment_counts = payments.groupby("order_id").size()
    single_payment_orders = [str(value) for value in payment_counts[payment_counts == 1].index]

    all_orders = _shuffled(order_ids, rng)
    single_payment = _shuffled(single_payment_orders, rng)
    cancelled_orders = [oid for oid in all_orders if status_by_order_id[oid] == "cancelled"]
    completed_orders = [oid for oid in all_orders if status_by_order_id[oid] == "completed"]

    used: set[str] = set()
    targets = InjectionTargets()
    targets.missing_payment = _pull(
        all_orders, counts.missing_payment, used, "missing payments"
    )
    targets.amount_mismatch = _pull(
        single_payment, counts.amount_mismatch, used, "amount mismatches"
    )
    targets.currency_mismatch = _pull(
        single_payment, counts.currency_mismatch, used, "currency mismatches"
    )
    targets.status_cancelled = _pull(
        cancelled_orders,
        counts.status_cancelled_with_successful_payment,
        used,
        "cancelled orders with successful payments",
    )
    targets.status_completed = _pull(
        completed_orders,
        counts.status_completed_with_failed_payment,
        used,
        "completed orders with failed payments",
    )
    targets.duplicate_order = _pull(
        single_payment, counts.duplicate_order, used, "duplicate orders"
    )
    targets.duplicate_payment_same_id = _pull(
        single_payment,
        counts.duplicate_payment_same_id,
        used,
        "duplicate payments sharing a payment_id",
    )
    targets.duplicate_payment_new_id = _pull(
        single_payment,
        counts.duplicate_payment_new_id,
        used,
        "duplicate payments with a new payment_id",
    )
    targets.qa_payments = _pull(
        single_payment, counts.qa_payment_rows, used, "payment data-quality issues"
    )
    targets.qa_orders = _pull(
        single_payment, counts.qa_order_rows, used, "order data-quality issues"
    )
    return targets


def _inject_missing_payments(payments: pd.DataFrame, target_order_ids: Sequence[str]) -> pd.DataFrame:
    """Remove every payment of the targeted orders: nothing valid is left to reconcile."""

    if not target_order_ids:
        return payments
    return payments[~payments["order_id"].isin(set(target_order_ids))].reset_index(drop=True)


def _inject_amount_mismatches(
    payments: pd.DataFrame, target_order_ids: Sequence[str], rng: np.random.Generator
) -> None:
    mask = payments["order_id"].isin(set(target_order_ids))
    count = int(mask.sum())
    if count == 0:
        return
    magnitudes = np.round(rng.uniform(1.0, 60.0, size=count), 2)
    signs = rng.choice(np.array([-1.0, 1.0]), size=count)
    current = payments.loc[mask, "payment_amount"].to_numpy(dtype=float)
    payments.loc[mask, "payment_amount"] = np.round(
        np.maximum(0.5, current + magnitudes * signs), 2
    )


def _inject_currency_mismatches(
    payments: pd.DataFrame, target_order_ids: Sequence[str], currencies: Sequence[str]
) -> None:
    """Pay in a different (but still valid) currency than the order."""

    mask = payments["order_id"].isin(set(target_order_ids))
    if not mask.any():
        return
    rotation = list(currencies)

    def rotate(currency: str) -> str:
        position = rotation.index(currency) if currency in rotation else 0
        return rotation[(position + 1) % len(rotation)]

    payments.loc[mask, "currency"] = payments.loc[mask, "currency"].map(rotate)


def _inject_status_inconsistencies(
    payments: pd.DataFrame,
    cancelled_order_ids: Sequence[str],
    completed_order_ids: Sequence[str],
) -> None:
    """Cancelled order paid successfully, and completed order whose payment failed."""

    if cancelled_order_ids:
        payments.loc[
            payments["order_id"].isin(set(cancelled_order_ids)), "payment_status"
        ] = "successful"
    if completed_order_ids:
        payments.loc[
            payments["order_id"].isin(set(completed_order_ids)), "payment_status"
        ] = "failed"


def _inject_duplicate_payments(
    payments: pd.DataFrame,
    same_id_order_ids: Sequence[str],
    new_id_order_ids: Sequence[str],
    id_generator: _PaymentIdGenerator,
) -> tuple[pd.DataFrame, list[str]]:
    """Duplicate payment rows, either reusing a payment_id or minting a new one."""

    additions: list[pd.DataFrame] = []
    new_payment_ids: list[str] = []
    for order_ids, reuse_payment_id in (
        (same_id_order_ids, True),
        (new_id_order_ids, False),
    ):
        if not order_ids:
            continue
        duplicate_rows = payments[payments["order_id"].isin(set(order_ids))].copy()
        if not reuse_payment_id:
            duplicate_rows["payment_id"] = [
                id_generator.next() for _ in range(len(duplicate_rows))
            ]
            new_payment_ids.extend(duplicate_rows["payment_id"].tolist())
        additions.append(duplicate_rows)

    if not additions:
        return payments, new_payment_ids
    return pd.concat([payments, *additions], ignore_index=True), new_payment_ids


def _inject_duplicate_orders(
    orders: pd.DataFrame, target_order_ids: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """Repeat whole order rows, which is how duplicate exports actually look."""

    if not target_order_ids:
        return orders, 0
    duplicate_rows = orders[orders["order_id"].isin(set(target_order_ids))].copy()
    return pd.concat([orders, duplicate_rows], ignore_index=True), len(duplicate_rows)


def _build_orphan_payments(
    count: int,
    id_generator: _PaymentIdGenerator,
    rng: np.random.Generator,
    date_pool: pd.DatetimeIndex,
    currency_labels: Sequence[str],
    currency_probabilities: np.ndarray,
    method_labels: Sequence[str],
    method_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Payments that reference an order_id which does not exist in the orders export."""

    if count == 0:
        return pd.DataFrame(columns=list(PAYMENT_COLUMNS))

    currencies = rng.choice(currency_labels, size=count, p=currency_probabilities)
    rates = np.array([CURRENCY_RATE_TO_EUR[currency] for currency in currencies])
    amounts = np.round(
        np.clip(
            rng.lognormal(BASE_AMOUNT_LOG_MEAN, BASE_AMOUNT_LOG_SIGMA, size=count) * rates,
            MIN_ORDER_AMOUNT,
            MAX_ORDER_AMOUNT,
        ),
        2,
    )
    dates = date_pool.take(rng.integers(0, len(date_pool), size=count))
    methods = rng.choice(method_labels, size=count, p=method_probabilities)
    statuses = rng.choice(
        np.array(["successful", "pending"]), size=count, p=np.array([0.85, 0.15])
    )

    return pd.DataFrame(
        {
            "payment_id": [id_generator.next() for _ in range(count)],
            "order_id": [f"ORD-{900_000 + index + 1:06d}" for index in range(count)],
            "payment_date": dates,
            "currency": currencies,
            "payment_amount": amounts,
            "payment_status": statuses,
            "payment_method": methods,
        },
        columns=list(PAYMENT_COLUMNS),
    )


def _assign(
    frame: pd.DataFrame,
    position_by_key: dict[str, int],
    column: str,
    keys: Sequence[str],
    values: Sequence[object],
) -> None:
    for key, value in zip(keys, values):
        frame.at[position_by_key[key], column] = value


def _inject_order_qa_issues(orders: pd.DataFrame, order_ids: Sequence[str]) -> dict[str, int]:
    """Corrupt individual order rows so input validation has rows to quarantine."""

    _cast_to_object(
        orders,
        ("order_id", "order_date", "currency", "order_amount", "order_status", "sales_channel"),
    )
    position = {str(key): int(index) for index, key in orders["order_id"].items()}
    missing, invalid_dates, invalid_numbers, malformed, categories = _chunks(
        order_ids, _proportional_split(len(order_ids), ORDER_QA_REASON_RATIOS)
    )
    date_keys, currency_keys, amount_keys = _chunks(
        missing, _proportional_split(len(missing), MISSING_VALUE_RATIOS)
    )

    _assign(orders, position, "order_date", date_keys, [None] * len(date_keys))
    _assign(orders, position, "currency", currency_keys, [None] * len(currency_keys))
    _assign(orders, position, "order_amount", amount_keys, [None] * len(amount_keys))

    _assign(
        orders, position, "order_date", invalid_dates, _cycle(INVALID_DATE_VALUES, len(invalid_dates))
    )
    _assign(
        orders,
        position,
        "order_amount",
        invalid_numbers,
        _cycle(INVALID_NUMERIC_VALUES, len(invalid_numbers)),
    )
    _assign(
        orders,
        position,
        "order_id",
        malformed,
        _cycle(MALFORMED_ORDER_ID_VALUES, len(malformed)),
    )

    currency_keys, status_keys, channel_keys = _chunks(
        categories, _proportional_split(len(categories), UNEXPECTED_CATEGORY_RATIOS)
    )
    _assign(
        orders,
        position,
        "currency",
        currency_keys,
        _cycle(UNEXPECTED_CURRENCIES, len(currency_keys)),
    )
    _assign(
        orders,
        position,
        "order_status",
        status_keys,
        _cycle(UNEXPECTED_ORDER_STATUSES, len(status_keys)),
    )
    _assign(
        orders,
        position,
        "sales_channel",
        channel_keys,
        _cycle(UNEXPECTED_SALES_CHANNELS, len(channel_keys)),
    )

    return {
        "missing_required_value": len(missing),
        "invalid_date": len(invalid_dates),
        "invalid_numeric": len(invalid_numbers),
        "malformed_id": len(malformed),
        "unexpected_category": len(categories),
    }


def _inject_payment_qa_issues(
    payments: pd.DataFrame, order_ids: Sequence[str]
) -> dict[str, int]:
    """Corrupt the single payment of each targeted order."""

    _cast_to_object(
        payments,
        (
            "payment_id",
            "payment_date",
            "currency",
            "payment_amount",
            "payment_status",
            "payment_method",
        ),
    )
    lookup = {str(key): int(index) for index, key in payments["order_id"].items()}
    position = {str(order_id): lookup[str(order_id)] for order_id in order_ids}

    missing, invalid_dates, invalid_numbers, malformed, categories = _chunks(
        order_ids, _proportional_split(len(order_ids), PAYMENT_QA_REASON_RATIOS)
    )
    date_keys, currency_keys, amount_keys = _chunks(
        missing, _proportional_split(len(missing), MISSING_VALUE_RATIOS)
    )

    _assign(payments, position, "payment_date", date_keys, [None] * len(date_keys))
    _assign(payments, position, "currency", currency_keys, [None] * len(currency_keys))
    _assign(payments, position, "payment_amount", amount_keys, [None] * len(amount_keys))

    _assign(
        payments,
        position,
        "payment_date",
        invalid_dates,
        _cycle(INVALID_DATE_VALUES, len(invalid_dates)),
    )
    _assign(
        payments,
        position,
        "payment_amount",
        invalid_numbers,
        _cycle(INVALID_NUMERIC_VALUES, len(invalid_numbers)),
    )
    _assign(
        payments,
        position,
        "payment_id",
        malformed,
        _cycle(MALFORMED_PAYMENT_ID_VALUES, len(malformed)),
    )

    currency_keys, status_keys, method_keys = _chunks(
        categories, _proportional_split(len(categories), PAYMENT_UNEXPECTED_CATEGORY_RATIOS)
    )
    _assign(
        payments,
        position,
        "currency",
        currency_keys,
        _cycle(UNEXPECTED_CURRENCIES, len(currency_keys)),
    )
    _assign(
        payments,
        position,
        "payment_status",
        status_keys,
        _cycle(UNEXPECTED_PAYMENT_STATUSES, len(status_keys)),
    )
    _assign(
        payments,
        position,
        "payment_method",
        method_keys,
        _cycle(UNEXPECTED_PAYMENT_METHODS, len(method_keys)),
    )

    return {
        "missing_required_value": len(missing),
        "invalid_date": len(invalid_dates),
        "invalid_numeric": len(invalid_numbers),
        "malformed_id": len(malformed),
        "unexpected_category": len(categories),
    }


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _derive_expectations(
    counts: InjectionCounts,
    order_rows: int,
    targets: InjectionTargets,
    order_qa_breakdown: dict[str, int],
    payment_qa_breakdown: dict[str, int],
    duplicate_order_rows: int,
    duplicate_payment_ids: Sequence[str],
    orphan_order_ids: Sequence[str],
    orphan_payment_ids: Sequence[str],
) -> dict:
    """Turn the injection bookkeeping into the exact numbers the pipeline must report."""

    # A QA row whose order_id was corrupted no longer matches any order, so the
    # payment of that order becomes a missing-order finding instead of a blocked one.
    broken_order_ids = order_qa_breakdown.get("malformed_id", 0)
    payments_blocked_by_quarantine = counts.qa_order_rows - broken_order_ids
    orphaned_by_broken_ids = broken_order_ids

    clean_primary_orders = order_rows - counts.qa_order_rows - duplicate_order_rows
    orders_without_valid_payment = counts.missing_payment + counts.qa_payment_rows
    orders_with_reconciliation_exception = (
        counts.amount_mismatch
        + counts.currency_mismatch
        + counts.status_cancelled_with_successful_payment
        + counts.status_completed_with_failed_payment
        + counts.duplicate_payment_same_id
        + counts.duplicate_payment_new_id
    )

    by_reason: dict[str, int] = {}
    for reason in order_qa_breakdown:
        by_reason[reason] = order_qa_breakdown[reason] + payment_qa_breakdown.get(reason, 0)

    return {
        "missing_payments": counts.missing_payment + counts.qa_payment_rows,
        "missing_orders": counts.missing_order + orphaned_by_broken_ids,
        "amount_mismatches": counts.amount_mismatch
        + counts.duplicate_payment_same_id
        + counts.duplicate_payment_new_id,
        "currency_mismatches": counts.currency_mismatch,
        "status_inconsistencies": counts.status_cancelled_with_successful_payment
        + counts.status_completed_with_failed_payment,
        "duplicate_order_findings": duplicate_order_rows,
        "duplicate_payment_findings": len(duplicate_payment_ids)
        + counts.duplicate_payment_same_id,
        "duplicate_payment_same_id_findings": counts.duplicate_payment_same_id,
        "duplicate_payment_new_id_findings": counts.duplicate_payment_new_id,
        "quarantined_rows": counts.quarantined_rows,
        "quarantined_rows_by_source": {
            "orders": counts.qa_order_rows,
            "payments": counts.qa_payment_rows,
        },
        "quarantined_rows_by_reason": by_reason,
        "payments_blocked_by_quarantined_orders": payments_blocked_by_quarantine,
        "input_data_quality_findings": counts.quarantined_rows
        + payments_blocked_by_quarantine,
        "expected_matched_orders": clean_primary_orders
        - orders_without_valid_payment
        - orders_with_reconciliation_exception,
        "orphan_payments_injected": len(orphan_payment_ids),
        "orphan_order_ids": list(orphan_order_ids),
    }


def _build_manifest(
    *,
    seed: int,
    order_count: int,
    counts: InjectionCounts,
    order_rows: int,
    payment_rows: int,
    unique_order_ids: int,
    unique_payment_ids: int,
    targets: InjectionTargets,
    order_qa_breakdown: dict[str, int],
    payment_qa_breakdown: dict[str, int],
    duplicate_order_rows: int,
    duplicate_payment_ids: Sequence[str],
    orphan_order_ids: Sequence[str],
    orphan_payment_ids: Sequence[str],
    orders_path: Path,
    payments_path: Path,
) -> dict:
    expectations = _derive_expectations(
        counts,
        order_rows,
        targets,
        order_qa_breakdown,
        payment_qa_breakdown,
        duplicate_order_rows,
        duplicate_payment_ids,
        orphan_order_ids,
        orphan_payment_ids,
    )
    return {
        "generator": "scripts/generate_sample_data.py",
        "generator_version": "1.0",
        "synthetic": True,
        "deterministic": True,
        "seed": seed,
        "parameters": {
            "order_count_requested": order_count,
            "customer_pool": max(50, order_count // 5),
            "date_range": [GENERATION_START, GENERATION_END],
            "split_payment_share": SPLIT_PAYMENT_SHARE,
            "triple_payment_share": TRIPLE_PAYMENT_SHARE,
        },
        "anomaly_counts_requested": {
            "missing_payment": counts.missing_payment,
            "missing_order": counts.missing_order,
            "amount_mismatch": counts.amount_mismatch,
            "currency_mismatch": counts.currency_mismatch,
            "status_cancelled_with_successful_payment": counts.status_cancelled_with_successful_payment,
            "status_completed_with_failed_payment": counts.status_completed_with_failed_payment,
            "duplicate_order": counts.duplicate_order,
            "duplicate_payment_same_id": counts.duplicate_payment_same_id,
            "duplicate_payment_new_id": counts.duplicate_payment_new_id,
            "qa_order_rows": counts.qa_order_rows,
            "qa_payment_rows": counts.qa_payment_rows,
        },
        "outputs": {
            "orders_file": _display_path(orders_path),
            "payments_file": _display_path(payments_path),
        },
        "file_totals": {
            "order_rows": order_rows,
            "payment_rows": payment_rows,
            "unique_order_ids": unique_order_ids,
            "unique_payment_ids": unique_payment_ids,
        },
        "injected": {
            "missing_payment_orders": list(targets.missing_payment),
            "amount_mismatch_orders": list(targets.amount_mismatch),
            "currency_mismatch_orders": list(targets.currency_mismatch),
            "status_inconsistency_orders_cancelled": list(targets.status_cancelled),
            "status_inconsistency_orders_completed": list(targets.status_completed),
            "duplicate_order_ids": list(targets.duplicate_order),
            "duplicate_payment_same_id_orders": list(targets.duplicate_payment_same_id),
            "duplicate_payment_new_id_orders": list(targets.duplicate_payment_new_id),
            "duplicate_payment_ids_added": list(duplicate_payment_ids),
            "qa_order_ids": list(targets.qa_orders),
            "qa_payment_orders": list(targets.qa_payments),
            "orphan_payment_ids": list(orphan_payment_ids),
        },
        "expected_detections": expectations,
        "notes": [
            "Base records are generated clean, so every exception in the report is attributable "
            "to a deliberately injected anomaly.",
            "Duplicate payment rows are kept inside the order's paid total on purpose, so those "
            "orders also appear as amount mismatches.",
            "Payments whose order row failed input validation cannot be reconciled and are "
            "reported as input data-quality findings, not as missing orders.",
            "A quarantined order row with a corrupted order_id leaves its payment without any "
            "order, which surfaces as a missing-order finding.",
            "Duplicate order rows are excluded from matching (the first row per order_id is the "
            "primary record) and are listed on the Duplicates tab.",
            "Payment rows removed by the missing-payment injection were deleted from the export; "
            "the affected order_ids are listed under injected.missing_payment_orders.",
        ],
    }


def _write_export(frame: pd.DataFrame, path: Path, sheet_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _display_path(path: Path) -> str:
    """Project-relative path when possible, so the manifest stays portable."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def generate_dataset(
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int = DEFAULT_RANDOM_SEED,
    output_dir: Path | str = DEFAULT_ORDERS_PATH.parent,
    counts: InjectionCounts | None = None,
) -> GenerationResult:
    """Generate the exports and the manifest; identical inputs give identical data."""

    injection_counts = counts or InjectionCounts()
    _check_value_ranges()
    directory = resolve_path(output_dir)
    orders_path = directory / DEFAULT_ORDERS_PATH.name
    payments_path = directory / DEFAULT_PAYMENTS_PATH.name
    manifest_path = directory / MANIFEST_FILE_NAME

    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic exports: order_count=%s seed=%s", f"{order_count:,}", seed)

    orders = _generate_orders(order_count, rng)
    payments, next_payment_sequence = _generate_payments(orders, rng)
    logger.info(
        "Base dataset: %s order rows, %s payment rows", f"{len(orders):,}", f"{len(payments):,}"
    )

    targets = _allocate_targets(orders, payments, injection_counts, rng)
    id_generator = _PaymentIdGenerator(
        next_payment_sequence, set(payments["payment_id"].astype(str))
    )

    payments = _inject_missing_payments(payments, targets.missing_payment)
    _inject_amount_mismatches(payments, targets.amount_mismatch, rng)
    _inject_currency_mismatches(payments, targets.currency_mismatch, tuple(CURRENCY_WEIGHTS))
    _inject_status_inconsistencies(
        payments, targets.status_cancelled, targets.status_completed
    )
    payments, duplicate_payment_ids = _inject_duplicate_payments(
        payments,
        targets.duplicate_payment_same_id,
        targets.duplicate_payment_new_id,
        id_generator,
    )
    orders, duplicate_order_rows = _inject_duplicate_orders(orders, targets.duplicate_order)

    currency_labels, currency_probabilities = _probabilities(CURRENCY_WEIGHTS)
    method_labels, method_probabilities = _probabilities(PAYMENT_METHOD_WEIGHTS)
    date_pool = _business_day_pool()
    orphan_payments = _build_orphan_payments(
        injection_counts.missing_order,
        id_generator,
        rng,
        date_pool,
        currency_labels,
        currency_probabilities,
        method_labels,
        method_probabilities,
    )
    orphan_payment_ids = orphan_payments["payment_id"].astype(str).tolist()
    orphan_order_ids = orphan_payments["order_id"].astype(str).tolist()
    if len(orphan_payments):
        payments = pd.concat([payments, orphan_payments], ignore_index=True)

    order_qa_breakdown = _inject_order_qa_issues(orders, targets.qa_orders)
    payment_qa_breakdown = _inject_payment_qa_issues(payments, targets.qa_payments)

    orders = orders.iloc[rng.permutation(len(orders))].reset_index(drop=True)
    payments = payments.iloc[rng.permutation(len(payments))].reset_index(drop=True)

    manifest = _build_manifest(
        seed=seed,
        order_count=order_count,
        counts=injection_counts,
        order_rows=int(len(orders)),
        payment_rows=int(len(payments)),
        unique_order_ids=int(orders["order_id"].nunique(dropna=False)),
        unique_payment_ids=int(payments["payment_id"].nunique(dropna=False)),
        targets=targets,
        order_qa_breakdown=order_qa_breakdown,
        payment_qa_breakdown=payment_qa_breakdown,
        duplicate_order_rows=duplicate_order_rows,
        duplicate_payment_ids=duplicate_payment_ids,
        orphan_order_ids=orphan_order_ids,
        orphan_payment_ids=orphan_payment_ids,
        orders_path=orders_path,
        payments_path=payments_path,
    )

    _write_export(orders, orders_path, ORDERS_SHEET_NAME)
    _write_export(payments, payments_path, PAYMENTS_SHEET_NAME)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote %s", orders_path.as_posix())
    logger.info("Wrote %s", payments_path.as_posix())
    logger.info("Wrote %s", manifest_path.as_posix())
    for name, value in manifest["expected_detections"].items():
        if isinstance(value, (int, float)):
            logger.info("Expected %-38s %s", name, value)

    return GenerationResult(orders_path, payments_path, manifest_path, manifest)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic orders/payments exports with known anomalies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--order-count",
        type=int,
        default=DEFAULT_ORDER_COUNT,
        help="number of synthetic orders to generate",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="random seed")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ORDERS_PATH.parent),
        help="directory for the exports and the manifest",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level)
    result = generate_dataset(
        order_count=args.order_count, seed=args.seed, output_dir=args.output_dir
    )
    expectations = result.manifest["expected_detections"]
    logger.info(
        "Done: %s order rows, %s payment rows, %s expected exception findings.",
        result.manifest["file_totals"]["order_rows"],
        result.manifest["file_totals"]["payment_rows"],
        sum(
            expectations[key]
            for key in (
                "missing_payments",
                "missing_orders",
                "amount_mismatches",
                "currency_mismatches",
                "status_inconsistencies",
                "duplicate_order_findings",
                "duplicate_payment_findings",
                "input_data_quality_findings",
            )
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
