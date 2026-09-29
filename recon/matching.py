"""Order-centric reconciliation of validated orders and payments.

Pure DataFrame logic: this module never reads or writes a workbook and never
configures logging. It takes the output of :mod:`recon.validation` and produces
one reviewable frame per check plus the ``All Exceptions`` union.

Matching model (``docs/PLAN.md`` §5)
-----------------------------------
The primary key is ``order_id`` and matching is order-centric with payment
aggregation: every valid payment is kept, grouped by order, and compared with the
order as a whole. A single order can therefore be matched by several payments.

* no valid payment at all -> ``MISSING_PAYMENT``
* ``abs(order_amount - sum(payment_amount)) > tolerance`` -> ``AMOUNT_MISMATCH``
* any payment in another currency -> ``CURRENCY_MISMATCH``
* any payment status not allowed for the order status -> ``STATUS_INCONSISTENCY``
* no failed check -> matched
* valid payment whose order is absent from the whole orders file -> ``MISSING_ORDER``
* valid payment whose order exists *only* as a quarantined row -> input data-quality
  finding, never an ordinary missing order

Findings are cumulative: one order can appear on several tabs at once. Duplicate
detection is independent of matching and always runs on the full valid frames;
duplicate payments stay inside the paid total (they are flagged, not removed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from recon.config import Config
from recon.excel_io import SOURCE_ROW_COLUMN
from recon.models import CHECKS, ExceptionType
from recon.validation import SOURCE_FILE_COLUMN, RowValidationResult

DEFAULT_CONFIG = Config()

EXCEPTION_TYPE_COLUMN = "exception_type"
DETAILS_COLUMN = "details"
DUPLICATE_TYPE_COLUMN = "duplicate_type"
DUPLICATE_KEY_COLUMN = "duplicate_key"
DUPLICATE_GROUP_SIZE_COLUMN = "duplicate_group_size"
IS_PRIMARY_COLUMN = "is_primary"

#: Values of the ``duplicate_type`` column on the Duplicates tab.
DUPLICATE_ORDER_ID = "duplicate_order_id"
DUPLICATE_PAYMENT_ID = "duplicate_payment_id"
DUPLICATE_TRANSACTION = "duplicate_transaction"

DUPLICATE_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    DUPLICATE_TYPE_COLUMN,
    DUPLICATE_KEY_COLUMN,
    DUPLICATE_GROUP_SIZE_COLUMN,
    IS_PRIMARY_COLUMN,
    "order_id",
    "payment_id",
    DETAILS_COLUMN,
)

EXCEPTION_COLUMNS: tuple[str, ...] = (
    EXCEPTION_TYPE_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "payment_id",
    DETAILS_COLUMN,
)

MATCHED_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "customer_id",
    "order_date",
    "currency",
    "order_amount",
    "order_status",
    "sales_channel",
    "payment_count",
    "total_paid",
)

MISSING_PAYMENT_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "customer_id",
    "order_date",
    "currency",
    "order_amount",
    "order_status",
    "sales_channel",
    DETAILS_COLUMN,
)

MISSING_ORDER_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "payment_id",
    "order_id",
    "payment_date",
    "currency",
    "payment_amount",
    "payment_status",
    "payment_method",
    DETAILS_COLUMN,
)

AMOUNT_MISMATCH_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "currency",
    "order_amount",
    "total_paid",
    "difference",
    "payment_count",
    "payment_ids",
    DETAILS_COLUMN,
)

CURRENCY_MISMATCH_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "order_currency",
    "payment_currencies",
    "payment_count",
    DETAILS_COLUMN,
)

STATUS_INCONSISTENCY_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    "order_id",
    "order_status",
    "allowed_payment_statuses",
    "payment_statuses",
    "disallowed_payment_statuses",
    "payment_count",
    DETAILS_COLUMN,
)

#: Sort order of the exception tabs, reused for the union sheet.
_CHECK_ORDER = {str(check.exception_type): index for index, check in enumerate(CHECKS)}
_SORT_KEY_COLUMN = "_exception_order"


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Every reconciliation output, ready for metrics and the report."""

    orders_file: str
    payments_file: str
    matched: pd.DataFrame
    missing_payments: pd.DataFrame
    missing_orders: pd.DataFrame
    amount_mismatches: pd.DataFrame
    currency_mismatches: pd.DataFrame
    status_inconsistencies: pd.DataFrame
    duplicate_orders: pd.DataFrame
    duplicate_payments: pd.DataFrame
    blocked_payments: pd.DataFrame
    exceptions: pd.DataFrame

    @property
    def duplicates(self) -> pd.DataFrame:
        """Both duplicate checks in one frame, for the Duplicates tab."""

        return pd.concat(
            [self.duplicate_orders, self.duplicate_payments], ignore_index=True
        )

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def exception_count(self) -> int:
        """Findings in ``All Exceptions`` (one row per record and failed check)."""

        return len(self.exceptions)

    def findings(self, exception_type: ExceptionType) -> pd.DataFrame:
        """All union rows of one exception type."""

        return self.exceptions.loc[
            self.exceptions[EXCEPTION_TYPE_COLUMN] == str(exception_type)
        ].reset_index(drop=True)

    def count_by_type(self) -> dict[str, int]:
        """Findings per exception type, keyed by the :class:`ExceptionType` label."""

        if self.exceptions.empty:
            return {}
        counts = self.exceptions[EXCEPTION_TYPE_COLUMN].value_counts()
        return {str(label): int(count) for label, count in counts.items()}


def reconcile(
    orders: RowValidationResult,
    payments: RowValidationResult,
    *,
    config: Config = DEFAULT_CONFIG,
) -> ReconciliationResult:
    """Reconcile validated orders and payments (pure function, no I/O)."""

    order_rows = orders.valid.reset_index(drop=True)
    payment_rows = payments.valid.reset_index(drop=True)
    quarantined_orders = orders.quarantined.reset_index(drop=True)
    quarantined_payments = payments.quarantined.reset_index(drop=True)

    duplicate_orders = find_duplicate_orders(
        order_rows, source_file=orders.source_file
    )
    duplicate_payments = find_duplicate_payments(
        payment_rows,
        source_file=payments.source_file,
        fingerprint_columns=config.duplicate_payment_fingerprint_columns,
    )

    # Duplicate order rows are listed on the Duplicates tab but only the first
    # row per order_id takes part in matching: one order, one verdict.
    primary_orders = order_rows.drop_duplicates(
        subset=["order_id"], keep="first"
    ).reset_index(drop=True)
    payment_groups = {
        order_id: group
        for order_id, group in payment_rows.groupby(
            "order_id", sort=False, dropna=False
        )
    }
    allowed_statuses_by_order_status = _allowed_statuses(config)

    matched_rows: list[dict[str, object]] = []
    missing_payment_rows: list[dict[str, object]] = []
    amount_mismatch_rows: list[dict[str, object]] = []
    currency_mismatch_rows: list[dict[str, object]] = []
    status_inconsistency_rows: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []

    for order in primary_orders.itertuples(index=False):
        order_id = str(order.order_id)
        group = payment_groups.get(order.order_id)
        if group is None:
            missing_payment_rows.append(
                {
                    SOURCE_FILE_COLUMN: orders.source_file,
                    SOURCE_ROW_COLUMN: int(order.source_row),
                    "order_id": order.order_id,
                    "customer_id": order.customer_id,
                    "order_date": order.order_date,
                    "currency": order.currency,
                    "order_amount": order.order_amount,
                    "order_status": order.order_status,
                    "sales_channel": order.sales_channel,
                    DETAILS_COLUMN: "order has no valid payment row",
                }
            )
            findings.append(
                _finding(
                    ExceptionType.MISSING_PAYMENT,
                    orders.source_file,
                    order.source_row,
                    order_id,
                    "",
                    "order has no valid payment row",
                )
            )
            continue

        aggregate = _aggregate_payments(group)
        failed = False

        difference = abs(float(order.order_amount) - aggregate.total_paid)
        if difference - config.amount_tolerance > config.float_epsilon:
            failed = True
            details = (
                f"order amount {float(order.order_amount):,.2f} {order.currency} vs "
                f"total paid {aggregate.total_paid:,.2f} "
                f"(difference {difference:,.2f}, tolerance {config.amount_tolerance:,.2f})"
            )
            amount_mismatch_rows.append(
                {
                    SOURCE_FILE_COLUMN: orders.source_file,
                    SOURCE_ROW_COLUMN: int(order.source_row),
                    "order_id": order.order_id,
                    "currency": order.currency,
                    "order_amount": float(order.order_amount),
                    "total_paid": aggregate.total_paid,
                    "difference": difference,
                    "payment_count": aggregate.payment_count,
                    "payment_ids": aggregate.payment_ids,
                    DETAILS_COLUMN: details,
                }
            )
            findings.append(
                _finding(
                    ExceptionType.AMOUNT_MISMATCH,
                    orders.source_file,
                    order.source_row,
                    order_id,
                    "",
                    details,
                )
            )

        unexpected_currencies = [
            currency
            for currency in aggregate.currencies
            if currency != str(order.currency)
        ]
        if unexpected_currencies:
            failed = True
            details = (
                f"payment currency {', '.join(unexpected_currencies)} differs from "
                f"order currency {order.currency}"
            )
            currency_mismatch_rows.append(
                {
                    SOURCE_FILE_COLUMN: orders.source_file,
                    SOURCE_ROW_COLUMN: int(order.source_row),
                    "order_id": order.order_id,
                    "order_currency": order.currency,
                    "payment_currencies": ", ".join(aggregate.currencies),
                    "payment_count": aggregate.payment_count,
                    DETAILS_COLUMN: details,
                }
            )
            findings.append(
                _finding(
                    ExceptionType.CURRENCY_MISMATCH,
                    orders.source_file,
                    order.source_row,
                    order_id,
                    "",
                    details,
                )
            )

        allowed_statuses = allowed_statuses_by_order_status.get(
            _case_fold(order.order_status), ()
        )
        disallowed_statuses = [
            status for status in aggregate.statuses if status not in allowed_statuses
        ]
        if disallowed_statuses:
            failed = True
            details = (
                f"payment status {', '.join(disallowed_statuses)} not allowed for "
                f"order status {order.order_status} "
                f"(allowed: {', '.join(allowed_statuses) or 'none'})"
            )
            status_inconsistency_rows.append(
                {
                    SOURCE_FILE_COLUMN: orders.source_file,
                    SOURCE_ROW_COLUMN: int(order.source_row),
                    "order_id": order.order_id,
                    "order_status": order.order_status,
                    "allowed_payment_statuses": ", ".join(allowed_statuses),
                    "payment_statuses": ", ".join(aggregate.statuses),
                    "disallowed_payment_statuses": ", ".join(disallowed_statuses),
                    "payment_count": aggregate.payment_count,
                    DETAILS_COLUMN: details,
                }
            )
            findings.append(
                _finding(
                    ExceptionType.STATUS_INCONSISTENCY,
                    orders.source_file,
                    order.source_row,
                    order_id,
                    "",
                    details,
                )
            )

        if not failed:
            matched_rows.append(
                {
                    SOURCE_FILE_COLUMN: orders.source_file,
                    SOURCE_ROW_COLUMN: int(order.source_row),
                    "order_id": order.order_id,
                    "customer_id": order.customer_id,
                    "order_date": order.order_date,
                    "currency": order.currency,
                    "order_amount": float(order.order_amount),
                    "order_status": order.order_status,
                    "sales_channel": order.sales_channel,
                    "payment_count": aggregate.payment_count,
                    "total_paid": aggregate.total_paid,
                }
            )

    missing_orders, blocked_payments = _split_unmatched_payments(
        payment_rows, primary_orders, quarantined_orders, payments.source_file
    )
    findings.extend(_missing_order_findings(missing_orders))
    findings.extend(_blocked_payment_findings(blocked_payments, quarantined_orders))
    findings.extend(
        _quarantined_findings(
            quarantined_orders, orders.source_file, "order", ExceptionType.QA_INPUT_ERROR
        )
    )
    findings.extend(
        _quarantined_findings(
            quarantined_payments,
            payments.source_file,
            "payment",
            ExceptionType.QA_INPUT_ERROR,
        )
    )
    findings.extend(_duplicate_findings(duplicate_orders))
    findings.extend(_duplicate_findings(duplicate_payments))

    return ReconciliationResult(
        orders_file=orders.source_file,
        payments_file=payments.source_file,
        matched=_frame(matched_rows, MATCHED_COLUMNS),
        missing_payments=_frame(missing_payment_rows, MISSING_PAYMENT_COLUMNS),
        missing_orders=missing_orders,
        amount_mismatches=_frame(amount_mismatch_rows, AMOUNT_MISMATCH_COLUMNS),
        currency_mismatches=_frame(currency_mismatch_rows, CURRENCY_MISMATCH_COLUMNS),
        status_inconsistencies=_frame(
            status_inconsistency_rows, STATUS_INCONSISTENCY_COLUMNS
        ),
        duplicate_orders=duplicate_orders,
        duplicate_payments=duplicate_payments,
        blocked_payments=blocked_payments,
        exceptions=_exception_union(findings),
    )


# --------------------------------------------------------------------------
# Payment aggregation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PaymentAggregate:
    """Everything matching needs to know about one order's valid payments."""

    total_paid: float
    payment_count: int
    payment_ids: str
    currencies: list[str]
    statuses: list[str]


def _aggregate_payments(group: pd.DataFrame) -> _PaymentAggregate:
    return _PaymentAggregate(
        total_paid=float(group["payment_amount"].sum()),
        payment_count=int(len(group)),
        payment_ids=", ".join(sorted(str(value) for value in group["payment_id"])),
        currencies=sorted({str(value) for value in group["currency"]}),
        statuses=sorted({str(value) for value in group["payment_status"]}),
    )


def _case_fold(value: object) -> str:
    """Compare statuses case-insensitively: config holds lower case, data is upper."""

    return str(value).strip().upper()


def _allowed_statuses(config: Config) -> dict[str, tuple[str, ...]]:
    """Case-folded view of ``allowed_payment_statuses_by_order_status``."""

    return {
        _case_fold(order_status): tuple(
            _case_fold(status) for status in allowed
        )
        for order_status, allowed in config.allowed_payment_statuses_by_order_status.items()
    }


# --------------------------------------------------------------------------
# Missing orders, quarantined orders
# --------------------------------------------------------------------------


def _split_unmatched_payments(
    payments: pd.DataFrame,
    primary_orders: pd.DataFrame,
    quarantined_orders: pd.DataFrame,
    payments_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split payments with no matchable order into missing orders and blocked ones.

    A payment is a missing order only when its ``order_id`` appears nowhere in the
    orders export. When the order row exists but failed input validation, the
    payment cannot be reconciled at all, which is a data-quality finding rather
    than a missing order.
    """

    primary_order_ids = set(primary_orders["order_id"])
    quarantined_order_ids = (
        set(quarantined_orders["order_id"]) if not quarantined_orders.empty else set()
    )
    order_ids_in_file = primary_order_ids | quarantined_order_ids

    missing_mask = ~payments["order_id"].isin(order_ids_in_file)
    blocked_mask = payments["order_id"].isin(quarantined_order_ids) & ~payments[
        "order_id"
    ].isin(primary_order_ids)

    missing_rows = payments.loc[missing_mask].copy()
    missing_rows.insert(0, SOURCE_FILE_COLUMN, payments_file)
    missing_rows[DETAILS_COLUMN] = [
        f"payment references order_id {order_id} which is not present in the orders export"
        for order_id in missing_rows["order_id"].astype(str)
    ]
    if not missing_rows.empty:
        missing_rows = missing_rows.reindex(
            columns=list(MISSING_ORDER_COLUMNS)
        ).reset_index(drop=True)
    else:
        missing_rows = _frame([], MISSING_ORDER_COLUMNS)

    blocked_rows = payments.loc[blocked_mask].copy().reset_index(drop=True)
    blocked_rows.insert(0, SOURCE_FILE_COLUMN, payments_file)
    return missing_rows, blocked_rows


def _missing_order_findings(missing_orders: pd.DataFrame) -> list[dict[str, object]]:
    return [
        _finding(
            ExceptionType.MISSING_ORDER,
            row.source_file,
            row.source_row,
            str(row.order_id),
            str(row.payment_id),
            str(row.details),
        )
        for row in missing_orders.itertuples(index=False)
    ]


def _blocked_payment_findings(
    blocked_payments: pd.DataFrame, quarantined_orders: pd.DataFrame
) -> list[dict[str, object]]:
    if blocked_payments.empty:
        return []

    quarantined_rows = {
        order_id: int(source_row)
        for order_id, source_row in zip(
            quarantined_orders["order_id"], quarantined_orders[SOURCE_ROW_COLUMN]
        )
    }
    findings = []
    for row in blocked_payments.itertuples(index=False):
        order_row = quarantined_rows.get(row.order_id)
        findings.append(
            _finding(
                ExceptionType.QA_INPUT_ERROR,
                row.source_file,
                row.source_row,
                str(row.order_id),
                str(row.payment_id),
                (
                    f"payment cannot be reconciled: order {row.order_id} "
                    f"(source row {order_row}) failed input validation"
                ),
            )
        )
    return findings


def _quarantined_findings(
    quarantined: pd.DataFrame,
    source_file: str,
    record_label: str,
    exception_type: ExceptionType,
) -> list[dict[str, object]]:
    """One input-data-quality finding per quarantined input row."""

    if quarantined.empty:
        return []
    return [
        _finding(
            exception_type,
            source_file,
            row.source_row,
            str(getattr(row, "order_id", "")),
            str(row.payment_id) if record_label == "payment" else "",
            f"{record_label} row failed input validation: {row.qa_details}",
        )
        for row in quarantined.itertuples(index=False)
    ]


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def find_duplicate_orders(
    orders: pd.DataFrame, *, source_file: str
) -> pd.DataFrame:
    """List every row of every duplicated ``order_id`` group."""

    if orders.empty:
        return _frame([], DUPLICATE_COLUMNS)

    positions = orders.reset_index(drop=True)
    group_sizes = positions.groupby("order_id", sort=False, dropna=False)[
        "order_id"
    ].transform("size")
    rows = positions.loc[group_sizes > 1].reset_index(drop=True)
    if rows.empty:
        return _frame([], DUPLICATE_COLUMNS)

    sizes = group_sizes.loc[group_sizes > 1].astype(int).reset_index(drop=True)
    keys = rows["order_id"].astype(str)
    return _frame(
        [
            {
                SOURCE_FILE_COLUMN: source_file,
                SOURCE_ROW_COLUMN: int(source_row),
                DUPLICATE_TYPE_COLUMN: DUPLICATE_ORDER_ID,
                DUPLICATE_KEY_COLUMN: key,
                DUPLICATE_GROUP_SIZE_COLUMN: int(size),
                IS_PRIMARY_COLUMN: bool(is_primary),
                "order_id": key,
                "payment_id": "",
                DETAILS_COLUMN: f"order_id {key} appears {int(size)} times in the orders export",
            }
            for source_row, key, size, is_primary in zip(
                rows[SOURCE_ROW_COLUMN],
                keys,
                sizes,
                ~rows["order_id"].duplicated(keep="first"),
            )
        ],
        DUPLICATE_COLUMNS,
    )


def find_duplicate_payments(
    payments: pd.DataFrame,
    *,
    source_file: str,
    fingerprint_columns: Sequence[str],
) -> pd.DataFrame:
    """List every row of every duplicated payment group.

    Two passes, so the same duplication is not reported twice: first rows that
    repeat a ``payment_id``, then - among the rows still unclaimed - rows that
    repeat a transaction fingerprint under a different ``payment_id``.
    """

    if payments.empty:
        return _frame([], DUPLICATE_COLUMNS)

    positions = payments.reset_index(drop=True)
    records: list[dict[str, object]] = []
    claimed = pd.Series(False, index=positions.index)

    for payment_id, group in positions.groupby("payment_id", sort=False, dropna=False):
        if len(group) < 2:
            continue
        claimed.loc[group.index] = True
        key = str(payment_id)
        records.extend(
            {
                SOURCE_FILE_COLUMN: source_file,
                SOURCE_ROW_COLUMN: int(source_row),
                DUPLICATE_TYPE_COLUMN: DUPLICATE_PAYMENT_ID,
                DUPLICATE_KEY_COLUMN: key,
                DUPLICATE_GROUP_SIZE_COLUMN: len(group),
                IS_PRIMARY_COLUMN: is_primary,
                "order_id": str(order_id),
                "payment_id": key,
                DETAILS_COLUMN: (
                    f"payment_id {key} appears {len(group)} times "
                    "in the payments export"
                ),
            }
            for (source_row, order_id), is_primary in zip(
                group[[SOURCE_ROW_COLUMN, "order_id"]].itertuples(index=False),
                _primary_flags(len(group)),
            )
        )

    remaining = positions.loc[~claimed]
    if not remaining.empty:
        for fingerprint, group in remaining.groupby(
            list(fingerprint_columns), sort=False, dropna=False
        ):
            if len(group) < 2:
                continue
            key = _format_fingerprint(fingerprint_columns, fingerprint)
            records.extend(
                {
                    SOURCE_FILE_COLUMN: source_file,
                    SOURCE_ROW_COLUMN: int(source_row),
                    DUPLICATE_TYPE_COLUMN: DUPLICATE_TRANSACTION,
                    DUPLICATE_KEY_COLUMN: key,
                    DUPLICATE_GROUP_SIZE_COLUMN: len(group),
                    IS_PRIMARY_COLUMN: is_primary,
                    "order_id": str(order_id),
                    "payment_id": str(payment_id),
                    DETAILS_COLUMN: (
                        f"identical transaction recorded {len(group)} times "
                        "under different payment ids"
                    ),
                }
                for (source_row, order_id, payment_id), is_primary in zip(
                    group[
                        [SOURCE_ROW_COLUMN, "order_id", "payment_id"]
                    ].itertuples(index=False),
                    _primary_flags(len(group)),
                )
            )

    return _frame(records, DUPLICATE_COLUMNS)


def _primary_flags(size: int) -> list[bool]:
    """First row of a duplicated group is the primary record."""

    return [True] + [False] * (size - 1)


def _format_fingerprint(
    columns: Sequence[str], fingerprint: Iterable[object]
) -> str:
    return " | ".join(
        f"{column}={value}" for column, value in zip(columns, fingerprint)
    )


def _duplicate_findings(duplicates: pd.DataFrame) -> list[dict[str, object]]:
    """One finding per duplicated row beyond the primary record."""

    if duplicates.empty:
        return []
    extra_rows = duplicates.loc[~duplicates[IS_PRIMARY_COLUMN].astype(bool)]
    return [
        _finding(
            ExceptionType.DUPLICATE_ORDER
            if row.duplicate_type == DUPLICATE_ORDER_ID
            else ExceptionType.DUPLICATE_PAYMENT,
            row.source_file,
            row.source_row,
            str(row.order_id),
            str(row.payment_id),
            str(row.details),
        )
        for row in extra_rows.itertuples(index=False)
    ]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _finding(
    exception_type: ExceptionType,
    source_file: str,
    source_row: object,
    order_id: str,
    payment_id: str,
    details: str,
) -> dict[str, object]:
    return {
        EXCEPTION_TYPE_COLUMN: str(exception_type),
        SOURCE_FILE_COLUMN: source_file,
        SOURCE_ROW_COLUMN: int(source_row),
        "order_id": order_id,
        "payment_id": payment_id,
        DETAILS_COLUMN: details,
    }


def _frame(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    """Build a frame with a fixed column set, even when there are no rows."""

    return pd.DataFrame(list(rows), columns=list(columns))


def _exception_union(findings: Sequence[dict[str, object]]) -> pd.DataFrame:
    """All findings in tab order, then by file and original row."""

    union = _frame(findings, EXCEPTION_COLUMNS)
    if union.empty:
        return union
    return (
        union.assign(
            **{
                _SORT_KEY_COLUMN: union[EXCEPTION_TYPE_COLUMN]
                .map(_CHECK_ORDER)
                .fillna(len(_CHECK_ORDER))
                .astype(int)
            }
        )
        .sort_values(
            [_SORT_KEY_COLUMN, SOURCE_FILE_COLUMN, "source_row"],
            kind="stable",
            ignore_index=True,
        )
        .drop(columns=[_SORT_KEY_COLUMN])
    )
