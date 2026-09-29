"""Contract tests for the deterministic synthetic data generator.

These lock in the promises the rest of the pipeline is built on: same seed gives
the same exports, and the manifest describes what is really in the workbooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recon.config import (
    DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS,
    EXPECTED_CURRENCIES,
    EXPECTED_ORDER_STATUSES,
    EXPECTED_PAYMENT_METHODS,
    EXPECTED_PAYMENT_STATUSES,
    EXPECTED_SALES_CHANNELS,
)
from scripts.generate_sample_data import InjectionCounts, _parse_args, generate_dataset

SMALL_ORDER_COUNT = 800
SMALL_COUNTS = InjectionCounts(
    missing_payment=12,
    missing_order=6,
    amount_mismatch=9,
    currency_mismatch=5,
    status_cancelled_with_successful_payment=3,
    status_completed_with_failed_payment=2,
    duplicate_order=3,
    duplicate_payment_same_id=3,
    duplicate_payment_new_id=2,
    qa_order_rows=10,
    qa_payment_rows=10,
)


def _generate(directory: Path, *, seed: int = 42):
    return generate_dataset(
        order_count=SMALL_ORDER_COUNT,
        seed=seed,
        output_dir=directory,
        counts=SMALL_COUNTS,
    )


def _read_orders(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="orders")


def _read_payments(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="payments")


def _invalid_order_mask(orders: pd.DataFrame) -> pd.Series:
    """Minimal version of the input rules; the full rules arrive with validation.py."""

    amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
    return (
        amounts.isna()
        | (amounts <= 0)
        | pd.to_datetime(orders["order_date"], errors="coerce").isna()
        | ~orders["order_id"].astype(str).str.fullmatch(r"ORD-\d{6}")
        | ~orders["currency"].astype(str).str.upper().isin(EXPECTED_CURRENCIES)
        | ~orders["order_status"].astype(str).str.lower().isin(EXPECTED_ORDER_STATUSES)
        | ~orders["sales_channel"].astype(str).str.lower().isin(EXPECTED_SALES_CHANNELS)
    )


def _invalid_payment_mask(payments: pd.DataFrame) -> pd.Series:
    amounts = pd.to_numeric(payments["payment_amount"], errors="coerce")
    return (
        amounts.isna()
        | (amounts <= 0)
        | pd.to_datetime(payments["payment_date"], errors="coerce").isna()
        | ~payments["payment_id"].astype(str).str.fullmatch(r"PAY-\d{7}")
        | ~payments["order_id"].astype(str).str.fullmatch(r"ORD-\d{6}")
        | ~payments["currency"].astype(str).str.upper().isin(EXPECTED_CURRENCIES)
        | ~payments["payment_status"].astype(str).str.lower().isin(EXPECTED_PAYMENT_STATUSES)
        | ~payments["payment_method"].astype(str).str.lower().isin(EXPECTED_PAYMENT_METHODS)
    )


def test_same_seed_produces_identical_exports(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")

    pd.testing.assert_frame_equal(_read_orders(first.orders_path), _read_orders(second.orders_path))
    pd.testing.assert_frame_equal(
        _read_payments(first.payments_path), _read_payments(second.payments_path)
    )

    # Only the recorded output paths may differ between two runs.
    first_manifest = dict(first.manifest)
    second_manifest = dict(second.manifest)
    first_manifest.pop("outputs")
    second_manifest.pop("outputs")
    assert first_manifest == second_manifest


def test_different_seed_changes_the_data(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first", seed=42)
    other = _generate(tmp_path / "other", seed=7)
    assert not _read_orders(first.orders_path).equals(_read_orders(other.orders_path))


def test_manifest_row_counts_match_the_workbooks(tmp_path: Path) -> None:
    result = _generate(tmp_path / "data")
    orders = _read_orders(result.orders_path)
    payments = _read_payments(result.payments_path)

    assert result.manifest["file_totals"]["order_rows"] == len(orders)
    assert result.manifest["file_totals"]["payment_rows"] == len(payments)
    assert result.manifest["file_totals"]["unique_order_ids"] == orders["order_id"].nunique()


def test_injected_quality_problems_are_really_invalid(tmp_path: Path) -> None:
    result = _generate(tmp_path / "data")
    orders = _read_orders(result.orders_path)
    payments = _read_payments(result.payments_path)
    sources = result.manifest["expected_detections"]["quarantined_rows_by_source"]

    assert int(_invalid_order_mask(orders).sum()) == sources["orders"]
    assert int(_invalid_payment_mask(payments).sum()) == sources["payments"]


def test_injected_reconciliation_anomalies_are_present(tmp_path: Path) -> None:
    result = _generate(tmp_path / "data")
    orders = _read_orders(result.orders_path)
    payments = _read_payments(result.payments_path)
    injected = result.manifest["injected"]
    expected = result.manifest["expected_detections"]

    order_ids = set(orders["order_id"].astype(str))
    payment_order_ids = payments["order_id"].astype(str)
    payments_by_order = {order_id: rows for order_id, rows in payments.groupby(payment_order_ids)}
    order_amounts = dict(
        zip(orders["order_id"].astype(str), pd.to_numeric(orders["order_amount"], errors="coerce"))
    )
    order_currencies = dict(
        zip(orders["order_id"].astype(str), orders["currency"].astype(str).str.upper())
    )

    # Missing payments: the payment rows were removed from the export entirely.
    for order_id in injected["missing_payment_orders"]:
        assert order_id not in payments_by_order

    # Missing orders: payments point at order ids that exist nowhere in the orders file.
    for order_id in expected["orphan_order_ids"]:
        assert order_id not in order_ids
        assert int((payment_order_ids == order_id).sum()) == 1

    # Duplicate orders: the same order row is present twice.
    order_id_counts = orders["order_id"].astype(str).value_counts()
    for order_id in injected["duplicate_order_ids"]:
        assert order_id_counts[order_id] == 2

    # Duplicate payments sharing a payment_id.
    payment_id_counts = payments["payment_id"].astype(str).value_counts()
    for order_id in injected["duplicate_payment_same_id_orders"]:
        rows = payments_by_order[order_id]
        assert rows["payment_id"].nunique() == 1
        assert payment_id_counts[str(rows["payment_id"].iloc[0])] == 2

    # Duplicate payments with a fresh payment_id but an identical transaction fingerprint.
    fingerprint_columns = list(DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS)
    for order_id in injected["duplicate_payment_new_id_orders"]:
        rows = payments_by_order[order_id]
        assert rows["payment_id"].nunique() == 2
        assert len(rows[fingerprint_columns].astype(str).drop_duplicates()) == 1

    # Amount mismatches exceed the configured tolerance.
    for order_id in injected["amount_mismatch_orders"]:
        paid = pd.to_numeric(payments_by_order[order_id]["payment_amount"], errors="coerce").sum()
        assert abs(float(order_amounts[order_id]) - float(paid)) > 0.01

    # Currency mismatches pay in a different currency than the order.
    for order_id in injected["currency_mismatch_orders"]:
        paid_currencies = set(payments_by_order[order_id]["currency"].astype(str).str.upper())
        assert paid_currencies and order_currencies[order_id] not in paid_currencies

    # Status inconsistencies: cancelled paid successfully, completed paid with a failure.
    for order_id in injected["status_inconsistency_orders_cancelled"]:
        statuses = set(payments_by_order[order_id]["payment_status"].astype(str).str.lower())
        assert "successful" in statuses
    for order_id in injected["status_inconsistency_orders_completed"]:
        statuses = set(payments_by_order[order_id]["payment_status"].astype(str).str.lower())
        assert "failed" in statuses


def test_too_few_orders_fails_with_a_clear_message(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not enough eligible orders"):
        generate_dataset(order_count=50, seed=42, output_dir=tmp_path / "data")


def test_order_count_flag_accepts_the_plan_alias() -> None:
    # docs/PLAN.md documents `--orders 7500`; --order-count remains the primary name.
    assert _parse_args(["--orders", "7500", "--seed", "42"]).order_count == 7500
    assert _parse_args(["--order-count", "500"]).order_count == 500
