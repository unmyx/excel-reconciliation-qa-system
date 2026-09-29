"""One test per reconciliation check: tolerance, currency, status, duplicates."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import (
    ORDERS_FILE,
    order_row,
    payment_row,
    quarantined_row,
    reconcile_exports,
)
from recon.config import (
    ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS,
    DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS,
    EXPECTED_PAYMENT_STATUSES,
    Config,
)
from recon.excel_io import SOURCE_ROW_COLUMN
from recon.matching import (
    DUPLICATE_PAYMENT_ID,
    DUPLICATE_TRANSACTION,
    IS_PRIMARY_COLUMN,
)
from recon.models import ExceptionType

STATUS_CASES = [
    (order_status, payment_status)
    for order_status in ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS
    for payment_status in EXPECTED_PAYMENT_STATUSES
]


# --------------------------------------------------------------------------
# Amount tolerance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("difference", [0.0, 0.004, 0.01])
def test_difference_at_or_below_tolerance_is_matched(difference: float) -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [payment_row("PAY-0000001", "ORD-000001", 100.0 - difference)],
    )

    assert result.matched_count == 1
    assert result.amount_mismatches.empty


def test_one_cent_over_tolerance_is_a_mismatch() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [payment_row("PAY-0000001", "ORD-000001", 99.98)],  # 0.02 off
    )

    assert result.matched_count == 0
    assert result.amount_mismatches["difference"].tolist() == [pytest.approx(0.02)]
    assert result.amount_mismatches["payment_ids"].tolist() == ["PAY-0000001"]


def test_tolerance_comes_from_config() -> None:
    tolerant = Config(amount_tolerance=1.0)

    within = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [payment_row("PAY-0000001", "ORD-000001", 99.5)],
        config=tolerant,
    )
    beyond = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [payment_row("PAY-0000001", "ORD-000001", 98.5)],
        config=tolerant,
    )

    assert within.matched_count == 1
    assert beyond.amount_mismatches["difference"].tolist() == [pytest.approx(1.5)]


# --------------------------------------------------------------------------
# Currency
# --------------------------------------------------------------------------


def test_currency_mismatch_is_reported() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, currency="EUR")],
        [payment_row("PAY-0000001", "ORD-000001", 100.0, currency="USD")],
    )

    assert result.matched_count == 0
    assert result.currency_mismatches["order_currency"].tolist() == ["EUR"]
    assert result.currency_mismatches["payment_currencies"].tolist() == ["USD"]
    assert result.amount_mismatches.empty  # amounts agree; only the currency differs


def test_currency_mismatch_is_reported_once_per_order() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, currency="EUR")],
        [
            payment_row("PAY-0000001", "ORD-000001", 60.0, currency="USD", source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 40.0, currency="GBP", source_row=3),
        ],
    )

    assert len(result.currency_mismatches) == 1
    assert result.currency_mismatches["payment_currencies"].tolist() == ["GBP, USD"]
    assert result.currency_mismatches["payment_count"].tolist() == [2]


# --------------------------------------------------------------------------
# Status consistency: every configured rule
# --------------------------------------------------------------------------


def test_status_rule_matrix_is_complete() -> None:
    assert len(STATUS_CASES) == len(ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS) * len(
        EXPECTED_PAYMENT_STATUSES
    )
    assert ("cancelled", "successful") in STATUS_CASES  # cancelled + successful payment
    assert ("completed", "failed") in STATUS_CASES  # paid order + failed payment


@pytest.mark.parametrize("order_status,payment_status", STATUS_CASES)
def test_configured_status_rules(order_status: str, payment_status: str) -> None:
    allowed = {
        status.upper()
        for status in ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS[order_status]
    }
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, status=order_status.upper())],
        [payment_row("PAY-0000001", "ORD-000001", 100.0, status=payment_status.upper())],
    )

    if payment_status.upper() in allowed:
        assert result.matched_count == 1
        assert result.status_inconsistencies.empty
    else:
        assert result.matched_count == 0
        assert len(result.status_inconsistencies) == 1
        assert result.status_inconsistencies["disallowed_payment_statuses"].tolist() == [
            payment_status.upper()
        ]
        assert result.findings(ExceptionType.STATUS_INCONSISTENCY)[
            SOURCE_ROW_COLUMN
        ].tolist() == [2]


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def test_duplicate_order_group_lists_every_row_with_size_and_primary_flag() -> None:
    result = reconcile_exports(
        [
            order_row("ORD-000001", 10.0, source_row=2),
            order_row("ORD-000001", 10.0, source_row=3),
            order_row("ORD-000001", 10.0, source_row=4),
        ],
        [payment_row("PAY-0000001", "ORD-000001", 10.0)],
    )

    assert result.duplicate_orders["duplicate_group_size"].tolist() == [3, 3, 3]
    assert result.duplicate_orders[IS_PRIMARY_COLUMN].tolist() == [True, False, False]
    assert result.duplicate_orders[SOURCE_ROW_COLUMN].tolist() == [2, 3, 4]
    # one finding per duplicated row beyond the primary record
    assert len(result.findings(ExceptionType.DUPLICATE_ORDER)) == 2


def test_duplicate_payment_by_payment_id_is_reported_once() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2),
            payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=3),
        ],
    )

    duplicates = result.duplicate_payments
    assert duplicates["duplicate_type"].tolist() == [DUPLICATE_PAYMENT_ID] * 2
    assert duplicates[IS_PRIMARY_COLUMN].tolist() == [True, False]
    # the identical fingerprint is not reported a second time
    assert len(result.findings(ExceptionType.DUPLICATE_PAYMENT)) == 1


def test_duplicate_payment_by_transaction_fingerprint_is_reported() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 50.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 50.0, source_row=3),
        ],
    )

    duplicates = result.duplicate_payments
    assert duplicates["duplicate_type"].tolist() == [DUPLICATE_TRANSACTION] * 2
    assert duplicates["duplicate_key"].nunique() == 1
    assert "different payment ids" in duplicates["details"].tolist()[0]
    assert len(result.findings(ExceptionType.DUPLICATE_PAYMENT)) == 1
    # both rows stay in the paid total: 50 + 50 still equals the order amount
    assert result.matched_count == 1
    assert result.matched["payment_count"].tolist() == [2]
    assert result.matched["total_paid"].tolist() == [pytest.approx(100.0)]
    assert result.amount_mismatches.empty


def test_mixed_duplicate_group_reports_every_extra_row() -> None:
    # One transaction recorded three times, twice under PAY-0000001 and once under
    # PAY-0000002: two rows are duplicates, so two findings are expected and all
    # three rows must be visible on the Duplicates tab.
    result = reconcile_exports(
        [order_row("ORD-000001", 30.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 10.0, source_row=2),
            payment_row("PAY-0000001", "ORD-000001", 10.0, source_row=3),
            payment_row("PAY-0000002", "ORD-000001", 10.0, source_row=4),
        ],
    )

    duplicates = result.duplicate_payments
    assert set(duplicates["duplicate_type"]) == {
        DUPLICATE_PAYMENT_ID,
        DUPLICATE_TRANSACTION,
    }
    assert duplicates[SOURCE_ROW_COLUMN].tolist() == [2, 3, 4]
    assert int((~duplicates[IS_PRIMARY_COLUMN].astype(bool)).sum()) == 2


@pytest.mark.parametrize("changed_column", list(DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS))
def test_fingerprint_requires_every_configured_column_to_match(
    changed_column: str,
) -> None:
    changed_values = {
        "order_id": "ORD-000002",
        "payment_date": pd.Timestamp("2025-03-03"),
        "payment_amount": 55.0,
        "currency": "USD",
        "payment_method": "PAYPAL",
    }
    altered = payment_row("PAY-0000002", "ORD-000001", 100.0, source_row=3)
    altered[changed_column] = changed_values[changed_column]

    result = reconcile_exports(
        [order_row("ORD-000001", 100.0), order_row("ORD-000002", 100.0)],
        [payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2), altered],
    )

    assert result.duplicate_payments.empty


def test_duplicate_detection_is_independent_of_matching() -> None:
    # The order still matches (100 = 60 + 40) while carrying a duplicate finding
    # for the second payment row, which repeats the first transaction.
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 60.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 60.0, source_row=3),
            payment_row("PAY-0000003", "ORD-000001", 40.0, source_row=4),
        ],
    )

    assert len(result.duplicate_payments) == 2
    assert result.duplicate_payments["payment_id"].tolist() == [
        "PAY-0000001",
        "PAY-0000002",
    ]
    assert result.matched_count == 0  # 160 paid against a 100 order
    assert result.amount_mismatches["payment_count"].tolist() == [3]


# --------------------------------------------------------------------------
# Missing orders vs quarantined orders
# --------------------------------------------------------------------------


def test_payment_for_a_quarantined_order_is_excluded_from_matching() -> None:
    result = reconcile_exports(
        [],
        [payment_row("PAY-0000001", "ORD-000007", 25.0)],
        quarantined_orders=[
            quarantined_row(
                source_row=6,
                source_file=ORDERS_FILE,
                order_id="ORD-000007",
                reason="invalid_date",
                details="order_date: invalid_date",
            )
        ],
    )

    assert result.missing_orders.empty  # the order is in the file, just unusable
    assert result.matched.empty
    assert result.missing_payments.empty
    assert len(result.blocked_payments) == 1
    assert result.count_by_type() == {str(ExceptionType.QA_INPUT_ERROR): 2}


def test_missing_order_and_missing_payment_are_different_checks() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 10.0)],
        [payment_row("PAY-0000002", "ORD-000099", 10.0)],
    )

    assert result.count_by_type() == {
        str(ExceptionType.MISSING_PAYMENT): 1,
        str(ExceptionType.MISSING_ORDER): 1,
    }
    assert result.missing_payments["order_id"].tolist() == ["ORD-000001"]
    assert result.missing_orders["order_id"].tolist() == ["ORD-000099"]
