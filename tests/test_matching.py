"""End-to-end tests for order-centric reconciliation (``recon.matching``)."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import (
    ORDERS_FILE,
    PAYMENTS_FILE,
    order_row,
    payment_row,
    quarantined_row,
    reconcile_exports,
)
from recon.excel_io import SOURCE_ROW_COLUMN
from recon.matching import (
    DETAILS_COLUMN,
    EXCEPTION_TYPE_COLUMN,
    IS_PRIMARY_COLUMN,
    SOURCE_FILE_COLUMN,
    find_duplicate_orders,
)
from recon.models import CHECKS, ExceptionType


def test_one_to_one_match() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, source_row=2)],
        [payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2)],
    )

    assert result.matched_count == 1
    assert result.matched["order_id"].tolist() == ["ORD-000001"]
    assert result.matched["payment_count"].tolist() == [1]
    assert result.matched["total_paid"].tolist() == [100.0]
    assert result.matched[SOURCE_ROW_COLUMN].tolist() == [2]
    assert result.matched[SOURCE_FILE_COLUMN].tolist() == [ORDERS_FILE]
    assert result.exceptions.empty
    assert result.count_by_type() == {}


def test_split_payments_aggregate_to_the_order_total() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 60.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 40.0, source_row=3),
        ],
    )

    assert result.matched_count == 1
    assert result.matched["payment_count"].tolist() == [2]
    assert result.matched["total_paid"].tolist() == pytest.approx([100.0])
    assert result.amount_mismatches.empty
    assert result.exceptions.empty


def test_split_payments_over_tolerance_are_an_amount_mismatch() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 60.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 30.0, source_row=3),
        ],
    )

    assert result.matched_count == 0
    assert result.amount_mismatches["difference"].tolist() == [pytest.approx(10.0)]
    assert result.amount_mismatches["payment_count"].tolist() == [2]
    assert result.amount_mismatches["payment_ids"].tolist() == [
        "PAY-0000001, PAY-0000002"
    ]


def test_missing_payment_is_reported() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, source_row=5)],
        [],
    )

    assert result.matched_count == 0
    assert result.missing_payments[SOURCE_ROW_COLUMN].tolist() == [5]
    assert result.missing_payments[DETAILS_COLUMN].tolist() == [
        "order has no valid payment row"
    ]
    assert result.findings(ExceptionType.MISSING_PAYMENT)[SOURCE_ROW_COLUMN].tolist() == [5]


def test_missing_order_is_reported() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000099", 10.0, source_row=9),
        ],
    )

    assert result.matched_count == 1
    assert result.missing_orders["payment_id"].tolist() == ["PAY-0000002"]
    assert result.missing_orders[SOURCE_ROW_COLUMN].tolist() == [9]
    assert result.missing_orders[SOURCE_FILE_COLUMN].tolist() == [PAYMENTS_FILE]
    details = result.missing_orders[DETAILS_COLUMN].tolist()[0]
    assert "ORD-000099" in details and "not present" in details


def test_duplicate_order_rows_are_listed_but_matched_once() -> None:
    result = reconcile_exports(
        [
            order_row("ORD-000001", 100.0, source_row=2),
            order_row("ORD-000001", 100.0, source_row=3),
        ],
        [payment_row("PAY-0000001", "ORD-000001", 100.0)],
    )

    assert result.matched_count == 1  # one order, one verdict
    assert len(result.duplicate_orders) == 2  # every row of the group is listed
    assert result.duplicate_orders[IS_PRIMARY_COLUMN].tolist() == [True, False]
    assert result.duplicate_orders["duplicate_group_size"].tolist() == [2, 2]
    assert result.findings(ExceptionType.DUPLICATE_ORDER)[SOURCE_ROW_COLUMN].tolist() == [3]


def test_duplicate_payments_stay_in_the_paid_total() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2),
            payment_row("PAY-0000002", "ORD-000001", 100.0, source_row=3),
        ],
    )

    # The duplicate is flagged but not removed: the order is now over-paid.
    assert result.matched_count == 0
    assert result.amount_mismatches["total_paid"].tolist() == [pytest.approx(200.0)]
    assert result.amount_mismatches["payment_count"].tolist() == [2]
    assert result.findings(ExceptionType.DUPLICATE_PAYMENT)[SOURCE_ROW_COLUMN].tolist() == [3]


def test_payment_for_a_quarantined_order_is_not_a_missing_order() -> None:
    result = reconcile_exports(
        [],
        [payment_row("PAY-0000001", "ORD-000002", 50.0, source_row=4)],
        quarantined_orders=[
            quarantined_row(
                source_row=7,
                source_file=ORDERS_FILE,
                order_id="ORD-000002",
                details="order_date: invalid_date",
            )
        ],
    )

    assert result.missing_orders.empty
    assert len(result.blocked_payments) == 1
    assert result.blocked_payments["payment_id"].tolist() == ["PAY-0000001"]

    quality_findings = result.findings(ExceptionType.QA_INPUT_ERROR)
    # the quarantined order row itself, then the payment it blocks
    assert quality_findings["payment_id"].tolist() == ["", "PAY-0000001"]
    details = quality_findings[DETAILS_COLUMN].tolist()[1]
    assert "cannot be reconciled" in details and "source row 7" in details


def test_payment_for_an_absent_order_is_a_missing_order() -> None:
    result = reconcile_exports(
        [],
        [payment_row("PAY-0000001", "ORD-000099", 50.0, source_row=4)],
        quarantined_orders=[
            quarantined_row(source_row=7, source_file=ORDERS_FILE, order_id="ORD-000002")
        ],
    )

    assert result.blocked_payments.empty
    assert result.missing_orders["order_id"].tolist() == ["ORD-000099"]


def test_exceptions_are_cumulative_for_one_record() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0, currency="EUR", status="CANCELLED")],
        [
            payment_row(
                "PAY-0000001",
                "ORD-000001",
                90.0,
                currency="USD",
                status="SUCCESSFUL",
            )
        ],
    )

    assert result.matched_count == 0
    assert len(result.amount_mismatches) == 1
    assert len(result.currency_mismatches) == 1
    assert len(result.status_inconsistencies) == 1
    assert len(result.exceptions) == 3
    assert set(result.exceptions[EXCEPTION_TYPE_COLUMN]) == {
        str(ExceptionType.AMOUNT_MISMATCH),
        str(ExceptionType.CURRENCY_MISMATCH),
        str(ExceptionType.STATUS_INCONSISTENCY),
    }
    # every finding points at the same order row
    assert result.exceptions[SOURCE_ROW_COLUMN].tolist() == [2, 2, 2]


def test_qa_findings_cover_quarantined_rows_and_blocked_payments() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0),
            payment_row("PAY-0000002", "ORD-000003", 10.0, source_row=5),
        ],
        quarantined_orders=[
            quarantined_row(source_row=3, source_file=ORDERS_FILE, order_id="ORD-000003")
        ],
        quarantined_payments=[
            quarantined_row(
                source_row=9,
                source_file=PAYMENTS_FILE,
                order_id="ORD-000001",
                payment_id="PAY-0000009",
                details="payment_amount: invalid_numeric",
            )
        ],
    )

    quality = result.findings(ExceptionType.QA_INPUT_ERROR)
    assert len(quality) == 3  # 1 quarantined order + 1 quarantined payment + 1 blocked payment
    assert quality[SOURCE_FILE_COLUMN].tolist() == [
        ORDERS_FILE,
        PAYMENTS_FILE,
        PAYMENTS_FILE,
    ]
    assert quality[SOURCE_ROW_COLUMN].tolist() == [3, 5, 9]
    details = quality[DETAILS_COLUMN].tolist()
    assert "order row failed input validation" in details[0]
    assert "cannot be reconciled" in details[1]
    assert "payment row failed input validation" in details[2]


def test_exception_union_follows_the_check_order_and_is_deterministic() -> None:
    orders = [order_row("ORD-000001", 100.0, source_row=5), order_row("ORD-000002", 50.0)]
    payments = [payment_row("PAY-0000001", "ORD-000002", 40.0, source_row=2)]
    quarantined = [
        quarantined_row(source_row=3, source_file=ORDERS_FILE, order_id="ORD-000009")
    ]

    first = reconcile_exports(orders, payments, quarantined_orders=quarantined)
    second = reconcile_exports(orders, payments, quarantined_orders=quarantined)

    tab_order = {str(check.exception_type): index for index, check in enumerate(CHECKS)}
    types = first.exceptions[EXCEPTION_TYPE_COLUMN].tolist()
    assert types == sorted(types, key=lambda label: tab_order[label])
    assert types == [
        str(ExceptionType.MISSING_PAYMENT),
        str(ExceptionType.AMOUNT_MISMATCH),
        str(ExceptionType.QA_INPUT_ERROR),
    ]

    pd.testing.assert_frame_equal(first.exceptions, second.exceptions)
    pd.testing.assert_frame_equal(first.matched, second.matched)
    pd.testing.assert_frame_equal(first.duplicates, second.duplicates)


def test_matched_rows_keep_the_input_order() -> None:
    result = reconcile_exports(
        [
            order_row("ORD-000002", 50.0, source_row=9),
            order_row("ORD-000001", 100.0, source_row=4),
        ],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0),
            payment_row("PAY-0000002", "ORD-000002", 50.0),
        ],
    )

    assert result.matched["order_id"].tolist() == ["ORD-000002", "ORD-000001"]
    assert result.matched[SOURCE_ROW_COLUMN].tolist() == [9, 4]


def test_result_helpers_summarise_by_exception_type() -> None:
    result = reconcile_exports(
        [order_row("ORD-000001", 100.0), order_row("ORD-000002", 20.0)],
        [payment_row("PAY-0000001", "ORD-000001", 90.0)],
    )

    assert result.matched_count == 0
    assert result.count_by_type() == {
        str(ExceptionType.MISSING_PAYMENT): 1,
        str(ExceptionType.AMOUNT_MISMATCH): 1,
    }
    assert len(result.findings(ExceptionType.AMOUNT_MISMATCH)) == 1
    assert result.findings(ExceptionType.DUPLICATE_PAYMENT).empty
    assert result.exception_count == 2


def test_empty_exports_produce_empty_frames_with_expected_columns() -> None:
    result = reconcile_exports([], [])

    assert result.matched.empty
    assert result.exceptions.empty
    assert list(result.matched.columns)[:3] == [
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_COLUMN,
        "order_id",
    ]
    assert list(result.exceptions.columns) == [
        EXCEPTION_TYPE_COLUMN,
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_COLUMN,
        "order_id",
        "payment_id",
        DETAILS_COLUMN,
    ]
    assert result.duplicates.empty
    assert result.count_by_type() == {}


def test_duplicate_ordering_follows_the_input_file() -> None:
    frame = pd.DataFrame(
        [
            order_row("ORD-000001", 10.0, source_row=4),
            order_row("ORD-000001", 10.0, source_row=2),
        ]
    )

    duplicates = find_duplicate_orders(frame, source_file=ORDERS_FILE)

    assert duplicates[SOURCE_ROW_COLUMN].tolist() == [4, 2]
    assert duplicates[IS_PRIMARY_COLUMN].tolist() == [True, False]
