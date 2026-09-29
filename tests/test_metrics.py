"""Tests for the Summary tab metrics (``recon.metrics``)."""

from __future__ import annotations

import pytest

from conftest import (
    ORDERS_FILE,
    PAYMENTS_FILE,
    order_row,
    orders_result,
    payment_row,
    payments_result,
    quarantined_row,
)
from recon.matching import reconcile
from recon.metrics import summarize
from recon.models import ExceptionType, SummaryMetrics


def _metrics(
    orders: list[dict[str, object]],
    payments: list[dict[str, object]],
    *,
    quarantined_orders: list[dict[str, object]] | None = None,
    quarantined_payments: list[dict[str, object]] | None = None,
) -> SummaryMetrics:
    checked_orders = orders_result(orders, quarantined=quarantined_orders or [])
    checked_payments = payments_result(payments, quarantined=quarantined_payments or [])
    return summarize(
        checked_orders,
        checked_payments,
        reconcile(checked_orders, checked_payments),
    )


def test_counts_for_a_simple_run() -> None:
    metrics = _metrics(
        [
            order_row("ORD-000001", 100.0, source_row=2),
            order_row("ORD-000002", 50.0, source_row=3),
        ],
        [payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2)],
    )

    assert metrics.total_orders == 2
    assert metrics.total_payments == 1
    assert metrics.matched_count == 1
    assert metrics.exception_count == 1
    assert metrics.quarantined_rows == 0
    assert metrics.reconciliation_rate == 0.5
    assert metrics.exception_counts == {str(ExceptionType.MISSING_PAYMENT): 1}


def test_totals_count_rows_read_before_quarantine() -> None:
    metrics = _metrics(
        [order_row("ORD-000001", 10.0)],
        [],
        quarantined_orders=[
            quarantined_row(source_row=9, source_file=ORDERS_FILE, order_id="ORD-000009")
        ],
        quarantined_payments=[
            quarantined_row(
                source_row=5, source_file=PAYMENTS_FILE, payment_id="PAY-0000005"
            )
        ],
    )

    assert metrics.total_orders == 2  # 1 valid + 1 quarantined
    assert metrics.total_payments == 1
    assert metrics.quarantined_rows == 2


def test_overlapping_exceptions_count_once_per_record() -> None:
    metrics = _metrics(
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

    # one order, three failed checks
    assert metrics.exception_counts == {
        str(ExceptionType.AMOUNT_MISMATCH): 1,
        str(ExceptionType.CURRENCY_MISMATCH): 1,
        str(ExceptionType.STATUS_INCONSISTENCY): 1,
    }
    assert sum(metrics.exception_counts.values()) == 3
    assert metrics.exception_count == 1


def test_exception_count_is_not_the_sum_of_per_type_counts() -> None:
    metrics = _metrics(
        [
            order_row("ORD-000001", 100.0, status="CANCELLED", source_row=2),
            order_row("ORD-000002", 20.0, source_row=3),
        ],
        [
            payment_row(
                "PAY-0000001", "ORD-000001", 90.0, status="SUCCESSFUL", source_row=2
            )
        ],
    )

    # ORD-000001 fails amount + status, ORD-000002 has no payment at all
    assert sum(metrics.exception_counts.values()) == 3
    assert metrics.exception_count == 2
    assert metrics.matched_count == 0


def test_distinct_records_are_counted_by_source_row() -> None:
    metrics = _metrics(
        [
            order_row("ORD-000001", 100.0, source_row=2),
            order_row("ORD-000001", 100.0, source_row=3),  # duplicated order row
        ],
        [payment_row("PAY-0000001", "ORD-000001", 90.0, source_row=2)],
    )

    # the amount mismatch sits on the primary row, the duplicate finding on the
    # repeated row, so two distinct source records are affected
    assert metrics.exception_counts == {
        str(ExceptionType.AMOUNT_MISMATCH): 1,
        str(ExceptionType.DUPLICATE_ORDER): 1,
    }
    assert metrics.exception_count == 2


def test_quarantine_is_reported_separately_from_reconciliation_exceptions() -> None:
    metrics = _metrics(
        [order_row("ORD-000001", 100.0)],
        [
            payment_row("PAY-0000001", "ORD-000001", 100.0, source_row=2),
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
            )
        ],
    )

    assert metrics.matched_count == 1
    assert metrics.exception_count == 0  # the clean order matches; no reconciliation exception
    assert metrics.quarantined_rows == 2  # two input rows were set aside
    # QA findings stay visible in the per-type table: two quarantined rows plus
    # the payment that the quarantined order blocks
    assert metrics.exception_counts == {str(ExceptionType.QA_INPUT_ERROR): 3}


def test_reconciliation_rate_uses_the_total_orders_read() -> None:
    orders = [
        order_row(f"ORD-{number:06d}", 10.0, source_row=number + 1)
        for number in range(1, 5)
    ]
    payments = [
        payment_row(f"PAY-{number:07d}", f"ORD-{number:06d}", 10.0, source_row=number + 1)
        for number in range(1, 4)
    ]

    metrics = _metrics(orders, payments)

    assert metrics.matched_count == 3
    assert metrics.total_orders == 4
    assert metrics.reconciliation_rate == pytest.approx(0.75)

    with_quarantine = _metrics(
        orders,
        payments,
        quarantined_orders=[
            quarantined_row(source_row=99, source_file=ORDERS_FILE, order_id="ORD-000099")
        ],
    )
    assert with_quarantine.reconciliation_rate == pytest.approx(3 / 5)


def test_rate_is_zero_when_no_orders_are_read() -> None:
    metrics = _metrics([], [])

    assert metrics.total_orders == 0
    assert metrics.total_payments == 0
    assert metrics.matched_count == 0
    assert metrics.exception_count == 0
    assert metrics.reconciliation_rate == 0.0
    assert metrics.exception_counts == {}


def test_types_without_findings_are_omitted() -> None:
    metrics = _metrics(
        [order_row("ORD-000001", 10.0)],
        [payment_row("PAY-0000001", "ORD-000001", 10.0)],
    )

    assert metrics.exception_counts == {}
    assert metrics.exception_count == 0
    assert metrics.matched_count == 1
    assert metrics.reconciliation_rate == 1.0


def test_exception_counts_follow_the_report_tab_order() -> None:
    metrics = _metrics(
        [
            order_row("ORD-000001", 100.0, source_row=2),
            order_row("ORD-000002", 20.0, source_row=3),
        ],
        [payment_row("PAY-0000001", "ORD-000001", 90.0, source_row=2)],
    )

    assert list(metrics.exception_counts) == [
        str(ExceptionType.MISSING_PAYMENT),
        str(ExceptionType.AMOUNT_MISMATCH),
    ]


def test_metrics_are_deterministic() -> None:
    orders = [order_row("ORD-000001", 100.0, status="CANCELLED")]
    payments = [payment_row("PAY-0000001", "ORD-000001", 90.0, currency="USD")]

    first = _metrics(orders, payments)
    second = _metrics(orders, payments)

    assert first == second
    assert list(first.exception_counts) == list(second.exception_counts)


def test_matched_count_agrees_with_the_reconciliation_result() -> None:
    checked_orders = orders_result([order_row("ORD-000001", 10.0)])
    checked_payments = payments_result(
        [payment_row("PAY-0000001", "ORD-000001", 10.0)]
    )
    result = reconcile(checked_orders, checked_payments)

    metrics = summarize(checked_orders, checked_payments, result)

    assert metrics.matched_count == result.matched_count == 1
    assert metrics.total_orders == checked_orders.total_rows
    assert metrics.total_payments == checked_payments.total_rows
    assert metrics.quarantined_rows == (
        checked_orders.quarantined_count + checked_payments.quarantined_count
    )
