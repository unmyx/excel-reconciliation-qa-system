"""Headline metrics for a reconciliation run.

Definitions follow ``docs/PLAN.md`` exactly:

``total_orders`` / ``total_payments``
    Rows read from each export *before* quarantine, so the rows validation set
    aside are still counted.
``matched_count``
    Valid orders with zero failed reconciliation checks.
``exception_count``
    Distinct records with at least one *reconciliation* exception. Per-type
    counts can sum higher because one record may fail several checks, so this is
    never computed by adding the per-type counts together. Data-quality findings
    are excluded here: they are input problems, not reconciliation exceptions.
``quarantined_rows``
    Input rows set aside by validation, reported separately from reconciliation
    exceptions.
``reconciliation_rate``
    ``matched_count / total_orders``, and ``0.0`` when no orders were read.
``exception_counts``
    Findings per exception type, in report tab order. Input data-quality findings
    are included so this table matches the ``All Exceptions`` tab, and counts may
    overlap.

Everything here is a pure function of the validation and reconciliation results;
no file or workbook access happens in this module.
"""

from __future__ import annotations

import pandas as pd

from recon.excel_io import SOURCE_ROW_COLUMN
from recon.matching import EXCEPTION_TYPE_COLUMN, ReconciliationResult
from recon.models import CHECKS, ExceptionType, SummaryMetrics
from recon.validation import SOURCE_FILE_COLUMN, RowValidationResult

#: Types that describe a reconciliation problem rather than an input problem.
RECONCILIATION_EXCEPTION_TYPES: tuple[ExceptionType, ...] = tuple(
    check.exception_type
    for check in CHECKS
    if check.exception_type is not ExceptionType.QA_INPUT_ERROR
)

_RECONCILIATION_LABELS = frozenset(
    str(exception_type) for exception_type in RECONCILIATION_EXCEPTION_TYPES
)
_TAB_ORDER = {str(check.exception_type): index for index, check in enumerate(CHECKS)}
#: One record is one source row; findings on several checks still count once.
_RECORD_COLUMNS: tuple[str, ...] = (SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN)


def summarize(
    orders: RowValidationResult,
    payments: RowValidationResult,
    result: ReconciliationResult,
) -> SummaryMetrics:
    """Build the Summary tab metrics from validation and matching output."""

    total_orders = orders.total_rows
    total_payments = payments.total_rows
    matched_count = result.matched_count

    return SummaryMetrics(
        total_orders=total_orders,
        total_payments=total_payments,
        matched_count=matched_count,
        exception_count=_distinct_exception_records(result.exceptions),
        quarantined_rows=orders.quarantined_count + payments.quarantined_count,
        reconciliation_rate=_reconciliation_rate(matched_count, total_orders),
        exception_counts=_counts_by_type(result.exceptions),
    )


def _reconciliation_rate(matched_count: int, total_orders: int) -> float:
    """``matched_count / total_orders``; zero orders reconciles to nothing, not a crash."""

    return matched_count / total_orders if total_orders else 0.0


def _distinct_exception_records(exceptions: pd.DataFrame) -> int:
    """How many source records failed at least one reconciliation check.

    The union holds one row per (record, failed check), so a record with an amount
    mismatch *and* a currency mismatch appears twice; this collapses those rows to
    the distinct source records behind them.
    """

    reconciliation = exceptions.loc[
        exceptions[EXCEPTION_TYPE_COLUMN].isin(_RECONCILIATION_LABELS)
    ]
    records = reconciliation[list(_RECORD_COLUMNS)].drop_duplicates()
    return int(len(records))


def _counts_by_type(exceptions: pd.DataFrame) -> dict[str, int]:
    """Findings per exception type, ordered like the report tabs."""

    if exceptions.empty:
        return {}
    counts = {
        str(label): int(count)
        for label, count in exceptions[EXCEPTION_TYPE_COLUMN].value_counts().items()
    }
    return dict(
        sorted(counts.items(), key=lambda item: _TAB_ORDER.get(item[0], len(_TAB_ORDER)))
    )
