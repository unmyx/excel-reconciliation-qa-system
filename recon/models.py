"""Shared enums and small dataclasses describing reconciliation findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExceptionType(StrEnum):
    """Categories of reconciliation and data-quality findings."""

    MISSING_PAYMENT = "Missing Payment"
    MISSING_ORDER = "Missing Order"
    AMOUNT_MISMATCH = "Amount Mismatch"
    DUPLICATE_ORDER = "Duplicate Order"
    DUPLICATE_PAYMENT = "Duplicate Payment"
    CURRENCY_MISMATCH = "Currency Mismatch"
    STATUS_INCONSISTENCY = "Status Inconsistency"
    QA_INPUT_ERROR = "Input Data Quality Error"


class QAReason(StrEnum):
    """Why a single input row was quarantined by validation."""

    MISSING_REQUIRED_VALUE = "missing_required_value"
    INVALID_DATE = "invalid_date"
    INVALID_NUMERIC = "invalid_numeric"
    MALFORMED_ID = "malformed_id"
    UNEXPECTED_CATEGORY = "unexpected_category"
    NON_POSITIVE_AMOUNT = "non_positive_amount"


@dataclass(frozen=True, slots=True)
class Check:
    """One reconciliation check, its report tab, and its reviewer-facing meaning."""

    exception_type: ExceptionType
    sheet_name: str
    description: str

    @property
    def key(self) -> str:
        """Stable machine-readable key, e.g. ``amount_mismatch``."""

        return self.exception_type.name.lower()

    @property
    def label(self) -> str:
        """Reviewer-facing label, e.g. ``Amount Mismatch``."""

        return str(self.exception_type)


# Order of this tuple is the order of the exception tabs in the report.
CHECKS: tuple[Check, ...] = (
    Check(
        ExceptionType.MISSING_PAYMENT,
        "Missing Payments",
        "Order has no valid payment transaction.",
    ),
    Check(
        ExceptionType.MISSING_ORDER,
        "Missing Orders",
        "Payment references an order_id that does not exist in the orders export.",
    ),
    Check(
        ExceptionType.AMOUNT_MISMATCH,
        "Amount Mismatches",
        "Order amount and total paid amount differ by more than the tolerance.",
    ),
    Check(
        ExceptionType.DUPLICATE_ORDER,
        "Duplicates",
        "Same order_id appears more than once in the orders export.",
    ),
    Check(
        ExceptionType.DUPLICATE_PAYMENT,
        "Duplicates",
        "Same payment_id or identical payment transaction appears more than once.",
    ),
    Check(
        ExceptionType.CURRENCY_MISMATCH,
        "Currency Mismatches",
        "Payment currency differs from the currency of its order.",
    ),
    Check(
        ExceptionType.STATUS_INCONSISTENCY,
        "Status Inconsistencies",
        "Payment status is not a valid combination for the order status.",
    ),
    Check(
        ExceptionType.QA_INPUT_ERROR,
        "QA Issues",
        "Row failed input validation and could not be reconciled.",
    ),
)

CHECK_BY_TYPE: dict[ExceptionType, Check] = {check.exception_type: check for check in CHECKS}


@dataclass(frozen=True, slots=True)
class SummaryMetrics:
    """Headline numbers shown on the Summary tab."""

    total_orders: int
    total_payments: int
    matched_count: int
    exception_count: int
    quarantined_rows: int
    reconciliation_rate: float
    exception_counts: dict[str, int] = field(default_factory=dict)
