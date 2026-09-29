"""Shared pytest setup and tiny deterministic fixture builders.

The project root is added to ``sys.path`` so tests import ``recon`` and
``scripts`` exactly the way the CLI does. The builders below create normalized
rows (the shape ``recon.normalize`` produces) and assemble the
``RowValidationResult`` objects that ``recon.matching`` consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections.abc import Sequence  # noqa: E402

import pandas as pd  # noqa: E402

from recon.config import Config  # noqa: E402
from recon.excel_io import SOURCE_ROW_COLUMN  # noqa: E402
from recon.matching import ReconciliationResult, reconcile  # noqa: E402
from recon.validation import (  # noqa: E402
    QA_DETAILS_COLUMN,
    QA_REASON_COLUMN,
    SOURCE_FILE_COLUMN,
    RowValidationResult,
)

ORDERS_FILE = "orders_export.xlsx"
PAYMENTS_FILE = "payments_export.xlsx"

ORDER_ROW_COLUMNS: tuple[str, ...] = (
    SOURCE_ROW_COLUMN,
    "order_id",
    "customer_id",
    "order_date",
    "currency",
    "order_amount",
    "order_status",
    "sales_channel",
)

PAYMENT_ROW_COLUMNS: tuple[str, ...] = (
    SOURCE_ROW_COLUMN,
    "payment_id",
    "order_id",
    "payment_date",
    "currency",
    "payment_amount",
    "payment_status",
    "payment_method",
)

QUARANTINED_COLUMNS: tuple[str, ...] = (
    SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN,
    QA_REASON_COLUMN,
    QA_DETAILS_COLUMN,
    "order_id",
    "payment_id",
)


def order_row(
    order_id: str,
    amount: float,
    *,
    source_row: int = 2,
    currency: str = "EUR",
    status: str = "COMPLETED",
    customer_id: str = "CUST-00001",
    order_date: str = "2025-01-01",
) -> dict[str, object]:
    """A normalized orders row (upper case currency/status, as normalize emits)."""

    return {
        SOURCE_ROW_COLUMN: source_row,
        "order_id": order_id,
        "customer_id": customer_id,
        "order_date": pd.Timestamp(order_date),
        "currency": currency,
        "order_amount": float(amount),
        "order_status": status,
        "sales_channel": "web",
    }


def payment_row(
    payment_id: str,
    order_id: str,
    amount: float,
    *,
    source_row: int = 2,
    currency: str = "EUR",
    status: str = "SUCCESSFUL",
    method: str = "CARD",
    payment_date: str = "2025-01-02",
) -> dict[str, object]:
    """A normalized payments row."""

    return {
        SOURCE_ROW_COLUMN: source_row,
        "payment_id": payment_id,
        "order_id": order_id,
        "payment_date": pd.Timestamp(payment_date),
        "currency": currency,
        "payment_amount": float(amount),
        "payment_status": status,
        "payment_method": method,
    }


def quarantined_row(
    *,
    source_row: int,
    source_file: str,
    order_id: str = "",
    payment_id: str = "",
    reason: str = "malformed_id",
    details: str = "order_id: malformed_id",
) -> dict[str, object]:
    """A row as ``recon.validation`` records it on the QA Issues sheet."""

    return {
        SOURCE_FILE_COLUMN: source_file,
        SOURCE_ROW_COLUMN: source_row,
        QA_REASON_COLUMN: reason,
        QA_DETAILS_COLUMN: details,
        "order_id": order_id,
        "payment_id": payment_id,
    }


def _frame(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def orders_result(
    rows: Sequence[dict[str, object]],
    *,
    quarantined: Sequence[dict[str, object]] = (),
) -> RowValidationResult:
    return RowValidationResult(
        label="orders",
        source_file=ORDERS_FILE,
        valid=_frame(rows, ORDER_ROW_COLUMNS),
        quarantined=_frame(quarantined, QUARANTINED_COLUMNS),
    )


def payments_result(
    rows: Sequence[dict[str, object]],
    *,
    quarantined: Sequence[dict[str, object]] = (),
) -> RowValidationResult:
    return RowValidationResult(
        label="payments",
        source_file=PAYMENTS_FILE,
        valid=_frame(rows, PAYMENT_ROW_COLUMNS),
        quarantined=_frame(quarantined, QUARANTINED_COLUMNS),
    )


def reconcile_exports(
    orders: Sequence[dict[str, object]],
    payments: Sequence[dict[str, object]],
    *,
    quarantined_orders: Sequence[dict[str, object]] = (),
    quarantined_payments: Sequence[dict[str, object]] = (),
    config: Config | None = None,
) -> ReconciliationResult:
    """Run matching on hand-built validated frames."""

    return reconcile(
        orders_result(orders, quarantined=quarantined_orders),
        payments_result(payments, quarantined=quarantined_payments),
        config=config or Config(),
    )
