"""Central configuration for the reconciliation pipeline.

Every tunable value lives here so no other module hard-codes business rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORDER_COLUMNS: tuple[str, ...] = (
    "order_id",
    "customer_id",
    "order_date",
    "currency",
    "order_amount",
    "order_status",
    "sales_channel",
)

PAYMENT_COLUMNS: tuple[str, ...] = (
    "payment_id",
    "order_id",
    "payment_date",
    "currency",
    "payment_amount",
    "payment_status",
    "payment_method",
)

DEFAULT_ORDERS_PATH = Path("data/orders_export.xlsx")
DEFAULT_PAYMENTS_PATH = Path("data/payments_export.xlsx")
DEFAULT_OUTPUT_PATH = Path("outputs/reconciliation_report.xlsx")
DEFAULT_MANIFEST_PATH = Path("data/generation_manifest.json")

DEFAULT_AMOUNT_TOLERANCE = 0.01
DEFAULT_FLOAT_EPSILON = 1e-9
DEFAULT_RANDOM_SEED = 42
DEFAULT_ORDER_COUNT = 7_500

EXPECTED_ORDER_STATUSES: tuple[str, ...] = (
    "pending",
    "completed",
    "shipped",
    "cancelled",
    "refunded",
)

EXPECTED_PAYMENT_STATUSES: tuple[str, ...] = (
    "pending",
    "successful",
    "failed",
    "refunded",
)

EXPECTED_CURRENCIES: tuple[str, ...] = ("EUR", "USD", "GBP", "CHF")

EXPECTED_SALES_CHANNELS: tuple[str, ...] = (
    "web",
    "mobile",
    "store",
    "phone",
    "partner",
)

EXPECTED_PAYMENT_METHODS: tuple[str, ...] = (
    "card",
    "bank_transfer",
    "paypal",
    "direct_debit",
)

# Status logic: a payment status is only consistent with the order status if it
# appears in the corresponding tuple below. This is what turns
# "cancelled order + successful payment" into a reported inconsistency.
ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS: dict[str, tuple[str, ...]] = {
    "pending": ("pending", "failed"),
    "completed": ("successful",),
    "shipped": ("successful",),
    "cancelled": ("failed", "refunded"),
    "refunded": ("refunded", "failed"),
}

# Two payment rows are considered the same transaction when these columns match.
DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS: tuple[str, ...] = (
    "order_id",
    "payment_date",
    "payment_amount",
    "currency",
    "payment_method",
)

ORDER_ID_PATTERN = r"^ORD-\d{6}$"
PAYMENT_ID_PATTERN = r"^PAY-\d{7}$"


def resolve_path(path: Path | str, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` against the project root when it is not absolute.

    Keeping relative paths project-root relative means the CLI behaves the same
    no matter which directory it is started from.
    """

    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate)


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime configuration for a reconciliation run."""

    orders_path: Path = DEFAULT_ORDERS_PATH
    payments_path: Path = DEFAULT_PAYMENTS_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH

    amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE
    float_epsilon: float = DEFAULT_FLOAT_EPSILON

    required_order_columns: tuple[str, ...] = ORDER_COLUMNS
    required_payment_columns: tuple[str, ...] = PAYMENT_COLUMNS

    expected_order_statuses: tuple[str, ...] = EXPECTED_ORDER_STATUSES
    expected_payment_statuses: tuple[str, ...] = EXPECTED_PAYMENT_STATUSES
    expected_currencies: tuple[str, ...] = EXPECTED_CURRENCIES
    expected_sales_channels: tuple[str, ...] = EXPECTED_SALES_CHANNELS
    expected_payment_methods: tuple[str, ...] = EXPECTED_PAYMENT_METHODS
    allowed_payment_statuses_by_order_status: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(ALLOWED_PAYMENT_STATUSES_BY_ORDER_STATUS)
    )

    order_id_pattern: str = ORDER_ID_PATTERN
    payment_id_pattern: str = PAYMENT_ID_PATTERN
    duplicate_payment_fingerprint_columns: tuple[str, ...] = (
        DUPLICATE_PAYMENT_FINGERPRINT_COLUMNS
    )

    random_seed: int = DEFAULT_RANDOM_SEED
    order_count: int = DEFAULT_ORDER_COUNT

    def __post_init__(self) -> None:
        if self.amount_tolerance < 0:
            raise ValueError("amount_tolerance must be zero or positive")
        if self.float_epsilon <= 0:
            raise ValueError("float_epsilon must be positive")
        if not self.required_order_columns:
            raise ValueError("required_order_columns must not be empty")
        if not self.required_payment_columns:
            raise ValueError("required_payment_columns must not be empty")
        if "order_id" not in self.required_order_columns:
            raise ValueError("required_order_columns must contain order_id")
        if "order_id" not in self.required_payment_columns:
            raise ValueError("required_payment_columns must contain order_id")
