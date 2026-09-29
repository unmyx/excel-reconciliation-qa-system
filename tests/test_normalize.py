"""Tests for field-level normalization of raw export frames.

Fixtures are tiny hand-built frames that already carry the reserved
``source_row`` column, exactly like frames coming out of ``recon.excel_io``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from recon.excel_io import SOURCE_ROW_COLUMN
from recon.models import QAReason
from recon.normalize import normalize_orders, normalize_payments


def _orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3, 4, 5],
            "order_id": [" ORD-000001", "ORD-000002 ", "ORD-000003", "ORD-000004"],
            "customer_id": ["CUST-00001", " CUST-00002 ", "CUST-00003", "CUST-00004"],
            "order_date": ["2025-01-02", " 2025-02-03 ", "not-a-date", ""],
            "currency": ["eur", " USD ", "gbp", "EUR"],
            "order_amount": ["1,234.56", "$1,500.00", "N/A", "  250  "],
            "order_status": ["Completed", " SHIPPED", "completed", "completed"],
            "sales_channel": [" web ", "store", "mobile", "phone"],
        }
    )


def _payments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3],
            "payment_id": [" pay-0000001 ", "PAY-0000002"],
            "order_id": ["ORD-000001", " ORD-000002"],
            "payment_date": ["2025-01-05 09:30:00", "31/02/2025"],
            "currency": ["eur", "usd"],
            "payment_amount": [120.5, "€1.000,50"],
            "payment_status": ["Successful", " failed "],
            "payment_method": ["Card", " DIRECT_DEBIT "],
        }
    )


# --------------------------------------------------------------------------
# Trimming and case
# --------------------------------------------------------------------------


def test_trims_whitespace_from_text_fields() -> None:
    result = normalize_orders(_orders_frame())

    assert result.frame["order_id"].tolist() == [
        "ORD-000001",
        "ORD-000002",
        "ORD-000003",
        "ORD-000004",
    ]
    assert result.frame["customer_id"].tolist() == [
        "CUST-00001",
        "CUST-00002",
        "CUST-00003",
        "CUST-00004",
    ]
    assert result.frame["sales_channel"].tolist() == ["web", "store", "mobile", "phone"]


def test_uppercases_configured_categorical_fields() -> None:
    result = normalize_orders(_orders_frame())

    assert result.frame["currency"].tolist() == ["EUR", "USD", "GBP", "EUR"]
    assert result.frame["order_status"].tolist() == [
        "COMPLETED",
        "SHIPPED",
        "COMPLETED",
        "COMPLETED",
    ]
    # sales_channel is trimmed but deliberately not upper-cased.
    assert result.frame["sales_channel"].tolist() == ["web", "store", "mobile", "phone"]


def test_payments_uppercase_payment_status_and_method() -> None:
    result = normalize_payments(_payments_frame())

    assert result.frame["currency"].tolist() == ["EUR", "USD"]
    assert result.frame["payment_status"].tolist() == ["SUCCESSFUL", "FAILED"]
    assert result.frame["payment_method"].tolist() == ["CARD", "DIRECT_DEBIT"]


# --------------------------------------------------------------------------
# Amounts and dates
# --------------------------------------------------------------------------


def test_cleans_amount_text_and_converts_to_numbers() -> None:
    result = normalize_orders(_orders_frame())

    amounts = result.frame["order_amount"].tolist()
    assert amounts[0] == pytest.approx(1234.56)  # thousands separator removed
    assert amounts[1] == pytest.approx(1500.00)  # currency symbol removed
    assert amounts[3] == pytest.approx(250.00)  # padded value trimmed
    assert pd.isna(amounts[2])  # "N/A" is not a number
    assert result.frame["order_amount"].dtype == "float64"


def test_ambiguous_separator_formats_are_flagged_not_misread() -> None:
    # "€1.000,50" must not be silently read as 1.0005: commas are only dropped
    # when they group thousands the conventional way.
    result = normalize_payments(_payments_frame())

    amounts = result.frame["payment_amount"].tolist()
    assert amounts[0] == pytest.approx(120.5)
    assert pd.isna(amounts[1])
    amount_problems = result.problems[result.problems["column"] == "payment_amount"]
    assert amount_problems["raw_value"].tolist() == ["€1.000,50"]
    assert amount_problems["reason"].tolist() == [str(QAReason.INVALID_NUMERIC)]


def test_grouped_thousands_are_removed_for_plain_text_amounts() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3, 4],
            "order_amount": ["1,234.56", "12,345,678.90", "1234,56"],
        }
    )

    result = normalize_orders(frame)

    amounts = result.frame["order_amount"].tolist()
    assert amounts[0] == pytest.approx(1234.56)
    assert amounts[1] == pytest.approx(12345678.90)
    assert pd.isna(amounts[2])  # ambiguous comma placement stays a reported problem
    assert result.problems["raw_value"].tolist() == ["1234,56"]


def test_coerces_dates_and_flags_unparseable_values() -> None:
    result = normalize_orders(_orders_frame())

    dates = result.frame["order_date"].tolist()
    assert dates[0] == pd.Timestamp("2025-01-02")
    assert dates[1] == pd.Timestamp("2025-02-03")  # padded ISO value still parses
    assert pd.isna(dates[2])  # "not-a-date"
    assert pd.isna(dates[3])  # empty cell
    assert pd.api.types.is_datetime64_any_dtype(result.frame["order_date"])


def test_parses_mixed_date_and_datetime_text_in_one_column() -> None:
    result = normalize_payments(_payments_frame())

    dates = result.frame["payment_date"].tolist()
    assert dates[0] == pd.Timestamp("2025-01-05 09:30:00")
    assert pd.isna(dates[1])  # 31/02/2025 does not exist


# --------------------------------------------------------------------------
# Row tracing, blanks and problems
# --------------------------------------------------------------------------


def test_preserves_source_row_and_row_order() -> None:
    frame = _orders_frame()

    result = normalize_orders(frame)

    assert result.frame[SOURCE_ROW_COLUMN].tolist() == [2, 3, 4, 5]
    assert result.frame[SOURCE_ROW_COLUMN].dtype.kind == "i"
    assert len(result.frame) == len(frame)
    assert result.frame.index.tolist() == frame.index.tolist()
    assert result.frame["order_id"].tolist()[0] == "ORD-000001"


def test_records_unparseable_cells_with_raw_values() -> None:
    result = normalize_orders(_orders_frame())

    records = result.problems.to_dict("records")
    assert records == [
        {
            SOURCE_ROW_COLUMN: 4,
            "column": "order_amount",
            "raw_value": "N/A",
            "reason": str(QAReason.INVALID_NUMERIC),
        },
        {
            SOURCE_ROW_COLUMN: 4,
            "column": "order_date",
            "raw_value": "not-a-date",
            "reason": str(QAReason.INVALID_DATE),
        },
    ]
    assert result.has_problems is True


def test_blank_values_are_missing_not_problems() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3],
            "order_id": ["ORD-000001", "ORD-000002"],
            "order_date": ["", "   "],
            "order_amount": [None, ""],
        }
    )

    result = normalize_orders(frame)

    assert pd.isna(result.frame["order_date"].tolist()[0])
    assert pd.isna(result.frame["order_amount"].tolist()[0])
    assert result.problems.empty
    assert result.has_problems is False
    assert list(result.problems.columns) == [
        SOURCE_ROW_COLUMN,
        "column",
        "raw_value",
        "reason",
    ]


def test_clean_frame_produces_no_problems() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2],
            "order_id": ["ORD-000001"],
            "order_date": ["2025-01-02"],
            "currency": ["eur"],
            "order_amount": [10.0],
        }
    )

    result = normalize_orders(frame)

    assert result.problems.empty
    assert result.label == "orders"


# --------------------------------------------------------------------------
# Contract guarantees
# --------------------------------------------------------------------------


def test_does_not_mutate_the_input_frame() -> None:
    frame = _orders_frame()
    original = frame.copy(deep=True)

    result = normalize_orders(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert result.frame is not frame
    assert result.frame["currency"].tolist()[0] == "EUR"
    assert frame["currency"].tolist()[0] == "eur"


def test_leaves_unknown_columns_untouched() -> None:
    frame = _orders_frame()
    frame["warehouse_note"] = [" keep Me ", "b", "c", "d"]

    result = normalize_orders(frame)

    assert result.frame["warehouse_note"].tolist() == [" keep Me ", "b", "c", "d"]


def test_missing_columns_are_skipped_not_invented() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2],
            "order_id": [" ORD-000001 "],
            "order_date": ["2025-01-02"],
        }
    )

    result = normalize_orders(frame)

    assert list(result.frame.columns) == [SOURCE_ROW_COLUMN, "order_id", "order_date"]
    assert result.frame["order_id"].tolist() == ["ORD-000001"]


def test_missing_source_row_column_is_rejected() -> None:
    frame = _orders_frame().drop(columns=[SOURCE_ROW_COLUMN])

    with pytest.raises(ValueError, match="source_row"):
        normalize_orders(frame)


def test_normalization_is_deterministic() -> None:
    first = normalize_orders(_orders_frame())
    second = normalize_orders(_orders_frame())

    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.problems, second.problems)
