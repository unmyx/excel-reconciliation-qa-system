"""Tests for file-level (fatal) and row-level (quarantine) validation.

Fixtures are tiny deterministic frames carrying the reserved ``source_row``
column, so the tests exercise the real normalize -> validate path.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from recon.config import Config
from recon.excel_io import SOURCE_ROW_COLUMN
from recon.models import QAReason
from recon.normalize import normalize_orders, normalize_payments
from recon.validation import (
    QA_DETAILS_COLUMN,
    QA_REASON_COLUMN,
    SOURCE_FILE_COLUMN,
    InputValidationError,
    RowValidationResult,
    validate_orders,
    validate_orders_structure,
    validate_payments,
    validate_payments_structure,
)

ORDERS_FILE = "orders_export.xlsx"
PAYMENTS_FILE = "payments_export.xlsx"


def _raw_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "order_id": [
                "ORD-000001",
                "ORD-000002",
                "ORD-000003",
                "ORD-000004",
                "ORD-12",
                "ORD-000006",
                "ORD-000007",
                "ORD-000008",
                "ORD-000010",
                "ORD-000011",
            ],
            "customer_id": [
                "CUST-00001",
                "",  # missing required value
                "CUST-00003",
                "CUST-00004",
                "CUST-00005",
                "CUST-00006",
                "CUST-00007",
                "CUST-00008",
                "CUST-00010",
                "CUST-00011",
            ],
            "order_date": [
                "2025-01-02",
                "2025-01-03",
                "not-a-date",  # invalid date
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "2025-01-10",
                "2025-01-11",
            ],
            "currency": [
                "EUR",
                "EUR",
                "EUR",
                "EUR",
                "EUR",
                "XYZ",  # unexpected category
                "EUR",
                "EUR",
                "EUR",
                "EUR",
            ],
            "order_amount": [100.0, 50.0, 75.0, "N/A", 10.0, 20.0, "-5.00", "0", 30.0, 40.0],
            "order_status": [
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
                "unknown",  # unexpected category
            ],
            "sales_channel": [
                "web",
                "web",
                "web",
                "web",
                "web",
                "web",
                "web",
                "web",
                "affiliate",  # unexpected category
                "web",
            ],
        }
    )


def _raw_payments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "payment_id": [
                "PAY-0000001",
                "PAY-ABC",  # malformed id
                "PAY-0000003",
                "PAY-0000004",
                "PAY-0000005",
                "PAY-0000006",
                "PAY-0000007",
                "PAY-0000008",
                "PAY-0000009",
            ],
            "order_id": [
                "ORD-000001",
                "ORD-000002",
                "ORD-12",  # malformed id
                "ORD-000004",
                "ORD-000005",
                "ORD-000006",
                "ORD-000007",
                "ORD-000008",
                "ORD-000009",
            ],
            "payment_date": [
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "31/02/2025",  # invalid date
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "2025-01-10",
                "2025-01-11",
            ],
            "currency": ["EUR", "EUR", "EUR", "EUR", "EUR", "EUR", "XYZ", "EUR", "EUR"],
            "payment_amount": [100.0, 50.0, 20.0, 20.0, 30.0, 30.0, 30.0, "", 0],
            "payment_status": [
                "successful",
                "successful",
                "successful",
                "successful",
                "successful",
                "unknown",  # unexpected category
                "successful",
                "successful",
                "successful",
            ],
            "payment_method": [
                "card",
                "card",
                "card",
                "card",
                "crypto",  # unexpected category
                "card",
                "card",
                "card",
                "card",
            ],
        }
    )


def _orders_result(config: Config | None = None) -> RowValidationResult:
    settings = config or Config()
    return validate_orders(
        normalize_orders(_raw_orders()), source_file=ORDERS_FILE, config=settings
    )


def _payments_result(config: Config | None = None) -> RowValidationResult:
    settings = config or Config()
    return validate_payments(
        normalize_payments(_raw_payments()), source_file=PAYMENTS_FILE, config=settings
    )


def _reasons_by_row(result: RowValidationResult) -> dict[int, str]:
    return {
        int(source_row): str(reason)
        for source_row, reason in zip(
            result.quarantined[SOURCE_ROW_COLUMN], result.quarantined[QA_REASON_COLUMN]
        )
    }


# --------------------------------------------------------------------------
# File-level validation (fatal)
# --------------------------------------------------------------------------


def test_missing_required_column_raises_input_validation_error() -> None:
    frame = _raw_orders().drop(columns=["currency", "order_status"])

    with pytest.raises(InputValidationError) as error:
        validate_orders_structure(frame, source="data/orders_export.xlsx")

    message = str(error.value)
    assert "data/orders_export.xlsx" in message  # names the file
    assert "currency" in message  # and every missing column
    assert "order_status" in message


def test_missing_required_column_raises_for_payments_too() -> None:
    frame = _raw_payments().drop(columns=["payment_amount"])

    with pytest.raises(InputValidationError, match="payment_amount"):
        validate_payments_structure(frame, source=PAYMENTS_FILE)


def test_empty_header_row_is_fatal() -> None:
    unnamed = pd.DataFrame({"Unnamed: 0": ["ORD-000001"], "Unnamed: 1": [100.0]})
    with pytest.raises(InputValidationError, match="no usable header row"):
        validate_orders_structure(unnamed, source=ORDERS_FILE)

    with pytest.raises(InputValidationError, match="no usable header row"):
        validate_orders_structure(pd.DataFrame(), source=ORDERS_FILE)


def test_extra_columns_warn_but_do_not_fail(caplog: pytest.LogCaptureFixture) -> None:
    frame = _raw_orders()
    frame["warehouse_note"] = "n/a"

    with caplog.at_level(logging.WARNING, logger="recon.validation"):
        validate_orders_structure(frame, source=ORDERS_FILE)

    assert any("warehouse_note" in record.getMessage() for record in caplog.records)


def test_source_row_column_is_not_reported_as_extra(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="recon.validation"):
        validate_orders_structure(_raw_orders(), source=ORDERS_FILE)

    assert not caplog.records


def test_header_only_export_is_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="recon.validation"):
        validate_payments_structure(_raw_payments().iloc[0:0], source=PAYMENTS_FILE)

    assert any("no data rows" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# Row-level validation (quarantine)
# --------------------------------------------------------------------------


def test_valid_rows_are_separated_from_quarantined_rows() -> None:
    result = _orders_result()

    assert len(result.valid) == 1
    assert result.valid[SOURCE_ROW_COLUMN].tolist() == [2]
    assert result.quarantined_count == 9
    assert result.total_rows == len(_raw_orders())  # no row is lost
    assert result.label == "orders"
    assert result.source_file == ORDERS_FILE

    valid_ids = set(result.valid["order_id"])
    quarantined_ids = set(result.quarantined["order_id"])
    assert valid_ids.isdisjoint(quarantined_ids)


def test_missing_required_value_is_quarantined() -> None:
    result = _orders_result()

    assert _reasons_by_row(result)[3] == str(QAReason.MISSING_REQUIRED_VALUE)
    details = result.quarantined.set_index(SOURCE_ROW_COLUMN).loc[3, QA_DETAILS_COLUMN]
    assert "customer_id" in details


def test_invalid_date_is_quarantined() -> None:
    order_result = _orders_result()
    payment_result = _payments_result()

    assert _reasons_by_row(order_result)[4] == str(QAReason.INVALID_DATE)
    assert _reasons_by_row(payment_result)[5] == str(QAReason.INVALID_DATE)


def test_invalid_numeric_value_is_quarantined() -> None:
    result = _orders_result()

    assert _reasons_by_row(result)[5] == str(QAReason.INVALID_NUMERIC)
    details = result.quarantined.set_index(SOURCE_ROW_COLUMN).loc[5, QA_DETAILS_COLUMN]
    assert "order_amount" in details


def test_malformed_ids_are_quarantined() -> None:
    order_result = _orders_result()
    payment_result = _payments_result()
    payment_reasons = _reasons_by_row(payment_result)

    assert _reasons_by_row(order_result)[6] == str(QAReason.MALFORMED_ID)
    assert payment_reasons[3] == str(QAReason.MALFORMED_ID)  # payment_id
    assert payment_reasons[4] == str(QAReason.MALFORMED_ID)  # order_id on a payment


def test_unexpected_categories_are_quarantined() -> None:
    order_reasons = _reasons_by_row(_orders_result())
    payment_reasons = _reasons_by_row(_payments_result())

    assert order_reasons[7] == str(QAReason.UNEXPECTED_CATEGORY)  # currency
    assert order_reasons[10] == str(QAReason.UNEXPECTED_CATEGORY)  # sales_channel
    assert order_reasons[11] == str(QAReason.UNEXPECTED_CATEGORY)  # order_status
    assert payment_reasons[6] == str(QAReason.UNEXPECTED_CATEGORY)  # currency
    assert payment_reasons[7] == str(QAReason.UNEXPECTED_CATEGORY)  # payment_status
    assert payment_reasons[8] == str(QAReason.UNEXPECTED_CATEGORY)  # payment_method


def test_non_positive_amounts_are_quarantined() -> None:
    order_reasons = _reasons_by_row(_orders_result())
    payment_reasons = _reasons_by_row(_payments_result())

    assert order_reasons[8] == str(QAReason.NON_POSITIVE_AMOUNT)  # negative
    assert order_reasons[9] == str(QAReason.NON_POSITIVE_AMOUNT)  # zero
    assert payment_reasons[10] == str(QAReason.NON_POSITIVE_AMOUNT)


def test_blank_required_payment_value_is_missing_not_invalid() -> None:
    result = _payments_result()

    assert _reasons_by_row(result)[9] == str(QAReason.MISSING_REQUIRED_VALUE)


def test_quarantined_rows_keep_original_data_and_tracing() -> None:
    result = _payments_result()

    assert list(result.quarantined.columns)[:4] == [
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_COLUMN,
        QA_REASON_COLUMN,
        QA_DETAILS_COLUMN,
    ]
    assert result.quarantined[SOURCE_FILE_COLUMN].tolist() == [PAYMENTS_FILE] * 8

    row = result.quarantined.set_index(SOURCE_ROW_COLUMN).loc[8]
    assert row["order_id"] == "ORD-000007"
    assert row["currency"] == "XYZ"
    assert row["payment_method"] == "CARD"
    assert row[QA_REASON_COLUMN] == str(QAReason.UNEXPECTED_CATEGORY)


def test_reason_counts_summarise_the_quarantine() -> None:
    result = _orders_result()

    assert result.reason_counts == {
        str(QAReason.NON_POSITIVE_AMOUNT): 2,
        str(QAReason.MISSING_REQUIRED_VALUE): 1,
        str(QAReason.INVALID_DATE): 1,
        str(QAReason.INVALID_NUMERIC): 1,
        str(QAReason.MALFORMED_ID): 1,
        str(QAReason.UNEXPECTED_CATEGORY): 3,
    }


def test_row_with_multiple_problems_uses_priority_and_lists_all_details() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2, 3],
            "order_id": ["ORD-1", "ORD-000003"],
            "customer_id": ["CUST-00001", ""],
            "order_date": ["not-a-date", "not-a-date"],
            "currency": ["XYZ", "EUR"],
            "order_amount": ["-1", 10.0],
            "order_status": ["completed", "completed"],
            "sales_channel": ["web", "web"],
        }
    )

    result = validate_orders(normalize_orders(frame), source_file=ORDERS_FILE)
    reasons = _reasons_by_row(result)
    details = dict(
        zip(result.quarantined[SOURCE_ROW_COLUMN], result.quarantined[QA_DETAILS_COLUMN])
    )

    # invalid_date outranks malformed_id, unexpected_category and non_positive_amount
    assert reasons[2] == str(QAReason.INVALID_DATE)
    assert details[2] == (
        "currency: unexpected_category; order_amount: non_positive_amount; "
        "order_date: invalid_date; order_id: malformed_id"
    )
    # a missing required value outranks everything else
    assert reasons[3] == str(QAReason.MISSING_REQUIRED_VALUE)
    assert details[3] == "customer_id: missing_required_value; order_date: invalid_date"


def test_every_input_row_ends_up_in_exactly_one_bucket() -> None:
    for result, raw in ((_orders_result(), _raw_orders()), (_payments_result(), _raw_payments())):
        assert result.total_rows == len(raw)
        traced = sorted(
            result.valid[SOURCE_ROW_COLUMN].tolist()
            + result.quarantined[SOURCE_ROW_COLUMN].tolist()
        )
        assert traced == raw[SOURCE_ROW_COLUMN].tolist()


def test_clean_export_produces_no_quarantine() -> None:
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2],
            "order_id": ["ORD-000001"],
            "customer_id": ["CUST-00001"],
            "order_date": ["2025-01-02"],
            "currency": ["eur"],
            "order_amount": [100.0],
            "order_status": ["completed"],
            "sales_channel": ["web"],
        }
    )

    result = validate_orders(normalize_orders(frame), source_file=ORDERS_FILE)

    assert result.quarantined.empty
    assert result.reason_counts == {}
    assert len(result.valid) == 1
    assert list(result.quarantined.columns)[:4] == [
        SOURCE_FILE_COLUMN,
        SOURCE_ROW_COLUMN,
        QA_REASON_COLUMN,
        QA_DETAILS_COLUMN,
    ]


# --------------------------------------------------------------------------
# Configuration drives the rules
# --------------------------------------------------------------------------


def test_categories_are_compared_case_insensitively() -> None:
    # normalize upper-cases payment_method while config holds lower case.
    frame = pd.DataFrame(
        {
            SOURCE_ROW_COLUMN: [2],
            "payment_id": ["PAY-0000001"],
            "order_id": ["ORD-000001"],
            "payment_date": ["2025-01-03"],
            "currency": ["eur"],
            "payment_amount": [100.0],
            "payment_status": ["SUCCESSFUL"],
            "payment_method": ["card"],
        }
    )

    result = validate_payments(
        normalize_payments(frame),
        source_file=PAYMENTS_FILE,
        config=Config(expected_payment_methods=("card",)),
    )

    assert result.quarantined.empty


def test_expected_categories_come_from_config() -> None:
    extended = Config(expected_currencies=("EUR", "XYZ"))

    default_result = _orders_result()
    extended_result = _orders_result(extended)

    assert _reasons_by_row(default_result)[7] == str(QAReason.UNEXPECTED_CATEGORY)
    # Accepting XYZ in the config makes that very same row valid.
    assert extended_result.valid[SOURCE_ROW_COLUMN].tolist() == [2, 7]
    assert 7 not in _reasons_by_row(extended_result)


def test_id_pattern_comes_from_config() -> None:
    strict = Config(order_id_pattern=r"^ORD-000001$")

    result = _orders_result(strict)
    reasons = _reasons_by_row(result)

    assert result.valid[SOURCE_ROW_COLUMN].tolist() == [2]
    # Row 8 is a non-positive amount under the default pattern, but its order id
    # fails the stricter pattern, which outranks it.
    assert reasons[8] == str(QAReason.MALFORMED_ID)
