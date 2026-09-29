"""Focused tests for the low-level Excel I/O module.

Every workbook here is built inside ``tmp_path``; the tests never touch the
generated data in the repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook
from openpyxl.styles import Font

from recon import excel_io
from recon.excel_io import (
    ExcelInputError,
    ExcelOutputError,
    ReportWorkbookWriter,
    read_orders,
    read_payments,
    write_report_workbook,
)


def _orders_frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [f"ORD-{number:06d}" for number in range(1, rows + 1)],
            "customer_id": [f"CUST-{number:05d}" for number in range(1, rows + 1)],
            "order_date": pd.to_datetime(
                [f"2025-01-{day:02d}" for day in range(1, rows + 1)]
            ),
            "currency": ["EUR"] * rows,
            "order_amount": [100.0 + number for number in range(rows)],
            "order_status": ["completed"] * rows,
            "sales_channel": ["web"] * rows,
        }
    )


def _payments_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "payment_id": ["PAY-0000001", "PAY-0000002"],
            "order_id": ["ORD-000001", "ORD-000002"],
            "payment_date": [pd.Timestamp("2025-01-02"), "not-a-date"],
            "currency": ["EUR", None],
            "payment_amount": [100.0, "N/A"],
            "payment_status": ["successful", "failed"],
            "payment_method": ["card", "paypal"],
        }
    )


def _write(path: Path, frame: pd.DataFrame, sheet_name: str = "Sheet1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, sheet_name=sheet_name, index=False)
    return path


def _summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"metric": ["total_orders", "matched_count"], "value": [3, 2]}
    )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_read_orders_adds_source_row_in_file_order(tmp_path: Path) -> None:
    path = _write(tmp_path / "orders.xlsx", _orders_frame(3), "orders")

    frame = read_orders(path)

    assert list(frame.columns)[0] == excel_io.SOURCE_ROW_COLUMN
    assert frame["source_row"].tolist() == [2, 3, 4]
    assert frame["source_row"].dtype.kind == "i"
    assert frame["order_id"].tolist() == ["ORD-000001", "ORD-000002", "ORD-000003"]
    assert frame["order_amount"].tolist() == [100.0, 101.0, 102.0]
    assert len(frame) == 3


def test_read_payments_returns_raw_values_unchanged(tmp_path: Path) -> None:
    path = _write(tmp_path / "payments.xlsx", _payments_frame(), "payments")

    frame = read_payments(path)

    assert frame["source_row"].tolist() == [2, 3]
    # I/O must not clean data: normalization and validation own that.
    assert frame["payment_amount"].tolist()[1] == "N/A"
    assert frame["payment_date"].astype(str).tolist()[1] == "not-a-date"
    assert frame["currency"].tolist()[1] == ""


def test_blank_cells_and_na_sentinels_stay_distinguishable(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "payment_id": ["PAY-0000001", "PAY-0000002", "PAY-0000003"],
            "order_id": ["ORD-000001", "ORD-000002", "ORD-000003"],
            "payment_date": [pd.Timestamp("2025-01-02"), None, "not-a-date"],
            "currency": ["EUR", None, "N/A"],
            "payment_amount": [100.0, None, "N/A"],
            "payment_status": ["successful", "pending", "failed"],
            "payment_method": ["card", "paypal", "N/A"],
        }
    )
    path = _write(tmp_path / "payments.xlsx", frame, "payments")

    loaded = read_payments(path)

    amounts = loaded["payment_amount"].tolist()
    assert float(amounts[0]) == 100.0
    assert amounts[1] == ""  # genuinely empty cell
    assert amounts[2] == "N/A"  # literal sentinel text, not silently turned into a blank
    assert loaded["payment_date"].astype(str).tolist()[1] == ""
    assert loaded["currency"].tolist()[2] == "N/A"


def test_reading_does_not_touch_headers_or_text(tmp_path: Path) -> None:
    raw = _orders_frame(1).rename(columns={"currency": " currency "})
    raw[" currency "] = " eur "
    path = _write(tmp_path / "orders.xlsx", raw, "orders")

    frame = read_orders(path)

    assert " currency " in frame.columns
    assert frame[" currency "].tolist() == [" eur "]


def test_read_orders_uses_first_sheet_by_default(tmp_path: Path) -> None:
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _orders_frame(1).to_excel(writer, sheet_name="cover", index=False)
        _orders_frame(2).to_excel(writer, sheet_name="orders", index=False)

    assert read_orders(path)["order_id"].tolist() == ["ORD-000001"]
    assert read_orders(path, sheet_name="orders")["order_id"].tolist() == [
        "ORD-000001",
        "ORD-000002",
    ]


def test_header_only_export_returns_empty_frame(tmp_path: Path) -> None:
    path = _write(tmp_path / "orders.xlsx", _orders_frame(0), "orders")

    frame = read_orders(path)

    assert frame.empty
    assert frame["source_row"].tolist() == []
    assert excel_io.SOURCE_ROW_COLUMN in frame.columns


def test_read_logs_path_and_row_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write(tmp_path / "orders.xlsx", _orders_frame(3), "orders")

    with caplog.at_level(logging.INFO, logger="recon.excel_io"):
        read_orders(path)

    messages = [record.getMessage() for record in caplog.records]
    assert any("3" in message and "rows from the orders export" in message for message in messages)
    assert any(str(path) in message for message in messages)


def test_missing_file_raises_excel_input_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.xlsx"

    with pytest.raises(ExcelInputError) as error:
        read_orders(missing)

    assert "not found" in str(error.value)
    assert str(missing) in str(error.value)


def test_directory_path_raises_excel_input_error(tmp_path: Path) -> None:
    with pytest.raises(ExcelInputError, match="not a file"):
        read_payments(tmp_path)


def test_unreadable_workbook_raises_excel_input_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.xlsx"
    broken.write_text("this is not a workbook", encoding="utf-8")

    with pytest.raises(ExcelInputError, match="could not read"):
        read_payments(broken)


def test_unknown_sheet_raises_excel_input_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "orders.xlsx", _orders_frame(1), "orders")

    with pytest.raises(ExcelInputError, match="could not read"):
        read_orders(path, sheet_name="payments")


def test_reserved_source_row_column_is_rejected(tmp_path: Path) -> None:
    frame = _orders_frame(1)
    frame.insert(0, excel_io.SOURCE_ROW_COLUMN, [1])
    path = _write(tmp_path / "orders.xlsx", frame, "orders")

    with pytest.raises(ExcelInputError, match="reserves"):
        read_orders(path)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def test_write_report_workbook_round_trips_sheets_in_order(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.xlsx"
    exceptions = pd.DataFrame(
        {"order_id": ["ORD-000002"], "exception_type": ["Amount Mismatch"]}
    )

    result = write_report_workbook(path, {"Summary": _summary_frame(), "All Exceptions": exceptions})

    assert result == path
    assert path.exists()
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        assert workbook.sheet_names == ["Summary", "All Exceptions"]
    written = pd.read_excel(path, sheet_name="All Exceptions", engine="openpyxl")
    assert written["exception_type"].tolist() == ["Amount Mismatch"]


def test_write_report_workbook_rejects_empty_sheet_mapping(tmp_path: Path) -> None:
    with pytest.raises(ExcelOutputError, match="no report sheets"):
        write_report_workbook(tmp_path / "report.xlsx", {})


def test_write_refuses_to_overwrite_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "report.xlsx"
    path.write_text("existing content", encoding="utf-8")

    with pytest.raises(ExcelOutputError, match="overwrite is disabled"):
        write_report_workbook(path, {"Summary": _summary_frame()}, overwrite=False)

    assert path.read_text(encoding="utf-8") == "existing content"


def test_unwritable_target_raises_excel_output_error(tmp_path: Path) -> None:
    with pytest.raises(ExcelOutputError, match="could not open report workbook"):
        write_report_workbook(tmp_path, {"Summary": _summary_frame()})


def test_writer_returns_stylable_worksheet(tmp_path: Path) -> None:
    path = tmp_path / "report.xlsx"

    with ReportWorkbookWriter(path) as writer:
        worksheet = writer.write_sheet("Summary", _summary_frame())
        worksheet["A1"].font = Font(bold=True)
        worksheet.freeze_panes = "A2"
        worksheet["B2"].number_format = "#,##0.00"

    reopened = load_workbook(path)["Summary"]
    assert reopened["A1"].font.bold is True
    assert reopened.freeze_panes == "A2"
    assert reopened["B2"].number_format == "#,##0.00"
    assert writer.sheet_names == ["Summary"]


def test_writer_logs_saved_workbook(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "report.xlsx"

    with caplog.at_level(logging.INFO, logger="recon.excel_io"):
        write_report_workbook(path, {"Summary": _summary_frame()})

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Wrote report workbook" in message and "Summary" in message for message in messages
    )
