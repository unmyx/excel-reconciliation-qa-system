"""Low-level Excel input/output for the reconciliation pipeline.

This module only moves data between ``.xlsx`` workbooks and pandas DataFrames.
It deliberately contains no reconciliation logic, no matching, no row-level QA
and no report styling.

Reading
-------
``read_orders`` / ``read_payments`` return the export exactly as pandas reads it,
plus one reserved leading column ``source_row`` holding the 1-based Excel row
number of every record (the header is row 1, so the first record is row 2).

Nothing is cleaned here: values are not trimmed, case-folded or type-coerced, so
``normalize`` and ``validation`` see the raw export, including padded headers.

Missing values
--------------
Exports are read with ``keep_default_na=False`` so an empty cell stays ``""``
while a literal ``"N/A"`` / ``"NA"`` / ``"NULL"`` stays text. The two cases stay
distinguishable for the data-quality layer: an empty required cell is a missing
value, a non-numeric sentinel is an invalid numeric value. ``validation`` owns
that decision; this module only preserves what the workbook contains.

Writing
-------
``ReportWorkbookWriter`` lays prepared frames out as worksheets and hands back the
openpyxl worksheet of each sheet, so ``report.py`` can apply presentation while
the workbook is open. ``write_report_workbook`` is the plain convenience wrapper
used by tests and by callers that do not need styling.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openpyxl.worksheet.worksheet import Worksheet

logger = logging.getLogger(__name__)

EXCEL_ENGINE = "openpyxl"

#: Reserved column added to every frame read from an export.
SOURCE_ROW_COLUMN = "source_row"
HEADER_ROW = 1
FIRST_DATA_ROW = HEADER_ROW + 1


class ExcelInputError(Exception):
    """Raised when an input export is missing, unreadable, or unusable."""


class ExcelOutputError(Exception):
    """Raised when the reconciliation workbook cannot be written."""


def read_orders(path: Path | str, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read the orders export.

    Args:
        path: Path to the orders ``.xlsx`` file.
        sheet_name: Worksheet to read; defaults to the first sheet.

    Returns:
        The raw orders rows with ``source_row`` as the first column.

    Raises:
        ExcelInputError: If the file is missing, not a file, or cannot be read.
    """

    return _read_export(path, label="orders", sheet_name=sheet_name)


def read_payments(path: Path | str, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read the payments export. See :func:`read_orders` for details."""

    return _read_export(path, label="payments", sheet_name=sheet_name)


def _read_export(
    path: Path | str, *, label: str, sheet_name: str | int = 0
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise ExcelInputError(f"{label} input file not found: {source}")
    if not source.is_file():
        raise ExcelInputError(f"{label} input path is not a file: {source}")

    try:
        frame = pd.read_excel(
            source,
            sheet_name=sheet_name,
            engine=EXCEL_ENGINE,
            keep_default_na=False,  # blank cells stay "", sentinel text stays text
        )
    except Exception as exc:
        # Single I/O boundary: every reader failure (corrupt workbook, unknown
        # sheet, missing permission, ...) becomes one clear, catchable error.
        raise ExcelInputError(
            f"could not read the {label} export {source} (sheet {sheet_name!r}): {exc}"
        ) from exc

    if not isinstance(frame, pd.DataFrame):  # sheet_name=None would return a dict
        raise ExcelInputError(
            f"expected a single worksheet from {source}, got {type(frame).__name__}"
        )
    if SOURCE_ROW_COLUMN in frame.columns:
        raise ExcelInputError(
            f"{source} already contains a {SOURCE_ROW_COLUMN!r} column, "
            "which this application reserves for row tracing"
        )

    frame.insert(
        0, SOURCE_ROW_COLUMN, np.arange(FIRST_DATA_ROW, FIRST_DATA_ROW + len(frame))
    )

    logger.info(
        "Loaded %s rows from the %s export %s (sheet %r)",
        f"{len(frame):,}",
        label,
        source,
        sheet_name,
    )
    logger.debug("%s columns: %s", label, list(frame.columns))
    if frame.empty:
        logger.warning("%s export %s contains no data rows", label, source)
    return frame


class ReportWorkbookWriter:
    """Write prepared report frames into a single workbook.

    Use it as a context manager; the workbook is saved on a clean exit:

        with ReportWorkbookWriter(path) as writer:
            worksheet = writer.write_sheet("Summary", summary_frame)
            worksheet.freeze_panes = "A2"  # report.py owns presentation
    """

    def __init__(self, path: Path | str, *, overwrite: bool = True) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self.sheet_names: list[str] = []
        self._writer: pd.ExcelWriter | None = None

    def __enter__(self) -> ReportWorkbookWriter:
        if self.path.exists() and not self.overwrite:
            raise ExcelOutputError(
                f"report workbook already exists and overwrite is disabled: {self.path}"
            )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pd.ExcelWriter(self.path, engine=EXCEL_ENGINE)
        except OSError as exc:
            raise ExcelOutputError(
                f"could not open report workbook {self.path}: {exc}"
            ) from exc
        return self

    def write_sheet(self, sheet_name: str, frame: pd.DataFrame) -> Worksheet:
        """Write ``frame`` to a new worksheet and return that worksheet."""

        if self._writer is None:
            raise ExcelOutputError(
                "ReportWorkbookWriter must be used as a context manager"
            )
        try:
            frame.to_excel(self._writer, sheet_name=sheet_name, index=False)
        except (OSError, ValueError) as exc:
            raise ExcelOutputError(
                f"could not write sheet {sheet_name!r} to {self.path}: {exc}"
            ) from exc

        self.sheet_names.append(sheet_name)
        logger.debug("Wrote sheet %r with %s data rows", sheet_name, f"{len(frame):,}")
        return self._writer.sheets[sheet_name]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        writer, self._writer = self._writer, None
        if writer is None:
            return

        if exc_type is not None:
            with suppress(Exception):
                writer.close()
            logger.warning(
                "Report workbook %s left incomplete: %s", self.path, exc_type.__name__
            )
            return

        try:
            writer.close()
        except Exception as exc:
            raise ExcelOutputError(
                f"could not save report workbook {self.path}: {exc}"
            ) from exc
        logger.info(
            "Wrote report workbook %s with sheets: %s",
            self.path,
            ", ".join(self.sheet_names) if self.sheet_names else "none",
        )


def write_report_workbook(
    path: Path | str,
    sheets: Mapping[str, pd.DataFrame],
    *,
    overwrite: bool = True,
) -> Path:
    """Write prepared frames to one workbook, one worksheet per entry.

    Sheets keep the order of ``sheets`` and receive no styling; ``report.py``
    uses :class:`ReportWorkbookWriter` directly when presentation is needed.
    """

    if not sheets:
        raise ExcelOutputError("no report sheets were provided")

    with ReportWorkbookWriter(path, overwrite=overwrite) as writer:
        for sheet_name, frame in sheets.items():
            writer.write_sheet(sheet_name, frame)
    return Path(path)
