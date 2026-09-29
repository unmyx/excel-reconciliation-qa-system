"""Normalize raw export values before row-level validation.

This module performs the field-level cleanup described in ``docs/PLAN.md``:
trimming text, upper-casing the agreed categorical columns, stripping currency
formatting from amounts, converting amounts to numbers and coercing dates. It
never drops a row, never matches records and never classifies an exception.

Unparseable values
------------------
``errors="coerce"`` turns an unparseable value into a missing one, which would
hide *why* a row is bad. The result therefore carries a small ``problems`` table
with one record per unparseable cell::

    source_row  column        raw_value     reason
    4           order_amount  N/A           invalid_numeric
    4           order_date    not-a-date    invalid_date

That is what lets validation report "invalid date" instead of "missing value"
without re-reading the workbook. Blank cells are never listed: they are missing
values, not parse failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from recon.excel_io import SOURCE_ROW_COLUMN
from recon.models import QAReason

logger = logging.getLogger(__name__)

THOUSANDS_SEPARATOR = ","
CURRENCY_SYMBOL_PATTERN = r"[$€£¥]"
#: A value whose commas are grouped the conventional way, e.g. ``1,234.56``.
GROUPED_THOUSANDS_PATTERN = r"-?\d{1,3}(,\d{3})+(\.\d+)?"
PROBLEM_COLUMNS: tuple[str, ...] = (SOURCE_ROW_COLUMN, "column", "raw_value", "reason")


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """Which columns of an export hold text, categorical, date and amount fields."""

    label: str
    text_columns: tuple[str, ...]
    upper_columns: tuple[str, ...]
    date_columns: tuple[str, ...]
    amount_columns: tuple[str, ...]


ORDERS_SPEC = NormalizationSpec(
    label="orders",
    text_columns=(
        "order_id",
        "customer_id",
        "currency",
        "order_status",
        "sales_channel",
    ),
    upper_columns=("currency", "order_status"),
    date_columns=("order_date",),
    amount_columns=("order_amount",),
)

PAYMENTS_SPEC = NormalizationSpec(
    label="payments",
    text_columns=(
        "payment_id",
        "order_id",
        "currency",
        "payment_status",
        "payment_method",
    ),
    upper_columns=("currency", "payment_status", "payment_method"),
    date_columns=("payment_date",),
    amount_columns=("payment_amount",),
)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """A normalized copy of an export plus the cells that could not be parsed.

    ``frame`` keeps every input row in its original order; ``problems`` is
    ordered by ``source_row`` then column name.
    """

    label: str
    frame: pd.DataFrame
    problems: pd.DataFrame

    @property
    def has_problems(self) -> bool:
        return not self.problems.empty


def normalize_orders(frame: pd.DataFrame) -> NormalizationResult:
    """Return a normalized copy of a raw orders export.

    Raises:
        ValueError: if ``frame`` has no ``source_row`` column. Row tracing is
            required and never invented.
    """

    return _normalize(frame, ORDERS_SPEC)


def normalize_payments(frame: pd.DataFrame) -> NormalizationResult:
    """Return a normalized copy of a raw payments export. See :func:`normalize_orders`."""

    return _normalize(frame, PAYMENTS_SPEC)


def _normalize(frame: pd.DataFrame, spec: NormalizationSpec) -> NormalizationResult:
    if SOURCE_ROW_COLUMN not in frame.columns:
        raise ValueError(
            f"cannot normalize the {spec.label} export: the reserved "
            f"{SOURCE_ROW_COLUMN!r} column is missing (read exports with recon.excel_io)"
        )

    normalized = frame.copy()
    problems: list[dict[str, object]] = []

    # Columns are only touched when they exist: a missing column is a structural
    # problem that validation reports, not something normalization invents.
    for column in spec.text_columns:
        if column in normalized.columns:
            normalized[column] = _as_text(normalized[column])
    for column in spec.upper_columns:
        if column in normalized.columns:
            normalized[column] = _upper_text(normalized[column])
    for column in spec.amount_columns:
        if column in normalized.columns:
            values, failed, raw_text = _convert_amounts(normalized[column])
            normalized[column] = values
            problems.extend(
                _problem_records(
                    normalized, column, raw_text, failed, QAReason.INVALID_NUMERIC
                )
            )
    for column in spec.date_columns:
        if column in normalized.columns:
            values, failed, raw_text = _convert_dates(normalized[column])
            normalized[column] = values
            problems.extend(
                _problem_records(
                    normalized, column, raw_text, failed, QAReason.INVALID_DATE
                )
            )

    result_problems = _problems_frame(problems)
    logger.info(
        "Normalized %s rows from the %s export (%s cells could not be parsed)",
        f"{len(normalized):,}",
        spec.label,
        f"{len(result_problems):,}",
    )
    if not result_problems.empty:
        logger.debug(
            "%s normalization problems: %s",
            spec.label,
            result_problems["reason"].value_counts().to_dict(),
        )
    return NormalizationResult(
        label=spec.label, frame=normalized, problems=result_problems
    )


def _as_text(series: pd.Series) -> pd.Series:
    """Text view of a column with surrounding whitespace removed.

    Converting to the nullable string dtype first matters: calling ``.str.strip()``
    straight on an object column silently turns numbers (and anything else that is
    not a string) into missing values.
    """

    return series.astype("string").str.strip()


def _upper_text(series: pd.Series) -> pd.Series:
    return _as_text(series).str.upper()


def _is_blank(text: pd.Series) -> pd.Series:
    """True where the text view is missing or empty after trimming."""

    blank = text.isna() | (text.fillna("") == "")
    return blank.fillna(False).astype(bool)


def _convert_amounts(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strip currency formatting, convert to numbers, flag unparseable cells.

    Commas are only removed when they are grouped the conventional way
    (``1,234.56``). A value such as ``1.000,50`` is ambiguous - dropping its
    comma would silently turn EUR 1.000,50 into 1.0005 - so it is left as text
    and reported as an invalid numeric value instead.
    """

    raw_text = _as_text(series)
    without_symbols = raw_text.str.replace(CURRENCY_SYMBOL_PATTERN, "", regex=True)
    grouped_thousands = without_symbols.str.fullmatch(
        GROUPED_THOUSANDS_PATTERN, na=False
    ).astype(bool)
    digits = without_symbols.str.replace(
        THOUSANDS_SEPARATOR, "", regex=False
    ).where(grouped_thousands, without_symbols)
    values = pd.to_numeric(digits, errors="coerce").astype("float64")
    failed = (values.isna() & ~_is_blank(raw_text)).astype(bool)
    return values, failed, raw_text


def _convert_dates(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Coerce dates, flag unparseable cells.

    ``format="mixed"`` parses every value on its own instead of forcing one
    inferred format on the whole column; real exports mix date-only and datetime
    text, which would otherwise be coerced to NaT and reported as invalid.
    """

    raw_text = _as_text(series)
    values = pd.to_datetime(raw_text, errors="coerce", format="mixed")
    failed = (values.isna() & ~_is_blank(raw_text)).astype(bool)
    return values, failed, raw_text


def _problem_records(
    frame: pd.DataFrame,
    column: str,
    raw_text: pd.Series,
    failed: pd.Series,
    reason: QAReason,
) -> list[dict[str, object]]:
    source_rows = frame[SOURCE_ROW_COLUMN]
    return [
        {
            SOURCE_ROW_COLUMN: int(source_rows.iloc[position]),
            "column": column,
            "raw_value": str(raw_text.iloc[position]),
            "reason": str(reason),
        }
        for position, is_failed in enumerate(failed.tolist())
        if is_failed
    ]


def _problems_frame(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    problems = pd.DataFrame(records, columns=list(PROBLEM_COLUMNS))
    problems = problems.astype(
        {
            SOURCE_ROW_COLUMN: "int64",
            "column": "string",
            "raw_value": "string",
            "reason": "string",
        }
    )
    return problems.sort_values(
        [SOURCE_ROW_COLUMN, "column"], kind="stable", ignore_index=True
    )
