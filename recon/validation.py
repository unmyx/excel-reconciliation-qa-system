"""Structural and row-level validation of the two input exports.

Two levels, deliberately kept apart (see ``docs/PLAN.md`` §4):

File level - fatal
    ``validate_orders_structure`` / ``validate_payments_structure`` check the
    worksheet before anything else happens: an empty header row, a missing
    required column or an unreadable file stops the run, because every number
    produced afterwards would be meaningless. These raise ``InputValidationError``
    (unreadable files surface from :mod:`recon.excel_io` as ``ExcelInputError``);
    the CLI turns both into exit code 2. Extra columns are logged and ignored.

Row level - quarantine
    ``validate_orders`` / ``validate_payments`` run on a
    :class:`recon.normalize.NormalizationResult` and split the rows into ``valid``
    and ``quarantined``. Quarantined rows keep their original data plus
    ``source_file``, ``source_row``, ``qa_reason`` and ``qa_details`` so a reviewer
    can trace every finding back to the workbook. Nothing is dropped, and only
    ``valid`` rows may continue into matching.

Accepted currencies, statuses, channels, methods, ID patterns and required
columns all come from :class:`recon.config.Config`; this module only knows which
*column* holds which kind of value.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from recon.config import Config
from recon.excel_io import SOURCE_ROW_COLUMN
from recon.models import QAReason
from recon.normalize import NormalizationResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Config()

#: Columns added to every quarantined row for the QA Issues tab.
SOURCE_FILE_COLUMN = "source_file"
QA_REASON_COLUMN = "qa_reason"
QA_DETAILS_COLUMN = "qa_details"

UNNAMED_COLUMN_PATTERN = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)
BLANK_COLUMN_NAMES = frozenset({"", "nan", "none", "<na>"})

#: Precedence used when one row has several problems, most serious first.
QA_REASON_PRIORITY: tuple[QAReason, ...] = (
    QAReason.MISSING_REQUIRED_VALUE,
    QAReason.INVALID_DATE,
    QAReason.INVALID_NUMERIC,
    QAReason.MALFORMED_ID,
    QAReason.UNEXPECTED_CATEGORY,
    QAReason.NON_POSITIVE_AMOUNT,
)
_REASON_ORDER = {str(reason): index for index, reason in enumerate(QA_REASON_PRIORITY)}

#: Signature of the helper that records "row N, column C, reason R".
_ReasonAdder = Callable[[int, str, QAReason], None]


class InputValidationError(Exception):
    """Raised when an export cannot be used at all (fatal, CLI exit code 2)."""


@dataclass(frozen=True, slots=True)
class RowValidationResult:
    """Valid rows, quarantined rows, and the reasons behind the quarantine.

    Both frames carry a clean positional index; ``source_row`` is the link back
    to the original worksheet row.
    """

    label: str
    source_file: str
    valid: pd.DataFrame
    quarantined: pd.DataFrame

    @property
    def total_rows(self) -> int:
        return len(self.valid) + len(self.quarantined)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined)

    @property
    def reason_counts(self) -> dict[str, int]:
        """Quarantined rows per primary reason."""

        if self.quarantined.empty:
            return {}
        counts = self.quarantined[QA_REASON_COLUMN].value_counts()
        return {str(reason): int(count) for reason, count in counts.items()}


@dataclass(frozen=True, slots=True)
class _RowCheckSpec:
    """Which columns of an export hold IDs, categoricals and amounts."""

    label: str
    required_columns: tuple[str, ...]
    id_checks: tuple[tuple[str, str], ...]
    category_checks: tuple[tuple[str, tuple[str, ...]], ...]
    amount_columns: tuple[str, ...]


def _orders_spec(config: Config) -> _RowCheckSpec:
    return _RowCheckSpec(
        label="orders",
        required_columns=config.required_order_columns,
        id_checks=(("order_id", config.order_id_pattern),),
        category_checks=(
            ("currency", config.expected_currencies),
            ("order_status", config.expected_order_statuses),
            ("sales_channel", config.expected_sales_channels),
        ),
        amount_columns=("order_amount",),
    )


def _payments_spec(config: Config) -> _RowCheckSpec:
    return _RowCheckSpec(
        label="payments",
        required_columns=config.required_payment_columns,
        # A payment also carries an order id, which must look like one.
        id_checks=(
            ("payment_id", config.payment_id_pattern),
            ("order_id", config.order_id_pattern),
        ),
        category_checks=(
            ("currency", config.expected_currencies),
            ("payment_status", config.expected_payment_statuses),
            ("payment_method", config.expected_payment_methods),
        ),
        amount_columns=("payment_amount",),
    )


# --------------------------------------------------------------------------
# File-level (fatal) validation
# --------------------------------------------------------------------------


def validate_orders_structure(
    frame: pd.DataFrame, *, source: Path | str, config: Config = DEFAULT_CONFIG
) -> None:
    """Raise :class:`InputValidationError` if the orders export is unusable."""

    _validate_structure(
        frame,
        required_columns=config.required_order_columns,
        source=source,
        label="orders",
    )


def validate_payments_structure(
    frame: pd.DataFrame, *, source: Path | str, config: Config = DEFAULT_CONFIG
) -> None:
    """Raise :class:`InputValidationError` if the payments export is unusable."""

    _validate_structure(
        frame,
        required_columns=config.required_payment_columns,
        source=source,
        label="payments",
    )


def _validate_structure(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    source: Path | str,
    label: str,
) -> None:
    if not _has_usable_header(frame):
        raise InputValidationError(
            f"{label} export {source} has no usable header row: the worksheet is "
            "empty or its first row contains no column names"
        )

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise InputValidationError(
            f"{label} export {source} is missing required column(s): "
            f"{', '.join(missing)}"
        )

    known = set(required_columns) | {SOURCE_ROW_COLUMN}
    extra = [str(column) for column in frame.columns if column not in known]
    if extra:
        logger.warning(
            "%s export %s has unexpected column(s) that will be ignored: %s",
            label,
            source,
            ", ".join(extra),
        )
    if frame.empty:
        logger.warning("%s export %s has a header but no data rows", label, source)


def _has_usable_header(frame: pd.DataFrame) -> bool:
    return any(_is_usable_column_name(column) for column in frame.columns)


def _is_usable_column_name(column: object) -> bool:
    text = str(column).strip()
    if text.lower() in BLANK_COLUMN_NAMES:
        return False
    return not UNNAMED_COLUMN_PATTERN.match(text)


# --------------------------------------------------------------------------
# Row-level (quarantine) validation
# --------------------------------------------------------------------------


def validate_orders(
    normalization: NormalizationResult,
    *,
    source_file: str,
    config: Config = DEFAULT_CONFIG,
) -> RowValidationResult:
    """Split normalized orders into valid rows and quarantined rows."""

    return _validate_rows(
        normalization, source_file=source_file, spec=_orders_spec(config)
    )


def validate_payments(
    normalization: NormalizationResult,
    *,
    source_file: str,
    config: Config = DEFAULT_CONFIG,
) -> RowValidationResult:
    """Split normalized payments into valid rows and quarantined rows."""

    return _validate_rows(
        normalization, source_file=source_file, spec=_payments_spec(config)
    )


def _validate_rows(
    normalization: NormalizationResult, *, source_file: str, spec: _RowCheckSpec
) -> RowValidationResult:
    frame = normalization.frame
    if SOURCE_ROW_COLUMN not in frame.columns:
        raise ValueError(
            f"cannot validate the {spec.label} export: the reserved "
            f"{SOURCE_ROW_COLUMN!r} column is missing"
        )

    reasons = _collect_row_reasons(frame, normalization.problems, spec)
    primary = [
        _primary_reason(reasons.get(position, {})) for position in range(len(frame))
    ]
    details = [
        _format_details(reasons.get(position, {})) for position in range(len(frame))
    ]

    flagged = [position for position, reason in enumerate(primary) if reason]
    flagged_positions = set(flagged)
    quarantined = frame.iloc[flagged].reset_index(drop=True)
    valid = frame.iloc[
        [position for position in range(len(frame)) if position not in flagged_positions]
    ].reset_index(drop=True)

    if len(valid) + len(quarantined) != len(frame):  # never lose a row silently
        raise AssertionError(
            f"row accounting mismatch for {source_file}: {len(valid)} valid + "
            f"{len(quarantined)} quarantined != {len(frame)} input rows"
        )

    quarantined = pd.concat(
        [
            pd.DataFrame(
                {
                    SOURCE_FILE_COLUMN: source_file,
                    SOURCE_ROW_COLUMN: quarantined[SOURCE_ROW_COLUMN].to_numpy(),
                    QA_REASON_COLUMN: [primary[position] for position in flagged],
                    QA_DETAILS_COLUMN: [details[position] for position in flagged],
                },
                index=quarantined.index,
            ),
            quarantined.drop(columns=[SOURCE_ROW_COLUMN]),
        ],
        axis=1,
    )

    result = RowValidationResult(
        label=spec.label,
        source_file=source_file,
        valid=valid,
        quarantined=quarantined,
    )
    logger.info(
        "Validated %s rows from %s: %s valid, %s quarantined",
        f"{result.total_rows:,}",
        source_file,
        f"{len(valid):,}",
        f"{len(quarantined):,}",
    )
    if result.reason_counts:
        logger.debug("%s QA reasons: %s", spec.label, result.reason_counts)
    return result


def _collect_row_reasons(
    frame: pd.DataFrame,
    problems: pd.DataFrame,
    spec: _RowCheckSpec,
) -> dict[int, dict[str, str]]:
    """Map row position -> {column: reason} for every problem found on that row.

    The first reason recorded for a cell wins, and normalization problems are
    recorded before the blank check, so an unparseable date is reported as
    ``invalid_date`` rather than as a missing value.
    """

    found: dict[int, dict[str, str]] = {}

    def add(position: int, column: str, reason: QAReason) -> None:
        found.setdefault(position, {}).setdefault(column, str(reason))

    position_by_source_row = {
        int(source_row): position
        for position, source_row in enumerate(frame[SOURCE_ROW_COLUMN].tolist())
    }
    for record in problems.to_dict("records"):
        position = position_by_source_row.get(int(record[SOURCE_ROW_COLUMN]))
        if position is not None:
            add(position, str(record["column"]), QAReason(str(record["reason"])))

    _add_missing_values(frame, spec, add)
    _add_malformed_ids(frame, spec, add)
    _add_unexpected_categories(frame, spec, add)
    _add_non_positive_amounts(frame, spec, add)
    return found


def _add_missing_values(
    frame: pd.DataFrame, spec: _RowCheckSpec, add: _ReasonAdder
) -> None:
    for column in spec.required_columns:
        if column not in frame.columns:
            continue  # structural validation already reports absent columns
        for position, blank in enumerate(_blank_mask(frame[column]).tolist()):
            if blank:
                add(position, column, QAReason.MISSING_REQUIRED_VALUE)


def _add_malformed_ids(
    frame: pd.DataFrame, spec: _RowCheckSpec, add: _ReasonAdder
) -> None:
    for column, pattern in spec.id_checks:
        if column not in frame.columns:
            continue
        text = _text_view(frame[column])
        blank = _blank_mask(frame[column])
        malformed = (~text.str.fullmatch(pattern, na=False).astype(bool)) & ~blank
        for position, is_malformed in enumerate(malformed.tolist()):
            if is_malformed:
                add(position, column, QAReason.MALFORMED_ID)


def _add_unexpected_categories(
    frame: pd.DataFrame, spec: _RowCheckSpec, add: _ReasonAdder
) -> None:
    for column, expected in spec.category_checks:
        if column not in frame.columns:
            continue
        allowed = {str(value).strip().upper() for value in expected}
        text = _text_view(frame[column])
        blank = _blank_mask(frame[column])
        unexpected = (~text.str.upper().isin(allowed)) & ~blank
        for position, is_unexpected in enumerate(unexpected.tolist()):
            if is_unexpected:
                add(position, column, QAReason.UNEXPECTED_CATEGORY)


def _add_non_positive_amounts(
    frame: pd.DataFrame, spec: _RowCheckSpec, add: _ReasonAdder
) -> None:
    for column in spec.amount_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_positive = ((values <= 0) & ~_blank_mask(frame[column])).fillna(False)
        for position, is_non_positive in enumerate(non_positive.astype(bool).tolist()):
            if is_non_positive:
                add(position, column, QAReason.NON_POSITIVE_AMOUNT)


def _text_view(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _blank_mask(series: pd.Series) -> pd.Series:
    """True where a normalized value counts as missing (blank, NaN or NaT)."""

    text = _text_view(series)
    blank = text.isna() | (text.fillna("") == "")
    return blank.fillna(False).astype(bool)


def _primary_reason(reasons: dict[str, str]) -> str:
    if not reasons:
        return ""
    return min(
        reasons.values(), key=lambda reason: _REASON_ORDER.get(reason, len(_REASON_ORDER))
    )


def _format_details(reasons: dict[str, str]) -> str:
    return "; ".join(f"{column}: {reason}" for column, reason in sorted(reasons.items()))
