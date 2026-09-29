"""End-to-end orchestration of the reconciliation stages.

The nine stages of ``docs/PLAN.md`` §3, in order:

1. load            ``excel_io.read_orders`` / ``read_payments``
2. validate        ``validation.validate_*_structure`` (fatal on a broken export)
3. normalize       ``normalize.normalize_orders`` / ``normalize_payments``
4. validate rows   ``validation.validate_orders`` / ``validate_payments`` (quarantine)
5. match           ``matching.reconcile`` - payments aggregated per order
6. reconcile       (same call: amount, currency and status checks)
7. classify        (same call: findings labelled with ``models.ExceptionType``)
8. summarize       ``metrics.summarize``
9. report          ``report.write_report``

Every stage is an existing module; this file only sequences them and logs what
happened. Fatal input problems propagate as :class:`recon.excel_io.ExcelInputError`
or :class:`recon.validation.InputValidationError` *before* anything is written, so
a failed run never leaves a partial workbook behind. Logging is configured by the
caller (``recon.logging_setup``), never here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from recon.config import Config, resolve_path
from recon.excel_io import read_orders, read_payments
from recon.matching import ReconciliationResult, reconcile
from recon.metrics import summarize
from recon.models import SummaryMetrics
from recon.normalize import normalize_orders, normalize_payments
from recon.report import write_report
from recon.validation import (
    RowValidationResult,
    validate_orders,
    validate_orders_structure,
    validate_payments,
    validate_payments_structure,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Config()
GENERATED_ORDERS_NAME = "orders_export.xlsx"
GENERATED_PAYMENTS_NAME = "payments_export.xlsx"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Everything a finished run produced."""

    orders_file: str
    payments_file: str
    output_path: Path
    orders: RowValidationResult
    payments: RowValidationResult
    reconciliation: ReconciliationResult
    metrics: SummaryMetrics


def run_pipeline(
    config: Config = DEFAULT_CONFIG, *, seed: int | None = None
) -> PipelineResult:
    """Run every stage and return the results.

    Args:
        config: Paths and business rules for the run.
        seed: Seed to record on the Summary tab. When omitted, the seed written
            by the sample-data generator is used if the exports came from it.

    Raises:
        ExcelInputError: An input file is missing or unreadable (fatal).
        InputValidationError: An input file has no usable header or is missing a
            required column (fatal).
    """

    orders_path = resolve_path(config.orders_path)
    payments_path = resolve_path(config.payments_path)
    output_path = resolve_path(config.output_path)
    orders_file = orders_path.name
    payments_file = payments_path.name

    logger.info(
        "Reconciling %s against %s (amount tolerance %s)",
        orders_path,
        payments_path,
        config.amount_tolerance,
    )

    logger.debug("stage 1/9 load exports")
    orders_raw = read_orders(orders_path)
    payments_raw = read_payments(payments_path)

    logger.debug("stage 2/9 validate input structure")
    validate_orders_structure(orders_raw, source=orders_path, config=config)
    validate_payments_structure(payments_raw, source=payments_path, config=config)

    logger.debug("stage 3/9 normalize fields")
    orders_normalized = normalize_orders(orders_raw)
    payments_normalized = normalize_payments(payments_raw)

    logger.debug("stage 4/9 validate rows and quarantine")
    orders = validate_orders(
        orders_normalized, source_file=orders_file, config=config
    )
    payments = validate_payments(
        payments_normalized, source_file=payments_file, config=config
    )

    logger.debug("stage 5-7/9 match, reconcile and classify")
    reconciliation = reconcile(orders, payments, config=config)

    logger.debug("stage 8/9 summarize metrics")
    metrics = summarize(orders, payments, reconciliation)

    logger.debug("stage 9/9 write report")
    written = write_report(
        output_path,
        orders=orders,
        payments=payments,
        reconciliation=reconciliation,
        metrics=metrics,
        config=config,
        seed=seed if seed is not None else _generator_seed(config, orders_file),
    )

    logger.info(
        "Matched %s of %s orders (%.2f%%) | affected records: %s | "
        "exception findings: %s | quarantined rows: %s",
        f"{metrics.matched_count:,}",
        f"{metrics.total_orders:,}",
        metrics.reconciliation_rate * 100,
        f"{metrics.exception_count:,}",
        f"{reconciliation.finding_count:,}",
        f"{metrics.quarantined_rows:,}",
    )
    logger.info("Reconciliation report written to %s", written)

    return PipelineResult(
        orders_file=orders_file,
        payments_file=payments_file,
        output_path=written,
        orders=orders,
        payments=payments,
        reconciliation=reconciliation,
        metrics=metrics,
    )


def generate_sample_exports(
    config: Config = DEFAULT_CONFIG, *, seed: int | None = None
) -> Config:
    """Write the deterministic sample exports and return a config pointing at them.

    The generator always names its files ``orders_export.xlsx`` and
    ``payments_export.xlsx`` inside the export directory, so the returned config
    reads exactly what was generated.
    """

    # Imported here so the normal run does not pull in the data generator.
    from scripts.generate_sample_data import generate_dataset

    export_directory = resolve_path(config.orders_path).parent
    sample_seed = config.random_seed if seed is None else seed
    logger.info(
        "Generating deterministic sample exports in %s (seed %s, %s orders)",
        export_directory,
        sample_seed,
        f"{config.order_count:,}",
    )
    generated = generate_dataset(
        order_count=config.order_count, seed=sample_seed, output_dir=export_directory
    )
    logger.info(
        "Sample exports ready: %s and %s",
        generated.orders_path.name,
        generated.payments_path.name,
    )
    return replace(
        config,
        orders_path=generated.orders_path,
        payments_path=generated.payments_path,
        manifest_path=generated.manifest_path,
    )


def _generator_seed(config: Config, orders_file: str) -> int | None:
    """Seed from the sample-data manifest, when the manifest matches this export."""

    manifest_path = resolve_path(config.manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable generation manifest %s: %s", manifest_path, exc)
        return None

    outputs = manifest.get("outputs")
    recorded = outputs.get("orders_file") if isinstance(outputs, dict) else None
    if recorded and Path(str(recorded)).name != orders_file:
        logger.debug(
            "Generation manifest %s describes %s, not %s; no seed recorded",
            manifest_path,
            recorded,
            orders_file,
        )
        return None

    seed = manifest.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return None
