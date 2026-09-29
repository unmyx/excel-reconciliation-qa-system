"""Command line entry point for the Excel Reconciliation & QA System.

Run from the project root::

    python run.py                          # data/*.xlsx -> outputs/reconciliation_report.xlsx
    python run.py --generate-sample        # build the sample exports first, then reconcile
    python run.py --tolerance 0.05 --log-level DEBUG
    python run.py --orders my_orders.xlsx --output outputs/custom.xlsx

Exit codes: ``0`` success, ``2`` fatal input/validation error, ``1`` unexpected
error. Fatal problems are logged at ERROR, reported concisely on stderr, and stop
the run before any report is written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from recon.config import Config
from recon.excel_io import ExcelInputError, ExcelOutputError
from recon.logging_setup import setup_logging
from recon.pipeline import PipelineResult, generate_sample_exports, run_pipeline
from recon.validation import InputValidationError

logger = logging.getLogger("run")

DEFAULT_CONFIG = Config()
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

EXIT_SUCCESS = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_FATAL_INPUT_ERROR = 2


def build_config(args: argparse.Namespace) -> Config:
    """Apply the command line overrides on top of the configured defaults."""

    overrides: dict[str, object] = {}
    if args.orders is not None:
        overrides["orders_path"] = Path(args.orders)
    if args.payments is not None:
        overrides["payments_path"] = Path(args.payments)
    if args.output is not None:
        overrides["output_path"] = Path(args.output)
    if args.tolerance is not None:
        overrides["amount_tolerance"] = float(args.tolerance)
    return replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Compare the orders and payments Excel exports and write a "
            "review-ready reconciliation workbook."
        ),
    )
    parser.add_argument(
        "--orders",
        type=Path,
        help=f"orders export (default: {DEFAULT_CONFIG.orders_path})",
    )
    parser.add_argument(
        "--payments",
        type=Path,
        help=f"payments export (default: {DEFAULT_CONFIG.payments_path})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"report path (default: {DEFAULT_CONFIG.output_path})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help=(
            "absolute amount tolerance in the order currency "
            f"(default: {DEFAULT_CONFIG.amount_tolerance})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "random seed for --generate-sample; also recorded on the Summary tab "
            "(default: the seed found in the generation manifest)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="INFO",
        help="logging verbosity (default: %(default)s)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help=(
            "build the deterministic sample exports in the export directory "
            "before reconciling them"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""

    args = parse_args(argv)
    setup_logging(args.log_level)

    if args.tolerance is not None and args.tolerance < 0:
        # config.Config would raise too, but as a usage error this reads better.
        print("error: --tolerance must be zero or positive", file=sys.stderr)
        return EXIT_FATAL_INPUT_ERROR

    config = build_config(args)

    try:
        if args.generate_sample:
            config = generate_sample_exports(config, seed=args.seed)
        result = run_pipeline(config, seed=args.seed)
    except (ExcelInputError, InputValidationError) as error:
        logger.error("Fatal input error: %s", error)
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FATAL_INPUT_ERROR
    except ExcelOutputError as error:
        logger.error("Could not write the report: %s", error)
        print(f"error: {error}", file=sys.stderr)
        return EXIT_UNEXPECTED_ERROR
    except Exception as error:  # noqa: BLE001 - last-resort guard for the CLI
        logger.exception("Unexpected error while reconciling")
        print(
            f"error: unexpected failure ({type(error).__name__}); "
            "see the log above for details",
            file=sys.stderr,
        )
        return EXIT_UNEXPECTED_ERROR

    _log_result(result)
    return EXIT_SUCCESS


def _log_result(result: PipelineResult) -> None:
    metrics = result.metrics
    logger.info(
        "Review %s | matched: %s | affected records: %s | quarantined rows: %s",
        result.output_path,
        f"{metrics.matched_count:,}",
        f"{metrics.exception_count:,}",
        f"{metrics.quarantined_rows:,}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
