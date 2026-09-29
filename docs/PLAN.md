# Excel Reconciliation & QA System — Implementation Plan

Status: proposed (no application code written yet)
Target: Python 3.11+ (local env: 3.14.7, pandas 3.0.6, openpyxl 3.1.5, pytest 9.1.1)

## 1. Project structure

```
excel-reconciliation-qa-system/
├── run.py                        # CLI entry point (argparse) -> recon.pipeline
├── requirements.txt              # pandas, openpyxl, pytest (+ pinned ranges)
├── pyproject.toml                # pytest config + project metadata only
├── README.md                     # written last, from real run output
├── .gitignore
├── docs/
│   └── PLAN.md
├── recon/                        # the application package
│   ├── __init__.py
│   ├── config.py                 # Config dataclass + defaults (tolerance, paths, columns, statuses)
│   ├── models.py                 # ExceptionType constants, result/summary dataclasses
│   ├── logging_setup.py          # one place that configures stdlib logging
│   ├── excel_io.py               # read exports, write report workbook (low-level I/O only)
│   ├── normalize.py              # trim/case/type normalization of raw frames
│   ├── validation.py             # column checks + row-level QA (quarantine, never drop silently)
│   ├── matching.py               # order_id based match of orders <-> payments
│   ├── metrics.py                # summary + per-exception counts + reconciliation rate
│   ├── report.py                 # sheet layout + openpyxl formatting
│   └── pipeline.py               # orchestration of the 9 workflow stages + logging
├── scripts/
│   └── generate_sample_data.py   # deterministic synthetic exports + anomaly manifest
├── tests/
│   ├── conftest.py               # tiny deterministic fixtures/builders
│   ├── test_validation.py
│   ├── test_normalize.py
│   ├── test_matching.py
│   ├── test_checks.py            # amount/currency/status/duplicate/missing rules
│   ├── test_metrics.py
│   ├── test_report.py            # workbook shape, sheet names, formatting smoke test
│   └── test_pipeline_e2e.py      # fixtures -> xlsx report end to end
├── data/                         # generated inputs (gitignored except .gitkeep)
│   ├── orders_export.xlsx
│   ├── payments_export.xlsx
│   └── generation_manifest.json  # counts + IDs of injected anomalies (seeded)
└── outputs/                      # generated reports
    └── reconciliation_report.xlsx
```

Rationale: a flat root-level package (no `src/` layout) keeps `python run.py` working
without path hacks. Each module owns one stage of the workflow and can be unit tested
with plain DataFrames — no file I/O required for the logic tests.

## 2. Configuration surface (`recon/config.py`)

| Setting | Default | Notes |
|---|---|---|
| `orders_path` / `payments_path` | `data/orders_export.xlsx` | CLI overridable |
| `output_path` | `outputs/reconciliation_report.xlsx` | CLI overridable |
| `amount_tolerance` | `0.01` | absolute, same-currency comparison |
| `required_columns` | per file | missing column = fatal, see §4 |
| `expected_order_statuses` | `pending, completed, shipped, cancelled, refunded` | unexpected value -> QA issue |
| `expected_payment_statuses` | `pending, successful, failed, refunded` | unexpected value -> QA issue |
| `expected_currencies` | `EUR, USD, GBP, CHF` | unexpected value -> QA issue |
| `order_id_pattern` / `payment_id_pattern` | `^ORD-\d{6}$` / `^PAY-\d{7}$` | malformed ID -> QA issue |
| `allowed_payment_statuses_by_order_status` | mapping | drives the status inconsistency rule |
| `random_seed` | `42` (generator only) | deterministic anomalies |

Everything a user might tune lives here and is echoed into the Summary sheet
(input file names, tolerance, generation seed) so a report is self-describing.

## 3. Workflow and module mapping

| Stage | Module | Output |
|---|---|---|
| 1. load | `excel_io` | raw frames + row counts logged |
| 2. validate structure | `validation` | fatal `InputValidationError` on missing columns |
| 3. normalize fields | `normalize` | trimmed/cased/typed frames |
| 4. validate rows | `validation` | `qa_issues` frame + clean frames |
| 5. match | `matching` | order-centric match table |
| 6. reconcile | `matching` | one row per (record, failed check) |
| 7. classify | `models` | `ExceptionType` labels per exception row |
| 8. summarize | `metrics` | `SummaryMetrics` dataclass |
| 9. report | `report` | `outputs/reconciliation_report.xlsx` |

## 4. Validation rules

Two levels, deliberately not conflated:

* **File level (fatal).** Missing required column, unreadable file, empty header row ->
  raise `InputValidationError` with the offending column name, log at ERROR, exit code 2.
  A missing key column makes any downstream "result" meaningless, so the run stops
  instead of producing a report that hides the problem. Extra/unknown columns are
  logged as a warning and ignored.
* **Row level (quarantine).** Each offending row is copied into `QA Issues` with
  `source_file`, `source_row` (1-based Excel row, header = 1) and `qa_reason`:
  `missing_required_value`, `invalid_date`, `invalid_numeric`, `malformed_id`,
  `unexpected_category`, `non_positive_amount`.
  Rows with a `qa_reason` are excluded from matching but are always visible in the report.
  If both an order and a payment for the same key are quarantined, both stay in `QA Issues`.

Normalization before validation of values: trim whitespace, uppercase `currency`,
`order_status`, `payment_status`, `payment_method`, strip currency symbols/thousands
separators from amounts, coerce dates with `errors="coerce"` and flag the failures.

## 5. Matching and reconciliation logic

Primary key: `order_id`. Matching is **order-centric with payment aggregation**, because a
single order can legitimately receive several payments and the sample generator will
produce such cases:

1. Drop nothing: start from all valid (non-quarantined) orders and payments.
2. Group valid payments by `order_id`.
3. For each valid order:
   * no payment group -> `MISSING_PAYMENT`
   * otherwise compare `abs(order_amount - sum(payment_amount)) <= tolerance`;
     over tolerance -> `AMOUNT_MISMATCH`
   * any payment currency != order currency -> `CURRENCY_MISMATCH`
   * any payment status not in `allowed_payment_statuses_by_order_status[order_status]`
     -> `STATUS_INCONSISTENCY` (covers cancelled+successful and paid+failed)
   * zero failed checks -> matched record
4. Valid payments whose `order_id` is absent from the (clean or quarantined) orders set
   -> `MISSING_ORDER`.
5. Duplicates are detected on the full valid frame, independently of matching:
   * `DUPLICATE_ORDER`: more than one order row per `order_id`
   * `DUPLICATE_PAYMENT`: repeated `payment_id`, or repeated transaction fingerprint
     (`order_id`, `payment_date`, `payment_amount`, `currency`, `payment_method`)
   * every row in a duplicated group is listed with `duplicate_group_size` and
     `is_primary`, so nothing is hidden; aggregate totals still include them and the
     summary states this explicitly.

Exception rows are cumulative: one record can appear in several check sheets
(e.g. amount + currency mismatch). `All Exceptions` is the union, one row per
(record, exception type), with a `exception_type` column and a stable sort.

## 6. Report workbook

Required sheets, plus one addition:

1. `Summary` — run metadata, totals, matched, exception count, counts by type, reconciliation rate
2. `Matched Records`
3. `Missing Payments`
4. `Missing Orders`
5. `Amount Mismatches`
6. `Duplicates` (`duplicate_type` column)
7. `Currency Mismatches`
8. `Status Inconsistencies` — added because it is a required check with no required sheet
9. `QA Issues`
10. `All Exceptions` (union sheet, the "read this first" tab for reviewers)

Formatting: bold white-on-slate header row, frozen header pane, autofilter,
per-column widths from content, `#,##0.00` amounts, `yyyy-mm-dd` dates, percentage
format for the reconciliation rate, red/amber fills via conditional formatting on
tolerance breaches and exception flags, tab colors per category.

Metric definitions (documented in the README, not left implicit):

* `total_orders` / `total_payments` = rows read from each export (before quarantine)
* `matched_count` = valid orders with zero failed checks
* `exception_count` = distinct records with at least one failed check (union), reported
  next to per-type counts, which can sum higher because checks overlap
* `reconciliation_rate` = `matched_count / total_orders`
* `qa_issue_count` = quarantined rows, reported separately from reconciliation exceptions

## 7. Sample data generator

`python scripts/generate_sample_data.py --orders 7500 --seed 42`

* deterministic via `random.Random(seed)` / seeded numpy generator
* realistic shape: ~7,500 orders, payment volume > order volume, 4 currencies,
  5 sales channels, 4 payment methods, ~18 months of dates, weekend/business-day effects
* exactly N injected anomalies per type (configurable constants), e.g. 120 missing
  payments, 60 missing orders, 90 amount mismatches, 40 duplicate payments, 25 duplicate
  orders, 50 currency mismatches, 45 status inconsistencies, 80 row-level QA issues
* writes `data/generation_manifest.json` with injected counts and IDs, which enables a
  test that asserts the pipeline detects exactly what was injected
* currency-specific amount rounding (2 decimals) and clearly synthetic IDs
  (`ORD-000001`, `PAY-000001`) so no one mistakes it for client data

## 8. Testing plan

| Test file | Covers |
|---|---|
| `test_validation.py` | missing required column (fatal), null required values, invalid dates, invalid numerics, malformed IDs, unexpected categories, source-row tracing |
| `test_normalize.py` | whitespace trimming, case folding, amount cleaning, date coercion |
| `test_matching.py` | 1:1 match, split payments aggregating to the order total, unmatched on both sides, deterministic ordering |
| `test_checks.py` | tolerance edge (exactly at tolerance = matched, one cent over = mismatch), tolerance override, currency mismatch, every status rule, duplicate order/payment, missing order/payment |
| `test_metrics.py` | counts, per-type counts, overlapping exceptions, reconciliation rate, QA counted separately |
| `test_report.py` | sheet names and order, header row presence, freeze panes/filters, numeric and date number formats |
| `test_pipeline_e2e.py` | generated fixtures -> `run.py`-equivalent pipeline -> workbook exists with expected sheet set and consistent summary numbers |

All fixtures are hand-built 5–20 row DataFrames in `conftest.py`; only the E2E test
touches Excel I/O, and it writes to `tmp_path`.

## 9. CLI

`python run.py` with optional flags (defaults from `config.py`):

```
--orders PATH      --payments PATH      --output PATH
--tolerance 0.01   --seed 42            --log-level INFO
--generate-sample  # convenience: build sample exports, then reconcile
```

Exit codes: `0` success, `2` fatal input/validation error, `1` unexpected error
(stack trace logged, short message on stderr).

## 10. Implementation order

1. Scaffolding: `.gitignore`, `requirements.txt`, `pyproject.toml`, `config.py`,
   `models.py`, `logging_setup.py`, git init
2. `scripts/generate_sample_data.py` + manifest
3. `excel_io.py` -> `normalize.py` -> `validation.py`
4. `matching.py` + `metrics.py` (pure functions, no I/O)
5. `report.py` formatting
6. `pipeline.py` + `run.py`
7. tests alongside steps 3–6 (they are written with each module, not at the end)
8. run the seeded pipeline, capture real numbers, write `README.md` from that output
9. final pass: `pytest -q`, one clean end-to-end run, verify workbook opens in openpyxl

## 11. Open decisions to confirm

1. Order-centric aggregated matching (§5) vs. strict 1:1 first-payment matching.
   Recommendation: aggregation — it avoids false "amount mismatch" noise on split payments.
2. Duplicate payments kept inside the order total (transparent, flagged) vs. excluded
   from the total (cleaner amounts, but silently changes numbers). Recommendation: keep + flag.
3. Adding the extra `Status Inconsistencies` sheet (10 tabs instead of 9).
   Recommendation: add it — the check is required but has no required sheet.
4. Missing required column = hard failure vs. empty report with a QA note.
   Recommendation: hard failure with exit code 2.
5. Sample size default of 7,500 orders at seed 42.

## 12. Explicitly out of scope (avoids over-engineering)

Genuine client data, databases, dashboards/BI, PDF export, scheduling, multi-user auth,
currency conversion using FX rates, fuzzy matching of near-miss IDs, and a plugin
framework for custom rules.
