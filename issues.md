# Repowise false positives — test matching / coverage attribution

Date: 2026-08-22 · Index: `87e4d25143bc` (current at time of check)

Repowise's health dashboard repeatedly reports missing tests / missing
coverage for files that are fully covered. Ground truth used here: the
pytest-cov table from the fast suite (`pytest tests/ -q`, total 89.47%,
run 2026-08-22 at HEAD 87e4d25). Rule of thumb until these are fixed:
**any repowise "untested_hotspot" / "0% coverage" claim for this repo must
be re-verified with `pytest --cov` before acting on it.**

## A. Outright wrong coverage claims

### A1 — All package `__init__.py` files reported as "100% of lines uncovered (0% line coverage)"
- **Claim (repowise):** `coverage_gradient` (severity: high, health_impact
  2.0) fired on 12 files: `src/arbfree_vol/__init__.py`, `arbitrage/`,
  `data/`, `models/`, `plotting/`, `pricing/`, `repair/`, `sabr/`, `ssvi/`,
  `svi/`, `viz/`, (and `ssvi`-subpackage) `__init__.py`.
- **Reality (pytest):** every one of them is 100% covered — they contain
  0–1 statements (e.g. `ingestion/__init__.py`: 1 stmt, 1 covered;
  `arbfree_vol/__init__.py`: 0 stmts). There is nothing uncovered.
- **Likely cause:** the coverage-map ingestion does not attribute executions
  to files with (near-)zero statements and defaults them to "0%".
- **Impact:** 12 × high-severity deductions drag `average_health` (8.01)
  and can put files in the wrong band; also inflates `top_findings` noise
  (268 findings total, ~12 of the top ones are these).

### A2 — `ssvi/term_structure.py` reported as untested_hotspot (CRITICAL, "11 dependents")
- **Claim:** "Hotspot with no paired test file and no coverage data".
- **Reality:** 100% covered (112/112 stmts). Tests:
  `tests/test_ssvi_term_structure.py`, `tests/test_essvi_term_structure.py`,
  plus heavy use via `tests/test_diagnose_fallback_slices.py`.
- **Likely cause:** test-file pairing expects `tests/test_<stem>.py`
  (`test_term_structure.py`); the actual names carry module prefixes.

### A3 — `viz/surface.py` reported as untested_hotspot ("5 dependents")
- **Claim:** "no paired test file and no coverage data".
- **Reality:** 97% covered (122 stmts, 4 missed). Tests:
  `tests/test_viz.py`, `tests/test_viz_invariants.py`.

## B. Pairing failures (`has_test_file: false`) — coverage itself was known

These did not produce "untested" biomarkers (coverage data existed), but the
paired-test metadata is wrong, which corrupts downstream features such as
`tests_to_run` in PR-risk mode:

| File | Repowise | Reality (pytest cov) | Actual tests |
|---|---|---|---|
| `repair/engine.py` | has_test_file: false | 100% | `tests/test_repair_engine.py` (963 NLOC) |
| `arbitrage/quote_detect.py` | has_test_file: false | 100% | `tests/test_arbitrage.py` |
| `ingestion/yahoo.py` | has_test_file: false | 100% | `tests/test_yfinance.py`, `tests/test_ingestion_yfinance.py` |
| `ingestion/_index_rates.py` | has_test_file: false | 98% | `tests/test_index_rates.py` |
| `svi/calibration.py` | has_test_file: false | 100% (module) / 90.14% (file) | `tests/test_svi.py` |
| `ssvi/model.py` | has_test_file: false | 100% | `tests/test_ssvi.py` |

**Pattern:** every miss is a test file that covers *multiple* modules or
uses a name other than the bare `test_<stem>.py` convention.

## C. Adjacent matching false positives (dead-code scanner)

- `_expiry_to_date_str` (`ingestion/openbb.py:80`) reported
  "Private symbol has no callers" — it is called at `openbb.py:168` and
  unit-tested at `tests/test_openbb.py:142`.
- "Zombie package `demo/`" / "unreachable" demo+script files: these are
  entry points by design (drivers with `__main__`), not dead code.

## Handling guidance

1. Treat the pytest-cov table as the single source of truth for coverage in
   this repo; ignore repowise coverage *deductions* for the files above.
2. The genuinely uncovered code remains what the audit found:
   `data/audit.py` 40%, `rates/curve.py` 57%, `ssvi/diagnostics.py` 74%,
   `time/__init__.py` 77% — those are real.
3. Upstream fixes that would eliminate most of the above: (a) treat
   0-statement files as covered rather than 0%; (b) pair tests by import
   graph (which test modules import the file) rather than filename
   convention; (c) same import-graph basis for "no callers" scans.
