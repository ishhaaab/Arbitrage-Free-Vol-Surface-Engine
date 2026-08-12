"""Independent ground-truth test module.

This package holds hand-derived ground-truth cases and the tests that assert
the repo's behavior AGREES with them.  The core anti-circularity rule:

- LABELS are derived by hand in the case definitions (analytic literals:
  the Gatheral-Jacquier bound arithmetic, the Hendriks-Martini Prop 3.1
  inequalities, the Dupire closed forms).
- The tests then assert the repo's detectors / verifiers / calibrators
  agree with the hand-derived label.
- If a case's repo behavior differs from the hand-derived label, the
  module STOPs and reports a potential real bug with evidence instead of
  weakening the label.

Module layout
-------------
- ``cases.py``             — the shared ``GroundTruthCase`` schema.
- ``arbitrage_cases.py``   — hand-derived eSSVI/SVI arbitrage labels.
- ``calibration_cases.py`` — parameter-recovery cases (SVI/eSSVI/SABR).
- ``dupire_cases.py``      — Dupire closed-form cases (constant-vol, linear-in-k).
- ``fit_quality.py``       — per-slice fit-quality harness (model IV vs mid IV).
- ``test_*_ground_truth.py`` — the tests asserting repo agreement.
"""
