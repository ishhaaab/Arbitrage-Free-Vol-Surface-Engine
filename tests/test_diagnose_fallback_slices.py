"""Regression tests for scripts/diagnose_fallback_slices.py helpers.

These cover the predecessor-selection edge cases that the script's
diagnostic loop relies on: unsorted fitted-slice input and repeated
parameter-object identity across maturities.
"""

import sys
from pathlib import Path

import pytest

_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from diagnose_fallback_slices import _find_predecessor  # noqa: E402


class _P:
    """Minimal stand-in for SSVIParams — identity is what matters here."""

    def __init__(self, theta):
        self.theta = theta


def test_find_predecessor_returns_last_below_T() -> None:
    fitted = [(0.1, _P(1)), (0.2, _P(2)), (0.3, _P(3))]
    params, T = _find_predecessor(fitted, 0.25)
    assert T == 0.2
    assert params.theta == 2


def test_find_predecessor_unsorted_input() -> None:
    # Input order must not matter: selection is by maturity.
    fitted = [(0.3, _P(3)), (0.1, _P(1)), (0.2, _P(2))]
    params, T = _find_predecessor(fitted, 0.25)
    assert T == 0.2
    assert params.theta == 2


def test_find_predecessor_returns_none_when_no_predecessor() -> None:
    fitted = [(0.1, _P(1)), (0.2, _P(2))]
    params, T = _find_predecessor(fitted, 0.05)
    assert params is None
    assert T is None


def test_find_predecessor_duplicate_identity_returns_matching_T() -> None:
    # A single params object shared by two maturities: the returned T must
    # be the maturity actually selected (the last one strictly below T),
    # not a re-derived match from an unsorted scan.
    shared = _P(7)
    fitted = [(0.1, shared), (0.2, shared), (0.3, _P(3))]
    params, T = _find_predecessor(fitted, 0.25)
    assert params is shared
    assert T == 0.2
