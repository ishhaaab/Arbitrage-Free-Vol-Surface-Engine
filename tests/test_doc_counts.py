"""Guard the suite-size claims in docs/AGENTS.md.

``docs/AGENTS.md`` states the current suite size ("53 test files, 628
tests").  The claim and the actual suite drift apart whenever files or
tests are added or removed, so this test pins them together: refresh the
numbers with ``pytest --collect-only -q`` and update ``docs/AGENTS.md``
and the ``_EXPECTED_TESTS`` constant in the same change.

Same philosophy as ``test_snapshot_guard`` — fail loudly, never skip.
"""

import re
from pathlib import Path

_DOC = Path(__file__).resolve().parent.parent / "docs" / "AGENTS.md"

# Refresh with `pytest --collect-only -q`:
#   tests   = collected items (725 = 711 passing + 14 deselected, 2026-08-22)
#   files   = derived from the tree below, no constant needed
_EXPECTED_TESTS = 725


def _suite_claim() -> tuple[int, int]:
    text = _DOC.read_text(encoding="utf-8")
    m = re.search(r"(\d+) test files, (\d+) tests", text)
    assert m is not None, (
        "docs/AGENTS.md no longer states the suite size "
        "('N test files, M tests')"
    )
    return int(m.group(1)), int(m.group(2))


def _test_files() -> list[Path]:
    return sorted(
        p
        for p in Path(__file__).parent.rglob("*.py")
        if p.name not in ("__init__.py", "conftest.py")
    )


def test_agents_md_file_count_matches_tree() -> None:
    """AGENTS.md's test-file count must match the *.py files under tests/."""
    claimed, _ = _suite_claim()
    actual = len(_test_files())
    assert claimed == actual, (
        f"docs/AGENTS.md test-file count is stale: claims {claimed}, "
        f"tree has {actual}. Update the doc and this guard together."
    )


def test_agents_md_test_count_matches_collection() -> None:
    """AGENTS.md's collected-test count must match the pinned number."""
    _, claimed = _suite_claim()
    assert claimed == _EXPECTED_TESTS, (
        f"docs/AGENTS.md test count is stale: claims {claimed}, "
        f"expected {_EXPECTED_TESTS} (from `pytest --collect-only -q`). "
        "Update the doc and this guard together."
    )
