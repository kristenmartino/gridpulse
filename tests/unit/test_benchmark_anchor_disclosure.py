"""The methodology must keep disclosing that our anchor can be seeded by `DF` (#539).

`forecast_mw` is not a model feature, and ADR-009's anchor substitution is
scoped to the `broken` class — both true, both checked elsewhere. What was
undisclosed is a second path to the same self-reference, on BAs the benchmark
*scores*: for an hour EIA has not metered yet it publishes the BA's day-ahead
value in the `D` field, and `_resolve_forecast_start` anchors on the last
*positive* `D`, not the last *metered* one.

The asymmetry is the point. `pair_hours` drops those hours from scoring, which
protects the truth side; nothing drops the hour that seeded the forecast. The
methodology named this exact objection for broken feeds — as a reason to
exclude them — and was silent about it for the scored set, which is what made
the silence read as a claim of independence.

The failure mode this file guards is not a wrong number. It is a later
rewrite quietly dropping the disclosure while the dependence stays, which is
how §5 came to imply the exclusion had disposed of it in the first place. So
the assertions are on the *claims*, not on any figure: the rates themselves are
recomputed hourly and live in the payload (`placeholder_pct`), never here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_METHODOLOGY = Path(__file__).resolve().parents[2] / "docs" / "BENCHMARK_METHODOLOGY.md"


@pytest.fixture(scope="module")
def doc() -> str:
    return _METHODOLOGY.read_text()


def test_the_anchor_dependence_is_stated(doc: str) -> None:
    """Naming the mechanism, not just the word — a reader has to be able to
    tell *how* their forecast reaches our input."""
    lowered = doc.lower()
    assert "_resolve_forecast_start" in doc
    assert "demand_lag_1h" in doc
    assert "placeholder_pct" in doc


def test_the_scoring_protection_is_not_presented_as_covering_the_anchor(doc: str) -> None:
    """The one-sidedness is the finding. A doc that mentions the drop rule and
    stops has restated §4, not disclosed §12's limit 11."""
    lowered = doc.lower()
    assert "one-sided" in lowered or "only the first is protected" in lowered
    assert "seeded" in lowered


def test_the_direction_of_the_effect_is_stated(doc: str) -> None:
    """Undisclosed *and* unsigned would be the worse reading. The direction is
    what makes this a disclosure rather than a bias we would have to correct:
    it correlates our error with theirs, it does not shrink ours."""
    lowered = doc.lower()
    assert "correlated" in lowered or "correlates" in lowered
    assert "does not make ours smaller" in lowered


def test_refusing_to_anchor_is_recorded_as_measurably_worse(doc: str) -> None:
    """The obvious "fix" is the trap, and it was measured before it was
    rejected (`data/vintage.py`: 6.55% against 7.72%, 9 of 12 BAs). Without
    those figures in the doc, the next reader re-derives the wrong conclusion
    from the disclosure this file exists to protect."""
    assert "6.55" in doc and "7.72" in doc


def test_broken_feed_exclusion_does_not_read_as_disposing_of_it(doc: str) -> None:
    """§5's row gives the ADR-009 self-reference as a *reason to exclude*.
    Left alone, that implies the scored set is free of it — the implication
    this issue was raised to correct."""
    assert "#539" in doc
    lowered = doc.lower()
    assert "must not be read as disposing" in lowered
