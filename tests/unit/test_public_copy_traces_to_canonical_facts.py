"""Every figure published on a public page still matches its source doc.

This file exists because the test it replaces asserted the drift instead of
catching it. ``test_landing.py::test_numbers_are_the_canonical_ones`` claimed
in its docstring that "every number traces to docs/CANONICAL_FACTS.md" — but
it only asserted the literal ``"4.8%"`` was *present in the page*. It never
read CANONICAL_FACTS. So when the 2026-08-07 retrain moved the served-ensemble
median to 4.35%, nothing failed, the page kept publishing 4.8% for four days,
and the test actively blocked the fix.

The assertion is therefore inverted. Each published figure is checked twice:

1. it still appears **in the page** (the page wasn't silently reworded), and
2. it still appears **in the source doc** (the source hasn't moved underneath).

Check 2 is the one that matters. When a retrain regenerates CANONICAL_FACTS,
this fails on the *source* side and names the page that has to change — which
is the direction the failure needs to travel. A test that only reads the page
can never notice that the world moved.

The sweep at the bottom covers the other half: a *new* unsourced percentage
being added to a public page. A fixed registry cannot see that by itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_WEB = _ROOT / "web"
_DOCS = _ROOT / "docs"

#: published literal -> the doc that sources it
#:
#: Keyed by literal rather than by page because a figure may legitimately
#: appear on more than one public page — the served-ensemble median is on
#: both /about and /methodology, and both must move together when it does.
#:
#: Keep the literal in the exact form BOTH files write it. If the page says
#: "4.35%" and the doc says "4.4%", that is a drift to resolve, not a reason
#: to loosen the match. Rounding a published figure for prose is how the
#: page and the source quietly stop meaning the same thing.
_PUBLIC_FIGURES: dict[str, Path] = {
    # ── Holdout accuracy, 168h recursive, per BA ──
    # Never quoted as a pooled across-51 figure (CANONICAL_FACTS
    # "Forecast accuracy" opens with that rule).
    "4.35%": _DOCS / "CANONICAL_FACTS.md",  # served ensemble, median
    "3.69%": _DOCS / "CANONICAL_FACTS.md",  # best-base, median
    "3.89%": _DOCS / "CANONICAL_FACTS.md",  # ensemble median, in-sample (withdrawn)
    "1.72%": _DOCS / "CANONICAL_FACTS.md",  # best-base, min (ERCOT)
    "23.26%": _DOCS / "CANONICAL_FACTS.md",  # best-base, max (SPA)
    "14.27%": _DOCS / "CANONICAL_FACTS.md",  # ensemble p90
    "9.87%": _DOCS / "CANONICAL_FACTS.md",  # XGBoost-alone p90
    # ── The withdrawn tail-robustness claim, and what replaced it ──
    "38.63%": _DOCS / "CANONICAL_FACTS.md",  # the figure the old claim rested on
    "13.61%": _DOCS / "CANONICAL_FACTS.md",  # ...and its counterpart
    "13.68%": _DOCS / "CANONICAL_FACTS.md",  # SEC under XGBoost now
    "14.72%": _DOCS / "CANONICAL_FACTS.md",  # SEC under the ensemble now — worse
    # ── Coverage and system facts ──
    "100%": _DOCS / "CANONICAL_FACTS.md",  # demand coverage, contiguous lower-48
    "51": _DOCS / "CANONICAL_FACTS.md",  # balancing authorities covered
    # ── Study results ──
    "27%": _DOCS / "HOW_IT_WORKS.md",  # vintages that dive in the serve regime
    "6.96%": _DOCS / "HOW_IT_WORKS.md",  # the visibility-gate generosity example
    "70%": _DOCS / "BENCHMARK_METHODOLOGY.md",  # first-published EIA revision error
}

#: Percentages that are presentation, not claims — they appear in prose-ish
#: positions but describe the page itself rather than the product.
_NON_CLAIM_PERCENTAGES = frozenset({"30%"})  # --accent-ring alpha, in a CSS comment


def _prose_of(path: Path) -> str:
    """Page text with <style> blocks and HTML comments removed.

    Both public pages inline their design tokens (the landing.py decoupling
    rationale), so a raw grep for percentages is dominated by widths, alphas
    and gradient stops. Those are not published claims and must not be
    registry entries — stripping them is what keeps the sweep meaningful
    rather than a wall of exemptions.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return text


def _public_pages() -> list[Path]:
    return sorted(_WEB.glob("*.html"))


@pytest.mark.parametrize(("literal", "source"), sorted(_PUBLIC_FIGURES.items()))
class TestPublishedFiguresTraceToSource:
    def test_literal_is_still_published_somewhere(self, literal, source) -> None:
        """Some public page still publishes the figure the registry claims.

        Keeps the registry from accumulating entries for copy that has been
        rewritten — a stale allow-list would silently widen the sweep below.
        """
        pages = [p.name for p in _public_pages() if literal in _prose_of(p)]
        assert pages, (
            f"No public page publishes {literal!r} any more. If the copy was "
            f"reworded, drop the entry from _PUBLIC_FIGURES."
        )

    def test_literal_is_still_in_the_source_doc(self, literal, source) -> None:
        """The source still says what the pages claim it says.

        THIS is the assertion that catches drift. It fails when a retrain
        regenerates the source and the published copy was not updated with
        it — naming every page that has to change so the fix is mechanical.
        """
        pages = [p.name for p in _public_pages() if literal in _prose_of(p)]
        assert literal in source.read_text(encoding="utf-8"), (
            f"{source.name} no longer contains {literal!r}, but {pages} still "
            f"publish it. The source moved — update those pages to the new "
            f"value, then update this registry."
        )


def test_every_published_percentage_is_registered() -> None:
    """No unsourced percentage may appear in public prose.

    The registry above can only defend figures someone thought to register.
    This catches the other failure: a new number added to a public page with
    no traceable source behind it.
    """
    registered = set(_PUBLIC_FIGURES) | _NON_CLAIM_PERCENTAGES
    for page in sorted(_WEB.glob("*.html")):
        found = set(re.findall(r"\d+(?:\.\d+)?%", _prose_of(page)))
        unregistered = found - registered
        assert not unregistered, (
            f"{page.name} publishes {sorted(unregistered)} with no source. "
            f"Add each to _PUBLIC_FIGURES with the doc it traces to, or to "
            f"_NON_CLAIM_PERCENTAGES if it is not a claim about the product."
        )


def test_no_public_page_publishes_a_count_of_adrs() -> None:
    """Counts are the highest-drift, lowest-value claim form.

    Both stale claims this file was written for were counts: "Ten
    architecture decision records" (there are 13) and "the full list of
    eight" limits (there are ten, and the same page said ten 80 lines
    later). A count buys nothing a reader wants and goes stale every time
    the underlying list grows.
    """
    countable = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|\d+)\s+(architecture decision|decision record|"
        r"known limit)",
        re.I,
    )
    for page in sorted(_WEB.glob("*.html")):
        match = countable.search(_prose_of(page))
        assert match is None, (
            f"{page.name} publishes a count: {match.group(0)!r}. Drop the "
            f"number — say 'every architecture decision' / 'the full list'."
        )
