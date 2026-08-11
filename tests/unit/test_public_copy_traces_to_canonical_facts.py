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

#: published literal -> (page it appears on, doc that sources it)
#:
#: Keep the literal in the exact form BOTH files write it. If the page says
#: "4.35%" and the doc says "4.4%", that is a drift to resolve, not a reason
#: to loosen the match.
_PUBLIC_FIGURES: dict[str, tuple[Path, Path]] = {
    # Served-ensemble median per-BA holdout error, 168h recursive.
    # Never a pooled across-51 figure (CANONICAL_FACTS "Forecast accuracy").
    "4.35%": (_WEB / "landing.html", _DOCS / "CANONICAL_FACTS.md"),
    # Demand coverage of the contiguous-US lower 48.
    "100%": (_WEB / "landing.html", _DOCS / "CANONICAL_FACTS.md"),
    # Balancing authorities covered.
    "51": (_WEB / "landing.html", _DOCS / "CANONICAL_FACTS.md"),
    # How wrong first-published EIA values run on high-revision feeds.
    "70%": (_WEB / "benchmark.html", _DOCS / "BENCHMARK_METHODOLOGY.md"),
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


@pytest.mark.parametrize(
    ("literal", "page", "source"),
    [(lit, page, src) for lit, (page, src) in _PUBLIC_FIGURES.items()],
)
class TestPublishedFiguresTraceToSource:
    def test_literal_is_still_on_the_page(self, literal, page, source) -> None:
        """The page still publishes the figure the registry says it does."""
        assert literal in _prose_of(page), (
            f"{page.name} no longer publishes {literal!r}. If the copy was "
            f"reworded, update _PUBLIC_FIGURES; if the figure was dropped, "
            f"remove its entry."
        )

    def test_literal_is_still_in_the_source_doc(self, literal, page, source) -> None:
        """The source still says what the page claims it says.

        THIS is the assertion that catches drift. It fails when a retrain
        regenerates the source and the published copy was not updated with
        it — naming both files in the message so the fix is mechanical.
        """
        assert literal in source.read_text(encoding="utf-8"), (
            f"{source.name} no longer contains {literal!r}, but "
            f"{page.name} still publishes it. The source moved — update the "
            f"page to the new value, then update this registry."
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
