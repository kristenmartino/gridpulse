"""No published surface may assert a standing scoreable count (#535).

`docs/BENCHMARK_METHODOLOGY.md` said "**44 of 51 BAs scoreable**" in the
present tense for three weeks while `/api/v1/benchmark` served **25**, and
`docs/BENCHMARK_SCOREABILITY.md` — the file the methodology pointed at for the
detail — said the same. Both were true when written on 2026-07-27 and neither
had any way to notice it had stopped being true: the doc is generated on demand
by a script that needs GCS credentials, and prose does not recompute.

So the rule is not "keep the number up to date". A number recomputed every hour
cannot be tracked by a sentence, and the previous fix — write the correct
figure — is the one that already failed. The rule is that **the count lives in
exactly one place, the live payload**, and every document either omits it or
stamps it with the date it was measured.

This is the `test_public_copy_traces_to_canonical_facts` lesson applied to a
figure that has no canonical *file* to trace to, because its source is an API.
Where that test asserts a page still matches its source doc, this one asserts
no page claims to BE the source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

#: Surfaces a reader can reach that discuss benchmark scoreability.
_SURFACES = [
    Path("docs/BENCHMARK_METHODOLOGY.md"),
    Path("docs/BENCHMARK_SCOREABILITY.md"),
    Path("web/benchmark.html"),
    Path("README.md"),
    Path("PRD.md"),
]

#: A **bolded** "N of 51 ... scoreable/excluded" claim — the assertive form.
#: Matched within a single line so a `**` ending one emphasis cannot pair with
#: a `**` opening the next one two lines down.
#:
#: The inverse phrasing is included because "**7 of 51 BAs are excluded**"
#: asserts the count just as firmly, and is how §11's "Not fleet-complete"
#: bullet carried a stale 7 alongside the stale 44.
_STANDING_CLAIM = re.compile(
    r"\*\*[^*\n]*?\b\d{1,2}\s+of\s+51\b[^*\n]*?(?:scoreable|exclud)[^*\n]*?\*\*",
    re.IGNORECASE,
)

#: A claim that carries the date it was measured is not a standing claim — it
#: is a dated observation, which is exactly the form this file is arguing for.
#: `BENCHMARK_SCOREABILITY.md`'s generated lede is the intended shape.
_DATED = re.compile(r"as measured on\s+\d{4}-\d{2}-\d{2}|as of\s+\d{4}-\d{2}-\d{2}", re.IGNORECASE)


@pytest.mark.parametrize("rel", _SURFACES, ids=lambda p: str(p))
def test_no_surface_asserts_a_standing_scoreable_count(rel: Path):
    path = _ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    text = path.read_text()
    undated = [hit for hit in _STANDING_CLAIM.findall(text) if not _DATED.search(hit)]
    assert not undated, (
        f"{rel} asserts a standing scoreable/excluded count: {undated!r}. That "
        "number is recomputed every scoring tick and this file cannot track it — "
        "which is #535. Either drop the figure and point at `n_scoreable` on "
        "/api/v1/benchmark, or stamp it with the date it was measured."
    )


def test_the_snapshot_says_it_is_a_snapshot():
    """`BENCHMARK_SCOREABILITY.md` is generated, dated, and self-describing.

    It is allowed to carry a count — it is the per-BA detail table, and a table
    with no figures is useless. What it may not do is present that count as
    current, which is why the generator stamps it and names the API as the
    authority. Asserted here rather than trusted to the generator, because the
    committed file is what a reader actually opens.
    """
    text = (_ROOT / "docs/BENCHMARK_SCOREABILITY.md").read_text()
    assert "As measured on" in text, "the count must be dated, not stated"
    assert "/api/v1/benchmark" in text, "the live payload must be named as the authority"
    assert "this file is the stale one" in text, (
        "the doc must say which side loses a disagreement — otherwise a reader "
        "who finds two different numbers has no way to pick"
    )


def test_the_snapshot_publishes_both_coverages():
    """The BA's publication rate and our capture rate, side by side.

    Publishing only the first is precisely how #535 hid: `df_coverage` read
    58% for NYISO while EIA published for 96.7% of hours, and no committed
    artifact carried the second number to contradict it.
    """
    text = (_ROOT / "docs/BENCHMARK_SCOREABILITY.md").read_text()
    assert "df_coverage_pct" in text and "df_asissued_pct" in text
