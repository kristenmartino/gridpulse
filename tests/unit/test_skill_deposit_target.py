"""A skill's frontmatter and its body must name the same deposit target.

The mistake Worklog moved from a list inside ``MISTAKES.md`` to one file per
deposit under ``.mistakes/worklog/`` (#588). For a stretch afterwards
``check-past-mistakes`` said **both**: its frontmatter named the directory while
its body still said "add exactly one line to ``MISTAKES.md``'s ``## Worklog``".
Those two are read by different consumers — the harness skill listing renders the
frontmatter, an agent that opens the file follows the body — so an agent would
deposit in one place or the other depending on which half it happened to read,
and a deposit written into the old list is invisible to the nudge hook and the
audit, both of which enumerate the directory.

It was not authored wrong. #588 fixed both halves; a later squash merge dropped
part of that change from ``main`` and #598 had to restore it. So the failure mode
this guards is **silent partial loss of a doc change**, which no amount of care
at authoring time prevents and which a reviewer reading the diff will not see.

Asserted on the *source* files rather than on behaviour, for the same reason
``test_public_copy_traces_to_canonical_facts.py`` inverts its check: the thing
that breaks is the document, so the failure has to travel from the document.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Skills that tell a session where to record a mistake candidate.
DEPOSIT_SKILLS = [
    ".claude/skills/check-past-mistakes/SKILL.md",
    ".claude/skills/audit-mistakes-log/SKILL.md",
]

#: The canonical location, as stated by `.mistakes/worklog/README.md`.
WORKLOG_DIR = ".mistakes/worklog"

#: Phrasings that route a deposit back into the retired inline list. Deliberately
#: narrow — `MISTAKES.md` is still where Analyzed entries live and is mentioned
#: legitimately all over these files, so only *deposit-destination* wording counts.
#:
#: Matched against NORMALISED text (see ``_normalise``). The first draft of this
#: guard matched the raw source and missed the very break it was written for:
#: the real instruction read ``Add **exactly one line** to`` and the ``**``
#: emphasis defeated a ``line to`` pattern. Markdown decoration is not a
#: meaningful difference and must not be one the guard can trip over.
STALE_DEPOSIT_PHRASINGS = [
    r"one line to mistakes\.md",
    r"line to mistakes\.md's #* ?worklog",
    r"worklog has accumulated a handful of new lines",
    r"add(?:ing)? (?:an? )?(?:entry|line) to mistakes\.md",
    # The audit's SUBJECT is the worklog directory; it only WRITES to MISTAKES.md.
    # Describing it as an "audit of MISTAKES.md" sends a reader to the wrong file
    # for the pending candidates, which is the same drift one level up.
    r"audit of mistakes\.md",
]


def _normalise(text: str) -> str:
    """Strip markdown decoration and wrapping so phrasing is what is matched.

    Emphasis, code spans and line breaks are all things an author changes
    freely without changing meaning — a guard that they can defeat is worse
    than none, because it reports green while the claim underneath has moved.
    """
    stripped = re.sub(r"[*`_]", "", text)
    return re.sub(r"\s+", " ", stripped).lower()


def _read(rel: str) -> str:
    path = REPO / rel
    assert path.exists(), f"{rel} is missing — this guard names a file that moved"
    return path.read_text()


def _split_frontmatter(text: str) -> tuple[str, str]:
    """``(frontmatter, body)``. Both halves are consumed, by different readers."""
    parts = text.split("---", 2)
    assert len(parts) == 3, "expected YAML frontmatter delimited by ---"
    return parts[1], parts[2]


@pytest.mark.parametrize("rel", DEPOSIT_SKILLS)
def test_frontmatter_and_body_agree_on_the_deposit_target(rel):
    """The half a reader happens to see must not change where they deposit."""
    frontmatter, body = _split_frontmatter(_read(rel))
    assert WORKLOG_DIR in frontmatter, f"{rel}: frontmatter does not name {WORKLOG_DIR}"
    assert WORKLOG_DIR in body, f"{rel}: body does not name {WORKLOG_DIR}"


@pytest.mark.parametrize("rel", DEPOSIT_SKILLS)
def test_no_skill_routes_a_deposit_into_the_retired_inline_list(rel):
    text = _normalise(_read(rel))
    for pattern in STALE_DEPOSIT_PHRASINGS:
        found = re.search(pattern, text)
        assert not found, f"{rel} still routes deposits at MISTAKES.md: {found.group(0)!r}"


def test_claude_md_names_the_directory():
    """CLAUDE.md is loaded into every session, so a stale line here is the
    costliest of the set — it is the copy an agent reads without opening
    anything."""
    raw = _read("CLAUDE.md")
    assert WORKLOG_DIR in raw
    text = _normalise(raw)
    for pattern in STALE_DEPOSIT_PHRASINGS:
        found = re.search(pattern, text)
        assert not found, f"CLAUDE.md still routes deposits at MISTAKES.md: {found.group(0)!r}"


def test_mistakes_md_worklog_section_points_at_the_directory():
    """The old location must forward rather than silently accept entries.

    Someone who goes to the section by memory has to be sent on, or the deposit
    lands where nothing reads it.
    """
    text = _read("MISTAKES.md")
    section = text.split("## Worklog", 1)
    assert len(section) == 2, "MISTAKES.md has no Worklog section to forward from"
    forward = section[1].split("## ", 1)[0]
    assert WORKLOG_DIR in forward, "the Worklog section does not point at the directory"
    inline = re.findall(r"^- 20\d\d-\d\d-\d\d \[", forward, re.MULTILINE)
    assert not inline, f"{len(inline)} entries still inline in MISTAKES.md; they are unread there"


def test_the_worklog_directory_and_its_readme_exist():
    """The guard above is only meaningful if the target it points at is real."""
    assert (REPO / WORKLOG_DIR).is_dir()
    assert (REPO / WORKLOG_DIR / "README.md").exists()


class TestTheNormaliserIsWhatMakesThisGuardWork:
    """The first draft failed because decoration defeated the match. Pin it."""

    @pytest.mark.parametrize(
        "decorated",
        [
            "Add **exactly one line** to `MISTAKES.md`'s `## Worklog`",
            "Add exactly one line to MISTAKES.md's ## Worklog",
            "add exactly one line to\n   MISTAKES.md's Worklog",
            "Add *one line* to __MISTAKES.md__'s Worklog",
        ],
    )
    def test_decoration_and_wrapping_do_not_hide_a_stale_instruction(self, decorated):
        text = _normalise(decorated)
        assert any(re.search(p, text) for p in STALE_DEPOSIT_PHRASINGS), (
            f"normalisation let this through: {decorated!r}"
        )

    def test_a_legitimate_mistakes_md_mention_is_not_flagged(self):
        """Analyzed entries live in MISTAKES.md; saying so must stay allowed."""
        text = _normalise("Add the Analyzed entry to `MISTAKES.md`, then stamp the marker.")
        assert not any(re.search(p, text) for p in STALE_DEPOSIT_PHRASINGS)
