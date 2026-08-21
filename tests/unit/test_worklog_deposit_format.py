"""Every file in `.mistakes/worklog/` must follow the deposit format.

`.mistakes/worklog/README.md` documents the contract: a filename of
`<UTC timestamp>-<category>.md` and a body holding a `[category]` tag. Both
halves are "best guess" -- the audit pass decides the real root cause and a
depositor's filename slug does not have to match their own inline tag (using
an *existing* graduated category name inline, e.g. `[claim-shipped-before-
measurement]`, under a more specific filename is normal and good). What is
not optional is having a tag at all: a deposit with no `[category]` gives a
decontextualized audit pass nothing structured to group by, and forces it to
infer one from prose alone. Caught 2026-08-21 when a real deposit
(`verification-checked-the-wrong-thing` by filename) turned out to have no
bracket tag anywhere in its body.

Asserted on every file currently in the directory, not a fixture snapshot --
the directory's contents change on every deposit and every audit consumption,
so a static fixture would drift out of sync with what is actually enforced.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKLOG_DIR = REPO / ".mistakes" / "worklog"

#: `date -u +%Y-%m-%dT%H%M%SZ` plus `-<category>`, per the README. Requiring
#: a literal `Z-` immediately after the seconds digits is what rejects the
#: `...Z1-` suffix trap the README calls out by name -- a stray character
#: between the timestamp and the dash is not this pattern.
FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")

#: A `[category]` tag: square brackets around a lowercase-kebab slug.
CATEGORY_TAG_RE = re.compile(r"\[[a-z][a-z0-9]*(?:-[a-z0-9]+)*\]")


def _deposit_files() -> list[pathlib.Path]:
    return sorted(p for p in WORKLOG_DIR.glob("*.md") if p.name != "README.md")


def test_the_worklog_directory_has_deposits_to_check():
    """The checks below are vacuously true on an empty directory -- a guard
    that never runs its assertions is indistinguishable from one that
    passed them, so confirm there is something here to actually check."""
    assert WORKLOG_DIR.is_dir()


def test_every_deposit_filename_matches_the_documented_timestamp_format():
    for path in _deposit_files():
        assert FILENAME_RE.match(path.name), (
            f"{path.name}: filename does not match "
            f"<YYYY-MM-DDTHHMMSSZ>-<category>.md — the audit and the nudge "
            f"hook both parse this leading stamp to decide what is new"
        )


def test_every_deposit_body_carries_a_category_tag():
    for path in _deposit_files():
        text = path.read_text()
        assert CATEGORY_TAG_RE.search(text), (
            f"{path.name}: body has no `[category]` tag — a decontextualized "
            f"audit pass needs a structured guess to group by, not just prose"
        )


def test_every_deposit_body_is_a_single_nonempty_entry():
    """`README.md` says the file holds exactly one line. A leading `- ` list
    marker is common and harmless (several real deposits carry one); this
    only guards against the file being empty or split across meaningfully
    different content blocks, not against markdown decoration."""
    for path in _deposit_files():
        text = path.read_text().strip()
        assert text, f"{path.name}: deposit is empty"
