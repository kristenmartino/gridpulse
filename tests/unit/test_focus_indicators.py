"""Focus indicators across the app's own stylesheet.

The app defines ONE focus ring, on the universal selector::

    *:focus-visible { outline: none;
                      box-shadow: 0 0 0 2px var(--bg-base),
                                  0 0 0 4px var(--accent-base); }

A per-component override that only sets ``outline`` does not replace that
ring — it *adds* a second indicator on top of it. That is how
``.gp-region-card`` ended up drawing the standard ring plus a redundant
30%-alpha outline 2px outside it, which is muddy rather than invisible.

The rule these tests encode: a component may opt out of the shared ring, but
it must then draw a compliant one of its own. ``--accent-ring`` is a 30%-alpha
token — correct as the *shadow* colour it was defined for, and a WCAG 1.4.11
failure (1.46:1 on ``--bg-base``) if used as an outline colour.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = REPO_ROOT / "assets" / "custom.css"

#: Selector → declaration body, for every ``:focus-visible`` rule in the sheet.
_RULE = re.compile(r"([^{}]*:focus-visible[^{}]*)\{([^}]*)\}", re.MULTILINE)
#: Comments are stripped first — this sheet documents its rules heavily, and a
#: comment block sits between the previous rule's ``}`` and the selector, so it
#: would otherwise be captured as part of the selector text.
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _focus_rules() -> list[tuple[str, str]]:
    css = _COMMENT.sub("", CSS_PATH.read_text())
    return [(m.group(1).strip(), m.group(2)) for m in _RULE.finditer(css)]


class TestFocusIndicators:
    def test_the_shared_ring_exists_and_is_opaque(self):
        """Everything else depends on this one rule being intact."""
        shared = next((body for sel, body in _focus_rules() if sel == "*:focus-visible"), None)
        assert shared is not None, "the universal focus ring is gone"
        assert "--accent-base" in shared, "the shared ring must use the opaque accent"
        assert "box-shadow" in shared

    def test_no_component_outlines_focus_with_the_alpha_token(self):
        """``--accent-ring`` is 30% alpha: 1.46:1 on --bg-base, against the
        3:1 WCAG 1.4.11 requires of a focus indicator. It is fine as the
        shadow colour it was named for, and wrong as an outline colour.

        ``.gp-region-card`` did exactly this. It was not invisible — the
        shared box-shadow ring still applied underneath, because the override
        set only ``outline`` — but it drew a second, near-invisible indicator
        2px outside the real one for no reason.
        """
        offenders = [
            sel for sel, body in _focus_rules() if "outline" in body and "--accent-ring" in body
        ]
        assert offenders == [], (
            f"focus outline drawn with the 30%-alpha token: {offenders}. "
            "Either drop the override and inherit the shared ring, or use "
            "--accent-base."
        )

    def test_region_cards_use_the_shared_ring(self):
        """No override at all is the cleanest form of correct: the card then
        focuses like every other control in the app."""
        selectors = [sel for sel, _ in _focus_rules()]
        assert not any("gp-region-card" in sel for sel in selectors)
