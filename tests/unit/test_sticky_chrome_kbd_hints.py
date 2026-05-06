"""Unit tests for the sticky-chrome + kbd-hint shell additions.

Covers issues #20 (sticky header + tab strip) and #23 (kbd chips +
Briefing Mode chrome). The Briefing Mode rules and the keyboard
handlers themselves were already shipped — these tests focus on the
new pieces and the structural contracts they depend on.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = REPO_ROOT / "assets" / "custom.css"
ACCESSIBILITY_JS_PATH = REPO_ROOT / "assets" / "accessibility.js"


def _read_css() -> str:
    return CSS_PATH.read_text()


class TestStickyChrome:
    def test_header_height_token_defined(self):
        """``--header-height`` drives the sticky-tab-strip offset.
        Must be defined on :root so the .nav-tabs rule can reference it."""
        css = _read_css()
        assert "--header-height: 56px" in css

    def test_gp_header_is_sticky(self):
        """Header pins to top of viewport on scroll. Without this the
        whole sticky-chrome story falls apart."""
        css = _read_css()
        # Find the .gp-header block and assert the three sticky props.
        block = _block(css, ".gp-header {")
        assert "position: sticky" in block
        assert "top: 0" in block
        # Z-index must beat the tab strip's 40 so the header sits above.
        assert "z-index: 50" in block

    def test_nav_tabs_sticky_under_header(self):
        """Tab strip pins immediately below the header — top offset
        must reference --header-height so they never overlap."""
        css = _read_css()
        block = _block(css, ".nav-tabs {")
        assert "position: sticky" in block
        assert "top: var(--header-height)" in block
        assert "z-index: 40" in block
        # Translucent background + blur so content underneath stays
        # readable when the strip pins.
        assert "backdrop-filter: blur(8px)" in block

    def test_sticky_zindex_ordering(self):
        """Header (50) > tab strip (40) > content (no z-index).
        Reverse ordering would put the tab strip on top of the header
        as it scrolls, which looks broken."""
        css = _read_css()
        header_block = _block(css, ".gp-header {")
        tabs_block = _block(css, ".nav-tabs {")
        header_z = int(_extract_value(header_block, "z-index"))
        tabs_z = int(_extract_value(tabs_block, "z-index"))
        assert header_z > tabs_z


class TestKbdHints:
    def test_kbd_hint_class_styled(self):
        """The .gp-header__kbd-hint kbd selector must produce a
        compact monospace chip — defining the rule but missing the
        visual properties would render an unstyled <kbd> from
        browser defaults."""
        css = _read_css()
        block = _block(css, ".gp-header__kbd-hint kbd {")
        assert "font-family: var(--font-mono)" in block
        assert "font-size: 10px" in block
        # Subtle border distinguishing the chip from surrounding text.
        assert "border:" in block
        assert "background: var(--bg-elevated)" in block

    def test_kbd_hint_hidden_in_briefing_mode(self):
        """Projection chrome shouldn't surface shortcut hints to a
        meeting audience. The body.briefing override removes them."""
        css = _read_css()
        # The selector must exist somewhere with display: none nearby.
        assert "body.briefing .gp-header__kbd-hint" in css
        block = _block(css, "body.briefing .gp-header__kbd-hint {")
        assert "display: none" in block

    def test_layout_renders_kbd_hints_with_aria_hidden(self):
        """The visible chip duplicates the aria-keyshortcuts attribute
        on the underlying selector. To avoid double-announcement, the
        chip itself must be aria-hidden so screen readers skip it."""
        from components.layout import _build_header

        # _build_header builds an html.Header — walk it for the kbd
        # hints and assert the aria-hidden attribute.
        header = _build_header()

        kbd_hints = []

        def walk(node):
            if hasattr(node, "className") and "gp-header__kbd-hint" in (node.className or ""):
                kbd_hints.append(node)
            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    walk(c)
            elif children is not None:
                walk(children)

        walk(header)
        assert len(kbd_hints) == 2, (
            f"Expected 2 kbd hints (region + persona), found {len(kbd_hints)}"
        )
        for hint in kbd_hints:
            # Dash routes aria-* through the prop-name kwargs (the
            # **{"aria-hidden": "true"} pattern in layout.py).
            assert getattr(hint, "aria-hidden", None) == "true", (
                "Kbd hint must be aria-hidden — the underlying selector "
                "carries aria-keyshortcuts already."
            )

    def test_layout_groups_carry_aria_label_with_shortcut(self):
        """``dbc.Select`` strict-validates its kwargs and rejects
        arbitrary aria-* attributes, so we can't put
        ``aria-keyshortcuts`` directly on the selector. Instead each
        selector is wrapped in a ``<div role="group">`` with an
        ``aria-label`` that names the shortcut. Screen readers
        announce the group label when focus enters."""
        from components.layout import _build_header

        header = _build_header()
        groups: list[object] = []

        def walk(node):
            cls = getattr(node, "className", None) or ""
            if "gp-header__shortcut-group" in cls:
                groups.append(node)
            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    walk(c)
            elif children is not None:
                walk(children)

        walk(header)
        assert len(groups) == 2, (
            f"Expected 2 shortcut groups (region + persona), found {len(groups)}"
        )
        labels = sorted(getattr(g, "aria-label", "") for g in groups)
        # Both labels mention the modifier + key.
        assert all("Alt+" in label for label in labels)
        # One must reference R, the other P.
        joined = " ".join(labels)
        assert "Alt+R" in joined
        assert "Alt+P" in joined


class TestKeyboardShortcutHandlersStillWired:
    """Defensive — the kbd chips advertise shortcuts that live in
    assets/accessibility.js. If the JS handler ever gets dropped, the
    chips would lie about a non-functional shortcut."""

    def test_alt_r_focuses_region_selector(self):
        js = ACCESSIBILITY_JS_PATH.read_text()
        # Loose match — handler may use 'r' or 'R' or both, with or
        # without altKey check. We just verify the binding exists.
        assert "region-selector" in js
        assert "'r'" in js or '"r"' in js or "'R'" in js or '"R"' in js
        assert "altKey" in js

    def test_alt_p_focuses_persona_selector(self):
        js = ACCESSIBILITY_JS_PATH.read_text()
        assert "persona-selector" in js
        assert "'p'" in js or '"p"' in js or "'P'" in js or '"P"' in js


# ── helpers ──────────────────────────────────────────────────────


def _block(css: str, selector_open: str) -> str:
    """Extract the body of a CSS block whose selector starts at the
    beginning of a line and exactly equals ``selector_open``.

    Anchored to start-of-line so a top-level ``.gp-header {`` rule
    isn't accidentally matched inside a compound selector like
    ``body.briefing .gp-header {`` that happens to appear earlier in
    the file. Handles nested braces correctly.
    """
    import re

    pattern = re.compile(
        r"(?:^|\n)" + re.escape(selector_open),
        re.MULTILINE,
    )
    m = pattern.search(css)
    if m is None:
        raise AssertionError(f"selector {selector_open!r} not found at start of line")
    start = m.end()
    depth = 1
    i = start
    while i < len(css) and depth:
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return css[start : i - 1]


def _extract_value(block: str, prop: str) -> str:
    """Pull the value of a single CSS declaration out of a block."""
    for line in block.splitlines():
        s = line.strip()
        if s.startswith(prop + ":") or s.startswith(prop + " :"):
            return s.split(":", 1)[1].rstrip(";").strip()
    raise AssertionError(f"property {prop!r} not found in block")
