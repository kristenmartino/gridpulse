"""Unit tests for the sticky-chrome shell additions (issue #20).

The sticky header / tab strip pin to the viewport on scroll. Without
the right z-index ordering and offset references, scrolling within a
long tab body would either let the chrome scroll out of view or render
the tab strip on top of the header during the pinning transition.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = REPO_ROOT / "assets" / "custom.css"


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
