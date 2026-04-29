"""Inline Lucide SVG icon library.

R5b of shell-redesign-v2.md (formerly G6 of the Phase G plan). Replaces
the ad-hoc emoji-as-icons pattern with vector glyphs that respect
``currentColor``, scale crisply at any DPI, and theme via CSS.

Path data copied from https://lucide.dev (MIT-licensed). Each entry
matches the SVG ``<path>`` attribute string exactly so the inline icon
renders identically to the upstream Lucide source.

Usage:
    from components.icons import icon
    icon("alert-triangle", size="sm")           # → 14×14 svg span
    icon("info", size="md", className="ml-2")   # → 16×16 with extra class
"""

from __future__ import annotations

from dash import dcc, html

# ── Lucide path data (MIT) ────────────────────────────────────────────
# Each value is an SVG inner-fragment string. The wrapper sets common
# attrs (viewBox, stroke, stroke-width, fill, line-cap, line-join) so
# the path entries stay minimal.
_PATHS: dict[str, str] = {
    "alert-triangle": (
        '<path d="M21.73 18 13.73 4a2 2 0 0 0-3.46 0l-8 14a2 2 0 0 0 1.73 3h16a2 2 0 0 0 1.73-3"/>'
        '<path d="M12 9v4"/>'
        '<path d="M12 17h.01"/>'
    ),
    "alert-circle": (
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" x2="12" y1="8" y2="12"/>'
        '<line x1="12" x2="12.01" y1="16" y2="16"/>'
    ),
    "info": ('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'),
    "check-circle": ('<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'),
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "trending-up": (
        '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>'
    ),
    "trending-down": (
        '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>'
    ),
    "clock": ('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    "calendar": (
        '<rect width="18" height="18" x="3" y="4" rx="2"/>'
        '<line x1="16" x2="16" y1="2" y2="6"/>'
        '<line x1="8" x2="8" y1="2" y2="6"/>'
        '<line x1="3" x2="21" y1="10" y2="10"/>'
    ),
    "search": ('<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/>'),
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "chevron-down": '<polyline points="6 9 12 15 18 9"/>',
    "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
    "external-link": (
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
        '<polyline points="15 3 21 3 21 9"/>'
        '<line x1="10" x2="21" y1="14" y2="3"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" x2="12" y1="15" y2="3"/>'
    ),
    "wind": (
        '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/>'
        '<path d="M9.6 4.6A2 2 0 1 1 11 8H2"/>'
        '<path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>'
    ),
    "sun": (
        '<circle cx="12" cy="12" r="4"/>'
        '<path d="M12 2v2"/>'
        '<path d="M12 20v2"/>'
        '<path d="m4.93 4.93 1.41 1.41"/>'
        '<path d="m17.66 17.66 1.41 1.41"/>'
        '<path d="M2 12h2"/>'
        '<path d="M20 12h2"/>'
        '<path d="m6.34 17.66-1.41 1.41"/>'
        '<path d="m19.07 4.93-1.41 1.41"/>'
    ),
    "thermometer": ('<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>'),
    # Pulse monogram glyph — same path as assets/favicon.svg, used inline
    # by the header. Kept here so other surfaces can reuse via icon().
    "pulse-mono": (
        '<path d="M4 16 L11 16 L14 8 L16 24 L18 12 L21 16 L28 16" '
        'transform="scale(0.75) translate(0, 0)"/>'
    ),
    "flask": (
        '<path d="M9 2v6.74A6 6 0 0 0 6 14a6 6 0 0 0 12 0 6 6 0 0 0-3-5.26V2"/>'
        '<line x1="6" x2="18" y1="2" y2="2"/>'
        '<line x1="9" x2="15" y1="14" y2="14"/>'
    ),
}

# Pixel sizes for the convenience aliases on the .icon class
_SIZES: dict[str, int] = {
    "xs": 12,
    "sm": 14,
    "md": 16,
    "lg": 20,
    "xl": 24,
}


def icon(name: str, size: str = "md", className: str | None = None) -> html.Span:  # noqa: N803
    """Render an inline Lucide SVG icon.

    Args:
        name: Lucide icon name (one of ``_PATHS`` keys).
        size: One of ``xs|sm|md|lg|xl`` (12 / 14 / 16 / 20 / 24 px).
        className: Optional extra CSS classes appended to the wrapper.

    Returns:
        ``html.Span`` whose innerHTML is the SVG. Stroke uses
        ``currentColor`` so callers can theme via ``color: ...``.
    """
    path = _PATHS.get(name)
    if path is None:
        # Unknown icon — render an empty span instead of raising so a
        # missing key doesn't crash the layout.
        return html.Span(className=f"icon icon--{size} icon--missing")

    px = _SIZES.get(size, _SIZES["md"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )
    cls = f"icon icon--{size}"
    if className:
        cls = f"{cls} {className}"
    # Dash-html doesn't expose a "raw HTML" sink, so the SVG goes through
    # dcc.Markdown which (with dangerously_allow_html) passes inline SVG
    # straight through to the DOM. Same pattern the header monogram uses.
    return html.Span(
        dcc.Markdown(svg, dangerously_allow_html=True),
        className=cls,
    )
