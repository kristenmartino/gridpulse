"""Unit tests for the inline Lucide icon library (``components/icons.py``).

The icons used to be emitted as inline ``<svg>`` markup through
``dcc.Markdown(dangerously_allow_html=True)``. Dash's Markdown sanitizer
strips the outer ``<svg>`` tag and keeps only its orphaned ``<path>``
children, so every icon rendered as nothing. They now render as an
``html.Span`` whose glyph is a CSS ``mask-image`` (a ``data:`` URI) over a
``currentColor`` background — which paints *and* keeps color inheritance.

These tests lock in that a real, self-contained ``<svg>`` reaches the DOM
(via the mask) and that the old ``dcc.Markdown`` path does not come back.
"""

from urllib.parse import unquote

from dash import dcc, html

from components.icons import _PATHS, _SIZES, icon


def _mask_svg(component) -> str:
    """Extract and URL-decode the masked SVG string from an icon span."""
    mask = component.style["maskImage"]
    assert mask.startswith("url('data:image/svg+xml,")
    assert mask.endswith("')")
    encoded = mask[len("url('data:image/svg+xml,") : -len("')")]
    return unquote(encoded)


class TestIconRendering:
    def test_returns_span_not_markdown(self):
        comp = icon("info")
        assert isinstance(comp, html.Span)
        # Regression guard: no dcc.Markdown anywhere — that path had its
        # <svg> stripped by the sanitizer and never painted.
        assert not isinstance(comp, dcc.Markdown)
        assert getattr(comp, "children", None) is None

    def test_mask_carries_complete_svg_with_path(self):
        comp = icon("info")
        decoded = _mask_svg(comp)
        # A full <svg> wrapper with the glyph path *inside* it, not orphaned.
        assert decoded.startswith("<svg")
        assert "</svg>" in decoded
        # The info glyph's exact path data survives into the mask.
        assert _PATHS["info"] in decoded
        assert decoded.index("<svg") < decoded.index("</svg>")

    def test_background_is_current_color_via_class(self):
        # Color inheritance lives in the .icon CSS (background-color:
        # currentColor); the span just needs the class so it applies.
        comp = icon("info")
        assert "icon" in comp.className
        # Only the dynamic mask-image is inline.
        assert set(comp.style) == {"maskImage", "WebkitMaskImage"}
        assert comp.style["maskImage"] == comp.style["WebkitMaskImage"]

    def test_size_class_and_svg_dimensions(self):
        comp = icon("zap", size="lg")
        assert "icon--lg" in comp.className
        decoded = _mask_svg(comp)
        px = _SIZES["lg"]
        assert f'width="{px}"' in decoded
        assert f'height="{px}"' in decoded

    def test_extra_classname_is_appended(self):
        comp = icon("info", className="ml-2")
        assert comp.className == "icon icon--md ml-2"

    def test_unknown_icon_is_empty_missing_span(self):
        comp = icon("does-not-exist")
        assert isinstance(comp, html.Span)
        assert "icon--missing" in comp.className
        # No mask on an unknown glyph — the .icon--missing CSS keeps it from
        # painting a solid currentColor block.
        assert not getattr(comp, "style", None)

    def test_all_known_icons_render_a_real_svg(self):
        for name in _PATHS:
            decoded = _mask_svg(icon(name))
            assert decoded.startswith("<svg"), name
            assert "</svg>" in decoded, name
