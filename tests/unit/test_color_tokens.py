"""The palette's contract: what is DECLARED must equal what is PAINTED.

Why this file exists
--------------------
``test_sprint3.TestColorblindPalette`` used to assert that ``FUEL_COLORS``
contained the KEYS "nuclear", "coal", "gas"... and that every value started
with "#". Both passed forever. Neither could fail for any reason that mattered:

  * ``FUEL_COLORS`` had ZERO callsites. The app painted a completely different,
    unverified palette from ``_callbacks_overview._FUEL_DISPLAY``. The test
    green-lit a dict nothing rendered.
  * In the palette that DID render, nuclear and hydro — bands that physically
    touch in the stacked area — measured CIEDE2000 1.0 under deuteranopia.
    A key-existence test cannot see that.

That is false assurance: a test whose passing tells you nothing, next to a
comment claiming the palette was "verified distinguishable under protanopia,
deuteranopia, tritanopia". So these tests assert two things instead:

  1. RENDERED colors — build the real figure, read the trace back.
  2. MEASURED separation — CIEDE2000 under simulated dichromacy, not eyeballs.

If you change a color and this file goes red, the palette regressed. Run
``python scripts/verify_palette.py`` to see every number at once.
"""

from __future__ import annotations

import pathlib
import re
import sys
from itertools import combinations

import pandas as pd
import pytest

# scripts/ is not a package (no __init__.py) and color_science must be
# importable by BOTH this file and the standalone CLI, so it lives there and we
# extend the path rather than duplicating the math in the test tree.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from color_science import ciede2000, contrast_ratio, lstar, simulate  # noqa: E402

from components import tokens  # noqa: E402

CVD = ("normal", "protan", "deutan", "tritan")

# Calibrated to the shipped baseline, not to a round number: the weakest
# co-occurring pair in the pre-existing Wong series (ARIMA green vs the EIA
# gray reference) measures 8.2. 12 is stricter than what already ships.
SHARED_FIGURE_FLOOR = 12.0


def min_cvd(a: str, b: str) -> float:
    return min(ciede2000(simulate(a, k), simulate(b, k)) for k in CVD)


class TestSingleSourceOfTruth:
    """The Python mirror and the stylesheet must not drift apart."""

    @pytest.fixture(scope="class")
    def css(self) -> str:
        return (pathlib.Path(__file__).resolve().parents[2] / "assets" / "custom.css").read_text()

    @pytest.mark.parametrize(
        ("css_var", "token_name"),
        [
            ("--bg-base", "BG_BASE"),
            ("--bg-raised", "BG_RAISED"),
            ("--bg-hover", "BG_HOVER"),
            ("--surface-sunken", "SURFACE_SUNKEN"),
            ("--text-primary", "TEXT_PRIMARY"),
            ("--text-secondary", "TEXT_SECONDARY"),
            ("--text-tertiary", "TEXT_TERTIARY"),
            ("--text-disabled", "TEXT_DISABLED"),
            ("--accent-base", "ACCENT"),
            ("--accent-hover", "ACCENT_SOFT"),
            ("--forecast", "FORECAST"),
            ("--success", "SUCCESS"),
            ("--warning", "WARNING"),
            ("--danger", "DANGER"),
            ("--info", "INFO"),
        ],
    )
    def test_css_root_matches_python_token(self, css: str, css_var: str, token_name: str):
        """Every mirrored token holds the same value on both sides.

        The Plotly layer cannot read CSS custom properties, so the value
        genuinely has to exist twice. This is the seam where "mirrored in
        Python as ACCENT" stops being a promise in a comment.
        """
        m = re.search(rf"^\s*{re.escape(css_var)}\s*:\s*([^;]+);", css, re.M)
        assert m, f"{css_var} not declared in :root"
        assert m.group(1).strip().lower() == getattr(tokens, token_name).lower(), (
            f"{css_var} and tokens.{token_name} have drifted apart"
        )

    def test_accent_is_declared_once_in_css(self, css: str):
        """The accent's channels may only appear in :root declarations.

        Three rules used to inline rgba(53, 198, 255, ...) directly — that is
        how one hex ended up copied four ways (CSS token, Python mirror, 13
        string literals, and these rules).
        """
        r, g, b = tokens.rgb(tokens.ACCENT)
        pattern = re.compile(rf"rgba?\(\s*{r}\s*,\s*{g}\s*,\s*{b}")
        for i, line in enumerate(css.splitlines(), 1):
            stripped = line.strip()
            # Comments describe the rule; they cannot paint. (This very rule is
            # explained in a :root comment that names the offending literal.)
            if stripped.startswith(("*", "/*", "//")):
                continue
            if pattern.search(line):
                assert re.match(r"\s*--[\w-]+\s*:", line), (
                    f"custom.css:{i} writes the accent's channels outside a "
                    f"custom-property declaration — use var(--accent-*)"
                )


class TestRenderedTraceColors:
    """Assert what the figure actually paints, not what a dict declares."""

    @staticmethod
    def _trace_colors(fig) -> set[str]:
        out: set[str] = set()
        for tr in fig.data:
            for attr in ("line", "marker"):
                obj = getattr(tr, attr, None)
                c = getattr(obj, "color", None) if obj is not None else None
                if isinstance(c, str):
                    out.add(c.lower())
            fc = getattr(tr, "fillcolor", None)
            if isinstance(fc, str):
                out.add(fc.lower())
        return out

    def test_fuel_display_renders_the_verified_palette(self):
        """The fuel chart paints FUEL_COLORS — the palette the tests verify.

        The regression this pins: FUEL_COLORS existed, was Okabe-Ito, was
        asserted by tests... and had zero callsites, while _FUEL_DISPLAY
        painted an entirely separate unverified palette.
        """
        from components._callbacks_overview import _FUEL_DISPLAY

        for fuel, color in tokens.FUEL_COLORS.items():
            assert _FUEL_DISPLAY[fuel]["color"] == color, (
                f"{fuel} renders {_FUEL_DISPLAY[fuel]['color']} but the verified "
                f"palette declares {color}"
            )
            # The 85% area fill must be derived from that same color.
            assert _FUEL_DISPLAY[fuel]["fill"] == tokens.alpha(color, 0.85)

    def test_fuel_stack_order_is_the_verified_order(self):
        """Adjacency guarantees only hold for the order they were verified in."""
        from components._callbacks_overview import _FUEL_STACK_ORDER

        assert tuple(_FUEL_STACK_ORDER) == tuple(tokens.FUEL_STACK_ORDER)
        assert set(_FUEL_STACK_ORDER) == set(tokens.FUEL_COLORS)

    def test_actual_demand_is_one_hex_everywhere(self):
        """ "Actual demand" resolves to a single color across every tab.

        It used to be the accent on Overview and Wong blue on five other tabs.
        """
        from components._callbacks_shared import COLORS
        from components.accessibility import LINE_STYLES

        assert COLORS["actual"] == tokens.ACCENT
        assert LINE_STYLES["actual"]["color"] == tokens.ACCENT

    def test_colorway_leads_with_the_accent(self):
        """Brand color and data color are one system: slot 0 is the accent."""
        from components._callbacks_shared import _COLORWAY, PLOT_LAYOUT

        assert _COLORWAY[0] == tokens.ACCENT
        assert PLOT_LAYOUT["colorway"][0] == tokens.ACCENT

    def test_empty_figure_uses_a_token(self):
        """Every tab falls back to _empty_figure — it had an orphaned literal."""
        from components._callbacks_shared import _empty_figure

        fig = _empty_figure("No data")
        assert fig.layout.annotations[0].font.color == tokens.TEXT_SECONDARY

    def test_overview_hero_fill_matches_its_own_line(self):
        """A fill must be derived from the line it sits under.

        The hero drew an ACCENT line over an rgba(59, 130, 246, 0.08) fill —
        stock Tailwind blue-500, left behind when the accent moved. A hex grep
        never saw it because it was spelled rgba().
        """
        from components._callbacks_overview import _build_overview_hero_chart

        idx = pd.date_range("2026-07-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "demand_mw": range(30000, 30048)})
        fig = _build_overview_hero_chart("FPL", df)
        actual = next((t for t in fig.data if t.name == "Actual"), None)
        assert actual is not None, "hero chart did not render an Actual trace"
        assert actual.line.color == tokens.ACCENT
        assert actual.fillcolor == tokens.alpha(tokens.ACCENT, 0.08)


class TestMeasuredSeparation:
    """CIEDE2000 under simulated dichromacy — the claim the comments made."""

    def test_adjacent_fuel_bands_separate_under_every_cvd_type(self):
        order = tokens.FUEL_STACK_ORDER
        for a, b in zip(order, order[1:], strict=False):  # pairwise: lengths differ by 1
            m = min_cvd(tokens.FUEL_COLORS[a], tokens.FUEL_COLORS[b])
            assert m >= SHARED_FIGURE_FLOOR, (
                f"fuel bands {a}|{b} touch in the stack but measure {m:.1f} "
                f"(floor {SHARED_FIGURE_FLOOR}). This is the nuclear/hydro "
                f"deutan collapse (1.0) regressing."
            )

    def test_every_fuel_band_clears_the_net_load_line(self):
        """The accent net-load line is drawn OVER the stack."""
        for fuel, color in tokens.FUEL_COLORS.items():
            m = min_cvd(color, tokens.ACCENT)
            assert m >= SHARED_FIGURE_FLOOR, f"fuel {fuel} vs accent = {m:.1f}"

    def test_accent_clears_every_series_it_shares_a_figure_with(self):
        rivals = [
            tokens.CB_PALETTE["orange"],
            tokens.CB_PALETTE["green"],
            tokens.CB_PALETTE["vermillion"],
            tokens.CB_PALETTE["yellow"],
            tokens.CB_PALETTE["blue"],
            tokens.NEUTRAL_SERIES,
            tokens.FORECAST,
        ]
        for r in rivals:
            assert min_cvd(tokens.ACCENT, r) >= SHARED_FIGURE_FLOOR

    def test_map_colorscale_is_luminance_monotonic(self):
        """The preserve-guardrail: the best-reasoned artifact in the repo."""
        ls = [lstar(c) for _, c in tokens.MAP_COLORSCALE]
        assert ls == sorted(ls), f"MAP_COLORSCALE lost luminance monotonicity: {ls}"
        assert ls[-1] - ls[0] > 50, "stress scale lost its luminance range"

    def test_temperature_and_hydro_have_distinct_hues(self):
        """They were byte-identical (#3b82f6)."""
        d = ciede2000(tokens.WEATHER_DRIVERS["temperature"], tokens.FUEL_COLORS["hydro"])
        assert d >= SHARED_FIGURE_FLOOR

    def test_wong_palette_stays_mutually_distinguishable(self):
        """CB_PALETTE is an external standard (Wong 2011), not a brand choice.

        The "did someone retype the published hexes" guard lives in
        scripts/verify_palette.py::check_wong_intact, which CI runs and which
        the color-literal gate excludes precisely because it must hold the
        reference values. Restating them HERE would just be a second copy of
        the palette inside the file that polices copies of the palette.

        What this asserts instead is the property the palette exists for, which
        a value-equality check cannot express: every pair stays apart under
        every CVD type.
        """
        values = [v for k, v in tokens.CB_PALETTE.items() if k != "black"]
        for a, b in combinations(values, 2):
            assert min_cvd(a, b) >= 8.0, f"Wong pair {a}/{b} collapsed to {min_cvd(a, b):.1f}"

    def test_model_identity_is_double_encoded(self):
        """WCAG 1.4.1: color is never the sole channel for model identity."""
        from components.accessibility import LINE_STYLES

        dashes = {k: v["dash"] for k, v in LINE_STYLES.items()}
        for name, style in LINE_STYLES.items():
            assert style.get("dash"), f"{name} lost its dash encoding"
            assert style.get("color")
        models = ["prophet", "arima", "xgboost", "ensemble"]
        assert len({dashes[m] for m in models}) == len(models), (
            "two models share a dash pattern — the second channel collapsed"
        )


class TestContrast:
    """WCAG AA — the preserve-guardrail."""

    @pytest.mark.parametrize(
        "name",
        [
            "TEXT_PRIMARY",
            "TEXT_SECONDARY",
            "ACCENT",
            "SUCCESS",
            "WARNING",
            "DANGER",
            "INFO",
            "FORECAST",
        ],
    )
    def test_meets_aa_normal_text_on_base(self, name: str):
        assert contrast_ratio(getattr(tokens, name), tokens.BG_BASE) >= 4.5

    def test_tertiary_meets_at_least_graphics_contrast(self):
        """--text-tertiary is stock zinc-500 and measures 4.09 on --bg-base.

        That clears WCAG AA for graphics/UI (3.0) but NOT for normal text
        (4.5) — and it is used for 11px chart tick labels, which are normal
        text. Pinned at the graphics floor so the known gap cannot silently
        get worse, and asserted here so the number is visible rather than
        implied. Closing it properly means rebuilding the neutral ramp.
        """
        ratio = contrast_ratio(tokens.TEXT_TERTIARY, tokens.BG_BASE)
        assert ratio >= 3.0
