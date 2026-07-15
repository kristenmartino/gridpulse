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

from color_science import (  # noqa: E402
    ciede2000,
    contrast_ratio,
    hex_to_oklch,
    lstar,
    simulate,
)
from stock_palettes import STOCK  # noqa: E402

# The derivation rules live with the CLI verifier so the script and these
# tests check ONE definition rather than two that can drift apart.
from verify_palette import (  # noqa: E402
    NEUTRAL_LIGHTNESS,
    SEMANTIC_COMPARED,
    SEMANTIC_SPEC,
    STOCK_FLOOR,
    _chroma,
    _fit,
)

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
            # Mirrored in both files and, until an audit noticed, checked in
            # neither — so the module's "the two are asserted equal" was false
            # for exactly the tokens nobody thought about.
            ("--border-subtle", "BORDER_SUBTLE"),
            ("--border-default", "BORDER_DEFAULT"),
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

    def test_every_derived_root_property_matches_its_source(self, css: str):
        """The :root block was the gate's designed blind spot, and the brand lived there.

        check_css skips any `--x:` line — that is where values are ALLOWED to be
        written. But only ~15 of them were mirrored against Python, so the rest
        were unguarded: an audit reverted --border-accent and the
        --accent-glow/dim/ring channels to the RETIRED accent and shipped green.
        They paint (a gradient, three backgrounds, a focus ring), so that was the
        5.5-era "accent copied N ways" defect, alive inside the one region built
        not to look.

        Every :root property that is DERIVED from a token is now checked against
        it, so "declared here" no longer means "unchecked here".
        """
        expected = {
            "--border-accent": tokens.ACCENT,
            "--accent-glow": tokens.alpha(tokens.ACCENT, 0.08),
            "--accent-dim": tokens.alpha(tokens.ACCENT, 0.15),
            "--accent-ring": tokens.alpha(tokens.ACCENT, 0.32),
            "--bg-overlay": tokens.alpha(tokens.BG_BASE, 0.72),
            "--forecast-dim": tokens.alpha(tokens.FORECAST, 0.12),
            "--success-dim": tokens.alpha(tokens.SUCCESS, 0.12),
            "--warning-dim": tokens.alpha(tokens.WARNING, 0.12),
            "--danger-dim": tokens.alpha(tokens.DANGER, 0.12),
            "--info-dim": tokens.alpha(tokens.INFO, 0.12),
        }
        for var, want in expected.items():
            m = re.search(rf"^\s*{re.escape(var)}\s*:\s*([^;]+);", css, re.M)
            assert m, f"{var} not declared in :root"
            got = m.group(1).strip()
            assert got.lower() == want.lower(), (
                f"{var} is {got} but derives from its token as {want} — a :root "
                f"property has drifted off the color it is supposed to tint"
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


class TestOwnership:
    """The palette must be THIS product's, and provably so.

    This class exists because an independent audit reverted ACCENT to the stock
    Tailwind near-duplicate the whole redesign was built to leave — and every
    gate and all 38 tests passed green. The dimension is scored on ownership;
    ownership was the one property nothing measured. Same shape as the defect
    this file was written to prevent ("nothing in CI failed on a raw hex, which
    is why it rotted"), one level up.

    The three claims are checked three different ways because "owned" is not one
    property. The accent is a CHOICE and is defended by not being a copy;
    everything else is DERIVED and is defended by reproducing from it.
    """

    def test_accent_is_not_a_stock_swatch(self):
        """The one color a human picked may not be one someone could download.

        Measured against every corpus in scripts/stock_palettes.py, not one.
        A Tailwind-only ruler passed an accent sitting 1.64 from CSS
        darkturquoise — below the ~2.3 JND, i.e. the same color.
        """
        near, hexv = min(STOCK.items(), key=lambda kv: ciede2000(tokens.ACCENT, kv[1]))
        d = ciede2000(tokens.ACCENT, hexv)
        assert d >= STOCK_FLOOR, (
            f"ACCENT {tokens.ACCENT} is CIEDE2000 {d:.1f} from {near} — a copy. "
            f"JND is ~2.3; both retired accents measured under it."
        )

    def test_neutrals_reproduce_from_the_stated_curve(self):
        """ "Derived" has to mean regenerable, or it is just a nice comment."""
        anchor_l, _, anchor_h = hex_to_oklch(tokens.ACCENT)
        for name, lightness in NEUTRAL_LIGHTNESS.items():
            regen = _fit(lightness, _chroma(lightness), anchor_h)
            d = ciede2000(getattr(tokens, name), regen)
            assert d <= 1.0, (
                f"{name} = {getattr(tokens, name)} does not reproduce from the ramp curve "
                f"at the accent's hue (regenerates to {regen}, dE {d:.1f})"
            )

    def test_semantics_reproduce_from_the_rule(self):
        """Anchor chroma at a solved lightness — see tokens.py for why not constant."""
        _, anchor_c, _ = hex_to_oklch(tokens.ACCENT)
        for name, (sem_l, hue) in SEMANTIC_SPEC.items():
            regen = _fit(sem_l, anchor_c, hue)
            d = ciede2000(getattr(tokens, name), regen)
            assert d <= 1.0, (
                f"{name} = {getattr(tokens, name)} does not reproduce from the semantic "
                f"rule (solved lightness {sem_l:.2f}, hue {hue} -> {regen}, dE {d:.1f})"
            )

    def test_severity_triad_separates_under_dichromacy(self):
        """The check that did not exist while a constant-lightness rule shipped.

        SUCCESS/WARNING/DANGER all collapse toward one yellow under deuteranopia;
        lightness is the only channel left. Holding it constant measured 1.5 —
        invisible — and nothing noticed, because verify_palette checked the fuel
        stack, the accent's co-occurring series, the drivers and the map, and
        never the semantics against each other.
        """
        for a, b in SEMANTIC_COMPARED:
            m = min_cvd(getattr(tokens, a), getattr(tokens, b))
            assert m >= SHARED_FIGURE_FLOOR, (
                f"severity {a}/{b} minCVD {m:.1f} — a constant-lightness rule "
                f"measured 1.5 here; stock Tailwind measured 3.3"
            )

    def test_no_semantic_is_a_stock_swatch(self):
        """The 5.5 finding: all five were dE 0.00 from the download."""
        for name in SEMANTIC_SPEC:
            value = getattr(tokens, name)
            near, hexv = min(STOCK.items(), key=lambda kv: ciede2000(value, kv[1]))
            d = ciede2000(value, hexv)
            assert d > 1.0, f"{name} = {value} is {near} verbatim (dE {d:.2f})"


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

    # test_adjacent_fuel_bands_separate_under_every_cvd_type,
    # test_every_fuel_band_clears_the_net_load_line, and
    # test_accent_clears_every_series_it_shares_a_figure_with lived here.
    # All three are DELETED — subsumed by tests/unit/test_rendered_figures.py,
    # which builds the real figures and measures every pair WITHIN each one.
    #
    # They were not merely redundant, they were WRONG, each in the same way:
    # each measured a list I maintained rather than the figure the app draws.
    # The adjacency check only compared bands next to each other in
    # FUEL_STACK_ORDER (a reader matches a legend swatch to a band anywhere) and
    # was colour-only (the palette now carries a fill pattern it cannot see).
    # The accent check consulted a hand-written list of rivals that omitted the
    # one colour it actually collides with. Keeping them beside the real check
    # would be keeping the second list that let the first go stale.

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

    def test_tertiary_meets_normal_text_aa(self):
        """--text-tertiary renders 11px chart ticks, so it owes normal-text AA.

        Its lightness is SOLVED for this number, not sampled — see the ramp
        derivation in components/tokens.py.

        This assertion used to read `>= 3.0` (the graphics floor) with a
        docstring explaining that stock zinc-500's 4.09 was a known gap. The
        ramp was rebuilt and the gap closed, but the floor stayed — so the one
        token advertised as "solved for AA" was the one token nothing held to
        AA, and reverting it to zinc-500 passed green. An audit found that by
        doing exactly that.
        """
        ratio = contrast_ratio(tokens.TEXT_TERTIARY, tokens.BG_BASE)
        assert ratio >= 4.5, (
            f"--text-tertiary is {ratio:.2f}:1 on --bg-base — below WCAG AA for the "
            f"11px tick labels it renders"
        )


class TestEveryTokenIsLive:
    """No token may exist without being painted. This is the repo's oldest bug.

    It has now happened four times, three of them inside the module written to
    stop it:

      * accessibility.FUEL_COLORS — CVD-safe, tested, ZERO callsites, while the
        app painted a different unverified palette. The 5.5 audit's headline.
      * accessibility.SEVERITY_COLORS — same defect, one constant over.
      * tokens.SEVERITY — invented while deleting those two, keyed to a
        vocabulary nothing in the product speaks. Now deleted.
      * tokens.WEATHER_DRIVERS — declared with three entries, wired with ONE.
        The Wind sparkline painted tokens.SUCCESS (a semantic token) and Solar
        painted tokens.FORECAST, while scripts/verify_palette.py dutifully
        measured the two colors that never rendered.

    A dead token is worse than no token: it is a claim the tests defend. So the
    rule is mechanical — every public token is painted by something, or it does
    not exist.
    """

    @staticmethod
    def _product_source() -> str:
        root = pathlib.Path(__file__).resolve().parents[2]
        parts = []
        for d in ("components", "personas"):
            for f in (root / d).rglob("*.py"):
                if f.name != "tokens.py" and "__pycache__" not in f.parts:
                    parts.append(f.read_text())
        parts.append((root / "app.py").read_text())
        parts.append((root / "scripts" / "generate_brand_assets.py").read_text())
        return "\n".join(parts)

    @staticmethod
    def _css() -> str:
        return (pathlib.Path(__file__).resolve().parents[2] / "assets" / "custom.css").read_text()

    # token -> the custom property it mirrors, for tokens only the browser paints
    CSS_NAMES = {
        "BG_BASE": "--bg-base",
        "BG_RAISED": "--bg-raised",
        "BG_HOVER": "--bg-hover",
        "SURFACE_SUNKEN": "--surface-sunken",
        "TEXT_PRIMARY": "--text-primary",
        "TEXT_SECONDARY": "--text-secondary",
        "TEXT_TERTIARY": "--text-tertiary",
        "TEXT_DISABLED": "--text-disabled",
        "ACCENT": "--accent-base",
        "ACCENT_SOFT": "--accent-hover",
        "FORECAST": "--forecast",
        "SUCCESS": "--success",
        "WARNING": "--warning",
        "DANGER": "--danger",
        "INFO": "--info",
        "BORDER_SUBTLE": "--border-subtle",
        "BORDER_DEFAULT": "--border-default",
    }

    def test_every_token_has_a_callsite(self):
        """Live in EITHER surface: Plotly reads the Python, the browser the CSS.

        --bg-hover is never used by a figure but paints every hover state; its
        Python mirror exists so test_css_root_matches_python_token can hold the
        two sides together. That is a real job, so "live" means painted by
        something, on either side.
        """
        src, css = self._product_source(), self._css()
        dead = [
            n
            for n in dir(tokens)
            if not (n.startswith("_") or n.islower() or n == "annotations")
            and f"tokens.{n}" not in src
            and not (self.CSS_NAMES.get(n) and f"var({self.CSS_NAMES[n]})" in css)
        ]
        assert not dead, (
            f"token(s) painted by nothing — no Python callsite and no var() use in "
            f"custom.css: {dead}. A token nothing paints is a claim nothing checks."
        )

    def test_every_entry_of_every_token_group_has_a_callsite(self):
        """A group can be two-thirds dead while the group itself looks used."""
        src = self._product_source()
        groups = {
            "WEATHER_DRIVERS": tokens.WEATHER_DRIVERS,
            "FUEL_COLORS": tokens.FUEL_COLORS,
            "PERSONA_COLORS": tokens.PERSONA_COLORS,
        }
        dead = []
        for gname, group in groups.items():
            # A comprehension over .items() reaches every entry wholesale.
            if f"{gname}.items()" in src:
                continue
            dead += [f"{gname}[{k!r}]" for k in group if f'{gname}["{k}"]' not in src]
        assert not dead, (
            f"token group entries with no callsite: {dead}. WEATHER_DRIVERS shipped "
            f"2 of 3 entries dead while the drivers painted semantic tokens instead."
        )


class TestGeneratedAssetsReproduce:
    """Generated artifacts are gated by reproduction, not literal-freedom.

    A rendered SVG necessarily contains literals, so the color gate skips it —
    safe only if something else proves it still matches the tokens. The
    generator once held stock blue-500 in a tuple commented "--accent-base"
    while the checked-in favicon had been hand-edited to the real accent; the
    two disagreed silently, and re-running the generator would have reverted the
    brand.
    """

    def test_favicon_svg_matches_its_generator(self):
        import tempfile

        root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "scripts"))
        import generate_brand_assets as gen

        out = pathlib.Path(tempfile.mkdtemp()) / "favicon.svg"
        gen.make_favicon_svg(out)
        assert out.read_text() == (root / "assets" / "favicon.svg").read_text(), (
            "assets/favicon.svg does not match what generate_brand_assets.py now "
            "produces — the brand mark and the token have drifted. Re-run "
            "`python scripts/generate_brand_assets.py --target favicon`."
        )
