"""Design tokens — the ONE place a color literal may appear in the Python layer.

Why this module exists
----------------------
Before this, GridPulse shipped three parallel color systems at once: eight
blues, three severity triads for one concept, and two fuel palettes (the
CVD-safe one had zero callsites while the app painted the unverified one).
The accent hex was copied three ways — the CSS token, a Python mirror, and 13
hardcoded ``"#35c6ff"`` string literals. Comments throughout the repo already
*described* single-source-of-truth discipline; nothing enforced it, so it
rotted.

``scripts/check_color_tokens.py`` now fails CI on any raw color literal
outside this file. That gate — not this docstring — is what keeps the palette
from drifting again.

Relationship to CSS
-------------------
``assets/custom.css`` ``:root`` is the source of truth for the *browser*; this
module mirrors it for the *Python/Plotly* layer, which cannot read CSS custom
properties. The two must be edited together. ``tests/unit/test_color_tokens.py``
parses the stylesheet and asserts the mirrors match, so a drift fails CI rather
than shipping.

Verification
------------
Every palette here is verified by ``scripts/verify_palette.py`` (OKLCh, CIE L*,
WCAG contrast, CIEDE2000 under normal/protan/deutan/tritan). Re-run it after
ANY edit; ``tests/unit/test_color_tokens.py`` runs the same invariants in CI.
"""

from __future__ import annotations

# ── Helpers ──────────────────────────────────────────────────────────


def rgb(color: str) -> tuple[int, int, int]:
    """Return ``color`` as an ``(r, g, b)`` tuple.

    For PIL and any other consumer that wants channels rather than a CSS
    string. ``scripts/generate_brand_assets.py`` previously kept its own
    ``ACCENT_BLUE = (59, 130, 246)`` tuple — commented "--accent-base" while
    actually holding the retired stock blue — so the generated favicon,
    og-image and touch icon painted the brand in a color the brand had
    already left behind. A tuple is a color literal that a hex grep cannot
    see; deriving it from a token closes that hole.

    Args:
        color: A ``#rrggbb`` token from this module.

    Returns:
        ``(r, g, b)`` with each channel in ``[0, 255]``.
    """
    h = color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"rgb() expects #rrggbb, got {color!r}")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def alpha(color: str, a: float) -> str:
    """Return ``color`` as an ``rgba()`` string at alpha ``a``.

    The reason this exists: an ``rgba(59, 130, 246, 0.08)`` literal is a color
    literal that a hex grep does not catch. Two such literals had already
    drifted off their own line color — the Overview hero drew an accent line
    over a stock-blue fill, and the Forecast band used the retired #38D0FF.
    Building translucent fills from a token keeps a fill tied to its source
    color by construction.

    Args:
        color: A ``#rrggbb`` token from this module.
        a: Alpha in ``[0, 1]``.

    Returns:
        An ``rgba(r, g, b, a)`` string.
    """
    h = color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"alpha() expects #rrggbb, got {color!r}")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {a})"


# ── Token values ─────────────────────────────────────────────────────
# ── The neutral ramp — DERIVED, not downloaded ───────────────────────
#
# Every neutral below is generated from the ACCENT's own hue (202°) at low
# chroma. It replaces stock Tailwind zinc, which sat at CIEDE2000 0.00 from the
# download — literally the shipped values, at a hue (~286°) unrelated to
# anything else in the product.
#
# The curve: chroma = 0.019 · exp(−((L − 0.38)/0.34)²).
#
# It is a Gaussian, so it is SYMMETRIC about its peak; what is deliberate is
# where the peak sits — L 0.38, below mid-tone. An earlier version of this
# comment claimed the curve was "skewed toward the dark end rather than
# symmetric", which is false about a Gaussian by construction, and it was
# written in the same commit that corrected a different false comment. Recorded
# rather than quietly deleted: this module's whole thesis is that prose drifts
# from code, and the author of the thesis is not exempt.
#
# The rationale for the low peak is TINT THE ROOM, NOT THE INK. Dark surfaces
# are large, so a trace of the brand hue there reads as atmosphere; the same
# chroma in text reads as a rendering fault, so it tapers as lightness rises.
# Zinc's curve peaks at ~0.0146 near mid-tone at a hue unrelated to its accent —
# not a worse shape, just an unchosen one.
#
# LIGHTNESS IS DELIBERATELY UNCHANGED from the zinc ramp it replaces (L* 2.8 →
# 2.9, 90.7 → 90.8, and so on). The UI's value architecture was already right;
# only the hue bias was borrowed. Changing both at once would make a visual
# regression impossible to attribute.
#
# The one exception is --text-tertiary, whose lightness is SOLVED for WCAG AA on
# --bg-base rather than chosen. At zinc's lightness it measured 4.09:1 — below
# AA for the 11px chart ticks it renders.

BG_BASE = "#050c0c"  # --bg-base: page background, deepest
BG_RAISED = "#0a1314"  # --bg-raised: cards, dropdowns, badges
BG_HOVER = "#101a1b"  # --bg-hover: interactive hover state
SURFACE_SUNKEN = "#030708"  # --surface-sunken: recessed track/inset

TEXT_PRIMARY = "#e3e5e5"  # --text-primary
TEXT_SECONDARY = "#9da3a4"  # --text-secondary
TEXT_TERTIARY = "#707c7c"  # --text-tertiary — solved for AA, not sampled
TEXT_DISABLED = "#485657"  # --text-disabled (WCAG 1.4.3 exempts inactive UI)

# ── Accent — the brand hue, and the anchor everything derives from ───
#
# OKLCh(0.82, 0.130, 202°). This is the ONE color a human chose. Every other
# non-external token in this module is generated from it, so it is the only one
# that needs defending by a different means: it must not be a copy.
#
# "Not a copy" is measured against 329 published swatches — Tailwind, the CSS
# named colors, Material, IBM Carbon, Ant Design, Chakra and Bootstrap
# (scripts/stock_palettes.py). Nearest today: material cyan-300 at CIEDE2000 4.24.
#
# That corpus is deliberately broad because the last two accents were not.
# #35c6ff was condemned at 2.5 from Tailwind sky-400 — and measures 1.51 from
# Material lightblue-300. Its replacement #33d3d5 scored 7.8 against a
# Tailwind-only ruler and shipped as "owned" while sitting 1.64 from CSS
# ``darkturquoise``, a color every browser ships. Both were below the ~2.3
# CIEDE2000 just-noticeable difference: perceptually the same color as something
# off a shelf. The gate reported otherwise because it had been given one shelf.
#
# The floor is 4.0 — see STOCK_FLOOR in scripts/verify_palette.py for why that
# number and not a rounder one. Short version: with a real corpus the metric
# nearly saturates (the greatest distance from all stock, anywhere in usable
# accent space, is 10.8), so a high floor stops being a floor and becomes an
# optimisation target — and optimising distance drives you to neon #00fdfd, the
# least owned color there is. 4.0 asks only the question worth asking: is this a
# copy? It is not; it is 1.7× the JND from anything published.
#
# What makes the palette this product's own is NOT that number. It is that one
# anchor generates the neutrals and the semantics through two stated rules, and
# every relationship in it is measured.
#
# This is ALSO the demand/primary data series (``COLORS["actual"]``,
# ``LINE_STYLES["actual"]``, ``_COLORWAY[0]``) so brand color and data color are
# one system rather than two.

ACCENT = "#35dde8"  # --accent-base
ACCENT_SOFT = "#59f4ff"  # --accent-hover

# ── Semantic ramp — DERIVED from the anchor ──────────────────────────
#
# THE RULE: chroma is the anchor's (0.130) or the sRGB gamut maximum at that hue,
# whichever is lower — so nothing can out-saturate the brand. Hue is convention.
# LIGHTNESS IS SOLVED for dichromatic separation, then clamped to a band where
# the color still reads as its meaning.
#
# The lightness rule exists because the previous one was an accessibility bug I
# shipped. It held every semantic at the anchor's lightness, on the reasoning
# that "no severity outshouts another by brightness" — and mocked the Tailwind
# row it replaced for spanning L 0.705-0.837 with "no reason anyone chose".
#
# That reasoning was exactly backwards. Under deuteranopia SUCCESS (green),
# WARNING (amber) and DANGER (red) all collapse toward the same yellow;
# lightness is the ONLY channel left. Holding it constant deleted the one thing
# telling them apart. Measured: the severity triad's worst pair went from 3.3
# (stock Tailwind) to 1.5 (the "designed" version) — and 1.5 is what
# scripts/color_science.py's own docstring calls invisible. The spread being
# mocked was doing the accessibility work.
#
# Nothing caught it because verify_palette.py checked fuel adjacency, the accent
# against its co-occurring series, the drivers and the map — and never the
# semantics against each other. It checks them now.
#
# Solved values give a worst pair of 15.3 (SUCCESS/WARNING), 18.8
# (SUCCESS/DANGER), 14.3 (WARNING/DANGER) — better than both the stock row and
# the constant-lightness version, with every color still reading correctly:
# light green, amber-gold, a real red.
#
# The lightness ordering that falls out (DANGER darkest, SUCCESS lightest) is
# NOT a claim that danger should be dim. It is what the sRGB gamut allows at
# constant chroma: red desaturates to pink as it lightens, so a bright DANGER
# stops looking like danger. The ordering is a consequence, not an intent.
#
# Colour alone still cannot carry severity — the triad rides the red-green axis
# that protan/deutan collapse, and no palette fixes that. That is precisely why
# WCAG 1.4.1 demands a second channel. Every severity callsite pairs the color
# with an icon or a label, and tests/unit/test_color_tokens.py now asserts that
# rather than trusting this paragraph.
#
# These replace SUCCESS=emerald-400, WARNING=amber-400, DANGER=red-400,
# INFO=blue-400, FORECAST=orange-500 — five tokens byte-identical to stock
# Tailwind (CIEDE2000 0.00), which an independent re-audit named as the specific
# reason this palette read as "downloaded, not designed".

FORECAST = "#fba962"  # --forecast: forward-looking series (hue 60)

SUCCESS = "#90e9a3"  # --success (hue 150)
WARNING = "#c5a93b"  # --warning (hue 95)
DANGER = "#cf6963"  # --danger (hue 25)
INFO = "#5aa3ec"  # --info (hue 250)

# There is no SEVERITY dict here, and that is deliberate — there was one, it had
# ZERO callsites, and this comment used to claim it meant severity "can never
# drift apart again".
#
# It was keyed critical/warning/info/ok. Nothing in the product speaks that
# vocabulary: insights.py grades info/notable/warning, error_handling.py grades
# confidence high/medium/low/demo. So it unified nothing; it was a fourth
# spelling of severity invented while deleting three, dressed as the fix. The
# FUEL_COLORS-with-zero-callsites defect, one file over, by the author of the
# module that exists to prevent it.
#
# The real contract is narrower and already holds: this module owns the VALUES
# (SUCCESS/WARNING/DANGER/INFO); each domain maps its own vocabulary onto them
# at its own callsite, which is where that knowledge belongs. Two mappings
# exist, they are not rivals, and neither carries a color literal.

# ── Borders ──────────────────────────────────────────────────────────

BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"  # --border-subtle
BORDER_DEFAULT = "rgba(255, 255, 255, 0.12)"  # --border-default
# (BORDER_STRONG existed here too, mirroring --border-strong, and was
#  referenced by neither Python nor any CSS rule. Removed by the callsite
#  test rather than by inspection — which is the point of the test.)

TRANSPARENT = "rgba(0,0,0,0)"

# ── Chart chrome ─────────────────────────────────────────────────────
#
# Deliberately below the border ramp: Plotly's stock dark grid competes with
# the data. These are barely visible against --bg-raised, which is the cue we
# want — gridlines guide the eye without becoming figure.

GRID_LINE = "rgba(255, 255, 255, 0.04)"
ZERO_LINE = "rgba(255, 255, 255, 0.08)"
AXIS_LINE = "rgba(255, 255, 255, 0.10)"

# Hover pill — reads as part of the app surface, not a Plotly-default light pill.
HOVER_BG = BG_RAISED
HOVER_BORDER = AXIS_LINE

# ── Colorblind-safe categorical palette ──────────────────────────────
#
# Wong (2011) "Points of View: Color blindness", Nature Methods. This is an
# external scientific standard, NOT a brand choice — do not retune it. It
# encodes MODEL IDENTITY, always paired with a dash pattern
# (``accessibility.LINE_STYLES``) so the encoding survives grayscale and all
# three CVD types (WCAG 1.4.1 double-encoding).
#
# ``blue`` no longer backs "actual demand" — the accent does. It remains the
# palette's blue slot and is used by ``_COLORWAY``.

CB_PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",  # Prophet
    "green": "#009E73",  # ARIMA
    "vermillion": "#D55E00",  # Ensemble
    "sky_blue": "#56B4E9",  # XGBoost
    "yellow": "#F0E442",  # Solar / warning
    "purple": "#CC79A7",
    "black": "#000000",
}

NEUTRAL_SERIES = "#7f7f7f"  # EIA reference forecast / coal — deliberately gray

# ── Fuel mix — designed for the 9-band stacked area ──────────────────
#
# NOT a downloaded palette and NOT Wong: nine categories exceed what any
# categorical scheme can keep distinguishable under CVD, so this is built on
# LUMINANCE separation between bands that physically touch in
# ``_FUEL_STACK_ORDER``, with hue carrying semantics (sooty coal, flame-orange
# gas, sun-yellow solar) as a secondary channel.
#
# It replaces a palette with two measured defects:
#   * nuclear #a855f7 vs hydro #3b82f6 — ADJACENT bands, deutan CIEDE2000 1.0
#     and dL* 2.1. Effectively one band for ~8% of men, and grayscale could
#     not recover it either.
#   * wind #34d399 vs the accent net-load line drawn OVER the stack — 6.5.
#
# Every adjacent pair now clears CIEDE2000 >= 12 under normal/protan/deutan/
# tritan, and every band clears the accent. ``scripts/verify_palette.py``
# proves it; ``tests/unit/test_color_tokens.py`` enforces it.
# Re-verify with a CVD simulator after ANY edit.
#
# WHY NUCLEAR IS WINE AND NOT PURPLE. Purple is the convention. Wine is a
# CHOICE, made for margin — and this comment previously claimed it was a
# necessity, which was false.
#
# The correction, recorded because it is the exact failure this module exists to
# prevent: the original sweep found zero violet, and that was true AGAINST THE
# PHASE-1 ACCENT (#35c6ff), a blue that violet collapses onto under deuteranopia.
# Phase 2 moved the accent to a teal-cyan and the sweep was never re-run, so the
# comment went on asserting a dead constraint. Re-run against the accent that
# actually ships, 61 violet candidates clear every constraint — including
# Tailwind purple-400, at other 19.5 / hydro 14.6 / accent 14.6.
#
# Wine stays because it holds 19.7 at its worst against a floor of 12.0, where
# the best violet holds 15.1 and the best-separated ones are near-duplicates of
# stock indigo-400. More headroom on the pair that was the original defect.
#
# The lesson is the one this repo keeps re-learning: a measurement is only true
# against the inputs it was run on. This comment was prose, so nothing re-ran
# it when the inputs changed. Anything load-bearing belongs in
# scripts/verify_palette.py, where it re-runs on every commit.

FUEL_COLORS = {
    "coal": "#5e646a",  # L* 42.1 — dark neutral, sooty
    "oil": "#4f3f31",  # L* 28.0 — near-black warm brown
    "gas": "#eb883b",  # L* 66.2 — flame orange
    "biomass": "#8a713c",  # L* 48.9 — olive-brown
    "other": "#898d91",  # L* 58.4 — neutral gray
    "nuclear": "#b13554",  # L* 42.1 — wine; see "why not purple" below
    "hydro": "#2672b7",  # L* 46.8 — mid blue; dropped to clear nuclear
    "wind": "#b7cca4",  # L* 79.6 — sage; pulled off the accent's cyan (was 12.0, now 19.4)
    "solar": "#f9e03f",  # L* 88.8 — bright sun yellow
}

# Stack order, bottom -> top (fossil -> firm -> renewable). The luminance
# separation above is guaranteed for ADJACENT pairs in THIS order; reordering
# the stack invalidates the verification.
FUEL_STACK_ORDER: tuple[str, ...] = (
    "coal",
    "oil",
    "gas",
    "biomass",
    "other",
    "nuclear",
    "hydro",
    "wind",
    "solar",
)

# ── Persona identity ─────────────────────────────────────────────────
#
# Rendered as the insight card's 4px left border. These were matplotlib's
# tab10 defaults (#1f77b4 / #2ca02c / #ff7f0e / #9467bd) — a fifth parallel
# color system, in a repo that already had three. Remapped onto Wong, which
# preserves each persona's established hue while removing the stray palette.
PERSONA_COLORS = {
    "grid_ops": CB_PALETTE["blue"],
    "renewables": CB_PALETTE["green"],
    "trader": CB_PALETTE["orange"],
    "data_scientist": CB_PALETTE["purple"],
}

# ── Weather drivers ──────────────────────────────────────────────────
#
# The Overview's three driver sparklines: Temperature, Wind, Solar irradiance.
#
# temperature is pinned to the Wong yellow because LINE_STYLES draws the same
# quantity in it on the Weather tab — one concept, one color. wind reuses the
# wind-GENERATION color: same physical driver, so sharing is correct rather than
# accidental.
#
# solar does NOT reuse FUEL_COLORS["solar"], and that is the interesting one:
# the solar-generation yellow measures CIEDE2000 0.4 from the temperature yellow
# it would sit beside — the same color. So the irradiance driver is derived on
# the semantic rule (anchor chroma, hue 50, L 0.78) at a hue solved to clear
# both neighbours. Reusing the "obviously right" token would have shipped two
# identical sparklines labelled Temperature and Solar.
#
# These were declared and then only ONE THIRD wired: the sparkline list passed
# tokens.SUCCESS for Wind and tokens.FORECAST for Solar — semantic tokens doing
# duty as driver identities — so this dict had two dead entries while
# verify_palette dutifully checked colors that never rendered. Exactly the
# FUEL_COLORS-with-zero-callsites defect this module was written to kill,
# reproduced by its own author. tests/unit/test_color_tokens.py now asserts every
# token group is actually referenced.
WEATHER_DRIVERS = {
    "temperature": CB_PALETTE["yellow"],
    "wind": FUEL_COLORS["wind"],
    "solar": "#fa9d68",
}

# ── US-Grid choropleth ───────────────────────────────────────────────

MAP_LAND = BG_RAISED
MAP_COASTLINE = "#1d2a2b"
MAP_SUBUNIT = "#162223"
MAP_AXIS_FONT = TEXT_TERTIARY
MAP_BORDER = alpha(TEXT_PRIMARY, 0.5)

# Utilization / grid-stress colorscale (0 = idle headroom -> 1 = peak).
#
# CVD-safe and LUMINANCE-MONOTONIC (L* 20.8 / 42.1 / 54.2 / 66.5 / 81.5): it
# rides the blue-yellow axis preserved under protan/deutan and climbs steadily
# in brightness, so stress reads in grayscale and under tritanopia. Replaced a
# emerald->red ramp that was a WCAG 1.4.1 failure (green<->red collapses for
# ~8% of men). Do not regress the monotonicity — re-verify any edit with
# scripts/verify_palette.py AND a CVD simulator.
MAP_COLORSCALE = [
    [0.00, "#26324a"],  # dim slate-blue — idle / comfortable headroom
    [0.40, "#3f6690"],  # muted blue — running easy
    [0.60, "#6f83a1"],  # blue-gray — getting tight
    [0.80, "#c99a44"],  # amber — warning
    [1.00, "#f4c531"],  # bright amber — peak / stressed
]


# ── Not a design color ───────────────────────────────────────────────

# components/icons.py builds an SVG that is used as a CSS mask: only the
# glyph's ALPHA matters, and the visible color comes from `background-color:
# currentColor` showing through. The stroke just has to be opaque. It lives
# here anyway so the rule stays absolute — hex appears in exactly one file,
# with no "this one doesn't count" escape hatch. The last convention rotted
# because it was a comment rather than a rule.
ICON_MASK_STROKE = "#000"
