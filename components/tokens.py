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
# Every neutral below is generated from the ACCENT's own hue (196°) at low
# chroma, on a deliberate curve. It replaces stock Tailwind zinc, which sat at
# CIEDE2000 0.00 from the download — literally the shipped values, at a hue
# (~286°) unrelated to anything else in the product.
#
# The curve: chroma = 0.019 · exp(−((L − 0.38)/0.34)²), skewed toward the DARK
# end rather than symmetric. The rationale is TINT THE ROOM, NOT THE INK. Dark
# surfaces are large, so a trace of the brand hue there reads as atmosphere;
# the same chroma in text reads as a rendering fault, so it tapers off as
# lightness rises. Zinc's curve peaks at ~0.0146 around mid-tone and is
# symmetric — a different shape, for no reason, because it was never chosen.
#
# LIGHTNESS IS DELIBERATELY UNCHANGED from the zinc ramp it replaces (L* 2.8 →
# 2.9, 90.7 → 90.8, and so on). The value architecture of the UI was already
# right; only the hue bias was borrowed. Changing both at once would have made
# a visual regression impossible to attribute.
#
# The one exception is --text-tertiary, lifted from L* 47.9 to 51.0. At zinc's
# lightness it measured 4.09:1 on --bg-base — below WCAG AA for normal text,
# while rendering 11px chart tick labels. Its lightness is now SOLVED for
# 4.55:1 rather than sampled and hoped for.

BG_BASE = "#050c0c"  # --bg-base: page background, deepest
BG_RAISED = "#0a1313"  # --bg-raised: cards, dropdowns, badges
BG_HOVER = "#101a1a"  # --bg-hover: interactive hover state
SURFACE_SUNKEN = "#030707"  # --surface-sunken: recessed track/inset

TEXT_PRIMARY = "#e3e5e5"  # --text-primary
TEXT_SECONDARY = "#9da4a3"  # --text-secondary
TEXT_TERTIARY = "#707c7b"  # --text-tertiary — solved for AA, not sampled
TEXT_DISABLED = "#485656"  # --text-disabled (WCAG 1.4.3 exempts inactive UI)

# ── Accent — the brand hue (mirrors --accent-base / --accent-hover) ───
#
# OKLCh(0.79, 0.125, 196°) — a spectral teal-cyan, and the anchor the neutral
# ramp above is generated from.
#
# It replaces #35c6ff, which was CIEDE2000 2.5 from Tailwind sky-400: a
# near-duplicate of a stock swatch, reached after a previous pass had moved off
# stock blue-500 and landed on a different stock blue. This one is 18.6 from
# sky-400 and 7.8 from its nearest Tailwind neighbour.
#
# Distance from Tailwind is NOT the point and is a poor target on its own —
# Tailwind's 242 swatches tile color space, so optimising for pure distance
# drives you to neon (#00fdfd), the least owned color there is. What makes this
# palette the product's own is that ONE anchor generates the neutrals through a
# stated curve, and every relationship in it is measured.
#
# This is ALSO the demand/primary data series (``COLORS["actual"]``,
# ``LINE_STYLES["actual"]``, ``_COLORWAY[0]``) so brand color and data color
# are one system rather than two. ``scripts/verify_palette.py`` proves it
# clears every series it shares a figure with under all three CVD types.

ACCENT = "#33d3d5"  # --accent-base
ACCENT_SOFT = "#7ce5e6"  # --accent-hover

# ── Forecast + semantic (mirrors --forecast / --success / ...) ────────

FORECAST = "#f97316"  # --forecast: forward-looking series

SUCCESS = "#34d399"  # --success
WARNING = "#fbbf24"  # --warning
DANGER = "#f87171"  # --danger
INFO = "#60a5fa"  # --info

# THE severity triad. Two rival triads used to exist alongside this one —
# (#FF5C7A/#FFB84D/#2BD67B) and an Okabe-Ito ``SEVERITY_COLORS`` — for the
# same three-level concept. Both are deleted; this maps severity onto the
# semantic tokens above so severity and status can never drift apart again.
#
# Severity is never carried by color ALONE at any callsite (each pairs the
# color with an icon, a label, or both) — required by WCAG 1.4.1, because
# --success/--danger sit on the red-green axis that protan/deutan collapse.
SEVERITY = {
    "critical": DANGER,
    "warning": WARNING,
    "info": INFO,
    "ok": SUCCESS,
}

# ── Borders ──────────────────────────────────────────────────────────

BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"  # --border-subtle
BORDER_DEFAULT = "rgba(255, 255, 255, 0.12)"  # --border-default
BORDER_STRONG = "rgba(255, 255, 255, 0.18)"  # --border-strong

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
# WHY NUCLEAR IS NOT PURPLE. Purple is the convention (and was the shipped
# choice) but it is not available here, and that is a measurement, not taste.
# Under deuteranopia violet collapses onto the same blue axis that hydro and
# the accent already occupy. Sweeping the whole violet->crimson space — hue
# 260-375 x L* 40-94 x five chroma steps — yields ZERO violet candidates that
# clear hydro (adjacent band) AND the accent net-load line drawn across the
# stack; the first shipped attempt at a violet measured 3.7 against the accent.
# Only crimson survives, at ~20. Wine it is.

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
# The Overview's three driver sparklines. Temperature used to be #3b82f6 here
# while the Weather tab drew the same quantity in Wong yellow — one concept,
# two hexes, and the #3b82f6 was also byte-identical to hydro (the "hydro and
# temperature share a hex" defect). Temperature now resolves to the Wong
# yellow everywhere: it is the hue that clears the accent demand line it is
# plotted against (CIEDE2000 39.2) and it is hue-distinct from hydro (71.1).
#
# Solar deliberately does NOT reuse FUEL_COLORS["solar"]: that yellow is
# CIEDE2000 3.2 from the temperature yellow, which would render the
# Temperature and Solar sparklines identical side by side. The trio below is
# mutually >= 15.5 under all three CVD types.
WEATHER_DRIVERS = {
    "temperature": CB_PALETTE["yellow"],
    "wind": FUEL_COLORS["wind"],  # same physical driver as wind generation
    "solar": FORECAST,
}

# ── US-Grid choropleth ───────────────────────────────────────────────

MAP_LAND = BG_RAISED
MAP_COASTLINE = "#1e2a2a"
MAP_SUBUNIT = "#172222"
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
