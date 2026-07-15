"""Design tokens — the only file in the Python layer where a color may be written.

``assets/custom.css`` ``:root`` is the browser's copy; this is the Plotly
layer's, because Plotly cannot read CSS custom properties. The two are asserted
equal by ``tests/unit/test_color_tokens.py``.

Enforcement, not convention:
  * ``scripts/check_color_tokens.py`` fails CI on a color literal anywhere else.
  * ``scripts/verify_palette.py`` recomputes every rule below and prints the
    measurements. It is the thing that states the numbers.
  * ``tests/unit/test_color_tokens.py`` asserts the same invariants, and that
    every token is actually painted.

Comments here state RULES and REASONS, not measurements — run the verifier for
those. This file previously carried ~230 lines of prose narrating its own
numbers and four of those claims were false, so the numbers now live only where
something recomputes them. The history is in git, where it cannot drift.
"""

from __future__ import annotations

# ── Helpers ──────────────────────────────────────────────────────────


def rgb(color: str) -> tuple[int, int, int]:
    """Return ``color`` as an ``(r, g, b)`` tuple, for PIL and other channel consumers.

    Exists so nothing hand-maintains a tuple: an RGB tuple is a color literal
    that a hex grep cannot see.

    Args:
        color: A ``#rrggbb`` token from this module.

    Returns:
        ``(r, g, b)``, each channel in ``[0, 255]``.
    """
    h = color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"rgb() expects #rrggbb, got {color!r}")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def alpha(color: str, a: float) -> str:
    """Return ``color`` as an ``rgba()`` string at alpha ``a``.

    Exists so a translucent fill is tied to its source color by construction — a
    hand-written ``rgba()`` is a literal the hex gate cannot see, and fills
    written that way have drifted off their own line color here before.

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


# ── The anchor ───────────────────────────────────────────────────────
#
# The one color a human chose. Every generated token below comes from it, so it
# is defended differently from the rest: it must not be a copy of an
# off-the-shelf swatch. Corpus: scripts/stock_palettes.py. Floor and rationale:
# verify_palette.STOCK_FLOOR.
#
# It is also the demand/primary data series (COLORS["actual"],
# LINE_STYLES["actual"], _COLORWAY[0]), so brand color and data color are one
# system rather than two.

ACCENT = "#35dde8"  # --accent-base
ACCENT_SOFT = "#59f4ff"  # --accent-hover: the accent lifted for hover

# ── Neutrals — generated from the anchor's hue at low chroma ─────────
#
# RULE: hue = the anchor's; chroma = a Gaussian on lightness, peaked below
# mid-tone; lightness = unchanged from the stock ramp this replaced, so any
# visual regression is attributable to the hue shift alone.
#
# WHY the peak is low: tint the room, not the ink. Dark surfaces are large, so a
# trace of the brand hue reads there as atmosphere; the same chroma in text
# reads as a rendering fault.
#
# --text-tertiary is the exception: its lightness is SOLVED for WCAG AA on
# --bg-base rather than chosen, because it renders 11px chart ticks.

BG_BASE = "#050c0c"  # --bg-base
BG_RAISED = "#0a1314"  # --bg-raised
BG_HOVER = "#101a1b"  # --bg-hover
SURFACE_SUNKEN = "#030708"  # --surface-sunken

TEXT_PRIMARY = "#e3e5e5"  # --text-primary
TEXT_SECONDARY = "#9da3a4"  # --text-secondary
TEXT_TERTIARY = "#707c7c"  # --text-tertiary — solved for AA
TEXT_DISABLED = "#485657"  # --text-disabled (WCAG 1.4.3 exempts inactive UI)

# ── Semantics — generated from the anchor ────────────────────────────
#
# RULE: chroma = the anchor's, or the sRGB gamut maximum at that hue, whichever
# is lower — nothing out-saturates the brand. Hue = convention. Lightness =
# SOLVED for separation under dichromacy, then clamped to a band where the color
# still reads as its meaning.
#
# WHY lightness is solved rather than held constant: under deuteranopia green,
# amber and red collapse toward one yellow, leaving lightness as the only
# channel. A constant-lightness version of this ramp shipped and was an
# accessibility regression.
#
# The lightness ORDER that falls out (danger darkest, success lightest) is not a
# claim that danger should be dim — it is what the gamut allows at constant
# chroma, since red desaturates toward pink as it lightens.
#
# Color alone cannot carry severity: the triad rides the red-green axis and no
# palette fixes that. WCAG 1.4.1 wants a second channel, and callsites do pair
# each color with an icon or a label — but NOTHING CHECKS THAT. Known gap,
# listed in verify_palette.

FORECAST = "#fba962"  # --forecast: forward-looking series
SUCCESS = "#90e9a3"  # --success
WARNING = "#c5a93b"  # --warning
DANGER = "#cf6963"  # --danger
INFO = "#5aa3ec"  # --info

# There is no SEVERITY dict here on purpose. One existed with zero callsites,
# keyed to a vocabulary nothing in the product speaks. This module owns the
# VALUES; each domain maps its own vocabulary onto them at its own callsite
# (insights.py grades info/notable/warning; error_handling.py grades confidence
# tiers), which is where that knowledge belongs.

# ── Borders and chart chrome ─────────────────────────────────────────
#
# Translucent white over the ground, so they pick up its hue. The chart lines
# sit deliberately below the border ramp: Plotly's stock dark grid competes with
# the data.

BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"  # --border-subtle
BORDER_DEFAULT = "rgba(255, 255, 255, 0.12)"  # --border-default

TRANSPARENT = "rgba(0,0,0,0)"

GRID_LINE = "rgba(255, 255, 255, 0.04)"
ZERO_LINE = "rgba(255, 255, 255, 0.08)"
AXIS_LINE = "rgba(255, 255, 255, 0.10)"

HOVER_BG = BG_RAISED
HOVER_BORDER = AXIS_LINE

# ── Wong palette — EXTERNAL, cited, not ours to retune ───────────────
#
# Wong (2011) "Points of View: Color blindness", Nature Methods. Encodes MODEL
# IDENTITY, always paired with a dash pattern (accessibility.LINE_STYLES) so the
# encoding survives grayscale and all three CVD types.

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

NEUTRAL_SERIES = "#7f7f7f"  # EIA reference forecast — deliberately gray

# ── Fuel mix — colour says the SOURCE, pattern says STORAGE ──────────
#
# Keyed to the full EIA-930 vocabulary (_callbacks_shared._EIA_FUEL_MAP), not to
# a list of fuels I enumerated. The previous nine covered eight real codes and
# invented "biomass", which EIA-930 does not have; BAT and SNB — which FPL
# returns every hour — fell through to ONE shared fallback and painted two
# adjacent bands in the same grey.
#
# COLOUR CANNOT CARRY THIS, and that is measured, not assumed: nine generation
# fuels need 36 mutually-separable pairs, and a search for a dark brown "oil"
# that clears the other eight returns ZERO candidates. Wong — the reference
# CVD-safe categorical set — tops out at EIGHT colours. Nine is over the limit.
#
# So there are two channels, and the pattern carries the FAMILY:
#
#   COLOUR  = the energy source, and it only has to work WITHIN a family.
#   PATTERN = which family. Fossil is hatched (the panel is titled "sorted by
#             emissions intensity", so the hatch says the thing the sort says).
#             SNB/WNB/PS are not new sources — they are solar/wind/hydro WITH
#             storage, so they carry their BASE fuel's colour and a different
#             hatch. Pure storage shares one colour and separates by pattern
#             alone: no third storage colour clears the generation set.
#
# tests/unit/test_rendered_figures.py measures every pair in the figure the app
# actually builds, against the fuels the DATA actually yields — not against a
# list of fuels or adjacencies enumerated here. The nine that preceded this
# guaranteed only ADJACENT pairs, which is the wrong bar: a reader matches a
# legend swatch to a band anywhere in the stack.
#
# Nuclear is wine rather than the conventional purple: violet collapses onto the
# blue axis hydro and the accent occupy under deuteranopia. Violet is available
# against the current accent but with less margin — a choice for headroom.
#
# Separation is verified. Ownership is NOT — known gap, listed in verify_palette.

FUEL_COLORS = {
    # generation — solid
    "coal": "#5e646a",
    "oil": "#4f3f31",
    "gas": "#eb883b",
    "nuclear": "#b13554",
    "geothermal": "#af7c5a",
    "hydro": "#2672b7",
    "wind": "#b7cca4",
    "solar": "#f9e03f",
    "other": "#898d91",
    # generation WITH integrated storage — base fuel's colour, hatched
    "wind_storage": "#b7cca4",
    "solar_storage": "#f9e03f",
    "pumped_storage": "#2672b7",
    # storage — one family, separated by pattern (they rarely co-occur, and no
    # third colour exists that clears the generation set)
    "battery": "#7a94b7",
    "other_storage": "#7a94b7",
    "unknown_storage": "#7a94b7",
    # unclassified
    "unknown": "#898d91",
}

# Plotly fillpattern shape per fuel — the FAMILY channel. "" is solid.
FUEL_PATTERNS = {
    # fossil — hatched: the panel sorts by emissions intensity, and this says so
    "coal": "\\",
    "oil": "\\",
    "gas": "\\",
    # clean generation — solid
    "nuclear": "",
    "geothermal": "",
    "hydro": "",
    "wind": "",
    "solar": "",
    "other": "",
    # generation WITH integrated storage — base fuel's colour, cross-hatched
    "wind_storage": "/",
    "solar_storage": "/",
    "pumped_storage": "/",
    # pure storage — one colour, separated by pattern alone
    "battery": "x",
    "other_storage": "+",
    "unknown_storage": "|",
    # unclassified
    "unknown": ".",
}

# Bottom -> top: fossil -> firm -> renewable -> storage -> unclassified.
# The luminance separation is guaranteed for ADJACENT pairs in THIS order;
# reordering invalidates it. Storage sits above generation because it is not
# generation — and BAT/PS go NEGATIVE while charging, which a stacked area
# cannot represent honestly. See the known gap in verify_palette.
FUEL_STACK_ORDER: tuple[str, ...] = (
    "coal",
    "oil",
    "gas",
    "geothermal",
    "other",
    "nuclear",
    "hydro",
    "pumped_storage",
    "wind",
    "wind_storage",
    "solar",
    "solar_storage",
    "battery",
    "other_storage",
    "unknown_storage",
    "unknown",
)

# ── Persona identity ─────────────────────────────────────────────────
#
# The insight card's left border. Mapped onto Wong, which keeps each persona's
# established hue without introducing another palette.

PERSONA_COLORS = {
    "grid_ops": CB_PALETTE["blue"],
    "renewables": CB_PALETTE["green"],
    "trader": CB_PALETTE["orange"],
    "data_scientist": CB_PALETTE["purple"],
}

# ── Weather drivers ──────────────────────────────────────────────────
#
# temperature is pinned to the Wong yellow because LINE_STYLES draws the same
# quantity in it on the Weather tab — one concept, one color. wind reuses the
# wind-GENERATION color: same physical driver, so sharing is correct.
#
# solar deliberately does NOT reuse FUEL_COLORS["solar"], which is
# indistinguishable from the temperature yellow it would sit beside. It is
# generated on the semantic rule at a hue solved to clear both neighbours.
#
# "Clear" there means the 10.0 adjacency floor, and that is the right floor for
# the one surface that draws these: _build_drivers_panel, three separately
# labeled sparkline cells, compared but never overplotted.
#
# It would NOT be enough for a figure that overplots two of them, and the
# margin to do that is not available: wind (sage green) and solar (orange) are
# 33.1 apart to normal vision but 11.5 under protanopia. Green-vs-orange IS the
# red-green axis, so no choice of hues rescues that pair for a dichromat — only
# lightness or form. If you ever draw two drivers as lines in one plot, they
# need a second channel (dash), the way LINE_STYLES carries the models.

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

# Utilization / grid stress (0 = idle -> 1 = peak).
#
# CVD-safe and LUMINANCE-MONOTONIC: it rides the blue-yellow axis preserved
# under protan/deutan and climbs steadily in brightness, so stress reads in
# grayscale and under tritanopia. Replaces an emerald->red ramp that was a WCAG
# 1.4.1 failure. Monotonicity is verified. Ownership is NOT — known gap.
MAP_COLORSCALE = [
    [0.00, "#26324a"],  # idle / comfortable headroom
    [0.40, "#3f6690"],  # running easy
    [0.60, "#6f83a1"],  # getting tight
    [0.80, "#c99a44"],  # warning
    [1.00, "#f4c531"],  # peak / stressed
]

# ── Not a design color ───────────────────────────────────────────────
#
# components/icons.py builds an SVG used as a CSS mask: only the glyph's ALPHA
# matters and the visible color comes from `background-color: currentColor`, so
# the stroke just has to be opaque. It lives here so the rule stays absolute —
# hex appears in one file, with no "this one doesn't count" exemption.
ICON_MASK_STROKE = "#000"
