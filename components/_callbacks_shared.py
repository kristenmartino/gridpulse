"""Shared infrastructure for ``components/callbacks.py``.

Step 1 of the ``callbacks.py`` decomposition tracked in issue #87. This
module owns the cross-cutting helpers and module-level state that every
tab callback uses — caches, design tokens, layout helpers, data-hash
computation, color palettes, basemap constants.

Per-tab helpers stay in ``callbacks.py`` for now and will move into
``_callbacks_<tab>.py`` modules in follow-up PRs. The decomposition
plan (``docs/internal/CALLBACKS_DECOMPOSITION_PLAN.md``) tracks the
full sequence.

## Public-import surface

``components/callbacks.py`` does ``from components._callbacks_shared
import *`` at the top, so every symbol listed in ``__all__`` becomes
accessible at ``from components.callbacks import <X>`` — preserving the
40+ import sites in ``app.py`` and ``tests/`` without any caller-side
changes.

Mutable singletons (``_MODEL_CACHE``, ``_PREDICTION_CACHE``, etc.) are
defined once here and re-exported by reference. Mutations from inside
``callbacks.py`` (and from the per-tab modules when they land) all see
the same dict — Python passes mutable objects by reference, so the
re-export pattern is safe for shared state as well as constants.
"""

from __future__ import annotations

import threading

import numpy as np

from components.accessibility import CB_PALETTE

# ── Cache state ──────────────────────────────────────────────────────
#
# Process-level caches used across multiple tabs. Module-level dicts so
# they survive across callbacks without needing a singleton class. The
# lock protects multi-step mutations (e.g. cache eviction + insert).

_cache_lock = threading.Lock()
_CACHE_VERSION = 3

_MODEL_CACHE: dict[tuple, tuple] = {}
"""(region, model_name, horizon) → (model, data_hash, timestamp)"""

_PREDICTION_CACHE: dict[tuple, tuple] = {}
"""(region, horizon) → (predictions, timestamps, data_hash, time)"""

_BACKTEST_CACHE: dict[tuple, dict] = {}
"""(region, horizon, model, exog_mode) → (result_dict, data_hash, time)"""

_GENERATION_CACHE: dict[str, tuple] = {}
"""region → (gen_df, fetch_timestamp)"""

BACKTEST_EXOG_MODES = {"oracle_exog", "forecast_exog"}
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"


# ── EIA fuel-type normalization ──────────────────────────────────────
#
# The EIA-930 generation API returns fuel codes that vary across endpoints
# (short codes like "SUN" from the v2 fuel-mix endpoint vs. spelled-out
# "Solar" from elsewhere). The map normalizes to the canonical lowercase
# names used by the UI's color palette + the generation-tab grouping.

_EIA_FUEL_MAP: dict[str, str] = {
    "SUN": "solar",
    "WND": "wind",
    "NG": "gas",
    "NUC": "nuclear",
    "COL": "coal",
    "WAT": "hydro",
    "OTH": "other",
    "Solar": "solar",
    "Wind": "wind",
    "Natural Gas": "gas",
    "Nuclear": "nuclear",
    "Coal": "coal",
    "Hydro": "hydro",
    "Other": "other",
}


# ── Plotly layout tokens ─────────────────────────────────────────────

PLOT_TEMPLATE = "plotly_dark"
PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        color="#a1a1aa",  # --text-secondary (v2 palette)
        size=11,
        family="Inter, 'Segoe UI', system-ui, sans-serif",
    ),
    margin=dict(l=48, r=16, t=24, b=36),
    legend=dict(
        orientation="h",
        y=-0.18,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)


def _layout(*, uirevision: str | None = None, **overrides) -> dict:
    """Compose a Plotly layout dict from PLOT_LAYOUT + per-call overrides.

    ``uirevision`` is the Plotly hook that preserves user-set UI state
    (zoom, pan, legend toggles) across figure re-renders. Tie it to
    *user-meaningful* state — region, horizon, model — NOT to a raw data
    hash, so that auto-refreshes (5-min interval, store updates) reuse
    the same revision string and the chart's interaction state survives.
    Changing the string forces Plotly to re-initialize the view, which
    is the desired behavior when the user picks a new region/horizon.
    """
    layout = {**PLOT_LAYOUT, **overrides}
    if uirevision is not None:
        layout["uirevision"] = uirevision
    return layout


# ── Color palette (colorblind-safe per Wong 2011) ────────────────────

COLORS = {
    "actual": CB_PALETTE["blue"],
    "prophet": CB_PALETTE["orange"],
    "arima": CB_PALETTE["green"],
    "xgboost": CB_PALETTE["sky_blue"],
    "ensemble": CB_PALETTE["vermillion"],
    "eia_forecast": "#7f7f7f",
    "temperature": CB_PALETTE["yellow"],
    "confidence": "rgba(213,94,0,0.15)",
    "gas": CB_PALETTE["orange"],
    "nuclear": CB_PALETTE["purple"],
    "coal": "#7f7f7f",
    "wind": CB_PALETTE["green"],
    "solar": CB_PALETTE["yellow"],
    "hydro": CB_PALETTE["blue"],
    "other": "#b0b0b0",
}

# Model-aware confidence-band fill colors — base model color at 12% opacity.
# Used by ``_add_confidence_bands`` to tint the band per-model so a
# multi-model overlay reads at a glance.
_MODEL_BAND_COLORS = {
    "xgboost": "rgba(86,180,233,0.12)",  # sky_blue
    "prophet": "rgba(230,159,0,0.12)",  # orange
    "arima": "rgba(0,158,115,0.12)",  # green
    "ensemble": "rgba(213,94,0,0.12)",  # vermillion
}


# ── Demand-series helpers ────────────────────────────────────────────


def _latest_real_demand(values, *, offset: int = 0):
    """Return the most recent real demand reading from a list / Series.

    Walks backward and returns the first value that is finite (not NaN
    or inf) and strictly positive. ``offset=0`` finds "now"; ``offset=24``
    finds "24 hours ago" while still skipping any NaN / zero rows that
    fall on the chosen position — so a NaN spike doesn't silently shift
    a trend baseline.

    Returns ``None`` when no usable reading exists in the window — the
    caller should render an ``"—"`` placeholder rather than a numeric.

    Cross-tab helper: consumed by both the Overview hero chart and the
    US Grid card-grid collector. Lives in shared because both tabs
    depend on the same NaN-guard semantics.
    """
    if values is None:
        return None
    try:
        n = len(values)
    except TypeError:
        return None
    if n == 0:
        return None
    skipped = 0
    for i in range(n - 1, -1, -1):
        try:
            v = float(values.iloc[i] if hasattr(values, "iloc") else values[i])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        if skipped < offset:
            skipped += 1
            continue
        return v
    return None


# ── Stress / reliability tokens ──────────────────────────────────────
#
# Stress ratios above this threshold are treated as structural import-
# dominance rather than measurable load. Surfaces in the US-Grid metrics
# bar (highest-stress region exclusion), the region cards (qualitative
# "imports" chip in place of a misleading percentage), and the polygon
# choropleth's stress-cap logic. ``IS_IMPORT_DOMINATED`` from config.py
# is the explicit tag; this threshold is the belt-and-braces fallback
# for BAs that aren't tagged but produce out-of-band ratios.

_STRESS_RELIABLE_CEILING = 2.0


# ── Choropleth + scatter-map basemap tokens ──────────────────────────
#
# Single source of truth for the US-Grid map visual style. The polygon
# border color is intentionally translucent so it reads against any
# colorscale fill — green, yellow, or red — without becoming the visual
# focus. See PR #79 for the regression that motivated the contrast bump.

_MAP_LAND_COLOR = "#111113"  # --bg-raised
_MAP_COASTLINE_COLOR = "#27272a"
_MAP_SUBUNIT_COLOR = "#1f1f23"
_MAP_AXIS_FONT_COLOR = "#71717a"  # --text-tertiary

# Polygon borders need to read against ANY colorscale fill — green, yellow,
# or red. A near-black border (#1f1f23) was nearly identical to the
# darkest polygon fill, so adjacent BAs visually fused into one blob on
# the Polygons view. A translucent zinc reads against everything.
_MAP_BORDER_COLOR = "rgba(228, 228, 231, 0.5)"

# Five-stop colorscale: the previous three-stop scale (0.0 → 0.7 → 1.0,
# green → yellow → red) put 30% and 60% utilization at nearly identical
# greens because both fall on the long [0.0, 0.7] interpolation segment.
# Most BAs operate in the 30–70% band, so spreading colors there is what
# the eye actually needs.
_MAP_COLORSCALE = [
    [0.00, "#10b981"],  # emerald-500 — idle / comfortable headroom
    [0.40, "#84cc16"],  # lime-500 — running easy
    [0.60, "#eab308"],  # yellow-500 — getting tight
    [0.80, "#f97316"],  # orange-500 — warning
    [1.00, "#dc2626"],  # red-600 — peak / stressed
]


# ── Module re-export surface ─────────────────────────────────────────
#
# ``components/callbacks.py`` does ``from components._callbacks_shared
# import *``. ``__all__`` controls which names that star-import exposes;
# we list both the public names (no leading underscore) and the private
# helpers / caches that ``app.py`` and tests import via
# ``from components.callbacks import _MODEL_CACHE`` etc.

__all__ = [
    # Caches + lock
    "_cache_lock",
    "_CACHE_VERSION",
    "_MODEL_CACHE",
    "_PREDICTION_CACHE",
    "_BACKTEST_CACHE",
    "_GENERATION_CACHE",
    # Backtest modes
    "BACKTEST_EXOG_MODES",
    "DEFAULT_BACKTEST_EXOG_MODE",
    # EIA normalization
    "_EIA_FUEL_MAP",
    # Plotly layout
    "PLOT_TEMPLATE",
    "PLOT_LAYOUT",
    "_layout",
    # Color palette
    "COLORS",
    "_MODEL_BAND_COLORS",
    # Demand-series helpers
    "_latest_real_demand",
    # Stress thresholds
    "_STRESS_RELIABLE_CEILING",
    # Map tokens
    "_MAP_LAND_COLOR",
    "_MAP_COASTLINE_COLOR",
    "_MAP_SUBUNIT_COLOR",
    "_MAP_AXIS_FONT_COLOR",
    "_MAP_BORDER_COLOR",
    "_MAP_COLORSCALE",
]
