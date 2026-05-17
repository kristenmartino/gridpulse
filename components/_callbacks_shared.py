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
import plotly.graph_objects as go

from components.accessibility import CB_PALETTE
from data.redis_client import redis_get

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


def _empty_figure(message: str = "") -> go.Figure:
    """Render an axis-less placeholder figure with a centered message.

    Cross-tab fallback: every tab needs the "no data yet / select X / loading"
    visual when its primary chart can't render. Centralised here so all tabs
    share the same dark-mode styling and the same uirevision key (``"empty"``)
    that keeps Plotly from re-initializing the view on each re-render.
    """
    fig = go.Figure()
    fig.update_layout(
        **_layout(
            uirevision="empty",
            annotations=[
                dict(
                    text=message,
                    showarrow=False,
                    font=dict(size=14, color="#A8B3C7"),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
    )
    return fig


# ── Data signature for cache correctness (cross-tab) ─────────────────
#
# Used by both the Forecast tab's outlook generator and the Backtest
# tab's horizon runner to key prediction / backtest caches on the
# *content* of the input frames (not just their shape). Two frames with
# different demand_mw values produce different hashes; identical content
# regardless of column ordering / timestamp parsing produces the same.


def _compute_data_hash(demand_df, weather_df, region: str) -> str:
    """Compute stable input signature for cache correctness.

    Signature includes:
    - region
    - row counts
    - normalized start/end timestamps
    - lightweight content checksums over key columns
    """
    import hashlib
    import json

    import pandas as pd

    def _normalize_ts(ts) -> str:
        """Strip timezone to produce a stable string regardless of tz-aware vs tz-naive."""
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return str(t)

    def _frame_sig(df, key_cols: list[str]) -> dict:
        frame_sig: dict[str, str | int] = {
            "rows": int(len(df)),
            "start": "",
            "end": "",
            "checksum": "",
        }
        if df.empty:
            return frame_sig

        if "timestamp" in df.columns:
            ts_bounds = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if ts_bounds.notna().any():
                frame_sig["start"] = _normalize_ts(ts_bounds.min())
                frame_sig["end"] = _normalize_ts(ts_bounds.max())

        cols = [c for c in key_cols if c in df.columns]
        if not cols:
            return frame_sig

        sample = df.loc[:, cols].copy()
        if "timestamp" in sample.columns:
            ts = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
            sample["timestamp"] = ts.astype("int64").fillna(-1).astype("int64")
            sample = sample.sort_values("timestamp", kind="mergesort")
        for col in cols:
            if col != "timestamp" and pd.api.types.is_numeric_dtype(sample[col]):
                sample[col] = sample[col].round(6)

        hashed = pd.util.hash_pandas_object(sample.fillna("<NA>"), index=False).to_numpy(
            dtype=np.uint64
        )
        frame_sig["checksum"] = f"{int(hashed.sum(dtype=np.uint64)):016x}"
        return frame_sig

    signature_payload = {
        "region": region,
        "demand": _frame_sig(demand_df, ["timestamp", "demand_mw"]),
        "weather": _frame_sig(
            weather_df,
            [
                "timestamp",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
            ],
        ),
    }
    signature_json = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


# ── Prediction-interval helpers (cross-tab) ──────────────────────────
#
# Both the Forecast tab (confidence bands around the live forecast)
# and the Backtest tab (residual-derived intervals shown alongside the
# holdout chart) need empirical-error quantiles. They share the same
# residual source: in-memory ``_BACKTEST_CACHE`` for recent in-process
# computes plus the scoring job's Redis backtests for the production
# baseline. Living in shared lets both tab modules import without
# cross-tab dependencies.


def _collect_backtest_residuals(region: str, model_name: str, horizon_hours: int) -> np.ndarray:
    """Collect recent backtest residuals by model/region/horizon from cache layers."""
    residual_chunks: list[np.ndarray] = []

    # In-memory cache (most recent in-process compute path)
    for (r, h, m, _mode), (cached_result, _hash, _time) in _BACKTEST_CACHE.items():
        if r != region or h != horizon_hours:
            continue
        if m != model_name and not (model_name == "ensemble" and m in ("ensemble",)):
            continue
        actual = np.asarray(cached_result.get("actual", []), dtype=float)
        pred = np.asarray(cached_result.get("predictions", []), dtype=float)
        n = min(len(actual), len(pred))
        if n > 0:
            residual_chunks.append(actual[:n] - pred[:n])

    # Redis pre-computed backtests (common production path)
    for key in (
        f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon_hours}",
        f"wattcast:backtest:{region}:{horizon_hours}",
    ):
        cached = redis_get(key)
        if not isinstance(cached, dict):
            continue
        actual = np.asarray(cached.get("actual", []), dtype=float)
        preds_map = cached.get("predictions", {})
        if isinstance(preds_map, dict):
            if model_name in preds_map:
                pred = np.asarray(preds_map.get(model_name, []), dtype=float)
            elif "ensemble" in preds_map:
                pred = np.asarray(preds_map.get("ensemble", []), dtype=float)
            elif preds_map:
                pred = np.asarray(next(iter(preds_map.values())), dtype=float)
            else:
                pred = np.array([], dtype=float)
            n = min(len(actual), len(pred))
            if n > 0:
                residual_chunks.append(actual[:n] - pred[:n])

    if not residual_chunks:
        return np.array([], dtype=float)
    residuals = np.concatenate(residual_chunks)
    return residuals[np.isfinite(residuals)]


def _empirical_interval_from_backtests(
    region: str,
    model_name: str,
    horizon_hours: int,
    target_coverage: float = 0.80,
) -> dict[str, float | int | bool]:
    """Estimate empirical prediction interval from recent backtest residuals."""
    from models.evaluation import empirical_error_quantiles

    residuals = _collect_backtest_residuals(region, model_name, horizon_hours)
    if residuals.size < max(24, horizon_hours // 2):
        return {"available": False}

    tail_size = int(min(residuals.size, max(horizon_hours * 5, 120)))
    recent = residuals[-tail_size:]
    alpha = (1.0 - target_coverage) / 2.0
    q = empirical_error_quantiles(recent, lower_q=alpha, upper_q=1.0 - alpha)
    return {
        "available": True,
        "lower_error": float(q["lower_error"]),
        "upper_error": float(q["upper_error"]),
        "sample_size": int(q["sample_size"]),
        "target_coverage": float(target_coverage),
        "calibration_window_hours": tail_size,
    }


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
    "_empty_figure",
    # Color palette
    "COLORS",
    "_MODEL_BAND_COLORS",
    # Demand-series helpers
    "_latest_real_demand",
    # Data signature
    "_compute_data_hash",
    # Prediction-interval helpers (cross-tab)
    "_collect_backtest_residuals",
    "_empirical_interval_from_backtests",
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
