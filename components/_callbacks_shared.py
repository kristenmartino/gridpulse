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

import textwrap
import threading

import numpy as np
import plotly.graph_objects as go

from components.accessibility import CB_PALETTE
from data.redis_client import redis_get, redis_key

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
#
# Issue #26: cross-cutting Plotly polish — hover label theming, axis
# tone, modebar trim. Defaults set here so every chart that flows
# through ``_layout()`` picks them up; per-chart overrides still win
# because ``_layout()`` does a shallow merge over ``PLOT_LAYOUT``.

PLOT_TEMPLATE = "plotly_dark"

# Modebar config for any chart that *wants* a visible modebar. The
# current GridPulse design hides the modebar entirely on every tab
# (each ``tab_*.py`` defines a local ``_GRAPH_CONFIG`` with
# ``displayModeBar: False`` — portfolio-dashboard context where no
# user is doing analytic exploration). This constant lives here as
# the cross-cutting alternative: when a future chart needs a
# user-facing zoom/pan/download bar, opt in with
# ``dcc.Graph(config=PLOT_CONFIG, ...)`` instead of inventing yet
# another local config dict.
#
# ``displaylogo=False`` strips the Plotly attribution.
# ``modeBarButtonsToRemove`` trims the lasso / select / autoscale /
# spike-line buttons (not useful for time-series demand data) so
# the only remaining buttons are zoom / pan / reset / download.
# ``responsive=True`` keeps charts legible on container resize.
PLOT_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
    "responsive": True,
}

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
    # Hover label — dark surface, Inter, left-aligned. Matches the
    # rest of the v2 chrome and reads as "part of the same surface"
    # rather than a Plotly-default light pill.
    hoverlabel=dict(
        bgcolor="#11141c",
        bordercolor="rgba(255,255,255,0.10)",
        font=dict(
            family="Inter, 'Segoe UI', system-ui, sans-serif",
            size=12,
            color="#F5F7FA",
        ),
        align="left",
    ),
    # Unified hover groups every trace's value at the same x — much
    # cleaner for time-series with multiple model overlays. Charts
    # that want trace-individual hovers (heatmaps, scatters) override
    # this in their own ``_layout()`` call.
    hovermode="x unified",
    # Subtle axis tone. Plotly's default dark-template grid is too
    # heavy — these are barely-visible against ``--bg-raised``,
    # which is exactly the cue we want (gridlines guide the eye
    # without competing with the data).
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.10)",
        tickfont=dict(color="#8892A5", size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.08)",
        linecolor="rgba(255,255,255,0.10)",
        tickfont=dict(color="#8892A5", size=11),
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

    Axis dicts merge instead of replace: ``PLOT_LAYOUT`` carries the
    shared axis tone (gridcolor / linecolor / tickfont) and callsites
    layer per-chart options (titles, tick format, showgrid toggles) on
    top. Without this merge, a callsite that passes ``xaxis=dict(...)``
    would silently lose the shared styling.
    """
    layout = {**PLOT_LAYOUT, **overrides}
    # Deep-merge xaxis / yaxis (and any yaxis2/yaxis3 etc. for dual-axis
    # plots): combine the PLOT_LAYOUT defaults with the per-call override,
    # with the override winning on conflicting keys.
    for axis_key, default_value in PLOT_LAYOUT.items():
        if not (axis_key.startswith("xaxis") or axis_key.startswith("yaxis")):
            continue
        override_value = overrides.get(axis_key)
        if override_value is None:
            continue  # only the default applies — leave PLOT_LAYOUT entry alone
        if isinstance(default_value, dict) and isinstance(override_value, dict):
            layout[axis_key] = {**default_value, **override_value}
        # else: scalar override (rare) — already handled by the outer
        # ``{**PLOT_LAYOUT, **overrides}`` above.
    # Plotly also accepts dual-axis configs the callsite didn't have a
    # default for (e.g. ``yaxis2``). Those pass through unchanged from
    # the override dict — they're already in ``layout`` from the
    # outer merge above.
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
                    # Plotly annotations never auto-wrap, and an explicit
                    # ``width`` CLIPS overflowing text instead of wrapping it
                    # (long messages rendered only their centered middle).
                    # ``<br>`` is the only line break Plotly honors, so wrap
                    # here: ~40 chars ≈ 240px at 13px, narrow enough for the
                    # 3-up residual grid on the Models tab.
                    text="<br>".join(textwrap.wrap(message, width=40)),
                    showarrow=False,
                    font=dict(size=13, color="#A8B3C7"),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    align="center",
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


def _collect_backtest_residuals(
    region: str,
    model_name: str,
    horizon_hours: int,
    horizon_resolved: bool = False,
) -> tuple[np.ndarray, str | None]:
    """Collect recent backtest residuals by model/region/horizon from cache layers.

    Returns ``(residuals, calibration_model)``. The production backtest
    payload only carries XGBoost predictions (``jobs/phases.py`` writer), so
    a request for e.g. ``"ensemble"`` residuals may be answered with a
    substitute model's residuals — ``calibration_model`` names the model
    whose residuals were actually used so callers can label the interval
    honestly instead of implying it was calibrated on the displayed model
    (2026-07 critical-review finding P1-2). Residuals from the exact
    requested model always take precedence over substitutes.

    ``horizon_resolved=True`` (the #283 Phase 3b widening estimator) EXCLUDES
    the horizon-agnostic training-holdout source below. The flat estimator's
    trade-off — "the right model's residuals beat a substitute model's
    right-horizon residuals" — INVERTS for a lead-resolved band: pooling the
    same 168h holdout into every anchor collapses the per-horizon spread into
    a window-size artifact (a fake fan), so there the right horizon matters
    more than the right model, and a per-horizon substitute (disclosed via
    ``calibration_model``) is the honest source.
    """
    exact_chunks: list[np.ndarray] = []
    substitute_chunks: list[np.ndarray] = []
    substitute_model: str | None = None

    # In-memory cache (most recent in-process compute path) — exact matches only
    for (r, h, m, _mode), (cached_result, _hash, _time) in _BACKTEST_CACHE.items():
        if r != region or h != horizon_hours:
            continue
        if m != model_name and not (model_name == "ensemble" and m in ("ensemble",)):
            continue
        actual = np.asarray(cached_result.get("actual", []), dtype=float)
        pred = np.asarray(cached_result.get("predictions", []), dtype=float)
        n = min(len(actual), len(pred))
        if n > 0:
            exact_chunks.append(actual[:n] - pred[:n])

    # Redis pre-computed backtests (common production path)
    for key in (
        redis_key(f"backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon_hours}"),
        redis_key(f"backtest:{region}:{horizon_hours}"),
    ):
        cached = redis_get(key)
        if not isinstance(cached, dict):
            continue
        actual = np.asarray(cached.get("actual", []), dtype=float)
        preds_map = cached.get("predictions", {})
        if isinstance(preds_map, dict):
            if model_name in preds_map:
                pred = np.asarray(preds_map.get(model_name, []), dtype=float)
                chunk_model = model_name
            elif "ensemble" in preds_map:
                pred = np.asarray(preds_map.get("ensemble", []), dtype=float)
                chunk_model = "ensemble"
            elif preds_map:
                chunk_model = next(iter(preds_map.keys()))
                pred = np.asarray(preds_map[chunk_model], dtype=float)
            else:
                pred = np.array([], dtype=float)
                chunk_model = None
            n = min(len(actual), len(pred))
            if n > 0:
                if chunk_model == model_name:
                    exact_chunks.append(actual[:n] - pred[:n])
                else:
                    substitute_chunks.append(actual[:n] - pred[:n])
                    substitute_model = chunk_model

    # Per-model training-holdout residuals (``gridpulse:holdout:{region}``).
    # The training job persists every model's holdout forecast against the
    # SAME actuals, so ensemble/prophet/arima self-calibrate from their OWN
    # residuals instead of substituting XGBoost (#196). Treated as an
    # exact-model source for the FLAT estimator (horizon-agnostic: a single
    # pooled quantile is applied across the horizon regardless, so the right
    # model's residuals beat a substitute model's right-horizon residuals).
    # Skipped when ``horizon_resolved`` — see the docstring.
    if not horizon_resolved:
        holdout = redis_get(redis_key(f"holdout:{region}"))
        if isinstance(holdout, dict):
            h_actual = np.asarray(holdout.get("actual", []), dtype=float)
            h_preds = holdout.get("predictions", {})
            if isinstance(h_preds, dict) and model_name in h_preds:
                h_pred = np.asarray(h_preds.get(model_name, []), dtype=float)
                n = min(len(h_actual), len(h_pred))
                if n > 0:
                    exact_chunks.append(h_actual[:n] - h_pred[:n])

    if exact_chunks:
        chunks, calibration_model = exact_chunks, model_name
    elif substitute_chunks:
        chunks, calibration_model = substitute_chunks, substitute_model
    else:
        return np.array([], dtype=float), None
    residuals = np.concatenate(chunks)
    return residuals[np.isfinite(residuals)], calibration_model


# Anchor lead times for the lead-resolved (widening) interval — mirrors
# ``jobs.phases.BACKTEST_HORIZONS`` (deliberately not imported: the web tier
# doesn't import the jobs module). Each anchor's backtest residuals estimate
# "typical error of an H-hour-ahead forecast", pinned at lead H.
_INTERVAL_ANCHOR_HORIZONS = (24, 168, 720)


def _widening_interval_from_backtests(
    region: str,
    model_name: str,
    target_coverage: float = 0.80,
) -> dict:
    """Lead-time-resolved empirical interval anchors (#283 Phase 3b).

    The flat estimator below applies one q10/q90 pair — pooled across every
    lead time of a single backtest horizon — uniformly over the whole chart,
    so it reads too wide near the origin and too narrow at the deep tail. This
    estimator computes the same empirical quantiles **per backtest horizon**
    (24h / 168h / 720h) and returns them as anchors at those lead times, so
    the caller can interpolate a band that *widens with lead time* — the
    honest shape of forecast uncertainty.

    Two statistical choices (both surfaced by the Phase 3b verification):

    * **Horizon-resolved residuals only** (``horizon_resolved=True``): the
      horizon-agnostic 168h training-holdout pool is excluded — feeding the
      same pool into every anchor collapses the per-horizon spread into a
      window-size artifact (a fake fan). A per-horizon XGBoost substitute
      (disclosed) beats horizon-agnostic exact-model residuals here.
    * **Effective-lead pinning**: a horizon-H backtest pools residuals over
      leads 1..H, so its quantiles measure roughly the mid-window (~H/2)
      error, not the lead-H error. Each anchor is therefore pinned at
      ``effective_lead = H/2``. Leads beyond the deepest effective lead hold
      flat at the deepest measured value — a KNOWN-NARROW deep-tail bias
      (the true lead-720 error exceeds the lead-1..720 pooled quantile);
      preferred over pretending the pool measured lead H exactly.

    Returns ``{"available": False}`` unless ≥2 anchors have enough residuals
    (the caller then falls back to the flat estimator, then the heuristic
    envelope). ``calibration_model`` follows the same disclosure contract as
    the flat estimator: exact-model calibration is claimed only when EVERY
    anchor used the requested model's residuals; otherwise it names the
    substitute so the band label can disclose it.
    """
    from models.evaluation import empirical_error_quantiles

    alpha = (1.0 - target_coverage) / 2.0
    anchors: list[dict] = []
    calib_models: set[str] = set()
    for h in _INTERVAL_ANCHOR_HORIZONS:
        residuals, calib = _collect_backtest_residuals(region, model_name, h, horizon_resolved=True)
        if residuals.size < max(24, h // 2):
            continue
        tail_size = int(min(residuals.size, max(h * 5, 120)))
        q = empirical_error_quantiles(residuals[-tail_size:], lower_q=alpha, upper_q=1.0 - alpha)
        anchors.append(
            {
                "horizon": int(h),
                "effective_lead": max(1, h // 2),
                "lower_error": float(q["lower_error"]),
                "upper_error": float(q["upper_error"]),
                "sample_size": int(q["sample_size"]),
            }
        )
        if calib is not None:
            calib_models.add(calib)

    if len(anchors) < 2:
        return {"available": False}

    substitutes = sorted(m for m in calib_models if m != model_name)
    calibration_model = model_name if not substitutes else substitutes[0]
    return {
        "available": True,
        "anchors": sorted(anchors, key=lambda a: a["horizon"]),
        "target_coverage": float(target_coverage),
        "calibration_model": calibration_model,
    }


def _empirical_interval_from_backtests(
    region: str,
    model_name: str,
    horizon_hours: int,
    target_coverage: float = 0.80,
) -> dict[str, float | int | bool]:
    """Estimate empirical prediction interval from recent backtest residuals.

    ``calibration_model`` in the returned dict names the model whose
    residuals actually calibrated the interval — it may differ from
    ``model_name`` (see ``_collect_backtest_residuals``), and callers must
    disclose that in the band label.
    """
    from models.evaluation import empirical_error_quantiles

    residuals, calibration_model = _collect_backtest_residuals(region, model_name, horizon_hours)
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
        "calibration_model": calibration_model,
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


def _pipeline_alive(region: str, max_age_hours: float = 3.0) -> bool:
    """True when the scoring pipeline is demonstrably writing this region —
    a fresh ``actuals:{region}`` ``scored_at`` within ``max_age_hours``
    (the job runs hourly; 3h = several consecutive missed ticks).

    P2-35 (#273): used to distinguish a genuine cold/warming state (nothing
    written yet — "will appear shortly" is honest) from a persistently
    unavailable surface (the pipeline IS running but this region's
    forecast/alert payload never lands — "shortly" was a forever-lie).
    Fails closed: any read/parse problem returns False, keeping the softer
    warming copy.
    """
    import pandas as pd

    from data.redis_client import redis_get, redis_key

    try:
        payload = redis_get(redis_key(f"actuals:{region}"))
        if not isinstance(payload, dict):
            return False
        scored_at = pd.Timestamp(payload.get("scored_at"))
        if scored_at.tzinfo is None:
            scored_at = scored_at.tz_localize("UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - scored_at) / pd.Timedelta(hours=1)
        return 0 <= age_hours <= max_age_hours
    except Exception:
        return False


def _guard_max_ok(entry) -> int | None:
    """Parse a ``horizon_guard`` entry's ``max_ok_horizon``; None if malformed.

    Shared by the Forecast tab and the Overview hero (#296). Malformed
    entries fail OPEN (the caller renders normally): the series is real
    model output, and a corrupt guard entry — e.g. writer/reader version
    skew between the independently-deployed scoring job and web tier —
    must never take down a healthy forecast.
    """
    if not isinstance(entry, dict):
        return None
    try:
        return int(entry.get("max_ok_horizon") or 0)
    except (TypeError, ValueError):
        return None


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
    "PLOT_CONFIG",
    "_layout",
    "_empty_figure",
    # #296 horizon guard
    "_guard_max_ok",
    # P2-35 warming-vs-unavailable discriminator
    "_pipeline_alive",
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
    "_widening_interval_from_backtests",
    "_INTERVAL_ANCHOR_HORIZONS",
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
