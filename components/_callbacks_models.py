"""Models tab helpers extracted from ``components/callbacks.py``.

Step 3 of the ``callbacks.py`` decomposition tracked in issue #87. This
module owns the Models / Diagnostics tab's data-shaping helpers:

* ``_format_metric`` — "N/A"-aware formatter so missing scores never
  display as a misleadingly perfect ``0``.
* ``_models_tab_from_redis`` — Redis fast path that builds the entire
  Models tab (metrics table + 5 charts) from the scoring-job's cached
  diagnostics payload.
* ``_get_feature_importance`` — pulls real XGBoost importances from the
  model cache when available, otherwise returns a static didactic
  fallback so the chart still renders.

## Why these three live together

All three are exclusively Models-tab data path. ``_models_tab_from_redis``
is the producer/consumer split between the scoring job (writes
``wattcast:diagnostics:{region}``) and the web request path. The
metric formatter is its only caller plus ``register_callbacks``'s v1
fallback branch (which mirrors the same table layout when Redis is
cold). ``_get_feature_importance`` is the Models-tab v1 fallback's
SHAP-chart data source.

## Public-import surface

``components/callbacks.py`` re-imports each function by name (no star
imports — ruff-clean). Tests use ``from components.callbacks import
_models_tab_from_redis`` etc, and the re-export shim keeps those import
sites valid without any caller-side changes.

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_models._get_feature_importance")  # ✓
    @patch("components.callbacks._get_feature_importance")          # ✗

The standard "patch where it's used, not where it's defined" rule. The
``register_callbacks`` v1 fallback path reads ``_get_feature_importance``
from the ``components.callbacks`` module's own namespace (via the
re-import), so tests against that specific call site can still use
``components.callbacks._get_feature_importance``.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import Input, Output, State, html, no_update

from components._callbacks_shared import (
    _MODEL_CACHE,
    COLORS,
    _empty_figure,
    _layout,
)
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


def _format_metric(m: dict, key: str, fmt: str) -> str:
    """Format a metric value, returning 'N/A' when the key is missing or None.

    Prevents unavailable metrics from displaying as ``0`` which users can
    misread as a real (and suspiciously perfect) model score.
    """
    val = m.get(key)
    if val is None:
        return "N/A"
    return fmt.format(val)


def _models_tab_from_redis(region, selected_models: list[str] | None = None):
    """Redis fast path for update_models_tab callback.

    Returns a 6-tuple (table, fig_resid_time, fig_resid_hist, fig_resid_pred,
    fig_heatmap, fig_shap) or None if cache miss.
    """
    default_models = ["prophet", "arima", "xgboost", "ensemble"]
    selected_models = selected_models or default_models
    # Current Redis diagnostics payload is ensemble-only for residual charts.
    # Keep this fast path only when callback explicitly requests ensemble-only.
    if selected_models is not default_models and set(selected_models) != {"ensemble"}:
        return None

    cached = redis_get(redis_key(f"diagnostics:{region}"))
    if cached is None:
        return None

    # uirevision keyed on region + sorted model selection so zoom/legend
    # state survives Redis refresh but resets when the user picks new models.
    uirev = f"{region}:{','.join(sorted(selected_models))}"

    log.info("diagnostics_redis_hit", region=region)
    # Read metrics through the shared resolver so the table and the
    # leaderboard stay locked together (real holdout MAPE per model
    # from each pickle's meta.json; RMSE/MAE/R² from this Redis payload).
    from models.model_service import get_model_metrics

    metrics = get_model_metrics(region)
    timestamps = pd.to_datetime(cached.get("timestamps", []))
    ensemble = np.array(cached.get("ensemble", []))
    residuals = np.array(cached.get("residuals", []))
    hourly_err = cached.get("hourly_error", {})
    fi = cached.get("feature_importance", {})

    # Metrics table
    name_map = {
        "Prophet": "prophet",
        "SARIMAX": "arima",
        "XGBoost": "xgboost",
        "Ensemble": "ensemble",
    }
    rows = []
    for display_name, key in name_map.items():
        if key not in selected_models:
            continue
        m = metrics.get(key, {})
        rows.append(
            html.Tr(
                [
                    html.Td(display_name, style={"fontWeight": "600"}),
                    html.Td(_format_metric(m, "mape", "{:.2f}%")),
                    html.Td(_format_metric(m, "rmse", "{:.0f}")),
                    html.Td(_format_metric(m, "mae", "{:.0f}")),
                    html.Td(_format_metric(m, "r2", "{:.4f}")),
                ]
            )
        )
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE", "MAE", "R²"]])),
            html.Tbody(rows),
        ],
        className="metrics-table",
    )

    # Residuals span the full training window (60-90 days hourly =
    # 1440-2160 points). At 1280px chart width that's ~1.7 raw points
    # per pixel — well past the resolution where the eye can resolve
    # them. LTTB downsample to ~720 keeps the silhouette intact, drops
    # ~70% of the wire JSON, and saves Plotly a real chunk of layout
    # cost on the client.
    from data.preprocessing import lttb_downsample

    resid_x, resid_y = lttb_downsample(
        np.asarray(timestamps.values),
        np.asarray(residuals),
        threshold=720,
    )
    fig_resid_time = go.Figure(
        go.Scatter(
            x=resid_x,
            y=resid_y,
            mode="lines",
            line=dict(color=COLORS["arima"], width=1),
        )
    )
    fig_resid_time.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
    fig_resid_time.update_layout(**_layout(uirevision=uirev, yaxis_title="Residual (MW)"))

    fig_resid_hist = go.Figure(
        go.Histogram(x=residuals, nbinsx=50, marker_color=COLORS["ensemble"])
    )
    fig_resid_hist.update_layout(
        **_layout(uirevision=uirev, xaxis_title="Residual (MW)", yaxis_title="Count")
    )

    fig_resid_pred = go.Figure(
        go.Scatter(
            x=ensemble,
            y=residuals,
            mode="markers",
            marker=dict(size=2, color=COLORS["xgboost"], opacity=0.3),
        )
    )
    fig_resid_pred.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
    fig_resid_pred.update_layout(
        **_layout(
            uirevision=uirev,
            xaxis_title="Predicted (MW)",
            yaxis_title="Residual (MW)",
        )
    )

    h_hours = hourly_err.get("hours", list(range(24)))
    h_vals = hourly_err.get("values", [0] * 24)
    h_median = float(np.median(h_vals)) if h_vals else 0
    fig_heatmap = go.Figure(
        go.Bar(
            x=h_hours,
            y=h_vals,
            marker_color=[COLORS["ensemble"] if e > h_median else COLORS["actual"] for e in h_vals],
        )
    )
    fig_heatmap.update_layout(
        **_layout(
            uirevision=uirev,
            xaxis_title="Hour of Day",
            yaxis_title="Mean |Error| (MW)",
        )
    )

    if "xgboost" in selected_models:
        fi_names = fi.get("names", [])
        fi_vals = fi.get("values", [])
        fig_shap = go.Figure(
            go.Bar(
                x=fi_vals[::-1],
                y=fi_names[::-1],
                orientation="h",
                marker_color=COLORS["xgboost"],
            )
        )
        fig_shap.update_layout(**_layout(uirevision=uirev, xaxis_title="Feature Importance"))
    else:
        fig_shap = _empty_figure("SHAP is available only for XGBoost. Select XGBoost above.")

    return table, fig_resid_time, fig_resid_hist, fig_resid_pred, fig_heatmap, fig_shap


def _get_feature_importance(region: str, top_n: int = 10) -> tuple[list[str], np.ndarray]:
    """Extract real feature importances from cached XGBoost model, or return defaults."""
    if (region, "xgboost", 0) in _MODEL_CACHE:
        model_dict, _, _ = _MODEL_CACHE[(region, "xgboost", 0)]
        if isinstance(model_dict, dict) and "feature_importances" in model_dict:
            imp = model_dict["feature_importances"]
            sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
            names = [f[0] for f in sorted_feats]
            vals = np.array([f[1] for f in sorted_feats])
            if vals.sum() > 0:
                return names, vals

    names = [
        "temperature_2m",
        "demand_lag_24h",
        "hour_sin",
        "cooling_degree_days",
        "wind_speed_80m",
        "demand_roll_24h_mean",
        "heating_degree_days",
        "solar_capacity_factor",
        "relative_humidity_2m",
        "is_holiday",
    ]
    vals = np.array([0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04])
    return names, vals


# ── Callback registration (Step 10c — register_callbacks split) ──────


def register_models_callbacks(app):
    """Register Models tab callbacks with the Dash app.

    Step 10c of the ``register_callbacks`` decomposition. Owns three
    callbacks driving the Models / Diagnostics tab:

    * ``update_models_title`` — page title block
    * ``update_models_leaderboard`` — 5-up model-comparison MetricsBar
    * ``update_models_tab`` — metrics table + 5 residual / SHAP figures,
      with Redis fast path (``_models_tab_from_redis``) + v1 compute
      fallback.

    The compute fallback uses ``COLORS`` and ``_layout`` from shared
    plus the legacy v1 in-line residual figure builders. Once every
    region is reliably populated by the scoring job, the fallback could
    be deleted entirely — see the per-model coverage check in
    ``_models_tab_from_redis``.
    """
    from components._callbacks_overview import _build_models_leaderboard
    from components.cards import build_page_title
    from config import REGION_NAMES

    @app.callback(
        Output("models-title", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_models_title(region, active_tab):
        """Page title for the Models tab."""
        if active_tab != "tab-models":
            return no_update

        region = region or "FPL"
        region_name = REGION_NAMES.get(region, region)
        return build_page_title(
            "Models",
            f"Forecast accuracy, residuals, and feature importance · {region_name}",
        )

    @app.callback(
        Output("models-leaderboard", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_models_leaderboard(region, active_tab):
        """5-up MetricsBar leaderboard — one column per model with MAPE."""
        if active_tab != "tab-models":
            return no_update
        return _build_models_leaderboard(region)

    @app.callback(
        [
            Output("tab3-metrics-table", "children"),
            Output("tab3-residuals-time", "figure"),
            Output("tab3-residuals-hist", "figure"),
            Output("tab3-residuals-pred", "figure"),
            Output("tab3-error-heatmap", "figure"),
            Output("tab3-shap", "figure"),
        ],
        [
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
            Input("tab3-model-selector", "value"),
        ],
        [State("region-selector", "value")],
        prevent_initial_call=True,
    )
    def update_models_tab(demand_json, active_tab, selected_models, region):
        """Update Models tab diagnostics using model service."""
        if active_tab != "tab-models":
            return [no_update] * 6
        if not demand_json:
            empty = _empty_figure("Loading...")
            return html.P("Loading..."), empty, empty, empty, empty, empty
        if not selected_models:
            empty = _empty_figure("Select at least one model to view diagnostics.")
            return html.P("No model selected."), empty, empty, empty, empty, empty

        # Redis fast path is valid only for ensemble-only diagnostics payloads.
        if region:
            redis_result = _models_tab_from_redis(region, selected_models)
            if redis_result is not None:
                return redis_result

        # ── v1 compute fallback ─────────────────────────────
        demand_df = pd.read_json(io.StringIO(demand_json))
        demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
        actual = demand_df["demand_mw"].values

        from models.model_service import get_forecasts, get_model_metrics

        forecasts = get_forecasts(region, demand_df, selected_models)
        # Read metrics from the same Redis-first source the leaderboard
        # uses (``get_model_metrics``) so the table and the leaderboard
        # always show consistent numbers — even if ``get_forecasts``
        # fell back to fresh simulated forecasts whose computed metrics
        # would otherwise diverge from the Redis payload's metrics.
        metrics = get_model_metrics(region) or forecasts.get("metrics", {})
        model_order = ["prophet", "arima", "xgboost", "ensemble"]
        selected = [m for m in model_order if m in set(selected_models)]

        # Metrics table
        name_map = {
            "Prophet": "prophet",
            "SARIMAX": "arima",
            "XGBoost": "xgboost",
            "Ensemble": "ensemble",
        }
        rows = []
        for display_name, key in name_map.items():
            if key not in selected:
                continue
            m = metrics.get(key, {})
            rows.append(
                html.Tr(
                    [
                        html.Td(display_name, style={"fontWeight": "600"}),
                        html.Td(_format_metric(m, "mape", "{:.2f}%")),
                        html.Td(_format_metric(m, "rmse", "{:.0f}")),
                        html.Td(_format_metric(m, "mae", "{:.0f}")),
                        html.Td(_format_metric(m, "r2", "{:.4f}")),
                    ]
                )
            )
        table = html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE", "MAE", "R²"]])),
                html.Tbody(rows),
            ],
            className="metrics-table",
        )

        timestamps = demand_df["timestamp"]
        model_labels = {
            "prophet": "Prophet",
            "arima": "SARIMAX",
            "xgboost": "XGBoost",
            "ensemble": "Ensemble",
        }
        model_colors = {
            "prophet": COLORS["prophet"],
            "arima": COLORS["arima"],
            "xgboost": COLORS["xgboost"],
            "ensemble": COLORS["ensemble"],
        }
        model_residuals: dict[str, np.ndarray] = {}
        model_predictions: dict[str, np.ndarray] = {}
        for model_key in selected:
            pred = forecasts.get(model_key)
            if isinstance(pred, np.ndarray) and len(pred) == len(actual):
                model_predictions[model_key] = pred
                model_residuals[model_key] = actual - pred

        if not model_residuals:
            empty = _empty_figure("No residual diagnostics available for the selected model(s).")
            return (
                table,
                empty,
                empty,
                empty,
                empty,
                _empty_figure("Select XGBoost to view SHAP feature importance."),
            )

        # uirevision keyed on region + sorted model selection so zoom/legend
        # state survives data refresh but resets when the user picks new models.
        uirev = f"{region}:{','.join(sorted(selected))}"

        fig_resid_time = go.Figure()
        for model_key, residuals in model_residuals.items():
            fig_resid_time.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=residuals,
                    mode="lines",
                    name=model_labels.get(model_key, model_key.title()),
                    line=dict(color=model_colors.get(model_key, COLORS["actual"]), width=1),
                )
            )
        fig_resid_time.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
        fig_resid_time.update_layout(**_layout(uirevision=uirev, yaxis_title="Residual (MW)"))

        fig_resid_hist = go.Figure()
        for model_key, residuals in model_residuals.items():
            fig_resid_hist.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=50,
                    name=model_labels.get(model_key, model_key.title()),
                    marker_color=model_colors.get(model_key, COLORS["actual"]),
                    opacity=0.6,
                )
            )
        fig_resid_hist.update_layout(
            **_layout(
                uirevision=uirev,
                barmode="overlay",
                xaxis_title="Residual (MW)",
                yaxis_title="Count",
            )
        )

        fig_resid_pred = go.Figure()
        for model_key, residuals in model_residuals.items():
            preds = model_predictions[model_key]
            fig_resid_pred.add_trace(
                go.Scatter(
                    x=preds,
                    y=residuals,
                    mode="markers",
                    name=model_labels.get(model_key, model_key.title()),
                    marker=dict(
                        size=3,
                        color=model_colors.get(model_key, COLORS["actual"]),
                        opacity=0.35,
                    ),
                )
            )
        fig_resid_pred.add_hline(y=0, line=dict(color="#F7FAFC", dash="dash", width=0.5))
        fig_resid_pred.update_layout(
            **_layout(uirevision=uirev, xaxis_title="Predicted (MW)", yaxis_title="Residual (MW)")
        )

        hours_of_day = timestamps.dt.hour
        fig_heatmap = go.Figure()
        for model_key, residuals in model_residuals.items():
            error_by_hour = pd.DataFrame({"hour": hours_of_day, "abs_error": np.abs(residuals)})
            hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()
            fig_heatmap.add_trace(
                go.Bar(
                    x=hourly_error.index,
                    y=hourly_error.values,
                    name=model_labels.get(model_key, model_key.title()),
                    marker_color=model_colors.get(model_key, COLORS["actual"]),
                    opacity=0.85,
                )
            )
        fig_heatmap.update_layout(**_layout(uirevision=uirev, barmode="group"))
        fig_heatmap.update_layout(
            **_layout(uirevision=uirev, xaxis_title="Hour of Day", yaxis_title="Mean |Error| (MW)")
        )

        if "xgboost" in selected:
            feature_names, importance_vals = _get_feature_importance(region)
            fig_shap = go.Figure(
                go.Bar(
                    x=importance_vals[::-1],
                    y=feature_names[::-1],
                    orientation="h",
                    marker_color=COLORS["xgboost"],
                )
            )
            fig_shap.update_layout(**_layout(uirevision=uirev, xaxis_title="Feature Importance"))
        else:
            fig_shap = _empty_figure("SHAP is available only for XGBoost. Select XGBoost above.")

        return table, fig_resid_time, fig_resid_hist, fig_resid_pred, fig_heatmap, fig_shap


__all__ = [
    "_format_metric",
    "_models_tab_from_redis",
    "_get_feature_importance",
    "register_models_callbacks",
]
