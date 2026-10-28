"""Models tab helpers extracted from ``components/callbacks.py``.

Step 3 of the ``callbacks.py`` decomposition tracked in issue #87. This
module owns the Models / Diagnostics tab's data-shaping helpers:

* ``_format_metric`` — "N/A"-aware formatter so missing scores never
  display as a misleadingly perfect ``0``.
* ``_models_tab_from_redis`` — Redis fast path that builds the entire
  Models tab (metrics table + 5 charts) from the scoring-job's cached
  diagnostics payload.
* ``_get_feature_importance`` — pulls real XGBoost importances from the
  model cache when available, otherwise returns no values.

## Why these three live together

All three are exclusively Models-tab data path. ``_models_tab_from_redis``
is the producer/consumer split between the scoring job (writes
``gridpulse:diagnostics:{region}``) and the web request path. The
metric formatter is its only caller plus ``register_callbacks``'s v1
fallback branch (which mirrors the same table layout when Redis is
cold). ``_get_feature_importance`` is the Models-tab v1 fallback's
feature-importance data source.

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

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import Input, Output, State, ctx, html, no_update

from components._callbacks_shared import (
    _MODEL_CACHE,
    COLORS,
    _empty_figure,
    _layout,
    _read_vintage_summary,
)
from components.accessibility import model_display_name
from components.cards import build_metrics_bar
from config import (
    ANCHOR_FED_MODELS,
    FEED_LIMITED_CLASSES,
    FEED_LIMITED_MIN_REVISION_PCT,
    MAPE_BY_HORIZON,
    mape_grade,
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


def _build_shap_fig(fi: dict, selected_models: list[str], uirev: str) -> go.Figure:
    """Build the model’s built-in feature-importance figure.

    Empty state (not a blank/fabricated bar chart) when XGBoost isn't selected
    or when no real importances exist (absent trained model) — the writer sends
    ``feature_importance=None`` in that case rather than a placeholder.
    """
    if "xgboost" not in selected_models:
        return _empty_figure(
            "Feature importance is available only for XGBoost. Select XGBoost above."
        )
    fi_names = fi.get("names", []) if fi else []
    fi_vals = fi.get("values", []) if fi else []
    if not fi_names:
        return _empty_figure("XGBoost feature importance unavailable for this region.")
    fig = go.Figure(
        go.Bar(
            x=fi_vals[::-1],
            y=fi_names[::-1],
            orientation="h",
            marker_color=COLORS["xgboost"],
        )
    )
    fig.update_layout(**_layout(uirevision=uirev, xaxis_title="Feature Importance"))
    return fig


#: Human-readable exog-mode labels for the residual-panel provenance caption
#: (#220). ``forecast_exog`` = climatology/naive exogenous inputs — the
#: deliberately degraded no-weather-info scenario; ``oracle_exog`` = actual
#: observed weather. Unknown modes fall through and display raw.
_EXOG_MODE_CAPTIONS = {"forecast_exog": "climatology-exog", "oracle_exog": "oracle-exog"}


def _models_tab_from_redis(region, selected_models: list[str] | None = None):
    """Redis fast path for update_models_tab callback.

    Returns a 6-tuple (table, fig_resid_time, fig_resid_hist, fig_resid_pred,
    fig_heatmap, fig_shap) or None if cache miss.
    """
    default_models = ["prophet", "arima", "xgboost", "ensemble"]
    selected_models = selected_models or default_models
    # The Redis diagnostics payload carries ONE residual series (the
    # walk-forward-backtest series named in ``residual_source`` — XGBoost in
    # production, disclosed on the charts), so this fast path can satisfy
    # exactly two selections: the default full model set (the Models tab's
    # initial state) and an ensemble-only selection. Any narrower per-model
    # subset needs per-model residuals this payload lacks, so it falls
    # through to compute.
    #
    # Compare by VALUE, not identity. The callback passes the checklist's
    # value — a distinct list object that *equals* ``default_models`` but is
    # not the same object — so the old ``is not default_models`` identity
    # check never matched the default view. That silently dropped the default
    # Models tab to the v1 compute path, which on the stateless web tier is
    # strict-gated to "unavailable" (#149) and produces no per-model
    # residuals → the "No residual diagnostics available" placeholder instead
    # of the real ensemble charts that were sitting in Redis the whole time.
    if set(selected_models) not in ({"ensemble"}, set(default_models)):
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
    # "predicted" is the canonical series name (#220 — the diagnostics payload
    # now carries walk-forward-backtest predictions, model named in
    # residual_source); "ensemble" is the legacy pre-#220 field kept as a
    # fallback for payloads still in Redis within their TTL.
    predicted = np.array(cached.get("predicted", cached.get("ensemble", [])))
    residuals = np.array(cached.get("residuals", []))
    hourly_err = cached.get("hourly_error", {})
    fi = cached.get("feature_importance") or {}
    diagnostics_source = cached.get("diagnostics_source")
    residual_source = cached.get("residual_source") or {}

    # Metrics table
    name_map = {model_display_name(k): k for k in ("prophet", "arima", "xgboost", "ensemble")}
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
            html.Thead(
                html.Tr([html.Th(h) for h in ["Model", "MAPE", "RMSE (MW)", "MAE (MW)", "R²"]])
            ),
            html.Tbody(rows),
        ],
        className="metrics-table",
    )

    # Honest empty state when no backtest results exist yet for this region
    # (fresh deploy before the first nightly training run). Do NOT render the
    # four residual charts from fabricated zero residuals (#166 / 2026-07
    # review P2-32). The metrics table above is real (meta.json holdout) and
    # Feature importance is real (trained model) — keep those; blank only the residuals.
    # The copy states the TRUE self-heal condition (#220): these populate from
    # the training job's backtests, not from scoring ticks — the old message
    # promised a fill that scoring could never deliver.
    # The length guard is belt-and-braces against a malformed/foreign payload
    # (the writer's horizon gate guarantees alignment): mismatched
    # timestamps/residuals would crash the LTTB downsample below and error the
    # whole Models-tab callback — honest degradation beats a dead tab.
    if (
        diagnostics_source == "unavailable"
        or residuals.size == 0
        or len(timestamps) != residuals.size
    ):
        log.info("diagnostics_unavailable_render", region=region, source=diagnostics_source)
        unavail = _empty_figure(
            "Residual diagnostics unavailable for this region. "
            "Published backtest predictions and matching timestamps are required."
        )
        return (
            table,
            unavail,
            unavail,
            unavail,
            unavail,
            _build_shap_fig(fi, selected_models, uirev),
        )

    # Provenance caption for the four residual panels (#220): these residuals
    # are HOLDOUT walk-forward-backtest errors (horizon + model from the
    # payload), not live-forecast errors — say so on the charts themselves.
    # The exog MODE is disclosed too: the production backtest runs with
    # climatology/naive exogenous inputs (``forecast_exog``), deliberately
    # measuring the degraded no-weather-info scenario — its residuals read
    # WORSE than the real-weather holdout MAPE in the metrics table above,
    # and without the label that contrast looks like a contradiction.
    if residual_source.get("kind") == "walk_forward_backtest":
        exog_mode = residual_source.get("exog_mode")
        exog_note = f" ({_EXOG_MODE_CAPTIONS.get(exog_mode, exog_mode)})" if exog_mode else ""
        provenance = (
            f"{residual_source.get('horizon', '?')}h walk-forward backtest residuals"
            f"{exog_note} · {str(residual_source.get('model', '')).upper()}"
        )
    else:
        provenance = None

    def _add_provenance(fig):
        if provenance:
            fig.add_annotation(
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                text=provenance,
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="rgba(160,180,200,0.85)"),
            )
        return fig

    # Residuals span the walk-forward backtest window (#220): ~120 points at
    # the preferred 24h horizon (LTTB no-ops at ≤720), up to ~840/~3600 on the
    # 168h/720h fallbacks — where downsampling to ~720 keeps the silhouette
    # intact, trims the wire JSON, and saves Plotly client layout cost.
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
            x=predicted,
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

    fig_shap = _build_shap_fig(fi, selected_models, uirev)

    return (
        table,
        _add_provenance(fig_resid_time),
        _add_provenance(fig_resid_hist),
        _add_provenance(fig_resid_pred),
        _add_provenance(fig_heatmap),
        fig_shap,
    )


#: Grades the verdict-confirmation rule recognizes. An unknown grade string
#: (e.g. from a writer with a different vocabulary) maps to *pending*, never
#: to health — matching the horizon panel's neutral tone for unknown grades.
_KNOWN_GRADES = ("excellent", "target", "acceptable", "rollback")


def _day_ahead_grade(horizon_payload: dict | None, model_key: str) -> str | None:
    """The model's resolved 24h-horizon grade from ``drift_horizon``, or ``None``.

    ``None`` when the payload / model / horizon block is missing or malformed,
    the MAPE is unresolved, fewer than 6 matured records exist (the same
    meaningfulness threshold ``_build_horizon_drift_panel`` applies), or the
    grade string isn't one of the known governance grades.
    """
    if not isinstance(horizon_payload, dict):
        return None
    models = horizon_payload.get("models")
    if not isinstance(models, dict):
        return None
    model_block = models.get(model_key)
    if not isinstance(model_block, dict):
        return None
    block = model_block.get("24h")
    if not isinstance(block, dict):
        return None
    # P2-21 (#273): gate the 7d figure on its own in-window sample count
    # (n_7d) — n_records is total history. Pre-#273 payloads lack n_7d;
    # unknown window counts cannot confirm an assessment.
    n_window = block.get("n_7d")
    n_eff = int(n_window) if n_window is not None else 0
    if block.get("rolling_mape_7d") is None or n_eff < 6:
        return None
    grade = block.get("grade")
    return grade if grade in _KNOWN_GRADES else None


def _error_band_label(grade: str | None, horizon: str) -> str:
    """Translate governance grades into the numeric band shown to readers."""
    bands = MAPE_BY_HORIZON.get(horizon, {})
    if grade in ("excellent", "target", "acceptable") and grade in bands:
        return f"≤{bands[grade]:g}% band"
    if grade == "rollback" and "acceptable" in bands:
        return f">{bands['acceptable']:g}% band"
    return "Unassessed"


def _build_drift_panel(region: str | None, metrics: dict | None = None) -> html.Div:
    """Show live 1h MAPE, with training holdout scores as separate context.

    Gate the seven-day average on 24 in-window observations. Above-band
    1h errors use the model’s 24h grade for confirmation; an absent horizon
    feed remains a visible warning. High-revision input conditions are
    disclosed without claiming they explain all of a model’s error.
    """
    region = region or "FPL"
    drift_payload = redis_get(redis_key(f"drift:{region}"))

    # Pull holdout MAPE from the same resolver the leaderboard uses,
    # so the "holdout" column here matches what the leaderboard shows.
    if metrics is None:
        from models.model_service import get_model_metrics

        metrics = get_model_metrics(region) or {}

    if not isinstance(drift_payload, dict) or not drift_payload.get("models"):
        return html.Div(
            [
                html.Div("Live error", className="gp-control-eyebrow"),
                html.Div(
                    (
                        "Live error unavailable. The seven-day average requires at least "
                        "24 scored observations in that window."
                    ),
                    className="gp-panel__placeholder",
                ),
            ],
            className="gp-models-drift-panel",
            id="drift-panel-warming",
        )

    drift_models = drift_payload["models"]

    # 24h-horizon grades for the verdict-confirmation rule (#217). Read after
    # the warming early-return so a cold region costs one Redis read, not two.
    # ``horizon_reporting`` distinguishes "feed alive, grades still maturing"
    # (pending, benign) from "feed absent" (broken >24h — fail closed).
    horizon_payload = redis_get(redis_key(f"drift_horizon:{region}"))
    horizon_reporting = isinstance(horizon_payload, dict) and bool(horizon_payload.get("models"))

    # Identify measured high-revision input conditions for anchor-fed models.
    # This association alone does not establish the cause of forecast error.
    vintage_summary = _read_vintage_summary(region)
    _cls = (vintage_summary or {}).get("revision_class")
    _rev = (vintage_summary or {}).get("mean_fresh_revision_pct")
    feed_limited_region = (
        _cls in FEED_LIMITED_CLASSES
        and isinstance(_rev, (int, float))
        and float(_rev) >= FEED_LIMITED_MIN_REVISION_PCT
    )

    # Render in the same order as the leaderboard so the eye can
    # diff across panels without re-anchoring.
    display_order = ("prophet", "arima", "xgboost", "ensemble")
    display_names = {k: model_display_name(k) for k in display_order}

    rows: list[html.Tr] = []
    degraded_names: list[str] = []  # off band at BOTH 1h and 24h — confirmed
    feed_limited_names: list[str] = []  # confirmed rollback, attributed to the feed
    off_band_healthy: list[str] = []  # above 1h band, healthy on matured 24h scores
    off_band_pending: list[str] = []  # above 1h band, 24h grade maturing (feed alive)
    off_band_unverified: list[str] = []  # above 1h band, horizon feed absent — fail closed
    unassessed_names: list[str] = []
    within_band_names: list[str] = []
    for key in display_order:
        if key not in drift_models:
            unassessed_names.append(display_names[key])
            continue
        drift = drift_models[key]
        # P2-21 (#273): the displayed stat is the 7d rolling figure — gate it
        # on the 7d in-window post-filter count, not total history. Pre-#273
        # payloads lacking n_7d cannot establish in-window sample size.
        n_window = drift.get("n_7d")
        n_eff = int(n_window) if n_window is not None else 0
        live_7d = drift.get("rolling_mape_7d")
        live_30d = drift.get("rolling_mape_30d")
        holdout = metrics.get(key, {}).get("mape")
        display = display_names.get(key, model_display_name(key))
        chip_title: str | None = None

        # Warming for this specific model when its window hasn't filled
        # enough to be statistically meaningful. Other models in the
        # same panel may have full records — surface mixed state.
        if n_eff < 24 or live_7d is None:
            status_label = "Not assessed"
            unassessed_names.append(display)
            status_tone = "secondary"
            # Don't print rolling means the chip just declared statistically
            # meaningless — a thin-window 180% next to "Warming" still reads
            # as a measurement (P2-21 verification catch).
            live_7d = None
            live_30d = None
        elif holdout is None or holdout <= 0:
            # Drift exists but the model's holdout MAPE is missing
            # (e.g. training-time meta lost). Show live MAPE without
            # the holdout context — better than dropping the row.
            status_label = "Live only"
            unassessed_names.append(display)
            status_tone = "secondary"
        else:
            # Grade derived from the live MAPE's own governance grade at the
            # 1-hour-ahead horizon (the drift metric IS first-hour) — never
            # from any cross-horizon holdout comparison (#273).
            grade = mape_grade(float(live_7d), horizon="1h")
            if grade == "excellent" or grade == "target":
                status_label = _error_band_label(grade, "1h")
                within_band_names.append(display)
                status_tone = "positive"
            elif grade == "acceptable":
                status_label = _error_band_label(grade, "1h")
                within_band_names.append(display)
                status_tone = "warning"
            else:
                # Above the 5% 1h band. Whether that is a *verdict* depends on
                # the model's own operating horizon (#217): a day-ahead model
                # with no 1-hour anchor is expected to sit above the 1h band.
                # Only confirm "Rollback" when the model is also off band at
                # 24h-horizon (its horizon-matched grade, #227).
                day_grade = _day_ahead_grade(horizon_payload, key)
                if day_grade == "rollback" and feed_limited_region and key in ANCHOR_FED_MODELS:
                    # Keep both above-band measurements visible alongside
                    # the measured input-revision condition.
                    status_label = "High input revisions"
                    status_tone = "warning"
                    chip_title = (
                        f"Above the configured 1h and 24h error bands. This model uses "
                        f"recent demand readings; the region’s reported demand revises "
                        f"by {float(_rev):.0f}% on average ({_cls})."
                    )
                    feed_limited_names.append(display)
                elif day_grade == "rollback":
                    status_label = "Above both bands"
                    status_tone = "negative"
                    chip_title = (
                        "Above the configured error bands at both first-hour and 24h-horizon. "
                        "This status does not indicate that a rollback has occurred."
                    )
                    degraded_names.append(display)
                elif day_grade is None and not horizon_reporting:
                    # FAIL CLOSED: the horizon feed writes hourly (even with
                    # zero resolved records) and its key carries a 24h TTL —
                    # a missing payload means the feed has been broken for a
                    # day+. Absence of the confirming signal must not read as
                    # health, or a dead pipeline silently disables the panel's
                    # only alarm path.
                    status_label = "Off band"
                    status_tone = "negative"
                    chip_title = (
                        "Above the 5% error band at first forecast hour. The 24h error feed is unavailable, "
                        "so the longer-lead comparison cannot be assessed."
                    )
                    off_band_unverified.append(display)
                elif day_grade is None:
                    status_label = "Off band"
                    status_tone = "warning"
                    chip_title = (
                        "Above the 5% 1h nowcast band; the 24h-horizon grade is "
                        "still resolving — see live error by forecast horizon below."
                    )
                    off_band_pending.append(display)
                else:
                    status_label = "Off band"
                    status_tone = "warning"
                    chip_title = (
                        f"Above the 5% 1h nowcast band, but graded "
                        f"{day_grade} on matured 24h-horizon predictions (24h "
                        f"grades include reporting delay) — models without a "
                        f"1-hour anchor are expected to sit above the 1h band. "
                        f"See live error by forecast horizon below."
                    )
                    off_band_healthy.append(display)

        chip_kwargs = {"title": chip_title} if chip_title else {}
        rows.append(
            html.Tr(
                [
                    html.Td(display, style={"fontWeight": "600"}),
                    html.Td(
                        f"{float(holdout):.2f}%" if holdout is not None else "—",
                        className="gp-drift-cell--holdout",
                    ),
                    html.Td(
                        f"{float(live_7d):.2f}%" if live_7d is not None else "—",
                        className="gp-drift-cell--live7d",
                    ),
                    html.Td(
                        f"{float(live_30d):.2f}%" if live_30d is not None else "—",
                        className="gp-drift-cell--live30d",
                    ),
                    html.Td(
                        html.Span(
                            status_label,
                            className=f"gp-status-chip gp-status-chip--{status_tone}",
                            **chip_kwargs,
                        ),
                        className="gp-drift-cell--status",
                    ),
                    # The gate's own denominator (7d in-window post-filter
                    # count when the payload carries it; legacy total
                    # otherwise) — showing total history next to a 7d stat
                    # overstated the sample (P2-21).
                    html.Td(str(n_eff), className="gp-drift-cell--n"),
                    html.Td(
                        str(n_30d)
                        if (n_30d := _count(drift.get("n_30d"))) is not None
                        else "Unavailable",
                        className="gp-drift-cell--n30d",
                    ),
                ]
            )
        )

    if not rows:
        return html.Div(
            [
                html.Div("Live error", className="gp-control-eyebrow"),
                html.Div(
                    "No live error records are available for this region.",
                    className="gp-panel__placeholder",
                ),
            ],
            className="gp-models-drift-panel",
        )

    # Headline severity follows the verdict-confirmation rule: alarm on a
    # CONFIRMED rollback (off band at both 1h and 24h) or on an off-band model
    # whose confirming feed is absent (fail closed); 1h-only excursions with a
    # live feed get neutral, explanatory copy instead of crying wolf (#217).
    feed_limited_clause = (
        f"{', '.join(feed_limited_names)} above both bands; high input revisions "
        f"(reported demand revises ~{float(_rev):.0f}% on average)"
        if feed_limited_names
        else ""
    )
    parts = []
    if degraded_names:
        parts.append(
            f"{', '.join(degraded_names)} off band at both first-hour and 24h-horizon — investigate"
        )
    if off_band_unverified:
        parts.append(
            f"{', '.join(off_band_unverified)} above the 1h band; horizon feed not reporting — investigate"
        )
    if feed_limited_clause:
        parts.append(feed_limited_clause)
    if off_band_healthy:
        parts.append(
            f"{', '.join(off_band_healthy)} above the 1h band, within band on matured 24h-horizon scores (includes reporting delay)"
        )
    if off_band_pending:
        parts.append(f"{', '.join(off_band_pending)} above the 1h band; 24h grade still resolving")
    if within_band_names:
        parts.append(
            f"{', '.join(within_band_names)} within the configured 1h band (MAPE ≤ {MAPE_BY_HORIZON['1h']['acceptable']:g}%)"
        )
    if unassessed_names:
        parts.append(
            f"{', '.join(unassessed_names)} not assessed: insufficient live records or missing holdout context"
        )
    headline_note = "; ".join(parts) + "."

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(h)
                        for h in [
                            "Model",
                            "Holdout (168h recursive)",
                            "Live first-hour (7d avg)",
                            "Live first-hour (30d avg)",
                            "Status",
                            "n (7d)",
                            "n (30d)",
                        ]
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        className="metrics-table gp-drift-table",
    )

    return html.Div(
        [
            html.Div(
                "Live error — first forecast hour · 7d / 30d averages",
                className="gp-control-eyebrow",
            ),
            html.Div(headline_note, className="gp-drift-headline"),
            html.Div(
                table,
                className="gp-drift-table-scroll",
                role="region",
                tabIndex=0,
                **{"aria-label": "Forecast error comparison table; scroll horizontally if needed"},
            ),
            html.Div(
                (
                    "Live columns show first-hour MAPE over the trailing 7 and 30 days, "
                    "recalculated as EIA revises reported demand. Holdout is the last "
                    "training run’s 168-hour recursive evaluation. These measure "
                    "different forecast horizons. Status bands apply to the seven-day live average. Horizon offsets are measured from the first forecast target, not the issue clock. Cross-horizon status uses the legacy offset diagnostic, separate from the new hourly samples."
                ),
                className="gp-panel__caption",
            ),
        ],
        className="gp-models-drift-panel",
        id="drift-panel",
    )


def _build_horizon_drift_panel(region: str | None, *, include_legacy: bool = True) -> html.Div:
    """Show sampled-lead evidence and aggregate categories, keeping legacy offsets advanced."""
    from data.forecast_evidence import utc_timestamp

    sampled = redis_get(redis_key(f"drift_hourly:{region or 'FPL'}:summary"))
    legacy = (
        html.Details(
            [
                html.Summary("Advanced: legacy forecast-start offsets · separate sample"),
                _build_legacy_horizon_drift_panel(region),
            ],
            className="gp-models-rail-disclosure gp-models-legacy-disclosure",
        )
        if include_legacy
        else None
    )
    if (
        not isinstance(sampled, dict)
        or sampled.get("schema_version") != 1
        or sampled.get("sampling_basis") != "hourly_observation_before_target"
    ):
        return html.Div(
            [
                html.P(
                    "Hourly-sampled error is unavailable for this region. Legacy offsets below "
                    "do not substitute for this sampling clock.",
                    className="gp-panel__placeholder",
                ),
                legacy,
            ],
            className="gp-models-drift-panel",
            id="hourly-horizon-drift-panel",
        )
    models = sampled.get("models") or {}
    models = models if isinstance(models, dict) else {}
    categories = (
        ("not_yet_tracked_hours", "Not yet tracked"),
        ("missing_sample_hours", "Missing samples"),
        ("series_unavailable_hours", "Series unavailable"),
        ("pending_actual_hours", "Pending actuals / quality resolution"),
        ("quality_guard_excluded_hours", "Quality-guard excluded actuals"),
        ("low_actual_excluded_hours", "Low-actual exclusions"),
    )
    rows = []
    details = []
    for model in (*_MODELS, "baseline"):
        if model not in models:
            continue
        cells = [html.Td("Seasonal baseline" if model == "baseline" else model_display_name(model))]
        for horizon in ("24h", "48h", "72h"):
            block = (models.get(model) or {}).get(horizon) or {}
            coverage = block.get("coverage_7d") or {}
            coverage = coverage if isinstance(coverage, dict) else {}
            count, score = _count(block.get("n_7d")), _number(block.get("rolling_mape_7d"))
            possible, scored = (
                _number(coverage.get(k)) for k in ("possible_target_hours", "scored_target_hours")
            )
            score_text = (
                f"{score:.2f}%"
                if (score is not None and score >= 0 and count is not None and count >= 6)
                else ("Collecting" if block else "Unavailable")
            )
            future = _count(block.get("future_sample_hours"))
            n_text = "Unavailable" if count is None else str(int(count))
            future_text = str(int(future)) if future is not None else "Unavailable"
            children = [html.Span(score_text, className="gp-drift-horizon-mape")]
            if (
                possible is not None
                and possible > 0
                and scored is not None
                and 0 <= scored <= possible
            ):
                children.extend(
                    [
                        html.Span(
                            f"{int(scored)} / {int(possible)} scored · future samples: {future_text}",
                            className="gp-models-coverage-caption",
                        ),
                        html.Div(
                            [
                                html.Span(style={"width": f"{scored / possible * 100:.3f}%"}),
                                html.I(),
                            ],
                            className="gp-models-coverage",
                            role="img",
                            **{
                                "aria-label": f"Aggregate coverage: {int(scored)} of {int(possible)} "
                                "possible target hours, not positions of missing hours"
                            },
                        ),
                    ]
                )
            else:
                children.append(
                    html.Span(
                        f"n = {n_text} scored · future samples: {future_text}. Target-hour coverage unavailable",
                        className="gp-models-coverage-caption",
                    )
                )
            if count is not None and scored is not None and count != scored:
                children.append(
                    html.Span(f"n = {n_text} observations", className="gp-models-coverage-caption")
                )
            details.append(
                html.Details(
                    [
                        html.Summary(
                            (
                                "Seasonal baseline"
                                if model == "baseline"
                                else model_display_name(model)
                            )
                            + " · "
                            + horizon
                            + " · categories & clocks"
                        ),
                        *[
                            html.P(
                                label
                                + ": "
                                + (
                                    str(int(value))
                                    if (value := _count(coverage.get(key))) is not None
                                    else "Unavailable"
                                )
                            )
                            for key, label in categories
                        ],
                        html.P(
                            "Window UTC: "
                            + (utc_timestamp(coverage.get("window_start")) or "Unavailable")
                            + " → "
                            + (utc_timestamp(coverage.get("window_end")) or "Unavailable")
                        ),
                        html.P(
                            "First tracked target UTC: "
                            + (utc_timestamp(block.get("first_target_at")) or "Unavailable")
                        ),
                    ]
                )
            )
            cells.append(
                html.Td(
                    html.Div(
                        children,
                        className="gp-models-sample-cell"
                        + (
                            " gp-models-collecting"
                            if score_text in ("Collecting", "Unavailable")
                            else ""
                        ),
                    ),
                    className="gp-drift-cell--horizon",
                )
            )
        rows.append(html.Tr(cells))
    return html.Div(
        [
            html.Div("Hourly-sampled lead · trailing 7d MAPE", className="gp-control-eyebrow"),
            html.P(
                "Separate hourly observations at 24 ≤ lead <25h, 48 ≤ lead <49h and 72 ≤ lead <73h, "
                "scored when usable actuals arrive.",
                className="gp-panel__caption",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Model"),
                                    *[html.Th(f"≥{h} sample lead") for h in ("24h", "48h", "72h")],
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="metrics-table",
                ),
                className="gp-drift-table-scroll",
                role="region",
                tabIndex=0,
                **{"aria-label": "Hourly-sampled model error and coverage; scroll horizontally"},
            ),
            html.P(
                "Six scored observations permit an average but do not establish representative coverage. "
                "Sampling lead differs from publication lead. Quality-flagged actuals stay excluded "
                "until a revision passes the same checks. Bars summarize counts, not missing-hour locations.",
                className="gp-panel__caption gp-models-coverage-explanation",
            ),
            html.Details(
                [
                    html.Summary("Per-series coverage categories and clocks"),
                    html.P(
                        "Updated UTC: "
                        + (utc_timestamp(sampled.get("last_updated_at")) or "Unavailable"),
                        className="gp-panel__caption",
                    ),
                    *details,
                ],
                className="gp-models-sample-details",
            ),
            legacy,
        ],
        className="gp-models-drift-panel",
        id="hourly-horizon-drift-panel",
    )


def _build_legacy_horizon_drift_panel(region: str | None) -> html.Div:
    """Per-horizon live drift panel — Models tab, #227.

    Reads ``gridpulse:drift_horizon:{region}`` (written by
    ``jobs.phases.write_horizon_drift_metrics``) and shows each model's live
    rolling-7d MAPE at **24h / 48h / 72h** ahead, each cell graded against
    **its own** ``MAPE_BY_HORIZON`` band (the grade is precomputed in the
    payload).

    This is the honest counterpart to the 1-hour panel above: Prophet and
    SARIMAX have no last-value anchor, so they look poor at 1h but can be
    legitimately strong day-ahead — this is the panel where that shows. Records
    mature slowly (a 72h prediction can't be scored until 72h later), so the
    warming state persists a few days post-deploy.
    """
    region = region or "FPL"
    payload = redis_get(redis_key(f"drift_horizon:{region}"))

    if not isinstance(payload, dict) or not payload.get("models"):
        return html.Div(
            [
                html.Div("Live error by forecast horizon", className="gp-control-eyebrow"),
                html.Div(
                    (
                        "No live error records available for these forecast horizons. "
                        "Each forecast can be scored once demand for its target hour is reported; "
                        "a displayed average requires at least six scored observations."
                    ),
                    className="gp-panel__placeholder",
                ),
            ],
            className="gp-models-drift-panel",
            id="horizon-drift-panel-warming",
        )

    horizons = payload.get("horizons") or ["24h", "48h", "72h"]
    drift_models = payload["models"]
    display_order = ("prophet", "arima", "xgboost", "ensemble")
    display_names = {k: model_display_name(k) for k in display_order}
    tone_by_grade = {
        "excellent": "positive",
        "target": "positive",
        "acceptable": "warning",
        "rollback": "negative",
    }
    # This display-count rule is not evidence of statistical sufficiency.
    # Show coverage separately; these records accrue only after targets resolve.
    min_records = 6

    rows: list[html.Tr] = []
    for key in display_order:
        model_block = drift_models.get(key)
        if not model_block:
            continue
        cells: list[html.Td] = [
            html.Td(display_names.get(key, model_display_name(key)), style={"fontWeight": "600"})
        ]
        for horizon in horizons:
            block = model_block.get(horizon) or {}
            mape = block.get("rolling_mape_7d")
            grade = block.get("grade")
            n_records = int(block.get("n_records", 0) or 0)
            # P2-21 (#273): gate the displayed 7d figure on its in-window
            # count. Total retained history is not a seven-day denominator.
            n_window = block.get("n_7d")
            n_eff = int(n_window) if n_window is not None else 0
            if mape is None or n_eff < min_records:
                cells.append(
                    html.Td(
                        html.Span(
                            "More records needed" if n_records else "—",
                            className="gp-status-chip gp-status-chip--secondary",
                        ),
                        className="gp-drift-cell--horizon",
                    )
                )
            else:
                tone = tone_by_grade.get(grade, "secondary")
                cells.append(
                    html.Td(
                        [
                            html.Span(f"{float(mape):.2f}%", className="gp-drift-horizon-mape"),
                            html.Span(
                                _error_band_label(grade, horizon),
                                className=f"gp-status-chip gp-status-chip--{tone}",
                            ),
                            html.Span(
                                f"{n_eff} scored observations",
                                className="gp-panel__caption",
                            ),
                        ],
                        className="gp-drift-cell--horizon",
                    )
                )
        rows.append(html.Tr(cells))

    if not rows:
        return html.Div(
            [
                html.Div("Live error by forecast horizon", className="gp-control-eyebrow"),
                html.Div(
                    "No live error records are available for these forecast horizons.",
                    className="gp-panel__placeholder",
                ),
            ],
            className="gp-models-drift-panel",
        )

    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Model")] + [html.Th(f"{h} horizon") for h in horizons])),
            html.Tbody(rows),
        ],
        className="metrics-table gp-drift-table gp-horizon-drift-table",
    )

    return html.Div(
        [
            html.Div("Live error by forecast horizon — 7d average", className="gp-control-eyebrow"),
            html.Div(
                (
                    "Each column compares MAPE with the configured error band for that forecast horizon."
                ),
                className="gp-drift-headline",
            ),
            html.Div(
                table,
                className="gp-drift-table-scroll",
                role="region",
                tabIndex=0,
                **{"aria-label": "Forecast error comparison table; scroll horizontally if needed"},
            ),
            html.Div(
                (
                    "Seven-day average MAPE, scored after demand for the target hour is reported. "
                    "Offsets are measured from the first forecast target, not the issue clock. "
                    "Counts are scored observations in the window, not seven complete days."
                ),
                className="gp-panel__caption",
            ),
        ],
        className="gp-models-drift-panel",
        id="horizon-drift-panel",
    )


def _get_feature_importance(region: str, top_n: int = 10) -> tuple[list[str], np.ndarray]:
    """Extract cached XGBoost importances; return no values when unavailable."""
    if (region, "xgboost", 0) in _MODEL_CACHE:
        model_dict, _, _ = _MODEL_CACHE[(region, "xgboost", 0)]
        if isinstance(model_dict, dict) and "feature_importances" in model_dict:
            imp = model_dict["feature_importances"]
            sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
            names = [f[0] for f in sorted_feats]
            vals = np.array([f[1] for f in sorted_feats])
            if vals.sum() > 0:
                return names, vals
    return [], np.array([], dtype=float)


_MODELS = ("prophet", "arima", "xgboost", "ensemble")
_DIAGNOSTIC_PAGE_SIZE = 50


def _number(value: object) -> float | None:
    """Return a finite numeric value, preserving real zero and signed residuals."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if np.isfinite(result) else None


def _count(value: object) -> int | None:
    """Unknown or invalid counts stay unavailable rather than becoming zero."""
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _cached_model_payload(region: str) -> dict:
    """Read the published metrics carrier without local model or network fallbacks."""
    payload = redis_get(redis_key(f"forecast:{region}:1h"))
    if not isinstance(payload, dict) or payload.get("region", region) != region:
        return {}
    return payload


def _training_table(region: str, selected: list[str] | None) -> html.Div | html.Table:
    """Filter only real cached training metrics; do not calculate predictions."""
    metrics = _cached_model_payload(region).get("model_metrics") or {}
    chosen = [key for key in _MODELS if key in (selected if isinstance(selected, list) else [])]
    if not chosen:
        return html.Div(
            "Select a model to compare training metrics.", className="gp-panel__placeholder"
        )
    if not isinstance(metrics, dict) or not metrics:
        return html.Div(
            "Training holdout metrics are not available in this region's cached payload.",
            className="gp-panel__placeholder",
        )
    rows = []
    for key in chosen:
        values = metrics.get(key) if isinstance(metrics.get(key), dict) else {}
        cells = []
        for field, fmt in (("mape", ".2f"), ("rmse", ",.0f"), ("mae", ",.0f"), ("r2", ".4f")):
            value = _number(values.get(field))
            if field != "r2" and value is not None and value < 0:
                value = None
            cells.append(
                html.Td(
                    "Unavailable"
                    if value is None
                    else format(value, fmt) + ("%" if field == "mape" else "")
                )
            )
        rows.append(html.Tr([html.Td(model_display_name(key)), *cells]))
    return html.Table(
        [
            html.Thead(
                html.Tr([html.Th(v) for v in ("Model", "MAPE", "RMSE · MW", "MAE · MW", "R²")])
            ),
            html.Tbody(rows),
        ],
        className="metrics-table",
    )


def _training_period(region: str) -> html.Div:
    """Keep payload generation distinct from unavailable holdout evaluation dates."""
    from data.forecast_evidence import utc_timestamp

    payload = _cached_model_payload(region)
    generated = utc_timestamp(payload.get("scored_at"))
    return html.Div(
        [
            html.P(
                "Holdout evaluation: 168h recursive. Exact evaluation start/end and training "
                "completion time are not published in the current metrics cache."
            ),
            html.P(
                f"Metrics carrier generated: {generated or 'Unavailable'}. "
                "This is the scoring payload clock, not the training period."
            ),
        ],
        className="gp-models-period",
    )


def _diagnostic_data(cached: dict | None) -> dict:
    """Retain aligned cached residuals and derive transparent summaries, never model output."""
    from data.forecast_evidence import utc_timestamp

    cached = cached if isinstance(cached, dict) else {}
    source = cached.get("residual_source") or {}
    source = source if isinstance(source, dict) else {}
    times = cached.get("timestamps")
    times = times if isinstance(times, list) else []
    residuals = cached.get("residuals")
    residuals = residuals if isinstance(residuals, list) else []
    predicted = cached.get("predicted", cached.get("ensemble", []))
    predicted = predicted if isinstance(predicted, list) else []
    actual = cached.get("actual")
    actual = actual if isinstance(actual, list) else []
    records: list[dict] = []
    if (
        cached.get("diagnostics_source") != "unavailable"
        and isinstance(times, list)
        and isinstance(residuals, list)
        and len(times) == len(residuals)
    ):
        for i, (stamp, raw) in enumerate(zip(times, residuals, strict=True)):
            value = _number(raw)
            if value is None:
                continue
            utc = utc_timestamp(stamp)
            inferred = False
            if (
                utc is None
                and cached.get("diagnostics_source") == "backtest"
                and source.get("kind") == "walk_forward_backtest"
                and isinstance(stamp, str)
            ):
                try:
                    parsed = pd.Timestamp(stamp)
                    if not pd.isna(parsed) and parsed.tzinfo is None:
                        utc = parsed.tz_localize("UTC").isoformat()
                        inferred = True
                except (ValueError, TypeError):
                    pass
            clock_basis = (
                "Pipeline UTC; legacy offset omitted"
                if inferred
                else "Explicit UTC offset"
                if utc
                else "UTC unavailable"
            )
            records.append(
                {
                    "source_time": str(stamp),
                    "utc": utc,
                    "clock_basis": clock_basis,
                    "inferred_utc": inferred,
                    "time": utc or f"{stamp} · clock unavailable / offset missing",
                    "residual": value,
                    "predicted": _number(predicted[i])
                    if len(predicted) == len(residuals)
                    else None,
                    "actual": _number(actual[i]) if len(actual) == len(residuals) else None,
                }
            )
    hist: list[dict] = []
    hourly: list[dict] = []
    if records:
        vals = np.asarray([r["residual"] for r in records])
        edges = np.histogram_bin_edges(vals, bins="auto")
        if len(edges) > 51:
            edges = np.linspace(float(vals.min()), float(vals.max()), 51)
        counts, edges = np.histogram(vals, bins=edges)
        hist = [
            {"lower": float(edges[i]), "upper": float(edges[i + 1]), "count": int(n)}
            for i, n in enumerate(counts)
        ]
        dated = [
            {"hour": pd.Timestamp(r["utc"]).hour, "error": abs(r["residual"])}
            for r in records
            if r["utc"]
        ]
        if dated:
            groups = pd.DataFrame(dated).groupby("hour")["error"]
            hourly = [
                {"hour": int(hour), "mae": float(values.mean()), "count": int(values.count())}
                for hour, values in groups
            ]
    fi = cached.get("feature_importance") or {}
    importance: list[dict] = []
    if (
        isinstance(fi, dict)
        and isinstance(fi.get("names"), list)
        and isinstance(fi.get("values"), list)
        and len(fi["names"]) == len(fi["values"])
    ):
        for name, raw in zip(fi["names"], fi["values"], strict=True):
            value = _number(raw)
            if isinstance(name, str) and value is not None and value >= 0:
                importance.append({"feature": name, "importance": value})
    return {
        "rows": records,
        "histogram": hist,
        "hourly": hourly,
        "importance": importance,
        "source": source,
        "raw_count": len(residuals) if isinstance(residuals, list) else 0,
    }


def _diagnostic_figures(data: dict, region: str) -> tuple[list[go.Figure], html.Div]:
    """Plot one fixed cached series, with signed zero and explicit sample/clock scope."""
    rows, hist, hourly, importance = (
        data[k] for k in ("rows", "histogram", "hourly", "importance")
    )
    source = data["source"]
    model = model_display_name(source.get("model")) if source.get("model") else "Model unspecified"
    kind = (
        "walk-forward backtest"
        if source.get("kind") == "walk_forward_backtest"
        else "cached residual series"
    )
    mode = source.get("exog_mode")
    mode_label = _EXOG_MODE_CAPTIONS.get(mode, mode) if mode else "exogenous mode unspecified"
    horizon = source.get("horizon")
    horizon_label = f"{horizon}h" if horizon is not None else "Horizon unavailable"
    dates = sorted(r["utc"] for r in rows if r["utc"])
    inferred_count = sum(bool(r.get("inferred_utc")) for r in rows)
    period = (
        f"{dates[0]} → {dates[-1]}" if dates else "UTC window unavailable; no qualified UTC targets"
    )
    legacy_clock_note = (
        f" {inferred_count} legacy offset-free timestamps use the documented UTC "
        "backtest-pipeline contract; other unknown clocks stay unknown."
        if inferred_count
        else ""
    )
    caption = html.Div(
        [
            html.Strong(f"{model} · {horizon_label} {kind} · {mode_label}"),
            html.P(
                f"{len(rows)} finite residual samples; {len(dates)} have UTC targets. Window: {period}."
            ),
            html.Details(
                [
                    html.Summary("Source clocks, sample exclusions and interpretation"),
                    html.P(
                        "Residual = actual − prediction. Positive means reported demand was above the "
                        "forecast; negative means below. This fixed cached series is independent of the "
                        "training comparison filter; it is not a live forecast-error series."
                    ),
                    html.P(
                        f"{data['raw_count'] - len(rows)} residual entries were excluded because values or "
                        "timestamp-array alignment were unavailable. UTC charts omit unknown clocks; "
                        "the exact table retains their source strings." + legacy_clock_note
                    ),
                ],
            ),
        ],
        className="gp-models-scope gp-models-diagnostic-callout",
    )
    empty = (
        "Cached backtest diagnostics unavailable. Published aligned backtest outputs are required."
    )
    figs = [_empty_figure(empty) for _ in range(5)]
    revision = f"{region}:fixed-cached-diagnostics"
    blue = "#5b9fff"
    dated = [r for r in rows if r["utc"]]
    if dated:
        from data.preprocessing import lttb_downsample

        times, values = lttb_downsample(
            pd.to_datetime([r["utc"] for r in dated], utc=True).tz_localize(None).to_numpy(),
            np.asarray([r["residual"] for r in dated]),
            threshold=720,
        )
        figs[0] = go.Figure(
            go.Scatter(
                x=times, y=values, mode="lines", name=model, line={"color": blue, "width": 1.5}
            )
        )
        figs[0].update_layout(
            **_layout(
                uirevision=revision,
                xaxis_title="Target time · UTC",
                yaxis_title="Actual − prediction · MW",
            )
        )
    if hist:
        figs[1] = go.Figure(
            go.Bar(
                x=[(r["lower"] + r["upper"]) / 2 for r in hist],
                y=[r["count"] for r in hist],
                width=[r["upper"] - r["lower"] for r in hist],
                marker_color=blue,
                customdata=[[r["lower"], r["upper"]] for r in hist],
                name="Backtest error count",
                hovertemplate="Residual bin %{customdata[0]:,.3f} to %{customdata[1]:,.3f} MW<br>Count %{y}<extra></extra>",
            )
        )
        figs[1].update_layout(
            **_layout(
                uirevision=revision,
                xaxis_title="Residual bin · MW",
                yaxis_title="Sample count",
                bargap=0.06,
            )
        )
        figs[1].update_yaxes(rangemode="tozero")
    paired = [r for r in rows if r["predicted"] is not None]
    if paired:
        figs[2] = go.Figure(
            go.Scatter(
                x=[r["predicted"] for r in paired],
                y=[r["residual"] for r in paired],
                mode="markers",
                marker={"color": blue, "size": 4, "opacity": 0.55},
                name=model,
            )
        )
        figs[2].update_layout(
            **_layout(
                uirevision=revision,
                xaxis_title="Predicted demand · MW",
                yaxis_title="Actual − prediction · MW",
            )
        )
    if hourly:
        figs[3] = go.Figure(
            go.Bar(
                x=[r["hour"] for r in hourly],
                y=[r["mae"] for r in hourly],
                customdata=[r["count"] for r in hourly],
                marker_color="#efb347",
                name="Hourly mean absolute error",
                hovertemplate="Target hour %{x}:00 UTC<br>MAE %{y:,.3f} MW<br>n=%{customdata}<extra></extra>",
            )
        )
        figs[3].update_layout(
            **_layout(
                uirevision=revision,
                xaxis_title="Target hour · UTC",
                yaxis_title="Mean absolute error · MW",
            )
        )
        figs[3].update_yaxes(rangemode="tozero")
    if importance:
        ordered = sorted(importance, key=lambda r: r["importance"])
        figs[4] = go.Figure(
            go.Bar(
                x=[r["importance"] for r in ordered],
                y=[r["feature"] for r in ordered],
                orientation="h",
                marker_color="#5b9fff",
                text=[f"{r['importance']:.4f}" for r in ordered],
                textposition="outside",
                cliponaxis=False,
                name="Built-in XGBoost importance",
            )
        )
        figs[4].update_layout(
            **_layout(uirevision=revision, xaxis_title="Built-in importance · unitless")
        )
        figs[4].update_layout(margin={"l": 155, "r": 20, "t": 20, "b": 45})
    else:
        figs[4] = _empty_figure("Cached XGBoost feature importance unavailable for this region.")
    if rows:
        values = [r["residual"] for r in rows]
        lo, hi = min(0.0, min(values)), max(0.0, max(values))
        pad = (hi - lo) * 0.08 or 1.0
        for index in (0, 2):
            if figs[index].data:
                figs[index].add_hline(y=0, line={"color": "#c5cddd", "dash": "dash", "width": 1})
                figs[index].update_yaxes(range=[lo - pad, hi + pad])
    for index, fig in enumerate(figs):
        fig.update_layout(
            paper_bgcolor="#111820",
            plot_bgcolor="#111820",
            font={
                "family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                "size": 10,
                "color": "#b4c4d8",
            },
            showlegend=False,
            legend={"orientation": "h", "x": 0, "y": 1.2, "font": {"size": 10}},
            margin={"l": 52 if index != 4 else 145, "r": 34, "t": 33, "b": 34},
            height=260,
            meta={"diagnostic_utc_qualified": index == 0 and bool(dated)},
        )
        fig.update_xaxes(title=None, showgrid=False, tickfont={"size": 10}, zeroline=False)
        fig.update_yaxes(
            title=None,
            showgrid=index != 4,
            gridcolor="#2b3a4c",
            griddash="dot",
            tickfont={"size": 10},
            zeroline=False,
        )
        if index == 4:
            fig.update_xaxes(visible=False)
            fig.update_layout(showlegend=False, margin={"l": 145, "r": 52, "t": 10, "b": 10})
    return figs, caption


def _exact_table(headers: list[str], rows: list[list]) -> html.Div:
    """Provide keyboard-scrollable exact values without decimal rounding."""
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody(
                    [
                        html.Tr([html.Td("Unavailable" if v is None else str(v)) for v in row])
                        for row in rows
                    ]
                ),
            ],
            className="metrics-table",
        ),
        className="gp-models-table-scroll",
        role="region",
        tabIndex=0,
        **{"aria-label": headers[0] + " diagnostic values; scroll horizontally"},
    )


def _diagnostic_evidence(data: dict | None, page: object) -> html.Div:
    """Page every retained residual; expose actual histogram counts, hourly n and importance."""
    if not isinstance(data, dict):
        return html.Div("Cached diagnostic values unavailable.")
    rows = data.get("rows") or []
    pages = max(1, (len(rows) + _DIAGNOSTIC_PAGE_SIZE - 1) // _DIAGNOSTIC_PAGE_SIZE)
    try:
        selected = max(1, min(pages, int(page or 1)))
    except (ValueError, TypeError):
        selected = 1
    start = (selected - 1) * _DIAGNOSTIC_PAGE_SIZE
    return html.Div(
        [
            html.P(
                f"Residual rows {start + 1 if rows else 0}–{min(start + _DIAGNOSTIC_PAGE_SIZE, len(rows))} "
                f"of {len(rows)}. Source strings are retained when no UTC offset was supplied.",
                className="gp-models-table-caption",
            ),
            _exact_table(
                [
                    "Target UTC / qualified source clock",
                    "Source timestamp",
                    "Clock basis",
                    "Actual · MW",
                    "Predicted · MW",
                    "Actual − prediction · MW",
                ],
                [
                    [
                        r["time"],
                        r["source_time"],
                        r.get("clock_basis"),
                        r["actual"],
                        r["predicted"],
                        r["residual"],
                    ]
                    for r in rows[start : start + _DIAGNOSTIC_PAGE_SIZE]
                ],
            ),
            html.H3("Residual histogram bins", className="gp-control-eyebrow"),
            html.P(
                "NumPy auto-width bins, capped at 50. Lower edges inclusive; upper edges exclusive "
                "except the final bin, whose upper edge is included. Counts sum to retained residuals."
            ),
            _exact_table(
                ["Lower residual · MW", "Upper residual · MW", "Count"],
                [[r["lower"], r["upper"], r["count"]] for r in data.get("histogram", [])],
            ),
            html.H3("Mean absolute error by target hour", className="gp-control-eyebrow"),
            _exact_table(
                ["Hour · UTC", "MAE · MW", "n"],
                [[r["hour"], r["mae"], r["count"]] for r in data.get("hourly", [])],
            ),
            html.H3("Built-in XGBoost importance", className="gp-control-eyebrow"),
            _exact_table(
                ["Feature", "Importance · unitless"],
                [[r["feature"], r["importance"]] for r in data.get("importance", [])],
            ),
        ]
    )


def _review_context(region: str) -> tuple[Any, Any, Any, Any]:
    """Read source-specific investigation context without substituting scores."""
    from urllib.parse import quote

    from data.forecast_evidence import utc_timestamp
    from data.forecast_identity import resolve_forecast_identity

    payload = _cached_model_payload(region)
    identity = resolve_forecast_identity(payload, "published")
    published = html.Div(
        [
            html.Strong(
                "Published forecast: " + (identity.label + "." if identity else "Unavailable.")
            ),
            " Comparison models, observed scores and backtest errors retain separate evidence scopes.",
        ]
    )
    sampled = redis_get(redis_key(f"drift_hourly:{region}:summary"))
    sampled = sampled if isinstance(sampled, dict) else {}
    models = sampled.get("models") or {}
    models = models if isinstance(models, dict) else {}
    key = identity.metric_key if identity and identity.metric_key in models else None
    if key is None:
        key = next(
            (name for name in ("ensemble", "xgboost", "prophet", "arima") if name in models), None
        )
    series = models.get(key) or {}
    block = series.get("24h") or {} if isinstance(series, dict) else {}
    block = block if isinstance(block, dict) else {}
    coverage = block.get("coverage_7d") or {}
    coverage = coverage if isinstance(coverage, dict) else {}

    def clock(value):
        stamp = utc_timestamp(value)
        return pd.Timestamp(stamp).strftime("%d %b %Y, %H:%M UTC") if stamp else "Unavailable"

    collection = html.P(
        "Collection began "
        + clock(sampled.get("capture_started_at"))
        + ". First 24h target: "
        + clock(block.get("first_target_at"))
        + ". Pending and future samples are distinct from historical scores.",
        className="gp-models-rail-clock",
    )
    source_label = (
        "Seasonal baseline"
        if key == "baseline"
        else model_display_name(key)
        if key
        else "Unavailable"
    )
    categories = [
        ("scored_target_hours", "Scored target hours"),
        ("possible_target_hours", "Possible target hours"),
        ("not_yet_tracked_hours", "Not yet tracked"),
        ("pending_actual_hours", "Pending actuals / quality resolution"),
        ("missing_sample_hours", "Missing samples"),
        ("series_unavailable_hours", "Series unavailable"),
        ("quality_guard_excluded_hours", "Quality-guard excluded"),
        ("low_actual_excluded_hours", "Low-actual exclusions"),
    ]
    coverage_context = html.Div(
        [
            html.P(
                f"Coverage reference: {source_label} · 24h sampled lead. Displayed models retain their own scores and counts."
            ),
            html.P(
                "Window: "
                + clock(coverage.get("window_start"))
                + " → "
                + clock(coverage.get("window_end"))
            ),
            html.P("Last updated: " + clock(sampled.get("last_updated_at"))),
            html.Dl(
                [
                    html.Div(
                        [
                            html.Dt(label),
                            html.Dd(
                                str(value)
                                if (value := _count(coverage.get(name))) is not None
                                else "Unavailable"
                            ),
                        ]
                    )
                    for name, label in categories
                ]
            ),
            html.P("Bars summarize the scored share, not the positions of missing hours."),
        ]
    )
    link = html.A(
        "Inspect forecast run evidence →",
        href=f"/?region={quote(region, safe='')}&tab=tab-outlook",
        className="gp-review-button",
    )
    return published, collection, coverage_context, link


def register_models_callbacks(app: Any) -> None:
    """Register cached, independently scoped model evidence callbacks."""
    from components.cards import build_page_title
    from config import REGION_NAMES

    @app.callback(
        Output("models-title", "children"),
        Input("region-selector", "value"),
        Input("dashboard-tabs", "active_tab"),
    )
    def update_models_title(region: str | None, active_tab: str) -> Any:
        if active_tab != "tab-models":
            return no_update
        return build_page_title(
            "Models",
            "Start with observed error and sample coverage; use training and backtest diagnostics to investigate · "
            + REGION_NAMES.get(region or "FPL", region or "FPL"),
        )

    @app.callback(
        Output("models-published-source", "children"),
        Output("models-investigation-context", "children"),
        Output("models-coverage-context", "children"),
        Output("models-forecast-evidence-link", "children"),
        Input("region-selector", "value"),
        Input("dashboard-tabs", "active_tab"),
        Input("demand-store", "data"),
    )
    def update_models_review_context(region, active_tab, demand_json):
        if active_tab != "tab-models":
            return (no_update,) * 4
        return _review_context(region or "FPL")

    @app.callback(
        Output("models-section", "value", allow_duplicate=True),
        Input("models-review-observed", "n_clicks"),
        Input("models-review-backtest", "n_clicks"),
        prevent_initial_call=True,
    )
    def navigate_models_evidence(observed_clicks, backtest_clicks):
        if ctx.triggered_id == "models-review-observed" and observed_clicks:
            return "observed"
        if ctx.triggered_id == "models-review-backtest" and backtest_clicks:
            return "backtest"
        return no_update

    @app.callback(
        Output("models-observed-section", "style"),
        Output("models-training-section", "style"),
        Output("models-backtest-section", "style"),
        Input("models-section", "value"),
    )
    def show_models_section(section: str | None) -> tuple[dict, dict, dict]:
        selected = section if section in {"observed", "training", "backtest"} else "observed"
        return tuple(
            {} if selected == name else {"display": "none"}
            for name in ("observed", "training", "backtest")
        )

    @app.callback(
        Output("models-drift-panel", "style"),
        Output("models-first-hour-panel", "style"),
        Input("models-observed-basis", "value"),
    )
    def show_models_observed_basis(basis: str | None) -> tuple[dict, dict]:
        return ({"display": "none"}, {}) if basis == "first" else ({}, {"display": "none"})

    @app.callback(
        Output("models-leaderboard", "children"),
        Input("region-selector", "value"),
        Input("dashboard-tabs", "active_tab"),
    )
    def update_models_leaderboard(region: str | None, active_tab: str) -> Any:
        return _training_period(region or "FPL") if active_tab == "tab-models" else no_update

    @app.callback(
        Output("models-drift-panel", "children"),
        Output("models-first-hour-panel", "children"),
        Output("models-legacy-panel", "children"),
        Input("region-selector", "value"),
        Input("dashboard-tabs", "active_tab"),
        Input("demand-store", "data"),
    )
    def update_models_drift_panel(
        region: str | None, active_tab: str, demand_json: str | None
    ) -> Any:
        if active_tab != "tab-models":
            return no_update, no_update, no_update
        metrics = _cached_model_payload(region or "FPL").get("model_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        return (
            _build_horizon_drift_panel(region, include_legacy=False),
            _build_drift_panel(region, metrics=metrics),
            html.Details(
                [
                    html.Summary("Advanced · legacy forecast-start offsets"),
                    _build_legacy_horizon_drift_panel(region),
                ],
                className="gp-models-rail-disclosure gp-models-legacy-disclosure",
            ),
        )

    @app.callback(
        Output("tab3-metrics-table", "children"),
        Input("tab3-model-selector", "value"),
        Input("region-selector", "value"),
        Input("dashboard-tabs", "active_tab"),
    )
    def update_models_training(
        selected: list[str] | None, region: str | None, active_tab: str
    ) -> Any:
        return (
            _training_table(region or "FPL", selected) if active_tab == "tab-models" else no_update
        )

    @app.callback(
        *[
            Output(ident, "figure")
            for ident in (
                "tab3-residuals-time",
                "tab3-residuals-hist",
                "tab3-residuals-pred",
                "tab3-error-heatmap",
                "tab3-shap",
            )
        ],
        Output("models-diagnostics-data", "data"),
        Output("tab3-diagnostics-scope", "children"),
        Output("models-diagnostics-page", "options"),
        Output("models-diagnostics-page", "value"),
        Input("demand-store", "data"),
        Input("dashboard-tabs", "active_tab"),
        Input("region-selector", "value"),
        State("models-diagnostics-page", "value"),
    )
    def update_models_tab(
        demand_json: str | None, active_tab: str, region: str | None, page: object = None
    ) -> Any:
        if active_tab != "tab-models":
            return [no_update] * 9
        region = region or "FPL"
        cached = redis_get(redis_key(f"diagnostics:{region}"))
        if isinstance(cached, dict) and cached.get("region", region) != region:
            cached = None
        data = _diagnostic_data(cached)
        figures, scope = _diagnostic_figures(data, region)
        pages = max(1, (len(data["rows"]) + _DIAGNOSTIC_PAGE_SIZE - 1) // _DIAGNOSTIC_PAGE_SIZE)
        options = [
            {
                "label": f"Rows {(n - 1) * _DIAGNOSTIC_PAGE_SIZE + 1}–{min(n * _DIAGNOSTIC_PAGE_SIZE, len(data['rows']))}",
                "value": n,
            }
            for n in range(1, pages + 1)
        ]
        if not data["rows"]:
            options = [{"label": "No residual rows", "value": 1}]
        try:
            selected_page = max(1, min(pages, int(page or 1)))
        except (ValueError, TypeError):
            selected_page = 1
        return *figures, data, scope, options, selected_page

    @app.callback(
        Output("tab3-diagnostics-evidence", "children"),
        Input("models-diagnostics-data", "data"),
        Input("models-diagnostics-page", "value"),
    )
    def update_models_exact_values(data: dict | None, page: object) -> html.Div:
        return _diagnostic_evidence(data, page)


# Leaderboard MAPEs are 168h-holdout numbers → grade them on the 7d governance
# band (config.MAPE_BY_HORIZON). Map the H2 grade to a MetricsBar tone so a
# model's color on the live surface matches the governance rule for the very
# metric shown, instead of ad-hoc 2.5/5.0 literals (2026-07 review P2-24).
_MAPE_GRADE_TONE = {
    "excellent": "positive",
    "target": "positive",
    "acceptable": "secondary",
    "rollback": "negative",
}

# ── Relocated from ``_callbacks_overview`` (§1c of the GP-P1-04 proposal) ──
#
# These render THIS tab's surfaces. They lived in the Overview module only
# because that module was written first and the panels were prototyped
# there; every caller has always been here.


def _leaderboard_mape_tone(mape: float) -> str:
    return _MAPE_GRADE_TONE.get(mape_grade(mape, "7d"), "secondary")


def _build_models_leaderboard(region: str | None) -> html.Div:
    """5-up MetricsBar leaderboard — Prophet / SARIMAX / XGBoost / Ensemble / EIA.

    Hero highlight goes to the model with the lowest MAPE. Sub-line shows
    MAE; tone comes from the H2 governance grade (``mape_grade`` on the 7d
    band, since these are 168h-holdout MAPEs) via ``_leaderboard_mape_tone``
    — not ad-hoc thresholds, so the color matches governance (#P2-24).
    """
    region = region or "FPL"
    try:
        from models.model_service import get_model_metrics
    except ImportError:  # pragma: no cover
        return html.Div()

    metrics_dict = get_model_metrics(region) or {}
    if not metrics_dict:
        return html.Div(
            "Model metrics not yet available for this region.",
            className="gp-panel__placeholder",
        )

    order = ("prophet", "arima", "xgboost", "ensemble", "eia")
    display_names = {k: model_display_name(k) for k in order}
    # EIA is the reference series on this panel, not one of our models —
    # the only place a longer label than the canonical one is wanted.
    display_names["eia"] = "EIA Reference"

    # Hero pick: model with lowest mape
    valid = [
        (k, v) for k, v in metrics_dict.items() if isinstance(v, dict) and v.get("mape") is not None
    ]
    hero_key = min(valid, key=lambda kv: float(kv[1].get("mape", 999)))[0] if valid else None

    items: list[dict] = []
    for key in order:
        if key not in metrics_dict:
            continue
        m = metrics_dict[key]
        if m.get("mape") is None:
            # An absent metric must render as unavailable, not as 0.0%.
            items.append(
                {
                    "label": display_names.get(key, model_display_name(key)),
                    "value": "—",
                    "unit": "metrics unavailable",
                    "tone": "secondary",
                    "hero": key == hero_key,
                }
            )
            continue
        mape = float(m["mape"])
        mae = m.get("mae")
        tone = _leaderboard_mape_tone(mape)
        items.append(
            {
                "label": display_names.get(key, model_display_name(key)),
                "value": f"{mape:.1f}%",
                "unit": f"MAPE · MAE {float(mae):,.0f} MW" if mae is not None else "MAPE",
                "tone": tone,
                "hero": key == hero_key,
            }
        )
    if not items:
        return html.Div(
            "Model metrics not yet available for this region.",
            className="gp-panel__placeholder",
        )
    bar = build_metrics_bar(items)
    bar.className = f"gp-metrics-bar gp-metrics-bar--{len(items)}up"
    return bar


__all__ = [
    "_format_metric",
    "_models_tab_from_redis",
    "_get_feature_importance",
    "register_models_callbacks",
]
