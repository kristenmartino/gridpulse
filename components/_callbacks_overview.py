"""Overview / mission-control tab helpers extracted from ``components/callbacks.py``.

Step 7a of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)
* #101 — Alerts tab (``_callbacks_alerts.py``)
* #102 — Generation tab (``_callbacks_generation.py``)
* #103 — Weather tab (``_callbacks_weather.py``)

## Scope of this module

The Overview tab is the largest single tab in callbacks.py (~1,900 lines
of helpers in total). To keep PR diffs reviewable, this module is being
populated in three sub-steps:

1. **This file's initial pass (PR #104):** the headline / mission-control
   block — title, 5-up MetricsBar items, hero chart, model card,
   insight narrative.
2. **Next:** Overview panels (drivers / generation / models leaderboard /
   risk / scenarios) — adds ~600 lines here.
3. **Next:** Overview briefing surface (sparklines / briefing / digest /
   spotlights / weather context / data-health / changes / news /
   persona KPIs) — adds ~900 lines here.

After all three land, the file is the single home for every
``_build_overview_*`` / ``_spotlight_*`` / ``_build_persona_kpis``
helper. ``register_callbacks`` re-imports them by name from here.

## What lives here today

* ``_build_overview_title`` — page-title block (region name + subtitle).
* ``_build_overview_metrics_items`` — the 5-up MetricsBar cells
  (Now / 7d Peak / 7d Low / Average / 24h Trend), NaN-aware so the
  hero metric never displays as ``nan`` when EIA-930 has a publishing
  lag on the most recent hour.
* ``_build_overview_hero_chart`` — 7d actual demand + 24h forecast bridge
  with 80% confidence band. Mirrors the v2 ``DemandChart.tsx`` shape.
* ``_build_overview_model_card`` — horizontal model-performance bar
  showing the primary (ensemble → xgboost → first available) model's
  MAPE / RMSE / MAE / R² with a trained-vs-simulated badge.
* ``_build_overview_insight`` — three-sentence narrative paragraph with
  semantic-color delta spans (rising demand is "warning" not "good").

## Public-import surface

``components/callbacks.py`` re-imports each function by name. Tests
import via ``from components.callbacks import _build_overview_*`` —
the re-export shim keeps those import sites valid without any
caller-side changes. ``register_callbacks`` continues to call the
helpers directly through the same namespace.
"""

from __future__ import annotations

import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import Input, Output, State, dcc, html, no_update

from components._callbacks_shared import (
    _BACKTEST_CACHE,
    _EIA_FUEL_MAP,
    _GENERATION_CACHE,
    _PREDICTION_CACHE,
    DEFAULT_BACKTEST_EXOG_MODE,
    _empirical_interval_from_backtests,
    _empty_figure,
    _latest_real_demand,
    _layout,
)
from components.accessibility import CB_PALETTE
from components.cards import (
    build_insight_card,
    build_kpi_row,
    build_metrics_bar,
    build_model_metrics_card,
    build_news_feed,
    build_page_title,
)
from config import (
    FRESHNESS_FRESH_MAX_AGE_HOURS,
    REGION_CAPACITY_MW,
    REGION_NAMES,
    REQUIRE_REDIS,
)
from data.redis_client import redis_get, redis_key
from personas.config import get_persona

log = structlog.get_logger()


def _read_ensemble_forecast_from_redis(
    region: str,
) -> tuple[list[pd.Timestamp], np.ndarray, str | None] | None:
    """Read the live ensemble forecast from ``gridpulse:forecast:{region}:1h``.

    The Forecast tab already reads from this key (via
    ``_outlook_tab_from_redis`` in ``_callbacks_forecast.py``). This helper
    extracts the same shape for the Overview hero chart + insight card so
    those surfaces stop falling back to ``models.model_service._simulate_forecasts``,
    which was producing noisy historical actuals displayed as forward
    forecasts (CLAUDE.md "Redis-only reads in the web tier" invariant
    violation discovered 2026-05-20).

    Returns:
        ``(timestamps, ensemble_predictions, scored_at)`` when the Redis
        payload exists and contains an ``ensemble`` key per row.
        ``None`` when Redis is cold, the payload is malformed, or the
        ensemble column isn't populated — the caller should render the
        actual-only / warming state in that case (never the simulated
        baseline).
    """
    cached = redis_get(redis_key(f"forecast:{region}:1h"))
    if not isinstance(cached, dict):
        return None
    forecasts = cached.get("forecasts") or []
    if not forecasts:
        return None

    # Prefer the explicit ensemble column. Fall back to predicted_demand_mw
    # (which mirrors the primary model — XGBoost by default) only when the
    # ensemble entry is missing entirely. We deliberately do not fall back
    # to a single base model masquerading as ensemble.
    first_row = forecasts[0]
    if "ensemble" in first_row:
        pred_key = "ensemble"
    elif "predicted_demand_mw" in first_row:
        pred_key = "predicted_demand_mw"
    else:
        return None

    try:
        timestamps = [pd.to_datetime(row["timestamp"]) for row in forecasts]
        predictions = np.array([float(row.get(pred_key, 0)) for row in forecasts], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("overview_forecast_redis_parse_failed", region=region, error=str(exc))
        return None

    return timestamps, predictions, cached.get("scored_at")


def _resolve_forecast_mape(region: str) -> tuple[float | None, str]:
    """Resolve the most-honest MAPE to display alongside the forecast.

    Returns ``(mape_value, source_label)`` where source_label carries both
    the window and the METRIC NAME actually used (e.g. ``"live 7d sMAPE"``,
    ``"live 30d MAPE"``, ``"holdout MAPE"``) — callers render it verbatim so
    an sMAPE value is never presented as MAPE:

    - ``"live 7d …"`` — rolling 7-day live drift error from forecast-vs-actual
      observations stored in ``gridpulse:drift:{region}`` (the headline
      number; reflects how the model is actually performing right now).
      Prefers sMAPE (bounded, robust to near-zero-actual artifacts — #142/
      PR-G9), falling back to rolling MAPE for pre-G9 payloads.
    - ``"live 30d"`` — rolling 30-day live drift error (fallback when 7d
      window has <24 records, e.g. first week post-deploy)
    - ``"holdout"`` — training-time holdout MAPE from each pickle's
      ``meta.extra["holdout_metrics"]`` (clearly labeled — this is what
      the model claimed at training time, not how it's doing live)
    - ``""`` (with value ``None``) — nothing reliable available; the
      forecast clause drops the MAPE annotation entirely

    The live drift data was written hourly to Redis by the scoring job
    since PR #126. The Overview clause didn't read from it until this
    PR — it was citing training holdout MAPE instead, which was
    technically truthful but misleading because users read the MAPE
    figure as "expected accuracy of this specific forecast."
    """
    # Layer 1: live rolling drift error. Prefer sMAPE (bounded; a near-zero
    # actual can't pin the headline at ~200% the way raw MAPE did for LDWP —
    # #142/PR-G9), falling back to the now-filtered rolling MAPE for payloads
    # written before G9 (which carry no sMAPE field).
    try:
        drift_payload = redis_get(redis_key(f"drift:{region}"))
        if isinstance(drift_payload, dict):
            models = drift_payload.get("models") or {}
            ens = models.get("ensemble") or {}
            # Require a meaningful window — 24 hourly records minimum before
            # the 7d figure is statistically defensible. Below that, it swings
            # wildly on each new tick.
            n_records = int(ens.get("n_records", 0) or 0)
            # Track WHICH metric supplied the value — an sMAPE number must
            # never be labeled "MAPE" (2026-07 critical-review finding P1-8;
            # for artifact-prone BAs the two can differ by an order of
            # magnitude, e.g. LDWP sMAPE ~13% vs raw MAPE ~190%).
            live_7d = ens.get("rolling_smape_7d")
            metric_7d = "sMAPE"
            if live_7d is None:
                live_7d = ens.get("rolling_mape_7d")
                metric_7d = "MAPE"
            live_30d = ens.get("rolling_smape_30d")
            metric_30d = "sMAPE"
            if live_30d is None:
                live_30d = ens.get("rolling_mape_30d")
                metric_30d = "MAPE"
            if live_7d is not None and n_records >= 24 and np.isfinite(float(live_7d)):
                return float(live_7d), f"live 7d {metric_7d}"
            if live_30d is not None and n_records >= 24 and np.isfinite(float(live_30d)):
                return float(live_30d), f"live 30d {metric_30d}"
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("forecast_mape_drift_read_failed", region=region, error=str(exc))

    # Layer 2: training-time holdout MAPE (existing path), clearly labeled
    try:
        from models.model_service import get_model_metrics

        metrics_dict = get_model_metrics(region) or {}
        ens_metrics = metrics_dict.get("ensemble") or metrics_dict.get("xgboost") or {}
        mape = ens_metrics.get("mape")
        if mape is not None:
            mape_f = float(mape)
            if mape_f > 0 and np.isfinite(mape_f):
                return mape_f, "holdout MAPE"
    except Exception:  # pragma: no cover — defensive
        pass

    return None, ""


def _build_overview_title(region: str) -> html.Div:
    """Page-title block: region name + 1-line subtitle."""

    region_name = REGION_NAMES.get(region, region)
    subtitle = f"Demand forecast and grid intelligence · {region}"
    return build_page_title(region_name, subtitle)


def _build_overview_metrics_items(demand_df: pd.DataFrame | None) -> list[dict]:
    """Compose the 5-up MetricsBar cells (Now / 7d Peak / 7d Low / Average / 24h Trend)."""
    placeholder_labels = ["Now", "7d Peak", "7d Low", "Average", "24h Trend"]
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        items = [
            {"label": label, "value": "—", "unit": None, "tone": "secondary"}
            for label in placeholder_labels
        ]
        items[0]["hero"] = True
        items[0]["tone"] = "primary"
        return items

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # Strip spurious zero-demand rows (EIA's missing-observation marker)
    # AND NaN rows (EIA-930's publishing lag for the most recent hour,
    # especially for newer / smaller BAs like PSCO / NEVP / AZPS — those
    # rows arrive at the next hourly tick instead). The ``> 0`` check
    # filters both: NaN > 0 is False.
    nonzero = df[df["demand_mw"] > 0].reset_index(drop=True)
    last_7d = nonzero.tail(168)

    # ``now_value`` reads from ``nonzero`` rather than ``df`` so the
    # most recent NaN / zero hour doesn't surface as "nan" / "0" in the
    # hero metric. Falls back to "—" when no usable reading exists.
    now_value = float(nonzero["demand_mw"].iloc[-1]) if not nonzero.empty else None
    now_ts = nonzero["timestamp"].iloc[-1] if not nonzero.empty else None
    peak_7d = float(last_7d["demand_mw"].max()) if not last_7d.empty else 0.0
    low_7d = float(last_7d["demand_mw"].min()) if not last_7d.empty else 0.0
    avg_7d = float(last_7d["demand_mw"].mean()) if not last_7d.empty else 0.0

    # 24h trend (#9 — 2026-05-20) — now uses TIMESTAMP-based lookup
    # instead of ``iloc[-25]`` of non-zero rows. The previous index-based
    # approach silently compared NOW against "the 25th-from-last published
    # hour," which is only "24h ago" when there are zero publishing gaps
    # in the last 24 hours. With EIA's 1-4h publishing lag and occasional
    # mid-day gaps, the index approach drifted in practice.
    #
    # Trend semantics: compare NOW to the demand value at (now_ts - 24h).
    # Tolerance window of ±90 min absorbs single-hour publishing gaps
    # (EIA's most common gap profile — one missing hour, neighbors at
    # exactly ±60 min from the 24h-ago target). Wider gaps surface "—"
    # rather than fabricating a comparison against a 2h-or-more-off
    # anchor.
    trend_pct: float | None
    trend_anchor_ts: pd.Timestamp | None = None
    if now_value is not None and now_ts is not None:
        target_ts = now_ts - pd.Timedelta(hours=24)
        window_lo = target_ts - pd.Timedelta(minutes=90)
        window_hi = target_ts + pd.Timedelta(minutes=90)
        candidates = nonzero[
            (nonzero["timestamp"] >= window_lo) & (nonzero["timestamp"] <= window_hi)
        ]
        if not candidates.empty:
            # Pick the candidate closest to the exact 24h-ago target.
            deltas = (candidates["timestamp"] - target_ts).abs()
            closest_idx = deltas.idxmin()
            ago_value = float(candidates.loc[closest_idx, "demand_mw"])
            trend_anchor_ts = candidates.loc[closest_idx, "timestamp"]
            trend_pct = ((now_value - ago_value) / ago_value * 100.0) if ago_value else None
        else:
            trend_pct = None
    else:
        trend_pct = None

    # Inverted semantic: rising demand reads as "warning" (negative tone),
    # falling demand reads as "positive" — matches v2 MetricsBar.tsx:64.
    if trend_pct is None:
        trend_tone = "secondary"
    elif trend_pct > 0.5:
        trend_tone = "negative"
    elif trend_pct < -0.5:
        trend_tone = "positive"
    else:
        trend_tone = "secondary"

    now_display = f"{now_value:,.0f}" if now_value is not None else "—"
    trend_display = f"{trend_pct:+.1f}%" if trend_pct is not None else "—"

    # Freshness subtext on NOW — "NOW" without context reads as
    # wall-clock now; in practice it's the most recent EIA-published
    # hour, which can be 1-4 hours behind because of EIA's publishing
    # lag. Make that explicit.
    now_subtext = f"as of {now_ts.strftime('%H:%M UTC')}" if now_ts is not None else None

    # Trend anchor subtext — if the 24h-ago row was off-target by more
    # than ~5 minutes (publishing gap absorbed by the ±30min tolerance),
    # surface the actual anchor time so users can see the comparison
    # isn't a perfect 24h.
    trend_subtext = None
    if trend_anchor_ts is not None and now_ts is not None:
        exact_target = now_ts - pd.Timedelta(hours=24)
        if abs((trend_anchor_ts - exact_target).total_seconds()) > 300:
            trend_subtext = f"vs {trend_anchor_ts.strftime('%H:%M UTC')}"

    return [
        {
            "label": "Now",
            "value": now_display,
            "unit": "MW",
            "hero": True,
            "subtext": now_subtext,
        },
        {
            "label": "7d Peak",
            "value": f"{peak_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "hourly max",
        },
        {
            "label": "7d Low",
            "value": f"{low_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "hourly min",
        },
        {
            "label": "Average",
            "value": f"{avg_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "7d hourly mean",
        },
        {
            "label": "24h Trend",
            "value": trend_display,
            "unit": None,
            "tone": trend_tone,
            "subtext": trend_subtext,
        },
    ]


def _build_overview_hero_chart(
    region: str,
    demand_df: pd.DataFrame | None,
) -> go.Figure:
    """7d actual demand + 24h forecast bridge with confidence band.

    Mirrors gridpulse-v2 components/DemandChart.tsx — blue solid actual with
    a faint area fill, orange dashed forecast bridged from the last actual
    point, and an orange-tinted confidence ribbon under the forecast.
    """
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return _empty_figure("No demand data")

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    actual = df.tail(168)
    if actual.empty:
        return _empty_figure("No recent demand")

    # ``last_mw`` is the bridge point between the actual line and the
    # forecast trace. A NaN tail (EIA-930 publishing lag) would render
    # the bridge as a gap; instead we walk back to the most recent real
    # reading. ``last_ts`` matches that row so the bridge stays time-aligned.
    real_actual = actual[actual["demand_mw"].notna() & (actual["demand_mw"] > 0)]
    if real_actual.empty:
        return _empty_figure("No recent demand")
    last_ts = real_actual["timestamp"].iloc[-1]
    last_mw = float(real_actual["demand_mw"].iloc[-1])

    fig = go.Figure()

    # Actual demand: blue solid + faint area fill below
    fig.add_trace(
        go.Scatter(
            x=actual["timestamp"],
            y=actual["demand_mw"].where(actual["demand_mw"] > 0),
            mode="lines",
            name="Actual",
            line=dict(color="#3b82f6", width=1.75),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)",
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>%{y:,.0f} MW<extra></extra>",
        )
    )

    # 24h forecast bridge (orange dashed) + confidence band.
    #
    # Reads the live ensemble forecast from gridpulse:forecast:{region}:1h
    # (written hourly by the scoring job). Falls back to actual-only chart
    # when Redis is cold rather than to models.model_service._simulate_forecasts,
    # which prior to 2026-05-20 was rendering noisy *historical* actuals at
    # *forward* timestamps — producing visibly wrong forecast traces (e.g. FPL
    # peak appearing at 04:00 instead of the real evening peak). See the
    # 2026-05-20 "looks off" debug + the fix branch's PR for the full
    # diagnosis.
    try:
        forecast_payload = _read_ensemble_forecast_from_redis(region)
        if forecast_payload is not None:
            forecast_ts_all, ensemble_arr, scored_at_iso = forecast_payload
            horizon = min(24, len(ensemble_arr))
            forecast_ts = forecast_ts_all[:horizon]
            ensemble_y = list(ensemble_arr[:horizon])

            # Confidence band (PR-B, 2026-05-20): prefer calibrated
            # empirical quantiles of recent backtest residuals, fall back
            # to a ±3 % heuristic only when the calibration window is too
            # small (typically <24 residual samples — first week
            # post-deploy, or for newly-added regions).
            #
            # The empirical method is the same one the Forecast tab uses
            # (``_empirical_interval_from_backtests`` →
            # ``apply_empirical_interval``); see
            # ``components._callbacks_shared:385-409`` and
            # ``components._callbacks_forecast:108-170``. Sharing the
            # method across surfaces means both views show a band
            # calibrated to the same residual distribution — no
            # surface-specific tuning that would silently diverge.
            interval_meta = _empirical_interval_from_backtests(region, "ensemble", horizon)
            empirical_ok = bool(interval_meta.get("available"))
            if empirical_ok:
                # Additive bands: lower_error / upper_error are quantiles
                # of (actual - predicted) residuals, so band_y = pred + q.
                lower_err = float(interval_meta["lower_error"])
                upper_err = float(interval_meta["upper_error"])
                upper_y = [v + upper_err for v in ensemble_y]
                lower_y = [v + lower_err for v in ensemble_y]
                # Disclose the calibration source when the residuals came
                # from a substitute model (the prod backtest payload only
                # carries XGBoost predictions), so the band never implies
                # it was calibrated on the displayed ensemble.
                calib = interval_meta.get("calibration_model")
                calib_note = "" if calib in (None, "ensemble") else f", {calib}-calibrated"
                band_name = (
                    f"80% prediction interval "
                    f"(empirical, n={int(interval_meta.get('sample_size', 0))}{calib_note})"
                )
            else:
                upper_y = [v * 1.03 for v in ensemble_y]
                lower_y = [v * 0.97 for v in ensemble_y]
                band_name = "±3% indicative range"
            fig.add_trace(
                go.Scatter(
                    x=list(forecast_ts) + list(forecast_ts[::-1]),
                    y=upper_y + lower_y[::-1],
                    fill="toself",
                    fillcolor="rgba(249, 115, 22, 0.12)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=band_name,
                )
            )

            # Forecast line bridged from the last actual point — gives a
            # visually continuous transition. The bridge segment uses the
            # last actual MW; the forward segment uses the real ensemble
            # predictions per their Redis-stored timestamps.
            bridge_x = [last_ts, *forecast_ts]
            bridge_y = [last_mw, *ensemble_y]
            fig.add_trace(
                go.Scatter(
                    x=bridge_x,
                    y=bridge_y,
                    mode="lines",
                    name="Forecast (24h)",
                    line=dict(color="#f97316", width=1.75, dash="dash"),
                    hovertemplate=(
                        "<b>%{x|%b %d, %H:%M}</b><br>%{y:,.0f} MW · forecast<extra></extra>"
                    ),
                )
            )

            # Surface the payload's own scored_at instead of discarding it —
            # a stale forecast must not render as the current outlook
            # (2026-07 critical-review finding P1-4).
            if scored_at_iso:
                scored_dt = pd.Timestamp(scored_at_iso)
                if scored_dt.tzinfo is None:
                    scored_dt = scored_dt.tz_localize("UTC")
                age_h = (pd.Timestamp.now(tz="UTC") - scored_dt).total_seconds() / 3600.0
                is_stale = age_h > FRESHNESS_FRESH_MAX_AGE_HOURS
                note = f"forecast scored {scored_dt.strftime('%b %d %H:%M')} UTC"
                if is_stale:
                    note += f" · {age_h:.0f}h ago — stale"
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=1.06,
                    xanchor="right",
                    showarrow=False,
                    text=note,
                    font=dict(size=10, color="#FFB84D" if is_stale else "#71717a"),
                )
    except Exception as exc:  # pragma: no cover — fall back to actual-only chart
        log.warning("overview_hero_forecast_failed", region=region, error=str(exc))

    fig.update_layout(
        **_layout(
            uirevision=region,
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(color="#71717a", size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                zeroline=False,
                tickformat=",.0f",
                tickfont=dict(color="#71717a", size=10),
                title=None,
            ),
        ),
    )
    return fig


def _build_overview_model_card(region: str) -> html.Div:
    """Horizontal model-performance bar (top/bottom borders only)."""
    try:
        from models.model_service import get_model_metrics, is_trained
    except ImportError:  # pragma: no cover — defensive
        return html.Div()

    metrics_dict = get_model_metrics(region)
    if not metrics_dict:
        return html.Div()

    # Prefer ensemble; fall back to xgboost; finally first available
    if "ensemble" in metrics_dict:
        primary_key = "ensemble"
    elif "xgboost" in metrics_dict:
        primary_key = "xgboost"
    else:
        primary_key = next(iter(metrics_dict.keys()), None)
    if primary_key is None:
        return html.Div()

    m = metrics_dict[primary_key]

    def _fmt(key: str, spec: str, suffix: str = "") -> str:
        # An absent metric must render as unavailable, not as a perfect score.
        value = m.get(key)
        if value is None:
            return "—"
        return f"{value:{spec}}{suffix}"

    formatted = {
        "MAPE": _fmt("mape", ".1f", "%"),
        "RMSE": _fmt("rmse", ",.0f", " MW"),
        "MAE": _fmt("mae", ",.0f", " MW"),
        "R²": _fmt("r2", ".3f"),
    }
    name = "XGBoost" if primary_key == "xgboost" else primary_key.title()
    badge = "trained" if is_trained(region) else "simulated"
    return build_model_metrics_card(model_name=name, metrics=formatted, badge=badge)


def _build_overview_insight(
    region: str,
    demand_df: pd.DataFrame | None,
    persona_id: str,
) -> html.Div:
    """3-sentence narrative paragraph with semantic-color delta spans."""
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return build_insight_card(
            "Summary",
            (
                "Awaiting demand data for this region. The forecast will populate once "
                "the next pipeline cycle completes."
            ),
        )

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    nonzero = df[df["demand_mw"] > 0]
    last_7d = nonzero.tail(168)

    # Read from ``nonzero`` so a NaN tail (EIA-930 publishing lag) doesn't
    # poison the narrative with literal "nan%" deltas.
    now_value = _latest_real_demand(nonzero["demand_mw"]) or 0.0
    avg_7d = float(last_7d["demand_mw"].mean()) if not last_7d.empty else 0.0
    delta_pct = ((now_value - avg_7d) / avg_7d * 100.0) if avg_7d else 0.0
    direction = "above" if delta_pct >= 0 else "below"
    # Inverted semantic: rising demand reads as warning (matches v2)
    delta_class = (
        "gp-insight-card__delta--negative" if delta_pct >= 0 else "gp-insight-card__delta--positive"
    )

    last_24h = df.tail(24)
    if not last_24h.empty:
        peak_idx = last_24h["demand_mw"].idxmax()
        peak_mw = float(last_24h.loc[peak_idx, "demand_mw"])
        peak_ts = pd.to_datetime(last_24h.loc[peak_idx, "timestamp"])
        peak_str = f"{peak_mw:,.0f} MW at {peak_ts.strftime('%H:%M')}"
    else:
        peak_str = "—"

    # Forecast clause — read from the same Redis payload the hero chart
    # uses. Drop the clause entirely when Redis is cold (warming state)
    # instead of fabricating it from the simulated baseline. Same fix
    # arc as the hero chart above; see comment there for the prior-bug
    # context.
    forecast_clause = "Next-cycle forecast confidence is updating."
    try:
        forecast_payload = _read_ensemble_forecast_from_redis(region)
        if forecast_payload is not None:
            forecast_ts_all, ensemble_arr, scored_at_iso = forecast_payload
            horizon = min(24, len(ensemble_arr))
            if horizon > 0:
                f_arr = ensemble_arr[:horizon]
                f_peak = float(f_arr.max())
                f_peak_idx = int(f_arr.argmax())
                # Use the REAL timestamp from the Redis payload, not a
                # computed offset off last_actual. The previous code
                # computed ``last_ts + (f_peak_idx + 1)h`` which is
                # meaningless against the simulated baseline (the
                # ensemble array represented historical hours, not
                # forward) and produced bogus peak times like "04:00".
                f_peak_ts = forecast_ts_all[f_peak_idx]

                # MAPE for this forecast (#4 — 2026-05-20).
                #
                # Pre-fix this cited ``get_model_metrics`` which returns
                # training-time HOLDOUT MAPE — the model's MAPE on its
                # validation slice from yesterday's training run. That's
                # not "how this specific 24h forecast is likely to do" —
                # it's "how the model did on a frozen slice last night."
                # Users reading "MAPE 1.6%" reasonably assumed the
                # forecast itself was expected to be 1.6% off. Fluff.
                #
                # Honest replacement: LIVE rolling 7d MAPE from
                # ``gridpulse:drift:{region}`` (#121 part 1, PR #126).
                # This is computed by comparing every previous tick's
                # 1-hour-ahead forecast against the now-known actual,
                # rolled over the last 7 days. It tells the user how
                # the model has actually been performing on real
                # forecasts of similar horizon.
                #
                # Fall-back order:
                #   1. Live 7d MAPE from drift (real, calibrated to
                #      recent reality)
                #   2. Live 30d MAPE from drift (more samples, slightly
                #      staler — used only when 7d has too few records)
                #   3. Training holdout MAPE (clearly labeled as such)
                #   4. No MAPE clause at all (when nothing is available)
                mape_value, mape_source = _resolve_forecast_mape(region)

                mape_clause = ""
                if mape_value is not None:
                    mape_clause = f" ({mape_source} {mape_value:.1f}%)"

                # A stale forecast must not narrate as the current outlook
                # (P1-4): disclose the scoring age once it exceeds the
                # missed-tick tolerance.
                stale_clause = ""
                if scored_at_iso:
                    scored_dt = pd.Timestamp(scored_at_iso)
                    if scored_dt.tzinfo is None:
                        scored_dt = scored_dt.tz_localize("UTC")
                    age_h = (pd.Timestamp.now(tz="UTC") - scored_dt).total_seconds() / 3600.0
                    if age_h > FRESHNESS_FRESH_MAX_AGE_HOURS:
                        stale_clause = f" Forecast last refreshed {age_h:.0f}h ago."

                forecast_clause = (
                    f"Next-24h forecast peaks at {f_peak:,.0f} MW around "
                    f"{f_peak_ts.strftime('%H:%M')} UTC{mape_clause}.{stale_clause}"
                )
    except Exception as exc:  # pragma: no cover
        log.warning("overview_insight_forecast_failed", region=region, error=str(exc))

    # #8 (2026-05-20): relabel "Recent peak" → "Last 24h peak" so the
    # window is explicit and consistent with the "7d Peak" cell in the
    # metrics bar above. The previous label was ambiguous about whether
    # "recent" meant 24h or the current 7d-peak window.
    body = [
        "Demand is ",
        html.Span(f"{abs(delta_pct):.1f}% {direction}", className=delta_class),
        " the 7-day average. ",
        "Last 24h peak: ",
        html.Span(peak_str, className="gp-insight-card__strong"),
        ". ",
        forecast_clause,
    ]
    # Persona influences eyebrow only — keeps the card tonally consistent
    eyebrow_map = {
        "grid_ops": "Operating summary",
        "renewables": "Renewables outlook",
        "trader": "Market signal",
        "data_scientist": "Model summary",
    }
    eyebrow = eyebrow_map.get(persona_id, "Summary")
    return build_insight_card(eyebrow, body)


# ── Overview panels block (Step 7b — drivers / generation / models / risk / scenarios) ──


def _generation_df_from_redis(region: str) -> pd.DataFrame | None:
    """Read the scoring job's ``gridpulse:generation:{region}`` payload and
    unpivot it to the long ``[timestamp, fuel_type, generation_mw, region]``
    frame the Generation panel expects.

    The scoring job writes a wide payload (``{timestamps, <fuel>: [...],
    renewable_pct: [...]}``, fuel names already normalized); this reverses that
    pivot so the web tier can render generation without touching EIA (#199).
    """
    payload = redis_get(redis_key(f"generation:{region}"))
    if not isinstance(payload, dict):
        return None
    timestamps = payload.get("timestamps")
    if not timestamps:
        return None
    skip = {"region", "timestamps", "renewable_pct", "scored_at"}
    rows: list[dict] = []
    for fuel, vals in payload.items():
        if fuel in skip or not isinstance(vals, list):
            continue
        for ts, mw in zip(timestamps, vals, strict=False):
            rows.append({"timestamp": ts, "fuel_type": fuel, "generation_mw": mw, "region": region})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_generation_cached(region: str) -> pd.DataFrame | None:
    """Return generation-by-fuel for a region, Redis-first.

    The stateless web tier must not fetch EIA in the request path — the scoring
    job writes ``gridpulse:generation:{region}`` hourly, and this reads it
    (#199 / the CLAUDE.md post-#130 web-tier I/O guardrail). Under
    ``REQUIRE_REDIS`` (staging/prod) a Redis miss returns None (warming state),
    never a live EIA call. The in-memory + EIA fetch tiers run only in
    development, where the scoring job may not be populating Redis.

    Returns a DataFrame ``[timestamp, fuel_type, generation_mw, region]`` or None.
    """
    import time as _time

    # Redis fast path (the only prod path).
    redis_df = _generation_df_from_redis(region)
    if redis_df is not None and not redis_df.empty:
        return redis_df

    if REQUIRE_REDIS:
        log.info("generation_warming", region=region)
        return None

    # ── development-only fallback (no scoring job populating Redis) ──
    # Tier 1: In-memory cache (5-minute TTL)
    if region in _GENERATION_CACHE:
        cached_df, cached_ts = _GENERATION_CACHE[region]
        if (_time.time() - cached_ts) < 300:
            log.info("generation_memory_cache_hit", region=region)
            return cached_df

    # Tier 2+3: fetch_generation_by_fuel handles SQLite cache + API call
    try:
        from config import EIA_API_KEY

        if EIA_API_KEY and EIA_API_KEY != "your_eia_api_key_here":
            from data.eia_client import fetch_generation_by_fuel

            gen_df = fetch_generation_by_fuel(region)
            if gen_df is not None and not gen_df.empty:
                # Normalize fuel type codes
                gen_df["fuel_type"] = (
                    gen_df["fuel_type"].map(_EIA_FUEL_MAP).fillna(gen_df["fuel_type"].str.lower())
                )
                _GENERATION_CACHE[region] = (gen_df, _time.time())
                log.info("generation_eia_fetched", region=region, rows=len(gen_df))
                return gen_df
    except Exception as e:
        log.warning("generation_eia_failed", region=region, error=str(e))

    # No demo fallback — return None so callers show "No data" or use
    # whatever is already in Redis rather than overwriting with fake values.
    log.warning("generation_no_data", region=region)
    return None


def _build_drivers_panel(weather_json: str | None) -> list:
    """3-up KPI cells (Temperature / Wind / Solar) with current value + 24h sparkline.

    The Forecast tab's Drivers inline panel calls this when its collapse
    opens. Each cell is a .gp-driver-cell with eyebrow / value / unit /
    sparkline. Sparkline reuses the same v2 minimal-axis style as
    _build_overview_sparkline.
    """
    if not weather_json:
        return _drivers_empty()

    try:
        wdf = pd.read_json(io.StringIO(weather_json))
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("forecast_drivers_parse_failed", error=str(exc))
        return _drivers_empty()

    if wdf.empty or "timestamp" not in wdf.columns:
        return _drivers_empty()

    wdf = wdf.copy()
    wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
    wdf = wdf.sort_values("timestamp")
    # Window: latest 24 rows (assume hourly cadence)
    horizon = wdf.tail(24)

    drivers = [
        {
            "label": "Temperature",
            "column": "temperature_2m",
            "unit": "°F",
            "color": "#3b82f6",
            "fillcolor": "rgba(59, 130, 246, 0.10)",
            "fmt": lambda v: f"{v:.0f}",
        },
        {
            "label": "Wind",
            "column": "wind_speed_80m",
            "unit": "mph",
            "color": "#34d399",
            "fillcolor": "rgba(52, 211, 153, 0.10)",
            "fmt": lambda v: f"{v:.1f}",
        },
        {
            "label": "Solar",
            "column": "shortwave_radiation",
            "unit": "W/m²",
            "color": "#f97316",
            "fillcolor": "rgba(249, 115, 22, 0.10)",
            "fmt": lambda v: f"{v:.0f}",
        },
    ]

    cells: list = []
    for d in drivers:
        col = d["column"]
        if col not in horizon.columns or horizon[col].isna().all():
            cells.append(_driver_cell_empty(d["label"]))
            continue
        latest = float(horizon[col].iloc[-1])
        avg = float(horizon[col].mean())
        delta = latest - avg
        delta_class = (
            "gp-metric-value--negative"
            if delta > 0.5
            else ("gp-metric-value--positive" if delta < -0.5 else "")
        )
        cells.append(
            html.Div(
                [
                    html.Div(d["label"], className="gp-metric-label"),
                    html.Div(
                        [
                            html.Span(
                                d["fmt"](latest),
                                className="gp-metric-value gp-metric-value--hero tabular",
                            ),
                            html.Span(d["unit"], className="gp-metric-unit"),
                        ],
                        className="gp-metric-value-row",
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{delta:+.1f} vs 24h avg",
                                className=f"gp-metric-sub {delta_class}",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        figure=_driver_sparkline(horizon, col, d["color"], d["fillcolor"]),
                        config={"displayModeBar": False, "responsive": True},
                        style={"height": "60px"},
                    ),
                ],
                className="gp-driver-cell",
            )
        )
    return cells


def _drivers_empty() -> list:
    cells = []
    for label in ("Temperature", "Wind", "Solar"):
        cells.append(_driver_cell_empty(label))
    return cells


def _driver_cell_empty(label: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="gp-metric-label"),
            html.Span("—", className="gp-metric-value tabular"),
            html.Div("No weather data", className="gp-metric-sub"),
        ],
        className="gp-driver-cell",
    )


# Fuel ordering: heaviest emissions at the bottom of the stack, zero-carbon
# on top. Within each bucket: dispatchable before intermittent.
_FUEL_STACK_ORDER: tuple[str, ...] = (
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

_FUEL_DISPLAY: dict[str, dict[str, str]] = {
    "coal": {"label": "Coal", "color": "#71717a", "fill": "rgba(113, 113, 122, 0.85)"},
    "oil": {"label": "Oil", "color": "#52525b", "fill": "rgba(82, 82, 91, 0.85)"},
    "gas": {"label": "Gas", "color": "#f97316", "fill": "rgba(249, 115, 22, 0.85)"},
    "biomass": {"label": "Biomass", "color": "#a16207", "fill": "rgba(161, 98, 7, 0.85)"},
    "other": {"label": "Other", "color": "#a1a1aa", "fill": "rgba(161, 161, 170, 0.85)"},
    "nuclear": {"label": "Nuclear", "color": "#a855f7", "fill": "rgba(168, 85, 247, 0.85)"},
    "hydro": {"label": "Hydro", "color": "#3b82f6", "fill": "rgba(59, 130, 246, 0.85)"},
    "wind": {"label": "Wind", "color": "#34d399", "fill": "rgba(52, 211, 153, 0.85)"},
    "solar": {"label": "Solar", "color": "#fbbf24", "fill": "rgba(251, 191, 36, 0.85)"},
}


def _build_generation_panel(region: str | None, demand_json: str | None) -> html.Div:
    """Stacked-area fuel mix + 3-up sub-MetricsBar (Net Load / Renewable / Largest)."""
    region = region or "FPL"

    try:
        gen_df = _fetch_generation_cached(region)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("forecast_generation_fetch_failed", region=region, error=str(exc))
        return _generation_empty()

    if gen_df is None or gen_df.empty:
        return _generation_empty()

    df = gen_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Window: latest 24 hours
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=24)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return _generation_empty()

    pivot = (
        df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        )
        .fillna(0)
        .clip(lower=0)
    )

    # Sort columns by emissions order (any unknown fuels go to the end)
    ordered_fuels = [f for f in _FUEL_STACK_ORDER if f in pivot.columns]
    extras = [f for f in pivot.columns if f not in _FUEL_STACK_ORDER]
    pivot = pivot[ordered_fuels + extras]

    # ── KPIs ───────────────────────────────────────────────────
    total_per_ts = pivot.sum(axis=1)
    avg_total = float(total_per_ts.mean()) if not total_per_ts.empty else 0.0

    fuel_avg = pivot.mean(axis=0).sort_values(ascending=False)
    largest_fuel = fuel_avg.index[0] if len(fuel_avg) else None
    largest_label = _FUEL_DISPLAY.get(str(largest_fuel), {}).get(
        "label", str(largest_fuel).title() if largest_fuel else "—"
    )
    largest_share_pct = (
        float(fuel_avg.iloc[0] / fuel_avg.sum() * 100.0)
        if len(fuel_avg) and fuel_avg.sum() > 0
        else 0.0
    )

    renewable_cols = [c for c in ("wind", "solar", "hydro") if c in pivot.columns]
    if renewable_cols and avg_total > 0:
        renewable_pct = float((pivot[renewable_cols].sum(axis=1) / total_per_ts * 100.0).mean())
    else:
        renewable_pct = 0.0

    # Net load (Demand - Wind - Solar) if demand available
    net_load_avg = avg_total
    if demand_json:
        try:
            ddf = pd.read_json(io.StringIO(demand_json))
            ddf["timestamp"] = pd.to_datetime(ddf["timestamp"])
            ddf = ddf.sort_values("timestamp")
            common = pivot.index.intersection(ddf.set_index("timestamp").index)
            if len(common) >= 2:
                d_aligned = ddf.set_index("timestamp").loc[common, "demand_mw"]
                wind_aligned = pivot.loc[common].get("wind", pd.Series(0.0, index=common))
                solar_aligned = pivot.loc[common].get("solar", pd.Series(0.0, index=common))
                net_load_series = d_aligned - wind_aligned - solar_aligned
                net_load_avg = float(net_load_series.mean())
        except Exception as exc:  # pragma: no cover
            log.warning("forecast_generation_netload_failed", region=region, error=str(exc))

    sub_metrics = build_metrics_bar(
        [
            {
                "label": "Net Load (avg)",
                "value": f"{net_load_avg:,.0f}",
                "unit": "MW",
                "hero": True,
            },
            {
                "label": "Renewable Share",
                "value": f"{renewable_pct:.1f}%",
                "tone": "positive" if renewable_pct >= 25 else "secondary",
            },
            {
                "label": "Largest Source",
                "value": largest_label,
                "unit": f"{largest_share_pct:.0f}%",
                "tone": "secondary",
            },
        ]
    )
    # Override the default 5-up class for a 3-up grid.
    sub_metrics.className = "gp-metrics-bar gp-metrics-bar--3up"

    # ── Stacked-area chart ─────────────────────────────────────
    fig = go.Figure()
    for fuel in pivot.columns:
        cfg = _FUEL_DISPLAY.get(
            str(fuel),
            {
                "label": str(fuel).title(),
                "color": "#a1a1aa",
                "fill": "rgba(161,161,170,0.7)",
            },
        )
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[fuel],
                mode="lines",
                stackgroup="gen",
                name=cfg["label"],
                line=dict(width=0, color=cfg["color"]),
                fillcolor=cfg["fill"],
                hovertemplate=(
                    f"<b>{cfg['label']}</b><br>%{{x|%H:%M}}<br>%{{y:,.0f}} MW<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        **_layout(
            uirevision=f"gen-{region}",
            showlegend=True,
            xaxis=dict(
                showgrid=False,
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(color="#71717a", size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                zeroline=False,
                tickformat=",.0f",
                tickfont=dict(color="#71717a", size=10),
                title=None,
            ),
            margin=dict(l=48, r=16, t=16, b=64),
        ),
    )

    return html.Div(
        [
            sub_metrics,
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "320px"},
            ),
        ],
        className="gp-generation-stack",
    )


def _generation_empty() -> html.Div:
    return html.Div(
        "No generation data available for this region.",
        className="gp-panel__placeholder",
    )


def _build_models_leaderboard(region: str | None) -> html.Div:
    """5-up MetricsBar leaderboard — Prophet / SARIMAX / XGBoost / Ensemble / EIA.

    Hero highlight goes to the model with the lowest MAPE. Sub-line shows
    MAE; tone reflects the MAPE grade band (positive ≤ 2.5%, secondary
    ≤ 5%, negative > 5%).
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
    display_names = {
        "prophet": "Prophet",
        "arima": "SARIMAX",
        "xgboost": "XGBoost",
        "ensemble": "Ensemble",
        "eia": "EIA Reference",
    }

    # Hero pick: model with lowest mape
    valid = [(k, v) for k, v in metrics_dict.items() if isinstance(v, dict) and "mape" in v]
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
                    "label": display_names.get(key, key.title()),
                    "value": "—",
                    "unit": "metrics unavailable",
                    "tone": "secondary",
                    "hero": key == hero_key,
                }
            )
            continue
        mape = float(m["mape"])
        mae = m.get("mae")
        if mape <= 2.5:
            tone = "positive"
        elif mape <= 5.0:
            tone = "secondary"
        else:
            tone = "negative"
        items.append(
            {
                "label": display_names.get(key, key.title()),
                "value": f"{mape:.1f}%",
                "unit": f"MAE {float(mae):,.0f}" if mae is not None else "",
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


def _build_risk_insight(
    region: str | None,
    demand_json: str | None,
    weather_json: str | None,
) -> html.Div:
    """3-sentence narrative for the Risk tab — composes the same fragments
    the alerts/stress callback already computes, just rendered as prose."""
    region = region or "FPL"

    # Demand-anomaly stat: pct of last-24h hours outside ±3σ
    anomaly_clause = "Demand sits within normal bounds."
    try:
        if demand_json:
            ddf = pd.read_json(io.StringIO(demand_json))
            ddf["timestamp"] = pd.to_datetime(ddf["timestamp"])
            ddf = ddf.sort_values("timestamp")
            last_24 = ddf.tail(24)["demand_mw"].dropna()
            if len(last_24) >= 6:
                m = float(last_24.mean())
                s = float(last_24.std()) or 1.0
                outliers = int(((last_24 - m).abs() > 2.5 * s).sum())
                if outliers > 0:
                    anomaly_clause = (
                        f"{outliers} hour{'s' if outliers > 1 else ''} of demand sat "
                        "outside ±2.5σ in the last 24h."
                    )
    except Exception as exc:  # pragma: no cover
        log.warning("risk_insight_anomaly_failed", region=region, error=str(exc))

    # Weather-severity stat: |temperature_2m − 65| ≥ 25 (peak heat / cold band)
    weather_clause = "Weather is in a comfortable band."
    try:
        if weather_json:
            wdf = pd.read_json(io.StringIO(weather_json))
            wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
            wdf = wdf.sort_values("timestamp")
            recent = wdf.tail(24)
            if "temperature_2m" in recent.columns and not recent["temperature_2m"].isna().all():
                temps = recent["temperature_2m"].dropna()
                t_max = float(temps.max())
                t_min = float(temps.min())
                if t_max >= 90:
                    weather_clause = f"Heat-driven demand risk is elevated (peak {t_max:.0f} °F)."
                elif t_min <= 30:
                    weather_clause = f"Cold-driven demand risk is elevated (low {t_min:.0f} °F)."
                elif t_max >= 80:
                    weather_clause = f"Temperatures trending warm (peak {t_max:.0f} °F)."
                elif t_min <= 40:
                    weather_clause = f"Temperatures trending cool (low {t_min:.0f} °F)."
    except Exception as exc:  # pragma: no cover
        log.warning("risk_insight_weather_failed", region=region, error=str(exc))

    body = [
        anomaly_clause,
        " ",
        weather_clause,
        " Check the timeline above for active alerts.",
    ]
    return build_insight_card("Risk summary", body)


def _scenario_demand_factor(temp_delta: float, wind_delta: float, solar_delta: float) -> float:
    """Linear demand-sensitivity factor for the scenario simulator heuristic.

    Returns a multiplicative factor to apply to a baseline 24h forecast.
    Coefficients are order-of-magnitude-defensible against load-research
    norms (not physically rigorous — full-fidelity physics lives in
    ``simulation/scenario_engine.py``):

      * temp_delta: ±2.5 % per 5 °F (existing — dominant driver)
      * solar_delta: +1.5 % per 100 W/m² (sun load → AC demand;
        meaningful for summer-peaking BAs like FPL/ERCOT/PJM)
      * wind_delta: +0.5 % per 10 mph (wind chill → heating demand;
        meaningful for winter-peaking BAs)

    All three combine linearly. Pulled out as a pure function so the
    heuristic is unit-testable without spinning up the Plotly render.
    """
    return 1.0 + (temp_delta / 5.0) * 0.025 + solar_delta * 0.00015 + wind_delta * 0.0005


def _build_scenarios_panel(
    temp_delta: int | float | None,
    wind_delta: int | float | None,
    solar_delta: int | float | None,
    region: str | None,
    demand_json: str | None,
) -> tuple[html.Div, go.Figure]:
    """Heuristic scenario impact + baseline-vs-scenario comparison chart.

    Returns ``(kpi_bar, figure)``. The math is a deliberate simplification:
    no model re-run, just a linear demand-sensitivity factor against the
    current 24h forecast. Real ensemble simulation lives in the (now hidden)
    Scenarios tab and the simulation/scenario_engine module — exposing
    full-fidelity here would need model loading on every slider drag.

    Demand sensitivities — see ``_scenario_demand_factor`` for coefficients.

    Renewable-share sensitivities (independent of demand):
      * wind_delta: ±0.6 pp per mph (caps at 30 pp)
      * solar_delta: ±0.05 pp per W/m² (caps at 30 pp)

    Confidence sensitivity: −1 pp per 5 °F of |temp_delta|, capped at −10 pp
    (forecast residuals grow with abs(temp_delta) outside ±10 °F).
    """
    region = region or "FPL"
    temp_delta = float(temp_delta or 0)
    wind_delta = float(wind_delta or 0)
    solar_delta = float(solar_delta or 0)

    # Base forecast (next 24h ensemble)
    horizon = 24
    base_y: np.ndarray | None = None
    last_actual_ts: pd.Timestamp | None = None

    if demand_json:
        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            demand_df = demand_df.sort_values("timestamp")

            # Baseline = the real scored ensemble from Redis (the scoring job's
            # own output), not model_service.get_forecasts — which on the
            # stateless web tier is strict-gated to "unavailable" in prod
            # (#149) and echoed actuals as a fake forecast in dev when only
            # "ensemble" is requested (2026-07 review P2-31). This is the same
            # reader the Overview hero uses.
            forecast_payload = _read_ensemble_forecast_from_redis(region)
            if forecast_payload is not None:
                _fc_ts, ensemble_arr, _scored_at = forecast_payload
                if ensemble_arr is not None and len(ensemble_arr) >= horizon:
                    base_y = np.asarray(ensemble_arr[:horizon], dtype=float)
                    last_actual_ts = demand_df["timestamp"].iloc[-1]
        except Exception as exc:  # pragma: no cover
            log.warning("forecast_scenario_baseline_failed", region=region, error=str(exc))

    if base_y is None or last_actual_ts is None:
        kpi_empty = build_metrics_bar(
            [
                {"label": "Δ Peak", "value": "—", "tone": "secondary", "hero": True},
                {"label": "Δ Reserve", "value": "—", "tone": "secondary"},
                {"label": "Δ Renewable", "value": "—", "tone": "secondary"},
                {"label": "Δ Confidence", "value": "—", "tone": "secondary"},
            ]
        )
        kpi_empty.className = "gp-metrics-bar gp-metrics-bar--4up"
        return (kpi_empty, _empty_figure("Awaiting baseline forecast"))

    # ── Heuristic scenario forecast ────────────────────────────
    # Demand response is dominated by temperature, plus smaller terms
    # for solar (AC load) and wind (wind chill). See
    # ``_scenario_demand_factor`` for the coefficient rationale.
    demand_factor = _scenario_demand_factor(temp_delta, wind_delta, solar_delta)
    scenario_y = base_y * demand_factor

    base_peak = float(np.max(base_y))
    scenario_peak = float(np.max(scenario_y))
    delta_peak_mw = scenario_peak - base_peak
    delta_peak_pct = (delta_peak_mw / base_peak * 100.0) if base_peak > 0 else 0.0

    capacity = REGION_CAPACITY_MW.get(region, 100_000)
    base_reserve = (capacity - base_peak) / capacity * 100.0
    scenario_reserve = (capacity - scenario_peak) / capacity * 100.0
    delta_reserve_pp = scenario_reserve - base_reserve

    # Renewable share heuristic — wind: 0.6 %/mph; solar: 0.05 %/(W/m²)
    delta_renewable_pp = wind_delta * 0.6 + solar_delta * 0.05
    delta_renewable_pp = max(min(delta_renewable_pp, 30.0), -30.0)

    # Confidence delta: bigger temp swings widen the band roughly linearly
    # (forecast residuals grow with abs(temp_delta) outside ±10°F band).
    delta_confidence_pp = -min(abs(temp_delta) / 5.0, 10.0)  # negative pp

    # ── KPI bar ────────────────────────────────────────────────
    peak_tone = (
        "negative"
        if delta_peak_pct > 0.5
        else ("positive" if delta_peak_pct < -0.5 else "secondary")
    )
    reserve_tone = (
        "positive"
        if delta_reserve_pp > 0.1
        else ("negative" if delta_reserve_pp < -0.1 else "secondary")
    )
    renewable_tone = (
        "positive"
        if delta_renewable_pp > 0.5
        else ("negative" if delta_renewable_pp < -0.5 else "secondary")
    )
    kpis = build_metrics_bar(
        [
            {
                "label": "Δ Peak",
                "value": f"{delta_peak_mw:+,.0f}",
                "unit": f"MW ({delta_peak_pct:+.1f}%)",
                "tone": peak_tone,
                "hero": True,
            },
            {
                "label": "Δ Reserve",
                "value": f"{delta_reserve_pp:+.1f}",
                "unit": "pp",
                "tone": reserve_tone,
            },
            {
                "label": "Δ Renewable",
                "value": f"{delta_renewable_pp:+.1f}",
                "unit": "pp",
                "tone": renewable_tone,
            },
            {
                "label": "Δ Confidence",
                "value": f"{delta_confidence_pp:+.1f}",
                "unit": "pp",
                "tone": "secondary",
            },
        ]
    )
    kpis.className = "gp-metrics-bar gp-metrics-bar--4up"

    # ── Baseline vs scenario chart ─────────────────────────────
    forecast_ts = pd.date_range(
        start=last_actual_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_ts,
            y=base_y,
            mode="lines",
            name="Baseline",
            line=dict(color="#3b82f6", width=1.75),
            hovertemplate="<b>Baseline</b><br>%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_ts,
            y=scenario_y,
            mode="lines",
            name="Scenario",
            line=dict(color="#f97316", width=1.75, dash="dash"),
            hovertemplate="<b>Scenario</b><br>%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            uirevision=f"scn-{region}",
            xaxis=dict(
                showgrid=False,
                linecolor="rgba(255,255,255,0.04)",
                tickfont=dict(color="#71717a", size=10),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                zeroline=False,
                tickformat=",.0f",
                tickfont=dict(color="#71717a", size=10),
                title=None,
            ),
            margin=dict(l=48, r=16, t=16, b=36),
            showlegend=True,
        ),
    )
    return kpis, fig


def _driver_sparkline(df: pd.DataFrame, column: str, color: str, fillcolor: str) -> go.Figure:
    """60px sparkline matching the v2 minimal-axes treatment."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=fillcolor,
            hovertemplate="%{x|%H:%M}<br>%{y:,.1f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        **_layout(
            uirevision=column,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=4, b=4),
        )
    )
    return fig


# ── Overview briefing block (Step 7c — sparklines / briefing / digest / spotlights / persona) ──


def _build_overview_sparkline(demand_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Build a compact 24h demand sparkline for the overview tab."""
    if demand_df is None or demand_df.empty or "demand_mw" not in demand_df.columns:
        return _empty_figure("No demand data")

    df = demand_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_24h = df.tail(24)

    if last_24h.empty:
        return _empty_figure("No recent demand data")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_24h["timestamp"],
            y=last_24h["demand_mw"],
            mode="lines",
            line=dict(color=CB_PALETTE["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,114,178,0.15)",
            name="Demand",
            hovertemplate="%{x|%H:%M}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    sparkline_layout = _layout(
        uirevision=region,
        showlegend=False,
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(
            showgrid=False,
            tickformat="%H:%M",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat=",.0f",
            title="MW",
        ),
    )
    fig.update_layout(**sparkline_layout)
    return fig


def _build_overview_briefing(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> html.Div:
    """Build the AI executive briefing section."""
    from data.ai_briefing import generate_briefing

    try:
        result = generate_briefing(persona_id, region, demand_df, weather_df)
    except Exception as exc:
        log.error("overview_briefing_failed", error=str(exc))
        return html.Div(
            "Briefing unavailable",
            style={"color": "#A8B3C7", "fontStyle": "italic"},
        )

    persona = get_persona(persona_id)

    children = [
        html.P(
            result.summary,
            style={
                "color": "#DDE6F2",
                "fontSize": "0.9rem",
                "lineHeight": "1.6",
                "marginBottom": "12px",
            },
        ),
    ]

    if result.observations:
        obs_items = []
        for obs in result.observations:
            obs_items.append(
                html.Li(
                    obs,
                    style={
                        "color": "#A8B3C7",
                        "fontSize": "0.82rem",
                        "marginBottom": "4px",
                        "lineHeight": "1.5",
                    },
                )
            )
        children.append(
            html.Ul(
                obs_items,
                style={"paddingLeft": "20px", "marginBottom": "8px"},
            )
        )

    source_label = "AI Analysis" if result.source == "claude" else "Data Summary"
    children.append(
        html.Span(
            source_label,
            style={
                "fontSize": "0.65rem",
                "color": "#A8B3C7",
                "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            },
        )
    )

    return html.Div(
        children,
        style={"borderLeft": f"4px solid {persona.color}"},
        className="briefing-card-content",
    )


def _build_weather_context(latest: pd.Series) -> html.Div:
    """Build a row of weather KPI mini-cards from the latest weather reading."""
    temp = latest.get("temperature_2m")
    wind = latest.get("wind_speed_80m", latest.get("wind_speed_10m"))
    humidity = latest.get("relative_humidity_2m")
    cloud = latest.get("cloud_cover")

    cards = []

    if temp is not None:
        t = float(temp)
        color = "#FF5C7A" if t >= 95 else ("#FFB84D" if t >= 85 else "#2BD67B")
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("TEMPERATURE", className="kpi-label"),
                        html.H4(
                            f"{t:.0f}°F",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if wind is not None:
        w = float(wind)
        color = "#FF5C7A" if w >= 40 else ("#FFB84D" if w >= 25 else "#2BD67B")
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("WIND SPEED", className="kpi-label"),
                        html.H4(
                            f"{w:.0f} mph",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if humidity is not None:
        h = float(humidity)
        color = "#FFB84D" if h >= 80 else "#2BD67B"
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("HUMIDITY", className="kpi-label"),
                        html.H4(
                            f"{h:.0f}%",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if cloud is not None:
        c = float(cloud)
        color = "#A8B3C7"
        cards.append(
            dbc.Col(
                html.Div(
                    [
                        html.P("CLOUD COVER", className="kpi-label"),
                        html.H4(
                            f"{c:.0f}%",
                            className="kpi-value",
                            style={"fontSize": "1.3rem"},
                        ),
                    ],
                    className="kpi-card",
                    style={"borderTop": f"3px solid {color}"},
                ),
                md=3,
            )
        )

    if not cards:
        return html.Div()

    return dbc.Row(cards, className="g-2")


def _build_changes_card(
    changes_json: str | None,
    persona: str,
    region: str,
    snapshots: dict | None,
) -> html.Div:
    """Build the 'Since your last visit' card for the Overview tab (NEXD-8)."""
    import json as _json

    from data.session_diff import format_relative_time

    if not changes_json:
        return html.Div()

    try:
        changes = _json.loads(changes_json) if isinstance(changes_json, str) else changes_json
    except Exception:
        return html.Div()

    if not changes:
        return html.Div()

    # Get previous visit timestamp for this region
    prev_timestamp = None
    if snapshots and isinstance(snapshots, dict):
        entry = snapshots.get(region)
        if entry and isinstance(entry, dict):
            prev_timestamp = entry.get("timestamp")

    rel_time = format_relative_time(prev_timestamp) if prev_timestamp else ""

    # Build bullet items
    items = []
    for change in changes[:5]:
        icon = change.get("icon", "")
        text = change.get("text", "")
        items.append(
            html.Div(
                [
                    html.Span(icon, className="change-icon"),
                    html.Span(text),
                ],
                className="change-item",
            )
        )

    header_parts = [
        html.Span("Since your last visit", className="changes-title"),
    ]
    if rel_time:
        header_parts.append(html.Span(rel_time, className="changes-timestamp"))

    return html.Div(
        [
            html.Div(header_parts, className="changes-header"),
            *items,
        ],
        className="changes-card",
    )


def _build_overview_data_health(freshness_data: dict | None) -> html.Div:
    """Build data health badges showing per-source freshness."""
    if not freshness_data:
        return html.Div()

    source_config = {
        "demand": {"label": "EIA Demand", "icon": "⚡"},
        "weather": {"label": "Weather", "icon": "☁"},
        # No live alert feed is wired yet (noaa_client has no scoring-path
        # caller) — do not attribute the alerts payload to NOAA.
        "alerts": {"label": "Alerts", "icon": "⚠"},
    }

    status_colors = {
        "fresh": "#2BD67B",
        "stale": "#FFB84D",
        "demo": "#A8B3C7",
        "unavailable": "#A8B3C7",
        "warming": "#7AA8FF",
        "error": "#FF5C7A",
    }

    badges = []
    for source, status in freshness_data.items():
        if source == "timestamp":
            continue
        cfg = source_config.get(source, {"label": source.title(), "icon": "●"})
        color = status_colors.get(status, "#A8B3C7")
        status_text = status.upper() if status != "fresh" else "LIVE"
        badges.append(
            html.Div(
                [
                    html.Span(cfg["icon"], style={"marginRight": "6px"}),
                    html.Span(
                        cfg["label"],
                        style={"fontWeight": "600", "marginRight": "6px"},
                    ),
                    html.Span(
                        status_text,
                        style={
                            "fontSize": "0.65rem",
                            "padding": "1px 6px",
                            "borderRadius": "3px",
                            "background": f"{color}20",
                            "color": color,
                        },
                    ),
                ],
                className="data-health-badge",
                style={
                    "display": "inline-flex",
                    "alignItems": "center",
                    "fontSize": "0.75rem",
                    "color": "#A8B3C7",
                    "padding": "4px 12px",
                    "marginRight": "12px",
                },
            )
        )

    if not badges:
        return html.Div()

    return html.Div(
        [
            html.Span(
                "DATA SOURCES",
                style={
                    "fontSize": "0.65rem",
                    "color": "#A8B3C7",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginRight": "16px",
                },
            ),
            *badges,
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "alignItems": "center",
            "padding": "8px 12px",
            "background": "#11182D",
            "borderRadius": "6px",
        },
    )


def _build_overview_spotlight(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> go.Figure:
    """Build persona-specific spotlight chart for overview."""
    if persona_id == "renewables":
        return _spotlight_renewables(weather_df, region)
    if persona_id == "trader":
        return _spotlight_trader(demand_df, region)
    if persona_id == "data_scientist":
        return _spotlight_model_accuracy(region)
    # Default: grid_ops → demand sparkline
    return _build_overview_sparkline(demand_df, region)


def _spotlight_renewables(weather_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Renewable generation potential chart."""
    if weather_df is None or weather_df.empty:
        return _empty_figure("No weather data for renewable outlook")

    df = weather_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_48h = df.tail(48)

    fig = go.Figure()
    if "wind_speed_80m" in last_48h.columns:
        fig.add_trace(
            go.Scatter(
                x=last_48h.get("timestamp", list(range(len(last_48h)))),
                y=last_48h["wind_speed_80m"],
                mode="lines",
                line=dict(color=CB_PALETTE.get("sky", "#56B4E9"), width=2),
                name="Wind (mph)",
                hovertemplate="%{y:.0f} mph<extra>Wind</extra>",
            )
        )
    if "shortwave_radiation" in last_48h.columns:
        fig.add_trace(
            go.Scatter(
                x=last_48h.get("timestamp", list(range(len(last_48h)))),
                y=last_48h["shortwave_radiation"],
                mode="lines",
                line=dict(color=CB_PALETTE.get("orange", "#E69F00"), width=2),
                name="Solar (W/m²)",
                yaxis="y2",
                hovertemplate="%{y:.0f} W/m²<extra>Solar</extra>",
            )
        )

    renew_layout = _layout(
        uirevision=region,
        margin=dict(l=45, r=45, t=35, b=40),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        # Axis overrides flow through ``_layout()`` so the shared
        # gridcolor / linecolor defaults in ``PLOT_LAYOUT`` deep-merge
        # with per-chart options. Passing these as kwargs to
        # ``update_layout()`` separately would conflict with the
        # ``xaxis`` / ``yaxis`` keys already in ``PLOT_LAYOUT``.
        yaxis=dict(
            title="Wind (mph)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis2=dict(
            title="Solar (W/m²)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
    )
    fig.update_layout(
        **renew_layout,
        title=dict(text="Renewable Potential (48h)", font=dict(size=13, color="#DDE6F2")),
        showlegend=True,
    )
    return fig


def _spotlight_trader(demand_df: pd.DataFrame | None, region: str) -> go.Figure:
    """Demand vs capacity utilization chart for traders."""
    capacity = REGION_CAPACITY_MW.get(region, 50000)

    if demand_df is None or demand_df.empty:
        return _empty_figure("No demand data for market view")

    df = demand_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_48h = df.tail(48)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_48h.get("timestamp", list(range(len(last_48h)))),
            y=last_48h["demand_mw"],
            mode="lines",
            line=dict(color=CB_PALETTE["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,114,178,0.15)",
            name="Demand",
            hovertemplate="%{y:,.0f} MW<extra>Demand</extra>",
        )
    )

    # Capacity line
    fig.add_hline(
        y=capacity,
        line_dash="dot",
        line_color="#FF5C7A",
        annotation_text=f"Capacity: {capacity:,.0f} MW",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#FF5C7A",
    )

    # Pricing tier thresholds
    for pct, label, color in [
        (0.85, "High tier (85%)", "#FFB84D"),
        (0.70, "Moderate (70%)", "#A8B3C7"),
    ]:
        fig.add_hline(
            y=capacity * pct,
            line_dash="dot",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="bottom left",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    trader_layout = _layout(
        uirevision=region,
        margin=dict(l=50, r=10, t=35, b=30),
        xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat=",.0f",
            title="MW",
        ),
    )
    fig.update_layout(
        **trader_layout,
        title=dict(text="Demand vs Capacity", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
    )
    return fig


def _spotlight_model_accuracy(region: str) -> go.Figure:
    """Model accuracy bar chart for data scientists."""
    # Pull from backtest cache if available
    models = ["prophet", "arima", "xgboost"]
    mape_values = []

    for model_name in models:
        mape = None
        for horizon in [168, 24, 720]:
            bt_key = (region, horizon, model_name, DEFAULT_BACKTEST_EXOG_MODE)
            if bt_key in _BACKTEST_CACHE:
                result_dict, _, _ = _BACKTEST_CACHE[bt_key]
                if isinstance(result_dict, dict) and "mape" in result_dict:
                    mape = result_dict["mape"]
                    break
        mape_values.append(mape if mape is not None else 4.5 + len(model_name) * 0.3)

    colors = [
        CB_PALETTE.get("vermillion", "#D55E00"),
        CB_PALETTE.get("blue", "#0072B2"),
        CB_PALETTE.get("green", "#009E73"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[m.title() for m in models],
            y=mape_values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in mape_values],
            textposition="outside",
            textfont=dict(color="#DDE6F2", size=11),
            hovertemplate="%{x}: %{y:.2f}% MAPE<extra></extra>",
        )
    )

    model_layout = _layout(
        uirevision=region,
        margin=dict(l=40, r=10, t=35, b=30),
        yaxis=dict(
            title="MAPE (%)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        xaxis=dict(showgrid=False),
    )
    fig.update_layout(
        **model_layout,
        title=dict(text="Model MAPE Comparison", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
    )
    return fig


def _build_overview_digest(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> html.Div:
    """Aggregate top insights from all tabs into a cross-tab digest."""
    from components.insights import (
        Insight,
        build_insight_card,
        generate_tab1_insights,
        generate_tab2_insights,
        generate_tab3_insights,
        generate_tab4_insights,
    )

    all_insights: list[Insight] = []

    # Tab 1: Historical demand insights
    try:
        tab1 = generate_tab1_insights(persona_id, region, demand_df, weather_df)
        all_insights.extend(tab1)
    except Exception:
        pass

    # Tab 2: Forecast insights (need predictions — skip if unavailable)
    try:
        for horizon in [168, 24]:
            pred_key = (region, horizon)
            if pred_key in _PREDICTION_CACHE:
                preds, timestamps, _, _ = _PREDICTION_CACHE[pred_key]
                tab2 = generate_tab2_insights(persona_id, region, preds, timestamps)
                all_insights.extend(tab2)
                break
    except Exception:
        pass

    # Validation tab: model accuracy insights (from backtest cache)
    try:
        for model_name in ["xgboost", "prophet", "arima"]:
            for horizon in [168, 24, 720]:
                bt_key = (region, horizon, model_name, DEFAULT_BACKTEST_EXOG_MODE)
                if bt_key in _BACKTEST_CACHE:
                    result_dict, _, _ = _BACKTEST_CACHE[bt_key]
                    if isinstance(result_dict, dict) and "mape" in result_dict:
                        metrics = {model_name: result_dict}
                        tab3 = generate_tab3_insights(
                            persona_id,
                            region,
                            metrics,
                            model_name=model_name,
                            horizon_hours=horizon,
                        )
                        all_insights.extend(tab3)
                        break
            else:
                continue
            break
    except Exception:
        pass

    # Tab 4: Grid insights (from generation cache)
    try:
        if region in _GENERATION_CACHE:
            gen_df, _ = _GENERATION_CACHE[region]
            if gen_df is not None and not gen_df.empty:
                gen_copy = gen_df.copy()
                gen_copy["timestamp"] = pd.to_datetime(gen_copy["timestamp"])
                pivot = gen_copy.pivot_table(
                    index="timestamp",
                    columns="fuel_type",
                    values="generation_mw",
                    aggfunc="sum",
                ).fillna(0)
                total_gen = pivot.sum(axis=1)
                renewable_cols = [
                    c for c in pivot.columns if c.lower() in ("wind", "solar", "hydro")
                ]
                renewable_gen = (
                    pivot[renewable_cols].sum(axis=1) if renewable_cols else total_gen * 0
                )
                renewable_pct = (
                    (renewable_gen.sum() / total_gen.sum() * 100) if total_gen.sum() > 0 else 0.0
                )
                net_load = total_gen - renewable_gen
                tab4 = generate_tab4_insights(
                    persona_id=persona_id,
                    region=region,
                    net_load=net_load,
                    demand=total_gen,
                    renewable_pct=renewable_pct,
                    pivot=pivot,
                    timestamps=pd.DatetimeIndex(pivot.index),
                )
                all_insights.extend(tab4)
    except Exception:
        pass

    if not all_insights:
        return html.Div(
            html.P(
                "No insights available yet. Explore tabs to generate data.",
                style={"color": "#A8B3C7", "fontSize": "0.82rem", "fontStyle": "italic"},
            )
        )

    # Sort by severity (warning first)
    severity_order = {"warning": 0, "notable": 1, "info": 2}
    all_insights.sort(key=lambda i: severity_order.get(i.severity, 2))

    return build_insight_card(all_insights, persona_id, "Overview", max_insights=5)


def _build_overview_news() -> html.Div:
    """Fetch and render energy news for the overview tab."""
    from data.news_client import fetch_energy_news

    try:
        articles = fetch_energy_news(page_size=10)
        if not articles:
            from data.news_client import _get_demo_news

            articles = _get_demo_news()
        return build_news_feed(articles)
    except Exception as e:
        log.error("overview_news_failed", error=str(e))
        from data.news_client import _get_demo_news

        return build_news_feed(_get_demo_news())


def _build_persona_kpis(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> dbc.Row:
    """Build persona-specific KPI cards from live demand/weather data."""
    capacity = REGION_CAPACITY_MW.get(region, 50000)

    # Extract real stats from demand data
    peak_mw = None
    avg_mw = None
    min_mw = None
    pct_of_capacity = None
    if demand_df is not None and "demand_mw" in demand_df.columns:
        # Drop NaN and non-positive values — EIA occasionally emits spurious
        # zero rows (e.g. NYISO 2026-02-10 had 6 consecutive zero-hours)
        # that collapse Demand Range to the full peak. Utility demand is
        # never physically 0 or negative for any covered BA.
        valid = demand_df.dropna(subset=["demand_mw"])
        valid = valid[valid["demand_mw"] > 0]
        if not valid.empty:
            peak_mw = valid["demand_mw"].max()
            avg_mw = valid["demand_mw"].mean()
            min_mw = valid["demand_mw"].min()
            pct_of_capacity = peak_mw / capacity * 100 if capacity > 0 else 0

    # Fallback: read demand stats from Redis
    if peak_mw is None:
        actuals_redis = redis_get(redis_key(f"actuals:{region}"))
        if actuals_redis and actuals_redis.get("demand_mw"):
            demand_vals = [v for v in actuals_redis["demand_mw"] if v is not None and v > 0]
            if demand_vals:
                peak_mw = max(demand_vals)
                avg_mw = sum(demand_vals) / len(demand_vals)
                min_mw = min(demand_vals)
                pct_of_capacity = peak_mw / capacity * 100 if capacity > 0 else 0

    # Extract weather stats. pandas ``.mean()`` is NaN-aware but returns NaN
    # when every value is null — coerce that to ``None`` so the downstream
    # ``is not None`` guards (and Redis fallback) behave consistently.
    avg_wind = None
    avg_solar = None
    if weather_df is not None:
        if "wind_speed_80m" in weather_df.columns:
            mean = weather_df["wind_speed_80m"].mean()
            avg_wind = float(mean) if pd.notna(mean) else None
        if "shortwave_radiation" in weather_df.columns:
            mean = weather_df["shortwave_radiation"].mean()
            avg_solar = float(mean) if pd.notna(mean) else None

    # Fallback: read weather stats from Redis. Gate each metric independently
    # so a missing column in ``weather_df`` still triggers the Redis lookup
    # for that metric (previously both had to be ``None`` for any fallback).
    # Also filter None/NaN entries before averaging — Redis arrays can have
    # gaps where Open-Meteo emitted nulls.
    if avg_wind is None or avg_solar is None:
        weather_redis = redis_get(redis_key(f"weather:{region}"))
        if weather_redis:
            if avg_wind is None and "wind_speed_80m" in weather_redis:
                vals = [v for v in weather_redis["wind_speed_80m"] if pd.notna(v)]
                if vals:
                    avg_wind = sum(vals) / len(vals)
            if avg_solar is None and "shortwave_radiation" in weather_redis:
                vals = [v for v in weather_redis["shortwave_radiation"] if pd.notna(v)]
                if vals:
                    avg_solar = sum(vals) / len(vals)

    # Get backtest MAPE from cache if available
    backtest_mape = None
    backtest_rmse = None
    for horizon in [168, 24, 720]:  # prefer 7-day, then 24h, then 30d
        bt_key = (region, horizon, "xgboost", DEFAULT_BACKTEST_EXOG_MODE)
        if bt_key in _BACKTEST_CACHE:
            cached_result, _, _ = _BACKTEST_CACHE[bt_key]
            if "metrics" in cached_result:
                backtest_mape = cached_result["metrics"].get("mape")
                backtest_rmse = cached_result["metrics"].get("rmse")
                break

    # Fallback: read from Redis if in-memory cache is empty
    if backtest_mape is None:
        for horizon in [168, 24, 720]:
            bt_redis = redis_get(
                redis_key(f"backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}")
            )
            if bt_redis is None:
                bt_redis = redis_get(redis_key(f"backtest:{region}:{horizon}"))
            if bt_redis and "metrics" in bt_redis:
                xgb_metrics = bt_redis["metrics"].get("xgboost", {})
                if xgb_metrics:
                    backtest_mape = xgb_metrics.get("mape")
                    backtest_rmse = xgb_metrics.get("rmse")
                    break

    # Compute derived metrics
    reserve_margin_pct = (100.0 - pct_of_capacity) if pct_of_capacity is not None else None
    demand_range = (peak_mw - min_mw) if peak_mw is not None and min_mw is not None else None

    # Wind capacity factor (approximate: avg_wind / rated_wind)
    wind_cf = None
    if avg_wind is not None:
        from config import WIND_CUTOUT_SPEED_MPH

        wind_cf = min(avg_wind / WIND_CUTOUT_SPEED_MPH * 100, 100.0)

    # Solar capacity factor (approximate: avg_irradiance / rated_irradiance)
    solar_cf = None
    if avg_solar is not None:
        from config import SOLAR_RATED_IRRADIANCE

        solar_cf = avg_solar / SOLAR_RATED_IRRADIANCE * 100

    # Estimate price from utilization (merit-order approximation)
    from config import PRICING_BASE_USD_MWH

    price_estimate = None
    if pct_of_capacity is not None:
        utilization = pct_of_capacity / 100
        if utilization < 0.70:
            price_estimate = PRICING_BASE_USD_MWH
        elif utilization < 0.90:
            price_estimate = PRICING_BASE_USD_MWH * (1 + (utilization - 0.70) * 5)
        else:
            price_estimate = PRICING_BASE_USD_MWH * (2 + (utilization - 0.90) * 20)

    # Format values
    peak_str = f"{int(peak_mw):,} MW" if peak_mw is not None else "No data"
    avg_str = f"{int(avg_mw):,} MW" if avg_mw is not None else "No data"
    cap_str = f"{pct_of_capacity:.0f}% of capacity" if pct_of_capacity is not None else ""
    mape_str = f"{backtest_mape:.1f}%" if backtest_mape is not None else "No data"
    mape_dir = "positive" if backtest_mape is not None and backtest_mape < 5 else "negative"
    rmse_str = f"{int(backtest_rmse):,} MW" if backtest_rmse is not None else "No data"

    persona_kpis = {
        "grid_ops": [
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "negative" if pct_of_capacity and pct_of_capacity > 80 else "neutral",
            },
            {
                "label": "Reserve Margin",
                "value": f"{reserve_margin_pct:.0f}%"
                if reserve_margin_pct is not None
                else "No data",
                "delta": "Below 15% is tight",
                "direction": "negative"
                if reserve_margin_pct is not None and reserve_margin_pct < 15
                else "positive"
                if reserve_margin_pct is not None
                else "neutral",
            },
            {
                "label": "Forecast Error",
                "value": mape_str,
                "delta": f"Walk-forward MAPE ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
            {
                "label": "Demand Range",
                "value": f"{int(demand_range):,} MW" if demand_range is not None else "No data",
                "delta": "Peak - Min",
                "direction": "neutral",
            },
        ],
        "renewables": [
            {
                "label": "Wind CF",
                "value": f"{wind_cf:.0f}%" if wind_cf is not None else "No data",
                "delta": "Capacity factor",
                "direction": "positive" if wind_cf is not None and wind_cf > 25 else "neutral",
            },
            {
                "label": "Solar CF",
                "value": f"{solar_cf:.0f}%" if solar_cf is not None else "No data",
                "delta": "Capacity factor",
                "direction": "positive" if solar_cf is not None and solar_cf > 15 else "neutral",
            },
            {
                "label": "Avg Wind",
                "value": f"{avg_wind:.1f} mph" if avg_wind is not None else "No data",
                "delta": "80m hub height",
                "direction": "neutral",
            },
            {
                "label": "Avg Solar",
                "value": f"{avg_solar:.0f} W/m²" if avg_solar is not None else "No data",
                "delta": "Shortwave radiation",
                "direction": "neutral",
            },
        ],
        "trader": [
            {
                "label": "Est. Price",
                "value": f"${price_estimate:.0f}/MWh" if price_estimate is not None else "No data",
                "delta": "Merit-order estimate",
                "direction": "negative"
                if price_estimate is not None and price_estimate > 100
                else "neutral",
            },
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "neutral",
            },
            {
                "label": "Avg Demand",
                "value": avg_str,
                "delta": f"Range: {int(demand_range):,} MW" if demand_range is not None else "",
                "direction": "neutral",
            },
            {
                "label": "Forecast Error",
                "value": mape_str,
                "delta": f"Walk-forward MAPE ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
        ],
        "data_scientist": [
            {
                "label": "XGBoost MAPE",
                "value": mape_str,
                "delta": f"Target: <5% ({DEFAULT_BACKTEST_EXOG_MODE})",
                "direction": mape_dir,
            },
            {
                "label": "RMSE",
                "value": rmse_str,
                "delta": "Walk-forward backtest",
                "direction": "neutral",
            },
            {
                "label": "Peak Demand",
                "value": peak_str,
                "delta": cap_str,
                "direction": "neutral",
            },
            {
                "label": "Demand Range",
                "value": f"{int(demand_range):,} MW" if demand_range is not None else "No data",
                "delta": "Max variability",
                "direction": "neutral",
            },
        ],
    }
    kpis = persona_kpis.get(persona_id, persona_kpis["grid_ops"])
    return build_kpi_row(kpis)


# ── Callback registration (Step 10 — register_callbacks split) ──────


def register_overview_callbacks(app):
    """Register Overview-tab callbacks with the Dash app.

    Step 10a of the ``register_callbacks`` decomposition. Called once
    by ``components.callbacks.register_callbacks`` at app boot. Owning
    the Dash decorator block here keeps the Overview tab's read path
    end-to-end inside this module — layout (``tab_overview.py``),
    helpers (the 17 functions above), and callback wiring all in
    coherent places.
    """

    @app.callback(
        [
            Output("overview-title", "children"),
            Output("overview-metrics-bar", "children"),
            Output("overview-spotlight-chart", "figure"),
            Output("overview-model-card", "children"),
            Output("overview-insight-card", "children"),
        ],
        [
            Input("demand-store", "data"),
            Input("dashboard-tabs", "active_tab"),
            Input("persona-selector", "value"),
        ],
        [
            State("weather-store", "data"),
            State("region-selector", "value"),
            State("data-freshness-store", "data"),
        ],
    )
    def update_overview_tab(
        demand_json, active_tab, persona_id, weather_json, region, freshness_data
    ):
        """Render the v2 linear-stack Overview: title, metrics, chart, model, insight."""
        if active_tab != "tab-overview":
            return [no_update] * 5

        persona_id = persona_id or "grid_ops"
        region = region or "FPL"

        try:
            demand_df = None
            if demand_json:
                demand_df = pd.read_json(io.StringIO(demand_json))
            # weather_json + freshness_data reserved for future inline drivers panel
            del weather_json, freshness_data

            # 1. Title block (region name + subtitle)
            title = _build_overview_title(region)

            # 2. MetricsBar (5-up KPI row)
            metrics_bar = build_metrics_bar(_build_overview_metrics_items(demand_df))

            # 3. Hero forecast chart (actual + dashed forecast + confidence band)
            chart = _build_overview_hero_chart(region, demand_df)

            # 4. ModelMetricsCard
            model_card = _build_overview_model_card(region)

            # 5. InsightCard
            insight = _build_overview_insight(region, demand_df, persona_id)

            return (title, metrics_bar, chart, model_card, insight)
        except Exception as exc:
            log.exception("update_overview_tab_failed")
            err_msg = f"{type(exc).__name__}: {exc}"
            err_div = html.Div(
                err_msg,
                style={"color": "var(--danger)", "fontSize": "0.8rem", "padding": "8px"},
            )
            return (err_div, html.Div(), _empty_figure(err_msg), html.Div(), err_div)


__all__ = [
    # 7a — Overview core
    "_build_overview_title",
    "_build_overview_metrics_items",
    "_build_overview_hero_chart",
    "_build_overview_model_card",
    "_build_overview_insight",
    # 7b — Overview panels
    "_fetch_generation_cached",
    "_build_drivers_panel",
    "_drivers_empty",
    "_driver_cell_empty",
    "_build_generation_panel",
    "_generation_empty",
    "_build_models_leaderboard",
    "_build_risk_insight",
    "_build_scenarios_panel",
    "_driver_sparkline",
    # 7c — Overview briefing surface
    "_build_overview_sparkline",
    "_build_overview_briefing",
    "_build_weather_context",
    "_build_changes_card",
    "_build_overview_data_health",
    "_build_overview_spotlight",
    "_spotlight_renewables",
    "_spotlight_trader",
    "_spotlight_model_accuracy",
    "_build_overview_digest",
    "_build_overview_news",
    "_build_persona_kpis",
    # 10a — Callback registration
    "register_overview_callbacks",
]
