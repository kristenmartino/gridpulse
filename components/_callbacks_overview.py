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

The three-sub-step population plan this docstring used to describe
(headline block → panels → briefing surface) completed, and then the
third sub-step was **deleted in full**. The briefing surface —
sparklines, AI briefing, digest, spotlights, data-health, changes card
and persona KPIs, ~2,200 lines across this module and two `data/`
clients — was dead code that no layout slot targeted, and several of its
builders invented signal they could not source. Removed across #513,
#518, #524, #525 and #526 under the GP-P1-04 decision (#221).

So this file is no longer "the single home for every
``_build_overview_*`` / ``_spotlight_*`` helper" — there is no
``_spotlight_*`` family any more, and the ``_build_overview_*`` helpers
that remain are exactly the five the live callback calls.

§1c is **done**. Every helper that rendered another tab's surface has
moved to the tab that renders it: the models leaderboard to
``_callbacks_models``, the risk insight and weather context to
``_callbacks_alerts``, and the drivers / generation / scenarios panels
to ``_callbacks_forecast``. ``_read_ensemble_forecast_from_redis`` went
to ``_callbacks_shared``, because the Overview hero chart and the
Forecast scenarios panel both read that payload and neither tab should
depend on the other for it.

No other tab module imports from this one. What is left here renders the
Overview and nothing else.

What replaces the deleted surface is §3 of
``docs/internal/OVERVIEW_DECISION_LAYER_PROPOSAL.md`` — an honest
Redis-backed briefing. #523 is its first piece.

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
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import Input, Output, State, html, no_update

from components._callbacks_shared import (
    _empirical_interval_from_backtests,
    _empty_figure,
    _latest_real_demand,
    _layout,
    _read_ensemble_forecast_from_redis,
)
from components.accessibility import model_display_name
from components.cards import (
    build_insight_card,
    build_metrics_bar,
    build_model_metrics_card,
    build_page_title,
)
from config import (
    FRESHNESS_FRESH_MAX_AGE_HOURS,
    REGION_NAMES,
)
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


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
            # Require a meaningful window — 24 records minimum inside the
            # WINDOW before its figure is statistically defensible. P2-21
            # (#273): ``n_records`` is total retained history (trimmed by
            # count, not age), so gating on it let a "live 7d" headline rest
            # on a handful of in-window observations. Gate each window on
            # its own post-filter count (``n_7d``/``n_30d``); payloads
            # written before #273 lack the counts — fall back to the old
            # total-count gate for the one tick until they rewrite.
            n_records = int(ens.get("n_records", 0) or 0)
            n_7d = ens.get("n_7d")
            n_30d = ens.get("n_30d")
            ok_7d = (int(n_7d) >= 24) if n_7d is not None else n_records >= 24
            ok_30d = (int(n_30d) >= 24) if n_30d is not None else n_records >= 24
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
            if live_7d is not None and ok_7d and np.isfinite(float(live_7d)):
                return float(live_7d), f"live 7d {metric_7d}"
            if live_30d is not None and ok_30d and np.isfinite(float(live_30d)):
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


def _exclusions_from_freshness(freshness_data) -> list[dict]:
    """Extract the #309 artifact-exclusion records from the freshness store.

    The data-freshness-store holds a JSON STRING (``json.dumps(freshness)`` in
    ``load_data`` — same contract ``update_fallback_banner`` parses), not a
    dict. The first cut of this wire checked ``isinstance(dict)`` and silently
    dropped the disclosure in prod while every builder-level render pin stayed
    green — the classic unpinned-wire gap. Tolerant of both formats and of
    malformed payloads; never raises.
    """
    if not freshness_data:
        return []
    try:
        parsed = json.loads(freshness_data) if isinstance(freshness_data, str) else freshness_data
        if isinstance(parsed, dict):
            exclusions = parsed.get("artifact_excluded") or []
            return exclusions if isinstance(exclusions, list) else []
    except (TypeError, ValueError):
        pass
    return []


def _build_overview_metrics_items(
    demand_df: pd.DataFrame | None,
    artifact_exclusions: list[dict] | None = None,
) -> list[dict]:
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

    # #309: the scoring job's quality guard excluded implausible trailing
    # readings (EIA partials — the LADWP "NOW 730 MW" class). The series here
    # is already cleaned, so the value above is the last REAL reading; this
    # discloses that newer-but-implausible readings exist.
    now_help = "Most recent actual demand reading (EIA-930)."
    exclusions = artifact_exclusions or []
    if exclusions:
        n_exc = len(exclusions)
        plural = "s" if n_exc != 1 else ""
        now_subtext = (
            f"{now_subtext} · {n_exc} newer reading{plural} excluded"
            if now_subtext
            else f"{n_exc} newer reading{plural} excluded"
        )
        newest = exclusions[-1]
        now_help = (
            f"Most recent PLAUSIBLE demand reading (EIA-930). Excluded: "
            f"{newest.get('mw'):,.0f} MW at {str(newest.get('ts'))[11:16]} UTC — "
            f"{newest.get('reason')}. EIA typically corrects these within the hour."
        )

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
            "help": now_help,
        },
        {
            "label": "7d Peak",
            "value": f"{peak_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "hourly max",
            "help": "Highest observed (actual) demand in the last 168h (7 days).",
        },
        {
            "label": "7d Low",
            "value": f"{low_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "hourly min",
            "help": "Lowest observed (actual) demand in the last 168h (7 days).",
        },
        {
            "label": "Average",
            "value": f"{avg_7d:,.0f}",
            "unit": "MW",
            "tone": "secondary",
            "subtext": "7d hourly mean",
            "help": "Mean observed (actual) demand over the last 168h (7 days).",
        },
        {
            "label": "24h Trend",
            "value": trend_display,
            "unit": None,
            "tone": trend_tone,
            "subtext": trend_subtext,
            "help": "Percent change from ~24h ago to now.",
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
    name = model_display_name(primary_key)
    badge = "trained" if is_trained(region) else "simulated"

    caption = None
    if primary_key == "ensemble":
        single = {
            k: v.get("mape")
            for k, v in metrics_dict.items()
            if k != "ensemble" and v.get("mape") is not None
        }
        if single:
            best_key = min(single, key=single.get)
            best_name = model_display_name(best_key)
            caption = f"Production blend — most accurate single model: {best_name} {single[best_key]:.1f}%"
        else:
            caption = "Production blend — combines all models for stability."

    return build_model_metrics_card(
        model_name=name, metrics=formatted, badge=badge, caption=caption
    )


def _provenance_note(region: str) -> list[str]:
    """One class-conditional sentence from ``gridpulse:vintage_summary`` (#319).

    Copy is deterministic and auditable: every number is a measured field from
    the vintage study, never generated. Silent for clean/unknown/missing.
    """
    from components._callbacks_shared import _read_vintage_summary
    from config import FEED_LIMITED_MIN_REVISION_PCT

    summary = _read_vintage_summary(region) or {}
    cls = summary.get("revision_class")
    rev = summary.get("mean_fresh_revision_pct")
    if not isinstance(rev, (int, float)):
        return []
    # Same magnitude floor as the Feed-limited pills: the first live fleet
    # classification (2026-07-17) put half the fleet in churn/bulk at trivial
    # magnitudes (AVA 0.9%, AECI 0.2%, ERCOT 1.8%) — frequency-based classes
    # are honest, but a "revises 2%" note on thirty regions is noise, and
    # callouts stay credible only while they are rare.
    if float(rev) < FEED_LIMITED_MIN_REVISION_PCT:
        return []
    if cls == "broken":
        return [
            f" Data note: this region's EIA feed publishes provisional readings "
            f"that revise {float(rev):.0f}% on average before settling — intraday "
            f"figures and live accuracy firm up over the following day."
        ]
    if cls == "bulk":
        return [
            f" Data note: same-day EIA values for this region are provisional "
            f"until the next-morning resubmission (measured revisions "
            f"{float(rev):.0f}%)."
        ]
    if cls == "churn":
        return [
            f" Data note: EIA readings here typically revise {float(rev):.0f}% "
            f"shortly after publication as metering completes."
        ]
    return []


def _build_overview_insight(
    region: str,
    demand_df: pd.DataFrame | None,
    persona_id: str,
    artifact_exclusions: list[dict] | None = None,
    weather_df: pd.DataFrame | None = None,
) -> html.Div:
    """Narrative paragraph with semantic-color delta spans, then role-specific lines.

    The narrative is identical for every persona — demand level, last-24h
    peak, next-24h forecast with live MAPE and staleness disclosure. What
    varies is what follows it: ``generate_tab1_insights`` emits observations
    tagged with ``persona_relevance``, and ``_filter_for_persona`` keeps and
    ranks the ones this role cares about. Grid Ops sees ramp rate and
    variability; Renewables sees the temperature-demand correlation; the
    Data Scientist leads with variability.

    Before #523 the persona changed only the eyebrow label — four words —
    which is why the landing page's "role-aware briefing" claim was reworded
    away in #522. This is the half that earns it back.

    The ``peak_demand`` insight is dropped: the narrative above already
    states the last-24h peak, and the two together read as a stutter. Matched
    on ``metric_name``, not on the rendered string.
    """
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
    # Demand above/below a 7-day average is routine — neither a risk nor a win —
    # so render the delta in neutral text both directions (#219 — 2026-07);
    # alert/semantic colors are reserved for genuine risk.
    delta_class = ""

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
    # #309: name the exclusion in prose — the tiles above already computed on
    # the cleaned series, so without this sentence the exclusion is invisible.
    for exc in (artifact_exclusions or [])[-1:]:
        body.append(
            f" Latest EIA reading ({exc.get('mw'):,.0f} MW at "
            f"{str(exc.get('ts'))[11:16]} UTC) excluded: {exc.get('reason')} — "
            f"an EIA reporting artifact, not demand."
        )

    # PR 3: the per-class data note, from the vintage study's measured
    # verdict. clean/unknown/missing → silence — callouts stay rare, which is
    # what keeps them credible.
    body.extend(_provenance_note(region))

    # #523: the role-aware half. Appended as sentences rather than a bullet
    # list because ``build_insight_card`` renders ``body`` inside an
    # ``html.P`` — nesting a ``<ul>`` there is invalid HTML and browsers
    # break the paragraph around it.
    try:
        from components.insights import generate_tab1_insights

        for ins in generate_tab1_insights(persona_id, region, demand_df, weather_df):
            # The narrative already reported the peak two sentences ago.
            if ins.metric_name == "peak_demand":
                continue
            body.append(f" {ins.text}")
    except Exception as exc:  # pragma: no cover — never let this drop the card
        log.warning("overview_insight_persona_lines_failed", region=region, error=str(exc))

    # Eyebrow names the role; the lines above are what make it true.
    eyebrow_map = {
        "grid_ops": "Operating summary",
        "renewables": "Renewables outlook",
        "trader": "Market signal",
        "data_scientist": "Model summary",
    }
    eyebrow = eyebrow_map.get(persona_id, "Summary")
    return build_insight_card(eyebrow, body)


# ── Overview panels block (Step 7b — drivers / generation / models / risk / scenarios) ──


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
            # #523: weather feeds the Renewables persona's temperature-demand
            # line in the insight card. Without it that persona gets nothing
            # role-specific at all — the one it most needs. Parsing a store
            # payload is CPU, not a web-tier I/O-guardrail read; the writer
            # serialises it exactly like demand-store above.
            #
            # Caught in its own try: weather here is an ENRICHMENT, so a
            # malformed payload must cost the enrichment, not the page. The
            # enclosing handler returns an error div for all five outputs,
            # which would blank the whole Overview over one optional sentence.
            weather_df = None
            if weather_json:
                try:
                    weather_df = pd.read_json(io.StringIO(weather_json))
                except Exception as exc:
                    log.warning("overview_weather_parse_failed", region=region, error=str(exc))
            artifact_exclusions = _exclusions_from_freshness(freshness_data)

            # 1. Title block (region name + subtitle)
            title = _build_overview_title(region)

            # 2. MetricsBar (5-up KPI row)
            metrics_bar = build_metrics_bar(
                _build_overview_metrics_items(demand_df, artifact_exclusions)
            )

            # 3. Hero forecast chart (actual + dashed forecast + confidence band)
            chart = _build_overview_hero_chart(region, demand_df)

            # 4. ModelMetricsCard
            model_card = _build_overview_model_card(region)

            # 5. InsightCard
            insight = _build_overview_insight(
                region, demand_df, persona_id, artifact_exclusions, weather_df
            )

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
    # 7c — Overview briefing surface
    # 10a — Callback registration
    "register_overview_callbacks",
]
