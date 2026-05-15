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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import html

from components._callbacks_shared import (
    _empty_figure,
    _latest_real_demand,
    _layout,
)
from components.cards import (
    build_insight_card,
    build_model_metrics_card,
    build_page_title,
)
from config import REGION_NAMES

log = structlog.get_logger()


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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Strip spurious zero-demand rows (EIA's missing-observation marker)
    # AND NaN rows (EIA-930's publishing lag for the most recent hour,
    # especially for newer / smaller BAs like PSCO / NEVP / AZPS — those
    # rows arrive at the next hourly tick instead). The ``> 0`` check
    # filters both: NaN > 0 is False.
    nonzero = df[df["demand_mw"] > 0]
    last_7d = nonzero.tail(168)

    # ``now_value`` reads from ``nonzero`` rather than ``df`` so the
    # most recent NaN / zero hour doesn't surface as "nan" / "0" in the
    # hero metric. Falls back to "—" when no usable reading exists.
    now_value = float(nonzero["demand_mw"].iloc[-1]) if not nonzero.empty else None
    peak_7d = float(last_7d["demand_mw"].max()) if not last_7d.empty else 0.0
    low_7d = float(last_7d["demand_mw"].min()) if not last_7d.empty else 0.0
    avg_7d = float(last_7d["demand_mw"].mean()) if not last_7d.empty else 0.0

    # 24h trend uses the same NaN-aware source as ``now_value`` so a
    # missing ago_24h hour doesn't poison the percentage with NaN.
    if now_value is not None and len(nonzero) >= 25:
        ago_24h = float(nonzero["demand_mw"].iloc[-25])
        trend_pct = ((now_value - ago_24h) / ago_24h * 100.0) if ago_24h else 0.0
    else:
        trend_pct = 0.0
    # Inverted semantic: rising demand reads as "warning" (negative tone),
    # falling demand reads as "positive" — matches v2 MetricsBar.tsx:64.
    trend_tone = (
        "negative" if trend_pct > 0.5 else ("positive" if trend_pct < -0.5 else "secondary")
    )

    now_display = f"{now_value:,.0f}" if now_value is not None else "—"
    trend_display = f"{trend_pct:+.1f}%" if now_value is not None else "—"

    return [
        {"label": "Now", "value": now_display, "unit": "MW", "hero": True},
        {"label": "7d Peak", "value": f"{peak_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "7d Low", "value": f"{low_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "Average", "value": f"{avg_7d:,.0f}", "unit": "MW", "tone": "secondary"},
        {"label": "24h Trend", "value": trend_display, "unit": None, "tone": trend_tone},
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

    # 24h forecast bridge (orange dashed) + confidence band
    try:
        from models.model_service import get_forecasts

        forecasts = get_forecasts(region, df, models_shown=["ensemble"])
        ensemble = forecasts.get("ensemble")
        upper_80 = forecasts.get("upper_80")
        lower_80 = forecasts.get("lower_80")
        if ensemble is not None and len(ensemble) > 0:
            horizon = min(24, len(ensemble))
            forecast_ts = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h",
            )
            ensemble_y = list(ensemble[:horizon])

            # Confidence band — drawn first so it sits behind the line
            if upper_80 is not None and lower_80 is not None and len(upper_80) >= horizon:
                upper_y = list(upper_80[:horizon])
                lower_y = list(lower_80[:horizon])
                fig.add_trace(
                    go.Scatter(
                        x=list(forecast_ts) + list(forecast_ts[::-1]),
                        y=upper_y + lower_y[::-1],
                        fill="toself",
                        fillcolor="rgba(249, 115, 22, 0.12)",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        name="80% confidence",
                    )
                )

            # Forecast line (bridged from last actual point — no gap)
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
    except Exception as exc:  # pragma: no cover — fall back to actual-only chart
        log.warning("overview_hero_forecast_failed", region=region, error=str(exc))

    fig.update_layout(
        **_layout(uirevision=region, showlegend=False),
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
    formatted = {
        "MAPE": f"{m.get('mape', 0.0):.1f}%",
        "RMSE": f"{m.get('rmse', 0.0):,.0f} MW",
        "MAE": f"{m.get('mae', 0.0):,.0f} MW",
        "R²": f"{m.get('r2', 0.0):.3f}",
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

    forecast_clause = "Next-cycle forecast confidence is updating."
    try:
        from models.model_service import get_forecasts

        forecasts = get_forecasts(region, df, models_shown=["ensemble"])
        ensemble = forecasts.get("ensemble")
        if ensemble is not None and len(ensemble) > 0:
            horizon = min(24, len(ensemble))
            f_arr = np.asarray(ensemble[:horizon])
            f_peak = float(f_arr.max())
            f_peak_idx = int(f_arr.argmax())
            f_peak_ts = df["timestamp"].iloc[-1] + pd.Timedelta(hours=f_peak_idx + 1)
            metrics_dict = forecasts.get("metrics") or {}
            ens_metrics = metrics_dict.get("ensemble") or metrics_dict.get("xgboost") or {}
            mape = float(ens_metrics.get("mape", 0.0))
            forecast_clause = (
                f"Next-24h forecast peaks at {f_peak:,.0f} MW around "
                f"{f_peak_ts.strftime('%H:%M')} (MAPE {mape:.1f}%)."
            )
    except Exception as exc:  # pragma: no cover
        log.warning("overview_insight_forecast_failed", region=region, error=str(exc))

    body = [
        "Demand is ",
        html.Span(f"{abs(delta_pct):.1f}% {direction}", className=delta_class),
        " the 7-day average. ",
        "Recent peak: ",
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


__all__ = [
    "_build_overview_title",
    "_build_overview_metrics_items",
    "_build_overview_hero_chart",
    "_build_overview_model_card",
    "_build_overview_insight",
]
