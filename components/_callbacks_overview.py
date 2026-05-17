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
from config import REGION_CAPACITY_MW, REGION_NAMES
from data.redis_client import redis_get
from personas.config import get_persona

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


# ── Overview panels block (Step 7b — drivers / generation / models / risk / scenarios) ──


def _fetch_generation_cached(region: str) -> pd.DataFrame | None:
    """Fetch generation data with 3-tier caching: memory -> SQLite/API -> demo.

    Args:
        region: Balancing authority code.

    Returns:
        DataFrame with [timestamp, fuel_type, generation_mw, region] or None.
    """
    import time as _time

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
        mape = float(m.get("mape", 0.0))
        mae = float(m.get("mae", 0.0))
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
                "unit": f"MAE {mae:,.0f}",
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
        " Check the timeline above for active NOAA alerts.",
    ]
    return build_insight_card("Risk summary", body)


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

    Sensitivities (calibrated against typical residential cooling/heating
    response in U.S. balancing authorities):
      * temp_delta: ±2.5 % demand per +5 °F above 65 °F (cooling) and per
        −5 °F below 65 °F (heating). Symmetric for simplicity.
      * wind_delta: ±0.6 % renewable share per +1 mph (caps at 30 % share).
      * solar_delta: ±0.05 % renewable share per +1 W/m² (caps similarly).
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
            from models.model_service import get_forecasts

            forecasts = get_forecasts(region, demand_df, models_shown=["ensemble"])
            ens = forecasts.get("ensemble")
            if ens is not None and len(ens) >= horizon:
                base_y = np.asarray(ens[:horizon], dtype=float)
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
    # Demand response is dominated by temperature; wind/solar shift the
    # renewable share (which we surface separately) but barely move demand.
    demand_factor = 1.0 + (temp_delta / 5.0) * 0.025  # ±2.5% per 5°F
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
        **_layout(uirevision=column),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=4, b=4),
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
    )
    fig.update_layout(
        **sparkline_layout,
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
        "alerts": {"label": "NOAA Alerts", "icon": "⚠"},
    }

    status_colors = {
        "fresh": "#2BD67B",
        "stale": "#FFB84D",
        "demo": "#A8B3C7",
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
    )
    fig.update_layout(
        **renew_layout,
        title=dict(text="Renewable Potential (48h)", font=dict(size=13, color="#DDE6F2")),
        showlegend=True,
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

    trader_layout = _layout(uirevision=region, margin=dict(l=50, r=10, t=35, b=30))
    fig.update_layout(
        **trader_layout,
        title=dict(text="Demand vs Capacity", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
        xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat=",.0f",
            title="MW",
        ),
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

    model_layout = _layout(uirevision=region, margin=dict(l=40, r=10, t=35, b=30))
    fig.update_layout(
        **model_layout,
        title=dict(text="Model MAPE Comparison", font=dict(size=13, color="#DDE6F2")),
        showlegend=False,
        yaxis=dict(
            title="MAPE (%)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        xaxis=dict(showgrid=False),
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
        actuals_redis = redis_get(f"wattcast:actuals:{region}")
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
        weather_redis = redis_get(f"wattcast:weather:{region}")
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
                f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}"
            )
            if bt_redis is None:
                bt_redis = redis_get(f"wattcast:backtest:{region}:{horizon}")
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
