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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import dcc, html

from components._callbacks_shared import (
    _EIA_FUEL_MAP,
    _GENERATION_CACHE,
    _empty_figure,
    _latest_real_demand,
    _layout,
)
from components.cards import (
    build_insight_card,
    build_metrics_bar,
    build_model_metrics_card,
    build_page_title,
)
from config import REGION_CAPACITY_MW, REGION_NAMES

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
]
