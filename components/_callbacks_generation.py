"""Generation & Net Load tab helpers extracted from ``components/callbacks.py``.

Step 5 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)
* #101 — Alerts tab (``_callbacks_alerts.py``)

## What lives here

``_generation_tab_from_redis`` — the Redis fast path that builds the
entire Generation & Net Load tab from the scoring job's hourly
``wattcast:generation:{region}`` payload.

The function shapes the parallel-array EIA payload into a fuel-pivot
DataFrame, joins with demand for net-load arithmetic, computes the four
KPIs (renewable share, peak ramp, min net load, curtailment hours),
renders the two stacked-area Plotly figures, and assembles a persona-
aware insight card via ``components.insights``.

## Public-import surface

``components/callbacks.py`` re-imports ``_generation_tab_from_redis``
by name. ``from components.callbacks import _generation_tab_from_redis``
in tests + the ``register_callbacks`` wiring continues to resolve.

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_generation.redis_get")  # ✓
    @patch("components.callbacks.redis_get")              # ✗ (no effect)
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import structlog

from components._callbacks_shared import COLORS, _layout
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


def _generation_tab_from_redis(region, range_hours, demand_json, persona_id):
    """Redis fast path for update_generation_tab callback.

    Returns a 7-tuple (fig_hero, fig_mix, ren_pct, peak_ramp, min_net,
    curtailment, insight_card) or None if cache miss.
    """
    cached_gen = redis_get(redis_key(f"generation:{region}"))
    if cached_gen is None or not cached_gen.get("timestamps"):
        return None

    # uirevision keyed on region + range so zoom survives Redis refresh
    # but resets when user picks a new date range.
    uirev = f"{region}:{range_hours}"

    log.info("generation_redis_hit", region=region)
    # Convert parallel-arrays to DataFrame
    gen_cols = {k: v for k, v in cached_gen.items() if k not in ("region",)}
    if "timestamps" in gen_cols:
        gen_cols["timestamp"] = gen_cols.pop("timestamps")
    gen_redis_df = pd.DataFrame(gen_cols)
    gen_redis_df["timestamp"] = pd.to_datetime(gen_redis_df["timestamp"])

    # Filter by date range
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=range_hours)
    if gen_redis_df["timestamp"].dt.tz is None:
        cutoff = cutoff.tz_localize(None)
    gen_redis_df = gen_redis_df[gen_redis_df["timestamp"] >= cutoff]

    if gen_redis_df.empty:
        return None

    # Fuel columns (everything except timestamp, region, renewable_pct)
    fuel_cols = [
        c for c in gen_redis_df.columns if c not in ("timestamp", "region", "renewable_pct")
    ]
    pivot = gen_redis_df.set_index("timestamp")[fuel_cols]
    total_gen = pivot.sum(axis=1)

    # Demand from demand-store or approximate as total gen
    demand_series = total_gen
    if demand_json:
        try:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            demand_aligned = demand_df.set_index("timestamp")["demand_mw"]
            common_idx = pivot.index.intersection(demand_aligned.index)
            if len(common_idx) > 24:
                demand_series = demand_aligned.loc[common_idx]
                pivot = pivot.loc[common_idx]
                total_gen = pivot.sum(axis=1)
        except Exception:
            pass

    wind_gen = pivot.get("wind", pd.Series(0, index=pivot.index))
    solar_gen = pivot.get("solar", pd.Series(0, index=pivot.index))
    net_load = demand_series - wind_gen - solar_gen
    common_idx = pivot.index

    # KPIs
    ren_cols = [c for c in ["wind", "solar", "hydro"] if c in pivot.columns]
    if ren_cols and total_gen.mean() > 0:
        renewable_pct = float((pivot[ren_cols].sum(axis=1) / total_gen * 100).mean())
    else:
        renewable_pct = 0.0
    net_load_diff = net_load.diff()
    peak_ramp = float(net_load_diff.max()) if not net_load_diff.isna().all() else 0.0
    min_net_load = float(net_load.min())
    peak_net_load = float(net_load.max())
    curtailment_hours = int((net_load < peak_net_load * 0.2).sum()) if peak_net_load > 0 else 0

    # Charts
    fig_hero = go.Figure()
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=demand_series.values if hasattr(demand_series, "values") else demand_series,
            mode="lines",
            name="Total Demand",
            line=dict(color=COLORS["actual"], width=2),
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=net_load.values,
            mode="lines",
            name="Net Load",
            line=dict(color=COLORS["ensemble"], width=2.5),
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=demand_series.values if hasattr(demand_series, "values") else demand_series,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_hero.add_trace(
        go.Scatter(
            x=common_idx,
            y=net_load.values,
            mode="lines",
            name="Renewable Contribution",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(45,226,196,0.15)",
        )
    )
    fig_hero.update_layout(
        **_layout(
            uirevision=uirev,
            yaxis_title="MW",
            hovermode="x unified",
            title=f"Demand vs Net Load — {region}",
        )
    )

    fig_mix = go.Figure()
    fuel_order = [
        "nuclear",
        "coal",
        "gas",
        "natural_gas",
        "hydro",
        "wind",
        "solar",
        "oil",
        "other",
    ]
    for fuel in fuel_order:
        if fuel in pivot.columns:
            fig_mix.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[fuel],
                    mode="lines",
                    name=fuel.replace("_", " ").title(),
                    stackgroup="one",
                    line=dict(width=0),
                    fillcolor=COLORS.get(fuel, "#95a5a6"),
                )
            )
    fig_mix.update_layout(
        **_layout(
            uirevision=uirev,
            yaxis_title="Generation (MW)",
            hovermode="x unified",
            title=f"Generation Mix — {region}",
        )
    )

    from components.insights import build_insight_card, generate_tab4_insights

    persona = persona_id or "grid_ops"
    insights = generate_tab4_insights(
        persona_id=persona,
        region=region,
        net_load=net_load,
        demand=demand_series,
        renewable_pct=renewable_pct,
        pivot=pivot,
        timestamps=pd.DatetimeIndex(common_idx),
    )
    insight_card = build_insight_card(insights, persona, "tab-generation")

    return (
        fig_hero,
        fig_mix,
        f"{renewable_pct:.1f}%",
        f"{peak_ramp:,.0f} MW/hr",
        f"{min_net_load:,.0f} MW",
        str(curtailment_hours),
        insight_card,
    )


__all__ = [
    "_generation_tab_from_redis",
]
