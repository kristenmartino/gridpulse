"""Weather-Energy Correlation tab helpers extracted from ``components/callbacks.py``.

Step 6 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)
* #101 — Alerts tab (``_callbacks_alerts.py``)
* #102 — Generation tab (``_callbacks_generation.py``)

## What lives here

``_weather_tab_from_redis`` — the Redis fast path that builds the six
Plotly figures (temp/wind/solar scatters, correlation heatmap, feature-
importance bar chart, seasonal decomposition subplots) for the
Weather-Energy Correlation tab from the scoring job's hourly
``wattcast:weather-correlation:{region}`` payload.

## Orphaned-callback observation

Like ``_generation_tab_from_redis`` before it (#102), this fast path is
no longer wired into ``register_callbacks``. Tests still exercise it,
so the re-export stays — but a follow-up should decide whether to fully
delete or re-wire.

## Public-import surface

``components/callbacks.py`` re-imports ``_weather_tab_from_redis`` by
name with an explicit ``# noqa: F401 — re-export`` marker (tests import
via ``from components.callbacks import``).

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_weather.redis_get")  # ✓
    @patch("components.callbacks.redis_get")           # ✗ (no effect)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from plotly.subplots import make_subplots

from components._callbacks_shared import COLORS, _layout
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


def _weather_tab_from_redis(region):
    """Redis fast path for update_weather_tab callback.

    Returns a 6-tuple of Plotly figures or None if cache miss.
    """
    cached = redis_get(redis_key(f"weather-correlation:{region}"))
    if cached is None:
        return None

    log.info("weather_redis_hit", region=region)
    corr_data = cached.get("correlation_matrix", {})
    imp_data = cached.get("importance", {})
    seasonal = cached.get("seasonal", {})

    # Scatter: Temperature vs Demand
    fig_temp = go.Figure(
        go.Scatter(
            x=cached.get("temperature_2m", []),
            y=cached.get("demand_mw", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["actual"], opacity=0.4),
        )
    )
    fig_temp.update_layout(
        **_layout(uirevision=region, xaxis_title="Temperature (°F)", yaxis_title="Demand (MW)")
    )

    # Scatter: Wind
    fig_wind = go.Figure(
        go.Scatter(
            x=cached.get("wind_speed_80m", []),
            y=cached.get("wind_power", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["wind"], opacity=0.5),
        )
    )
    fig_wind.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="Wind Speed (mph)",
            yaxis_title="Wind Power Estimate",
        )
    )

    # Scatter: Solar
    fig_solar = go.Figure(
        go.Scatter(
            x=cached.get("shortwave_radiation", []),
            y=cached.get("solar_cf", []),
            mode="markers",
            marker=dict(size=3, color=COLORS["solar"], opacity=0.5),
        )
    )
    fig_solar.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="GHI (W/m²)",
            yaxis_title="Solar Capacity Factor",
        )
    )

    # Heatmap
    corr_cols = corr_data.get("cols", [])
    corr_vals = corr_data.get("values", [])
    fig_heatmap = go.Figure(
        go.Heatmap(
            z=corr_vals,
            x=corr_cols,
            y=corr_cols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(np.array(corr_vals), 2) if corr_vals else [],
            texttemplate="%{text}",
        )
    )
    fig_heatmap.update_layout(**_layout(uirevision=region))

    # Feature importance
    imp_names = imp_data.get("names", [])
    imp_vals = imp_data.get("values", [])
    fig_importance = go.Figure(
        go.Bar(
            x=imp_vals,
            y=imp_names,
            orientation="h",
            marker_color=COLORS["ensemble"],
        )
    )
    fig_importance.update_layout(**_layout(uirevision=region, xaxis_title="Correlation Strength"))

    # Seasonal decomposition
    s_ts = pd.to_datetime(seasonal.get("timestamps", []))
    s_orig = seasonal.get("original", [])
    s_trend = seasonal.get("trend", [])
    s_resid = seasonal.get("residual", [])
    fig_seasonal = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Original", "Trend (7-day)", "Residual"],
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_orig,
            line=dict(color=COLORS["actual"], width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_trend,
            line=dict(color=COLORS["ensemble"], width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig_seasonal.add_trace(
        go.Scatter(
            x=s_ts,
            y=s_resid,
            line=dict(color=COLORS["arima"], width=1),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig_seasonal.update_layout(**_layout(uirevision=region, height=350))

    return fig_temp, fig_wind, fig_solar, fig_heatmap, fig_importance, fig_seasonal


__all__ = [
    "_weather_tab_from_redis",
]
