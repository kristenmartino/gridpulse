"""Alerts / Extreme Events tab helpers extracted from ``components/callbacks.py``.

Step 4 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)

## What lives here

Currently a single function: ``_alerts_tab_from_redis``. The Alerts tab
is structurally a "Redis-only" tab — every view it renders comes from
the ``wattcast:alerts:{region}`` payload that the scoring job writes
each hour. There's no fallback compute path because alert detection
itself runs in the scoring job, not on the web. That makes this module
small (~200 lines) and self-contained.

When the alerts tab grows additional helpers (severity classifiers,
event-timeline filters, etc.), they belong here.

## Public-import surface

``components/callbacks.py`` re-imports ``_alerts_tab_from_redis`` by
name, so ``from components.callbacks import _alerts_tab_from_redis``
in tests + the ``register_callbacks`` wiring continues to resolve.

When patching for tests, target the function's *new* namespace:

    @patch("components._callbacks_alerts.redis_get")          # ✓
    @patch("components.callbacks.redis_get")  # only-models  # ✗
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import html

from components._callbacks_shared import (
    COLORS,
    _empty_figure,
    _layout,
)
from components.cards import build_alert_card
from data.redis_client import redis_get

log = structlog.get_logger()


def _alerts_tab_from_redis(region):
    """Redis fast path for update_alerts_tab callback.

    Returns an 8-tuple (alert_cards, stress_str, stress_label_span, breakdown,
    fig_anomaly, fig_temp, fig_timeline, weather_context) or None if cache miss.
    """
    cached = redis_get(f"wattcast:alerts:{region}")
    if cached is None:
        return None

    empty = _empty_figure("Loading...")
    log.info("alerts_redis_hit", region=region)
    alerts = cached.get("alerts", [])
    stress = cached.get("stress_score", 20)
    stress_label = cached.get("stress_label", "Normal")
    counts = cached.get("alert_counts", {})
    anomaly = cached.get("anomaly", {})
    temp_data = cached.get("temperature", {})

    # Build alert cards
    alert_cards = []
    if alerts:
        for a in alerts:
            alert_cards.append(
                build_alert_card(
                    event=a["event"],
                    headline=a["headline"],
                    severity=a["severity"],
                    expires=a.get("expires", "")[:16] if a.get("expires") else None,
                )
            )
    else:
        alert_cards = [
            html.P(
                "No active alerts",
                style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
            )
        ]

    stress_color = "positive" if stress < 30 else ("negative" if stress >= 60 else "neutral")
    n_crit = counts.get("critical", 0)
    n_warn = counts.get("warning", 0)
    n_info = counts.get("info", 0)
    breakdown_items = []
    if n_crit:
        breakdown_items.append(
            html.Div(
                f"\U0001f534 Critical: {n_crit}",
                style={"fontSize": "0.75rem", "color": "#FF5C7A"},
            )
        )
    if n_warn:
        breakdown_items.append(
            html.Div(
                f"\U0001f7e1 Warning: {n_warn}",
                style={"fontSize": "0.75rem", "color": "#FFB84D"},
            )
        )
    if n_info:
        breakdown_items.append(
            html.Div(
                f"\U0001f535 Info: {n_info}",
                style={"fontSize": "0.75rem", "color": "#56B4E9"},
            )
        )
    if not alerts:
        breakdown_items.append(
            html.Div(
                "No active alerts",
                style={"fontSize": "0.75rem", "color": "#A8B3C7"},
            )
        )
    breakdown = html.Div(breakdown_items)

    # Anomaly detection chart
    a_ts = pd.to_datetime(anomaly.get("timestamps", []))
    a_demand = anomaly.get("demand", [])
    a_upper = anomaly.get("upper", [])
    a_lower = anomaly.get("lower", [])
    a_anom_ts = pd.to_datetime(anomaly.get("anomaly_timestamps", []))
    a_anom_vals = anomaly.get("anomaly_values", [])

    if len(a_ts) > 0:
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(
            go.Scatter(x=a_ts, y=a_demand, name="Demand", line=dict(color=COLORS["actual"]))
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=a_ts,
                y=a_upper,
                name="Upper (2σ)",
                line=dict(color="#FF5C7A", dash="dash", width=1),
            )
        )
        fig_anomaly.add_trace(
            go.Scatter(
                x=a_ts,
                y=a_lower,
                name="Lower (2σ)",
                line=dict(color="#FF5C7A", dash="dash", width=1),
            )
        )
        if len(a_anom_ts) > 0:
            fig_anomaly.add_trace(
                go.Scatter(
                    x=a_anom_ts,
                    y=a_anom_vals,
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="#FF5C7A", size=8, symbol="diamond"),
                )
            )
        fig_anomaly.update_layout(**_layout(uirevision=region, yaxis_title="MW"))
    else:
        fig_anomaly = empty

    # Temperature chart
    t_ts = pd.to_datetime(temp_data.get("timestamps", []))
    t_vals = temp_data.get("values", [])
    if len(t_ts) > 0:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(x=t_ts, y=t_vals, name="Temperature", line=dict(color=COLORS["temperature"]))
        )
        for t in [95, 100, 105]:
            fig_temp.add_hline(
                y=t,
                line=dict(color="#FF5C7A", dash="dot", width=1),
                annotation_text=f"{t}°F",
                annotation_position="right",
            )
        fig_temp.update_layout(**_layout(uirevision=region, yaxis_title="°F"))
    else:
        fig_temp = empty

    # Historical event timeline (static)
    events = [
        ("2021-02-15", "Winter Storm Uri", "ERCOT", 95),
        ("2022-09-06", "CA Heat Wave", "CAISO", 80),
        ("2023-07-20", "Heat Dome", "CAISO", 85),
        ("2024-04-08", "Solar Eclipse", "PJM", 40),
    ]
    fig_timeline = go.Figure()
    for date, name, reg, sev in events:
        color = COLORS["ensemble"] if reg == region else "#A8B3C7"
        fig_timeline.add_trace(
            go.Scatter(
                x=[date],
                y=[sev],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=12, color=color),
                showlegend=False,
            )
        )
    fig_timeline.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="Date",
            yaxis_title="Severity Score",
            yaxis_range=[0, 100],
        )
    )

    # Weather context: build from cached temperature if available
    weather_context = html.Div()
    if t_vals:
        import dash_bootstrap_components as dbc

        last_temp = float(t_vals[-1])
        color = "#FF5C7A" if last_temp >= 95 else ("#FFB84D" if last_temp >= 85 else "#2BD67B")
        weather_context = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P("TEMPERATURE", className="kpi-label"),
                            html.H4(
                                f"{last_temp:.0f}°F",
                                className="kpi-value",
                                style={"fontSize": "1.3rem"},
                            ),
                        ],
                        className="kpi-card",
                        style={"borderTop": f"3px solid {color}"},
                    ),
                    md=3,
                ),
            ],
            className="g-2",
        )

    return (
        alert_cards,
        str(stress),
        html.Span(stress_label, className=f"kpi-delta {stress_color}"),
        breakdown,
        fig_anomaly,
        fig_temp,
        fig_timeline,
        weather_context,
    )


__all__ = [
    "_alerts_tab_from_redis",
]
