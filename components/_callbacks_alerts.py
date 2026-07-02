"""Alerts / Extreme Events tab helpers extracted from ``components/callbacks.py``.

Step 4 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)

## What lives here

Currently a single function: ``_alerts_tab_from_redis``. The Alerts tab
is structurally a "Redis-only" tab — every view it renders comes from
the ``gridpulse:alerts:{region}`` payload that the scoring job writes
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

import io

import pandas as pd
import plotly.graph_objects as go
import structlog
from dash import Input, Output, html, no_update

from components._callbacks_shared import (
    COLORS,
    _empty_figure,
    _layout,
)
from components.cards import build_alert_card
from config import REQUIRE_REDIS
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


def _alerts_tab_from_redis(region):
    """Redis fast path for update_alerts_tab callback.

    Returns an 8-tuple (alert_cards, stress_str, stress_label_span, breakdown,
    fig_anomaly, fig_temp, fig_timeline, weather_context) or None if cache miss.
    """
    cached = redis_get(redis_key(f"alerts:{region}"))
    if cached is None:
        return None

    empty = _empty_figure("Loading...")
    log.info("alerts_redis_hit", region=region)
    alerts = cached.get("alerts", [])
    # Legacy payloads (pre-alerts_source) only ever carried demo content.
    alerts_source = cached.get("alerts_source", "demo")
    stress = cached.get("stress_score")
    stress_label = cached.get("stress_label", "Unavailable")
    counts = cached.get("alert_counts", {})
    anomaly = cached.get("anomaly", {})
    temp_data = cached.get("temperature", {})

    # Build alert cards
    alert_cards = []
    if alerts:
        if alerts_source == "demo":
            alert_cards.append(
                html.P(
                    "Demo data — not a live alert feed",
                    className="gp-demo-disclosure",
                    style={"color": "#FFB84D", "fontSize": "0.75rem", "textAlign": "center"},
                )
            )
        for a in alerts:
            alert_cards.append(
                build_alert_card(
                    event=a["event"],
                    headline=a["headline"],
                    severity=a["severity"],
                    expires=a.get("expires", "")[:16] if a.get("expires") else None,
                )
            )
    elif alerts_source == "unavailable":
        alert_cards = [
            html.P(
                "No live alert feed connected — severe-weather alerts are not "
                "yet integrated. Demand anomalies below are computed from real data.",
                style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
            )
        ]
    else:
        alert_cards = [
            html.P(
                "No active alerts",
                style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"},
            )
        ]

    if stress is None:
        stress_color = "neutral"
    else:
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
                "No alert feed" if alerts_source == "unavailable" else "No active alerts",
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
        "—" if stress is None else str(stress),
        html.Span(stress_label, className=f"kpi-delta {stress_color}"),
        breakdown,
        fig_anomaly,
        fig_temp,
        fig_timeline,
        weather_context,
    )


# ── Callback registration (Step 10d — register_callbacks split) ──────


def register_alerts_callbacks(app):
    """Register Risk / Alerts tab callbacks with the Dash app.

    Step 10d of the ``register_callbacks`` decomposition (issue #87).
    Owns the three callbacks driving the Risk tab:

    * ``update_risk_title`` — page title block.
    * ``update_risk_insight`` — 1-sentence narrative summary
      (``'all systems nominal'`` or an elevated-risk note).
    * ``update_alerts_tab`` — 8-output payload: alert cards, stress
      score / label / breakdown, anomaly chart, temperature exceedance,
      historical event timeline, weather-context KPIs. Has Redis fast
      path (``_alerts_tab_from_redis``) and a v1 demo-data compute
      fallback for dev mode.

    Section comment in callbacks.py was historically labeled
    ``# ── 7. TAB 4: GENERATION & NET LOAD ──`` even though every
    output id is ``tab5-*`` (the Risk tab). The label survived an
    earlier round of refactoring that deleted the Generation TAB 4
    callbacks but left the breadcrumb. Corrected when this section
    moved into ``register_alerts_callbacks(app)``.
    """
    from components._callbacks_overview import _build_risk_insight, _build_weather_context
    from components.cards import build_page_title
    from config import REGION_NAMES

    @app.callback(
        Output("risk-title", "children"),
        [
            Input("region-selector", "value"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def update_risk_title(region, active_tab):
        """Page-title block for the Risk tab."""
        if active_tab != "tab-alerts":
            return no_update

        region = region or "FPL"
        region_name = REGION_NAMES.get(region, region)
        return build_page_title(
            "Risk",
            f"Active alerts, demand anomalies, and grid stress · {region_name}",
        )

    @app.callback(
        Output("risk-insight-card", "children"),
        [
            Input("dashboard-tabs", "active_tab"),
            Input("region-selector", "value"),
            Input("demand-store", "data"),
            Input("weather-store", "data"),
        ],
    )
    def update_risk_insight(active_tab, region, demand_json, weather_json):
        """Narrative summary for the Risk tab — 'all systems nominal' or
        a 1-sentence elevated-risk note."""
        if active_tab != "tab-alerts":
            return no_update
        return _build_risk_insight(region, demand_json, weather_json)

    @app.callback(
        [
            Output("tab5-alerts-list", "children"),
            Output("tab5-stress-score", "children"),
            Output("tab5-stress-label", "children"),
            Output("tab5-stress-breakdown", "children"),
            Output("tab5-anomaly-chart", "figure"),
            Output("tab5-temp-exceedance", "figure"),
            Output("tab5-timeline", "figure"),
            Output("tab5-weather-context", "children"),
        ],
        [
            Input("region-selector", "value"),
            Input("demand-store", "data"),
            Input("weather-store", "data"),
            Input("dashboard-tabs", "active_tab"),
        ],
        prevent_initial_call=True,
    )
    def update_alerts_tab(region, demand_json, weather_json, active_tab):
        """Update Tab 5 alerts and stress indicators."""
        if active_tab != "tab-alerts":
            return [no_update] * 8
        empty = _empty_figure("Loading...")

        # ── v2 Redis fast path ──────────────────────────────
        if region:
            redis_result = _alerts_tab_from_redis(region)
            if redis_result is not None:
                return redis_result

        # ── warming gate (REQUIRE_REDIS deployments) ────────
        # The scoring job owns alert-payload generation in staging/prod;
        # a Redis miss there means the pipeline is warming, and the demo
        # fallback below must never render as real data.
        if REQUIRE_REDIS:
            from components.error_handling import warming_state

            log.info("alerts_warming_gate", region=region)
            warming = warming_state(
                title="Risk data is warming up",
                message=(
                    "The scheduled pipeline has not published alert/anomaly "
                    "data for this region yet. This page will populate after "
                    "the next scoring run."
                ),
            )
            return (
                [warming],
                "—",
                html.Span("Warming"),
                html.Div(),
                empty,
                empty,
                empty,
                html.Div(),
            )

        # ── v1 compute fallback (dev/demo only) ─────────────
        from data.demo_data import generate_demo_alerts

        alerts = generate_demo_alerts(region)

        alert_cards = []
        if alerts:
            alert_cards.append(
                html.P(
                    "Demo data — not a live alert feed",
                    className="gp-demo-disclosure",
                    style={"color": "#FFB84D", "fontSize": "0.75rem", "textAlign": "center"},
                )
            )
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

        n_crit = sum(1 for a in alerts if a["severity"] == "critical")
        n_warn = sum(1 for a in alerts if a["severity"] == "warning")
        n_info = sum(1 for a in alerts if a["severity"] == "info")
        stress = min(100, n_crit * 30 + n_warn * 15 + 20)
        stress_label = "Normal" if stress < 30 else ("Elevated" if stress < 60 else "Critical")
        stress_color = "positive" if stress < 30 else ("negative" if stress >= 60 else "neutral")

        from components.icons import icon as _icon

        breakdown_items = []
        if n_crit:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "alert-triangle",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--critical",
                        ),
                        html.Span(f"Critical: {n_crit}"),
                    ],
                    className="gp-stress-row gp-stress-row--critical",
                )
            )
        if n_warn:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "alert-circle",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--warning",
                        ),
                        html.Span(f"Warning: {n_warn}"),
                    ],
                    className="gp-stress-row gp-stress-row--warning",
                )
            )
        if n_info:
            breakdown_items.append(
                html.Div(
                    [
                        _icon(
                            "info",
                            size="xs",
                            className="gp-stress-row__icon gp-stress-row__icon--info",
                        ),
                        html.Span(f"Info: {n_info}"),
                    ],
                    className="gp-stress-row gp-stress-row--info",
                )
            )
        if not alerts:
            breakdown_items.append(
                html.Div("No active alerts", className="gp-stress-row gp-stress-row--empty")
            )
        breakdown = html.Div(breakdown_items)

        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            recent = demand_df.tail(168)
            rolling_mean = recent["demand_mw"].rolling(24).mean()
            rolling_std = recent["demand_mw"].rolling(24).std()
            upper = rolling_mean + 2 * rolling_std
            lower = rolling_mean - 2 * rolling_std
            anomalies = recent[recent["demand_mw"] > upper]

            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["demand_mw"],
                    name="Demand",
                    line=dict(color=COLORS["actual"]),
                )
            )
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=upper,
                    name="Upper (2σ)",
                    line=dict(color="#FF5C7A", dash="dash", width=1),
                )
            )
            fig_anomaly.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=lower,
                    name="Lower (2σ)",
                    line=dict(color="#FF5C7A", dash="dash", width=1),
                )
            )
            if not anomalies.empty:
                fig_anomaly.add_trace(
                    go.Scatter(
                        x=anomalies["timestamp"],
                        y=anomalies["demand_mw"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color="#FF5C7A", size=8, symbol="diamond"),
                    )
                )
            fig_anomaly.update_layout(**_layout(uirevision=region, yaxis_title="MW"))
        else:
            fig_anomaly = empty

        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            recent_w = weather_df.tail(168)
            fig_temp = go.Figure()
            fig_temp.add_trace(
                go.Scatter(
                    x=recent_w["timestamp"],
                    y=recent_w["temperature_2m"],
                    name="Temperature",
                    line=dict(color=COLORS["temperature"]),
                )
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

        # Build weather context from latest reading
        weather_context = html.Div()
        if weather_json:
            try:
                w_df = pd.read_json(io.StringIO(weather_json))
                if not w_df.empty:
                    weather_context = _build_weather_context(w_df.iloc[-1])
            except Exception:
                log.warning("weather_context_build_failed")

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
    "register_alerts_callbacks",
]
