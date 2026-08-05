"""Alerts / Extreme Events tab helpers extracted from ``components/callbacks.py``.

Step 4 of the ``callbacks.py`` decomposition tracked in issue #87.
Continues the per-tab split established by:

* #98 — shared infrastructure (``_callbacks_shared.py``)
* #99 — US Grid tab (``_callbacks_us_grid.py``)
* #100 — Models tab (``_callbacks_models.py``)

## What lives here

``_render_risk_tab`` plus the per-widget builders it composes, and the
two *data* paths that feed it:

* ``_alerts_tab_from_redis`` — production. Reads the
  ``gridpulse:alerts:{region}`` payload the scoring job writes hourly.
* the dev/demo fallback inside ``update_alerts_tab`` — computes the same
  shapes locally from ``demand-store`` / ``weather-store`` when
  ``REQUIRE_REDIS`` is off.

**One renderer, two data sources (P2-44 / #273).** These two paths used
to build the tab independently and had drifted into *different designs*:
production rendered the stress breakdown as emoji plus inline styles
while dev rendered the ``gp-stress-row`` icon components, and the dev
path printed a literal ``"None"`` for BAs whose capacity is unreliable.
The split that remains is the honest one — where the numbers come from —
and the rendering below it is shared, so the two can no longer diverge.

The docstring previously claimed this module held "a single function"
with "no fallback compute path"; the dev fallback has computed grid
stress inline since #265. Corrected here.

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


def _stress_tone(stress: int | None) -> str:
    """kpi-delta tone for the utilization-based grid-stress score (#265).

    Normal < 70 (positive) ≤ Elevated < 85 (neutral) ≤ High ≥ 85 (negative);
    a missing score is neutral. Bands mirror ``models.pricing.grid_stress``.
    """
    if stress is None:
        return "neutral"
    return "positive" if stress < 70 else ("negative" if stress >= 85 else "neutral")


#: Temperature-exceedance reference lines, °F. Hardcoded and fleet-uniform —
#: see P2-29 (#273), which tracks sourcing or per-region parameterization.
_TEMP_EXCEEDANCE_LINES_F = (95, 100, 105)

#: Historical extreme events plotted on the timeline. The severity scores are
#: unsourced editorial values, not measured — P2-29 (#273) tracks sourcing or
#: dropping them.
_HISTORICAL_EVENTS = (
    ("2021-02-15", "Winter Storm Uri", "ERCOT", 95),
    ("2022-09-06", "CA Heat Wave", "CAISO", 80),
    ("2023-07-20", "Heat Dome", "CAISO", 85),
    ("2024-04-08", "Solar Eclipse", "PJM", 40),
)


def _build_alert_cards(alerts, alerts_source, alerts_total=None):
    """Alert card list, including the provenance disclosure for the source."""
    if alerts:
        cards = []
        if alerts_source == "demo":
            cards.append(
                html.P(
                    "Demo data — not a live alert feed",
                    className="gp-demo-disclosure",
                    style={"color": "#FFB84D", "fontSize": "0.75rem", "textAlign": "center"},
                )
            )
        for a in alerts:
            cards.append(
                build_alert_card(
                    event=a["event"],
                    headline=a["headline"],
                    severity=a["severity"],
                    expires=a.get("expires", "")[:16] if a.get("expires") else None,
                )
            )
        if alerts_source == "noaa":
            total = int(alerts_total or len(alerts))
            more_note = f" · showing {len(alerts)} of {total}" if total > len(alerts) else ""
            cards.append(
                html.P(
                    f"Live severe-weather alerts · NOAA/NWS{more_note}",
                    className="gp-alerts-source",
                    style={"color": "#A8B3C7", "fontSize": "0.72rem", "textAlign": "center"},
                )
            )
        return cards

    if alerts_source == "unavailable":
        message = (
            "Severe-weather alerts (NOAA/NWS) are temporarily unavailable. "
            "The temperature and demand-anomaly charts below use live "
            "weather and demand data and are unaffected."
        )
    elif alerts_source == "noaa":
        message = "No active severe-weather alerts (NOAA/NWS live feed)"
    else:
        message = "No active alerts"
    return [
        html.P(message, style={"color": "#A8B3C7", "textAlign": "center", "padding": "20px"})
    ]


def _build_stress_breakdown(counts, has_alerts, alerts_source):
    """Severity breakdown rows.

    Unified on the ``gp-stress-row`` icon markup (P2-44 / #273). Production
    previously rendered emoji with inline styles here while dev rendered these
    rows, so the same tab had two visual languages depending on the data path.
    """
    from components.icons import icon as _icon

    rows = []
    for key, label, icon_name in (
        ("critical", "Critical", "alert-triangle"),
        ("warning", "Warning", "alert-circle"),
        ("info", "Info", "info"),
    ):
        n = counts.get(key, 0)
        if not n:
            continue
        rows.append(
            html.Div(
                [
                    _icon(
                        icon_name,
                        size="xs",
                        className=f"gp-stress-row__icon gp-stress-row__icon--{key}",
                    ),
                    html.Span(f"{label}: {n}"),
                ],
                className=f"gp-stress-row gp-stress-row--{key}",
            )
        )
    if not has_alerts:
        rows.append(
            html.Div(
                "No alert feed" if alerts_source == "unavailable" else "No active alerts",
                className="gp-stress-row gp-stress-row--empty",
            )
        )
    return html.Div(rows)


def _build_anomaly_figure(region, timestamps, demand, upper, lower, anom_ts, anom_vals):
    """Demand vs ±2σ band, with exceedances marked."""
    if len(timestamps) == 0:
        return _empty_figure("Loading...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=demand, name="Demand", line=dict(color=COLORS["actual"])))
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=upper, name="Upper (2σ)", line=dict(color="#FF5C7A", dash="dash", width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=lower, name="Lower (2σ)", line=dict(color="#FF5C7A", dash="dash", width=1)
        )
    )
    if len(anom_ts) > 0:
        fig.add_trace(
            go.Scatter(
                x=anom_ts,
                y=anom_vals,
                mode="markers",
                name="Anomaly",
                marker=dict(color="#FF5C7A", size=8, symbol="diamond"),
            )
        )
    fig.update_layout(**_layout(uirevision=region, yaxis_title="MW"))
    return fig


def _build_temp_figure(region, timestamps, values):
    """Temperature series with the fleet-uniform exceedance reference lines."""
    if len(timestamps) == 0:
        return _empty_figure("Loading...")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=timestamps, y=values, name="Temperature", line=dict(color=COLORS["temperature"]))
    )
    for t in _TEMP_EXCEEDANCE_LINES_F:
        fig.add_hline(
            y=t,
            line=dict(color="#FF5C7A", dash="dot", width=1),
            annotation_text=f"{t}°F",
            annotation_position="right",
        )
    fig.update_layout(**_layout(uirevision=region, yaxis_title="°F"))
    return fig


def _build_timeline_figure(region):
    """Historical extreme-event timeline, highlighting events in this region."""
    fig = go.Figure()
    for date, name, reg, sev in _HISTORICAL_EVENTS:
        color = COLORS["ensemble"] if reg == region else "#A8B3C7"
        fig.add_trace(
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
    fig.update_layout(
        **_layout(
            uirevision=region,
            xaxis_title="Date",
            yaxis_title="Severity Score",
            yaxis_range=[0, 100],
        )
    )
    return fig


def _render_risk_tab(
    *,
    region,
    alerts,
    alerts_source,
    stress,
    stress_label,
    counts,
    anomaly,
    temperature,
    weather_context,
    alerts_total=None,
):
    """Build the Risk tab's 8-output tuple from already-resolved data.

    The single render path for both the Redis and dev/demo data sources
    (P2-44 / #273). Callers differ only in where ``alerts`` / ``anomaly`` /
    ``temperature`` came from; everything below this line is identical.

    ``stress`` of ``None`` renders as an em dash, never the string "None" —
    ``models.pricing.grid_stress`` returns ``None`` deliberately for BAs
    without a reliable capacity plate (#254), and the dev path used to print
    it verbatim.
    """
    return (
        _build_alert_cards(alerts, alerts_source, alerts_total),
        "—" if stress is None else str(stress),
        html.Span(stress_label, className=f"kpi-delta {_stress_tone(stress)}"),
        _build_stress_breakdown(counts, bool(alerts), alerts_source),
        _build_anomaly_figure(
            region,
            anomaly.get("timestamps", []),
            anomaly.get("demand", []),
            anomaly.get("upper", []),
            anomaly.get("lower", []),
            anomaly.get("anomaly_timestamps", []),
            anomaly.get("anomaly_values", []),
        ),
        _build_temp_figure(region, temperature.get("timestamps", []), temperature.get("values", [])),
        _build_timeline_figure(region),
        weather_context,
    )


def _alerts_tab_from_redis(region):
    """Redis fast path for update_alerts_tab callback.

    Returns an 8-tuple (alert_cards, stress_str, stress_label_span, breakdown,
    fig_anomaly, fig_temp, fig_timeline, weather_context) or None if cache miss.
    """
    cached = redis_get(redis_key(f"alerts:{region}"))
    if cached is None:
        return None

    log.info("alerts_redis_hit", region=region)
    alerts = cached.get("alerts", [])
    # Legacy payloads (pre-alerts_source) only ever carried demo content.
    alerts_source = cached.get("alerts_source", "demo")
    stress = cached.get("stress_score")
    stress_label = cached.get("stress_label", "Unavailable")
    counts = cached.get("alert_counts", {})
    anomaly = dict(cached.get("anomaly", {}))
    temp_data = cached.get("temperature", {})
    t_vals = temp_data.get("values", [])
    anomaly["timestamps"] = pd.to_datetime(anomaly.get("timestamps", []))
    anomaly["anomaly_timestamps"] = pd.to_datetime(anomaly.get("anomaly_timestamps", []))

    # Cards, breakdown and figures are built by the shared renderer below;
    # this path's only job is resolving the data out of the payload.
    # Weather context — the full "Current Conditions" card row (temperature +
    # wind + humidity + cloud), built from the scoring job's latest reading. The
    # web tier used to only have the temperature series, so it rendered a lone
    # Temperature card; the scoring job now ships ``weather_current`` so this
    # matches the dev path's multi-card view.
    weather_context = html.Div()
    weather_current = cached.get("weather_current")
    if weather_current:
        from components._callbacks_overview import _build_weather_context

        weather_context = _build_weather_context(pd.Series(weather_current))
    elif t_vals:
        # Back-compat: older payloads (pre this change, or Redis not yet
        # refreshed) carry only the temperature series — show that one card.
        import dash_bootstrap_components as dbc

        last_temp = float(t_vals[-1])
        if pd.isna(last_temp):
            temp_display = "—"
            color = "#A8B3C7"
        else:
            temp_display = f"{last_temp:.0f}°F"
            color = "#FF5C7A" if last_temp >= 95 else ("#FFB84D" if last_temp >= 85 else "#2BD67B")
        weather_context = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.P("TEMPERATURE", className="kpi-label"),
                            html.H4(
                                temp_display,
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

    return _render_risk_tab(
        region=region,
        alerts=alerts,
        alerts_source=alerts_source,
        alerts_total=cached.get("alerts_total"),
        stress=stress,
        stress_label=stress_label,
        counts=counts,
        anomaly=anomaly,
        temperature=temp_data,
        weather_context=weather_context,
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
        # fallback below must never render as real data. P2-35 (#273):
        # when the pipeline is demonstrably alive (fresh actuals for this
        # region) yet the alert payload never lands, "will populate after
        # the next scoring run" is a forever-lie — escalate to an honest
        # persistent-unavailable state instead.
        if REQUIRE_REDIS:
            from components._callbacks_shared import (
                _pipeline_alive,
                _scoring_pass_completed_since_actuals,
            )
            from components.error_handling import warming_state

            # Escalate only on completed-pass evidence: within one scoring
            # pass a region's actuals land BEFORE its alert payload, so
            # fresh actuals alone would render the permanence claim during
            # the first pass after a flush (verification catch).
            if region and _pipeline_alive(region) and _scoring_pass_completed_since_actuals(region):
                log.info("alerts_unavailable_gate", region=region)
                state_card = warming_state(
                    title="Risk data unavailable",
                    message=(
                        "The scoring pipeline is live, but no alert/anomaly "
                        "payload exists for this region — its alert phase is "
                        "not producing data. This won't resolve on its own."
                    ),
                )
                chip_text = "Unavailable"
            else:
                log.info("alerts_warming_gate", region=region)
                state_card = warming_state(
                    title="Risk data is warming up",
                    message=(
                        "The scheduled pipeline has not published alert/anomaly "
                        "data for this region yet. This page will populate after "
                        "the next scoring run."
                    ),
                )
                chip_text = "Warming"
            return (
                [state_card],
                "—",
                html.Span(chip_text),
                html.Div(),
                empty,
                empty,
                empty,
                html.Div(),
            )

        # ── v1 compute fallback (dev/demo only) ─────────────
        from data.demo_data import generate_demo_alerts
        from models.pricing import grid_stress

        alerts = generate_demo_alerts(region)
        counts = {
            sev: sum(1 for a in alerts if a["severity"] == sev)
            for sev in ("critical", "warning", "info")
        }

        # Grid stress = demand / capacity supply tightness, not alert count
        # (#265). Same helper the scoring job calls, so the two paths cannot
        # disagree on the definition; they differ only in the demand reading
        # they have available.
        current_demand = None
        try:
            _dd = pd.read_json(io.StringIO(demand_json)) if demand_json else None
            if _dd is not None and not _dd.empty and "demand_mw" in _dd:
                _s = _dd["demand_mw"].dropna()
                current_demand = float(_s.iloc[-1]) if len(_s) else None
        except Exception:
            current_demand = None
        stress, stress_label = grid_stress(region, current_demand)

        # Anomaly band: compute the +/-2 sigma envelope over the FULL series,
        # then slice to the displayed 168h window -- so the window is warm at
        # its start and the bands span the whole demand line rather than
        # beginning a day late.
        anomaly = {}
        if demand_json:
            demand_df = pd.read_json(io.StringIO(demand_json))
            demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"])
            roll_mean_full = demand_df["demand_mw"].rolling(24, min_periods=1).mean()
            roll_std_full = demand_df["demand_mw"].rolling(24, min_periods=2).std()
            recent = demand_df.tail(168)
            upper = (roll_mean_full + 2 * roll_std_full).tail(168)
            lower = (roll_mean_full - 2 * roll_std_full).tail(168)
            anomalies = recent[recent["demand_mw"] > upper]
            anomaly = {
                "timestamps": recent["timestamp"],
                "demand": recent["demand_mw"],
                "upper": upper,
                "lower": lower,
                "anomaly_timestamps": anomalies["timestamp"],
                "anomaly_values": anomalies["demand_mw"],
            }

        temperature = {}
        if weather_json:
            weather_df = pd.read_json(io.StringIO(weather_json))
            weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
            recent_w = weather_df.tail(168)
            temperature = {
                "timestamps": recent_w["timestamp"],
                "values": recent_w["temperature_2m"],
            }


        # Build weather context from latest reading
        weather_context = html.Div()
        if weather_json:
            try:
                w_df = pd.read_json(io.StringIO(weather_json))
                if not w_df.empty:
                    weather_context = _build_weather_context(w_df.iloc[-1])
            except Exception:
                log.warning("weather_context_build_failed")

        return _render_risk_tab(
            region=region,
            alerts=alerts,
            alerts_source="demo",
            stress=stress,
            stress_label=stress_label,
            counts=counts,
            anomaly=anomaly,
            temperature=temperature,
            weather_context=weather_context,
        )


__all__ = [
    "_alerts_tab_from_redis",
    "register_alerts_callbacks",
]
