"""
Tab 6: Scenario Simulator ("What-If" Planner) — showstopper feature.

Components:
- Weather scenario builder (temperature, wind, cloud, humidity sliders)
- Preset historical scenarios (one-click replay)
- Impact dashboard (demand delta, price impact, reserve margin, renewable impact)
- Scenario comparison mode (up to 3 overlaid)
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.cards import build_chart_container
from simulation.presets import PRESETS


def layout() -> html.Div:
    """Build Tab 6 layout."""
    return html.Div(
        [
            dbc.Row(
                [
                    # ── Left panel: sliders + presets ──────────────────────
                    dbc.Col(
                        [
                            # Weather sliders
                            html.Div(
                                [
                                    html.H6(
                                        "Weather Scenario Builder",
                                        style={"color": "#ffffff", "marginBottom": "12px"},
                                    ),
                                    _slider("Temperature (°F)", "sim-temp", -10, 120, 75, 1, "°F"),
                                    _slider("Wind Speed (mph)", "sim-wind", 0, 80, 15, 1, "mph"),
                                    _slider("Cloud Cover (%)", "sim-cloud", 0, 100, 50, 5, "%"),
                                    _slider("Humidity (%)", "sim-humidity", 0, 100, 60, 5, "%"),
                                    _slider(
                                        "Solar Irradiance (W/m²)",
                                        "sim-solar",
                                        0,
                                        1000,
                                        500,
                                        25,
                                        "W/m²",
                                    ),
                                    # Duration
                                    html.Div(
                                        [
                                            html.Label("Duration", className="slider-label"),
                                            dbc.RadioItems(
                                                id="sim-duration",
                                                options=[
                                                    {"label": "24h", "value": 24},
                                                    {"label": "48h", "value": 48},
                                                    {"label": "72h", "value": 72},
                                                    {"label": "7d", "value": 168},
                                                ],
                                                value=24,
                                                inline=True,
                                                style={"fontSize": "0.8rem"},
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    # Action buttons
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "▶ Run Scenario",
                                                    id="sim-run-btn",
                                                    color="danger",
                                                    size="sm",
                                                    className="w-100",
                                                ),
                                                md=6,
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "↺ Reset",
                                                    id="sim-reset-btn",
                                                    outline=True,
                                                    color="secondary",
                                                    size="sm",
                                                    className="w-100",
                                                ),
                                                md=6,
                                            ),
                                        ],
                                        className="g-2",
                                    ),
                                ],
                                className="slider-container",
                            ),
                            # Preset scenarios
                            html.Div(
                                [
                                    html.H6(
                                        "Preset Historical Scenarios",
                                        style={"color": "#ffffff", "marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        [
                                            _preset_button(key, preset)
                                            for key, preset in PRESETS.items()
                                        ]
                                    ),
                                ],
                                className="slider-container mt-2",
                            ),
                        ],
                        md=3,
                        style={"maxHeight": "85vh", "overflowY": "auto"},
                    ),
                    # ── Right panel: impact dashboard ─────────────────────
                    dbc.Col(
                        [
                            # Impact KPIs
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.P("DEMAND DELTA", className="kpi-label"),
                                                    html.H4(
                                                        id="sim-demand-delta",
                                                        children="Loading...",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="sim-demand-delta-pct",
                                                        className="kpi-delta neutral",
                                                    ),
                                                ],
                                                className="kpi-card",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.P("PRICE IMPACT", className="kpi-label"),
                                                    html.H4(
                                                        id="sim-price-impact",
                                                        children="Loading...",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="sim-price-delta",
                                                        className="kpi-delta neutral",
                                                    ),
                                                ],
                                                className="kpi-card",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.P("RESERVE MARGIN", className="kpi-label"),
                                                    html.H4(
                                                        id="sim-reserve-margin",
                                                        children="Loading...",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="sim-reserve-status",
                                                        className="kpi-delta neutral",
                                                    ),
                                                ],
                                                className="kpi-card",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    html.P(
                                                        "RENEWABLE IMPACT", className="kpi-label"
                                                    ),
                                                    html.H4(
                                                        id="sim-renewable-impact",
                                                        children="Loading...",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="sim-renewable-detail",
                                                        className="kpi-delta neutral",
                                                    ),
                                                ],
                                                className="kpi-card",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                ],
                                className="g-2 mb-2",
                            ),
                            # Main comparison chart
                            build_chart_container(
                                "sim-forecast-chart",
                                "Baseline vs Scenario Forecast",
                                height="340px",
                            ),
                            # Bottom row: price curve + renewable breakdown
                            dbc.Row(
                                [
                                    dbc.Col(
                                        build_chart_container(
                                            "sim-price-chart", "Price Impact Curve", height="280px"
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        build_chart_container(
                                            "sim-renewable-chart",
                                            "Renewable Generation Impact",
                                            height="280px",
                                        ),
                                        md=6,
                                    ),
                                ],
                                className="g-2 mt-1",
                            ),
                        ],
                        md=9,
                    ),
                ],
                className="g-2",
            ),
            # Hidden stores for scenario state
            dcc.Store(id="sim-baseline-store"),
            dcc.Store(id="sim-scenario-store"),
        ]
    )


def _slider(
    label: str,
    slider_id: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float,
    unit: str,
) -> html.Div:
    """Build a labeled slider with value display."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label(label, className="slider-label"),
                    html.Span(
                        f"{default}{unit}",
                        id=f"{slider_id}-display",
                        className="slider-value",
                        style={"float": "right"},
                    ),
                ]
            ),
            dcc.Slider(
                id=slider_id,
                min=min_val,
                max=max_val,
                step=step,
                value=default,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="mb-3",
    )


def _preset_button(key: str, preset: dict) -> html.Button:
    """Build a preset scenario button."""
    return html.Button(
        [
            html.Span(preset["name"], className="preset-name"),
            html.Span(
                f"{preset['region']} • {preset['date']}",
                className="preset-meta",
            ),
        ],
        id={"type": "preset-btn", "index": key},
        className="preset-btn",
        n_clicks=0,
    )
