"""Tab: Forecast (formerly Demand Outlook).

R4a-1 of shell-redesign-v2.md. Rebuilt to the v2 linear-stack rhythm:
title / controls / hero chart / MetricsBar / ModelCard / InsightCard /
footer. R4a-2 will add the toggle strip + Drivers / Generation /
Scenario inline panels beneath the InsightCard.

All existing component IDs (outlook-chart, outlook-horizon, outlook-model,
outlook-peak, outlook-avg, outlook-min, outlook-range, outlook-peak-time,
outlook-min-time, outlook-data-through, tab2-insight-card, replay-*) are
preserved so the multi-output ``update_demand_outlook`` callback continues
to work without a signature change. The new visual structure wraps the
existing IDs as inner spans of MetricsBar cells / ModelMetricsCard slots.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.cards import build_page_footer

_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def _horizon_segmented() -> html.Div:
    """v2-style segmented control for forecast horizon."""
    return html.Div(
        [
            html.Div("Forecast Horizon", className="gp-control-eyebrow"),
            dbc.RadioItems(
                id="outlook-horizon",
                options=[
                    {"label": "24h", "value": "24"},
                    {"label": "7d", "value": "168"},
                    {"label": "30d", "value": "720"},
                ],
                value="168",
                inline=True,
                className="gp-segmented",
            ),
        ],
        className="gp-control",
    )


def _model_segmented() -> html.Div:
    """v2-style segmented control for active model."""
    return html.Div(
        [
            html.Div("Model", className="gp-control-eyebrow"),
            dbc.RadioItems(
                id="outlook-model",
                options=[
                    {"label": "XGBoost", "value": "xgboost"},
                    {"label": "Prophet", "value": "prophet"},
                    {"label": "ARIMA", "value": "arima"},
                    {"label": "Ensemble", "value": "ensemble"},
                ],
                value="xgboost",
                inline=True,
                className="gp-segmented",
            ),
        ],
        className="gp-control",
    )


def _replay_panel() -> html.Div:
    """Hidden replay panel — preserved for the existing replay callback.

    R4a-2 may surface this inside the future Drivers / Scenario panels;
    for now it stays in the DOM but ``style=display:none`` keeps it
    out of the visible flow.
    """
    return html.Div(
        id="replay-container",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Forecast Replay:",
                                className="fw-bold",
                                style={"fontSize": "0.85rem"},
                            ),
                            dcc.Dropdown(
                                id="replay-selector",
                                options=[{"label": "Current", "value": "current"}],
                                value="current",
                                clearable=False,
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                id="replay-label",
                                children="",
                                style={
                                    "fontSize": "0.8rem",
                                    "color": "var(--text-tertiary)",
                                    "paddingTop": "28px",
                                },
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
        ],
        style={"display": "none"},
    )


def _metrics_bar() -> html.Div:
    """4-up MetricsBar matching v2 — wraps existing IDs as inner spans.

    Cells: Peak Demand · Average · Min · Range. The existing callback
    populates the children of each ID; this layout just provides the
    structural shell + label/unit text.
    """
    cells = [
        # Peak (hero)
        html.Div(
            [
                html.Div("Peak Demand", className="gp-metric-label"),
                html.Div(
                    [
                        html.Span(
                            id="outlook-peak",
                            children="—",
                            className="gp-metric-value gp-metric-value--hero tabular",
                        ),
                        html.Span("MW", className="gp-metric-unit"),
                    ],
                    className="gp-metric-value-row",
                ),
                html.Div(id="outlook-peak-time", className="gp-metric-sub"),
            ],
            className="gp-metric-cell",
        ),
        # Average
        html.Div(
            [
                html.Div("Average", className="gp-metric-label"),
                html.Div(
                    [
                        html.Span(
                            id="outlook-avg",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
                        html.Span("MW", className="gp-metric-unit"),
                    ],
                    className="gp-metric-value-row",
                ),
                html.Div("Forecast period", className="gp-metric-sub"),
            ],
            className="gp-metric-cell",
        ),
        # Min
        html.Div(
            [
                html.Div("Min Demand", className="gp-metric-label"),
                html.Div(
                    [
                        html.Span(
                            id="outlook-min",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
                        html.Span("MW", className="gp-metric-unit"),
                    ],
                    className="gp-metric-value-row",
                ),
                html.Div(id="outlook-min-time", className="gp-metric-sub"),
            ],
            className="gp-metric-cell",
        ),
        # Range
        html.Div(
            [
                html.Div("Range", className="gp-metric-label"),
                html.Div(
                    [
                        html.Span(
                            id="outlook-range",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
                        html.Span("MW", className="gp-metric-unit"),
                    ],
                    className="gp-metric-value-row",
                ),
                html.Div("Peak − Min", className="gp-metric-sub"),
            ],
            className="gp-metric-cell",
        ),
    ]
    return html.Div(cells, className="gp-metrics-bar gp-metrics-bar--4up")


def _panel_toggle_strip() -> html.Div:
    """Three chip-buttons that expand inline panels under the Forecast tab.

    Each toggle drives a ``dbc.Collapse`` whose ``is_open`` is toggled by
    a tiny clientside callback. Lazy server-side rendering populates the
    panel content only when it opens (saves render cost when closed).
    """
    return html.Div(
        [
            html.Div("Expand", className="gp-control-eyebrow"),
            html.Div(
                [
                    _panel_toggle("drivers", "Drivers", "Weather"),
                    _panel_toggle("generation", "Generation", "Fuel mix"),
                    _panel_toggle("scenarios", "Scenarios", "What-if"),
                ],
                className="gp-panel-toggles",
            ),
        ],
        className="gp-control",
    )


def _panel_toggle(key: str, label: str, hint: str) -> html.Button:
    """Single toggle chip. Click flips the collapse panel's ``is_open``."""
    return html.Button(
        [
            html.Span("+ ", className="gp-panel-toggle__sigil"),
            html.Span(label, className="gp-panel-toggle__label"),
            html.Span(hint, className="gp-panel-toggle__hint"),
        ],
        id=f"forecast-panel-toggle-{key}",
        n_clicks=0,
        type="button",
        className="gp-panel-toggle",
        **{"aria-expanded": "false", "aria-controls": f"forecast-panel-{key}"},
    )


def _panel_header(eyebrow: str, hint: str | None = None) -> html.Div:
    children: list = [html.Div(eyebrow, className="gp-panel__eyebrow")]
    if hint:
        children.append(html.Span(hint, className="gp-panel__hint"))
    return html.Div(children, className="gp-panel__header")


def _panel_drivers() -> dbc.Collapse:
    """Drivers panel: 3-up KPI mini-bar (Temperature / Wind / Solar)."""
    return dbc.Collapse(
        html.Div(
            [
                _panel_header(
                    "Drivers",
                    "Weather signals shaping the next-24h forecast",
                ),
                # Filled by the update_forecast_drivers_panel callback when
                # the collapse opens. Initial children are skeleton cells.
                html.Div(
                    id="forecast-drivers-content",
                    children=_drivers_skeleton(),
                    className="gp-drivers-grid",
                ),
            ],
            className="gp-panel",
            id="forecast-panel-drivers",
        ),
        id="forecast-panel-drivers-collapse",
        is_open=False,
    )


def _drivers_skeleton() -> list:
    """3 placeholder cells while the Drivers callback computes."""
    cells = []
    for label in ("Temperature", "Wind", "Solar"):
        cells.append(
            html.Div(
                [
                    html.Div(label, className="gp-metric-label"),
                    html.Span("—", className="gp-metric-value tabular"),
                    html.Div("Loading…", className="gp-metric-sub"),
                ],
                className="gp-driver-cell",
            )
        )
    return cells


def _panel_generation() -> dbc.Collapse:
    """Generation panel: stacked-area fuel mix + renewable share sub-MetricsBar."""
    return dbc.Collapse(
        html.Div(
            [
                _panel_header(
                    "Generation",
                    "Fuel mix sorted by emissions intensity",
                ),
                html.Div(
                    id="forecast-generation-content",
                    children=html.Div(
                        "Loading generation data…",
                        className="gp-panel__placeholder",
                    ),
                ),
            ],
            className="gp-panel",
            id="forecast-panel-generation",
        ),
        id="forecast-panel-generation-collapse",
        is_open=False,
    )


_SCENARIO_PRESETS: dict[str, dict] = {
    # 5 preset chips — heuristic deltas vs the current baseline (not absolute
    # weather targets). Translated from simulation/presets.py historical
    # scenarios into "delta from typical conditions".
    "heat_dome": {
        "label": "Heat Dome",
        "sub": "+25°F · clear sky",
        "deltas": {"temp": 25, "wind": -5, "solar": 200},
    },
    "polar_vortex": {
        "label": "Polar Vortex",
        "sub": "−30°F · windy",
        "deltas": {"temp": -30, "wind": 10, "solar": -100},
    },
    "calm_overcast": {
        "label": "Calm Overcast",
        "sub": "Δ0°F · still / cloudy",
        "deltas": {"temp": 0, "wind": -8, "solar": -150},
    },
    "windy_cool": {
        "label": "Windy Cool",
        "sub": "−10°F · 20 mph",
        "deltas": {"temp": -10, "wind": 8, "solar": -50},
    },
    "eclipse": {
        "label": "Solar Eclipse",
        "sub": "Δ0°F · solar zero",
        "deltas": {"temp": 0, "wind": 0, "solar": -200},
    },
}


def _scenario_preset_chips() -> list:
    """Five preset buttons — clicking each writes deltas into the slider store."""
    return [
        html.Button(
            [
                html.Span(info["label"], className="gp-preset-chip__label"),
                html.Span(info["sub"], className="gp-preset-chip__sub"),
            ],
            id={"type": "scenario-preset", "index": key},
            n_clicks=0,
            type="button",
            className="gp-preset-chip",
        )
        for key, info in _SCENARIO_PRESETS.items()
    ]


def _scenario_slider(slider_id: str, label: str, lo: int, hi: int, unit: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, className="gp-slider__label"),
                    html.Span(
                        id=f"forecast-scn-{slider_id}-readout",
                        children=f"0 {unit}",
                        className="gp-slider__readout tabular",
                    ),
                ],
                className="gp-slider__header",
            ),
            dcc.Slider(
                id=f"forecast-scn-{slider_id}",
                min=lo,
                max=hi,
                step=1,
                value=0,
                marks={lo: f"{lo}", 0: "0", hi: f"+{hi}"},
                updatemode="mouseup",
                tooltip={"placement": "bottom", "always_visible": False},
                className="gp-slider",
            ),
        ],
        className="gp-slider-row",
    )


def _panel_scenarios() -> dbc.Collapse:
    """Scenarios panel: preset chips + 3 sliders + delta KPIs + comparison chart."""
    return dbc.Collapse(
        html.Div(
            [
                _panel_header(
                    "Scenarios",
                    "Stress-test demand against weather shifts",
                ),
                html.Div(_scenario_preset_chips(), className="gp-preset-chips"),
                html.Div(
                    [
                        _scenario_slider("temp", "Temperature Δ", -20, 20, "°F"),
                        _scenario_slider("wind", "Wind Δ", -10, 10, "mph"),
                        _scenario_slider("solar", "Solar Δ", -200, 200, "W/m²"),
                    ],
                    className="gp-scenario-sliders",
                ),
                # 4-up delta KPI bar (callback fills)
                html.Div(id="forecast-scenarios-kpis"),
                # Baseline vs scenario chart (callback fills)
                dcc.Graph(
                    id="forecast-scenarios-chart",
                    style={"height": "240px"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
            className="gp-panel",
            id="forecast-panel-scenarios",
        ),
        id="forecast-panel-scenarios-collapse",
        is_open=False,
    )


def layout() -> html.Div:
    """Build the v2 linear-stack Forecast tab layout."""
    return html.Div(
        [
            html.Div(
                [
                    # 1. Title block (callback fills outlook-title)
                    html.Div(id="outlook-title"),
                    # 2. Controls row (segmented horizon + model)
                    html.Div(
                        [_horizon_segmented(), _model_segmented()],
                        className="gp-controls-row",
                    ),
                    # 2b. Replay panel (hidden carrier — preserved for callback)
                    _replay_panel(),
                    # 3. Hero forecast chart (full-width, with confidence band)
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                id="outlook-chart",
                                style={"height": "380px"},
                                config=_GRAPH_CONFIG,
                            ),
                            type="circle",
                            color="#3b82f6",
                        ),
                        className="gp-chart-card",
                    ),
                    # 3b. Forecast as-of (preserved id, now rendered as a chip)
                    html.Div(
                        [
                            html.Span("Forecast as of ", className="gp-chip-label"),
                            html.Span(
                                id="outlook-data-through",
                                children="—",
                                className="gp-chip-value tabular",
                            ),
                        ],
                        className="gp-as-of-chip",
                    ),
                    # 4. MetricsBar (4-up, reuses existing outlook-* IDs)
                    _metrics_bar(),
                    # 5. ModelMetricsCard (callback fills)
                    html.Div(id="outlook-model-card"),
                    # 6. InsightCard (existing id; styled by the wrapper)
                    html.Div(id="tab2-insight-card", className="gp-insight-card-slot"),
                    # 7. Inline panel toggle strip (R4a-2)
                    _panel_toggle_strip(),
                    # 7a. Drivers panel (R4a-2 — working)
                    _panel_drivers(),
                    # 7b. Generation panel (R4a-3 — working)
                    _panel_generation(),
                    # 7c. Scenarios panel (R4a-4 — working)
                    _panel_scenarios(),
                    # 8. Footer
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
