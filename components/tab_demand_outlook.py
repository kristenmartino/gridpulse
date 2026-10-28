"""Forecast view with explicit source and error context above the chart.

Existing outlook, replay, and inline-panel IDs are retained for callbacks.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.accessibility import model_display_name
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
                ],
                value="24",
                inline=True,
                className="gp-segmented",
            ),
        ],
        className="gp-control",
    )


def _model_segmented() -> html.Div:
    """Select the forecast source to inspect in this view."""
    options = [{"label": "Published default", "value": "published"}] + [
        {"label": model_display_name(key), "value": key}
        for key in ("xgboost", "prophet", "arima", "ensemble")
    ]

    return html.Div(
        [
            html.Div("Forecast shown", className="gp-control-eyebrow"),
            dbc.RadioItems(
                id="outlook-model",
                options=options,
                value="published",
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
                html.Div(
                    "Peak Demand",
                    className="gp-metric-label",
                    title="Highest forecasted demand over the selected horizon.",
                ),
                html.Div(
                    [
                        html.Span(
                            id="outlook-peak",
                            children="—",
                            className="gp-metric-value gp-metric-value--hero tabular",
                        ),
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
                html.Div(
                    "Average",
                    className="gp-metric-label",
                    title="Mean forecasted demand over the selected horizon.",
                ),
                html.Div(
                    [
                        html.Span(
                            id="outlook-avg",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
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
                html.Div(
                    "Min Demand",
                    className="gp-metric-label",
                    title="Lowest forecasted demand over the selected horizon.",
                ),
                html.Div(
                    [
                        html.Span(
                            id="outlook-min",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
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
                html.Div(
                    "Range",
                    className="gp-metric-label",
                    title="Forecast peak − forecast min over the selected horizon.",
                ),
                html.Div(
                    [
                        html.Span(
                            id="outlook-range",
                            children="—",
                            className="gp-metric-value gp-metric-value--secondary tabular",
                        ),
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
                    _panel_toggle("drivers", "Weather context", "Aligned targets"),
                    _panel_toggle("generation", "Generation", "Fuel mix"),
                    _panel_toggle("scenarios", "Advanced sensitivity", "Experimental"),
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
                    "Weather context",
                    "Regional model weather aligned to the selected target hours",
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
                    "Observed generation",
                    "Its own reported window · separate from forecast targets",
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


_SCENARIO_PRESETS = {
    "reset": {
        "label": "Reset",
        "sub": "All deltas 0",
        "deltas": {"temp": 0, "wind": 0, "solar": 0},
    },
    "warmer": {"label": "Warmer", "sub": "+3°F", "deltas": {"temp": 3, "wind": 0, "solar": 0}},
    "cooler": {"label": "Cooler", "sub": "−3°F", "deltas": {"temp": -3, "wind": 0, "solar": 0}},
    "less_wind": {
        "label": "Less wind",
        "sub": "−2 mph",
        "deltas": {"temp": 0, "wind": -2, "solar": 0},
    },
    "less_sun": {
        "label": "Less sun",
        "sub": "−50 W/m²",
        "deltas": {"temp": 0, "wind": 0, "solar": -50},
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
                    "Compare demand under weather changes",
                ),
                # The source line lives in the callback output (below the
                # KPIs), not here: since #127 this panel may be served either
                # from real forecasts or from the #119 heuristic, and static
                # copy cannot tell the difference. It asserted "not a model
                # re-forecast" for a fortnight while production served exactly
                # that.
                html.P(
                    "Adjust temperature, wind, and solar radiation relative to the reference forecast. "
                    "The results identify the calculation used.",
                    className="gp-panel__disclosure",
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


def _publication_review():
    return html.Details(
        [
            html.Summary("Review published issues and matched actuals"),
            html.P(
                "Choose immutable cached issues to compare shared target hours. Later actuals and weather are context, not original inputs.",
                className="gp-panel__hint",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Earlier issue", htmlFor="published-review-earlier"),
                            dbc.Select(id="published-review-earlier"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Later issue", htmlFor="published-review-later"),
                            dbc.Select(id="published-review-later"),
                        ]
                    ),
                ],
                className="gp-review-toolbar",
            ),
            html.Div(id="published-review-content"),
        ],
        className="gp-review-disclosure",
    )


def layout() -> html.Div:
    """Build the Forecast view with context preceding the demand curve."""
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
                    # Source and matching error context precede the chart.
                    html.Div(id="outlook-model-card"),
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
                    # 6. InsightCard (existing id; styled by the wrapper)
                    html.Div(id="tab2-insight-card", style={"display": "none"}),
                    html.Details(
                        [
                            html.Summary("Forecast evidence and exact targets"),
                            html.Div(id="forecast-target-evidence"),
                        ],
                        className="gp-review-disclosure",
                    ),
                    html.Details(
                        [
                            html.Summary("Advanced · 30-day outlook"),
                            html.P(
                                "Long-range model comparisons may use seasonal weather context. The chart qualifies available cached weather coverage; it is not an original input vintage.",
                                className="gp-muted",
                            ),
                            html.Button(
                                "Show 30-day outlook",
                                id="forecast-extended-open",
                                n_clicks=0,
                                className="gp-review-button",
                            ),
                        ],
                        className="gp-review-disclosure",
                    ),
                    _publication_review(),
                    html.Div(
                        [
                            html.Button(
                                "Open weather & demand →",
                                id="forecast-open-explorer",
                                n_clicks=0,
                                className="gp-review-button",
                            ),
                            html.Button(
                                "Open error review →",
                                id="forecast-open-error",
                                n_clicks=0,
                                className="gp-review-button",
                            ),
                        ],
                        className="gp-explorer-window-actions",
                    ),
                    html.P(
                        "Future reviews the current cached forecast. Error reviews the selected stored published issue with later eligible actuals. Explorer windows and source selections are retained independently.",
                        className="gp-panel__caption",
                    ),
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
