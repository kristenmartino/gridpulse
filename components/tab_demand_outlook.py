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
            html.Div("Horizon", className="gp-control-eyebrow"),
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
            html.Label("Series", htmlFor="outlook-model"),
            dbc.Select(id="outlook-model", options=options, value="published"),
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
                    "Peak forecast",
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
                    "Mean demand",
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
                    "Minimum",
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
    cells[3].style = {"display": "none"}
    return html.Div(
        [cells[0], cells[2], cells[1], cells[3]], className="gp-metrics-bar gp-forecast-summary"
    )


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
            html.Span("+ ", className="gp-panel-toggle__sigil", **{"aria-hidden": "true"}),
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
    return html.Section(
        [
            html.Div(
                [
                    html.H2("Review published issues"),
                    html.P("Compare the same target hours before interpreting a curve change"),
                ],
                className="gp-workspace-panel-head",
            ),
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
            dbc.Tabs(
                [
                    dbc.Tab(label="Run-to-run change", tab_id="runs"),
                    dbc.Tab(label="Later reported demand", tab_id="actuals"),
                ],
                id="published-review-mode",
                active_tab="runs",
                className="gp-context-tabs",
            ),
            html.Div(
                [
                    html.Label("Shared-target window", htmlFor="published-review-window"),
                    dbc.Select(
                        id="published-review-window",
                        value="24",
                        options=[
                            {"label": "24 hours", "value": "24"},
                            {"label": "7 days · long-range", "value": "168"},
                            {"label": "30 days · advanced", "value": "720"},
                        ],
                    ),
                    html.Span("Independent of the outlook chart horizon."),
                ],
                id="published-review-window-control",
                className="gp-run-window-control",
            ),
            html.Div(id="published-review-content"),
        ],
        className="gp-workspace-panel gp-publication-review",
    )


def _context_panel():
    drivers = _panel_drivers()
    drivers.children.children.extend(
        [
            html.Details(
                [
                    html.Summary("Weather alignment and interpretation"),
                    html.P(
                        "Regional model weather is aligned to the selected forecast target hours. It provides context; this panel does not attribute demand or forecast error to weather. Historical error review uses later weather context, not the weather forecast originally issued with the demand run."
                    ),
                ],
                className="gp-review-disclosure",
            ),
            html.Button(
                "Open this weather & demand window →",
                id="forecast-context-open-explorer",
                type="button",
                className="gp-review-button",
            ),
        ]
    )
    return html.Section(
        [
            html.Div(
                [
                    html.H2("Supporting context"),
                    html.P("Weather and generation carry their own source windows"),
                ],
                className="gp-workspace-panel-head",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(label="Weather context", tab_id="weather"),
                    dbc.Tab(label="Observed generation", tab_id="generation"),
                ],
                id="forecast-context-tabs",
                active_tab="weather",
                className="gp-context-tabs",
            ),
            drivers,
            _panel_generation(),
        ],
        className="gp-workspace-panel gp-supporting-context",
    )


def layout() -> html.Div:
    """The approved chart-first workspace, with adjacent source evidence."""
    return html.Div(
        [
            html.Div(id="outlook-title"),
            _replay_panel(),
            html.Div(
                [
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H2("Demand outlook"),
                                            html.P(
                                                [
                                                    "Run generated ",
                                                    html.Span(
                                                        id="outlook-data-through", children="—"
                                                    ),
                                                    " · target timestamps UTC",
                                                ]
                                            ),
                                        ],
                                        className="gp-workspace-panel-head",
                                    ),
                                    html.Div(
                                        [
                                            _model_segmented(),
                                            _horizon_segmented(),
                                            dbc.Checkbox(
                                                id="outlook-show-history",
                                                value=True,
                                                label="Show demand history",
                                            ),
                                        ],
                                        className="gp-forecast-toolbar",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                [
                                                    html.I(style={"borderColor": "#5b9fff"}),
                                                    "Actual",
                                                ],
                                                id="forecast-legend-actual",
                                                type="button",
                                                **{"aria-pressed": "true"},
                                            ),
                                            html.Button(
                                                [
                                                    html.I(style={"borderColor": "#efb347"}),
                                                    html.Span(
                                                        "Published forecast",
                                                        id="forecast-legend-label",
                                                    ),
                                                ],
                                                id="forecast-legend-selected",
                                                type="button",
                                                **{"aria-pressed": "true"},
                                            ),
                                        ],
                                        className="gp-forecast-legend",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "Inspect with ← → · target timestamps UTC · MW",
                                                id="forecast-chart-inspection",
                                                role="status",
                                                **{"aria-live": "polite"},
                                            ),
                                            html.Button(
                                                "Zoom in",
                                                id="forecast-chart-zoom",
                                                n_clicks=0,
                                                className="gp-review-button",
                                            ),
                                            html.Button(
                                                "Reset",
                                                id="forecast-chart-reset",
                                                n_clicks=0,
                                                className="gp-review-button",
                                            ),
                                        ],
                                        className="gp-chart-tools",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Loading(
                                                dcc.Graph(
                                                    id="outlook-chart",
                                                    style={"height": "350px"},
                                                    config=_GRAPH_CONFIG,
                                                ),
                                                type="circle",
                                                color="#5b9fff",
                                            ),
                                        ],
                                        id="forecast-chart-keyboard",
                                        tabIndex=0,
                                        role="region",
                                        **{
                                            "aria-label": "Demand outlook chart. Arrow keys inspect forecast targets; Escape clears the inspection."
                                        },
                                    ),
                                    html.P(id="forecast-ramp", className="gp-forecast-ramp"),
                                    _metrics_bar(),
                                    html.P(
                                        id="forecast-curve-note",
                                        className="gp-workspace-panel-foot",
                                    ),
                                ],
                                className="gp-workspace-panel gp-forecast-outlook",
                            ),
                            html.Div(id="tab2-insight-card", style={"display": "none"}),
                            html.Details(
                                [
                                    html.Summary(
                                        "Forecast targets · exact values and input evidence"
                                    ),
                                    html.Div(id="forecast-target-evidence"),
                                ],
                                className="gp-review-disclosure",
                            ),
                            _publication_review(),
                            _context_panel(),
                            html.Details(
                                [
                                    html.Summary("Advanced · 30-day extended outlook"),
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
                                className="gp-review-disclosure gp-forecast-extended",
                            ),
                            html.Div(
                                [
                                    _panel_toggle(
                                        "scenarios",
                                        "Advanced · experimental weather sensitivity",
                                        "Experimental",
                                    ),
                                    _panel_scenarios(),
                                ],
                                className="gp-forecast-advanced",
                            ),
                            html.Div(
                                [
                                    _panel_toggle("drivers", "Weather context", "Aligned targets"),
                                    _panel_toggle("generation", "Observed generation", "Fuel mix"),
                                ],
                                style={"display": "none"},
                            ),
                        ],
                        className="gp-workspace-stack",
                    ),
                    html.Aside(
                        [
                            html.Div(id="outlook-model-card"),
                            html.Section(
                                [
                                    html.Div("Next step", className="gp-control-eyebrow"),
                                    html.H2("Investigate the error"),
                                    html.P(
                                        "Compare coverage, sampling lead and source before interpreting a percentage."
                                    ),
                                    html.Button(
                                        "Open error review →",
                                        id="forecast-open-error",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                    html.Button(
                                        "Check weather & demand →",
                                        id="forecast-open-explorer",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                    html.P(
                                        "Future reviews the current cached forecast. Error reviews the selected stored published issue with later eligible actuals. Explorer windows and source selections are retained independently.",
                                        className="gp-panel__caption",
                                    ),
                                ],
                                className="gp-workspace-panel gp-workspace-rail gp-forecast-next-step",
                            ),
                        ],
                        className="gp-workspace-stack",
                    ),
                ],
                className="gp-workspace-split",
            ),
            build_page_footer(),
        ],
        className="gp-page gp-section-stack gp-forecast-page",
    )
