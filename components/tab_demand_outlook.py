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
                    # 7. Footer
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
