"""
Tab: Demand Outlook - Forward-Looking Forecast.

Shows forecasted demand for 24 hours, 7 days, and 30 days
starting from the latest available data point.
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Demand Outlook tab layout."""
    return html.Div(
        [
            # Header with data through date
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Forecast as of: ",
                                        style={"fontSize": "0.9rem", "color": "#666"},
                                    ),
                                    html.Span(
                                        id="outlook-data-through",
                                        children="—",
                                        style={"fontSize": "0.9rem", "fontWeight": "bold"},
                                    ),
                                ],
                                className="mb-2",
                            ),
                        ]
                    ),
                ]
            ),
            # Horizon selector
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Forecast Horizon:",
                                className="fw-bold",
                                style={"fontSize": "0.85rem"},
                            ),
                            dbc.RadioItems(
                                id="outlook-horizon",
                                options=[
                                    {"label": "24 Hours", "value": "24"},
                                    {"label": "7 Days", "value": "168"},
                                    {"label": "30 Days", "value": "720"},
                                ],
                                value="168",
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                "Model:", className="fw-bold", style={"fontSize": "0.85rem"}
                            ),
                            dbc.RadioItems(
                                id="outlook-model",
                                options=[
                                    {"label": "XGBoost (Recommended)", "value": "xgboost"},
                                    {"label": "Ensemble", "value": "ensemble"},
                                ],
                                value="xgboost",
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            # Main forecast chart
            build_chart_container(
                "outlook-chart",
                "Demand Forecast",
                height="400px",
            ),
            # AI insight card
            html.Div(id="tab2-insight-card"),
            # KPI cards row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("PEAK DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="outlook-peak", children="— MW", className="kpi-value"
                                    ),
                                    html.P(
                                        id="outlook-peak-time",
                                        children="",
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
                                    html.P("AVERAGE DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="outlook-avg", children="— MW", className="kpi-value"
                                    ),
                                    html.P("Over forecast period", className="kpi-delta neutral"),
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
                                    html.P("MIN DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="outlook-min", children="— MW", className="kpi-value"
                                    ),
                                    html.P(
                                        id="outlook-min-time",
                                        children="",
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
                                    html.P("DEMAND RANGE", className="kpi-label"),
                                    html.H4(
                                        id="outlook-range", children="— MW", className="kpi-value"
                                    ),
                                    html.P("Peak - Min", className="kpi-delta neutral"),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mt-3 g-2",
            ),
            # Hourly breakdown table (for 24h view)
            html.Div(id="outlook-hourly-table", className="mt-3"),
        ],
        className="p-2",
    )
