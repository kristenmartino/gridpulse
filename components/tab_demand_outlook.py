"""
Tab: Demand Outlook - Forward-Looking Forecast.

Shows forecasted demand for 24 hours, 7 days, and 30 days
starting from the latest available data point.
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def _section_header(title: str, subtitle: str) -> html.Div:
    """Render a lightweight section header."""
    return html.Div(
        [
            html.Span(
                title,
                style={
                    "color": "#F7FAFC",
                    "fontSize": "0.85rem",
                    "fontWeight": "600",
                    "marginRight": "8px",
                },
            ),
            html.Span(
                subtitle,
                style={
                    "color": "#A8B3C7",
                    "fontSize": "0.75rem",
                },
            ),
        ],
        style={
            "padding": "10px 0 4px 0",
            "borderBottom": "1px solid #263556",
            "marginTop": "12px",
            "marginBottom": "8px",
        },
    )


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
                                        style={"fontSize": "0.9rem", "color": "#A8B3C7"},
                                    ),
                                    html.Span(
                                        id="outlook-data-through",
                                        children="Loading...",
                                        style={"fontSize": "0.9rem", "fontWeight": "bold"},
                                    ),
                                ],
                                className="mb-2",
                            ),
                        ]
                    ),
                ]
            ),
            # ── Forecast Controls ────────────────────────────
            _section_header("Forecast Controls", "Horizon and model selection"),
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
                                "Model:",
                                className="fw-bold",
                                style={"fontSize": "0.85rem"},
                            ),
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
            # ── Key Metrics ──────────────────────────────
            _section_header("Key Metrics", "Forecast summary statistics"),
            # KPI cards row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("PEAK DEMAND", className="kpi-label"),
                                    html.H4(
                                        id="outlook-peak",
                                        children="Loading...",
                                        className="kpi-value",
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
                                        id="outlook-avg",
                                        children="Loading...",
                                        className="kpi-value",
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
                                        id="outlook-min",
                                        children="Loading...",
                                        className="kpi-value",
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
                                        id="outlook-range",
                                        children="Loading...",
                                        className="kpi-value",
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
        ],
        className="p-2",
    )
