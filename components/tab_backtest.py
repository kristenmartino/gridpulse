"""
Tab 7: Backtest - Forecast Accuracy Analysis.

Shows forecast vs actual demand comparison with different forecast horizons.
Allows users to visualize model accuracy at 24h, 7-day, and 30-day horizons.
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
    """Build Tab 7 (Backtest) layout."""
    return html.Div(
        [
            # Header with description
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                "Compare forecast predictions against actual demand at different forecast horizons. "
                                "Select a horizon to see how accuracy varies based on how far ahead the forecast was made.",
                                className="text-muted mb-3",
                                style={"fontSize": "0.9rem"},
                            ),
                        ]
                    ),
                ]
            ),
            # Controls row
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
                                id="backtest-horizon",
                                options=[
                                    {"label": "24 hours ahead", "value": "24"},
                                    {"label": "7 days ahead", "value": "168"},
                                    {"label": "30 days ahead", "value": "720"},
                                ],
                                value="24",
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=12,
                    ),
                    # Hidden model selector — XGBoost only (other models deferred)
                    dbc.RadioItems(
                        id="backtest-model",
                        options=[{"label": "XGBoost", "value": "xgboost"}],
                        value="xgboost",
                        style={"display": "none"},
                    ),
                ],
                className="mb-3",
            ),
            # Main chart: Forecast vs Actual
            build_chart_container(
                "backtest-chart",
                "Forecast vs Actual Demand",
                height="450px",
            ),
            # AI insight card
            html.Div(id="tab3-insight-card"),
            # ── Accuracy Metrics ─────────────────────────
            _section_header("Accuracy Metrics", "Model performance on holdout data"),
            # Metrics row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("MAPE", className="kpi-label"),
                                    html.H4(
                                        id="backtest-mape",
                                        children="Loading...",
                                        className="kpi-value",
                                    ),
                                    html.P("Mean Absolute % Error", className="kpi-delta neutral"),
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
                                    html.P("RMSE", className="kpi-label"),
                                    html.H4(
                                        id="backtest-rmse",
                                        children="Loading...",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Root Mean Squared Error", className="kpi-delta neutral"
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
                                    html.P("MAE", className="kpi-label"),
                                    html.H4(
                                        id="backtest-mae",
                                        children="Loading...",
                                        className="kpi-value",
                                    ),
                                    html.P("Mean Absolute Error", className="kpi-delta neutral"),
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
                                    html.P("R²", className="kpi-label"),
                                    html.H4(
                                        id="backtest-r2",
                                        children="Loading...",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Coefficient of Determination",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mt-3 g-2",
            ),
            # ── Horizon Analysis ─────────────────────────
            _section_header("Horizon Analysis", "How accuracy varies with forecast distance"),
            # Info panel
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Alert(
                                [
                                    html.I(className="bi bi-info-circle me-2"),
                                    html.Strong("Horizon Explanation: "),
                                    html.Span(id="backtest-horizon-explanation"),
                                ],
                                color="info",
                                className="mt-3 mb-0",
                                style={"fontSize": "0.85rem"},
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="p-2",
    )
