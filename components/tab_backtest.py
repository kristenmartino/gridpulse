"""
Tab 7: Backtest - Forecast Accuracy Analysis.

Shows forecast vs actual demand comparison with different forecast horizons.
Allows users to visualize model accuracy at 24h, 7-day, and 30-day horizons.
"""

import dash_bootstrap_components as dbc
from dash import html, dcc

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Tab 7 (Backtest) layout."""
    return html.Div([
        # Header with description
        dbc.Row([
            dbc.Col([
                html.P(
                    "Compare forecast predictions against actual demand at different forecast horizons. "
                    "Select a horizon to see how accuracy varies based on how far ahead the forecast was made.",
                    className="text-muted mb-3",
                    style={"fontSize": "0.9rem"},
                ),
            ]),
        ]),

        # Controls row
        dbc.Row([
            dbc.Col([
                html.Label("Forecast Horizon:", className="fw-bold", style={"fontSize": "0.85rem"}),
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
            ], md=6),
            dbc.Col([
                html.Label("Model:", className="fw-bold", style={"fontSize": "0.85rem"}),
                dbc.RadioItems(
                    id="backtest-model",
                    options=[
                        {"label": "XGBoost (Recommended)", "value": "xgboost"},
                        {"label": "Ensemble", "value": "ensemble"},
                        {"label": "Prophet", "value": "prophet"},
                        {"label": "ARIMA", "value": "arima"},
                    ],
                    value="xgboost",
                    inline=True,
                    className="mt-1",
                    style={"fontSize": "0.85rem"},
                ),
            ], md=6),
        ], className="mb-3"),

        # Main chart: Forecast vs Actual
        build_chart_container(
            "backtest-chart",
            "Forecast vs Actual Demand",
            height="450px",
        ),

        # Metrics row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P("MAPE", className="kpi-label"),
                    html.H4(id="backtest-mape", children="—%", className="kpi-value"),
                    html.P("Mean Absolute % Error", className="kpi-delta neutral"),
                ], className="kpi-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.P("RMSE", className="kpi-label"),
                    html.H4(id="backtest-rmse", children="— MW", className="kpi-value"),
                    html.P("Root Mean Squared Error", className="kpi-delta neutral"),
                ], className="kpi-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.P("MAE", className="kpi-label"),
                    html.H4(id="backtest-mae", children="— MW", className="kpi-value"),
                    html.P("Mean Absolute Error", className="kpi-delta neutral"),
                ], className="kpi-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.P("R²", className="kpi-label"),
                    html.H4(id="backtest-r2", children="—", className="kpi-value"),
                    html.P("Coefficient of Determination", className="kpi-delta neutral"),
                ], className="kpi-card"),
            ], md=3),
        ], className="mt-3 g-2"),

        # Info panel
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("Horizon Explanation: "),
                    html.Span(id="backtest-horizon-explanation"),
                ], color="info", className="mt-3 mb-0", style={"fontSize": "0.85rem"}),
            ]),
        ]),
    ], className="p-2")
