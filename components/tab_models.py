"""
Models tab: Model comparison, diagnostics, and oversight.

Components:
- Model selector toggle
- Metrics table (MAPE, RMSE, MAE, R² for all 4 models)
- Residual analysis: time series, histogram, vs predicted
- Error by hour heatmap
- SHAP feature importance
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
    """Build the Models tab layout."""
    return html.Div(
        [
            # ── Model Selection ─────────────────────────────
            _section_header("Model Selection", "Select models for comparative oversight"),
            # Model selector
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Checklist(
                                id="tab3-model-selector",
                                options=[
                                    {"label": "Prophet", "value": "prophet"},
                                    {"label": "SARIMAX", "value": "arima"},
                                    {"label": "XGBoost", "value": "xgboost"},
                                    {"label": "Ensemble", "value": "ensemble"},
                                ],
                                value=["prophet", "arima", "xgboost", "ensemble"],
                                inline=True,
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="mb-2",
            ),
            # Metrics table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span("Model Performance Metrics", className="chart-title"),
                                    html.Div(id="tab3-metrics-table"),
                                ],
                                className="chart-container",
                            ),
                        ],
                        md=12,
                    ),
                ]
            ),
            # ── Residual Analysis ────────────────────────
            _section_header("Residual Analysis", "Diagnose systematic error patterns"),
            # Residual analysis row
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab3-residuals-time", "Residuals Over Time", height="280px"
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab3-residuals-hist", "Residual Distribution", height="280px"
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab3-residuals-pred", "Residuals vs Predicted", height="280px"
                        ),
                        md=4,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # ── Error Patterns ───────────────────────────
            _section_header("Error Patterns", "When and why forecasts degrade"),
            # Bottom row: error heatmap + SHAP
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab3-error-heatmap", "Forecast Error by Hour of Day", height="300px"
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab3-shap", "SHAP Feature Importance (XGBoost)", height="300px"
                        ),
                        md=6,
                    ),
                ],
                className="g-2 mt-1",
            ),
        ]
    )
