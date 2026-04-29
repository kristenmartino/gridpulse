"""Tab: Models (formerly Model Diagnostics / tab-models).

R4c of shell-redesign-v2.md. Restructures the Models tab into the
v2 linear-stack rhythm:

  Title → Model leaderboard MetricsBar → multi-select selector →
  metrics table → residual analysis 3-up grid → error+SHAP 2-up grid
  → InsightCard → footer

All existing component IDs are preserved (tab3-model-selector,
tab3-metrics-table, tab3-residuals-time, -hist, -pred,
tab3-error-heatmap, tab3-shap, tab3-insight-card) so the existing
6-output update_models_tab and 1-output insight callbacks continue
to work without signature changes.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.cards import build_page_footer

_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def _model_selector() -> html.Div:
    """v2-styled multi-select for which models to compare in the diagnostics."""
    return html.Div(
        [
            html.Div("Compare Models", className="gp-control-eyebrow"),
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
                className="gp-segmented gp-segmented--multi",
            ),
        ],
        className="gp-control",
    )


def _metrics_table_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Model Performance", className="gp-panel__eyebrow"),
                    html.Span(
                        "MAPE / RMSE / MAE / R² across selected models",
                        className="gp-panel__hint",
                    ),
                ],
                className="gp-panel__header",
            ),
            html.Div(id="tab3-metrics-table", className="gp-metrics-table-slot"),
        ],
        className="gp-panel",
    )


def _residual_grid() -> html.Div:
    """3-up residual analysis grid (preserves tab3-residuals-* IDs)."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Residuals Over Time", className="gp-control-eyebrow"),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab3-residuals-time",
                            style={"height": "260px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
            html.Div(
                [
                    html.Div("Residual Distribution", className="gp-control-eyebrow"),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab3-residuals-hist",
                            style={"height": "260px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
            html.Div(
                [
                    html.Div("Residuals vs Predicted", className="gp-control-eyebrow"),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab3-residuals-pred",
                            style={"height": "260px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
        ],
        className="gp-residual-grid",
    )


def _error_shap_grid() -> html.Div:
    """2-up: error-by-hour heatmap + SHAP feature importance."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "Forecast Error by Hour",
                        className="gp-control-eyebrow",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab3-error-heatmap",
                            style={"height": "300px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
            html.Div(
                [
                    html.Div(
                        "SHAP Feature Importance · XGBoost",
                        className="gp-control-eyebrow",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="tab3-shap",
                            style={"height": "300px"},
                            config=_GRAPH_CONFIG,
                        ),
                        type="circle",
                        color="#3b82f6",
                    ),
                ],
                className="gp-chart-card",
            ),
        ],
        className="gp-models-secondary",
    )


def layout() -> html.Div:
    """Build the v2 linear-stack Models tab layout."""
    return html.Div(
        [
            html.Div(
                [
                    # 1. Title block (callback fills models-title)
                    html.Div(id="models-title"),
                    # 2. Model leaderboard MetricsBar (callback fills 5-up bar)
                    html.Div(id="models-leaderboard"),
                    # 3. Compare-models multi-select
                    _model_selector(),
                    # 4. Metrics table
                    _metrics_table_card(),
                    # 5. Residual analysis 3-up grid
                    _residual_grid(),
                    # 6. Error + SHAP 2-up grid
                    _error_shap_grid(),
                    # 7. InsightCard (existing tab3-insight-card)
                    html.Div(id="tab3-insight-card", className="gp-insight-card-slot"),
                    # 8. Footer
                    build_page_footer(
                        sources=["EIA", "Open-Meteo"],
                        note="Backtests run on the most recent week of holdout data.",
                    ),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
