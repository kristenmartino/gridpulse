"""Cached model evidence, separated by measurement scope."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.accessibility import model_display_name
from components.cards import build_page_footer

_GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}


def _choices(ident: str, options: list[tuple[str, str]], value: str) -> html.Div:
    """Native radio controls with a shared visual tab baseline."""
    return html.Div(
        dbc.RadioItems(
            id=ident,
            options=[{"label": label, "value": key} for key, label in options],
            value=value,
            inline=True,
            className="gp-models-tabs",
        ),
        className="gp-models-tabbar",
    )


def _graph(ident: str, title: str) -> html.Div:
    """One cached diagnostic figure, with an exact table alternative below."""
    return html.Div(
        [
            html.H3(title, className="gp-control-eyebrow"),
            dcc.Loading(
                dcc.Graph(
                    id=ident, config=_GRAPH_CONFIG, responsive=True, style={"height": "290px"}
                ),
                type="circle",
                color="#3b82f6",
            ),
        ],
        className="gp-chart-card",
    )


def layout() -> html.Div:
    """Build observed error first; retain all graph IDs and native disclosure state."""
    return html.Div(
        [
            html.Div(id="models-title"),
            html.P(
                "Observed scores, training holdouts and backtest residuals use different "
                "targets and clocks. Each section keeps its own source and sample count.",
                className="gp-panel__caption",
            ),
            _choices(
                "models-section",
                [
                    ("observed", "Observed error"),
                    ("training", "Training reference"),
                    ("backtest", "Backtest diagnostics"),
                ],
                "observed",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Observed forecast error", className="gp-panel__eyebrow"),
                            html.P(
                                "Choose the scoring basis before comparing percentages.",
                                className="gp-panel__caption",
                            ),
                        ],
                        className="gp-models-panel-head",
                    ),
                    _choices(
                        "models-observed-basis",
                        [("sampled", "Hourly-sampled lead"), ("first", "First-hour scores")],
                        "sampled",
                    ),
                    html.Div(id="models-drift-panel"),
                    html.Div(id="models-first-hour-panel", style={"display": "none"}),
                ],
                id="models-observed-section",
                className="gp-panel gp-models-tabbed-panel",
            ),
            html.Section(
                [
                    html.H2("Training reference", className="gp-panel__eyebrow"),
                    html.P(
                        "168h recursive holdout · MAPE is mean absolute percentage error; "
                        "RMSE and MAE are MW. These results do not identify the published default "
                        "or predict its next forecast accuracy.",
                        className="gp-panel__caption",
                    ),
                    html.Div(id="models-leaderboard"),
                    html.Div(
                        [
                            html.Div("Show in training comparison", className="gp-control-eyebrow"),
                            dbc.Checklist(
                                id="tab3-model-selector",
                                options=[
                                    {"label": model_display_name(k), "value": k}
                                    for k in ("prophet", "arima", "xgboost", "ensemble")
                                ],
                                value=["prophet", "arima", "xgboost", "ensemble"],
                                inline=True,
                            ),
                            html.P(
                                "This filter changes the training table only. The cached backtest "
                                "series and XGBoost importance remain independent.",
                                className="gp-panel__caption",
                            ),
                        ],
                        className="gp-models-training-filter",
                    ),
                    html.Div(
                        id="tab3-metrics-table",
                        role="region",
                        tabIndex=0,
                        className="gp-models-table-scroll",
                        **{"aria-label": "Training metrics; scroll horizontally"},
                    ),
                ],
                id="models-training-section",
                className="gp-panel gp-models-training",
                style={"display": "none"},
            ),
            html.Section(
                [
                    html.H2("Backtest diagnostics", className="gp-panel__eyebrow"),
                    html.Div(id="tab3-diagnostics-scope"),
                    html.Div(
                        [
                            _graph("tab3-residuals-time", "Residual over time · MW"),
                            _graph("tab3-residuals-hist", "Residual distribution · count"),
                            _graph("tab3-residuals-pred", "Residual vs predicted demand · MW"),
                            _graph("tab3-error-heatmap", "Mean absolute error by target hour · MW"),
                        ],
                        className="gp-models-diagnostic-grid",
                    ),
                    _graph("tab3-shap", "Built-in XGBoost feature importance · unitless"),
                    html.P(
                        "Importance describes the fitted XGBoost model. It is not SHAP, causal "
                        "impact or an explanation of an individual forecast.",
                        className="gp-panel__caption",
                    ),
                    dcc.Store(id="models-diagnostics-data"),
                    html.Details(
                        [
                            html.Summary("Exact diagnostic values and sample counts"),
                            html.Label("Residual rows", htmlFor="models-diagnostics-page"),
                            dbc.Select(
                                id="models-diagnostics-page",
                                value=1,
                                options=[{"label": "Page 1", "value": 1}],
                            ),
                            html.Div(id="tab3-diagnostics-evidence"),
                        ],
                        className="gp-models-exact",
                    ),
                ],
                id="models-backtest-section",
                style={"display": "none"},
            ),
            build_page_footer(
                sources=["EIA", "Open-Meteo"],
                note="Model evidence is read from cached scoring and training outputs.",
            ),
        ],
        className="gp-page gp-models-review gp-section-stack",
    )
