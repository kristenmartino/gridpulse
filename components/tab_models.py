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
    """A native cached chart with visible, keyboard-accessible review controls."""
    units = (
        "Count"
        if ident.endswith("hist")
        else "Relative importance"
        if ident.endswith("shap")
        else "MW"
    )
    notes = {
        "tab3-residuals-time": "Signed actual − prediction across qualified UTC backtest targets",
        "tab3-residuals-hist": "Distribution of cached signed backtest errors; count per bin",
        "tab3-residuals-pred": "Cached predicted-demand / error pairs; fixed source series",
        "tab3-error-heatmap": "Mean absolute error by UTC target hour; sample counts in hover",
        "tab3-shap": "Captured model importance · not SHAP, causality or per-forecast attribution",
    }
    return html.Div(
        [
            html.Div([html.H3(title), html.P(notes[ident])], className="gp-workspace-panel-head"),
            html.Div(id=ident + "-legend", className="gp-model-chart-legend"),
            html.Div(
                [
                    html.Span("Inspect with ← →", className="gp-model-chart-inspect-hint"),
                    html.Span(units, className="gp-model-chart-unit"),
                    html.Button(
                        "Zoom in",
                        className="gp-review-button",
                        **{"data-model-chart": ident, "data-chart-action": "zoom"},
                    ),
                    html.Button(
                        "Reset",
                        className="gp-review-button",
                        **{"data-model-chart": ident, "data-chart-action": "reset"},
                    ),
                ],
                className="gp-model-chart-tools",
            ),
            html.Div(
                dcc.Loading(
                    dcc.Graph(
                        id=ident,
                        config=_GRAPH_CONFIG,
                        responsive=True,
                        style={"height": "260px"},
                        className="gp-models-graph",
                    ),
                    type="circle",
                    color="#3b82f6",
                ),
                className="gp-model-chart-keyboard",
                tabIndex=0,
                role="img",
                **{
                    "aria-label": title
                    + ". Use Left and Right, Home and End to inspect cached values. Escape dismisses inspection."
                },
            ),
            html.Div(
                id=ident + "-inspected",
                className="gp-model-chart-readout",
                role="status",
                **{"aria-live": "polite"},
            ),
        ],
        className="gp-chart-card gp-model-chart-card",
    )


def _panel_head(title: str, caption: str) -> html.Div:
    return html.Div([html.H2(title), html.P(caption)], className="gp-workspace-panel-head")


def _observed_rail() -> html.Aside:
    return html.Aside(
        [
            html.Section(
                [
                    html.Div("Investigation checklist", className="gp-control-eyebrow"),
                    html.H2("Is the difference interpretable?"),
                    html.Ol(
                        [
                            html.Li("Check that the series and measurement lead match."),
                            html.Li("Inspect the target window and score coverage."),
                            html.Li("Check data revisions, unavailable inputs and exclusions."),
                            html.Li("Use the model’s own backtest context."),
                        ]
                    ),
                    html.Div(id="models-investigation-context"),
                    html.Div(id="models-forecast-evidence-link"),
                ],
                className="gp-workspace-panel gp-workspace-rail",
            ),
            html.Details(
                [
                    html.Summary("Scoring window and coverage categories"),
                    html.Div(id="models-coverage-context"),
                ],
                className="gp-models-rail-disclosure",
            ),
            html.Details(
                [
                    html.Summary("Actual quality and revision context"),
                    html.P(
                        "Quality-flagged actuals remain excluded until a revision passes the "
                        "same checks. A later revision alone does not explain every model error. "
                        "Coverage categories retain each source’s reported counts; unknown "
                        "exclusions are not treated as zero."
                    ),
                ],
                className="gp-models-rail-disclosure",
            ),
        ],
        className="gp-workspace-stack gp-models-investigation-rail",
        **{"aria-label": "Observed error investigation context"},
    )


def _training_rail() -> html.Aside:
    return html.Aside(
        html.Section(
            [
                html.Div("Investigation scope", className="gp-control-eyebrow"),
                html.H2("Different evidence, different questions"),
                html.P(
                    "The training window contains multiple forecast leads. Compare observed "
                    "scores at a defined lead before deciding that a model’s recent "
                    "performance differs."
                ),
                html.P(
                    "Checkboxes filter only this table. The cached backtest series and "
                    "XGBoost importance remain independent; selecting a training model "
                    "does not create that model’s residuals."
                ),
                html.Button(
                    "Review observed error →",
                    id="models-review-observed",
                    n_clicks=0,
                    className="gp-review-button",
                ),
                html.Button(
                    "Inspect backtest diagnostics →",
                    id="models-review-backtest",
                    n_clicks=0,
                    className="gp-review-button",
                ),
            ],
            className="gp-workspace-panel gp-workspace-rail",
        ),
        className="gp-workspace-stack",
        **{"aria-label": "Training reference interpretation"},
    )


def layout() -> html.Div:
    """Match the reviewed workspace hierarchy while retaining native callback IDs."""
    return html.Div(
        [
            html.Div(
                "Investigate forecast performance",
                className="gp-control-eyebrow gp-models-view-eyebrow",
            ),
            html.Div(id="models-title"),
            html.Div(id="models-published-source", className="gp-models-published-callout"),
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
                            html.Section(
                                [
                                    _panel_head(
                                        "Observed forecast error",
                                        "Choose a measurement clock before comparing percentages",
                                    ),
                                    _choices(
                                        "models-observed-basis",
                                        [
                                            ("sampled", "Hourly-sampled lead"),
                                            ("first", "First-hour scores"),
                                        ],
                                        "sampled",
                                    ),
                                    html.Div(id="models-drift-panel"),
                                    html.Div(
                                        id="models-first-hour-panel", style={"display": "none"}
                                    ),
                                ],
                                className="gp-workspace-panel gp-models-tabbed-panel",
                            ),
                            html.Div(id="models-legacy-panel"),
                        ],
                        className="gp-workspace-stack",
                    ),
                    _observed_rail(),
                ],
                id="models-observed-section",
                className="gp-workspace-split",
            ),
            html.Section(
                [
                    html.Div(
                        html.Section(
                            [
                                _panel_head(
                                    "Training holdout",
                                    "168h recursive forecast · archived training reference",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "Training comparison only",
                                            className="gp-panel__caption",
                                        ),
                                        dbc.Checklist(
                                            id="tab3-model-selector",
                                            options=[
                                                {"label": model_display_name(k), "value": k}
                                                for k in ("prophet", "arima", "xgboost", "ensemble")
                                            ],
                                            value=["prophet", "arima", "xgboost", "ensemble"],
                                            inline=True,
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
                                html.Div(
                                    [
                                        html.P(
                                            "MAPE is percentage error; MAE and RMSE are MW. R² "
                                            "describes this evaluation. These results do not rank "
                                            "the accuracy of the next published forecast."
                                        ),
                                        html.Details(
                                            [
                                                html.Summary("Evaluation dates and source clock"),
                                                html.Div(id="models-leaderboard"),
                                            ]
                                        ),
                                    ],
                                    className="gp-models-training-notes",
                                ),
                            ],
                            className="gp-workspace-panel gp-models-training-panel",
                        ),
                        className="gp-workspace-stack",
                    ),
                    _training_rail(),
                ],
                id="models-training-section",
                className="gp-workspace-split",
                style={"display": "none"},
            ),
            html.Section(
                [
                    html.Div(id="tab3-diagnostics-scope"),
                    html.Div(
                        [
                            _graph("tab3-residuals-time", "Backtest error over time"),
                            _graph("tab3-residuals-hist", "Backtest error distribution"),
                            _graph("tab3-residuals-pred", "Backtest error vs predicted demand"),
                            _graph("tab3-error-heatmap", "Mean absolute error by target hour"),
                        ],
                        className="gp-models-diagnostic-grid",
                    ),
                    _graph("tab3-shap", "Built-in XGBoost feature importance"),
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
                className="gp-workspace-stack",
            ),
            build_page_footer(
                sources=["EIA", "Open-Meteo"],
                note="Model evidence is read from cached scoring and training outputs.",
            ),
        ],
        className="gp-page gp-models-review gp-section-stack gp-reviewed-page",
    )
