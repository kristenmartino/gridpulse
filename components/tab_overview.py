"""Overview: proposed analyst review workflow and displayed forecast context.

Dynamic content retains its callback IDs. Reported-demand history sits below
the forecast review so historical KPIs do not compete with the next decision.
"""

from dash import dcc, html

from components.cards import build_page_footer


def layout() -> html.Div:
    """Build the forecast-review Overview using the existing page shell."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(id="overview-title"),
                    html.Div(
                        [
                            html.P(
                                "For the Forecast Review Analyst",
                                className="gp-review-workflow__audience",
                            ),
                            html.P(
                                "Does this forecast need closer review?",
                                className="gp-review-workflow__question",
                            ),
                            html.P(
                                "Check the forecast source, its recent observed error, "
                                "and the demand outlook before deciding what to investigate.",
                                className="gp-review-workflow__note",
                            ),
                        ],
                        className="gp-review-workflow",
                    ),
                    html.Div(id="overview-insight-card"),
                    html.Div(id="overview-model-card"),
                    html.Div(
                        dcc.Loading(
                            dcc.Graph(
                                id="overview-spotlight-chart",
                                style={"height": "380px"},
                                config={
                                    "displayModeBar": False,
                                    "responsive": True,
                                },
                            ),
                            type="circle",
                            color="#3b82f6",
                        ),
                        className="gp-chart-card",
                    ),
                    html.Details(
                        [
                            html.Summary("Reported demand history"),
                            html.Div(id="overview-metrics-bar"),
                        ],
                        className="gp-review-workflow__history",
                    ),
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
