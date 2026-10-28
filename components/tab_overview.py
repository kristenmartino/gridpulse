"""Overview: published outlook, matching evidence and role-specific context."""

from dash import dcc, html

from components.cards import build_page_footer


def layout() -> html.Div:
    """Keep the real application shell and put the forecast near its evidence."""
    page = html.Div(
        [
            html.Div(id="overview-title"),
            html.Div(
                [
                    html.Strong("Does this forecast need closer review?"),
                    " Check its source, recent observed error and demand outlook.",
                ],
                className="gp-overview__question",
            ),
            html.Div(id="overview-review-summary"),
            html.Div(id="overview-insight-card"),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2("Demand outlook"),
                                            html.P(
                                                "Reported demand and the next 24h published forecast"
                                            ),
                                        ]
                                    ),
                                    dcc.RadioItems(
                                        id="overview-history-range",
                                        options=[
                                            {"label": "24h history", "value": "24"},
                                            {"label": "7d history", "value": "168"},
                                        ],
                                        value="24",
                                        inline=True,
                                        className="gp-overview__ranges",
                                        inputClassName="gp-overview__range-input",
                                        labelClassName="gp-overview__range-label",
                                    ),
                                ],
                                className="gp-overview__chart-header",
                            ),
                            dcc.Loading(
                                dcc.Graph(
                                    id="overview-spotlight-chart",
                                    className="gp-overview__chart",
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                                type="circle",
                                color="#3b82f6",
                            ),
                            html.P(
                                "History controls change reported demand only. The published forecast remains 24 hours; all chart times are UTC.",
                                className="gp-overview__chart-note",
                            ),
                        ],
                        className="gp-overview__chart-card",
                        **{"aria-label": "Demand outlook"},
                    ),
                    html.Aside(
                        id="overview-model-card",
                        **{"aria-label": "Scoring coverage and training context"},
                    ),
                ],
                className="gp-overview__main-grid",
            ),
            html.Details(
                [html.Summary("Inspect evidence"), html.Div(id="overview-evidence-panel")],
                className="gp-overview__evidence",
            ),
            html.Div(id="overview-source-details"),
            html.Details(
                [html.Summary("Reported demand history"), html.Div(id="overview-metrics-bar")],
                className="gp-review-workflow__history gp-review-disclosure",
            ),
            build_page_footer(),
        ],
        className="gp-page gp-overview",
    )
    chart = page.children[4].children[0]
    header = chart.children[0]
    history = header.children[1]
    header.children = [header.children[0]]
    chart.children.insert(
        1,
        html.Div(
            [html.Span("Reported demand history"), history],
            className="gp-overview__history-toolbar",
        ),
    )
    page.children[5].id = "overview-evidence-disclosure"
    page.children[5].children[0].tabIndex = -1
    jump = html.Button(
        "Inspect evidence ↓",
        id="overview-inspect-evidence",
        n_clicks=0,
        className="gp-review-button gp-overview__evidence-jump",
    )
    page.children[4].children[0] = html.Div([chart, jump], className="gp-workspace-stack")
    page.children.insert(
        -1,
        html.Div(
            [
                html.Span("Continue the review"),
                html.Button(
                    "Review forecast →",
                    id="overview-open-forecast",
                    n_clicks=0,
                    className="gp-review-button",
                ),
                html.Button(
                    "Investigate error →",
                    id="overview-open-models",
                    n_clicks=0,
                    className="gp-review-button",
                ),
                html.Button(
                    "Check conditions →",
                    id="overview-open-conditions",
                    n_clicks=0,
                    className="gp-review-button",
                ),
            ],
            className="gp-overview__next",
        ),
    )
    return page
