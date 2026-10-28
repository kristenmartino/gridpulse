"""Conditions: linked weather/demand review and a separate source-message inbox."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.cards import build_page_footer
from components.weather_explorer import layout as explorer_layout


def layout():
    return html.Div(
        [
            html.Div(id="risk-title"),
            dbc.RadioItems(
                id="conditions-section",
                options=[
                    {"label": "Weather & demand", "value": "explorer"},
                    {"label": "Weather messages", "value": "messages"},
                ],
                value="explorer",
                inline=True,
                className="gp-segmented gp-conditions-sections",
            ),
            explorer_layout(),
            html.Div(
                [
                    dcc.Store(id="conditions-message-data"),
                    dcc.Store(id="conditions-selected-message"),
                    dcc.Store(id="conditions-focus-event"),
                    dcc.Store(id="conditions-back-event"),
                    html.P(
                        "NWS messages for associated states are regional context, not measured demand impacts or verified territory intersections.",
                        className="gp-panel__hint",
                    ),
                    html.Div(
                        id="conditions-message-status", role="status", className="gp-review-source"
                    ),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.H2("Weather messages"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Search messages",
                                                        htmlFor="conditions-message-search",
                                                    ),
                                                    dbc.Input(
                                                        id="conditions-message-search",
                                                        type="search",
                                                        value="",
                                                        placeholder="Event, source or headline…",
                                                        debounce=0.3,
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Category",
                                                        htmlFor="conditions-message-severity",
                                                    ),
                                                    dbc.Select(
                                                        id="conditions-message-severity",
                                                        value="all",
                                                        options=[
                                                            {
                                                                "label": "All source groups",
                                                                "value": "all",
                                                            }
                                                        ]
                                                        + [
                                                            {"label": x.title(), "value": x}
                                                            for x in ("critical", "warning", "info")
                                                        ],
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="gp-review-toolbar",
                                    ),
                                    html.Div(id="conditions-message-list"),
                                ],
                                className="gp-panel gp-review-inventory",
                            ),
                            html.Aside(
                                [
                                    html.Button(
                                        "← Back to messages",
                                        id="conditions-message-back",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                    html.Div(id="conditions-message-detail"),
                                ],
                                id="conditions-message-inspector",
                                className="gp-panel gp-review-inspector",
                            ),
                        ],
                        className="gp-review-split",
                    ),
                    html.Details(
                        [
                            html.Summary("Recent weather and descriptive demand references"),
                            html.P(
                                "Weather is regional model context. Demand bands are descriptive rolling statistics, not forecast intervals or emergency thresholds.",
                                className="gp-panel__hint",
                            ),
                            html.Div(id="tab5-weather-context"),
                            dcc.Graph(
                                id="tab5-anomaly-chart",
                                config={"displayModeBar": False, "responsive": True},
                            ),
                            dcc.Graph(
                                id="tab5-temp-exceedance",
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                        className="gp-review-disclosure",
                    ),
                ],
                id="conditions-messages",
                style={"display": "none"},
            ),
            # Compatibility carriers for the existing source callback; no stress or event-gallery UI.
            html.Div(
                [
                    html.Span(id="tab5-stress-score"),
                    html.Span(id="tab5-stress-label"),
                    html.Div(id="tab5-stress-breakdown"),
                    html.Div(id="tab5-alerts-list"),
                    dcc.Graph(id="tab5-timeline"),
                    html.Div(id="risk-insight-card"),
                ],
                style={"display": "none"},
            ),
            build_page_footer(
                note="Weather estimates and NWS messages provide context; neither establishes causal demand or asset impact."
            ),
        ],
        className="gp-page gp-reviewed-page gp-conditions-page",
    )
