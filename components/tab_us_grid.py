"""Authority directory, inspected snapshots and one genuine territory map."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.cards import build_page_footer
from config import REGION_GROUPS


def layout():
    page = html.Div(
        [
            dcc.Store(id="us-grid-selected", data="FPL", storage_type="session"),
            dcc.Store(id="us-grid-data"),
            dcc.Store(id="us-grid-jump-store"),
            html.Div(id="us-grid-title"),
            html.Div(id="us-grid-metrics-bar"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("View", htmlFor="us-grid-view-toggle"),
                            dbc.RadioItems(
                                id="us-grid-view-toggle",
                                options=[
                                    {"label": x.title(), "value": x}
                                    for x in ("table", "cards", "map")
                                ],
                                value="table",
                                inline=True,
                                className="gp-segmented",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Search authorities", htmlFor="us-grid-search"),
                            dbc.Input(
                                id="us-grid-search",
                                placeholder="Name or code",
                                type="search",
                                debounce=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Group", htmlFor="us-grid-group"),
                            dbc.Select(
                                id="us-grid-group",
                                options=[{"label": "All groups", "value": "all"}]
                                + [{"label": x, "value": x} for x in REGION_GROUPS],
                                value="all",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Sort", htmlFor="us-grid-sort"),
                            dbc.Select(
                                id="us-grid-sort",
                                options=[
                                    {"label": "Hourly movement", "value": "change"},
                                    {"label": "Demand · high to low", "value": "demand"},
                                    {"label": "Name · A–Z", "value": "name"},
                                ],
                                value="change",
                            ),
                        ]
                    ),
                ],
                className="gp-grid-controls",
            ),
            html.Div(
                [
                    html.Section(id="us-grid-region-grid", className="gp-grid-directory"),
                    html.Aside(
                        id="us-grid-inspector",
                        className="gp-grid-inspector",
                        **{"aria-label": "Selected authority"},
                    ),
                ],
                className="gp-grid-workspace",
            ),
            build_page_footer(),
        ],
        className="gp-page gp-section-stack gp-grid-review",
    )
    toolbar = page.children[5]
    workspace = page.children[6]
    directory = workspace.children[0]
    directory.children = [
        html.Div(
            [
                html.H2("Regional inventory"),
                html.P("Inspect one authority, then open its forecast"),
            ],
            className="gp-workspace-panel-head",
        ),
        toolbar,
        html.Section(id="us-grid-region-grid", className="gp-grid-directory-results"),
    ]
    directory.id = "us-grid-directory-panel"
    directory.className = "gp-workspace-panel gp-grid-directory"
    page.children.pop(5)
    return page
