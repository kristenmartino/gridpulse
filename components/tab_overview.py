"""
Tab 0: Overview — AI-powered executive briefing and dashboard digest.

Sections:
A. AI Executive Briefing (Claude-generated, persona-specific)
B. Data Health (per-source freshness badges)
C. Spotlight Chart (persona-relevant metric) + Insight Digest (cross-tab)
D. Energy News Feed (moved from global footer)
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    """Build Tab 0 (Overview) layout."""
    return html.Div(
        [
            # ── Section A: Greeting + AI Executive Briefing ───────
            dbc.Row(
                dbc.Col(
                    html.Div(id="overview-greeting"),
                    md=12,
                ),
                className="g-2 mb-2",
            ),
            html.Div(
                id="overview-briefing",
                children=_skeleton_briefing(),
                className="briefing-card",
            ),
            # ── Section B: Data Health ────────────────────────────
            html.Div(id="overview-data-health", className="mt-2"),
            # ── Section C: Spotlight Chart + Insight Digest ───────
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Loading(
                                dcc.Graph(
                                    id="overview-spotlight-chart",
                                    style={"height": "260px"},
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                ),
                                type="circle",
                                color="#00d4aa",
                            ),
                            className="chart-container",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            id="overview-insight-digest",
                            className="insight-digest",
                        ),
                        md=4,
                    ),
                ],
                className="g-2 mt-2",
            ),
            # ── Section D: Energy News ────────────────────────────
            html.Div(id="overview-news-feed", className="mt-2"),
        ]
    )


def _skeleton_briefing() -> html.Div:
    """Pulsing skeleton placeholder while briefing loads."""
    return html.Div(
        [
            html.Div(className="skeleton-line skeleton-line-long"),
            html.Div(className="skeleton-line skeleton-line-long"),
            html.Div(className="skeleton-line skeleton-line-medium"),
        ],
        className="skeleton-pulse",
    )
