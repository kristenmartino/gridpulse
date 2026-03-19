"""
Tab 4: Generation & Net Load.

Net Load = Demand − Wind − Solar = load that must be met by dispatchable sources.

Components:
- Hero chart: Demand vs Net Load (two lines, shaded renewable contribution)
- Supporting chart: Generation mix stacked area by fuel type
- 4 KPIs: Renewable share, peak ramp, min net load, curtailment risk hours
- Insight card: Persona-aware observations
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Tab 4 (Generation & Net Load) layout."""
    return html.Div(
        [
            # Description
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P(
                                "Net Load = Demand \u2212 Wind \u2212 Solar. "
                                "This is the demand that must be met by dispatchable "
                                "generation (gas, nuclear, hydro). The shaded area "
                                "shows the renewable contribution.",
                                className="text-muted mb-3",
                                style={"fontSize": "0.9rem"},
                            ),
                        ]
                    ),
                ]
            ),
            # Date range selector
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                "Time Range:",
                                className="fw-bold",
                                style={"fontSize": "0.85rem"},
                            ),
                            dbc.RadioItems(
                                id="gen-date-range",
                                options=[
                                    {"label": "Last 24 hours", "value": "24"},
                                    {"label": "Last 7 days", "value": "168"},
                                    {"label": "Last 30 days", "value": "720"},
                                    {"label": "Last 90 days", "value": "2160"},
                                ],
                                value="168",
                                inline=True,
                                className="mt-1",
                                style={"fontSize": "0.85rem"},
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="mb-3",
            ),
            # Hero chart: Demand vs Net Load
            build_chart_container(
                "tab4-net-load-chart",
                "Demand vs Net Load",
                height="400px",
            ),
            # AI insight card
            html.Div(id="tab4-insight-card"),
            # KPI row (4 cards)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("RENEWABLE SHARE", className="kpi-label"),
                                    html.H4(
                                        id="tab4-renewable-pct",
                                        children="\u2014%",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Wind + Solar + Hydro",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("PEAK RAMP", className="kpi-label"),
                                    html.H4(
                                        id="tab4-peak-ramp",
                                        children="\u2014 MW/hr",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Max hourly net load increase",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("MIN NET LOAD", className="kpi-label"),
                                    html.H4(
                                        id="tab4-min-net-load",
                                        children="\u2014 MW",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Duck curve belly",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("LOW NET LOAD HRS", className="kpi-label"),
                                    html.H4(
                                        id="tab4-curtailment-hours",
                                        children="\u2014",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        "Hours < 20% of peak",
                                        className="kpi-delta neutral",
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mt-3 g-2",
            ),
            # Supporting chart: Generation Mix (stacked area)
            html.Div(className="mt-3"),
            build_chart_container(
                "tab4-gen-mix-chart",
                "Generation Mix by Fuel Type",
                height="320px",
            ),
        ],
        className="p-2",
    )
