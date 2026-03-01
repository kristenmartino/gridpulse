"""
Tab 4: Generation Mix & Renewables (James + Maria's view).

Components:
- Stacked area chart: generation by fuel type
- Renewable penetration % line
- Wind generation vs wind speed overlay
- Solar generation vs irradiance
- Duck curve visualization (net demand = total - solar)
- Carbon intensity estimate
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def layout() -> html.Div:
    """Build Tab 4 layout."""
    return html.Div(
        [
            # Top: stacked area
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab4-gen-mix", "Generation by Fuel Type", height="380px"
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.P("RENEWABLE SHARE", className="kpi-label"),
                                    html.H3(
                                        id="tab4-renewable-pct",
                                        children="— %",
                                        className="kpi-value",
                                    ),
                                    html.P(
                                        id="tab4-renewable-delta", className="kpi-delta neutral"
                                    ),
                                ],
                                className="kpi-card mb-2",
                            ),
                            html.Div(
                                [
                                    html.P("WIND CAPACITY FACTOR", className="kpi-label"),
                                    html.H3(
                                        id="tab4-wind-cf", children="— %", className="kpi-value"
                                    ),
                                ],
                                className="kpi-card mb-2",
                            ),
                            html.Div(
                                [
                                    html.P("SOLAR CAPACITY FACTOR", className="kpi-label"),
                                    html.H3(
                                        id="tab4-solar-cf", children="— %", className="kpi-value"
                                    ),
                                ],
                                className="kpi-card mb-2",
                            ),
                            html.Div(
                                [
                                    html.P("EST. CARBON INTENSITY", className="kpi-label"),
                                    html.H3(
                                        id="tab4-carbon", children="— kg/MWh", className="kpi-value"
                                    ),
                                ],
                                className="kpi-card",
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-2",
            ),
            # Middle row: wind + solar overlays
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab4-wind-overlay", "Wind Generation vs Wind Speed", height="300px"
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab4-solar-overlay", "Solar Generation vs Irradiance", height="300px"
                        ),
                        md=6,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # Bottom: duck curve + renewable penetration
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab4-duck-curve",
                            "Duck Curve (Net Demand = Total − Solar)",
                            height="320px",
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab4-renewable-trend",
                            "Renewable Penetration Over Time",
                            height="320px",
                        ),
                        md=5,
                    ),
                ],
                className="g-2 mt-1",
            ),
        ]
    )
