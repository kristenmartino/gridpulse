"""
Tab 2: Weather-Energy Correlation Explorer (James's primary view).

Components:
- Scatter plot matrix: Temp vs Demand, Wind vs Wind Gen, GHI vs Solar Gen
- Correlation heatmap (all weather features vs demand)
- Feature importance bar chart
- Seasonal decomposition
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_chart_container


def _section_header(title: str, subtitle: str) -> html.Div:
    """Render a lightweight section header."""
    return html.Div(
        [
            html.Span(
                title,
                style={
                    "color": "#F7FAFC",
                    "fontSize": "0.85rem",
                    "fontWeight": "600",
                    "marginRight": "8px",
                },
            ),
            html.Span(
                subtitle,
                style={
                    "color": "#A8B3C7",
                    "fontSize": "0.75rem",
                },
            ),
        ],
        style={
            "padding": "10px 0 4px 0",
            "borderBottom": "1px solid #263556",
            "marginTop": "12px",
            "marginBottom": "8px",
        },
    )


def layout() -> html.Div:
    """Build Tab 2 layout."""
    return html.Div(
        [
            # ── Weather Correlations ────────────────────────
            _section_header(
                "Weather Correlations", "Direct relationships between weather and energy"
            ),
            # Top row: scatter plots
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab2-scatter-temp", "Temperature vs Demand", height="300px"
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab2-scatter-wind", "Wind Speed vs Wind Power", height="300px"
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab2-scatter-solar", "Solar Irradiance vs Solar CF", height="300px"
                        ),
                        md=4,
                    ),
                ],
                className="g-2",
            ),
            # ── Feature Analysis ─────────────────────────
            _section_header(
                "Feature Analysis", "Multi-variable correlation and importance ranking"
            ),
            # Middle row: heatmap + feature importance
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab2-heatmap", "Feature Correlation Heatmap", height="380px"
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        build_chart_container(
                            "tab2-feature-importance", "Weather Feature Importance", height="380px"
                        ),
                        md=5,
                    ),
                ],
                className="g-2 mt-1",
            ),
            # ── Decomposition ───────────────────────────
            _section_header("Decomposition", "Trend, seasonal, and residual components"),
            # Bottom row: seasonal decomposition
            dbc.Row(
                [
                    dbc.Col(
                        build_chart_container(
                            "tab2-seasonal", "Demand: Trend + Seasonal + Residual", height="350px"
                        ),
                        md=12,
                    ),
                ],
                className="g-2 mt-1",
            ),
        ]
    )
