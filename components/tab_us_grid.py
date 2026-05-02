"""Tab: US Grid — small-multiples view of all 16 balancing authorities.

V1.β + V1.γ of NEXT_UP.md. A bird's-eye snapshot of every BA in the
system, available as either a card grid or a Plotly ``scatter_geo``
map of BA centroids. Each card / map point is clickable — drilling
down lands the user on the Forecast tab pre-set to that region.

Structure (top → bottom):
1. Page title block (region count + national now-demand)
2. MetricsBar — 4-up (Total Demand · National Peak 24h · Highest-Stress Region · Lowest Reserve)
3. View toggle — Cards | Map (segmented control, default Cards)
4. Body — region card grid OR scatter_geo map (driven by toggle)
5. Footer — static attribution

Dynamic content (1, 2, 4) is filled by ``update_us_grid_snapshot`` in
``components/callbacks.py``; the toggle and footer are static.
"""

import dash_bootstrap_components as dbc
from dash import html

from components.cards import build_page_footer


def _view_toggle() -> html.Div:
    """v2-style segmented control for the Cards | Map | Polygons switch.

    V3.β added the Polygons option — a Plotly Choropleth driven by
    ``assets/ba_polygons.geojson`` (51 BA service-territory polygons
    sourced from the MIT-licensed electricitymaps-contrib repo, ~165 KB
    pre-simplified). Map (centroid scatter) is preserved as a fallback.
    """
    return html.Div(
        [
            html.Div("View", className="gp-control-eyebrow"),
            dbc.RadioItems(
                id="us-grid-view-toggle",
                options=[
                    {"label": "Cards", "value": "cards"},
                    {"label": "Map", "value": "map"},
                    {"label": "Polygons", "value": "polygons"},
                ],
                value="cards",
                inline=True,
                className="gp-segmented",
            ),
        ],
        className="gp-control gp-us-grid-view-control",
    )


def layout() -> html.Div:
    """Build the US Grid tab's structural skeleton."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(id="us-grid-title"),
                    html.Div(id="us-grid-metrics-bar"),
                    _view_toggle(),
                    html.Div(id="us-grid-region-grid"),
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
