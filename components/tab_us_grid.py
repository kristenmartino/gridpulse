"""Tab: US Grid — small-multiples view of all 16 balancing authorities.

V1.β of NEXT_UP.md. A bird's-eye snapshot of every BA in the system as
a card grid, with a 4-up MetricsBar for national rollups. Each card is
clickable — drilling down lands the user on the Forecast tab pre-set
to that region.

Structure (top → bottom):
1. Page title block (region count + national now-demand)
2. MetricsBar — 4-up (Total Demand · National Peak Today · Highest-Stress Region · Lowest Reserve)
3. Region card grid (16 cards on desktop, 4-col)
   each card: name + demand (hero) + delta chip + 24h sparkline + stress chip
4. Footer — static attribution

Dynamic content (1–3) is filled by ``update_us_grid_snapshot`` in
``components/callbacks.py``; the footer is rendered statically.
"""

from dash import html

from components.cards import build_page_footer


def layout() -> html.Div:
    """Build the US Grid tab's structural skeleton."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(id="us-grid-title"),
                    html.Div(id="us-grid-metrics-bar"),
                    html.Div(id="us-grid-region-grid"),
                    build_page_footer(),
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
