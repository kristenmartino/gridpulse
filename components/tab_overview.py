"""Tab 0: Overview — mission-control linear stack.

R2 of shell-redesign-v2.md. Replaces the prior 8-card sprawl (greeting /
briefing / what-changed / data-health / quick-nav / spotlight / insight-digest /
news-feed) with the gridpulse-v2 dashboard rhythm: a single-column space-y-8
stack of seven sections.

Structure (top → bottom):
1. Page title block (region name + 1-line subtitle)
2. MetricsBar — 5-up KPI row (Now / 7d Peak / 7d Low / Average / 24h Trend)
3. Hero demand chart — full-width forecast with confidence band
4. ModelMetricsCard — horizontal model-performance bar (MAPE / RMSE / MAE / R²)
5. InsightCard — single narrative paragraph (eyebrow + body)

Dynamic content (1–5) is filled by ``update_overview_tab`` in
``components/callbacks.py``. Attribution now lives in the single app-level
footer built by ``components/layout.py`` (no per-tab footer).
"""

from dash import dcc, html


def layout() -> html.Div:
    """Build the v2-style Overview linear stack."""
    return html.Div(
        [
            html.Div(
                [
                    # 1. Title block (callback fills overview-title)
                    html.Div(id="overview-title"),
                    # 2. MetricsBar (5-up KPI row)
                    html.Div(id="overview-metrics-bar"),
                    # 3. Hero demand chart (full-width with confidence band)
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
                            color="#35c6ff",
                        ),
                        className="gp-chart-card",
                    ),
                    # 4. ModelMetricsCard (horizontal model performance bar)
                    html.Div(id="overview-model-card"),
                    # 5. InsightCard (narrative summary)
                    html.Div(id="overview-insight-card"),
                    # Footer promoted to a single app-level footer in
                    # components/layout.py build_layout().
                ],
                className="gp-section-stack",
            ),
        ],
        className="gp-page",
    )
