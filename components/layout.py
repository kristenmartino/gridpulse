"""Main dashboard layout — header, tab strip, data stores.

R3 of shell-redesign-v2.md. Header rebuilt to mirror gridpulse-v2's
56px h-14 strip with monogram + Grid|Pulse wordmark on the left and
region/persona/mode controls on the right. Tab strip reduced to 4
visible tabs (Overview, Forecast, Risk, Models); five other tabs are
still rendered into the DOM via ``tab_class_name="d-none"`` so their
component IDs exist for callback safety. R4 will fold their content
into the four visible tabs and remove the hidden tabs entirely.

CRITICAL: All 9 tab layouts are still rendered statically inside
``dbc.Tab(children=...)``. Hiding the tab strip button via Bootstrap's
``d-none`` class keeps the panel content in the DOM (callbacks resolve)
without exposing the navigation pill.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components import (
    tab_alerts,
    tab_backtest,
    tab_demand_outlook,
    tab_forecast,
    tab_generation,
    tab_models,
    tab_overview,
    tab_simulator,
    tab_weather,
)
from config import REGION_NAMES, TAB_LABELS
from personas.config import list_personas

# R3 visible-tab whitelist. Other tabs render but their pill is hidden.
_VISIBLE_TABS = {"tab-overview", "tab-outlook", "tab-alerts", "tab-models"}

# R3 surface the four visible tabs under v2-aligned labels regardless of
# whatever name a hidden tab happens to use today.
_VISIBLE_LABEL_OVERRIDES = {
    "tab-overview": "Overview",
    "tab-outlook": "Forecast",
    "tab-alerts": "Risk",
    "tab-models": "Models",
}


def _monogram() -> html.Span:
    """Inline 24×24 SVG monogram. Same path data as ``assets/favicon.svg``."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" '
        'class="gp-header__monogram" aria-hidden="true">'
        '<rect width="32" height="32" rx="6" fill="#0a0a0b"/>'
        '<path d="M4 16 L11 16 L14 8 L16 24 L18 12 L21 16 L28 16" '
        'stroke="#3b82f6" stroke-width="2.25" '
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        "</svg>"
    )
    return html.Span(
        dcc.Markdown(svg, dangerously_allow_html=True),
        className="gp-header__monogram-wrap",
    )


def _build_header() -> html.Header:
    """v2-aligned header: monogram + wordmark + region/persona/mode controls."""
    persona_options = [
        {"label": f"{p['avatar']} {p['title']}", "value": p["id"]} for p in list_personas()
    ]
    region_options = [{"label": name, "value": code} for code, name in REGION_NAMES.items()]

    brand = html.Div(
        [
            _monogram(),
            html.H1(
                [
                    html.Span("Grid", className="gp-header__wordmark-grid"),
                    html.Span("Pulse", className="gp-header__wordmark-pulse"),
                ],
                className="gp-header__wordmark dashboard-title",
            ),
        ],
        className="gp-header__brand",
    )

    controls = html.Div(
        [
            dbc.Select(
                id="region-selector",
                options=region_options,
                value="FPL",
                className="gp-header__select region-selector",
            ),
            dbc.Select(
                id="persona-selector",
                options=persona_options,
                value="grid_ops",
                className="gp-header__chip persona-switcher",
            ),
            html.Button(
                "Briefing Mode",
                id="meeting-mode-btn",
                n_clicks=0,
                className="gp-header__link",
            ),
            html.Button(
                "Save View",
                id="bookmark-btn",
                n_clicks=0,
                className="gp-header__link",
            ),
            # Hidden — kept for the freshness callback that writes here.
            html.Div(id="header-freshness", style={"display": "none"}),
        ],
        className="gp-header__controls",
    )

    return html.Header(
        html.Div([brand, controls], className="gp-header__inner"),
        className="dashboard-header gp-header",
        id="dashboard-header",
    )


def _tab(tab_id: str, layout_fn) -> dbc.Tab:
    """Wrap a tab module's layout in ``dbc.Tab``. Hidden tabs use ``d-none``
    on the pill so the panel renders without surfacing in the strip."""
    label = _VISIBLE_LABEL_OVERRIDES.get(tab_id, TAB_LABELS.get(tab_id, tab_id))
    is_visible = tab_id in _VISIBLE_TABS
    return dbc.Tab(
        layout_fn(),
        label=label,
        tab_id=tab_id,
        tab_class_name="" if is_visible else "d-none",
    )


def build_layout() -> dbc.Container:
    """Build the full dashboard layout (R3 — 4 visible tabs)."""
    return dbc.Container(
        [
            # URL state for bookmarks (C2)
            dcc.Location(id="url", refresh=False),
            # Auto-refresh interval
            dcc.Interval(id="refresh-interval", interval=300_000, n_intervals=0),
            # Header
            _build_header(),
            # Bookmark / save-view toast
            html.Div(
                id="bookmark-toast",
                style={
                    "position": "fixed",
                    "top": "10px",
                    "right": "10px",
                    "zIndex": 9999,
                },
            ),
            # Data freshness banner (G2 — visible only when degraded)
            html.Div(id="fallback-banner"),
            # Per-Widget confidence badges (A4 + E3) — hidden carrier element
            html.Div(id="widget-confidence-bar", style={"display": "none"}),
            # Tab strip (4 visible: Overview / Forecast / Risk / Models)
            # R5b: wrapped in <main role="main"> so the skip-to-content
            # link lands here and screen readers announce the page's
            # main content region. The dbc.Tabs strip itself is the
            # primary navigation control.
            html.Main(
                dbc.Tabs(
                    id="dashboard-tabs",
                    active_tab="tab-overview",
                    children=[
                        _tab("tab-overview", tab_overview.layout),
                        _tab("tab-outlook", tab_demand_outlook.layout),
                        _tab("tab-alerts", tab_alerts.layout),
                        _tab("tab-models", tab_models.layout),
                        # ── Hidden tabs (DOM resident; absorbed in R4) ──
                        _tab("tab-forecast", tab_forecast.layout),
                        _tab("tab-backtest", tab_backtest.layout),
                        _tab("tab-generation", tab_generation.layout),
                        _tab("tab-weather", tab_weather.layout),
                        _tab("tab-simulator", tab_simulator.layout),
                    ],
                ),
                id="main-content",
                role="main",
            ),
            # Data Stores
            dcc.Store(id="news-store"),
            dcc.Store(id="demand-store"),
            dcc.Store(id="weather-store"),
            dcc.Store(id="features-store"),
            dcc.Store(id="models-store"),
            dcc.Store(id="alerts-store"),
            dcc.Store(id="data-freshness-store"),
            dcc.Store(id="meeting-mode-store", data="false"),
            dcc.Store(id="audit-store"),
            dcc.Store(id="pipeline-log-store"),
            dcc.Store(id="briefing-store"),
            # NEXD-8: Session snapshot for "What Changed" (R4a-reserved)
            dcc.Store(id="session-snapshot-store", storage_type="local"),
            dcc.Store(id="changes-store"),
            # NEXD-9: User preferences
            dcc.Store(id="user-prefs-store", storage_type="local"),
        ],
        fluid=True,
        style={"padding": 0},
    )
