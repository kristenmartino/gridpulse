"""
Main dashboard layout — header, persona switcher, region selector, KPI bar, tabs.

CRITICAL: All 9 tab layouts are rendered statically inside dbc.Tab(children=...).
This ensures every component ID always exists in the DOM. Dash Bootstrap handles
showing/hiding tabs — we do NOT dynamically render tab content via callbacks.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# Import all tab layouts
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


def build_layout() -> dbc.Container:
    """Build the full dashboard layout with all 8 tabs pre-rendered."""
    return dbc.Container(
        [
            # ── URL state for bookmarks (C2) ──────────────────────
            dcc.Location(id="url", refresh=False),
            # ── Interval for auto-refresh ──────────────────────────
            dcc.Interval(id="refresh-interval", interval=300_000, n_intervals=0),
            # ── Header ─────────────────────────────────────────────
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H1("GridPulse", className="dashboard-title"),
                                    html.P(
                                        "Energy Intelligence Platform",
                                        className="dashboard-subtitle",
                                    ),
                                ],
                                md=4,
                                className="d-flex flex-column justify-content-center",
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Balancing Authority",
                                        style={"color": "#A8B3C7", "fontSize": "0.7rem"},
                                    ),
                                    dbc.Select(
                                        id="region-selector",
                                        options=[
                                            {"label": name, "value": code}
                                            for code, name in REGION_NAMES.items()
                                        ],
                                        value="FPL",
                                        className="region-selector",
                                    ),
                                ],
                                md=3,
                                className="d-flex flex-column justify-content-center",
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "View", style={"color": "#A8B3C7", "fontSize": "0.7rem"}
                                    ),
                                    dbc.Select(
                                        id="persona-selector",
                                        options=[
                                            {
                                                "label": f"{p['avatar']} {p['title']}",
                                                "value": p["id"],
                                            }
                                            for p in list_personas()
                                        ],
                                        value="grid_ops",
                                        className="persona-switcher",
                                    ),
                                ],
                                md=2,
                                className="d-flex flex-column justify-content-center",
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Briefing Mode",
                                                id="meeting-mode-btn",
                                                size="sm",
                                                color="info",
                                                outline=True,
                                                className="me-2",
                                                style={"fontSize": "0.75rem"},
                                            ),
                                            dbc.Button(
                                                "Save View",
                                                id="bookmark-btn",
                                                size="sm",
                                                color="secondary",
                                                outline=True,
                                                className="me-2",
                                                style={"fontSize": "0.75rem"},
                                            ),
                                            html.Div(
                                                id="header-freshness", style={"display": "none"}
                                            ),
                                        ],
                                        className="d-flex flex-column align-items-end justify-content-center h-100",
                                    ),
                                ],
                                md=3,
                            ),
                        ],
                        align="center",
                    ),
                ],
                className="dashboard-header",
                id="dashboard-header",
            ),
            # ── Bookmark notification ──────────────────────────────
            html.Div(
                id="bookmark-toast",
                style={"position": "fixed", "top": "10px", "right": "10px", "zIndex": 9999},
            ),
            # ── Data Freshness Banner (G2) ─────────────────────────
            html.Div(id="fallback-banner"),
            # ── Per-Widget Confidence Badges (A4 + E3) — hidden, moved to overview data health
            html.Div(id="widget-confidence-bar", style={"display": "none"}),
            # ── Welcome Card ───────────────────────────────────────
            html.Div(id="welcome-card"),
            # ── KPI Row ────────────────────────────────────────────
            html.Div(id="kpi-cards"),
            # ── Tabs (full width) ─────────────────────────────────
            dbc.Tabs(
                id="dashboard-tabs",
                active_tab="tab-overview",
                children=[
                    dbc.Tab(
                        tab_overview.layout(),
                        label=TAB_LABELS["tab-overview"],
                        tab_id="tab-overview",
                    ),
                    dbc.Tab(
                        tab_forecast.layout(),
                        label=TAB_LABELS["tab-forecast"],
                        tab_id="tab-forecast",
                    ),
                    dbc.Tab(
                        tab_demand_outlook.layout(),
                        label=TAB_LABELS["tab-outlook"],
                        tab_id="tab-outlook",
                    ),
                    dbc.Tab(
                        tab_models.layout(),
                        label=TAB_LABELS["tab-models"],
                        tab_id="tab-models",
                    ),
                    dbc.Tab(
                        tab_backtest.layout(),
                        label=TAB_LABELS["tab-backtest"],
                        tab_id="tab-backtest",
                    ),
                    dbc.Tab(
                        tab_generation.layout(),
                        label=TAB_LABELS["tab-generation"],
                        tab_id="tab-generation",
                    ),
                    dbc.Tab(
                        tab_weather.layout(),
                        label=TAB_LABELS["tab-weather"],
                        tab_id="tab-weather",
                    ),
                    dbc.Tab(
                        tab_alerts.layout(),
                        label=TAB_LABELS["tab-alerts"],
                        tab_id="tab-alerts",
                    ),
                    dbc.Tab(
                        tab_simulator.layout(),
                        label=TAB_LABELS["tab-simulator"],
                        tab_id="tab-simulator",
                    ),
                ],
            ),
            # ── Data Stores ────────────────────────────────────────
            dcc.Store(id="news-store"),
            dcc.Store(id="demand-store"),
            dcc.Store(id="weather-store"),
            dcc.Store(id="features-store"),
            dcc.Store(id="models-store"),
            dcc.Store(id="alerts-store"),
            # G2: Track whether each data source served fresh or stale data
            dcc.Store(id="data-freshness-store"),
            # C9: Meeting-ready mode state (True/False)
            dcc.Store(id="meeting-mode-store", data="false"),
            # D2: Latest audit record for display
            dcc.Store(id="audit-store"),
            # I1: Latest pipeline log for diagnostics
            dcc.Store(id="pipeline-log-store"),
            # AI briefing cache for Overview tab
            dcc.Store(id="briefing-store"),
        ],
        fluid=True,
        style={"padding": 0},
    )
