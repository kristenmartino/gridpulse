"""
Main dashboard layout — header, persona switcher, region selector, KPI bar, tabs.

CRITICAL: All 4 tab layouts are rendered statically inside dbc.Tab(children=...).
This ensures every component ID always exists in the DOM. Dash Bootstrap handles
showing/hiding tabs — we do NOT dynamically render tab content via callbacks.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

# Import all tab layouts
from components import (
    tab_backtest,
    tab_demand_outlook,
    tab_forecast,
    tab_generation,
)
from config import REGION_NAMES, TAB_LABELS
from personas.config import list_personas


def build_layout() -> dbc.Container:
    """Build the full dashboard layout with all tabs pre-rendered."""
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
                                    html.H1("\u26a1 GridPulse", className="dashboard-title"),
                                    html.P(
                                        "Demand Forecasting & Analytics",
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
                                        style={"color": "#8a8fa8", "fontSize": "0.7rem"},
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
                                        "Persona", style={"color": "#8a8fa8", "fontSize": "0.7rem"}
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
                                                "\U0001f4f7 Present",
                                                id="meeting-mode-btn",
                                                size="sm",
                                                color="info",
                                                outline=True,
                                                className="me-2",
                                                style={"fontSize": "0.75rem"},
                                            ),
                                            dbc.Button(
                                                "\U0001f517 Bookmark",
                                                id="bookmark-btn",
                                                size="sm",
                                                color="secondary",
                                                outline=True,
                                                className="me-2",
                                                style={"fontSize": "0.75rem"},
                                            ),
                                            html.Div(
                                                id="header-freshness", style={"textAlign": "right"}
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
            # ── Per-Widget Confidence Badges (A4 + E3) ────────────
            html.Div(id="widget-confidence-bar"),
            # ── Welcome Card ───────────────────────────────────────
            html.Div(id="welcome-card"),
            # ── KPI Row ────────────────────────────────────────────
            html.Div(id="kpi-cards"),
            # ── Tabs (full width) ─────────────────────────────────
            dbc.Tabs(
                id="dashboard-tabs",
                active_tab="tab-forecast",
                children=[
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
                        tab_backtest.layout(),
                        label=TAB_LABELS["tab-backtest"],
                        tab_id="tab-backtest",
                    ),
                    dbc.Tab(
                        tab_generation.layout(),
                        label=TAB_LABELS["tab-generation"],
                        tab_id="tab-generation",
                    ),
                ],
            ),
            # ── News Ribbon (below tabs, full width) ─────────────
            html.Div(id="news-feed"),
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
        ],
        fluid=True,
        style={"padding": 0},
    )
