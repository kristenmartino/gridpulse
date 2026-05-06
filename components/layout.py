"""Main dashboard layout — header, tab strip, data stores.

V1.β adds US Grid as the fifth visible tab on top of the V2.1 shell.
After V2.1 there are no hidden tabs — the five DOM-resident modules R3
once kept around (Historical / Backtest / Generation / Weather /
Simulator) were absorbed into the visible tabs in R4 and removed
entirely. Visible tabs now: Overview, US Grid, Forecast, Risk, Models.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components import (
    tab_alerts,
    tab_demand_outlook,
    tab_models,
    tab_overview,
    tab_us_grid,
)
from config import REGION_GROUPS, REGION_NAMES, TAB_LABELS
from personas.config import list_personas

# All five tabs in the strip are visible. ``_VISIBLE_TABS`` is kept as a
# constant so the smoke test can lock in the v2 shell composition.
_VISIBLE_TABS = {"tab-overview", "tab-us-grid", "tab-outlook", "tab-alerts", "tab-models"}

# Display labels for the visible tabs (overrides whatever TAB_LABELS
# says) — kept so the v2 naming is enforced regardless of config drift.
_VISIBLE_LABEL_OVERRIDES = {
    "tab-overview": "Overview",
    "tab-us-grid": "US Grid",
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
    # Region options are grouped geographically (Central / Northeast /
    # Southeast / West). dbc.Select doesn't support native <optgroup>,
    # so we surface each group with a disabled separator option that's
    # visually distinct but unselectable. The empty value="" never
    # reaches a callback because Bootstrap honors the disabled attribute
    # on click + keyboard nav.
    #
    # V3.ζ follow-up: the forecast quality gate
    # (``is_forecast_quality_acceptable``) hides BAs whose XGBoost
    # holdout MAPE is in the ``rollback`` grade (>22% on 7d horizon).
    # Empty groups (e.g. all members hidden) drop their separator too.
    from models.model_service import is_forecast_quality_acceptable

    region_options: list[dict] = []
    for group_name, codes in REGION_GROUPS.items():
        visible_codes = [c for c in codes if is_forecast_quality_acceptable(c)]
        if not visible_codes:
            continue
        region_options.append({"label": f"── {group_name} ──", "value": "", "disabled": True})
        for code in visible_codes:
            region_options.append({"label": REGION_NAMES.get(code, code), "value": code})

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

    # Keyboard-shortcut hints. The actual handlers live in
    # ``assets/accessibility.js`` (Alt+R focuses region, Alt+P focuses
    # persona, Alt+1..4 switch tabs). The visible <kbd> chips advertise
    # the shortcut without consuming label-row chrome — they sit
    # adjacent to the selector they target. Hidden under
    # ``body.briefing`` so the projection chrome stays clean.
    region_kbd_hint = html.Span(
        html.Kbd("⌥R"),
        className="gp-header__kbd-hint",
        title="Alt+R focuses the region selector",
        **{"aria-hidden": "true"},
    )
    persona_kbd_hint = html.Span(
        html.Kbd("⌥P"),
        className="gp-header__kbd-hint",
        title="Alt+P focuses the persona selector",
        **{"aria-hidden": "true"},
    )

    # The dbc.Select component strict-validates its kwargs and rejects
    # arbitrary aria-* attributes, so we can't put aria-keyshortcuts
    # directly on the selector. Wrap each selector + its kbd hint in
    # a labelled <div role="group"> instead — the role + the visible
    # chip together convey the shortcut to assistive tech, and the
    # browser's own focus model handles the Alt+R keystroke via
    # ``assets/accessibility.js``.
    region_group = html.Div(
        [
            dbc.Select(
                id="region-selector",
                options=region_options,
                value="FPL",
                className="gp-header__select region-selector",
            ),
            region_kbd_hint,
        ],
        className="gp-header__shortcut-group",
        role="group",
        **{"aria-label": "Balancing authority — Alt+R"},
    )
    persona_group = html.Div(
        [
            dbc.Select(
                id="persona-selector",
                options=persona_options,
                value="grid_ops",
                className="gp-header__chip persona-switcher",
            ),
            persona_kbd_hint,
        ],
        className="gp-header__shortcut-group",
        role="group",
        **{"aria-label": "Persona — Alt+P"},
    )

    controls = html.Div(
        [
            region_group,
            persona_group,
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
    """Wrap a tab module's layout in ``dbc.Tab``."""
    label = _VISIBLE_LABEL_OVERRIDES.get(tab_id, TAB_LABELS.get(tab_id, tab_id))
    return dbc.Tab(layout_fn(), label=label, tab_id=tab_id)


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
                        _tab("tab-us-grid", tab_us_grid.layout),
                        _tab("tab-outlook", tab_demand_outlook.layout),
                        _tab("tab-alerts", tab_alerts.layout),
                        _tab("tab-models", tab_models.layout),
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
