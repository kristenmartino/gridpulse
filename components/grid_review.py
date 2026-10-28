"""Reviewed authority directory: cache snapshots, one territory map and inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from config import REGION_COORDINATES, REGION_GROUPS, REGION_NAMES


def group_for(code):
    return next((name for name, codes in REGION_GROUPS.items() if code in codes), "Other")


def clock(value):
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return stamp.strftime("%d %b %H:%M UTC") if pd.notna(stamp) else "Clock unavailable"


def real(value):
    try:
        return value is not None and np.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def delta(data):
    current, previous = data.get("current_mw"), data.get("prev_mw")
    if not real(current) or not real(previous) or not data.get("hourly_pair"):
        return None
    return (current - previous) / previous * 100


def format_demand(data):
    return f"{data['current_mw']:,.0f} MW" if real(data.get("current_mw")) else "Unavailable"


def movement(data):
    value = delta(data)
    return f"{value:+.1f}%" if value is not None else "—"


def spark(data):
    values = data.get("today_mw") or []
    times = data.get("history_timestamps") or []
    if len(values) != len(times) or not any(real(x) for x in values):
        return html.Span("History unavailable", className="gp-muted")
    fig = go.Figure(
        go.Scatter(
            x=times,
            y=[float(x) if real(x) else None for x in values],
            mode="lines",
            connectgaps=False,
            line={"color": "#68a5f7", "width": 1.6},
            hovertemplate="%{x|%d %b %H:%M UTC}<br>%{y:,.0f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 2, "b": 2},
        height=42,
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
    )
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "responsive": True},
        style={"height": "42px", "minWidth": "100px"},
    )


def net(data):
    payload = data.get("interchange") or {}
    value = payload.get("net_mw")
    try:
        if value is not None and np.isfinite(float(value)):
            return f"{abs(float(value)):,.0f} MW {'export' if float(value) > 0 else 'import' if float(value) < 0 else 'net'}"
    except (TypeError, ValueError):
        pass
    return "Unavailable"


def ordered(data, search="", group="all", sort="change"):
    codes = [
        c
        for c in data
        if (group == "all" or group_for(c) == group)
        and (search.casefold() in (c + " " + REGION_NAMES.get(c, c)).casefold())
    ]
    if sort == "name":
        return sorted(codes, key=lambda c: REGION_NAMES.get(c, c))
    if sort == "demand":
        return sorted(codes, key=lambda c: -(data[c].get("current_mw") or 0))
    return sorted(codes, key=lambda c: -(abs(delta(data[c])) if delta(data[c]) is not None else -1))


def directory(data, codes, view, selected):
    if not codes:
        return html.P("No authorities match these filters.", className="gp-review-empty")
    if view == "cards":
        return html.Div(
            [
                html.Article(
                    [
                        html.Button(
                            [html.Strong(REGION_NAMES[c]), html.Small(c)],
                            id={"type": "us-grid-inspect", "region": c},
                            className="gp-grid-authority",
                        ),
                        html.H3(format_demand(data[c])),
                        html.P(
                            movement(data[c])
                            + " hourly movement · "
                            + clock(data[c].get("latest_at")),
                            className="gp-muted",
                        ),
                        spark(data[c]),
                        html.P(net(data[c]), className="gp-muted"),
                    ],
                    className="gp-review-panel" + (" is-selected" if c == selected else ""),
                )
                for c in codes
            ],
            className="gp-grid-cards",
        )
    return html.Div(
        html.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th(x, scope="col")
                            for x in (
                                "Authority",
                                "Reported demand",
                                "Δ 1h",
                                "Latest 24 records",
                                "Net interchange",
                                "Reported at",
                            )
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Th(
                                    html.Button(
                                        [
                                            html.Strong(REGION_NAMES[c]),
                                            html.Small(c + " · " + group_for(c)),
                                        ],
                                        id={"type": "us-grid-inspect", "region": c},
                                        className="gp-grid-authority",
                                    ),
                                    scope="row",
                                ),
                                html.Td(format_demand(data[c])),
                                html.Td(movement(data[c])),
                                html.Td(spark(data[c])),
                                html.Td(net(data[c])),
                                html.Td(clock(data[c].get("latest_at"))),
                            ],
                            className="is-selected" if c == selected else "",
                        )
                        for c in codes
                    ]
                ),
            ],
            className="gp-review-table",
        ),
        className="gp-table-scroll",
        tabIndex=0,
        role="region",
        **{"aria-label": "Authority directory"},
    )


def map_view(data, codes, selected, geojson):
    features = (geojson or {}).get("features", [])
    # Canonical asset features use the BA code as id; never synthesize territory.
    feature_codes = {str(f.get("properties", {}).get("region")) for f in features}
    mapped = [c for c in codes if c in feature_codes]
    missing = [c for c in codes if c not in feature_codes]
    fig = go.Figure()
    if mapped:
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                featureidkey="properties.region",
                locations=mapped,
                z=[
                    2 if c == selected else 1 if real(data[c].get("current_mw")) else 0
                    for c in mapped
                ],
                zmin=0,
                zmax=2,
                colorscale=[[0, "#283447"], [0.49, "#42698b"], [0.5, "#42698b"], [1, "#8cc9fc"]],
                showscale=False,
                marker_line_color="#9baabd",
                marker_line_width=0.5,
                customdata=[[c] for c in mapped],
                text=[
                    REGION_NAMES[c]
                    + "<br>"
                    + format_demand(data[c])
                    + "<br>"
                    + clock(data[c].get("latest_at"))
                    for c in mapped
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fallback = [c for c in missing if c in REGION_COORDINATES]
    if fallback:
        fig.add_trace(
            go.Scattergeo(
                lat=[REGION_COORDINATES[c][0] for c in fallback],
                lon=[REGION_COORDINATES[c][1] for c in fallback],
                mode="markers",
                marker={"size": 9, "color": "#8cc9fc", "symbol": "diamond"},
                customdata=[[c] for c in fallback],
                text=[REGION_NAMES[c] + " · representative weather lookup point" for c in fallback],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=440,
        geo={
            "scope": "usa",
            "projection_type": "albers usa",
            "bgcolor": "rgba(0,0,0,0)",
            "showland": True,
            "landcolor": "#151e2c",
            "showlakes": False,
            "showsubunits": True,
            "subunitcolor": "#344257",
            "coastlinecolor": "#344257",
        },
        clickmode="event+select",
        uirevision="grid-map",
    )
    return html.Div(
        [
            html.P(
                f"{len(mapped)}/{len(codes)} filtered authorities have territory geometry · {len(fallback)} representative point fallbacks. CONUS tracked fleet; not nationwide coverage or grid risk.",
                className="gp-muted",
            ),
            dcc.Graph(
                id="us-grid-map",
                figure=fig,
                config={"displayModeBar": False, "responsive": True, "topojsonURL": "/assets/geo/"},
            ),
            html.P(
                [
                    "Territories: ",
                    html.A(
                        "electricitymaps-contrib",
                        href="https://github.com/electricitymaps/electricitymaps-contrib",
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    " · AGPL-3.0. Shapes can overlap. Use the directory below for small authorities.",
                ],
                className="gp-muted",
            ),
            directory(data, codes, "table", selected),
        ]
    )


def inspector(code, data):
    if code not in data:
        return html.Section(
            [
                html.Div("Authority inspector", className="gp-control-eyebrow"),
                html.H3("Inspect an authority"),
                html.P(
                    "Select a directory entry or territory to review its source snapshot before opening Forecast.",
                    className="gp-muted",
                ),
            ],
            className="gp-workspace-panel gp-workspace-rail",
        )
    from config import (
        IS_IMPORT_DOMINATED,
        PEAK_DERIVED_CAPACITY,
        REGION_CAPACITY_MW,
        UNRELIABLE_CAPACITY,
    )

    row = data[code]
    capacity = REGION_CAPACITY_MW.get(code)
    capacity_label = f"{capacity:,.0f} MW" if real(capacity) else "Unavailable"
    qualification = (
        "Configured installed-capacity reference; availability and outages are not measured here."
    )
    if code in PEAK_DERIVED_CAPACITY:
        qualification = "Peak-derived capacity estimate, not measured installed nameplate."
    elif code in IS_IMPORT_DOMINATED or code in UNRELIABLE_CAPACITY:
        qualification = "Local capacity does not represent all supply serving this authority; demand/nameplate ratios are not used as risk grades."
    return [
        html.Section(
            [
                html.Div("Authority inspector", className="gp-control-eyebrow"),
                html.H3(REGION_NAMES[code], id="us-grid-inspector-title", tabIndex=-1),
                html.P(code + " · " + group_for(code), className="gp-muted"),
                html.H2(format_demand(row)),
                html.Div(
                    [
                        html.Div([html.Span("Δ 1h"), html.Strong(movement(row))]),
                        html.Div([html.Span("Net interchange"), html.Strong(net(row))]),
                    ],
                    className="gp-inspector-metrics",
                ),
                html.P(
                    "Reported at "
                    + clock(row.get("latest_at"))
                    + " · cache "
                    + clock(row.get("scored_at")),
                    className="gp-muted",
                ),
                spark(row),
                html.P(
                    f"{len(row.get('artifact_excluded') or [])} source-stamped quality exclusions. Readings retain their individual reporting clocks.",
                    className="gp-muted",
                ),
                html.Button(
                    "Review forecast →",
                    id="us-grid-review",
                    n_clicks=0,
                    className="gp-review-action",
                ),
            ],
            className="gp-workspace-panel gp-workspace-rail gp-authority-card",
        ),
        html.Details(
            [
                html.Summary("Installed-capacity context"),
                html.Div(
                    [
                        html.Strong(capacity_label),
                        html.P(qualification),
                        html.P(
                            "Reference configuration only; this is not operating reserves, outage probability or grid stress."
                        ),
                    ],
                    className="gp-grid-scope-copy",
                ),
            ],
            className="gp-review-disclosure",
        ),
    ]
