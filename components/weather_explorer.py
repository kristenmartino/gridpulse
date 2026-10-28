"""Native Dash/Plotly explorer layout and pure view rendering."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from components._callbacks_shared import _empty_figure, _layout
from components.weather_explorer_data import WEATHER_FIELDS, number, utc

VARIABLES = [
    {"value": k, "label": f"{label} · {unit}"} for k, (_, label, unit) in WEATHER_FIELDS.items()
]
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


def clock(value):
    stamp = utc(value)
    return stamp.strftime("%d %b %Y, %H:%M UTC") if stamp is not None else "Clock unavailable"


def exact_table(headings, rows, label="Exact source values"):
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(x) for x in headings])),
                html.Tbody([html.Tr([html.Td(value) for value in row]) for row in rows]),
            ],
            className="gp-review-table gp-exact-table",
        ),
        className="gp-review-table-scroll",
        role="region",
        tabIndex=0,
        **{"aria-label": label + "; scroll horizontally if needed"},
    )


def layout():
    return html.Div(
        [
            dcc.Store(id="explorer-data"),
            dcc.Store(id="explorer-state", storage_type="session"),
            dcc.Store(id="explorer-window"),
            dcc.Store(id="explorer-cursor"),
            dcc.Store(id="explorer-date-edit"),
            html.Div(
                [
                    html.H2("Shared UTC window", className="sr-only"),
                    dbc.RadioItems(
                        id="explorer-mode",
                        options=[
                            {"label": "Past", "value": "past"},
                            {"label": "Future", "value": "future"},
                            {"label": "Error review", "value": "error"},
                        ],
                        value="past",
                        inline=True,
                        className="gp-explorer-modes",
                    ),
                    html.Div(id="explorer-source-badge", className="gp-explorer-source"),
                ],
                className="gp-explorer-heading",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Window begins · UTC", htmlFor="explorer-start"),
                                    dbc.Input(
                                        id="explorer-start",
                                        type="datetime-local",
                                        step=3600,
                                        debounce=True,
                                    ),
                                    html.Div(
                                        id="explorer-date-feedback",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                ],
                                className="gp-explorer-control gp-explorer-date-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Window length", htmlFor="explorer-span"),
                                    dbc.RadioItems(
                                        id="explorer-span",
                                        options=[
                                            {"label": "24h", "value": 24},
                                            {"label": "7d", "value": 168},
                                        ],
                                        value=168,
                                        inline=True,
                                        className="gp-segmented",
                                    ),
                                ],
                                className="gp-explorer-control gp-explorer-span-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Weather variable", htmlFor="explorer-variable"),
                                    dbc.Select(
                                        id="explorer-variable",
                                        options=VARIABLES,
                                        value="temperature_f",
                                    ),
                                ],
                                className="gp-explorer-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Forecast source", htmlFor="explorer-source"),
                                    dbc.Select(id="explorer-source", value="published"),
                                ],
                                id="explorer-source-control",
                                className="gp-explorer-control",
                            ),
                            html.Div(
                                [
                                    html.Label("Published issue", htmlFor="explorer-issue"),
                                    dbc.Select(id="explorer-issue"),
                                ],
                                id="explorer-issue-control",
                                className="gp-explorer-control",
                            ),
                            html.Div(
                                id="explorer-series-scope",
                                className="gp-explorer-series-scope",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "← Previous",
                                        id="explorer-prev",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                    html.Button(
                                        "Next →",
                                        id="explorer-next",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                    html.Button(
                                        "Latest window",
                                        id="explorer-reset",
                                        n_clicks=0,
                                        className="gp-review-button",
                                    ),
                                ],
                                className="gp-explorer-window-actions",
                            ),
                        ],
                        className="gp-explorer-toolbar",
                    ),
                    html.Div(id="explorer-summary", className="gp-explorer-summary"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3(id="explorer-demand-title"),
                                            html.Span("MW", id="explorer-demand-unit"),
                                        ],
                                        className="gp-explorer-plot-head",
                                    ),
                                    dcc.Graph(
                                        id="explorer-demand",
                                        config=GRAPH_CONFIG,
                                        className="gp-explorer-graph",
                                    ),
                                ],
                                className="gp-explorer-plot",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3(id="explorer-weather-title"),
                                            html.Span(id="explorer-weather-unit"),
                                        ],
                                        className="gp-explorer-plot-head",
                                    ),
                                    dcc.Graph(
                                        id="explorer-weather",
                                        config=GRAPH_CONFIG,
                                        className="gp-explorer-graph",
                                    ),
                                ],
                                className="gp-explorer-plot",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [html.H3("Signed forecast error"), html.Span("MW")],
                                        className="gp-explorer-plot-head",
                                    ),
                                    dcc.Graph(
                                        id="explorer-error",
                                        config=GRAPH_CONFIG,
                                        className="gp-explorer-graph",
                                    ),
                                ],
                                id="explorer-error-panel",
                                className="gp-explorer-plot",
                            ),
                        ],
                        id="explorer-plots",
                        className="gp-explorer-plots",
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="explorer-inspected", role="status", **{"aria-live": "polite"}
                            ),
                            html.Div(
                                [
                                    html.Label("Inspect a shared hour", htmlFor="explorer-hour"),
                                    dbc.Input(
                                        id="explorer-hour",
                                        type="range",
                                        min=0,
                                        max=0,
                                        value=0,
                                        step=1,
                                    ),
                                ],
                                className="gp-explorer-scrubber",
                            ),
                            html.P(
                                "Drag the shared hour or use arrow keys, Home and End. Each plot highlights the same UTC hour.",
                                className="gp-panel__caption",
                            ),
                        ],
                        className="gp-explorer-inspection",
                    ),
                    html.P(id="explorer-scope", className="gp-explorer-scope"),
                ],
                className="gp-panel gp-explorer-panel",
            ),
            html.Div(
                [
                    html.Details(
                        [
                            html.Summary("Sources and interpretation"),
                            html.Div(id="explorer-provenance"),
                        ],
                        className="gp-review-disclosure",
                    ),
                    html.Details(
                        [html.Summary("Exact values"), html.Div(id="explorer-exact")],
                        className="gp-review-disclosure",
                    ),
                ],
                className="gp-explorer-evidence",
            ),
            html.P(id="explorer-available-range", className="gp-explorer-available-range"),
        ],
        id="weather-explorer",
        className="gp-weather-explorer",
    )


def figure(rows, series, key, unit="MW", zero=False, error=False, cursor=None):
    if not rows:
        return _empty_figure("No matching source records are available for this selection.")
    if not any(number(row.get(column)) is not None for _, column, _ in series for row in rows):
        return _empty_figure("No usable source values in this window. Gaps are not filled.")
    fig = go.Figure()
    for name, column, color in series:
        values, stamps, previous = [], [], None
        for row in rows:
            current = utc(row["timestamp"])
            if previous is not None and current - previous > pd.Timedelta(hours=1):
                stamps.append((previous + pd.Timedelta(hours=1)).isoformat())
                values.append(None)
            stamps.append(row["timestamp"])
            values.append(row.get(column))
            previous = current
        fig.add_trace(
            go.Scatter(
                x=stamps,
                y=values,
                name=name,
                mode="lines+markers" if len(rows) == 1 else "lines",
                connectgaps=False,
                line={"color": color, "width": 2},
                marker={"size": 6},
                hovertemplate="%{x|%b %d %H:%M UTC}<br>%{y:,.2f} "
                + unit
                + "<extra>"
                + name
                + "</extra>",
            )
        )
        selected = next((r for r in rows if r["timestamp"] == cursor), rows[0])
        fig.add_trace(
            go.Scatter(
                x=[selected["timestamp"]],
                y=[selected.get(column)],
                mode="markers",
                marker={"size": 7, "color": color, "line": {"color": "#dce8f4", "width": 1}},
                name="Selected hour",
                showlegend=False,
                hoverinfo="skip",
                meta={"cursor_column": column},
            )
        )
    first, last = utc(rows[0]["timestamp"]), utc(rows[-1]["timestamp"])
    if first == last:
        first, last = first - pd.Timedelta(minutes=30), last + pd.Timedelta(minutes=30)
    fig.update_layout(
        **_layout(
            height=190,
            margin={"l": 52, "r": 12, "t": 25, "b": 32},
            paper_bgcolor="#111820",
            plot_bgcolor="#111820",
            font={
                "family": "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                "size": 10,
                "color": "#a5b7c9",
            },
            uirevision=key,
            hovermode="x",
            yaxis_title=None,
            xaxis={
                "range": [first, last],
                "tickformat": "%b %d, %H",
                "nticks": 3,
                "title": None,
                "showgrid": False,
            },
            legend={"orientation": "h", "y": 1.17, "font": {"size": 10}},
            meta={"window_key": key, "unit": unit},
        )
    )
    fig.update_yaxes(
        rangemode="tozero" if zero or error else "normal",
        zeroline=error,
        zerolinecolor="#acbfd2",
        fixedrange=True,
        gridcolor="#2b3a4c",
        griddash="dot",
        tickformat=",.0f" if unit == "MW" else ".1f",
    )
    fig.update_xaxes(fixedrange=True)
    if cursor:
        fig.add_vline(x=cursor, line={"color": "#a9bed4", "dash": "dot", "width": 1})
    return fig


def statistics(rows, mode, variable, source_label):
    column = "demand_mw" if mode == "past" else "actual_mw" if mode == "error" else "forecast_mw"
    values = [number(r.get(column)) for r in rows]
    values = [x for x in values if x is not None]
    weather = [number(r.get(variable)) for r in rows]
    weather = [x for x in weather if x is not None]
    _, label, unit = WEATHER_FIELDS[variable]
    result = [
        html.Span(
            [
                html.Strong(str(len(rows))),
                " target hours",
                " · " + clock(rows[0]["timestamp"]) + " → " + clock(rows[-1]["timestamp"])
                if rows
                else "",
            ]
        )
    ]
    if values:
        result.append(
            html.Span(
                f"Mean {'forecast' if mode == 'future' else 'demand'} {sum(values) / len(values):,.0f} MW"
            )
        )
    if weather:
        result.append(
            html.Span(
                f"{label} {min(weather):.1f}–{max(weather):.1f} {unit} · {len(weather)}/{len(rows)} values"
            )
        )
    elif rows:
        result.append(html.Span(f"{label}: no matched values in this window"))
    if mode == "error" and rows:
        result.append(
            html.Span(
                f"Selected-window MAE {sum(abs(r['error_mw']) for r in rows) / len(rows):,.0f} MW"
            )
        )
    before = sum(r.get("generated_before_target") is False for r in rows) if mode == "future" else 0
    if before:
        result.insert(
            1,
            html.Span(
                f"Includes {before} stored target(s) before generation",
                className="gp-review-caution",
            ),
        )
    return result
