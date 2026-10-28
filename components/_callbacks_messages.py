"""Source-scoped message filtering and explicit inspect/return behavior."""

from dash import ALL, Input, Output, State, ctx, html, no_update

from components.weather_explorer import exact_table
from data.redis_client import redis_get, redis_key


def _message_source(data):
    if not data or data.get("source") == "unavailable":
        return "Weather messages unavailable; active-message status is unknown."
    if data.get("source") == "demo":
        return "Demonstration messages · not a live NWS feed."
    if not data.get("alerts"):
        return "The latest source update reported no active messages."
    return f"{len(data['alerts'])} source messages · {data.get('source', 'Source unspecified')} · updated {data.get('as_of') or 'clock unavailable'}. Counts are messages, not grid incidents."


def message_view(data, query, severity, selected):
    alerts = data.get("alerts") or []
    current = (
        selected.get("index", 0)
        if isinstance(selected, dict) and selected.get("region") == data.get("region")
        else 0
    )
    if not isinstance(current, int) or not 0 <= current < len(alerts):
        current = 0
    rows = []
    for i, a in enumerate(alerts):
        if severity not in ("all", a.get("severity")):
            continue
        if (
            str(query or "").lower()
            not in " ".join(str(a.get(k, "")) for k in ("event", "headline", "source")).lower()
        ):
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(a.get("severity", "Unspecified")),
                    html.Td(
                        [
                            html.Strong(a.get("event", "Weather message")),
                            html.Div(
                                a.get("source") or a.get("sender_name") or "NWS source",
                                className="gp-panel__caption",
                            ),
                        ]
                    ),
                    html.Td(str(a.get("expires") or "Unavailable")),
                    html.Td(
                        html.Button(
                            "Inspect →",
                            id={"type": "condition-message", "index": i},
                            n_clicks=0,
                            className="gp-review-button",
                            **{"aria-label": f"Inspect {a.get('event', 'message')} entry {i + 1}"},
                        )
                    ),
                ],
                className="gp-selected-row" if i == current else "",
            )
        )
    table = html.Div(
        [
            html.P(f"{len(rows)} of {len(alerts)} source entries", className="gp-panel__caption"),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(x)
                                    for x in (
                                        "App group",
                                        "Event / source",
                                        "Source expiry",
                                        "Inspect",
                                    )
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="gp-review-table gp-message-table",
                ),
                className="gp-review-table-scroll",
                role="region",
                tabIndex=0,
                **{"aria-label": "Weather messages; scroll horizontally if needed"},
            )
            if rows
            else html.P(
                "No messages match these filters. This is not a live all-clear status."
                if alerts
                else _message_source(data)
            ),
        ]
    )
    if not alerts:
        detail = html.P("Select a source message when the feed is available.")
    else:
        a = alerts[current]
        detail = html.Div(
            [
                html.Div("Selected source message", className="gp-control-eyebrow"),
                html.H2(
                    f"Entry {current + 1} · {a.get('event', 'Weather message')}",
                    id="conditions-detail-heading",
                    tabIndex=-1,
                    **{"data-message-index": str(current)},
                ),
                html.P(a.get("headline") or "Headline unavailable"),
                html.Hr(),
                html.Div("Source expiry", className="gp-control-eyebrow"),
                html.P(str(a.get("expires") or "Unavailable")),
                html.P(a.get("source") or a.get("sender_name") or "Source not retained"),
                html.Details(
                    [
                        html.Summary("Message evidence"),
                        exact_table(
                            ["Field", "Source value"],
                            [
                                [name, str(a.get(name, "Not retained"))]
                                for name in (
                                    "id",
                                    "severity",
                                    "urgency",
                                    "certainty",
                                    "area_desc",
                                    "effective",
                                    "onset",
                                    "expires",
                                    "source",
                                )
                            ],
                        ),
                        html.P(
                            "The source expiry keeps its supplied offset. If it was truncated upstream, no timezone is inferred. State association does not establish authority/asset intersection.",
                            className="gp-panel__hint",
                        ),
                    ],
                    className="gp-review-disclosure gp-message-evidence",
                ),
            ]
        )
    return table, detail


def register_message_callbacks(app):
    @app.callback(
        Output("conditions-message-data", "data"),
        [
            Input("region-selector", "value"),
            Input("refresh-interval", "n_intervals"),
            Input("dashboard-tabs", "active_tab"),
        ],
    )
    def load(region, _ticks, tab):
        if tab != "tab-alerts":
            return no_update
        raw = redis_get(redis_key(f"alerts:{region}"))
        return {
            "region": region,
            "alerts": raw.get("alerts", []) if isinstance(raw, dict) else [],
            "source": raw.get("alerts_source", "unavailable")
            if isinstance(raw, dict)
            else "unavailable",
            "as_of": raw.get("scored_at") if isinstance(raw, dict) else None,
        }

    @app.callback(
        [
            Output("conditions-message-list", "children"),
            Output("conditions-message-detail", "children"),
            Output("conditions-message-status", "children"),
        ],
        [
            Input("conditions-message-data", "data"),
            Input("conditions-message-search", "value"),
            Input("conditions-message-severity", "value"),
            Input("conditions-selected-message", "data"),
            Input("region-selector", "value"),
        ],
    )
    def render(data, query, severity, selected, region):
        if not isinstance(data, dict) or data.get("region") != region:
            return (
                html.P("Loading this authority’s source messages…"),
                html.Div(),
                "Message source pending",
            )
        table, detail = message_view(data, query, severity or "all", selected)
        return table, detail, _message_source(data)

    @app.callback(
        Output("conditions-selected-message", "data"),
        Input({"type": "condition-message", "index": ALL}, "n_clicks"),
        State("region-selector", "value"),
        prevent_initial_call=True,
    )
    def select(clicks, region):
        trigger = ctx.triggered_id
        if not any(clicks or []) or not isinstance(trigger, dict):
            return no_update
        return {"region": region, "index": trigger["index"]}

    app.clientside_callback(
        """function(selected){if(!selected)return window.dash_clientside.no_update;let attempts=0;const show=()=>{const h=document.getElementById('conditions-detail-heading');if(h&&h.dataset.messageIndex===String(selected.index)){window.gridpulseReview.reveal('conditions-message-inspector','conditions-detail-heading');}else if(attempts++<12)setTimeout(show,50);};setTimeout(show,50);return selected;}""",
        Output("conditions-focus-event", "data"),
        Input("conditions-selected-message", "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """function(clicks,selected){if(!clicks)return window.dash_clientside.no_update;const id=selected?JSON.stringify({index:selected.index,type:'condition-message'}):null;const el=(id&&document.getElementById(id))||document.getElementById('conditions-message-search');if(el){const box=el.closest('.gp-review-table-scroll');if(box){const a=el.getBoundingClientRect(),b=box.getBoundingClientRect();if(a.right>b.right)box.scrollLeft+=a.right-b.right+12;if(a.left<b.left)box.scrollLeft-=b.left-a.left+12;}window.gridpulseReview.reveal(el.id,el.id);}return clicks;}""",
        Output("conditions-back-event", "data"),
        Input("conditions-message-back", "n_clicks"),
        State("conditions-selected-message", "data"),
        prevent_initial_call=True,
    )
