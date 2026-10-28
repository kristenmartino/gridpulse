"""Cache-only explorer control/render callbacks; inspection stays client-side."""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, ctx, html, no_update

from components.explorer_state import (
    SPANS,
    clean_state,
    default_start,
    normalize_window,
    rows_for,
    validate_start,
)
from components.weather_explorer import clock, exact_table, figure, statistics
from components.weather_explorer_data import WEATHER_FIELDS, load_explorer_data, utc


def _date(value):
    stamp = utc(value)
    return stamp.strftime("%Y-%m-%dT%H:00") if stamp is not None else ""


def render_explorer(data, state):
    mode, variable = state["mode"], state["variable"]
    state, rows = normalize_window(data, state)
    source = (data.get("future") or {}).get(state["source"]) or {}
    window = state["windows"][mode]
    withheld = (
        mode == "future"
        and source.get("max_horizon") is not None
        and window["hours"] > source["max_horizon"]
    )
    if withheld:
        rows = []
    issue = next((x for x in data.get("issues", []) if x["publication_id"] == state["issue"]), {})
    label = source.get("label", "Selected forecast")
    key = ":".join(
        map(
            str,
            (
                data.get("region"),
                mode,
                variable,
                state["source"],
                state["issue"],
                window["start"],
                window["hours"],
            ),
        )
    )
    selected = next((i for i, r in enumerate(rows) if r["timestamp"] == window["cursor"]), 0)
    demand_col = (
        "demand_mw" if mode == "past" else "actual_mw" if mode == "error" else "forecast_mw"
    )
    demand_series = (
        [
            ("Earlier published forecast", "forecast_mw", "#efa95a"),
            ("Reported demand", "actual_mw", "#5b9fff"),
        ]
        if mode == "error"
        else [
            (
                "Reported demand" if mode == "past" else label,
                demand_col,
                "#5b9fff" if mode == "past" else "#efa95a",
            )
        ]
    )
    _, weather_label, unit = WEATHER_FIELDS[variable]
    weather_kind = (
        "Forecast weather"
        if mode == "future"
        else "Later historical weather"
        if mode == "error"
        else "Historical model weather"
    )
    figs = [
        figure(rows, demand_series, key + ":demand", zero=True, cursor=window["cursor"]),
        figure(
            rows,
            [(weather_kind, variable, "#70bcb2")],
            key + ":weather",
            unit=unit,
            zero=variable != "temperature_f",
            cursor=window["cursor"],
        ),
        figure(
            rows,
            [("Actual − forecast", "error_mw", "#ba9cda")],
            key + ":error",
            error=True,
            cursor=window["cursor"],
        )
        if mode == "error"
        else figure([], [], key),
    ]
    if not rows:
        if withheld:
            reason = f"The {window['hours']}h {label} curve is withheld by its horizon guard. Choose a supported shorter horizon."
        elif mode == "error":
            reason = "No verified issue/eligible actual matches are available. Choose a cached published issue; original weather as issued is not available."
            if issue and issue.get("receipt_status") == "unavailable":
                reason = "This issue receipt is unavailable in the read cache. Choose another issue; no history is reconstructed."
            elif issue and not issue.get("loaded"):
                reason = "Loading the selected cached published issue."
        else:
            reason = "The selected source has no cached target records for this authority. No replacement curve is generated."
        from components._callbacks_shared import _empty_figure

        figs = [_empty_figure(reason) for _ in figs]
        summary = [html.Span(reason)]
    else:
        summary = statistics(rows, mode, variable, label)
    if mode == "past":
        badge = [
            html.Strong("Reported demand + historical weather"),
            html.Small("Demand cache · " + clock(data.get("actuals_at"))),
            html.Small("Weather cache · " + clock(data.get("weather_at"))),
        ]
        scope = (
            "Historical weather is regional model context. Shared movement does not prove cause."
        )
    elif mode == "future":
        badge = [
            html.Strong(label + " · " + source.get("status", "Unavailable")),
            html.Small("Generated · " + clock(source.get("generated_at"))),
        ]
        scope = (
            "This published baseline repeats the last observed demand day without explicit future-weather adjustment."
            if source.get("metric_key") == "baseline"
            else "Trained comparisons use weather with other inputs. This view does not quantify causal factor contributions or establish original weather issuance."
        )
        if source.get("metric_key") is None:
            scope = (
                source.get("note")
                or "The selected forecast source is unavailable; no weather-response attribution is made."
            )
    else:
        badge = [
            html.Strong(issue.get("series_label") or "Published issue unavailable"),
            html.Small("Generated · " + clock(issue.get("timing", {}).get("generated_at"))),
            html.Small("Actual cache · " + clock(data.get("actuals_at"))),
        ]
        scope = "Signed error is actual − forecast. Weather is later historical context; expected weather as issued was not retained. Selected-window MAE is separate from the official hourly-sampled score."
    provenance = [
        html.P(
            [
                html.Strong("Sources for this selection. "),
                f"Demand cache {clock(data.get('actuals_at'))}; weather cache {clock(data.get('weather_at'))}. Historical weather includes regional reanalysis/recent backfill, not station measurements.",
            ]
        )
    ]
    if mode == "future":
        provenance += [
            html.P(
                f"The selected {label} run was generated {clock(source.get('generated_at'))}. Only this cached payload supplies comparison model values. The exact original weather input/issue vintage is not retained. Stored targets before generation remain marked; the default begins after that clock."
            )
        ]
    if mode == "error":
        provenance += [
            html.P(
                f"Original issue {state['issue'] or 'unavailable'} · generated {clock(issue.get('timing', {}).get('generated_at'))}; observed published {clock(issue.get('observed_published_at'))}. Actuals are the later cache snapshot shown above and may revise. Only targets after the issue's recorded clocks and quality-eligible actuals within the last {data.get('error_window_hours', 168)}h are compared."
            ),
            html.P(
                f"Quality exclusions in the retained scoring window: {len(data.get('actual_exclusions') or {})}. Expected weather as issued was not retained, so weather-input error cannot be separated from model error here."
            ),
        ]
    if mode == "error" and data.get("quality_inputs") == "legacy_without_operator_forecast":
        provenance.append(
            html.P(
                "This legacy actual cache does not retain the operator forecast array, so the day-ahead-ratio quality signal is unavailable. Other retained checks and stamped exclusions are still applied."
            )
        )
    if mode == "error" and data.get("quality_inputs") == "operator_forecast_provided":
        provenance.append(
            html.P(
                f"Latest observed operator forecasts are provided for {data.get('operator_forecast_finite_hours', 0)} retained hours. A missing per-hour input disables the day-ahead-ratio signal for that hour; these are not original issue vintages."
            )
        )
    provenance += [
        html.P(
            "Reported demand may contain provider revisions and historical outliers. Missing values remain gaps. This view never replays a model or reconstructs an original weather forecast."
        ),
        html.P(
            "An empty publication index means issue history is unavailable in this read cache; legacy chart replay is a separate dataset."
        ),
    ]
    headers = [
        "Target · UTC",
        "Reported demand · MW"
        if mode == "past"
        else label + " · MW"
        if mode == "future"
        else "Earlier forecast · MW",
    ]
    if mode == "error":
        headers += ["Reported actual · MW", "Actual − forecast · MW"]
    headers += [weather_label + " · " + unit]
    table_rows = []
    for row in rows:
        first = (
            (str(row.get(demand_col)) if row.get(demand_col) is not None else "Gap")
            if mode != "error"
            else str(row.get("forecast_mw"))
        )
        values = [
            row["timestamp"]
            + (
                " · stored before generation"
                if mode == "future" and row.get("generated_before_target") is False
                else ""
            ),
            first,
        ]
        if mode == "error":
            values += [str(row.get("actual_mw")), str(row.get("error_mw"))]
        values += [str(row.get(variable)) if row.get(variable) is not None else "Gap"]
        table_rows.append(values)
    payload = {
        "key": key,
        "region": data.get("region"),
        "mode": mode,
        "variable": variable,
        "weather_label": weather_label,
        "unit": unit,
        "demand_column": demand_col,
        "demand_label": "Reported actual"
        if mode == "error"
        else "Reported demand"
        if mode == "past"
        else label,
        "rows": rows,
        "selected_index": selected,
        "window_start": window["start"],
    }
    return (
        summary,
        *figs,
        payload,
        badge,
        scope,
        provenance,
        exact_table(headers, table_rows),
        "Reported demand"
        if mode == "past"
        else "Forecast and reported demand"
        if mode == "error"
        else "Demand forecast",
        weather_label,
        {"display": "block" if mode == "error" else "none"},
        "gp-explorer-plots gp-explorer-plots--error" if mode == "error" else "gp-explorer-plots",
        max(0, len(rows) - 1),
    )


def register_explorer_callbacks(app):
    @app.callback(
        [
            Output("explorer-state", "data", allow_duplicate=True),
            Output("conditions-section", "value"),
            Output("explorer-mode", "value"),
            Output("explorer-span", "value"),
            Output("explorer-span", "options"),
            Output("explorer-variable", "value"),
            Output("explorer-source", "value"),
            Output("explorer-source", "options"),
            Output("explorer-issue", "value"),
            Output("explorer-issue", "options"),
            Output("explorer-start", "value"),
            Output("explorer-start", "min"),
            Output("explorer-start", "max"),
            Output("explorer-date-feedback", "children"),
            Output("explorer-start", "invalid"),
            Output("explorer-prev", "disabled"),
            Output("explorer-next", "disabled"),
            Output("explorer-reset", "children"),
            Output("explorer-source-control", "style"),
            Output("explorer-issue-control", "style"),
            Output("weather-explorer", "style"),
            Output("conditions-messages", "style"),
            Output("explorer-data", "data"),
            Output("published-review-earlier", "options"),
            Output("published-review-earlier", "value"),
            Output("published-review-later", "options"),
            Output("published-review-later", "value"),
        ],
        [
            Input("explorer-data", "data"),
            Input("explorer-state", "data"),
            Input("conditions-section", "value"),
            Input("explorer-mode", "value"),
            Input("explorer-span", "value"),
            Input("explorer-variable", "value"),
            Input("explorer-source", "value"),
            Input("explorer-issue", "value"),
            Input("explorer-start", "value"),
            Input("explorer-start", "n_blur"),
            Input("explorer-start", "n_submit"),
            Input("explorer-date-edit", "data"),
            Input("explorer-prev", "n_clicks"),
            Input("explorer-next", "n_clicks"),
            Input("explorer-reset", "n_clicks"),
            Input("region-selector", "value"),
            Input("refresh-interval", "n_intervals"),
            Input("dashboard-tabs", "active_tab"),
            Input("published-review-earlier", "value"),
            Input("published-review-later", "value"),
        ],
        [State("explorer-cursor", "data"), State("explorer-date-feedback", "children")],
        prevent_initial_call="initial_duplicate",
    )
    def controls(
        data,
        raw,
        section,
        mode,
        span,
        variable,
        source,
        issue,
        start,
        _blur,
        _submit,
        date_edit,
        _prev,
        _next,
        _reset,
        region,
        _ticks,
        tab,
        earlier,
        later,
        cursor,
        prior_feedback,
    ):
        state = clean_state(raw, region)
        old_mode = state["mode"]
        owner_start = _date(state["windows"][old_mode]["start"])
        trigger = ctx.triggered_id
        changed = set(ctx.triggered_prop_ids.values())
        if tab not in ("tab-alerts", "tab-outlook"):
            return [no_update] * 27
        selected = tuple(x for x in (issue, earlier, later) if x)
        indexed = (
            {x.get("publication_id"): x for x in (data or {}).get("issues", [])}
            if isinstance(data, dict)
            else {}
        )
        reload_data = (
            not isinstance(data, dict)
            or data.get("region") != region
            or trigger in ("region-selector", "refresh-interval", "dashboard-tabs")
            or any(
                indexed.get(key, {}).get("receipt_status") == "not_requested" for key in selected
            )
        )
        if reload_data:
            data = load_explorer_data(region, selected_ids=selected)
        data_output = data if reload_data else no_update
        if (
            isinstance(cursor, dict)
            and cursor.get("region") == region
            and cursor.get("mode") == state["mode"]
        ):
            state["windows"][state["mode"]]["cursor"] = cursor.get("timestamp") or ""
        if "conditions-section" in changed:
            state["section"] = section if section in ("explorer", "messages") else "explorer"
        if "explorer-mode" in changed and mode in SPANS:
            state["mode"] = mode
        if "explorer-variable" in changed and variable in WEATHER_FIELDS:
            state["variable"] = variable
        if "explorer-source" in changed and source in (
            "published",
            "prophet",
            "arima",
            "xgboost",
            "ensemble",
        ):
            state["source"] = source
        if "explorer-issue" in changed:
            state["issue"] = issue or ""
        win = state["windows"][state["mode"]]
        if "explorer-span" in changed and span in SPANS[state["mode"]]:
            win["hours"] = span
        state, rows = normalize_window(data, state)
        win = state["windows"][state["mode"]]
        all_rows = rows_for(data, state)
        feedback = prior_feedback or ""
        if trigger in (
            "explorer-mode",
            "conditions-section",
            "region-selector",
            "explorer-source",
            "explorer-issue",
            "explorer-reset",
            "explorer-prev",
            "explorer-next",
        ):
            feedback = ""
        edit_changed = "explorer-date-edit" in changed
        owner = date_edit.get("scope") or {} if isinstance(date_edit, dict) else {}
        owned_edit = (
            edit_changed
            and owner.get("region") == region
            and owner.get("mode") == state["mode"]
            and owner.get("window_start") == owner_start
            and (state["mode"] != "future" or owner.get("source") == state["source"])
        )
        switching_mode = "explorer-mode" in changed and old_mode != state["mode"]
        date_changed = (
            "explorer-start" in changed and not switching_mode and "region-selector" not in changed
        ) or owned_edit
        edited_value = date_edit.get("value") if owned_edit else start
        if trigger in ("explorer-prev", "explorer-next", "explorer-reset"):
            date_changed = False
        if date_changed:
            problem = validate_start(edited_value, all_rows)
            if problem:
                feedback = problem + " Start unchanged; the last valid window is still shown."
            else:
                if utc(edited_value) != utc(win["start"]):
                    feedback = ""
                win["start"] = utc(edited_value).isoformat()
        if (
            "explorer-span" in changed
            and not date_changed
            and validate_start(start, all_rows) is None
        ):
            win["start"] = utc(start).isoformat()
        if trigger in ("explorer-prev", "explorer-next") and all_rows:
            amount = (
                max(
                    1,
                    int(
                        (
                            utc(all_rows[-1]["timestamp"]) - utc(all_rows[0]["timestamp"])
                        ).total_seconds()
                        / 3600
                    )
                    + 1,
                )
                if win["hours"] == "all"
                else win["hours"]
            ) * (1 if trigger == "explorer-next" else -1)
            win["start"] = (utc(win["start"]) + pd.Timedelta(hours=amount)).isoformat()
            win["cursor"] = ""
        if trigger == "explorer-reset" and all_rows:
            generated = (data.get("future", {}).get(state["source"]) or {}).get("generated_at")
            win["start"] = default_start(all_rows, state["mode"], win["hours"], generated)
            win["cursor"] = ""
        state, rows = normalize_window(data, state)
        win = state["windows"][state["mode"]]
        issue_options = [
            {
                "label": clock(x["timing"].get("generated_at"))
                + " · "
                + str(x.get("series_label") or "Published"),
                "value": x["publication_id"],
            }
            for x in data.get("issues", [])
        ]
        source_options = list(data.get("sources") or [])
        if state["source"] not in {x["value"] for x in source_options}:
            source_options.append(
                {"label": state["source"] + " · unavailable", "value": state["source"]}
            )
        span_options = [
            {"label": "All matches" if h == "all" else "7d" if h == 168 else f"{h}h", "value": h}
            for h in SPANS[state["mode"]]
        ]
        end = (
            (
                utc(all_rows[-1]["timestamp"]) + pd.Timedelta(hours=1)
                if win["hours"] == "all"
                else utc(win["start"]) + pd.Timedelta(hours=win["hours"])
            )
            if all_rows
            else None
        )
        publication_options = [
            {
                "label": clock(x["timing"].get("generated_at"))
                + " · "
                + str(x.get("series_label") or "Published"),
                "value": x["publication_id"],
            }
            for x in data.get("issues", [])
        ]
        publication_ids = [x["value"] for x in publication_options]
        selected_earlier = (
            earlier
            if earlier in publication_ids
            else (
                publication_ids[1]
                if len(publication_ids) > 1
                else publication_ids[0]
                if publication_ids
                else None
            )
        )
        selected_later = (
            later if later in publication_ids else (publication_ids[0] if publication_ids else None)
        )
        return (
            state if state != raw else no_update,
            state["section"],
            state["mode"],
            win["hours"],
            span_options,
            state["variable"],
            state["source"],
            source_options,
            state["issue"] or None,
            issue_options,
            _date(win["start"]),
            _date(all_rows[0]["timestamp"]) if all_rows else None,
            _date(all_rows[-1]["timestamp"]) if all_rows else None,
            feedback,
            bool(feedback),
            not all_rows or utc(win["start"]) <= utc(all_rows[0]["timestamp"]),
            not all_rows or end > utc(all_rows[-1]["timestamp"]),
            "First prospective window" if state["mode"] == "future" else "Latest window",
            {"display": "block" if state["mode"] == "future" else "none"},
            {"display": "block" if state["mode"] == "error" else "none"},
            {"display": "block" if state["section"] == "explorer" else "none"},
            {"display": "block" if state["section"] == "messages" else "none"},
            data_output,
            publication_options,
            selected_earlier,
            publication_options,
            selected_later,
        )

    @app.callback(
        [
            Output("explorer-summary", "children"),
            Output("explorer-demand", "figure"),
            Output("explorer-weather", "figure"),
            Output("explorer-error", "figure"),
            Output("explorer-window", "data"),
            Output("explorer-source-badge", "children"),
            Output("explorer-scope", "children"),
            Output("explorer-provenance", "children"),
            Output("explorer-exact", "children"),
            Output("explorer-demand-title", "children"),
            Output("explorer-weather-title", "children"),
            Output("explorer-error-panel", "style"),
            Output("explorer-plots", "className"),
            Output("explorer-hour", "max"),
        ],
        [
            Input("explorer-data", "data"),
            Input("explorer-state", "data"),
            Input("region-selector", "value"),
        ],
    )
    def render(data, state, region):
        if not isinstance(data, dict) or data.get("region") != region:
            data = {"region": region, "past": [], "future": {}, "issues": [], "error": {}}
        return render_explorer(data, clean_state(state, region))

    app.clientside_callback(
        """function(windowData,index,demandHover,weatherHover,errorHover,demandClick,weatherClick,errorClick,previousCursor){return window.gridpulseReview.inspect(windowData,index,[demandHover,weatherHover,errorHover,demandClick,weatherClick,errorClick],previousCursor);}""",
        [
            Output("explorer-cursor", "data"),
            Output("explorer-inspected", "children"),
            Output("explorer-hour", "value"),
        ],
        [
            Input("explorer-window", "data"),
            Input("explorer-hour", "value"),
            Input("explorer-demand", "hoverData"),
            Input("explorer-weather", "hoverData"),
            Input("explorer-error", "hoverData"),
            Input("explorer-demand", "clickData"),
            Input("explorer-weather", "clickData"),
            Input("explorer-error", "clickData"),
        ],
        State("explorer-cursor", "data"),
    )

    @app.callback(
        Output("published-review-content", "children"),
        [
            Input("explorer-data", "data"),
            Input("published-review-earlier", "value"),
            Input("published-review-later", "value"),
            Input("region-selector", "value"),
        ],
    )
    def publication_review(data, earlier_id, later_id, region):
        from dash import dcc

        from components.weather_explorer_data import compare_issues

        if not isinstance(data, dict) or data.get("region") != region:
            return html.P("Loading this authority’s issue history…")
        issues = {x["publication_id"]: x for x in data.get("issues", [])}
        earlier, later = issues.get(earlier_id), issues.get(later_id)
        if not earlier or not later:
            return html.P(
                "Published issue history is not available in the read cache for this authority. No past run is reconstructed. Error review becomes available when an immutable issue and usable actuals overlap."
            )
        if any(x.get("receipt_status") == "unavailable" for x in (earlier, later)):
            return html.P(
                "A selected issue receipt is unavailable in this read cache. Choose another issue."
            )
        if any(x.get("receipt_status") == "unavailable" for x in (earlier, later)):
            return html.P(
                "A selected issue receipt is unavailable in this read cache. Choose another issue; no history is reconstructed."
            )
        if not earlier.get("loaded") or not later.get("loaded"):
            return html.P("Loading selected cached issues…")
        if earlier_id == later_id:
            return html.P("Choose two different published issues to inspect run-to-run changes.")
        if utc(earlier["timing"]["generated_at"]) >= utc(later["timing"]["generated_at"]):
            return html.P(
                "The earlier issue must precede the later issue. Check the generation clocks in the selectors."
            )
        rows = compare_issues(earlier, later)
        body = [
            html.P(
                f"Earlier: {earlier.get('series_label')} · {clock(earlier['timing']['generated_at'])}. Later: {later.get('series_label')} · {clock(later['timing']['generated_at'])}. Differences describe published curve changes, not forecast error.",
                className="gp-panel__hint",
            )
        ]
        if rows:
            body += [
                dcc.Graph(
                    figure=figure(
                        rows,
                        [
                            ("Earlier issue", "earlier_mw", "#5b9fff"),
                            ("Later issue", "later_mw", "#efa95a"),
                        ],
                        earlier_id + later_id,
                        zero=True,
                    ),
                    config={"displayModeBar": False, "responsive": True},
                ),
                html.Details(
                    [
                        html.Summary(f"Exact shared future targets · {len(rows)}"),
                        exact_table(
                            ["UTC target", "Earlier · MW", "Later · MW", "Later − earlier · MW"],
                            [
                                [r["timestamp"], r["earlier_mw"], r["later_mw"], r["change_mw"]]
                                for r in rows
                            ],
                        ),
                    ],
                    className="gp-review-disclosure",
                ),
            ]
        else:
            body.append(
                html.P(
                    "These issues have no shared targets strictly after both recorded issue clocks."
                )
            )
        matched = data.get("error", {}).get(earlier_id) or []
        body.append(
            html.Details(
                [
                    html.Summary(
                        f"Earlier issue vs later reported demand · {len(matched)} eligible matches"
                    ),
                    html.P(
                        "Actual − forecast, MW. Later actuals may revise; these matches are separate from the official ≥24h observation-lead scoring window. Original expected weather is not archived.",
                        className="gp-panel__hint",
                    ),
                    exact_table(
                        [
                            "UTC target",
                            "Earlier forecast · MW",
                            "Reported actual · MW",
                            "Actual − forecast · MW",
                        ],
                        [
                            [r["timestamp"], r["forecast_mw"], r["actual_mw"], r["error_mw"]]
                            for r in matched
                        ],
                    )
                    if matched
                    else html.P(
                        "No quality-eligible actuals overlap this issue in the retained comparison window."
                    ),
                ],
                className="gp-review-disclosure",
            )
        )
        return body

    @app.callback(
        [
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
            Output("explorer-state", "data", allow_duplicate=True),
            Output("conditions-section", "value", allow_duplicate=True),
            Output("explorer-mode", "value", allow_duplicate=True),
        ],
        [Input("forecast-open-explorer", "n_clicks"), Input("forecast-open-error", "n_clicks")],
        [
            State("explorer-state", "data"),
            State("explorer-cursor", "data"),
            State("region-selector", "value"),
            State("published-review-earlier", "value"),
        ],
        prevent_initial_call=True,
    )
    def open_explorer(_future, _error, raw, cursor, region, earlier):
        if not (_future or _error):
            return [no_update] * 4
        state = clean_state(raw, region)
        if (
            isinstance(cursor, dict)
            and cursor.get("region") == region
            and cursor.get("mode") == state["mode"]
        ):
            state["windows"][state["mode"]]["cursor"] = cursor.get("timestamp") or ""
        mode = "error" if ctx.triggered_id == "forecast-open-error" else "future"
        state["section"] = "explorer"
        state["mode"] = mode
        if mode == "error" and earlier:
            state["issue"] = earlier
        return "tab-alerts", state, "explorer", mode
