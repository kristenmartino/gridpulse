"""Overview-only presentation of the shared, source-matched review context.

No data fetching or metric selection lives here. All surfaces consume the same
published payload and the context resolved by the main Overview callback.
"""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

from dash import html

from data.forecast_identity import resolve_forecast_identity, selection_values


def _utc_clock(value: object, fmt: str) -> str | None:
    """Format one recorded ISO timestamp without per-row dataframe parsing."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    stamp = stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)
    return stamp.strftime(fmt)


def _cell(label: str, value: str | list, note: str | list, *, source: bool = False) -> html.Section:
    return html.Section(
        [
            html.P(label, className="gp-overview__metric-label"),
            html.P(
                value,
                className="gp-overview__metric-value"
                + (" gp-overview__metric-value--source" if source else ""),
            ),
            html.P(note, className="gp-overview__metric-note"),
        ],
        className="gp-overview__metric",
    )


def build_review_summary(
    context: dict, cached: dict | None, *, latest: dict | None = None
) -> html.Div:
    """Next peak, published identity and matching historical error, never a substitute."""
    peak, when = "Unavailable", "Awaiting the next scoring update."
    identity = resolve_forecast_identity(cached)
    selected = selection_values(cached, identity, 24) if identity else None
    if selected is not None:
        timestamps, values = selected
        idx = max(range(len(values)), key=values.__getitem__)
        peak = [f"{values[idx]:,.0f}", html.Small(" MW")]
        when = _utc_clock(timestamps[idx], "%a %d %b, %H:%M UTC") or "Target time unavailable"
    peak_note = [when]
    if latest is not None:
        peak_note.extend([html.Br(), f"Latest reported demand: {latest.get('value', '—')} MW"])
    return html.Div(
        [
            html.Div(
                [
                    _cell("Next 24h peak · published forecast", peak, peak_note),
                    _cell(
                        "Forecast shown",
                        context["display_label"],
                        context["display_status"],
                        source=True,
                    ),
                    _cell(
                        context["error_label"],
                        context["error_value"],
                        context["coverage_summary"],
                    ),
                ],
                className="gp-overview__metrics",
            ),
            html.P(context["timing_summary"], className="gp-overview__timing"),
        ]
    )


def build_review_sidebar(context: dict) -> html.Div:
    """Aggregate coverage is neither a time mask nor an accuracy/risk grade."""
    scored, possible = context.get("coverage_scored"), context.get("coverage_possible")
    known = (
        type(scored) is int and type(possible) is int and 0 <= scored <= possible and possible > 0
    )
    coverage = [
        html.H3("Aggregate scoring coverage"),
        html.P(context["coverage_summary"], className="gp-overview__coverage-value"),
    ]
    if known:
        coverage.extend(
            [
                html.Div(
                    [
                        html.Span(
                            style={"width": f"{100 * scored / possible:.6f}%"},
                            className="gp-overview__coverage-scored",
                        ),
                        html.Span(className="gp-overview__coverage-unscored"),
                    ],
                    className="gp-overview__coverage-bar",
                    role="img",
                    **{
                        "aria-label": f"{scored} of {possible} hourly targets scored; {possible - scored} not scored. Aggregate count, not a timeline."
                    },
                ),
                html.Div(
                    [html.Span(f"{scored} scored"), html.Span(f"{possible - scored} not scored")],
                    className="gp-overview__coverage-legend",
                ),
                html.P(
                    "Counts describe historical scoring coverage, not when missing targets occurred.",
                    className="gp-overview__note",
                ),
            ]
        )
    else:
        coverage.append(
            html.P(
                "Target-hour completeness is not reported. No coverage fraction is inferred.",
                className="gp-overview__note",
            )
        )
    return html.Div(
        [
            html.Section(
                [
                    *coverage,
                    html.Div(
                        [
                            html.H3(context["holdout_label"]),
                            html.P(
                                context["holdout_value"], className="gp-overview__training-value"
                            ),
                            html.P(context["holdout_note"], className="gp-overview__note"),
                        ],
                        className="gp-overview__training",
                    ),
                ],
                className="gp-overview__rail-card",
            ),
            html.Section(
                [
                    html.H3("Error measurement context"),
                    html.P(context["error_note"], className="gp-overview__note"),
                ],
                className="gp-overview__rail-card gp-overview__rail-card--timing",
            ),
        ],
        className="gp-overview__rail",
    )


def _target_table(context: dict, cached: dict | None) -> html.Div | html.P:
    identity = resolve_forecast_identity(cached)
    if identity is None:
        return html.P(
            "Published forecast targets are unavailable. Awaiting the next scoring update."
        )
    guard_map = cached.get("horizon_guard")
    guard = guard_map.get(identity.metric_key) if isinstance(guard_map, dict) else None
    max_ok = guard.get("max_ok_horizon") if isinstance(guard, dict) else None
    if isinstance(max_ok, (int, float)) and isfinite(max_ok) and max_ok < 24:
        return html.P("Published forecast targets are withheld at this horizon.")
    rows = []
    for row in (cached.get("forecasts") or [])[:24]:
        row = row if isinstance(row, dict) else {}
        clock = _utc_clock(row.get("timestamp"), "%d %b %Y, %H:%M UTC") or "Timestamp unavailable"
        raw = row.get(identity.column)
        try:
            number = float(raw)
            value = f"{number:,.0f}" if isfinite(number) else "Unavailable"
        except (TypeError, ValueError, OverflowError):
            value = "Unavailable"
        rows.append(html.Tr([html.Td(clock), html.Td(value), html.Td(identity.label)]))
    return html.Div(
        [
            html.P(
                f"{len(rows)} stored published forecast targets (up to 24h), not scored observations. "
                f"Published series: {identity.label}. The chart's historical continuity point is not a forecast target. Missing values remain unavailable."
                + (
                    " The published curve is unavailable or withheld for this selection."
                    if not context.get("display_available")
                    else ""
                ),
                className="gp-overview__note",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Target hour · UTC", scope="col"),
                                    html.Th("Forecast · MW", scope="col"),
                                    html.Th("Published series", scope="col"),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="gp-overview__target-table",
                ),
                className="gp-overview__table-scroll",
                tabIndex=0,
                role="region",
                **{"aria-label": "Published hourly forecast target values"},
            ),
        ]
    )


def build_review_evidence(context: dict, cached: dict | None) -> html.Div:
    """Expose score metadata and target rows through native keyboard disclosures."""
    summary = html.Div(
        [
            html.H3(context["error_label"]),
            html.P([html.Strong(context["error_value"]), " · ", context["coverage_summary"]]),
            html.P(context["error_note"]),
            html.P(context["coverage_note"]),
            html.P(context["sampling_note"]) if context.get("sampling_note") else None,
            html.P(
                "Matched observation rows are not included here. The target table contains forecasts, not per-hour observed errors."
            ),
        ],
        className="gp-overview__evidence-copy",
    )
    return html.Div(
        [
            html.Details(
                [html.Summary("Scoring summary"), summary],
                className="gp-overview__evidence-section",
            ),
            html.Details(
                [
                    html.Summary("Forecast targets"),
                    html.Div(_target_table(context, cached), className="gp-overview__targets"),
                ],
                className="gp-overview__evidence-section",
            ),
        ],
        className="gp-overview__evidence-sections",
    )


def build_review_definitions(context: dict) -> html.Div:
    def section(title: str, keys: list[str]) -> html.Details:
        return html.Details(
            [
                html.Summary(title),
                html.Div(
                    [html.P(context[key]) for key in keys if context.get(key)],
                    className="gp-overview__details-copy",
                ),
            ]
        )

    return html.Div(
        [
            section("Forecast source & selection", ["source_note", "baseline_note"]),
            section("Timing & scoring definitions", ["timing_note", "error_note", "sampling_note"]),
        ],
        className="gp-overview__definitions",
    )
