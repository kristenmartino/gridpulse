"""Visible source and measurement context for a displayed demand forecast."""

from dash import html


def build_forecast_review(context: dict) -> html.Div:
    """Render resolved forecast identity without deriving or substituting metrics."""
    context = context or {}

    def value(key: str, default: str) -> str:
        return str(context.get(key) or default)

    return html.Div(
        [
            html.Div(
                [
                    html.Section(
                        [
                            html.H3("Forecast shown", className="gp-forecast-review__label"),
                            html.P(
                                value("display_label", "Unavailable"),
                                className="gp-forecast-review__value",
                            ),
                            html.P(
                                value("display_status", "Unavailable"),
                                className="gp-forecast-review__status",
                            ),
                            html.P(
                                [
                                    "GridPulse published forecast: ",
                                    html.Strong(value("published_label", "Unavailable")),
                                ],
                                className="gp-forecast-review__note",
                            )
                            if context.get("display_label") != context.get("published_label")
                            else None,
                        ],
                        className="gp-forecast-review__cell",
                    ),
                    html.Section(
                        [
                            html.H3(
                                value("error_label", "Observed error"),
                                className="gp-forecast-review__label",
                            ),
                            html.P(
                                value("error_value", "Not available"),
                                className="gp-forecast-review__value tabular",
                            ),
                            html.P(
                                value("coverage_summary", "Scoring coverage unavailable"),
                                className="gp-forecast-review__status",
                            ),
                            html.P(
                                value(
                                    "error_note",
                                    "No matching observed error is available for the forecast shown.",
                                ),
                                className="gp-forecast-review__note",
                            ),
                        ],
                        className="gp-forecast-review__cell",
                    ),
                    html.Section(
                        [
                            html.H3(
                                value("holdout_label", "Training reference"),
                                className="gp-forecast-review__label",
                            ),
                            html.P(
                                value("holdout_value", "Not available"),
                                className="gp-forecast-review__value tabular",
                            ),
                            html.P(
                                value(
                                    "holdout_note",
                                    "No matching training evaluation is available for the forecast shown.",
                                ),
                                className="gp-forecast-review__note",
                            ),
                        ],
                        className="gp-forecast-review__cell",
                    ),
                ],
                className="gp-forecast-review__grid",
            ),
            html.P(
                value("source_note", "Forecast source details are not available."),
                className="gp-forecast-review__source-note",
            ),
            *(
                [html.P(context["baseline_note"], className="gp-forecast-review__source-note")]
                if context.get("baseline_note")
                else []
            ),
            html.P(
                value("timing_summary", "Target window and generation time unavailable."),
                className="gp-forecast-review__source-note",
            ),
            html.Details(
                [
                    html.Summary("Timing and scoring coverage"),
                    html.P(value("timing_note", "Timing metadata unavailable.")),
                    html.P(value("coverage_note", "Coverage metadata unavailable.")),
                ],
                className="gp-forecast-review__source-note",
            ),
        ],
        className="gp-forecast-review",
        role="region",
        **{"aria-label": "Forecast source and error context"},
    )
