"""Visible forecast identity and evidence, matched to the displayed series and lead."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any

from data.forecast_evidence import baseline_selection, forecast_timing
from data.forecast_identity import resolve_forecast_identity, selection_values
from data.redis_client import redis_get, redis_key


def forecast_review_context(
    region: str, payload: dict | None, requested: str, hours: int
) -> dict[str, Any]:
    identity = resolve_forecast_identity(payload, requested)
    context = {
        "display_label": "Unavailable",
        "display_status": "Unavailable",
        "published_label": "Unavailable",
        "source_note": "No published values are available for this selection.",
        "error_label": "Observed error",
        "error_value": "Not available",
        "error_note": "No error score is attributed to an unavailable forecast.",
        "holdout_label": "Training reference",
        "holdout_value": "Not available",
        "holdout_note": "No training result is attributed to an unavailable forecast.",
        "coverage_summary": "Scoring coverage unavailable",
        "coverage_note": "The scored target hours and measurement window are not available.",
        "sampling_note": "",
        "coverage_scored": None,
        "coverage_possible": None,
        "display_available": False,
    }
    timing = forecast_timing(payload, hours)

    def clock(value):
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M") if value else "not recorded"

    context["timing_summary"] = (
        f"Targets: {clock(timing['forecast_start_at'])} to {clock(timing['target_end_at'])} UTC. "
        f"Generated: {clock(timing['generated_at'])} UTC."
    )
    context["timing_note"] = (
        f"Targets (UTC): {timing['forecast_start_at'] or 'unavailable'} to "
        f"{timing['target_end_at'] or 'unavailable'}. "
        f"Generated (UTC): {timing['generated_at'] or 'not recorded'}. "
        f"Input anchor (UTC): {timing['input_anchor_at'] or 'not recorded'}. "
        "The input anchor is not a guarantee of finalized metered demand. "
        "This is a rolling target window, not a calendar operating day."
    )
    selection = baseline_selection(payload)
    context["baseline_note"] = (
        f"Published baseline selection — recorded policy reason: {selection['reason'] or 'not recorded'}. "
        f"Comparison series: {selection['compared_series'] or 'not recorded'}. "
        "When this selection first began is not recorded. "
        + (
            "The policy compares unpaired target sets; this is not a paired benchmark."
            if selection.get("comparison_basis") == "unpaired_trailing_window"
            else ""
        )
        if selection
        else ""
    )
    if identity is None:
        return context
    context.update(
        display_label=identity.label,
        display_status=(
            "GridPulse published forecast"
            if identity.status == "Published default"
            else "GridPulse model comparison"
        ),
        published_label=identity.published_label,
        source_note=identity.note,
    )
    if selection_values(payload, identity, hours) is None:
        context.update(
            display_status="Not shown at this horizon",
            source_note=f"{identity.label} is unavailable or withheld for the selected {hours}-hour window.",
        )
        return context
    context["display_available"] = True
    key = identity.metric_key
    context.update(
        error_label=f"Observed error · {hours}h horizon · 7d MAPE",
        error_note=f"No matching observed error is available for {identity.label} at this forecast horizon.",
    )
    if hours in (24, 48, 72) and key:
        drift = redis_get(redis_key(f"drift_horizon:{region}"))
        models = drift.get("models") if isinstance(drift, dict) else None
        series = models.get(key) if isinstance(models, dict) else None
        block = series.get(f"{hours}h") if isinstance(series, dict) else None
        if isinstance(block, dict):
            n = block.get("n_7d")
            value = block.get("rolling_mape_7d")
            coverage = block.get("coverage_7d")
            if isinstance(coverage, dict):
                count = coverage.get("scored_target_hours")
                possible = coverage.get("possible_target_hours")
                if type(count) is int and type(possible) is int and 0 <= count <= possible:
                    context["coverage_scored"] = count
                    context["coverage_possible"] = possible
                    context["coverage_summary"] = f"{count} / {possible} hourly targets scored"
                    context["coverage_note"] = (
                        f"Scored targets: {coverage.get('scored_start') or 'none'} to "
                        f"{coverage.get('scored_end') or 'none'}. "
                        f"Window evaluated at {coverage.get('as_of') or 'unknown time'}. "
                        f"Score uses {n} observations; {coverage.get('duplicate_records', 'unknown')} "
                        "duplicate target records. "
                        "The denominator is a full seven-day hourly window, not a claim that "
                        "this series existed for every hour. Unscored hours may have several causes; "
                        f"{coverage.get('low_actual_excluded', 'unknown')} low-actual records excluded."
                        f" Diagnostic events: repeated-origin skips {block.get('n_dedup_skipped_7d', 'not recorded')}, "
                        f"expired unresolved {block.get('n_expired_unresolved_7d', 'not recorded')}, "
                        f"malformed pending {block.get('n_malformed_7d', 'not recorded')}. "
                        "These event counts are not an additive explanation of every unscored hour."
                    )
            elif isinstance(n, (int, float)) and isfinite(n) and n >= 0:
                context["coverage_summary"] = (
                    f"{int(n)} scored observations · completeness not reported"
                )
                context["coverage_note"] = "This legacy score does not report target-hour coverage."
            if (
                isinstance(n, (int, float))
                and isfinite(n)
                and n >= 6
                and isinstance(value, (int, float))
                and isfinite(value)
                and value >= 0
            ):
                context.update(
                    error_value=f"{value:.2f}%",
                    error_note=f"{identity.label}, measured {hours} hours after the first forecast target. "
                    "This is an offset from forecast start, not an exact publication-to-target lead. "
                    "MAPE means mean absolute percentage error; lower is better. "
                    "Historical error does not guarantee this forecast's accuracy.",
                )
            elif isinstance(n, (int, float)) and isfinite(n) and n < 6:
                context["error_note"] = (
                    f"{identity.label} has {int(n)} scored observations at {hours}h ahead; at least 6 are required to display the seven-day error."
                )
    elif hours not in (24, 48, 72):
        context["error_note"] = (
            f"No matching live score is published for the {hours}-hour lead. Scores from shorter leads are not substituted here."
        )
    if key and hours in (24, 48, 72):
        sampled = redis_get(redis_key(f"drift_hourly:{region}:summary"))
        if (
            isinstance(sampled, dict)
            and sampled.get("schema_version") == 1
            and sampled.get("sampling_basis") == "hourly_observation_before_target"
        ):
            new_block = (sampled.get("models", {}).get(key) or {}).get(f"{hours}h") or {}
            n = new_block.get("n_7d", 0)
            value = new_block.get("rolling_mape_7d")
            context["sampling_note"] = (
                f"Hourly sampling: first {hours}h target {clock(new_block.get('first_target_at'))} UTC. "
                f"Scored observations: {n}; future samples: {new_block.get('future_sample_hours', 0)}."
            )
            if (
                isinstance(n, int)
                and n >= 6
                and isinstance(value, (int, float))
                and isfinite(value)
                and value >= 0
            ):
                coverage = new_block.get("coverage_7d") or {}
                context.update(
                    error_label=f"Observed error · sampled ≥{hours}h before target · 7d MAPE",
                    error_value=f"{value:.2f}%",
                    error_note=(
                        f"{identity.label} as observed by the hourly job, at least {hours} "
                        f"and less than {hours + 1} hours before each target. This is the sampling "
                        "lead, not the original publication lead. MAPE is mean absolute percentage "
                        "error; lower is better. Historical error does not guarantee this forecast's accuracy."
                    ),
                    coverage_scored=n if 0 <= n <= 168 else None,
                    coverage_possible=168 if 0 <= n <= 168 else None,
                    coverage_summary=f"{n} / 168 hourly targets scored",
                    coverage_note=(
                        f"Window evaluated at {coverage.get('as_of') or 'unknown time'}. "
                        f"Of 168 target hours: {n} scored, "
                        f"{coverage.get('not_yet_tracked_hours')} before collection, "
                        f"{coverage.get('missing_sample_hours')} missing samples, "
                        f"{coverage.get('series_unavailable_hours')} without this series, "
                        f"{coverage.get('pending_actual_hours')} waiting for actual demand, "
                        f"{coverage.get('low_actual_excluded_hours')} low-actual exclusions. "
                        "These categories partition the window. Actuals may still be revised. "
                        "Legacy forecast-start-offset records are not included."
                    ),
                )
            else:
                context["error_label"] = f"Observed error · legacy {hours}h offset · 7d MAPE"
                context["sampling_note"] += (
                    " The score above remains the legacy forecast-start-offset diagnostic "
                    "until six new observations are scored; the two samples are not combined."
                    if context["error_value"] != "Not available"
                    else " No hourly error average is shown until six observations are scored; legacy histories stay separate."
                )
    if key == "baseline":
        context.update(
            holdout_value="Not applicable",
            holdout_note="This baseline has no trained-model holdout. Other models' scores do not describe it.",
        )
    elif key:
        # Read the scoring snapshot's recorded holdout only. The general
        # model service can fall back to simulated diagnostics in dev/cold states.
        published_metrics = payload.get("model_metrics")
        metrics = published_metrics.get(key) if isinstance(published_metrics, dict) else None
        metrics = metrics if isinstance(metrics, dict) else {}
        value = metrics.get("mape")
        context["holdout_label"] = f"Training reference · {identity.label}"
        context["holdout_note"] = (
            "MAPE means mean absolute percentage error; lower is better. Last training evaluation: 168-hour recursive holdout across multiple leads. This is not live error or expected accuracy for the displayed curve."
        )
        if isinstance(value, (int, float)) and isfinite(value) and value >= 0:
            context["holdout_value"] = f"{value:.2f}% MAPE"
    return context
