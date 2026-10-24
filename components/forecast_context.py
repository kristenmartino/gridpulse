"""Visible forecast identity and evidence, matched to the displayed series and lead."""

from __future__ import annotations

from math import isfinite

from data.forecast_identity import resolve_forecast_identity, selection_values
from data.redis_client import redis_get, redis_key


def forecast_review_context(
    region: str, payload: dict | None, requested: str, hours: int
) -> dict[str, str]:
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
    }
    if identity is None:
        return context
    context.update(
        display_label=identity.label,
        display_status=identity.status,
        published_label=identity.published_label,
        source_note=identity.note,
    )
    if selection_values(payload, identity, hours) is None:
        context.update(
            display_status="Not shown at this horizon",
            source_note=f"{identity.label} is unavailable or withheld for the selected {hours}-hour window.",
        )
        return context
    key = identity.metric_key
    context.update(
        error_label=f"Observed error · {hours}h ahead · 7d MAPE",
        error_note=f"No matching live error is available for {identity.label} at {hours}h ahead. This chart has no single validated accuracy score.",
    )
    if hours in (24, 48, 72) and key:
        drift = redis_get(redis_key(f"drift_horizon:{region}"))
        models = drift.get("models") if isinstance(drift, dict) else None
        series = models.get(key) if isinstance(models, dict) else None
        block = series.get(f"{hours}h") if isinstance(series, dict) else None
        if isinstance(block, dict):
            n = block.get("n_7d", block.get("n_records", 0))
            value = block.get("rolling_mape_7d")
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
                    error_note=f"{identity.label} forecasts issued {hours}h before their target hour; {int(n)} scored observations in the last 7 days. MAPE means mean absolute percentage error; lower is better. This is not a guarantee for this forecast.",
                )
            elif isinstance(n, (int, float)) and isfinite(n) and n < 6:
                context["error_note"] = (
                    f"{identity.label} has {int(n)} scored observations at {hours}h ahead; at least 6 are required to display the seven-day error."
                )
    elif hours not in (24, 48, 72):
        context["error_note"] = (
            f"No matching live score is published for the {hours}-hour lead. Scores from shorter leads are not substituted here."
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
