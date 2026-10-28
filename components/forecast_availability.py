"""Forecast presentation states; cache absence is never an active request."""

from dataclasses import dataclass

from components.accessibility import model_display_name
from data.forecast_identity import resolve_forecast_identity, selection_values


@dataclass(frozen=True)
class ForecastAvailability:
    state: str
    label: str
    message: str


def forecast_availability(
    payload, requested, hours, region, *, alive=False, completed=False, read_failed=False
):
    """Resolve the same selection for source, plot and empty statistics.

    Pending requests belong to Dash's loading state. Recent regional writes
    support warming; a completed pass supports the stronger unavailable copy.
    No forecast or source input is borrowed from another region.
    """
    if read_failed:
        return ForecastAvailability(
            "error",
            "Could not load forecast",
            "Forecast could not be loaded. Refresh the page to retry, or select another model.",
        )
    identity = resolve_forecast_identity(payload, requested)
    if identity and selection_values(payload, identity, hours) is not None:
        return ForecastAvailability("available", identity.label, identity.note)
    if isinstance(payload, dict) and payload.get("forecasts"):
        return ForecastAvailability(
            "unavailable_selection",
            "Not shown for this selection",
            f"No {model_display_name(requested)} forecast in the latest update for {region or 'this region'} "
            f"at the selected {hours}-hour horizon. Select another model or check again later.",
        )
    if payload is not None and not isinstance(payload, dict):
        return forecast_availability(None, requested, hours, region, read_failed=True)
    if alive and completed:
        return ForecastAvailability(
            "unavailable",
            "Forecast unavailable",
            f"Forecast unavailable for {region or 'this region'}. The latest completed data update "
            "did not publish a forecast for this region. Try another region or check again later.",
        )
    if alive:
        return ForecastAvailability(
            "warming",
            "Warming",
            "Recent regional inputs are available, but a completed forecast update is not yet recorded. "
            "Try another region or check again later.",
        )
    return ForecastAvailability(
        "empty",
        "No cached forecast available",
        "No cached forecast is available for this region. Try another region or check again later.",
    )
