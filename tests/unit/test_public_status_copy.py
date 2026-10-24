"""Availability and attribution copy must match the state the caller knows."""

from unittest.mock import patch

from components._callbacks_us_grid import _build_us_grid_region_card
from components.error_handling import api_error_state, confidence_badge, error_state, warming_state


def test_api_failure_does_not_invent_a_cached_result():
    assert "Showing cached data" not in str(api_error_state("EIA"))
    assert "Showing cached data" in str(api_error_state("EIA", cached_available=True))


def test_missing_data_and_freshness_do_not_claim_recovery_or_accuracy():
    missing = str(warming_state())
    assert "No published data" in missing
    assert "Try another region" in missing
    assert "shortly" not in missing and "refreshing data" not in missing
    assert "does not assess data accuracy" in confidence_badge("Demand", "high").title
    assert "no demand data available" in _build_us_grid_region_card("FPL", {}).title


def test_generic_error_keeps_details_in_logs():
    with patch("components.error_handling.log.warning") as logger:
        visible = str(error_state(error_detail="TRACE_ONLY_MARKER"))
    assert "TRACE_ONLY_MARKER" not in visible
    logger.assert_called_once_with("error_state_detail", detail="TRACE_ONLY_MARKER")
