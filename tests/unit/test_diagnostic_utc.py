"""Known UTC fold clocks must keep their annotation when serialized to Redis."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jobs import phases


def _data():
    stamps = pd.date_range("2026-10-20", periods=48, freq="h", tz="UTC")
    return phases.RegionData(
        "FPL",
        pd.DataFrame({"timestamp": stamps, "demand_mw": 10000.0}),
        pd.DataFrame({"timestamp": stamps, "temperature_2m": 75.0}),
    )


@pytest.mark.parametrize("offset", [None, "US/Eastern"])
def test_backtest_writer_preserves_utc_instants_without_computing_models(offset):
    stamps = pd.date_range("2026-10-20", periods=24, freq="h", tz="UTC")
    # Production fold code uses .values, which produces UTC NumPy clocks with
    # no annotation. An aware input exercises conversion, not relabeling.
    source = stamps.values if offset is None else stamps.tz_convert(offset)
    cached = {}
    backtest = {
        "metrics": {"mape": 1.0, "rmse": 10.0, "mae": 10.0, "r2": 0.9},
        "actual": np.full(24, 1000.0),
        "predictions": np.full(24, 990.0),
        "timestamps": source,
    }
    with (
        patch("data.redis_client.redis_get", return_value=None),
        patch("components.callbacks._run_backtest_for_horizon", return_value=backtest) as run,
        patch(
            "data.redis_client.redis_set", side_effect=lambda k, v, **kw: cached.update({k: v})
        ) as write,
    ):
        result = phases.write_backtests(_data())
    assert result.ok
    assert run.call_count == write.call_count == len(phases.BACKTEST_HORIZONS)
    for payload in cached.values():
        assert payload["timestamps"] == [t.isoformat() for t in stamps]
        assert payload["residuals"] == [10.0] * 24


def test_diagnostic_writer_restores_legacy_backtest_utc_before_hourly_grouping():
    stamps = pd.date_range("2026-10-20", periods=24, freq="h")
    payload = {
        "actual": [1000.0] * 24,
        "predictions": {"xgboost": [990.0] * 24},
        "timestamps": [t.isoformat() for t in stamps],
        "metrics": {},
    }
    captured = {}
    with (
        patch("data.redis_client.redis_get", return_value=payload),
        patch(
            "data.redis_client.redis_set", side_effect=lambda k, v, **kw: captured.update(v)
        ) as write,
        patch(
            "models.model_service.get_forecasts", side_effect=AssertionError("No model inference")
        ),
    ):
        result = phases.write_diagnostics(_data(), None)
    assert result.ok
    write.assert_called_once()
    assert captured["timestamps"] == [t.tz_localize("UTC").isoformat() for t in stamps]
    assert captured["hourly_error"]["hours"] == list(range(24))
    assert captured["hourly_error"]["values"] == [10.0] * 24
