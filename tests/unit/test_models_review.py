"""The Models review must preserve independent clocks, real counts and signed errors."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from components._callbacks_models import (
    _build_horizon_drift_panel,
    _diagnostic_data,
    _diagnostic_evidence,
    _diagnostic_figures,
    _training_table,
)
from tests.integration.dash_driver import component_text


def diagnostic_payload(n: int = 7) -> dict:
    """Unequal/signed values make count, zero and alignment errors observable."""
    import pandas as pd

    timestamps = pd.date_range("2026-02-01", periods=n, freq="h", tz="UTC")
    residuals = [float((i % 5) * 7 - 11) for i in range(n)]
    predicted = [100.0 + i for i in range(n)]
    return {
        "region": "ERCOT",
        "diagnostics_source": "backtest",
        "residual_source": {
            "kind": "walk_forward_backtest",
            "model": "xgboost",
            "horizon": 24,
            "exog_mode": "forecast_exog",
        },
        "timestamps": [t.isoformat() for t in timestamps],
        "predicted": predicted,
        "residuals": residuals,
        "actual": [p + r for p, r in zip(predicted, residuals, strict=True)],
        "feature_importance": {"names": ["lag", "weather"], "values": [0.8, 0.2]},
    }


def text(component: object) -> str:
    """Use Dash's own serialization, including nested component children."""
    import json

    from plotly.utils import PlotlyJSONEncoder

    return component_text(json.loads(json.dumps(component, cls=PlotlyJSONEncoder)))


def test_signed_values_counts_and_exact_histogram_are_from_payload() -> None:
    payload = diagnostic_payload(29)
    data = _diagnostic_data(payload)
    figures, scope = _diagnostic_figures(data, "ERCOT")
    assert len(data["rows"]) == 29
    assert sum(b["count"] for b in data["histogram"]) == 29
    assert sum(h["count"] for h in data["hourly"]) == 29
    assert data["hourly"][0]["count"] == 2
    assert data["hourly"][0]["mae"] == pytest.approx((11 + 17) / 2)
    assert list(figures[2].data[0].y) == payload["residuals"]
    assert figures[0].layout.yaxis.range[0] < 0 < figures[0].layout.yaxis.range[1]
    assert figures[2].layout.shapes[0].y0 == 0
    assert "29 finite" in text(scope)
    assert "actual − prediction" in text(scope)
    assert "climatology-exog" in text(scope)
    exact = text(_diagnostic_evidence(data, 1))
    assert "-11.0" in exact and "2026-02-01T00:00:00+00:00" in exact
    assert "NumPy" in exact and "final bin" in exact


def test_unknown_clock_is_not_silently_assigned_utc() -> None:
    payload = diagnostic_payload(3)
    payload["timestamps"] = [s.removesuffix("+00:00") for s in payload["timestamps"]]
    payload["diagnostics_source"] = "legacy_unknown"
    data = _diagnostic_data(payload)
    figures, scope = _diagnostic_figures(data, "ERCOT")
    assert all(r["utc"] is None for r in data["rows"])
    assert not data["hourly"]
    assert not figures[0].data
    assert figures[0].layout.meta["diagnostic_utc_qualified"] is False
    assert "UTC window unavailable" in text(scope)
    assert "offset missing" in text(_diagnostic_evidence(data, 1))
    assert sum(b["count"] for b in data["histogram"]) == 3


def test_bad_alignment_is_unavailable_not_truncated_or_zero_filled() -> None:
    payload = diagnostic_payload()
    payload["timestamps"] = payload["timestamps"][:-1]
    data = _diagnostic_data(payload)
    assert data["rows"] == [] and data["histogram"] == []
    assert data["importance"]  # independent cached XGBoost importance still valid
    figs, _ = _diagnostic_figures(data, "ERCOT")
    assert not figs[0].data and not figs[2].data
    assert figs[4].data
    payload = diagnostic_payload()
    payload["predicted"] = [100.0]
    data = _diagnostic_data(payload)
    assert len(data["rows"]) == 7 and all(r["predicted"] is None for r in data["rows"])
    assert not _diagnostic_figures(data, "ERCOT")[0][2].data


def test_nonfinite_and_zero_residuals_are_distinguished() -> None:
    payload = diagnostic_payload(4)
    payload["residuals"] = [0.0, float("nan"), float("inf"), -3.25]
    data = _diagnostic_data(payload)
    assert [r["residual"] for r in data["rows"]] == [0.0, -3.25]
    assert sum(b["count"] for b in data["histogram"]) == 2
    assert "-3.25" in text(_diagnostic_evidence(data, 1))


def test_long_residual_series_downsamples_plot_not_exact_values() -> None:
    data = _diagnostic_data(diagnostic_payload(801))
    figures, _ = _diagnostic_figures(data, "ERCOT")
    assert len(figures[0].data[0].y) == 720
    assert len(data["rows"]) == 801
    final_page = text(_diagnostic_evidence(data, 17))
    assert "801–801 of 801" in final_page
    assert data["rows"][-1]["time"] in final_page


def test_training_filter_is_cached_and_preserves_missing_metrics() -> None:
    payload = {"model_metrics": {"xgboost": {"mape": 7.123, "rmse": 21, "mae": None, "r2": -0.4}}}
    with patch("components._callbacks_models.redis_get", return_value=payload) as read:
        result = text(_training_table("ERCOT", ["xgboost"]))
        assert "XGBoost" in result and "7.12%" in result and "-0.4000" in result
        assert "Unavailable" in result and "Prophet" not in result
        read.assert_called_once()
    with patch("components._callbacks_models.redis_get", return_value=None) as read:
        assert "not available" in text(_training_table("ERCOT", ["prophet"]))
        read.assert_called_once()


def test_sampled_categories_are_structured_counts_not_hour_positions() -> None:
    block = {
        "n_7d": 12,
        "rolling_mape_7d": 7.125,
        "future_sample_hours": 9,
        "coverage_7d": {
            "possible_target_hours": 96,
            "scored_target_hours": 12,
            "quality_guard_excluded_hours": 3,
            "pending_actual_hours": 1,
        },
    }
    sampled = {
        "schema_version": 1,
        "sampling_basis": "hourly_observation_before_target",
        "models": {"xgboost": {"24h": block}},
    }

    def read(key: str) -> dict | None:
        return sampled if "drift_hourly" in key else None

    with patch("components._callbacks_models.redis_get", side_effect=read) as redis:
        result = text(_build_horizon_drift_panel("ERCOT"))
        assert "7.12%" in result and "12 / 96" in result and "future samples: 9" in result
        assert "Quality-guard excluded actuals: 3" in result
        assert "Missing samples: Unavailable" in result
        assert "not missing-hour locations" in result
        assert redis.call_count == 2
    block["n_7d"] = 5
    with patch("components._callbacks_models.redis_get", side_effect=read):
        result = text(_build_horizon_drift_panel("ERCOT"))
        assert "Collecting" in result and "7.12%" not in result


def test_missing_sampled_feed_does_not_substitute_legacy_offset() -> None:
    with patch("components._callbacks_models.redis_get", return_value=None) as read:
        result = text(_build_horizon_drift_panel("ERCOT"))
        assert "Hourly-sampled error is unavailable" in result
        assert "Legacy offsets" in result
        assert read.call_count == 2


def test_legacy_backtest_pipeline_clock_is_utc_with_explicit_basis() -> None:
    payload = diagnostic_payload(9)
    payload["timestamps"] = [stamp.removesuffix("+00:00") for stamp in payload["timestamps"]]
    data = _diagnostic_data(payload)
    figs, scope = _diagnostic_figures(data, "ERCOT")
    assert len(figs[0].data[0].y) == 9
    assert sum(r["count"] for r in data["hourly"]) == 9
    assert all(r["utc"].endswith("+00:00") and r["inferred_utc"] for r in data["rows"])
    assert data["rows"][0]["source_time"] == payload["timestamps"][0]
    assert "9 legacy offset-free timestamps" in text(scope)
    assert "Pipeline UTC; legacy offset omitted" in text(_diagnostic_evidence(data, 1))
