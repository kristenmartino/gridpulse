"""Regression tests for the Overview hero chart + insight Redis-forecast fix.

A bug discovered 2026-05-20: ``_build_overview_hero_chart`` and
``_build_overview_insight`` were calling ``models.model_service.get_forecasts``
inline. In the web tier that always falls back to ``_simulate_forecasts``
(no trained pickles on the web container's disk), which returns
``actual * (1 + noise)`` — an array of length ``len(actual)`` representing
noisy *historical* predictions, not forward forecasts. The hero chart
then plotted ``ensemble[:24]`` (first 24 hours of HISTORICAL data) at
the *next 24 hours of timestamps*, producing a visibly wrong forecast
trace. The insight summary computed peak timestamp the same way and
emitted nonsense like "peaks at 04:00."

The fix routes both surfaces to ``gridpulse:forecast:{region}:1h`` (same
key the Forecast tab already reads). These tests pin the fix:

- The new helper ``_read_ensemble_forecast_from_redis`` parses the
  payload correctly and returns ``None`` on cold cache / malformed input
- The hero chart includes a forecast trace when Redis is warm with the
  correct timestamps + values from the payload
- The hero chart omits the forecast trace (renders actual-only) when
  Redis is cold — does NOT fall back to simulated
- The insight summary's forecast clause uses the REAL timestamp from
  the Redis payload (not a computed offset off last_actual)
- The insight summary drops the forecast clause when Redis is cold —
  does NOT fabricate one from simulated
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _demand_df(hours: int = 168) -> pd.DataFrame:
    """Build a synthetic actuals dataframe ending at 2026-05-20 05:00 UTC."""
    end = pd.Timestamp("2026-05-20 05:00:00", tz="UTC")
    ts = pd.date_range(end=end, periods=hours, freq="h")
    # Daily cycle 15k–22k MW (Florida-ish shape, May)
    hours_in = np.arange(hours)
    demand = 18500 + 3500 * np.sin(2 * np.pi * (hours_in - 18) / 24)
    return pd.DataFrame({"timestamp": ts, "demand_mw": demand})


def _redis_forecast_payload(
    *, n_rows: int = 24, start_ts: str = "2026-05-20T06:00:00+00:00"
) -> dict:
    """Build a realistic gridpulse:forecast:{region}:1h payload.

    Mirrors the shape ``jobs.phases.predict_and_write_forecast`` writes:
    list of rows each with timestamp + predicted_demand_mw + per-model
    keys + (optionally) ensemble + ensemble_weights at the top level.
    """
    start = pd.Timestamp(start_ts)
    timestamps = pd.date_range(start=start, periods=n_rows, freq="h")
    hours_in = np.arange(n_rows)
    # Florida-style daily cycle, peak around 19:00 UTC = 15:00 EDT.
    # start_ts = 06:00 UTC → peak at hours_in=13 → 19:00 UTC.
    # sin((i-7)*2π/24) is 1 when i-7 = 6, i.e. i = 13. ✓
    ensemble_pred = 17500 + 5000 * np.sin(2 * np.pi * (hours_in - 7) / 24)
    forecasts = []
    for i, ts in enumerate(timestamps):
        forecasts.append(
            {
                "timestamp": ts.isoformat(),
                "predicted_demand_mw": float(ensemble_pred[i]),
                "xgboost": float(ensemble_pred[i] * 1.005),
                "prophet": float(ensemble_pred[i] * 0.995),
                "arima": float(ensemble_pred[i] * 1.010),
                "ensemble": float(ensemble_pred[i]),
            }
        )
    return {
        "region": "FPL",
        "scored_at": "2026-05-20T09:02:00+00:00",
        "granularity": "1h",
        "primary_model": "xgboost",
        "forecasts": forecasts,
        "ensemble_weights": {"xgboost": 0.58, "prophet": 0.29, "arima": 0.13},
    }


class TestReadEnsembleForecastFromRedis:
    """The new helper ``_read_ensemble_forecast_from_redis``."""

    @patch("components._callbacks_overview.redis_get")
    def test_returns_none_when_cache_miss(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        mock_redis_get.return_value = None
        assert _read_ensemble_forecast_from_redis("FPL") is None

    @patch("components._callbacks_overview.redis_get")
    def test_returns_none_when_forecasts_empty(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        mock_redis_get.return_value = {"region": "FPL", "forecasts": []}
        assert _read_ensemble_forecast_from_redis("FPL") is None

    @patch("components._callbacks_overview.redis_get")
    def test_returns_none_when_payload_is_not_dict(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        mock_redis_get.return_value = "not a dict"  # malformed
        assert _read_ensemble_forecast_from_redis("FPL") is None

    @patch("components._callbacks_overview.redis_get")
    def test_returns_timestamps_predictions_scored_at_when_warm(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        mock_redis_get.return_value = _redis_forecast_payload()
        result = _read_ensemble_forecast_from_redis("FPL")
        assert result is not None
        timestamps, predictions, scored_at = result
        assert len(timestamps) == 24
        assert len(predictions) == 24
        assert isinstance(predictions, np.ndarray)
        # First timestamp matches the payload
        assert timestamps[0] == pd.Timestamp("2026-05-20T06:00:00+00:00")
        # scored_at threaded through
        assert scored_at == "2026-05-20T09:02:00+00:00"

    @patch("components._callbacks_overview.redis_get")
    def test_falls_back_to_predicted_demand_mw_when_ensemble_missing(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        # Build a payload where rows don't have ensemble (e.g. only XGBoost
        # was loadable during scoring). predicted_demand_mw fallback should
        # kick in.
        payload = _redis_forecast_payload()
        for row in payload["forecasts"]:
            del row["ensemble"]
        mock_redis_get.return_value = payload
        result = _read_ensemble_forecast_from_redis("FPL")
        assert result is not None
        _ts, predictions, _scored = result
        # Should equal predicted_demand_mw column (which equals the
        # synthetic ensemble in our fixture, but the resolver chose the
        # fallback path)
        assert predictions[0] == pytest.approx(payload["forecasts"][0]["predicted_demand_mw"])

    @patch("components._callbacks_overview.redis_get")
    def test_horizon_guard_total_collapse_withholds_hero_series(self, mock_redis_get):
        """#296: when the served series is flagged even at 24h
        (max_ok_horizon < 24 — total collapse), the hero must fall back to
        the actual-only/warming render (None), not draw the degenerate
        line. Flags at longer horizons don't affect the 24h hero bridge."""
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        payload = _redis_forecast_payload()
        payload["horizon_guard"] = {
            "ensemble": {"max_ok_horizon": 0, "flagged_horizon": 24, "reason": "non_finite"}
        }
        mock_redis_get.return_value = payload
        assert _read_ensemble_forecast_from_redis("FPL") is None

        # Flagged only past 24h -> hero unaffected.
        payload = _redis_forecast_payload()
        payload["horizon_guard"] = {
            "ensemble": {
                "max_ok_horizon": 168,
                "flagged_horizon": 720,
                "reason": "below_recent_band",
            }
        }
        mock_redis_get.return_value = payload
        assert _read_ensemble_forecast_from_redis("FPL") is not None

        # Guard on the primary applies when rows lack the ensemble column.
        payload = _redis_forecast_payload()
        for row in payload["forecasts"]:
            del row["ensemble"]
        payload["horizon_guard"] = {
            "xgboost": {"max_ok_horizon": 0, "flagged_horizon": 24, "reason": "non_finite"}
        }
        mock_redis_get.return_value = payload
        assert _read_ensemble_forecast_from_redis("FPL") is None

        # Malformed guard fails open.
        payload = _redis_forecast_payload()
        payload["horizon_guard"] = {"ensemble": "corrupt"}
        mock_redis_get.return_value = payload
        assert _read_ensemble_forecast_from_redis("FPL") is not None

    @patch("components._callbacks_overview.redis_get")
    def test_reads_correct_redis_key(self, mock_redis_get):
        from components._callbacks_overview import _read_ensemble_forecast_from_redis

        mock_redis_get.return_value = None
        _read_ensemble_forecast_from_redis("ERCOT")
        # Key composition: gridpulse:forecast:{region}:1h
        called_key = mock_redis_get.call_args[0][0]
        assert called_key == "gridpulse:forecast:ERCOT:1h"


class TestHeroChartReadsRedis:
    """``_build_overview_hero_chart`` no longer calls get_forecasts inline."""

    @patch("components._callbacks_overview.redis_get")
    def test_includes_forecast_trace_when_redis_warm(self, mock_redis_get):
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        fig = _build_overview_hero_chart("FPL", _demand_df())

        trace_names = [t.name for t in fig.data]
        assert "Forecast (24h)" in trace_names

        # The forecast trace's x-axis values match the timestamps in the
        # Redis payload — NOT a computed offset. First forecast point
        # after the bridge segment is 2026-05-20 06:00 UTC.
        forecast_trace = next(t for t in fig.data if t.name == "Forecast (24h)")
        # bridge_x = [last_actual_ts, *forecast_timestamps]
        # last_actual_ts = 2026-05-20 05:00; forecast_ts[0] = 2026-05-20 06:00
        assert pd.Timestamp(str(forecast_trace.x[1])) == pd.Timestamp("2026-05-20T06:00:00+00:00")

    @patch("components._callbacks_overview.redis_get")
    def test_omits_forecast_trace_when_redis_cold(self, mock_redis_get):
        """Cold cache → actual-only chart. Does NOT fall back to simulated."""
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = None
        fig = _build_overview_hero_chart("FPL", _demand_df())

        trace_names = [t.name for t in fig.data]
        assert "Forecast (24h)" not in trace_names
        # Actuals still render
        assert any("Actual" in str(t.name) or t.name is None for t in fig.data)

    @patch("components._callbacks_overview.redis_get")
    def test_does_not_call_get_forecasts(self, mock_redis_get):
        """Sanity: the new path doesn't accidentally also trigger the
        old inline compute via model_service.get_forecasts. This pins
        the architectural fix — web tier reads only."""
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        with patch("models.model_service.get_forecasts") as mock_get_forecasts:
            _build_overview_hero_chart("FPL", _demand_df())
            mock_get_forecasts.assert_not_called()

    @patch("components._callbacks_overview.redis_get")
    def test_confidence_band_renders_around_real_ensemble(self, mock_redis_get):
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        fig = _build_overview_hero_chart("FPL", _demand_df())

        # The confidence band is drawn as a filled scatter — the ±3 %
        # heuristic band should bracket the forecast ensemble values.
        band_traces = [t for t in fig.data if t.fill == "toself"]
        assert len(band_traces) >= 1  # at least the actual area + the band

        # Find the band whose y-range tracks the ensemble (the actual
        # area uses fill="tozeroy", not "toself" — band is the toself one).
        band = band_traces[0]
        # Band y is [upper, ...upper, ...lower reversed], so the max
        # should be > the max of the ensemble (because upper = ensemble * 1.03)
        band_y = np.array([v for v in band.y if v is not None], dtype=float)
        # Sample sanity check — the band's max should be > 22k (since
        # ensemble peaks at ~22.5k in the fixture and 1.03× = 23.2k)
        assert band_y.max() > 22_000


# ────────────────────────────────────────────────────────────────────────
# PR-B (2026-05-20) — empirical confidence interval on Overview hero chart
# ────────────────────────────────────────────────────────────────────────


class TestHeroChartEmpiricalConfidenceInterval:
    """``_build_overview_hero_chart`` uses empirical residual quantiles when
    enough backtest residuals exist; falls back to ±3 % heuristic otherwise.

    PR-B (2026-05-20): replaces the always-heuristic ±3% band with the
    same calibration the Forecast tab uses, so both surfaces show
    intervals derived from the same residual distribution. See
    ``components._callbacks_shared._empirical_interval_from_backtests``
    for the data source and gating rules.
    """

    @patch("components._callbacks_overview._empirical_interval_from_backtests")
    @patch("components._callbacks_overview.redis_get")
    def test_empirical_band_used_when_residuals_available(self, mock_redis_get, mock_empirical):
        """When backtest residuals exist (sample_size >= threshold), the
        band is built from additive residual quantiles, NOT ±3 % of
        each prediction. The band-trace name carries the source label
        so the legend / debug can show calibration provenance."""
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        # Asymmetric quantiles to distinguish from heuristic (which is
        # symmetric ±3%): lower at -800 MW, upper at +1200 MW.
        mock_empirical.return_value = {
            "available": True,
            "lower_error": -800.0,
            "upper_error": 1200.0,
            "sample_size": 144,
            "target_coverage": 0.80,
            "calibration_window_hours": 144,
        }

        fig = _build_overview_hero_chart("FPL", _demand_df())

        band_traces = [t for t in fig.data if t.fill == "toself"]
        # The toself band is the empirical interval (the actuals area
        # uses fill="tozeroy"). Walk fig.data to find the band whose
        # name contains "empirical".
        emp_band = next(
            (t for t in fig.data if t.name and "empirical" in t.name.lower()),
            None,
        )
        assert emp_band is not None, (
            f"No empirical band found. Trace names: {[t.name for t in band_traces]}"
        )
        # Band name carries sample size — surfaces calibration provenance
        assert "n=144" in emp_band.name

        # Band asymmetry verifies additive residual quantiles, not
        # multiplicative ±x%. Upper offset (+1200) > lower offset (800)
        # in absolute value → the band should sit higher above the
        # ensemble than below.
        band_y = np.array([v for v in emp_band.y if v is not None], dtype=float)
        # The band's max should exceed ~22500 + 1200 ≈ 23700 (ensemble peak + upper_error)
        assert band_y.max() > 23_500, (
            f"Empirical band max {band_y.max()} doesn't reflect +1200 offset above ensemble peak"
        )

    @patch("components._callbacks_overview._empirical_interval_from_backtests")
    @patch("components._callbacks_overview.redis_get")
    def test_heuristic_fallback_when_residuals_insufficient(self, mock_redis_get, mock_empirical):
        """When the calibration window is too small (first week
        post-deploy / newly-added region), the helper returns
        ``{available: False}`` and the chart falls back to the ±3 %
        heuristic band, labeled as ``"±3% indicative range"`` so the
        legend never claims a calibration it doesn't have."""
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        mock_empirical.return_value = {"available": False}

        fig = _build_overview_hero_chart("FPL", _demand_df())

        # Should NOT have an empirical-labeled band
        emp_bands = [t for t in fig.data if t.name and "empirical" in t.name.lower()]
        assert emp_bands == [], (
            f"Empirical band rendered despite insufficient residuals; "
            f"names: {[t.name for t in fig.data]}"
        )

        # Should have the heuristic band, labeled honestly
        heur_band = next(
            (t for t in fig.data if t.name == "±3% indicative range"),
            None,
        )
        assert heur_band is not None, (
            f"Heuristic fallback band missing. Trace names: {[t.name for t in fig.data]}"
        )

    @patch("components._callbacks_overview._empirical_interval_from_backtests")
    @patch("components._callbacks_overview.redis_get")
    def test_empirical_helper_called_with_correct_args(self, mock_redis_get, mock_empirical):
        """The hero chart shows a 24-hour ensemble forecast, so the
        empirical helper must be called with model_name='ensemble' and
        horizon_hours=24. Pinning the args prevents a future refactor
        from silently querying a different residual bucket."""
        from components._callbacks_overview import _build_overview_hero_chart

        mock_redis_get.return_value = _redis_forecast_payload()
        mock_empirical.return_value = {"available": False}

        _build_overview_hero_chart("FPL", _demand_df())

        mock_empirical.assert_called_once()
        call_args = mock_empirical.call_args
        # Positional args: (region, model_name, horizon_hours)
        assert call_args[0][0] == "FPL"
        assert call_args[0][1] == "ensemble"
        assert call_args[0][2] == 24


class TestInsightSummaryReadsRedis:
    """``_build_overview_insight``'s forecast clause uses real Redis timestamps."""

    @patch("components._callbacks_overview.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_forecast_clause_uses_real_peak_timestamp(self, mock_metrics, mock_redis_get):
        from components._callbacks_overview import _build_overview_insight

        # Fixture peaks at index 13 of the 24-hour window starting at
        # 2026-05-20 06:00 UTC → 2026-05-20 19:00 UTC.
        mock_redis_get.return_value = _redis_forecast_payload()
        mock_metrics.return_value = {"ensemble": {"mape": 3.85}}

        card = _build_overview_insight("FPL", _demand_df(), "data_scientist")

        # Walk the card's children to find the forecast clause text
        rendered_text = _all_text(card)
        # Real peak time is 19:00 UTC — not "04:00" or some computed offset
        assert "19:00 UTC" in rendered_text
        assert "MAPE 3.9%" in rendered_text
        # Peak value should appear too (~22.5k MW per the fixture)
        assert (
            "22,500" in rendered_text or "22,499" in rendered_text or "22,500 MW" in rendered_text
        )

    @patch("components._callbacks_overview.redis_get")
    def test_forecast_clause_drops_when_redis_cold(self, mock_redis_get):
        """Cold cache → no fabricated forecast clause from simulated."""
        from components._callbacks_overview import _build_overview_insight

        mock_redis_get.return_value = None

        card = _build_overview_insight("FPL", _demand_df(), "data_scientist")

        text = _all_text(card)
        # The fallback copy stays — no fake numbers
        assert "Next-cycle forecast confidence is updating." in text
        # No phantom MAPE / peak from the simulated path
        assert "Next-24h forecast peaks" not in text


def _all_text(node) -> str:
    """Walk a Dash component tree and concatenate string children."""
    pieces: list[str] = []

    def walk(n):
        if isinstance(n, str):
            pieces.append(n)
            return
        children = getattr(n, "children", None)
        if isinstance(children, str):
            pieces.append(children)
        elif isinstance(children, (list, tuple)):
            for c in children:
                walk(c)
        elif children is not None:
            walk(children)

    walk(node)
    return " ".join(pieces)
