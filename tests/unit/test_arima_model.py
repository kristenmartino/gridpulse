"""Focused unit tests for ``models.arima_model``.

Closes a coverage gap from issue #88. ARIMA training + prediction is
already exercised via ``test_training_job_holdout_mape.py`` and
``test_ensemble.py``, but those are integration-tier — they wire the
model into a larger pipeline. These tests target the model module's
own contracts in isolation, with the underlying SARIMAX library
mocked so the suite stays fast (per ``tests/TEST_PYRAMID.md``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sarimax_df():
    """Compact synthetic feature frame for fast unit tests. 96 rows
    (4 daily cycles) is enough for ``train_arima`` to construct a
    payload without hitting the seasonal-D edge case, and SARIMAX
    fitting is mocked at the module level so input size doesn't
    drive runtime anyway."""
    n = 96
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 50_000.0 + 5_000.0 * np.sin(2 * np.pi * np.arange(n) / 24),
            "temperature_2m": 20.0 + 5 * np.cos(2 * np.pi * np.arange(n) / 24),
            "wind_speed_80m": 5.0,
            "shortwave_radiation": 0.5,
            "cooling_degree_days": 0.0,
            "heating_degree_days": 0.0,
        }
    )


@pytest.fixture
def mock_sarimax():
    """Patch ``statsmodels.tsa.statespace.sarimax.SARIMAX`` so
    ``train_arima`` builds a payload without paying the actual MLE
    fitting cost (~5-10s on real data). The fake fit produces:
    - ``params``: a small synthetic coefficient vector (matches the
      lean-payload shape contract)
    - ``resid``: an array of zeros so the drift-detection check (line
      ~140 of arima_model.py) sees no drift and doesn't trigger the
      ``DEFAULT_ORDER`` retrain that would clobber the cached order
    - ``aic``: a finite float so the ``round(fitted.aic, 1)`` log call
      doesn't blow up
    """
    with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_class:
        fake_fitted = MagicMock()
        fake_fitted.params = np.array([0.5, 0.3, -0.1, 0.2, 0.1], dtype=np.float64)
        fake_fitted.resid = np.zeros(100, dtype=np.float64)  # no drift
        fake_fitted.aic = 1000.0
        instance = MagicMock()
        instance.fit.return_value = fake_fitted
        mock_class.return_value = instance
        yield mock_class


class TestTrainArimaPayload:
    """``train_arima`` returns a lean dict — params + tail_y/tail_exog +
    order/seasonal_order — designed to pickle in kilobytes rather than
    the ~500 MB a full SARIMAXResults would. The Kalman filter state is
    reconstructed at predict time from the params + tail data."""

    def test_returns_lean_payload_keys(self, sarimax_df, mock_sarimax):
        from models.arima_model import train_arima

        # Force the cached-order fast path so we skip pmdarima entirely.
        # Combined with the SARIMAX mock, training is ~milliseconds.
        result = train_arima(
            sarimax_df,
            cached_order=(2, 1, 2),
            cached_seasonal_order=(1, 1, 1, 24),
        )

        # Lean payload contract — every key the predict path reads.
        for key in ("params", "order", "seasonal_order", "exog_cols", "tail_y", "tail_exog"):
            assert key in result, f"train_arima payload missing {key!r}"

        assert result["order"] == (2, 1, 2)
        assert result["seasonal_order"] == (1, 1, 1, 24)
        assert isinstance(result["params"], np.ndarray)
        # Params should be a 1-D coefficient vector
        assert result["params"].ndim == 1
        # Tail rows preserved as float32 to keep the pickle small.
        assert result["tail_y"].dtype == np.float32

    def test_tail_size_capped_at_pickle_tail_rows(self, mock_sarimax):
        """The tail kept with the payload is capped at PICKLE_TAIL_ROWS
        (240 = 10 daily cycles) regardless of input length — the cap
        is what keeps the pickle in kilobytes."""
        from models.arima_model import PICKLE_TAIL_ROWS, train_arima

        # Build an oversized frame to exercise the cap. Mocked SARIMAX
        # still runs in milliseconds regardless of size.
        n = PICKLE_TAIL_ROWS * 4
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": 50_000.0 + np.sin(np.arange(n) / 24.0) * 1000,
                "temperature_2m": 20.0,
                "wind_speed_80m": 5.0,
                "shortwave_radiation": 0.5,
                "cooling_degree_days": 0.0,
                "heating_degree_days": 0.0,
            }
        )
        result = train_arima(
            df,
            cached_order=(2, 1, 2),
            cached_seasonal_order=(1, 1, 1, 24),
        )
        assert len(result["tail_y"]) == PICKLE_TAIL_ROWS


class TestCachedOrderFastPath:
    """The cached-order parameter (added in the resume + speedups PR
    #84) is the dominant per-BA training speedup. When supplied it
    must skip ``_auto_select_order`` entirely — the test patches that
    helper to a sentinel and asserts it's never called."""

    def test_cached_order_skips_auto_select(self, sarimax_df, mock_sarimax):
        from models import arima_model

        with patch.object(arima_model, "_auto_select_order") as mock_auto:
            mock_auto.return_value = ((9, 9, 9), (9, 9, 9, 24))
            arima_model.train_arima(
                sarimax_df,
                cached_order=(2, 1, 2),
                cached_seasonal_order=(1, 1, 1, 24),
            )
            mock_auto.assert_not_called()

    def test_cached_order_value_round_trips_to_payload(self, sarimax_df, mock_sarimax):
        """The cached order must appear verbatim in the saved payload —
        if train_arima silently re-selected, the next run's cache
        lookup would chase the wrong order."""
        from models.arima_model import train_arima

        result = train_arima(
            sarimax_df,
            cached_order=(1, 1, 1),
            cached_seasonal_order=(0, 1, 0, 24),
        )
        assert result["order"] == (1, 1, 1)
        assert result["seasonal_order"] == (0, 1, 0, 24)


class TestPredictArima:
    """Prediction supports both the lean payload (params + tail) and a
    legacy payload that stored a fitted ``SARIMAXResults`` under
    ``"model"`` for one roll-forward cycle of backward compatibility."""

    def test_predict_clamps_negative_forecasts_to_zero(self):
        """Demand can't be negative — the predict path explicitly
        clamps the SARIMAX output to ``np.maximum(forecast, 0)``."""
        from models.arima_model import predict_arima

        # Legacy-style payload with a mocked fitted model that returns
        # negatives. Avoids actually fitting SARIMAX (slow + flaky).
        fitted = MagicMock()
        fitted.forecast.return_value = np.array([-100.0, 50.0, -200.0, 0.0])
        legacy_payload = {"model": fitted}

        future_exog = pd.DataFrame(
            {
                "temperature_2m": [20.0] * 4,
                "wind_speed_80m": [5.0] * 4,
                "shortwave_radiation": [0.5] * 4,
                "cooling_degree_days": [0.0] * 4,
                "heating_degree_days": [0.0] * 4,
            }
        )
        result = predict_arima(legacy_payload, future_exog, periods=4)
        # Negatives clamped, non-negatives preserved
        assert (result >= 0).all()
        assert result[0] == 0.0
        assert result[1] == 50.0

    def test_predict_returns_nan_array_on_failure(self):
        """When SARIMAX raises during forecast, the predict path
        catches it and returns ``np.full(periods, np.nan)`` so the
        scoring caller can treat it as a per-model failure without
        the broader region-loop crashing."""
        from models.arima_model import predict_arima

        fitted = MagicMock()
        fitted.forecast.side_effect = ValueError("synthetic SARIMAX failure")
        legacy_payload = {"model": fitted}

        result = predict_arima(legacy_payload, pd.DataFrame(), periods=10)
        assert result.shape == (10,)
        assert np.isnan(result).all()

    def test_predict_arima_handles_missing_exog_columns(self):
        """If the future-exog DataFrame is missing some expected
        columns, ``_get_exog`` silently drops them. Predict still
        succeeds with whatever columns are present (or returns None
        exog if all expected cols are missing)."""
        from models.arima_model import predict_arima

        fitted = MagicMock()
        fitted.forecast.return_value = np.array([100.0] * 5)
        legacy_payload = {"model": fitted}

        # No exog columns at all — predict should still work.
        result = predict_arima(legacy_payload, pd.DataFrame(index=range(5)), periods=5)
        assert result.shape == (5,)
        # forecast was called at least once
        assert fitted.forecast.called


class TestPredictArimaGapActuals:
    """#226: with a train->score gap, the frozen Kalman state must be advanced
    through the observed gap actuals (``append()``) before forecasting, so the
    horizon origin is the last real value, not the stale ``train_end``."""

    def _legacy(self, forecast_ramp, train_end):
        fitted = MagicMock()
        advanced = MagicMock()
        fitted.append.return_value = advanced
        fitted.forecast.return_value = forecast_ramp
        advanced.forecast.return_value = forecast_ramp
        return {"model": fitted, "train_end": train_end}, fitted, advanced

    def test_full_gap_actuals_gives_true_one_step_origin(self):
        from models.arima_model import predict_arima

        train_end = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=6)  # offset 6 -> gap 5
        ramp = np.arange(100.0, 200.0)
        legacy, fitted, advanced = self._legacy(ramp, train_end)
        gap_actuals = np.array([1500.0, 1510, 1490, 1505, 1500])  # 5 == gap

        res = predict_arima(
            legacy,
            pd.DataFrame(index=range(20)),
            periods=3,
            start_ts=start_ts,
            gap_actuals=gap_actuals,
        )
        # append() advanced the state through all 5 gap actuals
        assert fitted.append.called
        assert len(fitted.append.call_args.args[0]) == 5
        # a == gap => slice offset 0 => horizon starts at the advanced forecast[0]
        assert list(res["forecast"]) == [100.0, 101.0, 102.0]

    def test_partial_gap_actuals_reduces_slice_offset(self):
        from models.arima_model import predict_arima

        train_end = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=6)  # gap 5
        ramp = np.arange(100.0, 200.0)
        legacy, fitted, advanced = self._legacy(ramp, train_end)
        gap_actuals = np.array([1500.0, 1510, 1490])  # only 3 of 5 (EIA lag)

        res = predict_arima(
            legacy,
            pd.DataFrame(index=range(20)),
            periods=3,
            start_ts=start_ts,
            gap_actuals=gap_actuals,
        )
        assert len(fitted.append.call_args.args[0]) == 3
        # a=3 -> slice offset = gap-a = 2 -> advanced.forecast[2:5]
        assert list(res["forecast"]) == [102.0, 103.0, 104.0]

    def test_no_gap_actuals_keeps_stale_slice(self):
        from models.arima_model import predict_arima

        train_end = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=6)  # gap 5
        ramp = np.arange(100.0, 200.0)
        legacy, fitted, advanced = self._legacy(ramp, train_end)

        res = predict_arima(
            legacy,
            pd.DataFrame(index=range(20)),
            periods=3,
            start_ts=start_ts,
            gap_actuals=None,
        )
        # No append; the stale path slices [gap:gap+periods] = [5:8]
        assert not fitted.append.called
        assert list(res["forecast"]) == [105.0, 106.0, 107.0]

    def test_append_failure_falls_back_to_stale(self):
        from models.arima_model import predict_arima

        train_end = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        start_ts = train_end + pd.Timedelta(hours=6)
        ramp = np.arange(100.0, 200.0)
        legacy, fitted, advanced = self._legacy(ramp, train_end)
        fitted.append.side_effect = ValueError("synthetic append failure")

        res = predict_arima(
            legacy,
            pd.DataFrame(index=range(20)),
            periods=3,
            start_ts=start_ts,
            gap_actuals=np.array([1.0, 2, 3, 4, 5]),
        )
        # append raised -> fall back to the original stale forecast, no crash
        assert list(res["forecast"]) == [105.0, 106.0, 107.0]


class TestArimaConstants:
    """Regression: the exog column list and pickle-tail size are
    structural contracts the scoring + persistence paths depend on."""

    def test_exog_cols_match_training_features(self):
        from models.arima_model import ARIMA_EXOG_COLS

        # The five weather features the training pipeline produces
        # (data/feature_engineering.py). Drift here would silently
        # drop weather signal from SARIMAX.
        assert "temperature_2m" in ARIMA_EXOG_COLS
        assert "wind_speed_80m" in ARIMA_EXOG_COLS
        assert "shortwave_radiation" in ARIMA_EXOG_COLS
        assert "cooling_degree_days" in ARIMA_EXOG_COLS
        assert "heating_degree_days" in ARIMA_EXOG_COLS

    def test_default_seasonal_order_enforces_d1(self):
        """``D=1`` (seasonal differencing) is critical — without it,
        SARIMAX's integrated component causes forecast drift away
        from the daily cycle. Issue surfaced during V0.3 backtesting."""
        from models.arima_model import DEFAULT_SEASONAL_ORDER

        assert DEFAULT_SEASONAL_ORDER[1] >= 1, (
            "Seasonal D must be >= 1; D=0 produces drifting forecasts"
        )
        assert DEFAULT_SEASONAL_ORDER[3] == 24, "Daily seasonality (m=24) expected"


class TestGetExogNaNHandling:
    """``_get_exog`` feeds SARIMAX's exogenous matrix. SARIMAX cannot
    fit/forecast with NaN in exog, so the function must return a clean
    float array. The archive-unstable ``wind_speed_80m`` column (#164)
    can arrive object-dtype with ``None``/``NaN`` — which made the old
    ``np.isnan`` guard raise ``ufunc 'isnan' not supported ... casting
    rule 'safe'`` *before* the fill ran, dropping ARIMA from every
    region's holdout ensemble (#176)."""

    def test_object_dtype_with_none_does_not_raise_and_is_nan_free(self):
        from models.arima_model import _get_exog

        n = 48
        df = pd.DataFrame(
            {
                "temperature_2m": np.full(n, 20.0),
                # object-dtype column: floats interspersed with None.
                "wind_speed_80m": pd.Series(
                    [None, None] + [5.0] * (n - 4) + [None, None], dtype=object
                ),
                "shortwave_radiation": np.full(n, 0.5),
                "cooling_degree_days": np.zeros(n),
                "heating_degree_days": np.zeros(n),
            }
        )
        # Sanity: the column really is object dtype (the crash precondition).
        assert df["wind_speed_80m"].dtype == object

        exog = _get_exog(df)

        assert exog is not None
        assert exog.dtype == np.float64
        assert not np.isnan(exog).any(), (
            "exog reaching SARIMAX must be NaN-free; object-dtype None/NaN "
            "in an archive-unstable regressor would otherwise drop ARIMA "
            "from the holdout ensemble (#176)."
        )

    def test_interior_nan_is_forward_filled(self):
        from models.arima_model import _get_exog

        n = 24
        wind = np.full(n, 5.0)
        wind[10:13] = np.nan
        df = pd.DataFrame(
            {
                "temperature_2m": np.full(n, 20.0),
                "wind_speed_80m": wind,
                "shortwave_radiation": np.full(n, 0.5),
                "cooling_degree_days": np.zeros(n),
                "heating_degree_days": np.zeros(n),
            }
        )

        exog = _get_exog(df)

        assert not np.isnan(exog).any()
        # Forward-fill carries the last good value (5.0) across the gap.
        wind_col = exog[:, 1]
        assert wind_col[10] == pytest.approx(5.0)
        assert wind_col[12] == pytest.approx(5.0)

    def test_all_nan_column_degrades_to_zero(self):
        from models.arima_model import _get_exog

        n = 24
        df = pd.DataFrame(
            {
                "temperature_2m": np.full(n, 20.0),
                "wind_speed_80m": np.full(n, np.nan),
                "shortwave_radiation": np.full(n, 0.5),
                "cooling_degree_days": np.zeros(n),
                "heating_degree_days": np.zeros(n),
            }
        )

        exog = _get_exog(df)

        assert not np.isnan(exog).any()
        # An all-NaN column can't be ff/bfilled; it zeroes out.
        assert (exog[:, 1] == 0.0).all()
