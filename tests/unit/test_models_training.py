"""
Comprehensive tests for model training modules.

Covers:
- models/arima_model.py — SARIMAX with pmdarima order selection
- models/prophet_model.py — Prophet with weather regressors
- models/xgboost_model.py — XGBoost with TimeSeriesSplit CV + SHAP
- models/training.py — Training orchestrator: train_all_models, save/load, validation

Mocking strategy:
- pmdarima.auto_arima: always mocked (slow fitting)
- Prophet class: always mocked (slow fitting)
- SARIMAX: mocked in most tests (slow fitting)
- XGBoost on small data: runs real training (fast enough)
- pickle.dump/load: mocked for serialization roundtrip tests
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.arima_model import (
    ARIMA_EXOG_COLS,
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    _auto_select_order,
    _get_exog,
    predict_arima,
    train_arima,
)
from models.prophet_model import (
    PROPHET_REGRESSORS,
    _get_prophet,
    create_prophet_model,
    predict_prophet,
    train_prophet,
)
from models.training import (
    _safe_model_path,
    _validate_region,
    load_models,
    save_models,
    train_all_models,
)
from models.xgboost_model import (
    EXCLUDE_COLS,
    _compute_mape,
    _get_feature_cols,
    _top_features,
    compute_shap_values,
    predict_xgboost,
    train_xgboost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_feature_df(n_rows: int = 500) -> pd.DataFrame:
    """Build a minimal feature DataFrame for fast tests.

    Contains all columns needed by ARIMA exog, Prophet regressors,
    and XGBoost feature selection.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")
    hours = np.arange(n_rows)

    daily = 5000 * np.sin(2 * np.pi * (hours - 6) / 24)
    demand = 40000 + daily + rng.normal(0, 500, n_rows)
    demand = np.maximum(demand, 5000)

    temp = 75 + 10 * np.sin(2 * np.pi * (hours - 6) / 24) + rng.normal(0, 3, n_rows)
    solar = np.maximum(0, 800 * np.sin(2 * np.pi * (hours % 24 - 6) / 24))

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": demand,
            "forecast_mw": demand + rng.normal(0, 1000, n_rows),
            "region": "ERCOT",
            "temperature_2m": temp,
            "apparent_temperature": temp - 3,
            "wind_speed_80m": np.abs(15 + rng.normal(0, 6, n_rows)),
            "shortwave_radiation": solar,
            "cooling_degree_days": np.maximum(temp - 65, 0),
            "heating_degree_days": np.maximum(65 - temp, 0),
            "is_holiday": rng.choice([0, 1], n_rows, p=[0.97, 0.03]).astype(float),
            "wind_speed_10m": np.abs(10 + rng.normal(0, 5, n_rows)),
            "hour_of_day": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "month": timestamps.month,
            "demand_lag_24": np.roll(demand, 24),
            "demand_lag_168": np.roll(demand, 168),
            "demand_rolling_24_mean": pd.Series(demand).rolling(24, min_periods=1).mean().values,
            "demand_rolling_168_mean": pd.Series(demand).rolling(168, min_periods=1).mean().values,
            "temp_rolling_24_mean": pd.Series(temp).rolling(24, min_periods=1).mean().values,
        }
    )
    return df


# ---------------------------------------------------------------------------
# TestArimaModel
# ---------------------------------------------------------------------------


class TestArimaModel:
    """Tests for models/arima_model.py functions."""

    def test_get_exog_extracts_available_columns(self):
        """_get_exog returns array for columns present in df."""
        df = _make_small_feature_df(50)
        exog = _get_exog(df)
        assert exog is not None
        # Should have as many columns as ARIMA_EXOG_COLS present in df
        available = [c for c in ARIMA_EXOG_COLS if c in df.columns]
        assert exog.shape == (len(df), len(available))

    def test_get_exog_returns_none_when_no_columns(self):
        """_get_exog returns None when no exog columns are in the df."""
        df = pd.DataFrame({"timestamp": [1, 2], "demand_mw": [100, 200]})
        assert _get_exog(df) is None

    def test_get_exog_handles_nan_values(self):
        """_get_exog fills NaN values (forward/backward/zero fill)."""
        df = _make_small_feature_df(20)
        # Inject NaN values
        df.loc[5:8, "temperature_2m"] = np.nan
        df.loc[0:2, "wind_speed_80m"] = np.nan
        exog = _get_exog(df)
        assert exog is not None
        assert not np.isnan(exog).any(), "NaN values should be filled"

    def test_get_exog_pads_when_n_rows_exceeds_data(self):
        """_get_exog pads with last row when n_rows > len(df)."""
        df = _make_small_feature_df(10)
        exog = _get_exog(df, n_rows=20)
        assert exog is not None
        assert exog.shape[0] == 20
        # Padded rows should equal last row of original
        np.testing.assert_array_equal(exog[10], exog[9])

    def test_get_exog_truncates_when_n_rows_less_than_data(self):
        """_get_exog truncates when n_rows < len(df)."""
        df = _make_small_feature_df(20)
        exog = _get_exog(df, n_rows=5)
        assert exog is not None
        assert exog.shape[0] == 5

    @patch("models.arima_model._auto_select_order")
    @patch("models.arima_model.SARIMAX", create=True)
    def test_train_arima_returns_expected_keys(self, mock_sarimax_cls, mock_auto):
        """train_arima returns dict with model, order, seasonal_order, exog_cols."""
        mock_auto.return_value = (DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER)

        mock_fitted = MagicMock()
        mock_fitted.aic = 1000.0
        mock_fitted.resid = np.random.normal(0, 100, 100)
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted
        mock_sarimax_cls.return_value = mock_model

        with (
            patch("models.arima_model.SARIMAX", mock_sarimax_cls),
            patch.dict(
                "sys.modules",
                {"statsmodels.tsa.statespace.sarimax": MagicMock(SARIMAX=mock_sarimax_cls)},
            ),
        ):
            df = _make_small_feature_df(300)
            result = train_arima(df, auto_order=True)

        assert "model" in result
        assert "order" in result
        assert "seasonal_order" in result
        assert "exog_cols" in result
        assert result["exog_cols"] == ARIMA_EXOG_COLS

    @patch("models.arima_model._auto_select_order")
    def test_train_arima_uses_default_order_when_auto_disabled(self, mock_auto):
        """train_arima uses DEFAULT_ORDER when auto_order=False."""
        mock_fitted = MagicMock()
        mock_fitted.aic = 500.0
        mock_fitted.resid = np.random.normal(0, 50, 100)
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted

        mock_sarimax = MagicMock(return_value=mock_model)
        with patch.dict(
            "sys.modules", {"statsmodels.tsa.statespace.sarimax": MagicMock(SARIMAX=mock_sarimax)}
        ):
            df = _make_small_feature_df(300)
            result = train_arima(df, auto_order=False)

        # auto_select_order should NOT have been called
        mock_auto.assert_not_called()
        assert result["order"] == DEFAULT_ORDER
        assert result["seasonal_order"] == DEFAULT_SEASONAL_ORDER

    @patch("models.arima_model._auto_select_order")
    def test_train_arima_limits_rows(self, mock_auto):
        """train_arima uses at most max_training_rows of data."""
        mock_auto.return_value = (DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER)

        calls_y_length = []

        def capture_sarimax(y, exog=None, **kwargs):
            calls_y_length.append(len(y))
            mock_fitted = MagicMock()
            mock_fitted.aic = 500.0
            mock_fitted.resid = np.random.normal(0, 50, min(100, len(y)))
            mock_inst = MagicMock()
            mock_inst.fit.return_value = mock_fitted
            return mock_inst

        mock_sarimax = MagicMock(side_effect=capture_sarimax)
        with patch.dict(
            "sys.modules", {"statsmodels.tsa.statespace.sarimax": MagicMock(SARIMAX=mock_sarimax)}
        ):
            df = _make_small_feature_df(3000)
            train_arima(df, auto_order=True, max_training_rows=500)

        # The y array passed to SARIMAX should be <= 500
        assert calls_y_length[0] <= 500

    def test_predict_arima_shape_matches_periods(self):
        """predict_arima returns array of length == periods."""
        periods = 48
        mock_fitted = MagicMock()
        mock_fitted.forecast.return_value = np.random.uniform(30000, 50000, periods)

        model_dict = {
            "model": mock_fitted,
            "order": DEFAULT_ORDER,
            "seasonal_order": DEFAULT_SEASONAL_ORDER,
            "exog_cols": ARIMA_EXOG_COLS,
        }
        future_df = _make_small_feature_df(periods)
        result = predict_arima(model_dict, future_df, periods=periods)
        assert len(result) == periods

    def test_predict_arima_clamps_negative_values(self):
        """predict_arima clamps negative forecasts to 0."""
        periods = 24
        raw_forecast = np.array([-100, 500, -50] + [1000] * 21, dtype=float)
        mock_fitted = MagicMock()
        mock_fitted.forecast.return_value = raw_forecast

        model_dict = {
            "model": mock_fitted,
            "order": DEFAULT_ORDER,
            "seasonal_order": DEFAULT_SEASONAL_ORDER,
            "exog_cols": ARIMA_EXOG_COLS,
        }
        future_df = _make_small_feature_df(periods)
        result = predict_arima(model_dict, future_df, periods=periods)
        assert (result >= 0).all(), "All forecasts should be non-negative"

    def test_predict_arima_returns_nan_on_failure(self):
        """predict_arima returns NaN array when forecast raises."""
        periods = 24
        mock_fitted = MagicMock()
        mock_fitted.forecast.side_effect = ValueError("Singular matrix")

        model_dict = {
            "model": mock_fitted,
            "order": DEFAULT_ORDER,
            "seasonal_order": DEFAULT_SEASONAL_ORDER,
            "exog_cols": ARIMA_EXOG_COLS,
        }
        future_df = _make_small_feature_df(periods)
        result = predict_arima(model_dict, future_df, periods=periods)
        assert len(result) == periods
        assert np.isnan(result).all()

    def test_auto_select_order_returns_valid_tuple(self):
        """_auto_select_order returns (order, seasonal_order) tuples."""
        mock_auto = MagicMock()
        mock_auto.order = (1, 1, 1)
        mock_auto.seasonal_order = (1, 1, 1, 24)

        mock_pm = MagicMock()
        mock_pm.auto_arima.return_value = mock_auto

        with patch.dict("sys.modules", {"pmdarima": mock_pm}):
            y = np.random.uniform(30000, 50000, 200)
            order, seasonal_order = _auto_select_order(y, None)

        assert isinstance(order, tuple)
        assert len(order) == 3
        assert isinstance(seasonal_order, tuple)
        assert len(seasonal_order) == 4

    def test_auto_select_order_fallback_on_failure(self):
        """_auto_select_order returns defaults when auto_arima fails."""
        mock_pm = MagicMock()
        mock_pm.auto_arima.side_effect = RuntimeError("Convergence failed")

        with patch.dict("sys.modules", {"pmdarima": mock_pm}):
            y = np.random.uniform(30000, 50000, 200)
            order, seasonal_order = _auto_select_order(y, None)

        assert order == DEFAULT_ORDER
        assert seasonal_order == DEFAULT_SEASONAL_ORDER


# ---------------------------------------------------------------------------
# TestProphetModel
# ---------------------------------------------------------------------------


class TestProphetModel:
    """Tests for models/prophet_model.py functions."""

    def test_get_prophet_returns_class(self):
        """_get_prophet returns the Prophet class (lazy import)."""
        mock_prophet_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.Prophet = mock_prophet_cls

        import models.prophet_model as pm

        original = pm._Prophet
        try:
            pm._Prophet = None  # Reset cached class
            with patch.dict("sys.modules", {"prophet": mock_module}):
                result = _get_prophet()
            assert result is mock_prophet_cls
        finally:
            pm._Prophet = original

    def test_get_prophet_caches_import(self):
        """_get_prophet caches the Prophet class after first import."""
        import models.prophet_model as pm

        sentinel = MagicMock()
        original = pm._Prophet
        try:
            pm._Prophet = sentinel
            assert _get_prophet() is sentinel
        finally:
            pm._Prophet = original

    @patch("models.prophet_model._get_prophet")
    def test_create_prophet_model_attaches_regressors(self, mock_get):
        """create_prophet_model adds all PROPHET_REGRESSORS to the model."""
        mock_instance = MagicMock()
        mock_get.return_value = MagicMock(return_value=mock_instance)

        model = create_prophet_model()

        assert model.add_regressor.call_count == len(PROPHET_REGRESSORS)
        # Verify each regressor name was passed
        called_names = {call.args[0] for call in model.add_regressor.call_args_list}
        expected_names = {name for name, _ in PROPHET_REGRESSORS}
        assert called_names == expected_names

    @patch("models.prophet_model._get_prophet")
    def test_create_prophet_model_uses_logistic_growth(self, mock_get):
        """create_prophet_model configures logistic growth."""
        mock_cls = MagicMock()
        mock_get.return_value = mock_cls

        create_prophet_model()

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["growth"] == "logistic"

    @patch("models.prophet_model.create_prophet_model")
    def test_train_prophet_calls_fit(self, mock_create):
        """train_prophet calls model.fit with ds, y, cap, floor columns."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        df = _make_small_feature_df(100)
        train_prophet(df, target_col="demand_mw")

        mock_model.fit.assert_called_once()
        train_df = mock_model.fit.call_args[0][0]
        assert "ds" in train_df.columns
        assert "y" in train_df.columns
        assert "cap" in train_df.columns
        assert "floor" in train_df.columns
        assert (train_df["floor"] == 0).all()

    @patch("models.prophet_model.create_prophet_model")
    def test_train_prophet_stores_demand_cap(self, mock_create):
        """train_prophet stores _demand_cap on the model for predict."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        df = _make_small_feature_df(100)
        result = train_prophet(df, target_col="demand_mw")

        assert hasattr(result, "_demand_cap")
        expected_cap = float(df["demand_mw"].max() * 1.5)
        assert result._demand_cap == pytest.approx(expected_cap)

    @patch("models.prophet_model.create_prophet_model")
    def test_train_prophet_handles_missing_regressors(self, mock_create):
        """train_prophet fills missing regressors with 0.0."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        # DataFrame missing some Prophet regressors
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
                "demand_mw": np.random.uniform(30000, 50000, 50),
                "temperature_2m": np.random.uniform(60, 90, 50),
            }
        )
        train_prophet(df, target_col="demand_mw")

        mock_model.fit.assert_called_once()
        train_df = mock_model.fit.call_args[0][0]
        # All regressor columns should exist in the training df
        for regressor_name, _ in PROPHET_REGRESSORS:
            assert regressor_name in train_df.columns

    def test_predict_prophet_returns_expected_keys(self):
        """predict_prophet returns dict with forecast, lower/upper bands, timestamps."""
        periods = 48
        mock_model = MagicMock()
        mock_model._demand_cap = 60000

        # Build a realistic forecast DataFrame that model.predict would return
        future_dates = pd.date_range("2024-01-01", periods=200 + periods, freq="h")
        mock_forecast_df = pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": np.random.uniform(30000, 50000, len(future_dates)),
                "yhat_lower": np.random.uniform(25000, 35000, len(future_dates)),
                "yhat_upper": np.random.uniform(45000, 60000, len(future_dates)),
            }
        )
        mock_model.predict.return_value = mock_forecast_df
        mock_model.make_future_dataframe.return_value = pd.DataFrame(
            {
                "ds": future_dates,
            }
        )

        df = _make_small_feature_df(200)
        result = predict_prophet(mock_model, df, periods=periods)

        assert "forecast" in result
        assert "lower_80" in result
        assert "upper_80" in result
        assert "lower_95" in result
        assert "upper_95" in result
        assert "timestamps" in result
        assert len(result["forecast"]) == periods

    def test_predict_prophet_shape_matches_periods(self):
        """predict_prophet arrays all have length == periods."""
        periods = 72
        mock_model = MagicMock()
        mock_model._demand_cap = 60000

        future_dates = pd.date_range("2024-01-01", periods=100 + periods, freq="h")
        mock_forecast_df = pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": np.random.uniform(30000, 50000, len(future_dates)),
                "yhat_lower": np.random.uniform(25000, 35000, len(future_dates)),
                "yhat_upper": np.random.uniform(45000, 60000, len(future_dates)),
            }
        )
        mock_model.predict.return_value = mock_forecast_df
        mock_model.make_future_dataframe.return_value = pd.DataFrame(
            {
                "ds": future_dates,
            }
        )

        df = _make_small_feature_df(100)
        result = predict_prophet(mock_model, df, periods=periods)

        for key in ["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]:
            assert len(result[key]) == periods, f"{key} has wrong length"


# ---------------------------------------------------------------------------
# TestXGBoostModel
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    """Tests for models/xgboost_model.py functions."""

    def test_get_feature_cols_excludes_target_and_metadata(self):
        """_get_feature_cols excludes EXCLUDE_COLS from feature list."""
        df = _make_small_feature_df(10)
        cols = _get_feature_cols(df)
        for excluded in EXCLUDE_COLS:
            assert excluded not in cols, f"{excluded} should be excluded"

    def test_get_feature_cols_returns_numeric_only(self):
        """_get_feature_cols only returns numeric columns."""
        df = _make_small_feature_df(10)
        df["string_col"] = "abc"
        cols = _get_feature_cols(df)
        assert "string_col" not in cols

    def test_get_feature_cols_returns_nonempty(self):
        """_get_feature_cols returns at least some feature columns."""
        df = _make_small_feature_df(10)
        cols = _get_feature_cols(df)
        assert len(cols) > 0

    def test_compute_mape_known_values(self):
        """_compute_mape returns correct MAPE for known inputs."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.10 + 0.10 → mean = 0.10 → 10%
        result = _compute_mape(actual, predicted)
        assert result == pytest.approx(10.0)

    def test_compute_mape_zero_handling(self):
        """_compute_mape returns inf when all actuals are near-zero."""
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert _compute_mape(actual, predicted) == float("inf")

    def test_compute_mape_excludes_near_zero(self):
        """_compute_mape excludes near-zero actuals from calculation."""
        actual = np.array([0.0, 100.0])
        predicted = np.array([10.0, 110.0])
        result = _compute_mape(actual, predicted)
        # Only the second element counts: |10/100| = 10%
        assert result == pytest.approx(10.0)

    def test_top_features_returns_sorted(self):
        """_top_features returns top N features by importance, descending."""
        names = ["a", "b", "c", "d", "e"]
        importances = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
        top3 = _top_features(names, importances, n=3)
        assert top3 == ["b", "c", "a"]

    def test_top_features_handles_n_larger_than_features(self):
        """_top_features returns all features when n > len(names)."""
        names = ["a", "b"]
        importances = np.array([0.3, 0.7])
        result = _top_features(names, importances, n=10)
        assert len(result) == 2
        assert result[0] == "b"  # Higher importance first

    def test_train_xgboost_returns_expected_keys(self, feature_df):
        """train_xgboost returns dict with model, feature_names, feature_importances, cv_scores."""
        # Use small params for speed
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        result = train_xgboost(feature_df, params=params, n_splits=2)

        assert "model" in result
        assert "feature_names" in result
        assert "feature_importances" in result
        assert "cv_scores" in result
        assert isinstance(result["feature_importances"], dict)
        assert len(result["cv_scores"]) == 2  # n_splits=2

    def test_train_xgboost_cv_scores_are_finite(self, feature_df):
        """train_xgboost CV scores are finite MAPE values."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        result = train_xgboost(feature_df, params=params, n_splits=2)
        for score in result["cv_scores"]:
            assert np.isfinite(score)
            assert score > 0

    def test_predict_xgboost_shape_matches_input(self, feature_df):
        """predict_xgboost returns array matching input DataFrame length."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        model_dict = train_xgboost(feature_df, params=params, n_splits=2)
        predictions = predict_xgboost(model_dict, feature_df)
        assert len(predictions) == len(feature_df)

    def test_predict_xgboost_positive_values(self, feature_df):
        """predict_xgboost clamps all predictions to non-negative."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        model_dict = train_xgboost(feature_df, params=params, n_splits=2)
        predictions = predict_xgboost(model_dict, feature_df)
        assert (predictions >= 0).all()

    def test_predict_xgboost_deterministic(self, feature_df):
        """predict_xgboost with same seed gives identical results."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        model_dict = train_xgboost(feature_df, params=params, n_splits=2)
        pred1 = predict_xgboost(model_dict, feature_df)
        pred2 = predict_xgboost(model_dict, feature_df)
        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_xgboost_handles_missing_features(self, feature_df):
        """predict_xgboost fills missing feature columns with 0.0."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        model_dict = train_xgboost(feature_df, params=params, n_splits=2)

        # Drop some feature columns from the prediction df
        incomplete_df = feature_df.drop(columns=[model_dict["feature_names"][0]])
        predictions = predict_xgboost(model_dict, incomplete_df)
        assert len(predictions) == len(incomplete_df)

    def test_compute_shap_values_returns_dict_or_none(self, feature_df):
        """compute_shap_values returns dict with shap_values, feature_names, base_value."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
        }
        model_dict = train_xgboost(feature_df, params=params, n_splits=2)

        try:
            result = compute_shap_values(model_dict, feature_df, max_samples=50)
            assert "shap_values" in result
            assert "feature_names" in result
            assert "base_value" in result
            assert isinstance(result["base_value"], float)
        except ImportError:
            # shap not installed — acceptable in CI
            pytest.skip("shap not installed")

    def test_compute_shap_values_returns_none_on_error(self):
        """compute_shap_values returns error or raises when shap fails."""
        model_dict = {
            "model": MagicMock(),
            "feature_names": ["a", "b"],
        }
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = Exception("SHAP error")

        with (
            patch.dict("sys.modules", {"shap": mock_shap}),
            pytest.raises(Exception, match="SHAP error"),
        ):
            compute_shap_values(model_dict, df, max_samples=10)


# ---------------------------------------------------------------------------
# TestTrainingOrchestrator
# ---------------------------------------------------------------------------


class TestTrainingOrchestrator:
    """Tests for models/training.py functions."""

    def test_validate_region_accepts_valid(self):
        """_validate_region passes for known BA codes."""
        # Should not raise
        _validate_region("ERCOT")
        _validate_region("CAISO")
        _validate_region("PJM")
        _validate_region("FPL")

    def test_validate_region_rejects_invalid_format(self):
        """_validate_region raises on non-uppercase-alphanumeric strings."""
        with pytest.raises(ValueError, match="Invalid region format"):
            _validate_region("ercot")
        with pytest.raises(ValueError, match="Invalid region format"):
            _validate_region("../etc")
        with pytest.raises(ValueError, match="Invalid region format"):
            _validate_region("ERCOT; DROP TABLE")

    def test_validate_region_rejects_unknown_region(self):
        """_validate_region raises for valid format but unknown BA code."""
        with pytest.raises(ValueError, match="Unknown region"):
            _validate_region("ZZZZZ")

    def test_safe_model_path_builds_correct_path(self, tmp_path):
        """_safe_model_path returns base_dir / <region>_models.pkl."""
        path = _safe_model_path(str(tmp_path), "ERCOT")
        assert path.endswith("ERCOT_models.pkl")
        assert str(tmp_path) in path

    def test_safe_model_path_prevents_traversal(self, tmp_path):
        """_safe_model_path raises on path traversal attempts."""
        # Path traversal via region name with dots should be caught
        # by _validate_region first, but _safe_model_path has its own check
        with pytest.raises(ValueError):
            _safe_model_path(str(tmp_path), "../../etc/passwd")

    @patch("models.training.train_prophet")
    @patch("models.training.predict_prophet")
    @patch("models.training.train_arima")
    @patch("models.training.predict_arima")
    @patch("models.training.train_xgboost")
    @patch("models.training.predict_xgboost")
    @patch("models.training.compute_all_metrics")
    @patch("models.training.compute_ensemble_weights")
    def test_train_all_models_returns_expected_structure(
        self,
        mock_weights,
        mock_metrics,
        mock_xgb_pred,
        mock_xgb_train,
        mock_arima_pred,
        mock_arima_train,
        mock_prophet_pred,
        mock_prophet_train,
    ):
        """train_all_models returns dict with region, models, metrics, weights."""
        n = 500
        df = _make_small_feature_df(n)

        mock_prophet_train.return_value = MagicMock()
        mock_prophet_pred.return_value = {"forecast": np.random.uniform(30000, 50000, 168)}
        mock_arima_train.return_value = {
            "model": MagicMock(),
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 24),
            "exog_cols": [],
        }
        mock_arima_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_xgb_train.return_value = {
            "model": MagicMock(),
            "feature_names": ["a"],
            "feature_importances": {"a": 1.0},
            "cv_scores": [5.0],
        }
        mock_xgb_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_metrics.return_value = {"mape": 5.0, "rmse": 1000.0, "mae": 800.0, "r2": 0.9}
        mock_weights.return_value = {"prophet": 0.4, "arima": 0.3, "xgboost": 0.3}

        result = train_all_models(df, "ERCOT", validation_hours=168)

        assert result["region"] == "ERCOT"
        assert "models" in result
        assert "metrics" in result
        assert "ensemble_weights" in result
        assert "validation_actual" in result

    @patch("models.training.train_prophet", side_effect=RuntimeError("Prophet failed"))
    @patch("models.training.train_arima")
    @patch("models.training.predict_arima")
    @patch("models.training.train_xgboost")
    @patch("models.training.predict_xgboost")
    @patch("models.training.compute_all_metrics")
    @patch("models.training.compute_ensemble_weights")
    def test_train_all_models_handles_prophet_failure(
        self,
        mock_weights,
        mock_metrics,
        mock_xgb_pred,
        mock_xgb_train,
        mock_arima_pred,
        mock_arima_train,
        mock_prophet_train,
    ):
        """train_all_models continues when Prophet training fails."""
        n = 500
        df = _make_small_feature_df(n)

        mock_arima_train.return_value = {
            "model": MagicMock(),
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 24),
            "exog_cols": [],
        }
        mock_arima_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_xgb_train.return_value = {
            "model": MagicMock(),
            "feature_names": ["a"],
            "feature_importances": {"a": 1.0},
            "cv_scores": [5.0],
        }
        mock_xgb_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_metrics.return_value = {"mape": 5.0, "rmse": 1000.0, "mae": 800.0, "r2": 0.9}
        mock_weights.return_value = {"arima": 0.5, "xgboost": 0.5}

        result = train_all_models(df, "ERCOT", validation_hours=168)

        # Prophet should have inf MAPE in metrics
        assert result["metrics"]["prophet"]["mape"] == float("inf")
        # Other models should still be trained
        assert "arima" in result["models"]
        assert "xgboost" in result["models"]

    @patch("models.training.train_prophet")
    @patch("models.training.predict_prophet")
    @patch("models.training.train_arima")
    @patch("models.training.predict_arima")
    @patch("models.training.train_xgboost")
    @patch("models.training.predict_xgboost")
    @patch("models.training.compute_all_metrics")
    @patch("models.training.compute_ensemble_weights")
    def test_train_all_models_calls_all_three(
        self,
        mock_weights,
        mock_metrics,
        mock_xgb_pred,
        mock_xgb_train,
        mock_arima_pred,
        mock_arima_train,
        mock_prophet_pred,
        mock_prophet_train,
    ):
        """train_all_models invokes all three model train functions."""
        n = 500
        df = _make_small_feature_df(n)

        mock_prophet_train.return_value = MagicMock()
        mock_prophet_pred.return_value = {"forecast": np.random.uniform(30000, 50000, 168)}
        mock_arima_train.return_value = {
            "model": MagicMock(),
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 24),
            "exog_cols": [],
        }
        mock_arima_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_xgb_train.return_value = {
            "model": MagicMock(),
            "feature_names": ["a"],
            "feature_importances": {"a": 1.0},
            "cv_scores": [5.0],
        }
        mock_xgb_pred.return_value = np.random.uniform(30000, 50000, 168)
        mock_metrics.return_value = {"mape": 5.0, "rmse": 1000.0, "mae": 800.0, "r2": 0.9}
        mock_weights.return_value = {"prophet": 0.33, "arima": 0.33, "xgboost": 0.34}

        train_all_models(df, "ERCOT", validation_hours=168)

        mock_prophet_train.assert_called_once()
        mock_arima_train.assert_called_once()
        mock_xgb_train.assert_called_once()

    def test_save_models_creates_file(self, tmp_path):
        """save_models writes a pickle file to the output directory."""
        training_result = {
            "region": "ERCOT",
            "models": {
                "xgboost": {"model": {"model": "fake_xgb_model", "feature_names": ["a"]}},
            },
            "metrics": {"xgboost": {"mape": 5.0}},
            "ensemble_weights": {"xgboost": 1.0},
        }
        filepath = save_models(training_result, output_dir=str(tmp_path))
        assert os.path.exists(filepath)
        assert filepath.endswith("ERCOT_models.pkl")

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_models + load_models roundtrip preserves region, weights, metrics."""
        training_result = {
            "region": "ERCOT",
            "models": {
                "xgboost": {"model": {"model": "fake_model", "feature_names": ["a", "b"]}},
            },
            "metrics": {"xgboost": {"mape": 5.0, "rmse": 1000.0, "mae": 800.0, "r2": 0.9}},
            "ensemble_weights": {"xgboost": 1.0},
        }
        save_models(training_result, output_dir=str(tmp_path))
        loaded = load_models("ERCOT", model_dir=str(tmp_path))

        assert loaded["region"] == "ERCOT"
        assert loaded["ensemble_weights"] == {"xgboost": 1.0}
        assert loaded["metrics"]["xgboost"]["mape"] == 5.0

    def test_load_models_raises_on_missing_file(self, tmp_path):
        """load_models raises FileNotFoundError when model file does not exist."""
        with pytest.raises(FileNotFoundError, match="No trained models"):
            load_models("ERCOT", model_dir=str(tmp_path))

    def test_load_models_validates_region(self, tmp_path):
        """load_models validates region before attempting to load."""
        with pytest.raises(ValueError, match="Invalid region format"):
            load_models("../../etc", model_dir=str(tmp_path))

    @patch("models.training.train_prophet", side_effect=RuntimeError("fail"))
    @patch("models.training.train_arima", side_effect=RuntimeError("fail"))
    @patch("models.training.train_xgboost", side_effect=RuntimeError("fail"))
    def test_train_all_models_all_fail_equal_weights(self, mock_xgb, mock_arima, mock_prophet):
        """When all models fail, ensemble uses equal weights."""
        df = _make_small_feature_df(500)
        result = train_all_models(df, "ERCOT", validation_hours=168)

        # All models failed, so metrics should all be inf
        for model_name in ["prophet", "arima", "xgboost"]:
            assert result["metrics"][model_name]["mape"] == float("inf")

        # With no finite MAPE, weights should be equal
        weights = result["ensemble_weights"]
        expected = 1.0 / 3
        for w in weights.values():
            assert w == pytest.approx(expected, abs=0.01)
