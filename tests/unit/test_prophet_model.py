"""Focused unit tests for ``models.prophet_model``.

Closes a coverage gap from issue #88. Prophet end-to-end is exercised
in ``test_ensemble.py`` / ``test_models_training.py``, but those rely
on real Prophet fitting (~20-30s per fit) which is too slow for the
unit tier. These tests target the module's own contracts — model
configuration, regressor list, demand-cap behavior, predict-output
shape — with the heavy ``Prophet`` class mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def feature_df():
    """Compact synthetic feature frame with all expected regressors.
    Length kept small (96 rows = 4 daily cycles); Prophet's actual
    fitting is mocked so length doesn't drive runtime."""
    n = 96
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 50_000.0 + 5_000.0 * np.sin(2 * np.pi * np.arange(n) / 24),
            "temperature_2m": 70.0,
            "apparent_temperature": 72.0,
            "wind_speed_10m": 5.0,
            "shortwave_radiation": 0.5,
            "cooling_degree_days": 5.0,
            "heating_degree_days": 0.0,
            "is_holiday": 0,
        }
    )


@pytest.fixture
def mock_prophet_class():
    """Patch the lazy ``_get_prophet`` resolver so ``create_prophet_model``
    returns a configurable mock. Avoids importing Prophet's heavy
    cmdstanpy dependency tree at all in the unit suite."""
    fake_prophet_class = MagicMock(name="FakeProphetClass")
    fake_instance = MagicMock(name="FakeProphetInstance")
    # Track which regressors were attached
    fake_instance._added_regressors = []

    def add_regressor(name, mode="additive"):
        fake_instance._added_regressors.append((name, mode))

    fake_instance.add_regressor.side_effect = add_regressor
    fake_prophet_class.return_value = fake_instance

    with patch("models.prophet_model._get_prophet", return_value=fake_prophet_class):
        yield fake_prophet_class, fake_instance


class TestCreateProphetModel:
    """The model constructor configures growth, seasonality, and
    attaches all regressors in ``PROPHET_REGRESSORS``. A drift in this
    set silently drops weather signal from the production model."""

    def test_creates_with_logistic_growth(self, mock_prophet_class):
        from models.prophet_model import create_prophet_model

        cls, _ = mock_prophet_class
        create_prophet_model()

        # Prophet was instantiated once; verify growth=logistic in kwargs.
        cls.assert_called_once()
        kwargs = cls.call_args.kwargs
        assert kwargs["growth"] == "logistic", (
            "logistic growth (with floor=0) is what structurally "
            "prevents negative forecasts — non-negotiable contract."
        )

    def test_attaches_all_seven_regressors(self, mock_prophet_class):
        from models.prophet_model import PROPHET_REGRESSORS, create_prophet_model

        _, instance = mock_prophet_class
        create_prophet_model()

        attached = {name for name, _ in instance._added_regressors}
        expected = {name for name, _ in PROPHET_REGRESSORS}
        assert attached == expected, f"Missing regressors: {expected - attached}"

    def test_all_regressors_are_additive_at_attach_time(self, mock_prophet_class):
        """Comment in source explains: multiplicative regressors on a
        logistic growth trend cause erratic extrapolation. Even though
        ``PROPHET_REGRESSORS`` lists some as multiplicative, the
        ``add_regressor`` call passes ``mode='additive'`` for all."""
        from models.prophet_model import create_prophet_model

        _, instance = mock_prophet_class
        create_prophet_model()

        modes = {mode for _, mode in instance._added_regressors}
        assert modes == {"additive"}, f"All regressors must attach as 'additive', got modes={modes}"


class TestTrainProphet:
    """Training stitches the input frame into Prophet's required ``ds``/
    ``y`` shape, computes a ``demand_cap`` from training-window max,
    and forwards regressor columns. The ``_demand_cap`` attribute on
    the returned model is read at predict time."""

    def test_demand_cap_is_one_point_five_x_max(self, feature_df, mock_prophet_class):
        from models.prophet_model import train_prophet

        _, instance = mock_prophet_class
        model = train_prophet(feature_df)

        expected_cap = float(feature_df["demand_mw"].max() * 1.5)
        assert model._demand_cap == pytest.approx(expected_cap, abs=0.01), (
            "demand_cap = 1.5x training max is the bound that prevents "
            "extrapolation drift on 7-30d horizons; the value flows "
            "through to the predict path's logistic-growth ceiling."
        )

    def test_calls_fit_with_ds_y_floor_cap_columns(self, feature_df, mock_prophet_class):
        from models.prophet_model import train_prophet

        _, instance = mock_prophet_class
        train_prophet(feature_df)

        instance.fit.assert_called_once()
        train_df = instance.fit.call_args.args[0]
        # Logistic growth requires both 'cap' and 'floor'.
        for col in ("ds", "y", "cap", "floor"):
            assert col in train_df.columns, (
                f"Prophet expects {col!r} on the training frame; "
                "logistic growth refuses to fit without cap/floor."
            )
        # Floor=0 is the structural negative-forecast bound.
        assert (train_df["floor"] == 0).all()

    def test_missing_regressor_logged_and_zero_filled(self, feature_df, mock_prophet_class):
        """If the upstream feature pipeline drops a regressor column,
        Prophet would raise on fit. The defensive zero-fill keeps the
        broader region-loop alive at the cost of dropping that
        regressor's signal — a per-region warning is emitted."""
        from models.prophet_model import train_prophet

        df_missing = feature_df.drop(columns=["is_holiday"])
        _, instance = mock_prophet_class
        train_prophet(df_missing)

        train_df = instance.fit.call_args.args[0]
        # Column was zero-filled rather than missing.
        assert "is_holiday" in train_df.columns
        assert (train_df["is_holiday"] == 0.0).all()

    def test_nan_in_regressor_is_filled_not_passed_to_fit(self, feature_df, mock_prophet_class):
        """A NaN in a present regressor (the ffill/bfill/zero defense
        added for the archive-unstable #164 columns still guards any
        regressor) must not reach ``fit`` — Prophet
        raises "Found NaN in column ...", which silently dropped Prophet
        from every region's holdout ensemble. The fit frame's regressors
        must be NaN-free after ffill/bfill/zero sanitation."""
        from models.prophet_model import PROPHET_REGRESSORS, train_prophet

        df_gappy = feature_df.copy()
        # Leading + interior NaNs (bfill covers the lead, ffill the interior).
        df_gappy.loc[df_gappy.index[:3], "wind_speed_10m"] = np.nan
        df_gappy.loc[df_gappy.index[40:45], "wind_speed_10m"] = np.nan

        _, instance = mock_prophet_class
        train_prophet(df_gappy)

        train_df = instance.fit.call_args.args[0]
        for regressor_name, _mode in PROPHET_REGRESSORS:
            assert not train_df[regressor_name].isna().any(), (
                f"regressor {regressor_name!r} reached Prophet.fit with NaN — "
                "Prophet would raise and drop this model from the ensemble."
            )

    def test_all_nan_regressor_degrades_to_zero(self, feature_df, mock_prophet_class):
        """An entirely-NaN regressor column can't be ff/bfilled; it must
        degrade to zero (signal dropped) rather than crash fit."""
        from models.prophet_model import train_prophet

        df_gappy = feature_df.copy()
        df_gappy["wind_speed_10m"] = np.nan

        _, instance = mock_prophet_class
        train_prophet(df_gappy)

        train_df = instance.fit.call_args.args[0]
        assert (train_df["wind_speed_10m"] == 0.0).all()


class TestPredictProphet:
    """The predict path returns a structured dict with point + interval
    forecasts plus timestamps. The interval columns drive the
    confidence band on the Forecast tab."""

    def test_predict_returns_expected_keys_and_lengths(self, feature_df, mock_prophet_class):
        from models.prophet_model import predict_prophet

        # Set up a fake fitted Prophet model whose .predict returns a
        # synthetic forecast frame — enough to exercise the predict
        # path's slicing + dict-build logic.
        _, instance = mock_prophet_class
        instance._demand_cap = 75_000.0
        # make_future_dataframe returns a frame with 'ds' column.
        future_n = 96 + 168
        future_ds = pd.date_range("2024-01-01", periods=future_n, freq="h")
        instance.make_future_dataframe.return_value = pd.DataFrame({"ds": future_ds})
        # Prophet.predict returns a frame with yhat / yhat_lower / yhat_upper.
        instance.predict.return_value = pd.DataFrame(
            {
                "ds": future_ds,
                "yhat": np.full(future_n, 50_000.0),
                "yhat_lower": np.full(future_n, 48_000.0),
                "yhat_upper": np.full(future_n, 52_000.0),
            }
        )

        result = predict_prophet(instance, feature_df, periods=168)

        for key in ("forecast", "lower_80", "upper_80", "timestamps"):
            assert key in result
        assert len(result["forecast"]) == 168
        assert len(result["timestamps"]) == 168

    def test_predict_emits_no_fabricated_95_interval(self, feature_df, mock_prophet_class):
        """#150: predict_prophet must NOT emit a fabricated 95% band.

        The old output scaled the real 80% bounds (``yhat_lower*0.95`` /
        ``yhat_upper*1.05``) and labelled it 95% — an uncalibrated heuristic.
        The honest output keeps only Prophet's genuine 80% posterior, which
        must bracket the point forecast.
        """
        from models.prophet_model import predict_prophet

        _, instance = mock_prophet_class
        instance._demand_cap = 75_000.0
        n = 96 + 24
        future_ds = pd.date_range("2024-01-01", periods=n, freq="h")
        instance.make_future_dataframe.return_value = pd.DataFrame({"ds": future_ds})
        instance.predict.return_value = pd.DataFrame(
            {
                "ds": future_ds,
                "yhat": np.full(n, 50_000.0),
                "yhat_lower": np.full(n, 45_000.0),
                "yhat_upper": np.full(n, 55_000.0),
            }
        )

        result = predict_prophet(instance, feature_df, periods=24)

        # No fabricated 95% claim survives anywhere in the output.
        assert "lower_95" not in result
        assert "upper_95" not in result
        # The honest 80% posterior brackets the point forecast.
        assert (result["upper_80"] >= result["forecast"]).all()
        assert (result["lower_80"] <= result["forecast"]).all()
