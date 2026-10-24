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
            "wind_speed_80m": 5.0,
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
            "logistic growth bounds the trend to [floor, cap]; note this bounds "
            "the TREND only, not the additive composite — non-negativity is "
            "enforced by the explicit clip in predict_prophet (#281)."
        )

    def test_yearly_seasonality_off_by_default(self, mock_prophet_class):
        """#281: yearly seasonality defaults OFF — it needs ≥2yr of history to
        identify a 365-day cycle; fitting it on the ~90d window drove a phantom
        multi-GW summer decline into negative demand."""
        from models.prophet_model import create_prophet_model

        cls, _ = mock_prophet_class
        create_prophet_model()
        assert cls.call_args.kwargs["yearly_seasonality"] is False


class TestYearlySeasonalityGate:
    """#281: train_prophet enables yearly seasonality only when the training
    span can actually identify a 365-day cycle."""

    def _df(self, days: int):
        import pandas as pd

        ts = pd.date_range("2024-01-01", periods=days, freq="D")
        return pd.DataFrame({"timestamp": ts, "demand_mw": 15000.0})

    def test_off_for_short_default_window(self, mock_prophet_class):
        from models.prophet_model import train_prophet

        cls, _ = mock_prophet_class
        train_prophet(self._df(90))  # ~90-day production window
        assert cls.call_args.kwargs["yearly_seasonality"] is False

    def test_on_for_multiyear_window(self, mock_prophet_class):
        from models.prophet_model import train_prophet

        cls, _ = mock_prophet_class
        train_prophet(self._df(760))  # > YEARLY_SEASONALITY_MIN_DAYS (730)
        assert cls.call_args.kwargs["yearly_seasonality"] is True

    def test_on_at_exactly_min_days_span(self, mock_prophet_class):
        """Boundary: span == YEARLY_SEASONALITY_MIN_DAYS (730) → yearly ON.

        span_days = periods - 1 (a daily range of N rows spans N-1 days), so
        731 daily rows give a 730-day span — exactly the gate. The far-from-
        boundary tests above (89/759) leave ``>=`` vs ``>`` and the constant's
        value unpinned; this case fails if the gate drifts to ``>`` or the
        constant moves up."""
        from models.prophet_model import train_prophet

        cls, _ = mock_prophet_class
        train_prophet(self._df(731))  # span == 730
        assert cls.call_args.kwargs["yearly_seasonality"] is True

    def test_off_just_below_min_days_span(self, mock_prophet_class):
        """Boundary: span == 729 (one day under the 730 gate) → yearly OFF.

        730 daily rows → 729-day span. Pins the other side of the ``>=``: a
        gate that drifted down (e.g. constant → 729) would wrongly turn yearly
        ON here."""
        from models.prophet_model import train_prophet

        cls, _ = mock_prophet_class
        train_prophet(self._df(730))  # span == 729
        assert cls.call_args.kwargs["yearly_seasonality"] is False


class TestPredictProphetFloorsNegative:
    """#281: predict_prophet clips the composite yhat (and its lower band) at 0 —
    Prophet's logistic floor bounds only the trend, so the additive composite can
    still go negative. Non-negativity must be enforced explicitly."""

    def test_negative_composite_is_clipped(self, mock_prophet_class):
        import pandas as pd

        from models.prophet_model import predict_prophet

        _, instance = mock_prophet_class
        ts = pd.date_range("2026-07-10", periods=3, freq="h")
        instance.make_future_dataframe.return_value = pd.DataFrame({"ds": ts})
        instance.predict.return_value = pd.DataFrame(
            {
                "ds": ts,
                "yhat": [-500.0, 12000.0, -20.0],  # negative composite despite floor=0
                "yhat_lower": [-800.0, 9000.0, -100.0],
                # A negative UPPER band is possible when the entire posterior
                # sits below zero (#281); upper_80's own np.maximum clip must
                # floor it too — an all-positive yhat_upper here would leave
                # that clip untested (a dropped clip would survive).
                "yhat_upper": [-50.0, 15000.0, 150.0],
            }
        )
        instance._demand_cap = 30000.0

        out = predict_prophet(instance, pd.DataFrame({"timestamp": ts}), periods=3)

        assert (out["forecast"] >= 0).all(), "negative demand must be clipped"
        assert (out["lower_80"] >= 0).all(), "negative lower band must be clipped"
        assert (out["upper_80"] >= 0).all(), "negative upper band must be clipped"
        assert out["forecast"][0] == 0.0 and out["forecast"][2] == 0.0
        assert out["upper_80"][0] == 0.0  # −50 upper band → 0
        assert out["forecast"][1] == 12000.0  # positive values untouched

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
        """A NaN in a present regressor (e.g. the archive-unstable
        ``wind_speed_80m``, #164/#176) must not reach ``fit`` — Prophet
        raises "Found NaN in column ...", which silently dropped Prophet
        from every region's holdout ensemble. The fit frame's regressors
        must be NaN-free after ffill/bfill/zero sanitation."""
        from models.prophet_model import PROPHET_REGRESSORS, train_prophet

        df_gappy = feature_df.copy()
        # Leading + interior NaNs (bfill covers the lead, ffill the interior).
        df_gappy.loc[df_gappy.index[:3], "wind_speed_80m"] = np.nan
        df_gappy.loc[df_gappy.index[40:45], "wind_speed_80m"] = np.nan

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
        df_gappy["wind_speed_80m"] = np.nan

        _, instance = mock_prophet_class
        train_prophet(df_gappy)

        train_df = instance.fit.call_args.args[0]
        assert (train_df["wind_speed_80m"] == 0.0).all()


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


class TestDegenerateRegressorsNotAttached:
    """A regressor whose training values barely vary has a coefficient the data
    cannot identify, and Prophet standardizes it by that near-zero std — so the
    first forecast-time value outside the training range is amplified into an
    arbitrary demand swing. Measured 2026-10-21: ERCOT's 2026-10-16 Prophet
    trained on 1 of 1,996 hours with any HDD (std 0.0095); HDD alone moved its
    08:00 forecast from 53.6 GW (HDD 0) to 237.6 GW (HDD 15), and the served
    ensemble reached 124-134 GW on the 10-17 cold front.

    Requirement: a non-binary regressor whose training std is below
    ``PROPHET_MIN_REGRESSOR_STD`` is not attached; binary regressors are exempt
    (Prophet does not standardize them, so there is nothing to amplify)."""

    def _df(self, **overrides):
        n = 240
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-07-25", periods=n, freq="h", tz="UTC"),
                "demand_mw": 50_000.0 + rng.normal(0, 1_000, n),
                "temperature_2m": 85.0 + rng.normal(0, 7, n),
                "apparent_temperature": 90.0 + rng.normal(0, 7, n),
                "wind_speed_80m": 10.0 + rng.normal(0, 3, n),
                "shortwave_radiation": 250.0 + rng.normal(0, 300, n),
                "cooling_degree_days": 20.0 + rng.normal(0, 7, n),
                "heating_degree_days": 0.0,
                "is_holiday": 0,
            }
        )
        for k, v in overrides.items():
            df[k] = v
        return df

    def _attached(self, instance):
        return {name for name, _ in instance._added_regressors}

    def test_one_hour_of_hdd_is_not_attached(self, mock_prophet_class):
        """The ERCOT 2026-10-16 shape: HDD is zero except one hour at 0.4."""
        from models.prophet_model import train_prophet

        _, instance = mock_prophet_class
        hdd = np.zeros(240)
        hdd[100] = 0.4
        train_prophet(self._df(heating_degree_days=hdd))
        assert "heating_degree_days" not in self._attached(instance)

    def test_varying_regressors_still_attached(self, mock_prophet_class):
        from models.prophet_model import train_prophet

        _, instance = mock_prophet_class
        train_prophet(self._df())
        assert {
            "temperature_2m",
            "apparent_temperature",
            "wind_speed_80m",
            "shortwave_radiation",
            "cooling_degree_days",
        } <= self._attached(instance)

    def test_binary_regressor_kept_even_when_rare(self, mock_prophet_class):
        """24 holiday hours in 240 has std ~0.3 — under the bar, but binary."""
        from models.prophet_model import train_prophet

        _, instance = mock_prophet_class
        hol = np.zeros(240, dtype=int)
        hol[:24] = 1
        train_prophet(self._df(is_holiday=hol))
        assert "is_holiday" in self._attached(instance)

    def test_boundary_at_min_std(self, mock_prophet_class):
        """At exactly the bar the regressor is attached; just below, it is not."""
        from models.prophet_model import PROPHET_MIN_REGRESSOR_STD, train_prophet

        base = np.tile([-1.0, 1.0], 120)  # sample std (ddof=1) of +/-1 alternation
        unit_std = float(pd.Series(base).std())

        _, instance = mock_prophet_class
        at_bar = 10.0 + base * (PROPHET_MIN_REGRESSOR_STD / unit_std)
        train_prophet(self._df(heating_degree_days=at_bar))
        assert "heating_degree_days" in self._attached(instance)

        instance._added_regressors.clear()
        below = 10.0 + base * (0.99 * PROPHET_MIN_REGRESSOR_STD / unit_std)
        train_prophet(self._df(heating_degree_days=below))
        assert "heating_degree_days" not in self._attached(instance)

    def test_flag_off_attaches_everything(self, mock_prophet_class):
        """Rollback path: with the flag off, every regressor is attached as before."""
        from models.prophet_model import PROPHET_REGRESSORS, train_prophet

        _, instance = mock_prophet_class
        hdd = np.zeros(240)
        hdd[100] = 0.4  # would be skipped with the flag on
        with patch.dict("config.FEATURE_FLAGS", {"prophet_skip_degenerate_regressors": False}):
            train_prophet(self._df(heating_degree_days=hdd))
        assert self._attached(instance) == {name for name, _ in PROPHET_REGRESSORS}

    def test_fit_frame_still_carries_every_regressor_column(self, mock_prophet_class):
        """Dropping a regressor must not drop its column — predict_prophet and
        the NaN-sanitation contract above both assume the full set exists."""
        from models.prophet_model import PROPHET_REGRESSORS, train_prophet

        _, instance = mock_prophet_class
        train_prophet(self._df())
        fit_df = instance.fit.call_args.args[0]
        for name, _ in PROPHET_REGRESSORS:
            assert name in fit_df.columns


@pytest.mark.slow
def test_real_prophet_does_not_blow_up_on_first_cold_hour():
    """End-to-end on real Prophet: summer-only training with one 0.4-HDD hour,
    then a forecast hour at HDD 15 (50 F). Before the fix this reproduced the
    ERCOT 2026-10-17 shape — a forecast several times the training max. The
    bar is the logistic cap (1.5x training max), which bounds only the trend,
    so a forecast above it can only come from a regressor term."""
    from models.prophet_model import predict_prophet, train_prophet

    n = 24 * 30
    rng = np.random.default_rng(1)
    ts = pd.date_range("2026-07-25", periods=n, freq="h", tz="UTC")
    hour = np.arange(n) % 24
    temp = 85.0 + 8.0 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2, n)
    temp = np.maximum(temp, 65.0)
    temp[300] = 64.6
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 50_000.0 + 600.0 * (temp - 85.0) + rng.normal(0, 800, n),
            "temperature_2m": temp,
            "apparent_temperature": temp + 3.0,
            "wind_speed_80m": 10.0 + rng.normal(0, 3, n),
            "shortwave_radiation": np.clip(800 * np.sin(2 * np.pi * (hour - 6) / 24), 0, None),
            "cooling_degree_days": np.maximum(temp - 65.0, 0),
            "heating_degree_days": np.maximum(65.0 - temp, 0),
            "is_holiday": 0,
        }
    )
    model = train_prophet(df)
    assert "heating_degree_days" not in model.extra_regressors

    fut_ts = pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    cold = pd.DataFrame(
        {
            "timestamp": fut_ts,
            "temperature_2m": 50.0,
            "apparent_temperature": 46.0,
            "wind_speed_80m": 10.0,
            "shortwave_radiation": 0.0,
            "cooling_degree_days": 0.0,
            "heating_degree_days": 15.0,
            "is_holiday": 0,
        }
    )
    out = predict_prophet(model, cold, periods=24)
    assert out["forecast"].max() <= 1.5 * df["demand_mw"].max(), (
        f"forecast {out['forecast'].max():.0f} MW exceeds the trend cap "
        f"{1.5 * df['demand_mw'].max():.0f} MW — a regressor term is extrapolating"
    )
