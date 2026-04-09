"""
Tests for helper functions in components/callbacks.py.

Covers ~500 statements across all module-level helpers:
- _empty_figure
- _compute_data_hash
- _confidence_half_width
- _add_confidence_bands
- _add_trailing_actuals
- _create_future_features
- _fetch_generation_cached
- _get_feature_importance
- _build_persona_kpis
- _predict_single_fold
- _ensemble_fold
- _run_backtest_for_horizon
- _run_forecast_outlook

No live API calls. All external dependencies are mocked.
"""

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demand_df():
    """Small demand DataFrame for testing helpers."""
    n = 720  # 30 days of hourly data
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    demand = 30000 + 5000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": demand,
            "region": "ERCOT",
        }
    )


@pytest.fixture
def weather_df():
    """Small weather DataFrame for testing helpers."""
    n = 720
    start = datetime(2024, 6, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": 85 + rng.normal(0, 5, n),
            "wind_speed_80m": 12 + rng.normal(0, 3, n),
            "shortwave_radiation": np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(n) / 24)),
            "relative_humidity_2m": 60 + rng.normal(0, 10, n),
            "cloud_cover": 50 + rng.normal(0, 15, n),
        }
    )


@pytest.fixture
def featured_train_df():
    """Training DataFrame with engineered features for _create_future_features tests."""
    n = 800
    start = datetime(2024, 3, 1, tzinfo=UTC)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(99)
    hours = np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 30000 + 5000 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 300, n),
            "region": "FPL",
            "temperature_2m": 80 + rng.normal(0, 5, n),
            "wind_speed_80m": 10 + rng.normal(0, 2, n),
            "shortwave_radiation": np.maximum(0, 500 * np.sin(2 * np.pi * hours / 24)),
            "cooling_degree_days": rng.uniform(5, 20, n),
            "heating_degree_days": rng.uniform(0, 2, n),
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "month": ts.month,
            "demand_lag_24h": rng.uniform(25000, 35000, n),
            "demand_roll_24h_mean": rng.uniform(28000, 32000, n),
        }
    )


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Clear all in-memory caches in callbacks before each test."""
    import components.callbacks as cb

    cb._MODEL_CACHE.clear()
    cb._PREDICTION_CACHE.clear()
    cb._BACKTEST_CACHE.clear()
    cb._GENERATION_CACHE.clear()
    yield
    cb._MODEL_CACHE.clear()
    cb._PREDICTION_CACHE.clear()
    cb._BACKTEST_CACHE.clear()
    cb._GENERATION_CACHE.clear()


# ===========================================================================
# _empty_figure
# ===========================================================================


class TestEmptyFigure:
    """Tests for _empty_figure()."""

    def test_returns_go_figure(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("Test message")
        assert isinstance(fig, go.Figure)

    def test_annotation_text(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("No data loaded")
        annotations = fig.layout.annotations
        assert len(annotations) == 1
        assert annotations[0].text == "No data loaded"

    def test_empty_message(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("")
        annotations = fig.layout.annotations
        assert annotations[0].text == ""

    def test_axes_hidden(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("hidden axes")
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False

    def test_annotation_centered(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("center")
        ann = fig.layout.annotations[0]
        assert ann.x == 0.5
        assert ann.y == 0.5
        assert ann.xref == "paper"
        assert ann.yref == "paper"

    def test_no_arrow(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("msg")
        assert fig.layout.annotations[0].showarrow is False

    def test_uses_plot_layout_template(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("t")
        assert fig.layout.template.layout is not None or str(fig.layout.template) != ""


# ===========================================================================
# _compute_data_hash
# ===========================================================================


class TestComputeDataHash:
    """Tests for _compute_data_hash()."""

    def test_same_input_same_hash(self, demand_df, weather_df):
        from components.callbacks import _compute_data_hash

        h1 = _compute_data_hash(demand_df, weather_df, "ERCOT")
        h2 = _compute_data_hash(demand_df, weather_df, "ERCOT")
        assert h1 == h2

    def test_different_region_different_hash(self, demand_df, weather_df):
        from components.callbacks import _compute_data_hash

        h1 = _compute_data_hash(demand_df, weather_df, "ERCOT")
        h2 = _compute_data_hash(demand_df, weather_df, "CAISO")
        assert h1 != h2

    def test_different_data_different_hash(self, weather_df):
        from components.callbacks import _compute_data_hash

        ts1 = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        ts2 = pd.date_range("2024-02-01", periods=48, freq="h", tz="UTC")
        df1 = pd.DataFrame({"timestamp": ts1, "demand_mw": np.ones(48)})
        df2 = pd.DataFrame({"timestamp": ts2, "demand_mw": np.ones(48)})
        h1 = _compute_data_hash(df1, weather_df, "FPL")
        h2 = _compute_data_hash(df2, weather_df, "FPL")
        assert h1 != h2

    def test_empty_dataframes(self):
        from components.callbacks import _compute_data_hash

        empty = pd.DataFrame(columns=["timestamp", "demand_mw"])
        h = _compute_data_hash(empty, empty, "PJM")
        assert isinstance(h, str)

    def test_no_timestamp_column(self):
        from components.callbacks import _compute_data_hash

        df = pd.DataFrame({"value": [1, 2, 3]})
        h = _compute_data_hash(df, df, "MISO")
        assert isinstance(h, str)

    def test_hash_returns_int(self, demand_df, weather_df):
        from components.callbacks import _compute_data_hash

        h = _compute_data_hash(demand_df, weather_df, "FPL")
        assert isinstance(h, str)

    @pytest.mark.parametrize(
        "region", ["ERCOT", "CAISO", "PJM", "MISO", "NYISO", "FPL", "SPP", "ISONE"]
    )
    def test_all_regions_produce_valid_hash(self, demand_df, weather_df, region):
        from components.callbacks import _compute_data_hash

        h = _compute_data_hash(demand_df, weather_df, region)
        assert isinstance(h, str)

    def test_tz_naive_vs_tz_aware_same_hash(self):
        """Timestamps with and without tz info should hash identically (by design)."""
        from components.callbacks import _compute_data_hash

        ts_aware = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        ts_naive = ts_aware.tz_localize(None)
        df_aware = pd.DataFrame({"timestamp": ts_aware, "demand_mw": np.ones(24)})
        df_naive = pd.DataFrame({"timestamp": ts_naive, "demand_mw": np.ones(24)})
        weather = pd.DataFrame({"timestamp": ts_aware, "temp": np.ones(24)})
        weather_naive = pd.DataFrame({"timestamp": ts_naive, "temp": np.ones(24)})
        h1 = _compute_data_hash(df_aware, weather, "ERCOT")
        h2 = _compute_data_hash(df_naive, weather_naive, "ERCOT")
        assert h1 == h2


# ===========================================================================
# _confidence_half_width
# ===========================================================================


class TestConfidenceHalfWidth:
    """Tests for _confidence_half_width()."""

    def test_24h_horizon(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(24) == 0.03

    def test_short_horizon(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(1) == 0.03
        assert _confidence_half_width(12) == 0.03

    def test_medium_horizon_168h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(168) == 0.06

    def test_medium_horizon_48h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(48) == 0.06

    def test_long_horizon_720h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(720) == 0.10

    def test_boundary_25h(self):
        """25 hours is > 24, should return 0.06."""
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(25) == 0.06

    def test_boundary_169h(self):
        """169 hours is > 168, should return 0.10."""
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(169) == 0.10

    @pytest.mark.parametrize(
        "horizon,expected",
        [
            (1, 0.03),
            (6, 0.03),
            (24, 0.03),
            (25, 0.06),
            (72, 0.06),
            (168, 0.06),
            (169, 0.10),
            (336, 0.10),
            (720, 0.10),
        ],
    )
    def test_all_thresholds(self, horizon, expected):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(horizon) == expected


# ===========================================================================
# _add_confidence_bands
# ===========================================================================


class TestAddConfidenceBands:
    """Tests for _add_confidence_bands()."""

    def test_adds_two_traces(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-01-01", periods=24, freq="h")
        preds = np.full(24, 30000.0)
        initial_count = len(fig.data)
        _add_confidence_bands(fig, ts, preds, 24)
        assert len(fig.data) == initial_count + 2

    def test_upper_lower_values(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-01-01", periods=10, freq="h")
        preds = np.full(10, 10000.0)
        _add_confidence_bands(fig, ts, preds, 24)
        upper_trace = fig.data[0]
        lower_trace = fig.data[1]
        # hw=0.03 for 24h
        np.testing.assert_allclose(upper_trace.y, 10300.0, rtol=1e-6)
        np.testing.assert_allclose(lower_trace.y, 9700.0, rtol=1e-6)

    def test_fill_tonexty(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        preds = np.ones(5) * 5000
        _add_confidence_bands(fig, ts, preds, 48)
        lower_trace = fig.data[1]
        assert lower_trace.fill == "tonexty"

    def test_upper_no_legend(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        preds = np.ones(5) * 5000
        _add_confidence_bands(fig, ts, preds, 24)
        assert fig.data[0].showlegend is False

    def test_lower_named_80ci(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        preds = np.ones(5) * 5000
        _add_confidence_bands(fig, ts, preds, 24)
        assert fig.data[1].name == "80% CI"

    def test_wider_bands_at_longer_horizon(self):
        from components.callbacks import _add_confidence_bands

        fig24 = go.Figure()
        fig720 = go.Figure()
        ts = pd.date_range("2024-01-01", periods=5, freq="h")
        preds = np.full(5, 20000.0)
        _add_confidence_bands(fig24, ts, preds, 24)
        _add_confidence_bands(fig720, ts, preds, 720)
        # upper band should be wider at 720h
        upper_24 = fig24.data[0].y[0]
        upper_720 = fig720.data[0].y[0]
        assert upper_720 > upper_24


# ===========================================================================
# _add_trailing_actuals
# ===========================================================================


class TestAddTrailingActuals:
    """Tests for _add_trailing_actuals()."""

    def test_adds_trace_with_valid_json(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
                "demand_mw": np.random.default_rng(1).normal(30000, 500, 100),
            }
        )
        json_str = df.to_json(date_format="iso")
        _add_trailing_actuals(fig, json_str, tail_hours=48)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Actual"

    def test_none_json_no_trace(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, None)
        assert len(fig.data) == 0

    def test_empty_string_no_trace(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, "")
        assert len(fig.data) == 0

    def test_invalid_json_no_crash(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, "{bad json!!!")
        assert len(fig.data) == 0  # silently fails

    def test_tail_hours_limits_data(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
                "demand_mw": np.ones(200) * 25000,
            }
        )
        json_str = df.to_json(date_format="iso")
        _add_trailing_actuals(fig, json_str, tail_hours=24)
        # The trace should have at most 24 points
        assert len(fig.data[0].y) == 24

    def test_trace_style(self):
        from components.callbacks import COLORS, _add_trailing_actuals

        fig = go.Figure()
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
                "demand_mw": np.ones(50) * 20000,
            }
        )
        json_str = df.to_json(date_format="iso")
        _add_trailing_actuals(fig, json_str)
        trace = fig.data[0]
        assert trace.line.color == COLORS["actual"]
        assert trace.line.dash == "dot"


# ===========================================================================
# _create_future_features
# ===========================================================================


class TestCreateFutureFeatures:
    """Tests for _create_future_features()."""

    def test_output_shape_short_horizon(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=48, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert len(result) == 48

    def test_output_shape_long_horizon(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=720, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert len(result) == 720

    def test_has_timestamp_column(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert "timestamp" in result.columns

    def test_has_time_features(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        expected_cols = [
            "hour",
            "day_of_week",
            "month",
            "day_of_year",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing time feature: {col}"

    def test_hour_encoding_correct(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=48, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        # hour should match timestamp
        expected_hours = future_ts.hour
        np.testing.assert_array_equal(result["hour"].values, expected_hours)

    def test_dow_encoding_correct(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=48, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        expected_dow = future_ts.dayofweek
        np.testing.assert_array_equal(result["day_of_week"].values, expected_dow)

    def test_is_weekend_correct(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=168, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        expected = (future_ts.dayofweek >= 5).astype(int)
        np.testing.assert_array_equal(result["is_weekend"].values, expected)

    def test_month_encoding_correct(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        expected_months = future_ts.month
        np.testing.assert_array_equal(result["month"].values, expected_months)

    def test_short_horizon_uses_last_row(self, featured_train_df):
        """For short horizons (<168), non-time features should use last training row values."""
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        last_row = featured_train_df.iloc[-1]
        # temperature_2m should be constant from last row (for short horizon)
        assert result["temperature_2m"].iloc[0] == pytest.approx(last_row["temperature_2m"])

    def test_long_horizon_uses_group_means(self, featured_train_df):
        """For long horizons (>=168), non-time features should vary by (hour, dow)."""
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=720, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        # For long horizon, temperatures should NOT all be the same
        temps = result["temperature_2m"].values
        assert len(np.unique(np.round(temps, 2))) > 1

    def test_no_demand_mw_in_output(self, featured_train_df):
        """demand_mw should not be a feature column in future features."""
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert "demand_mw" not in result.columns

    def test_no_region_in_output(self, featured_train_df):
        """region should not be a feature column in future features."""
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert "region" not in result.columns

    def test_hour_sin_cos_bounds(self, featured_train_df):
        from components.callbacks import _create_future_features

        last_ts = featured_train_df["timestamp"].max()
        future_ts = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1), periods=48, freq="h", tz="UTC"
        )
        result = _create_future_features(featured_train_df, future_ts)
        assert result["hour_sin"].between(-1.0, 1.0).all()
        assert result["hour_cos"].between(-1.0, 1.0).all()
        assert result["dow_sin"].between(-1.0, 1.0).all()
        assert result["dow_cos"].between(-1.0, 1.0).all()


# ===========================================================================
# _fetch_generation_cached
# ===========================================================================


class TestFetchGenerationCached:
    """Tests for _fetch_generation_cached()."""

    def test_memory_cache_hit(self):
        import components.callbacks as cb
        from components.callbacks import _fetch_generation_cached

        fake_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "fuel_type": ["gas"] * 24,
                "generation_mw": np.ones(24) * 5000,
            }
        )
        cb._GENERATION_CACHE["ERCOT"] = (fake_df, time.time())
        result = _fetch_generation_cached("ERCOT")
        assert result is not None
        pd.testing.assert_frame_equal(result, fake_df)

    def test_memory_cache_expired(self):
        """If the memory cache is older than 300s, it should be bypassed."""
        import components.callbacks as cb
        from components.callbacks import _fetch_generation_cached

        fake_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "fuel_type": ["gas"] * 24,
                "generation_mw": np.ones(24) * 5000,
            }
        )
        # Set cache timestamp to 10 minutes ago
        cb._GENERATION_CACHE["ERCOT"] = (fake_df, time.time() - 600)

        # With no EIA_API_KEY, it should fall through to demo data
        with patch("components.callbacks.EIA_API_KEY", ""):
            result = _fetch_generation_cached("ERCOT")
            assert result is not None  # should get demo data

    @patch("config.EIA_API_KEY", "test-key")
    def test_eia_api_path(self):
        from components.callbacks import _fetch_generation_cached

        gen_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=24, freq="h"),
                "fuel_type": ["NG"] * 24,
                "generation_mw": np.ones(24) * 3000,
            }
        )
        with patch("data.eia_client.fetch_generation_by_fuel", return_value=gen_df):
            result = _fetch_generation_cached("CAISO")
            assert result is not None
            # fuel type should be normalized
            assert (result["fuel_type"] == "gas").all()

    @patch("config.EIA_API_KEY", "test-key")
    def test_eia_api_failure_falls_to_demo(self):
        from components.callbacks import _fetch_generation_cached

        with patch("data.eia_client.fetch_generation_by_fuel", side_effect=Exception("API down")):
            result = _fetch_generation_cached("PJM")
            assert result is not None  # demo fallback
            assert "fuel_type" in result.columns

    @patch("config.EIA_API_KEY", "")
    def test_no_api_key_uses_demo(self):
        from components.callbacks import _fetch_generation_cached

        result = _fetch_generation_cached("FPL")
        assert result is not None
        assert len(result) > 0

    @patch("config.EIA_API_KEY", "")
    def test_demo_fallback_populates_memory_cache(self):
        import components.callbacks as cb
        from components.callbacks import _fetch_generation_cached

        _fetch_generation_cached("SPP")
        assert "SPP" in cb._GENERATION_CACHE

    def test_demo_failure_returns_none(self):
        from components.callbacks import _fetch_generation_cached

        with (
            patch("config.EIA_API_KEY", ""),
            patch("data.demo_data.generate_demo_generation", side_effect=Exception("demo broken")),
        ):
            result = _fetch_generation_cached("ISONE")
            assert result is None

    @patch("config.EIA_API_KEY", "test-key")
    def test_eia_returns_empty_df_falls_to_demo(self):
        from components.callbacks import _fetch_generation_cached

        empty = pd.DataFrame()
        with patch("data.eia_client.fetch_generation_by_fuel", return_value=empty):
            result = _fetch_generation_cached("MISO")
            assert result is not None  # falls through to demo

    @patch("config.EIA_API_KEY", "test-key")
    def test_eia_returns_none_falls_to_demo(self):
        from components.callbacks import _fetch_generation_cached

        with patch("data.eia_client.fetch_generation_by_fuel", return_value=None):
            result = _fetch_generation_cached("NYISO")
            assert result is not None  # demo fallback


# ===========================================================================
# _get_feature_importance
# ===========================================================================


class TestGetFeatureImportance:
    """Tests for _get_feature_importance()."""

    def test_default_when_no_cache(self):
        from components.callbacks import _get_feature_importance

        names, vals = _get_feature_importance("ERCOT")
        assert len(names) == 10
        assert len(vals) == 10
        assert "temperature_2m" in names

    def test_default_values_sum_to_one(self):
        from components.callbacks import _get_feature_importance

        _, vals = _get_feature_importance("ERCOT")
        assert abs(vals.sum() - 1.0) < 1e-6

    def test_cached_model_with_importances(self):
        import components.callbacks as cb
        from components.callbacks import _get_feature_importance

        model_dict = {
            "feature_importances": {
                "temperature_2m": 0.4,
                "demand_lag_24h": 0.3,
                "hour_sin": 0.2,
                "wind_speed_80m": 0.1,
            }
        }
        cb._MODEL_CACHE[("FPL", "xgboost", 0)] = (model_dict, 0, time.time())
        names, vals = _get_feature_importance("FPL", top_n=3)
        assert len(names) == 3
        assert names[0] == "temperature_2m"  # highest importance first

    def test_cached_model_all_zero_importances_returns_defaults(self):
        import components.callbacks as cb
        from components.callbacks import _get_feature_importance

        model_dict = {
            "feature_importances": {
                "a": 0.0,
                "b": 0.0,
            }
        }
        cb._MODEL_CACHE[("CAISO", "xgboost", 0)] = (model_dict, 0, time.time())
        names, vals = _get_feature_importance("CAISO")
        # Should return defaults since vals.sum() == 0
        assert len(names) == 10

    def test_cached_model_not_dict(self):
        """If cached model is not a dict with 'feature_importances', return defaults."""
        import components.callbacks as cb
        from components.callbacks import _get_feature_importance

        cb._MODEL_CACHE[("PJM", "xgboost", 0)] = ("not_a_dict", 0, time.time())
        names, vals = _get_feature_importance("PJM")
        assert len(names) == 10

    def test_top_n_parameter(self):
        import components.callbacks as cb
        from components.callbacks import _get_feature_importance

        model_dict = {
            "feature_importances": {
                "a": 0.5,
                "b": 0.3,
                "c": 0.1,
                "d": 0.05,
                "e": 0.05,
            }
        }
        cb._MODEL_CACHE[("MISO", "xgboost", 0)] = (model_dict, 0, time.time())
        names, vals = _get_feature_importance("MISO", top_n=2)
        assert len(names) == 2
        assert names[0] == "a"


# ===========================================================================
# _build_persona_kpis
# ===========================================================================


class TestBuildPersonaKpis:
    """Tests for _build_persona_kpis()."""

    @patch("components.callbacks.redis_get", return_value=None)
    def test_grid_ops_with_data(self, mock_redis, demand_df, weather_df):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("grid_ops", "ERCOT", demand_df, weather_df)
        assert result is not None
        # build_kpi_row returns dbc.Row
        import dash_bootstrap_components as dbc

        assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    def test_renewables_with_data(self, mock_redis, demand_df, weather_df):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("renewables", "CAISO", demand_df, weather_df)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_trader_with_data(self, mock_redis, demand_df, weather_df):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("trader", "PJM", demand_df, weather_df)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_data_scientist_with_data(self, mock_redis, demand_df, weather_df):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("data_scientist", "FPL", demand_df, weather_df)
        assert result is not None

    @pytest.mark.parametrize("persona", ["grid_ops", "renewables", "trader", "data_scientist"])
    @patch("components.callbacks.redis_get", return_value=None)
    def test_all_personas_no_data(self, mock_redis, persona):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis(persona, "ERCOT", None, None)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_unknown_persona_falls_to_grid_ops(self, mock_redis, demand_df):
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("unknown_persona", "ERCOT", demand_df, None)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_empty_demand_df(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        empty_df = pd.DataFrame(columns=["timestamp", "demand_mw"])
        result = _build_persona_kpis("grid_ops", "FPL", empty_df, None)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_demand_df_with_nans(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": [np.nan] * 48,
            }
        )
        result = _build_persona_kpis("grid_ops", "ERCOT", df, None)
        assert result is not None

    @patch("components.callbacks.redis_get")
    def test_redis_fallback_for_demand(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        def fake_redis_get(key):
            if "actuals" in key:
                return {"demand_mw": [30000, 35000, 28000], "timestamps": ["t1", "t2", "t3"]}
            if "weather" in key:
                return {"wind_speed_80m": [10, 15, 12], "shortwave_radiation": [300, 500, 400]}
            return None

        mock_redis.side_effect = fake_redis_get
        result = _build_persona_kpis("grid_ops", "FPL", None, None)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_backtest_cache_used_for_mape(self, mock_redis, demand_df):
        import components.callbacks as cb
        from components.callbacks import _build_persona_kpis

        bt_result = {"metrics": {"mape": 3.5, "rmse": 800}}
        cb._BACKTEST_CACHE[("ERCOT", 168, "xgboost", "forecast_exog")] = (
            bt_result,
            0,
            time.time(),
        )
        result = _build_persona_kpis("grid_ops", "ERCOT", demand_df, None)
        assert result is not None

    @patch("components.callbacks.redis_get", return_value=None)
    def test_high_utilization_price(self, mock_redis):
        """When peak demand is close to capacity, price should be elevated."""
        from components.callbacks import _build_persona_kpis

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "demand_mw": np.full(48, 125000.0),  # ~96% of ERCOT 130k capacity
            }
        )
        result = _build_persona_kpis("trader", "ERCOT", df, None)
        assert result is not None


# ===========================================================================
# _predict_single_fold
# ===========================================================================


class TestPredictSingleFold:
    """Tests for _predict_single_fold()."""

    def test_xgboost_fold(self):
        from components.callbacks import _predict_single_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, 500),
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-22", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(2).normal(30000, 1000, 24),
            }
        )
        mock_model = {"model": "xgb_mock"}
        preds = np.random.default_rng(3).normal(30000, 500, 50)

        with (
            patch("models.xgboost_model.train_xgboost", return_value=mock_model) as m_train,
            patch("models.xgboost_model.predict_xgboost", return_value=preds) as m_pred,
        ):
            result = _predict_single_fold("xgboost", train, test)
            m_train.assert_called_once_with(train)
            # Stepwise autoregressive inference calls predict once per forecast step.
            assert m_pred.call_count == len(test)
            for call_args in m_pred.call_args_list:
                assert call_args.args[0] == mock_model
                assert len(call_args.args[1]) == 1
            assert len(result) == 24  # clipped to n_test

    def test_prophet_fold(self):
        from components.callbacks import _predict_single_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, 500),
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-22", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(2).normal(30000, 1000, 24),
            }
        )
        mock_model = MagicMock()
        prophet_result = {"forecast": np.random.default_rng(4).normal(30000, 500, 100)}

        with (
            patch("models.prophet_model.train_prophet", return_value=mock_model),
            patch("models.prophet_model.predict_prophet", return_value=prophet_result),
        ):
            result = _predict_single_fold("prophet", train, test)
            assert len(result) == 24

    def test_arima_fold(self):
        from components.callbacks import _predict_single_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, 500),
                "temperature_2m": np.ones(500) * 85,
                "wind_speed_80m": np.ones(500) * 10,
                "shortwave_radiation": np.ones(500) * 400,
                "cooling_degree_days": np.ones(500) * 15,
                "heating_degree_days": np.ones(500) * 0,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-22", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.random.default_rng(2).normal(30000, 1000, 24),
                "temperature_2m": np.ones(24) * 85,
                "wind_speed_80m": np.ones(24) * 10,
                "shortwave_radiation": np.ones(24) * 400,
                "cooling_degree_days": np.ones(24) * 15,
                "heating_degree_days": np.ones(24) * 0,
            }
        )
        mock_model = MagicMock()
        arima_preds = np.random.default_rng(5).normal(30000, 500, 50)

        with (
            patch("models.arima_model.train_arima", return_value=mock_model),
            patch("models.arima_model.predict_arima", return_value=arima_preds),
        ):
            result = _predict_single_fold("arima", train, test)
            assert len(result) == 24

    def test_unknown_model_returns_none(self):
        from components.callbacks import _predict_single_fold

        train = pd.DataFrame({"timestamp": [], "demand_mw": []})
        test = pd.DataFrame({"timestamp": [], "demand_mw": []})
        result = _predict_single_fold("unknown_model", train, test)
        assert result is None

    def test_arima_fills_nan_exog(self):
        """ARIMA path should forward-fill NaN in exogenous columns."""
        from components.callbacks import _predict_single_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
                "demand_mw": np.ones(100) * 30000,
                "temperature_2m": np.ones(100) * 80,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-05", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
                "temperature_2m": [np.nan] * 12 + [80.0] * 12,
            }
        )
        mock_model = MagicMock()
        arima_preds = np.ones(30) * 30000

        with (
            patch("models.arima_model.train_arima", return_value=mock_model),
            patch("models.arima_model.predict_arima", return_value=arima_preds) as m_pred,
        ):
            result = _predict_single_fold("arima", train, test)
            assert result is not None
            # Verify the test_clean passed to predict_arima has no NaNs in temperature
            call_args = m_pred.call_args[0]
            test_passed = call_args[1]
            assert not test_passed["temperature_2m"].isna().any()


# ===========================================================================
# _ensemble_fold
# ===========================================================================


class TestEnsembleFold:
    """Tests for _ensemble_fold()."""

    def test_combines_three_models(self):
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )
        xgb_preds = np.ones(24) * 29000
        prophet_preds = np.ones(24) * 31000
        arima_preds = np.ones(24) * 30500

        def mock_predict(name, train_df, test_df, **kwargs):
            return {"xgboost": xgb_preds, "prophet": prophet_preds, "arima": arima_preds}[name]

        with (
            patch("components.callbacks._predict_single_fold", side_effect=mock_predict),
            patch("models.evaluation.compute_mape", return_value=5.0),
        ):
            result = _ensemble_fold(train, test)
            assert result is not None
            assert len(result) == 24

    def test_all_models_fail_returns_none(self):
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )

        with patch("components.callbacks._predict_single_fold", return_value=None):
            result = _ensemble_fold(train, test)
            assert result is None

    def test_one_model_fails_still_works(self):
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )

        call_count = 0

        def mock_predict(name, train_df, test_df, **kwargs):
            nonlocal call_count
            call_count += 1
            if name == "prophet":
                return None  # prophet fails
            return np.ones(24) * 30000

        with (
            patch("components.callbacks._predict_single_fold", side_effect=mock_predict),
            patch("models.evaluation.compute_mape", return_value=3.0),
        ):
            result = _ensemble_fold(train, test)
            assert result is not None
            assert len(result) == 24

    def test_mape_weighting_correct(self):
        """Model with lower MAPE should get higher weight."""
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )

        xgb_pred = np.ones(24) * 29000  # closer to actual
        prophet_pred = np.ones(24) * 35000  # farther from actual

        def mock_predict(name, train_df, test_df, **kwargs):
            if name == "xgboost":
                return xgb_pred
            if name == "prophet":
                return prophet_pred
            return None  # arima fails

        # xgboost: MAPE~3.3, prophet: MAPE~16.7
        # xgboost should have higher weight → ensemble closer to 29000
        with patch("components.callbacks._predict_single_fold", side_effect=mock_predict):
            result = _ensemble_fold(train, test)
            assert result is not None
            avg = result.mean()
            # Ensemble should be closer to xgboost (29000) than prophet (35000)
            assert avg < 32000

    def test_nan_predictions_excluded(self):
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )

        def mock_predict(name, train_df, test_df, **kwargs):
            if name == "xgboost":
                return np.full(24, np.nan)  # all NaN
            if name == "prophet":
                return np.ones(24) * 30000
            return np.ones(24) * 31000

        with (
            patch("components.callbacks._predict_single_fold", side_effect=mock_predict),
            patch("models.evaluation.compute_mape", return_value=3.0),
        ):
            result = _ensemble_fold(train, test)
            assert result is not None

    def test_zero_mape_fallback_to_uniform(self):
        """If all MAPE values are 0 (perfect fit), should fall back to uniform average."""
        from components.callbacks import _ensemble_fold

        train = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
                "demand_mw": np.ones(200) * 30000,
            }
        )
        test = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-09", periods=24, freq="h", tz="UTC"),
                "demand_mw": np.ones(24) * 30000,
            }
        )

        xgb = np.ones(24) * 30000
        prophet = np.ones(24) * 30000

        def mock_predict(name, train_df, test_df, **kwargs):
            if name == "xgboost":
                return xgb
            if name == "prophet":
                return prophet
            return None

        # MAPE=0 for perfect predictions → 1/MAPE is infinite → fallback to uniform
        with (
            patch("components.callbacks._predict_single_fold", side_effect=mock_predict),
            patch("models.evaluation.compute_mape", return_value=0.0),
        ):
            result = _ensemble_fold(train, test)
            assert result is not None
            # uniform average of 30000 and 30000 = 30000
            np.testing.assert_allclose(result, 30000.0, rtol=1e-6)


# ===========================================================================
# _run_backtest_for_horizon
# ===========================================================================


class TestRunBacktestForHorizon:
    """Tests for _run_backtest_for_horizon()."""

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    @patch("models.evaluation.compute_all_metrics")
    def test_basic_backtest_result_shape(
        self,
        mock_metrics,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_backtest_for_horizon

        n = 1200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "temperature_2m": np.ones(n) * 80,
                "hour": ts.hour,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured
        mock_metrics.return_value = {"mape": 4.5, "rmse": 900, "mae": 700, "r2": 0.92}

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch("components.callbacks._predict_single_fold", return_value=np.ones(24) * 30000):
            result = _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")

        assert "predictions" in result
        assert "actual" in result
        assert "timestamps" in result
        assert "metrics" in result
        assert "num_folds" in result
        assert "fold_boundaries" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_insufficient_data_returns_error(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_backtest_for_horizon

        # Only 100 rows -- insufficient for min_train_size=720 + horizon=24
        featured = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
                "demand_mw": np.ones(100) * 30000,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        result = _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "error" in result

    def test_memory_cache_hit(self, demand_df, weather_df):
        import components.callbacks as cb
        from components.callbacks import _compute_data_hash, _run_backtest_for_horizon

        data_hash = _compute_data_hash(demand_df, weather_df, "ERCOT")
        cached_result = {
            "predictions": np.ones(24),
            "actual": np.ones(24),
            "timestamps": np.arange(24),
            "metrics": {"mape": 3.0},
            "num_folds": 1,
            "fold_boundaries": [0],
        }
        cb._BACKTEST_CACHE[("ERCOT", 24, "xgboost", "forecast_exog")] = (
            cached_result,
            data_hash,
            time.time(),
        )

        result = _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert result["metrics"]["mape"] == 3.0

    @patch("data.cache.get_cache")
    def test_sqlite_cache_hit(self, mock_get_cache, demand_df, weather_df):
        from components.callbacks import (
            _CACHE_VERSION,
            _compute_data_hash,
            _run_backtest_for_horizon,
        )

        data_hash = _compute_data_hash(demand_df, weather_df, "ERCOT")
        sqlite_result = {
            "cache_version": _CACHE_VERSION,
            "data_hash": data_hash,
            "actual": [30000, 31000, 29000],
            "predictions": [30100, 30900, 29100],
            "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "metrics": {"mape": 2.5, "rmse": 500},
            "num_folds": 1,
            "fold_boundaries": [0],
        }
        mock_cache = MagicMock()
        mock_cache.get.return_value = sqlite_result
        mock_get_cache.return_value = mock_cache

        result = _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert isinstance(result["actual"], np.ndarray)
        assert isinstance(result["predictions"], np.ndarray)

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    @patch("models.evaluation.compute_all_metrics")
    def test_ensemble_backtest(
        self,
        mock_metrics,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_backtest_for_horizon

        n = 1200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured
        mock_metrics.return_value = {"mape": 4.0, "rmse": 800, "mae": 600, "r2": 0.93}

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch("components.callbacks._ensemble_fold", return_value=np.ones(24) * 30000):
            result = _run_backtest_for_horizon(demand_df, weather_df, 24, "ensemble", "ERCOT")

        assert "metrics" in result
        assert result["metrics"]["mape"] == 4.0

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    @patch("models.evaluation.compute_all_metrics")
    def test_backtest_caches_result(
        self,
        mock_metrics,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        """After a successful backtest, result should be in _BACKTEST_CACHE."""
        import components.callbacks as cb
        from components.callbacks import _run_backtest_for_horizon

        n = 1200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured
        mock_metrics.return_value = {"mape": 5.0, "rmse": 1000, "mae": 800, "r2": 0.90}

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch("components.callbacks._predict_single_fold", return_value=np.ones(24) * 30000):
            _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")

        assert ("ERCOT", 24, "xgboost", "forecast_exog") in cb._BACKTEST_CACHE

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_all_folds_fail_returns_error(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_backtest_for_horizon

        n = 1200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch("components.callbacks._predict_single_fold", return_value=None):
            result = _run_backtest_for_horizon(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "error" in result


# ===========================================================================
# _run_forecast_outlook
# ===========================================================================


class TestRunForecastOutlook:
    """Tests for _run_forecast_outlook()."""

    def test_prediction_cache_hit(self, demand_df, weather_df):
        import components.callbacks as cb
        from components.callbacks import _compute_data_hash, _run_forecast_outlook

        data_hash = _compute_data_hash(demand_df, weather_df, "ERCOT")
        future_ts = pd.date_range("2024-07-01", periods=24, freq="h", tz="UTC")
        cached_preds = np.ones(24) * 32000

        cb._PREDICTION_CACHE[("ERCOT", 24, "xgboost")] = (
            cached_preds,
            future_ts,
            data_hash,
            time.time(),
        )

        result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "predictions" in result
        np.testing.assert_array_equal(result["predictions"], cached_preds)

    @patch("data.cache.get_cache")
    def test_sqlite_cache_hit(self, mock_get_cache, demand_df, weather_df):
        from components.callbacks import _run_forecast_outlook

        sqlite_result = {
            "predictions": [30000, 31000],
            "timestamps": ["2024-07-01 01:00:00", "2024-07-01 02:00:00"],
        }
        mock_cache = MagicMock()
        mock_cache.get.return_value = sqlite_result
        mock_get_cache.return_value = mock_cache

        result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "predictions" in result
        assert isinstance(result["predictions"], np.ndarray)

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_xgboost_model_path(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "temperature_2m": np.ones(n) * 80,
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_model = {"model": "xgb"}
        predictions = np.ones(50) * 30000

        with (
            patch("models.xgboost_model.train_xgboost", return_value=mock_model),
            patch("models.xgboost_model.predict_xgboost", return_value=predictions),
        ):
            result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")

        assert "predictions" in result
        assert len(result["predictions"]) == 24

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_insufficient_data_returns_error(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        # Too little data
        featured = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
                "demand_mw": np.ones(50) * 30000,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "error" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_unknown_model_returns_error(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "hour": ts.hour,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        result = _run_forecast_outlook(demand_df, weather_df, 24, "nonexistent", "ERCOT")
        assert "error" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_prophet_model_path(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_model = MagicMock()
        prophet_result = {"forecast": np.ones(50) * 31000}

        with (
            patch("models.prophet_model.train_prophet", return_value=mock_model),
            patch("models.prophet_model.predict_prophet", return_value=prophet_result),
        ):
            result = _run_forecast_outlook(demand_df, weather_df, 24, "prophet", "ERCOT")
        assert "predictions" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_arima_model_path(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_model = MagicMock()
        arima_preds = np.ones(50) * 29000

        with (
            patch("models.arima_model.train_arima", return_value=mock_model),
            patch("models.arima_model.predict_arima", return_value=arima_preds),
        ):
            result = _run_forecast_outlook(demand_df, weather_df, 24, "arima", "ERCOT")
        assert "predictions" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_model_failure_returns_error(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        from components.callbacks import _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        with patch("models.xgboost_model.train_xgboost", side_effect=Exception("Training failed")):
            result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
        assert "error" in result

    @patch("data.cache.get_cache")
    @patch("data.preprocessing.merge_demand_weather")
    @patch("data.feature_engineering.engineer_features")
    def test_xgboost_model_cache_hit(
        self,
        mock_features,
        mock_merge,
        mock_get_cache,
        demand_df,
        weather_df,
    ):
        """If a trained XGBoost model is in _MODEL_CACHE, it should be reused."""
        import components.callbacks as cb
        from components.callbacks import _compute_data_hash, _run_forecast_outlook

        n = 800
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        featured = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.random.default_rng(1).normal(30000, 1000, n),
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "month": ts.month,
            }
        )
        mock_merge.return_value = featured
        mock_features.return_value = featured

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        data_hash = _compute_data_hash(demand_df, weather_df, "ERCOT")
        mock_model = {"model": "cached_xgb"}
        cb._MODEL_CACHE[("ERCOT", "xgboost", 0)] = (mock_model, data_hash, time.time())

        predictions = np.ones(50) * 30000
        with (
            patch("models.xgboost_model.train_xgboost") as m_train,
            patch("models.xgboost_model.predict_xgboost", return_value=predictions),
        ):
            result = _run_forecast_outlook(demand_df, weather_df, 24, "xgboost", "ERCOT")
            # train should NOT be called since model is cached
            m_train.assert_not_called()

        assert "predictions" in result


# ===========================================================================
# Module-level constants / EIA fuel map
# ===========================================================================


class TestModuleConstants:
    """Tests for module-level constants used by helpers."""

    def test_eia_fuel_map_covers_short_codes(self):
        from components.callbacks import _EIA_FUEL_MAP

        assert _EIA_FUEL_MAP["SUN"] == "solar"
        assert _EIA_FUEL_MAP["WND"] == "wind"
        assert _EIA_FUEL_MAP["NG"] == "gas"
        assert _EIA_FUEL_MAP["NUC"] == "nuclear"
        assert _EIA_FUEL_MAP["COL"] == "coal"
        assert _EIA_FUEL_MAP["WAT"] == "hydro"

    def test_eia_fuel_map_covers_long_names(self):
        from components.callbacks import _EIA_FUEL_MAP

        assert _EIA_FUEL_MAP["Solar"] == "solar"
        assert _EIA_FUEL_MAP["Wind"] == "wind"
        assert _EIA_FUEL_MAP["Natural Gas"] == "gas"

    def test_colors_dict_has_all_keys(self):
        from components.callbacks import COLORS

        expected = [
            "actual",
            "prophet",
            "arima",
            "xgboost",
            "ensemble",
            "eia_forecast",
            "temperature",
            "confidence",
            "gas",
            "nuclear",
            "coal",
            "wind",
            "solar",
            "hydro",
            "other",
        ]
        for key in expected:
            assert key in COLORS

    def test_plot_layout_has_template(self):
        from components.callbacks import PLOT_LAYOUT

        assert "template" in PLOT_LAYOUT
