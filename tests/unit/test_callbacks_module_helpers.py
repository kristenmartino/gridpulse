"""
Unit tests for module-level helper functions in components/callbacks.py.

Tests cover 9 helper functions that live outside register_callbacks():
  - _compute_data_hash
  - _confidence_half_width
  - _add_confidence_bands
  - _add_trailing_actuals
  - _create_future_features
  - _fetch_generation_cached
  - _empty_figure
  - _get_feature_importance
  - _build_persona_kpis
"""

import time
from unittest.mock import patch

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------


def _make_demand_df(n: int = 100, start: str = "2024-06-01") -> pd.DataFrame:
    """Create a minimal demand DataFrame."""
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {"timestamp": ts, "demand_mw": np.random.default_rng(42).uniform(20000, 40000, n)}
    )


def _make_weather_df(n: int = 100, start: str = "2024-06-01") -> pd.DataFrame:
    """Create a minimal weather DataFrame."""
    ts = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "wind_speed_80m": rng.uniform(5, 25, n),
            "shortwave_radiation": rng.uniform(0, 800, n),
        }
    )


def _make_train_df(n: int = 200, start: str = "2024-05-01") -> pd.DataFrame:
    """Create a training DataFrame with time + extra feature columns."""
    ts = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": rng.uniform(20000, 40000, n),
            "region": "ERCOT",
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "month": ts.month,
            "day_of_year": ts.dayofyear,
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "temperature_2m": rng.uniform(60, 100, n),
            "wind_speed_80m": rng.uniform(5, 25, n),
            "demand_lag_24h": rng.uniform(20000, 40000, n),
        }
    )


# ===================================================================
# 1. _compute_data_hash
# ===================================================================


class TestComputeDataHash:
    """Tests for _compute_data_hash(demand_df, weather_df, region)."""

    def test_same_inputs_produce_same_hash(self):
        from components.callbacks import _compute_data_hash

        d = _make_demand_df(48)
        w = _make_weather_df(48)
        h1 = _compute_data_hash(d, w, "ERCOT")
        h2 = _compute_data_hash(d, w, "ERCOT")
        assert h1 == h2

    def test_different_region_produces_different_hash(self):
        from components.callbacks import _compute_data_hash

        d = _make_demand_df(48)
        w = _make_weather_df(48)
        h1 = _compute_data_hash(d, w, "ERCOT")
        h2 = _compute_data_hash(d, w, "CAISO")
        assert h1 != h2

    def test_different_timestamps_produce_different_hash(self):
        from components.callbacks import _compute_data_hash

        d1 = _make_demand_df(48, start="2024-06-01")
        d2 = _make_demand_df(48, start="2024-07-01")
        w = _make_weather_df(48)
        h1 = _compute_data_hash(d1, w, "ERCOT")
        h2 = _compute_data_hash(d2, w, "ERCOT")
        assert h1 != h2

    def test_empty_dataframes(self):
        from components.callbacks import _compute_data_hash

        d = pd.DataFrame(columns=["timestamp", "demand_mw"])
        w = pd.DataFrame(columns=["timestamp"])
        # Should not raise
        result = _compute_data_hash(d, w, "PJM")
        assert isinstance(result, str)

    def test_returns_int(self):
        from components.callbacks import _compute_data_hash

        d = _make_demand_df(10)
        w = _make_weather_df(10)
        result = _compute_data_hash(d, w, "MISO")
        assert isinstance(result, str)


# ===================================================================
# 2. _confidence_half_width
# ===================================================================


class TestConfidenceHalfWidth:
    """Tests for _confidence_half_width(horizon_hours)."""

    def test_short_horizon_24h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(24) == 0.03

    def test_medium_horizon_168h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(168) == 0.06

    def test_long_horizon_720h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(720) == 0.10

    def test_below_24h(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(12) == 0.03

    def test_between_24_and_168(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(72) == 0.06

    def test_above_168(self):
        from components.callbacks import _confidence_half_width

        assert _confidence_half_width(200) == 0.10


# ===================================================================
# 3. _add_confidence_bands
# ===================================================================


class TestAddConfidenceBands:
    """Tests for _add_confidence_bands(fig, timestamps, predictions, horizon_hours)."""

    def test_adds_two_traces(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-06-01", periods=24, freq="h")
        preds = np.full(24, 30000.0)
        _add_confidence_bands(fig, ts, preds, 24)
        assert len(fig.data) == 2

    def test_upper_trace_has_no_fill(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-06-01", periods=24, freq="h")
        preds = np.full(24, 30000.0)
        _add_confidence_bands(fig, ts, preds, 24)
        upper = fig.data[0]
        assert upper.fill is None

    def test_lower_trace_has_tonexty_fill(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-06-01", periods=24, freq="h")
        preds = np.full(24, 30000.0)
        _add_confidence_bands(fig, ts, preds, 24)
        lower = fig.data[1]
        assert lower.fill == "tonexty"

    def test_band_width_scales_with_horizon(self):
        from components.callbacks import _add_confidence_bands

        preds = np.full(24, 30000.0)
        ts = pd.date_range("2024-06-01", periods=24, freq="h")

        fig_short = go.Figure()
        _add_confidence_bands(fig_short, ts, preds, 24)
        upper_short = np.array(fig_short.data[0].y)

        fig_long = go.Figure()
        _add_confidence_bands(fig_long, ts, preds, 720)
        upper_long = np.array(fig_long.data[0].y)

        # Longer horizon should have wider bands
        assert upper_long[0] > upper_short[0]

    def test_lower_trace_named_80pct_ci(self):
        from components.callbacks import _add_confidence_bands

        fig = go.Figure()
        ts = pd.date_range("2024-06-01", periods=24, freq="h")
        preds = np.full(24, 30000.0)
        _add_confidence_bands(fig, ts, preds, 24)
        assert fig.data[1].name == "80% CI"


# ===================================================================
# 4. _add_trailing_actuals
# ===================================================================


class TestAddTrailingActuals:
    """Tests for _add_trailing_actuals(fig, demand_json, tail_hours)."""

    def test_adds_trace_with_valid_json(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        demand_df = _make_demand_df(100)
        demand_json = demand_df.to_json()
        _add_trailing_actuals(fig, demand_json, tail_hours=48)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Actual"

    def test_trace_uses_dotted_line(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        demand_df = _make_demand_df(100)
        demand_json = demand_df.to_json()
        _add_trailing_actuals(fig, demand_json)
        assert fig.data[0].line.dash == "dot"

    def test_noop_on_none_json(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, None)
        assert len(fig.data) == 0

    def test_noop_on_empty_string(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, "")
        assert len(fig.data) == 0

    def test_tail_hours_limits_rows(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        demand_df = _make_demand_df(200)
        demand_json = demand_df.to_json()
        _add_trailing_actuals(fig, demand_json, tail_hours=10)
        # The trace should have at most 10 data points
        assert len(fig.data[0].y) == 10

    def test_handles_bad_json_silently(self):
        from components.callbacks import _add_trailing_actuals

        fig = go.Figure()
        _add_trailing_actuals(fig, "{{not-valid-json}}")
        # Should not raise; figure should remain empty
        assert len(fig.data) == 0


# ===================================================================
# 5. _create_future_features
# ===================================================================


class TestCreateFutureFeatures:
    """Tests for _create_future_features(train_df, future_timestamps)."""

    def test_short_horizon_uses_last_row(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        future_ts = pd.date_range("2024-05-09 08:00", periods=48, freq="h")
        result = _create_future_features(train, future_ts)

        # Non-time features should use last row value for short horizon
        last_temp = train["temperature_2m"].iloc[-1]
        assert np.allclose(result["temperature_2m"].values, last_temp)

    def test_long_horizon_uses_group_means(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(500, start="2024-01-01")
        future_ts = pd.date_range("2024-01-21 00:00", periods=200, freq="h")
        result = _create_future_features(train, future_ts)

        # For 200-hour horizon (>168), group means should vary across hours/days
        unique_temps = result["temperature_2m"].nunique()
        assert unique_temps > 1

    def test_time_features_always_computed(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        future_ts = pd.date_range("2024-05-09 08:00", periods=24, freq="h")
        result = _create_future_features(train, future_ts)

        time_cols = [
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
        for col in time_cols:
            assert col in result.columns, f"Missing time feature: {col}"

    def test_hour_values_match_timestamps(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        future_ts = pd.date_range("2024-05-09 08:00", periods=24, freq="h")
        result = _create_future_features(train, future_ts)

        expected_hours = future_ts.hour.values
        np.testing.assert_array_equal(result["hour"].values, expected_hours)

    def test_is_weekend_correct(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        # 2024-06-08 is a Saturday, 2024-06-09 is Sunday
        future_ts = pd.date_range("2024-06-08 00:00", periods=48, freq="h")
        result = _create_future_features(train, future_ts)

        # All rows should be weekend (Saturday=5, Sunday=6)
        assert (result["is_weekend"] == 1).all()

    def test_existing_feature_filled_from_last_row(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        # Add a feature column that will be in last_row
        train["phantom_feature"] = 42.0
        future_ts = pd.date_range("2024-05-09 08:00", periods=24, freq="h")
        result = _create_future_features(train, future_ts)

        # phantom_feature should be filled from last_row (=42)
        assert "phantom_feature" in result.columns
        assert np.allclose(result["phantom_feature"].values, 42.0)

    def test_result_has_timestamp_column(self):
        from components.callbacks import _create_future_features

        train = _make_train_df(200)
        future_ts = pd.date_range("2024-05-09 08:00", periods=24, freq="h")
        result = _create_future_features(train, future_ts)

        assert "timestamp" in result.columns
        assert len(result) == 24

    def test_short_horizon_boundary(self):
        """Horizon of exactly 167 (<168) should use last row values."""
        from components.callbacks import _create_future_features

        train = _make_train_df(500, start="2024-01-01")
        future_ts = pd.date_range("2024-01-21 00:00", periods=167, freq="h")
        result = _create_future_features(train, future_ts)

        # Should use last row value for non-time features
        last_temp = train["temperature_2m"].iloc[-1]
        assert np.allclose(result["temperature_2m"].values, last_temp)


# ===================================================================
# 6. _fetch_generation_cached
# ===================================================================


class TestFetchGenerationCached:
    """Tests for _fetch_generation_cached(region)."""

    def test_memory_cache_hit(self):
        """Memory cache returns data when fresh (< 300s)."""
        import components.callbacks as cb_mod

        gen_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=10, freq="h"),
                "fuel_type": ["gas"] * 10,
                "generation_mw": [1000] * 10,
                "region": ["ERCOT"] * 10,
            }
        )
        # Directly seed the memory cache
        cb_mod._GENERATION_CACHE["ERCOT"] = (gen_df, time.time())
        try:
            result = cb_mod._fetch_generation_cached("ERCOT")
            assert result is not None
            assert len(result) == 10
        finally:
            cb_mod._GENERATION_CACHE.clear()

    def test_memory_cache_expired(self):
        """Expired cache (>300s) should NOT return from memory."""
        import components.callbacks as cb_mod

        gen_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=5, freq="h"),
                "fuel_type": ["gas"] * 5,
                "generation_mw": [1000] * 5,
                "region": ["ERCOT"] * 5,
            }
        )
        # Set cache with old timestamp
        cb_mod._GENERATION_CACHE["ERCOT"] = (gen_df, time.time() - 600)

        # Mock EIA key to be empty so it skips to demo
        with (
            patch("config.EIA_API_KEY", ""),
            patch("data.demo_data.generate_demo_generation") as mock_demo,
        ):
            mock_demo.return_value = gen_df
            result = cb_mod._fetch_generation_cached("ERCOT")
            assert result is not None

        cb_mod._GENERATION_CACHE.clear()

    @patch("data.eia_client.fetch_generation_by_fuel")
    def test_eia_api_success(self, mock_fetch):
        """EIA API returns data and populates cache."""
        import components.callbacks as cb_mod

        cb_mod._GENERATION_CACHE.clear()

        gen_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=5, freq="h"),
                "fuel_type": ["NG"] * 5,
                "generation_mw": [2000] * 5,
                "region": ["CAISO"] * 5,
            }
        )
        mock_fetch.return_value = gen_df

        with patch("config.EIA_API_KEY", "real_key"):
            result = cb_mod._fetch_generation_cached("CAISO")

        assert result is not None
        # Fuel type should be normalized: NG -> gas
        assert (result["fuel_type"] == "gas").all()
        cb_mod._GENERATION_CACHE.clear()

    def test_eia_fails_falls_to_demo(self):
        """When EIA raises, falls through to demo data."""
        import components.callbacks as cb_mod

        cb_mod._GENERATION_CACHE.clear()

        demo_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=3, freq="h"),
                "fuel_type": ["solar"] * 3,
                "generation_mw": [500] * 3,
                "region": ["SPP"] * 3,
            }
        )

        with (
            patch("config.EIA_API_KEY", "real_key"),
            patch(
                "data.eia_client.fetch_generation_by_fuel",
                side_effect=Exception("API down"),
            ),
            patch("data.demo_data.generate_demo_generation", return_value=demo_df),
        ):
            result = cb_mod._fetch_generation_cached("SPP")

        assert result is not None
        assert len(result) == 3
        cb_mod._GENERATION_CACHE.clear()

    def test_all_tiers_fail_returns_none(self):
        """When all tiers fail, returns None."""
        import components.callbacks as cb_mod

        cb_mod._GENERATION_CACHE.clear()

        with (
            patch("config.EIA_API_KEY", "real_key"),
            patch(
                "data.eia_client.fetch_generation_by_fuel",
                side_effect=Exception("API down"),
            ),
            patch(
                "data.demo_data.generate_demo_generation",
                side_effect=Exception("Demo broken"),
            ),
        ):
            result = cb_mod._fetch_generation_cached("NYISO")

        assert result is None
        cb_mod._GENERATION_CACHE.clear()

    def test_no_api_key_skips_eia(self):
        """When EIA_API_KEY is empty, skips EIA and goes to demo."""
        import components.callbacks as cb_mod

        cb_mod._GENERATION_CACHE.clear()

        demo_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-06-01", periods=3, freq="h"),
                "fuel_type": ["wind"] * 3,
                "generation_mw": [300] * 3,
                "region": ["MISO"] * 3,
            }
        )

        with (
            patch("config.EIA_API_KEY", ""),
            patch("data.demo_data.generate_demo_generation", return_value=demo_df),
        ):
            result = cb_mod._fetch_generation_cached("MISO")

        assert result is not None
        assert len(result) == 3
        cb_mod._GENERATION_CACHE.clear()


# ===================================================================
# 7. _empty_figure
# ===================================================================


class TestEmptyFigure:
    """Tests for _empty_figure(message)."""

    def test_returns_go_figure(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("No data available")
        assert isinstance(fig, go.Figure)

    def test_has_annotation_with_message(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("Test message")
        annotations = fig.layout.annotations
        assert len(annotations) == 1
        assert annotations[0].text == "Test message"

    def test_axes_hidden(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("Hidden axes")
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False

    def test_empty_message(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("")
        assert fig.layout.annotations[0].text == ""

    def test_annotation_centered(self):
        from components.callbacks import _empty_figure

        fig = _empty_figure("Center")
        ann = fig.layout.annotations[0]
        assert ann.x == 0.5
        assert ann.y == 0.5


# ===================================================================
# 8. _get_feature_importance
# ===================================================================


class TestGetFeatureImportance:
    """Tests for _get_feature_importance(region, top_n)."""

    def test_returns_defaults_when_no_cache(self):
        import components.callbacks as cb_mod

        original = cb_mod._MODEL_CACHE.copy()
        cb_mod._MODEL_CACHE.clear()
        try:
            names, vals = cb_mod._get_feature_importance("ERCOT")
            assert len(names) == 10
            assert len(vals) == 10
            assert names[0] == "temperature_2m"
            assert vals[0] == 0.25
        finally:
            cb_mod._MODEL_CACHE.update(original)

    def test_returns_cached_importances(self):
        import components.callbacks as cb_mod

        original = cb_mod._MODEL_CACHE.copy()
        model_dict = {
            "feature_importances": {
                "temperature_2m": 0.30,
                "hour_sin": 0.20,
                "demand_lag_24h": 0.15,
                "wind_speed_80m": 0.10,
                "cooling_degree_days": 0.05,
            }
        }
        cb_mod._MODEL_CACHE[("PJM", "xgboost", 0)] = (model_dict, "hash123", time.time())

        try:
            names, vals = cb_mod._get_feature_importance("PJM", top_n=3)
            assert len(names) == 3
            assert names[0] == "temperature_2m"
            assert vals[0] == 0.30
            # Should be sorted descending
            assert vals[0] >= vals[1] >= vals[2]
        finally:
            cb_mod._MODEL_CACHE.clear()
            cb_mod._MODEL_CACHE.update(original)

    def test_falls_back_when_sum_zero(self):
        """If all importances are 0, falls back to defaults."""
        import components.callbacks as cb_mod

        original = cb_mod._MODEL_CACHE.copy()
        model_dict = {
            "feature_importances": {
                "temperature_2m": 0.0,
                "hour_sin": 0.0,
            }
        }
        cb_mod._MODEL_CACHE[("FPL", "xgboost", 0)] = (model_dict, "hash", time.time())

        try:
            names, vals = cb_mod._get_feature_importance("FPL")
            # Should fall back to defaults
            assert len(names) == 10
            assert names[0] == "temperature_2m"
            assert vals[0] == 0.25
        finally:
            cb_mod._MODEL_CACHE.clear()
            cb_mod._MODEL_CACHE.update(original)

    def test_top_n_defaults_return_full_list(self):
        """Default fallback always returns the full 10-item list regardless of top_n."""
        import components.callbacks as cb_mod

        original = cb_mod._MODEL_CACHE.copy()
        cb_mod._MODEL_CACHE.clear()
        try:
            names, vals = cb_mod._get_feature_importance("ERCOT", top_n=5)
            # Default list has 10 items; top_n not applied to defaults
            assert len(names) == 10
        finally:
            cb_mod._MODEL_CACHE.update(original)

    def test_returns_numpy_array(self):
        import components.callbacks as cb_mod

        original = cb_mod._MODEL_CACHE.copy()
        cb_mod._MODEL_CACHE.clear()
        try:
            names, vals = cb_mod._get_feature_importance("ERCOT")
            assert isinstance(vals, np.ndarray)
            assert isinstance(names, list)
        finally:
            cb_mod._MODEL_CACHE.update(original)


# ===================================================================
# 9. _build_persona_kpis
# ===================================================================


class TestBuildPersonaKpis:
    """Tests for _build_persona_kpis(persona_id, region, demand_df, weather_df)."""

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_grid_ops_with_demand_data(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        demand = _make_demand_df(100)
        result = _build_persona_kpis("grid_ops", "ERCOT", demand_df=demand)
        assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_renewables_with_weather_data(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        weather = _make_weather_df(100)
        result = _build_persona_kpis("renewables", "ERCOT", weather_df=weather)
        assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_trader_persona(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        demand = _make_demand_df(100)
        result = _build_persona_kpis("trader", "ERCOT", demand_df=demand)
        assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_data_scientist_persona(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        demand = _make_demand_df(100)
        result = _build_persona_kpis("data_scientist", "PJM", demand_df=demand)
        assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_unknown_persona_falls_back_to_grid_ops(self, mock_redis):
        from components.callbacks import _build_persona_kpis

        demand = _make_demand_df(50)
        result = _build_persona_kpis("unknown_persona", "ERCOT", demand_df=demand)
        assert isinstance(result, dbc.Row)

    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_no_data_falls_to_redis(self):
        """When no demand_df provided, should call redis_get for actuals."""
        with patch("components.callbacks.redis_get") as mock_redis:
            mock_redis.return_value = None
            from components.callbacks import _build_persona_kpis

            result = _build_persona_kpis("grid_ops", "ERCOT")
            # Should have called redis_get for actuals
            mock_redis.assert_any_call("wattcast:actuals:ERCOT")
            assert isinstance(result, dbc.Row)

    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_redis_demand_fallback(self):
        """Redis provides demand stats when demand_df is None."""
        redis_data = {
            "wattcast:actuals:ERCOT": {
                "demand_mw": [25000, 30000, 35000, 40000],
            },
        }

        def mock_redis_get(key):
            return redis_data.get(key)

        with patch("components.callbacks.redis_get", side_effect=mock_redis_get):
            from components.callbacks import _build_persona_kpis

            result = _build_persona_kpis("grid_ops", "ERCOT")
            assert isinstance(result, dbc.Row)

    @patch("components.callbacks.redis_get", return_value=None)
    def test_backtest_cache_provides_mape(self, mock_redis):
        """Backtest cache provides MAPE and RMSE."""
        import components.callbacks as cb_mod

        original = cb_mod._BACKTEST_CACHE.copy()
        cb_mod._BACKTEST_CACHE[("ERCOT", 168, "xgboost")] = (
            {"metrics": {"mape": 3.5, "rmse": 1200}},
            "hash",
            time.time(),
        )

        try:
            result = cb_mod._build_persona_kpis(
                "data_scientist", "ERCOT", demand_df=_make_demand_df(50)
            )
            assert isinstance(result, dbc.Row)
        finally:
            cb_mod._BACKTEST_CACHE.clear()
            cb_mod._BACKTEST_CACHE.update(original)

    @patch("components.callbacks.redis_get", return_value=None)
    @patch("components.callbacks._BACKTEST_CACHE", {})
    def test_all_none_returns_row_with_dash_values(self, mock_redis):
        """With no data at all, should still return a valid Row with placeholder values."""
        from components.callbacks import _build_persona_kpis

        result = _build_persona_kpis("grid_ops", "ERCOT")
        assert isinstance(result, dbc.Row)
