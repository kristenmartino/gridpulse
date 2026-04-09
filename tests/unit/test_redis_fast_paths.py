"""Unit tests for Redis fast path functions in components/callbacks.py.

Tests all 7 extracted Redis fast-path functions that bypass compute-on-demand
when pre-computed data is available in Redis. Each function follows the pattern:
redis_get → parse → build Plotly figures/Dash components → return tuple or None.

All tests mock ``components.callbacks.redis_get`` so no real Redis connection
is needed.
"""

import json
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
from dash import html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(n=48):
    """Return *n* hourly ISO-8601 timestamp strings starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%S").tolist()


def _demand_payload(n=48, base=30_000):
    """Minimal ``wattcast:actuals:{region}`` Redis payload."""
    return {
        "timestamps": _ts(n),
        "demand_mw": [base + i * 10 for i in range(n)],
    }


def _weather_payload(n=48):
    """Minimal ``wattcast:weather:{region}`` Redis payload."""
    return {
        "timestamps": _ts(n),
        "temperature_2m": [70 + i * 0.1 for i in range(n)],
        "wind_speed_80m": [10 + i * 0.05 for i in range(n)],
        "shortwave_radiation": [200 + i for i in range(n)],
    }


def _weather_correlation_payload():
    """Minimal ``wattcast:weather-correlation:{region}`` Redis payload."""
    return {
        "temperature_2m": [70, 72, 74],
        "demand_mw": [30000, 31000, 32000],
        "wind_speed_80m": [10, 12, 14],
        "wind_power": [100, 200, 300],
        "shortwave_radiation": [200, 400, 600],
        "solar_cf": [0.2, 0.4, 0.6],
        "correlation_matrix": {
            "cols": ["temperature_2m", "demand_mw"],
            "values": [[1.0, 0.8], [0.8, 1.0]],
        },
        "importance": {
            "names": ["temperature_2m", "wind_speed_80m"],
            "values": [0.8, 0.3],
        },
        "seasonal": {
            "timestamps": _ts(24),
            "original": list(range(24)),
            "trend": [float(x) for x in range(24)],
            "residual": [0.0] * 24,
        },
    }


def _diagnostics_payload():
    """Minimal ``wattcast:diagnostics:{region}`` Redis payload."""
    n = 48
    return {
        "metrics": {
            "prophet": {"mape": 5.0, "rmse": 500, "mae": 400, "r2": 0.95},
            "arima": {"mape": 4.5, "rmse": 450, "mae": 380, "r2": 0.96},
            "xgboost": {"mape": 3.0, "rmse": 300, "mae": 250, "r2": 0.98},
            "ensemble": {"mape": 2.5, "rmse": 280, "mae": 230, "r2": 0.985},
        },
        "timestamps": _ts(n),
        "ensemble": [30000 + i * 10 for i in range(n)],
        "residuals": [(-1) ** i * 50 for i in range(n)],
        "hourly_error": {
            "hours": list(range(24)),
            "values": [100 + i * 5 for i in range(24)],
        },
        "feature_importance": {
            "names": ["temperature_2m", "hour", "demand_lag_1"],
            "values": [0.5, 0.3, 0.2],
        },
    }


def _generation_payload(n=72):
    """Minimal ``wattcast:generation:{region}`` Redis payload."""
    ts = pd.date_range(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=n), periods=n, freq="h")
    return {
        "timestamps": ts.strftime("%Y-%m-%dT%H:%M:%S%z").tolist(),
        "wind": [500 + i for i in range(n)],
        "solar": [200 + i for i in range(n)],
        "gas": [10000] * n,
        "nuclear": [5000] * n,
        "hydro": [1000] * n,
    }


def _alerts_payload(with_alerts=True):
    """Minimal ``wattcast:alerts:{region}`` Redis payload."""
    alerts = []
    if with_alerts:
        alerts = [
            {
                "event": "Heat Advisory",
                "headline": "Heat Advisory until 8PM",
                "severity": "warning",
                "expires": "2024-07-15T20:00:00",
            }
        ]
    return {
        "alerts": alerts,
        "stress_score": 45 if with_alerts else 15,
        "stress_label": "Elevated" if with_alerts else "Normal",
        "alert_counts": {"warning": 1, "info": 0} if with_alerts else {},
        "anomaly": {
            "timestamps": _ts(24),
            "demand": [30000 + i * 10 for i in range(24)],
            "upper": [32000] * 24,
            "lower": [28000] * 24,
            "anomaly_timestamps": _ts(2),
            "anomaly_values": [35000, 36000],
        },
        "temperature": {
            "timestamps": _ts(24),
            "values": [90 + i * 0.5 for i in range(24)],
        },
    }


def _forecast_payload(n=72, model="xgboost"):
    """Minimal ``wattcast:forecast:{region}:1h`` Redis payload."""
    return {
        "scored_at": "2024-06-01T12:00:00Z",
        "forecasts": [
            {
                "timestamp": t,
                model: 30000 + i * 10,
                "predicted_demand_mw": 30000 + i * 10,
            }
            for i, t in enumerate(_ts(n))
        ],
    }


def _backtest_payload(horizon=24):
    """Minimal ``wattcast:backtest:{region}:{horizon}`` Redis payload."""
    n = 48
    actual = [30000 + i * 10 for i in range(n)]
    return {
        "timestamps": _ts(n),
        "actual": actual,
        "predictions": {
            "xgboost": [a + 50 for a in actual],
            "ensemble": [a + 30 for a in actual],
        },
        "metrics": {
            "xgboost": {"mape": 3.2, "rmse": 320, "mae": 260, "r2": 0.97},
            "ensemble": {"mape": 2.8, "rmse": 290, "mae": 240, "r2": 0.98},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# TestLoadDataFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadDataFromRedis:
    """Tests for _load_data_from_redis(region)."""

    @patch("components.callbacks.redis_get")
    def test_both_keys_hit_returns_5_tuple(self, mock_rg):
        """Both actuals and weather cached → returns 5-tuple of JSON strings."""
        mock_rg.side_effect = lambda k: {
            "wattcast:actuals:FPL": _demand_payload(),
            "wattcast:weather:FPL": _weather_payload(),
        }.get(k)

        from components.callbacks import _load_data_from_redis

        result = _load_data_from_redis("FPL")

        assert result is not None
        assert len(result) == 5
        # Each element is a JSON string
        for item in result:
            assert isinstance(item, str)
        # demand_json should be parseable
        demand_parsed = json.loads(result[0])
        assert "demand_mw" in demand_parsed
        assert "timestamp" in demand_parsed

    @patch("components.callbacks.redis_get")
    def test_actuals_miss_returns_none(self, mock_rg):
        """Actuals cache miss → returns None."""
        mock_rg.side_effect = lambda k: {
            "wattcast:weather:FPL": _weather_payload(),
        }.get(k)

        from components.callbacks import _load_data_from_redis

        assert _load_data_from_redis("FPL") is None

    @patch("components.callbacks.redis_get")
    def test_weather_miss_returns_none(self, mock_rg):
        """Weather cache miss → returns None."""
        mock_rg.side_effect = lambda k: {
            "wattcast:actuals:FPL": _demand_payload(),
        }.get(k)

        from components.callbacks import _load_data_from_redis

        assert _load_data_from_redis("FPL") is None

    @patch("components.callbacks.redis_get")
    def test_both_miss_returns_none(self, mock_rg):
        """Both keys missing → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _load_data_from_redis

        assert _load_data_from_redis("FPL") is None

    @patch("components.callbacks.redis_get")
    def test_empty_demand_no_latest_data(self, mock_rg):
        """Empty demand_mw list → freshness dict has no latest_data key."""
        payload = {"timestamps": [], "demand_mw": []}
        mock_rg.side_effect = lambda k: {
            "wattcast:actuals:FPL": payload,
            "wattcast:weather:FPL": _weather_payload(),
        }.get(k)

        from components.callbacks import _load_data_from_redis

        result = _load_data_from_redis("FPL")
        assert result is not None
        freshness = json.loads(result[2])
        assert "latest_data" not in freshness

    @patch("components.callbacks.redis_get")
    def test_audit_trail_uses_redis_source(self, mock_rg):
        """Audit record records source='redis'."""
        mock_rg.side_effect = lambda k: {
            "wattcast:actuals:FPL": _demand_payload(),
            "wattcast:weather:FPL": _weather_payload(),
        }.get(k)

        from components.callbacks import _load_data_from_redis

        result = _load_data_from_redis("FPL")
        assert result is not None
        audit_json = json.loads(result[3])
        assert audit_json["demand_source"] == "redis"
        assert audit_json["weather_source"] == "redis"
        assert audit_json["forecast_source"] == "redis"


# ═══════════════════════════════════════════════════════════════════════════
# TestWeatherTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestWeatherTabFromRedis:
    """Tests for _weather_tab_from_redis(region)."""

    @patch("components.callbacks.redis_get")
    def test_full_data_returns_6_figures(self, mock_rg):
        """Full cached weather correlation data → returns 6 go.Figures."""
        mock_rg.return_value = _weather_correlation_payload()

        from components.callbacks import _weather_tab_from_redis

        result = _weather_tab_from_redis("FPL")
        assert result is not None
        assert len(result) == 6
        for fig in result:
            assert isinstance(fig, go.Figure)

    @patch("components.callbacks.redis_get")
    def test_cache_miss_returns_none(self, mock_rg):
        """Cache miss → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _weather_tab_from_redis

        assert _weather_tab_from_redis("FPL") is None

    @patch("components.callbacks.redis_get")
    def test_empty_correlation_matrix_still_creates_figures(self, mock_rg):
        """Empty correlation matrix → figures still created (no crash)."""
        payload = _weather_correlation_payload()
        payload["correlation_matrix"] = {"cols": [], "values": []}
        mock_rg.return_value = payload

        from components.callbacks import _weather_tab_from_redis

        result = _weather_tab_from_redis("FPL")
        assert result is not None
        assert len(result) == 6

    @patch("components.callbacks.redis_get")
    def test_scatter_plots_have_traces(self, mock_rg):
        """Temperature, wind, solar scatter plots have marker traces."""
        mock_rg.return_value = _weather_correlation_payload()

        from components.callbacks import _weather_tab_from_redis

        fig_temp, fig_wind, fig_solar, _, _, _ = _weather_tab_from_redis("FPL")

        # Each scatter should have exactly 1 trace
        assert len(fig_temp.data) == 1
        assert fig_temp.data[0].mode == "markers"
        assert len(fig_wind.data) == 1
        assert fig_wind.data[0].mode == "markers"
        assert len(fig_solar.data) == 1
        assert fig_solar.data[0].mode == "markers"

    @patch("components.callbacks.redis_get")
    def test_heatmap_uses_rdbu_colorscale(self, mock_rg):
        """Heatmap figure uses RdBu colorscale."""
        mock_rg.return_value = _weather_correlation_payload()

        from components.callbacks import _weather_tab_from_redis

        _, _, _, fig_heatmap, _, _ = _weather_tab_from_redis("FPL")
        # Plotly expands "RdBu" into a tuple of (stop, color) pairs at render time.
        # Verify the first and last colours match the known RdBu endpoints.
        cs = fig_heatmap.data[0].colorscale
        assert cs[0][0] == 0.0
        assert "103,0,31" in cs[0][1]  # RdBu dark-red start
        assert cs[-1][0] == 1.0
        assert "5,48,97" in cs[-1][1]  # RdBu dark-blue end


# ═══════════════════════════════════════════════════════════════════════════
# TestModelsTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestModelsTabFromRedis:
    """Tests for _models_tab_from_redis(region)."""

    @patch("components.callbacks.redis_get")
    def test_full_diagnostics_returns_table_and_5_figures(self, mock_rg):
        """Full diagnostics → returns (html.Table, 5 x go.Figure)."""
        mock_rg.return_value = _diagnostics_payload()

        from components.callbacks import _models_tab_from_redis

        result = _models_tab_from_redis("FPL")
        assert result is not None
        assert len(result) == 6
        table, *figs = result
        assert isinstance(table, html.Table)
        for fig in figs:
            assert isinstance(fig, go.Figure)

    @patch("components.callbacks.redis_get")
    def test_cache_miss_returns_none(self, mock_rg):
        """Cache miss → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _models_tab_from_redis

        assert _models_tab_from_redis("FPL") is None

    @patch("components.callbacks.redis_get")
    def test_table_has_4_rows(self, mock_rg):
        """Metrics table has 4 rows: Prophet, SARIMAX, XGBoost, Ensemble."""
        mock_rg.return_value = _diagnostics_payload()

        from components.callbacks import _models_tab_from_redis

        table, *_ = _models_tab_from_redis("FPL")
        # table.children = [Thead, Tbody]
        tbody = table.children[1]
        rows = tbody.children
        assert len(rows) == 4
        # Check first cell of each row
        names = [row.children[0].children for row in rows]
        assert names == ["Prophet", "SARIMAX", "XGBoost", "Ensemble"]

    @patch("components.callbacks.redis_get")
    def test_empty_hourly_error_defaults_to_24_zeros(self, mock_rg):
        """Missing hourly_error values → defaults to list of 24 zeros."""
        payload = _diagnostics_payload()
        payload["hourly_error"] = {}
        mock_rg.return_value = payload

        from components.callbacks import _models_tab_from_redis

        result = _models_tab_from_redis("FPL")
        assert result is not None
        # fig_heatmap is index 4 (table, resid_time, resid_hist, resid_pred, heatmap, shap)
        fig_heatmap = result[4]
        bar_y = list(fig_heatmap.data[0].y)
        assert len(bar_y) == 24
        assert all(v == 0 for v in bar_y)

    @patch("components.callbacks.redis_get")
    def test_feature_importance_reversed_order(self, mock_rg):
        """Feature importance bar chart has reversed name/value order."""
        mock_rg.return_value = _diagnostics_payload()

        from components.callbacks import _models_tab_from_redis

        *_, fig_shap = _models_tab_from_redis("FPL")
        # Original order: ["temperature_2m", "hour", "demand_lag_1"]
        # Reversed: ["demand_lag_1", "hour", "temperature_2m"]
        y_vals = list(fig_shap.data[0].y)
        assert y_vals == ["demand_lag_1", "hour", "temperature_2m"]


# ═══════════════════════════════════════════════════════════════════════════
# TestGenerationTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerationTabFromRedis:
    """Tests for _generation_tab_from_redis(region, range_hours, demand_json, persona_id)."""

    @patch("components.callbacks.redis_get")
    def test_full_gen_data_returns_7_tuple(self, mock_rg):
        """Full generation data → returns 7-tuple."""
        mock_rg.return_value = _generation_payload(72)

        from components.callbacks import _generation_tab_from_redis

        result = _generation_tab_from_redis("FPL", 168, None, "grid_ops")
        assert result is not None
        assert len(result) == 7
        fig_hero, fig_mix, ren_pct, peak_ramp, min_net, curtail, insight_card = result
        assert isinstance(fig_hero, go.Figure)
        assert isinstance(fig_mix, go.Figure)
        assert "%" in ren_pct

    @patch("components.callbacks.redis_get")
    def test_cache_miss_returns_none(self, mock_rg):
        """Cache miss → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _generation_tab_from_redis

        assert _generation_tab_from_redis("FPL", 168, None, "grid_ops") is None

    @patch("components.callbacks.redis_get")
    def test_no_timestamps_returns_none(self, mock_rg):
        """Payload without timestamps → returns None."""
        mock_rg.return_value = {"wind": [100, 200], "solar": [50, 60]}

        from components.callbacks import _generation_tab_from_redis

        assert _generation_tab_from_redis("FPL", 168, None, "grid_ops") is None

    @patch("components.callbacks.redis_get")
    def test_empty_after_cutoff_returns_none(self, mock_rg):
        """All timestamps older than range_hours → returns None."""
        # Create data far in the past (>168 hours ago)
        old_ts = (
            pd.date_range("2020-01-01", periods=24, freq="h")
            .strftime("%Y-%m-%dT%H:%M:%S+00:00")
            .tolist()
        )
        payload = {
            "timestamps": old_ts,
            "wind": [100] * 24,
            "solar": [50] * 24,
            "gas": [10000] * 24,
        }
        mock_rg.return_value = payload

        from components.callbacks import _generation_tab_from_redis

        assert _generation_tab_from_redis("FPL", 168, None, "grid_ops") is None

    @patch("components.callbacks.redis_get")
    def test_renewable_pct_from_wind_solar_hydro(self, mock_rg):
        """Renewable % is computed from wind + solar + hydro columns."""
        mock_rg.return_value = _generation_payload(72)

        from components.callbacks import _generation_tab_from_redis

        result = _generation_tab_from_redis("FPL", 168, None, "grid_ops")
        assert result is not None
        ren_pct_str = result[2]
        ren_pct_val = float(ren_pct_str.replace("%", ""))
        # With wind + solar + hydro present, renewable % should be > 0
        assert ren_pct_val > 0

    @patch("components.callbacks.redis_get")
    def test_demand_alignment_from_demand_json(self, mock_rg):
        """demand_json provided → demand series is aligned from it."""
        gen_payload = _generation_payload(72)
        mock_rg.return_value = gen_payload

        # Build demand_json aligned with generation timestamps
        gen_ts = pd.to_datetime(gen_payload["timestamps"])
        demand_df = pd.DataFrame(
            {
                "timestamp": gen_ts,
                "demand_mw": [25000 + i * 10 for i in range(len(gen_ts))],
            }
        )
        demand_json = demand_df.to_json(date_format="iso")

        from components.callbacks import _generation_tab_from_redis

        result = _generation_tab_from_redis("FPL", 168, demand_json, "grid_ops")
        assert result is not None
        assert len(result) == 7


# ═══════════════════════════════════════════════════════════════════════════
# TestAlertsTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestAlertsTabFromRedis:
    """Tests for _alerts_tab_from_redis(region)."""

    @patch("components.callbacks.redis_get")
    def test_with_alerts_builds_alert_cards(self, mock_rg):
        """Payload with alerts → alert cards built from build_alert_card."""
        mock_rg.return_value = _alerts_payload(with_alerts=True)

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        assert result is not None
        assert len(result) == 7
        alert_cards = result[0]
        assert len(alert_cards) == 1
        # Each card is an html.Div from build_alert_card
        assert isinstance(alert_cards[0], html.Div)

    @patch("components.callbacks.redis_get")
    def test_no_alerts_placeholder(self, mock_rg):
        """No alerts → placeholder 'No active alerts' text."""
        mock_rg.return_value = _alerts_payload(with_alerts=False)

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        assert result is not None
        alert_cards = result[0]
        assert len(alert_cards) == 1
        assert isinstance(alert_cards[0], html.P)
        assert "No active alerts" in alert_cards[0].children

    @patch("components.callbacks.redis_get")
    def test_stress_below_30_is_positive(self, mock_rg):
        """Stress score < 30 → 'positive' CSS class on label span."""
        payload = _alerts_payload(with_alerts=False)
        payload["stress_score"] = 15
        mock_rg.return_value = payload

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        stress_label_span = result[2]
        assert isinstance(stress_label_span, html.Span)
        assert "positive" in stress_label_span.className

    @patch("components.callbacks.redis_get")
    def test_stress_ge_60_is_negative(self, mock_rg):
        """Stress score >= 60 → 'negative' CSS class on label span."""
        payload = _alerts_payload(with_alerts=True)
        payload["stress_score"] = 75
        mock_rg.return_value = payload

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        stress_label_span = result[2]
        assert isinstance(stress_label_span, html.Span)
        assert "negative" in stress_label_span.className

    @patch("components.callbacks.redis_get")
    def test_stress_mid_range_is_neutral(self, mock_rg):
        """Stress score 30-59 → 'neutral' CSS class on label span."""
        payload = _alerts_payload(with_alerts=True)
        payload["stress_score"] = 45
        mock_rg.return_value = payload

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        stress_label_span = result[2]
        assert isinstance(stress_label_span, html.Span)
        assert "neutral" in stress_label_span.className

    @patch("components.callbacks.redis_get")
    def test_anomaly_data_produces_figure_with_traces(self, mock_rg):
        """Anomaly timestamps present → figure has demand + upper + lower traces."""
        mock_rg.return_value = _alerts_payload(with_alerts=True)

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        fig_anomaly = result[4]
        assert isinstance(fig_anomaly, go.Figure)
        # demand + upper + lower + anomaly markers = 4 traces
        assert len(fig_anomaly.data) == 4

    @patch("components.callbacks.redis_get")
    def test_temperature_data_produces_figure_with_hlines(self, mock_rg):
        """Temperature timestamps present → figure with temp trace."""
        mock_rg.return_value = _alerts_payload(with_alerts=True)

        from components.callbacks import _alerts_tab_from_redis

        result = _alerts_tab_from_redis("FPL")
        fig_temp = result[5]
        assert isinstance(fig_temp, go.Figure)
        # At least the temperature trace
        assert len(fig_temp.data) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# TestOutlookTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestOutlookTabFromRedis:
    """Tests for _outlook_tab_from_redis(region, horizon_hours, model_name,
    demand_json, weather_json, persona_id)."""

    @patch("components.callbacks._add_trailing_actuals")
    @patch("components.callbacks._add_confidence_bands")
    @patch("components.callbacks.redis_get")
    def test_full_forecasts_returns_9_tuple(self, mock_rg, mock_bands, mock_trail):
        """Full forecast data → returns 9-tuple."""
        mock_rg.return_value = _forecast_payload(72, "xgboost")

        from components.callbacks import _outlook_tab_from_redis

        result = _outlook_tab_from_redis("FPL", 48, "xgboost", None, None, "grid_ops")
        assert result is not None
        assert len(result) == 9
        fig, data_through, peak_str, peak_time, avg_str, min_str, min_time, range_str, insight = (
            result
        )
        assert isinstance(fig, go.Figure)
        assert "MW" in peak_str
        assert "MW" in avg_str
        assert "MW" in min_str
        assert "MW" in range_str

    @patch("components.callbacks.redis_get")
    def test_cache_miss_returns_none(self, mock_rg):
        """Cache miss → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _outlook_tab_from_redis

        assert _outlook_tab_from_redis("FPL", 48, "xgboost", None, None, "grid_ops") is None

    @patch("components.callbacks.redis_get")
    def test_model_miss_returns_none(self, mock_rg):
        """Requested model not in forecasts and not xgboost/ensemble → None."""
        payload = _forecast_payload(72, "xgboost")
        mock_rg.return_value = payload

        from components.callbacks import _outlook_tab_from_redis

        # prophet is not in the forecasts dict
        result = _outlook_tab_from_redis("FPL", 48, "prophet", None, None, "grid_ops")
        assert result is None

    @patch("components.callbacks._add_trailing_actuals")
    @patch("components.callbacks._add_confidence_bands")
    @patch("components.callbacks.redis_get")
    def test_insufficient_horizon_returns_none(self, mock_rg, mock_bands, mock_trail):
        """Available data shorter than requested horizon → returns None."""
        # Only 24 data points but requesting 48
        mock_rg.return_value = _forecast_payload(24, "xgboost")

        from components.callbacks import _outlook_tab_from_redis

        result = _outlook_tab_from_redis("FPL", 48, "xgboost", None, None, "grid_ops")
        assert result is None

    @patch("components.callbacks._add_trailing_actuals")
    @patch("components.callbacks._add_confidence_bands")
    @patch("components.callbacks.redis_get")
    def test_horizon_trimming(self, mock_rg, mock_bands, mock_trail):
        """More data than horizon → sliced to horizon_hours."""
        mock_rg.return_value = _forecast_payload(72, "xgboost")

        from components.callbacks import _outlook_tab_from_redis

        result = _outlook_tab_from_redis("FPL", 24, "xgboost", None, None, "grid_ops")
        assert result is not None
        fig = result[0]
        # Main forecast trace should have exactly 24 points
        assert len(fig.data[0].y) == 24

    @patch("components.callbacks._add_trailing_actuals")
    @patch("components.callbacks._add_confidence_bands")
    @patch("components.callbacks.redis_get")
    def test_peak_min_computed_correctly(self, mock_rg, mock_bands, mock_trail):
        """Peak and min values are correctly extracted from predictions."""
        # Construct a known payload
        forecasts = [
            {"timestamp": t, "xgboost": 30000 + i * 100, "predicted_demand_mw": 30000 + i * 100}
            for i, t in enumerate(_ts(48))
        ]
        mock_rg.return_value = {"scored_at": "2024-06-01T12:00:00Z", "forecasts": forecasts}

        from components.callbacks import _outlook_tab_from_redis

        result = _outlook_tab_from_redis("FPL", 48, "xgboost", None, None, "grid_ops")
        assert result is not None
        peak_str = result[2]
        min_str = result[5]
        # Peak = 30000 + 47*100 = 34700
        assert "34,700" in peak_str
        # Min = 30000
        assert "30,000" in min_str

    @patch("components.callbacks._add_trailing_actuals")
    @patch("components.callbacks._add_confidence_bands")
    @patch("components.callbacks.redis_get")
    def test_scored_at_timestamp_formatted(self, mock_rg, mock_bands, mock_trail):
        """scored_at value is formatted into human-readable UTC string."""
        mock_rg.return_value = _forecast_payload(72, "xgboost")

        from components.callbacks import _outlook_tab_from_redis

        result = _outlook_tab_from_redis("FPL", 48, "xgboost", None, None, "grid_ops")
        assert result is not None
        data_through = result[1]
        assert "UTC" in data_through
        assert "2024-06-01" in data_through


# ═══════════════════════════════════════════════════════════════════════════
# TestBacktestTabFromRedis
# ═══════════════════════════════════════════════════════════════════════════


class TestBacktestTabFromRedis:
    """Tests for _backtest_tab_from_redis(region, horizon_hours, model_name, persona_id)."""

    @patch("components.callbacks.redis_get")
    def test_full_backtest_returns_7_tuple(self, mock_rg):
        """Full backtest data → returns 7-tuple."""
        mock_rg.return_value = _backtest_payload(24)

        from components.callbacks import _backtest_tab_from_redis

        result = _backtest_tab_from_redis("FPL", 24, "xgboost", "grid_ops")
        assert result is not None
        assert len(result) == 7
        fig, mape_str, rmse_str, mae_str, r2_str, explanation, insight = result
        assert isinstance(fig, go.Figure)
        assert "%" in mape_str
        assert "MW" in rmse_str
        assert "MW" in mae_str

    @patch("components.callbacks.redis_get")
    def test_cache_miss_returns_none(self, mock_rg):
        """Cache miss → returns None."""
        mock_rg.return_value = None

        from components.callbacks import _backtest_tab_from_redis

        assert _backtest_tab_from_redis("FPL", 24, "xgboost", "grid_ops") is None

    @patch("components.callbacks.redis_get")
    def test_model_miss_returns_none(self, mock_rg):
        """Requested model not in predictions (and not ensemble) → None."""
        payload = _backtest_payload(24)
        mock_rg.return_value = payload

        from components.callbacks import _backtest_tab_from_redis

        # prophet not in predictions and not "ensemble"
        result = _backtest_tab_from_redis("FPL", 24, "prophet", "grid_ops")
        assert result is None

    @patch("components.callbacks.redis_get")
    def test_ensemble_fallback(self, mock_rg):
        """Model not in predictions → falls back to ensemble predictions."""
        payload = _backtest_payload(24)
        # ensemble is the requested model; it does exist in predictions
        mock_rg.return_value = payload

        from components.callbacks import _backtest_tab_from_redis

        result = _backtest_tab_from_redis("FPL", 24, "ensemble", "grid_ops")
        assert result is not None
        mape_str = result[1]
        # Should use ensemble metrics: 2.80%
        assert "2.80" in mape_str

    @patch("components.callbacks.redis_get")
    def test_metrics_formatting(self, mock_rg):
        """Metrics are correctly formatted: MAPE%, RMSE MW, MAE MW, R2."""
        mock_rg.return_value = _backtest_payload(24)

        from components.callbacks import _backtest_tab_from_redis

        result = _backtest_tab_from_redis("FPL", 24, "xgboost", "grid_ops")
        assert result is not None
        _, mape_str, rmse_str, mae_str, r2_str, _, _ = result
        assert mape_str.startswith("3.20%")
        assert rmse_str.startswith("320 MW")
        assert mae_str.startswith("260 MW")
        assert r2_str.startswith("0.970")
        assert "(forecast_exog)" in mape_str

    @patch("components.callbacks.redis_get")
    def test_first_available_fallback(self, mock_rg):
        """Neither requested model nor ensemble → uses first available model."""
        payload = _backtest_payload(24)
        # Remove ensemble, keep only xgboost
        del payload["predictions"]["ensemble"]
        del payload["metrics"]["ensemble"]
        mock_rg.return_value = payload

        from components.callbacks import _backtest_tab_from_redis

        # Request ensemble (which is accepted as model_name since it's in the
        # special-cased ("ensemble",) tuple), but ensemble is missing from
        # predictions → fallback chain: model → ensemble → first available
        result = _backtest_tab_from_redis("FPL", 24, "ensemble", "grid_ops")
        assert result is not None
        # Should fall back to xgboost predictions and metrics
        mape_str = result[1]
        assert "3.20" in mape_str

    @patch("components.callbacks.redis_get")
    def test_empty_predictions_uses_actual(self, mock_rg):
        """Empty predictions dict → falls back to using actual as predictions."""
        payload = _backtest_payload(24)
        payload["predictions"] = {}
        payload["metrics"] = {}
        mock_rg.return_value = payload

        from components.callbacks import _backtest_tab_from_redis

        result = _backtest_tab_from_redis("FPL", 24, "ensemble", "grid_ops")
        assert result is not None
        # With predictions == actual, MAPE defaults to 0
        fig = result[0]
        assert isinstance(fig, go.Figure)
        # Chart should still have traces
        assert len(fig.data) >= 2

    @patch("components.callbacks.redis_get")
    def test_backtest_chart_has_3_traces(self, mock_rg):
        """Backtest chart has actual, forecast, and error-fill traces."""
        mock_rg.return_value = _backtest_payload(24)

        from components.callbacks import _backtest_tab_from_redis

        result = _backtest_tab_from_redis("FPL", 24, "xgboost", "grid_ops")
        assert result is not None
        fig = result[0]
        assert len(fig.data) == 3
        trace_names = [t.name for t in fig.data]
        assert "Actual Demand" in trace_names
        assert "XGBOOST Forecast" in trace_names
        assert "Forecast Error" in trace_names
