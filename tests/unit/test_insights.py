"""Unit tests for components/insights.py — AI insight engine."""

import numpy as np
import pandas as pd
from dash import html

from components.insights import (
    Insight,
    _extract_backtest_stats,
    _extract_forecast_stats,
    _extract_historical_stats,
    _filter_for_persona,
    build_insight_card,
    generate_tab1_insights,
    generate_tab2_insights,
    generate_tab3_insights,
)

# ── Test helpers ──────────────────────────────────────────


def _make_demand_df(hours: int = 168) -> pd.DataFrame:
    """Create synthetic demand DataFrame for testing."""
    timestamps = pd.date_range("2024-07-01", periods=hours, freq="h", tz="UTC")
    np.random.seed(42)
    base = 25000 + 3000 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    noise = np.random.normal(0, 500, hours)
    return pd.DataFrame({"timestamp": timestamps, "demand_mw": base + noise})


def _make_weather_df(hours: int = 168) -> pd.DataFrame:
    """Create synthetic weather DataFrame for testing."""
    timestamps = pd.date_range("2024-07-01", periods=hours, freq="h", tz="UTC")
    np.random.seed(42)
    temp = 75 + 15 * np.sin(np.arange(hours) * 2 * np.pi / 24) + np.random.normal(0, 2, hours)
    return pd.DataFrame({"timestamp": timestamps, "temperature_2m": temp})


# ── Insight dataclass ─────────────────────────────────────


class TestInsightDataclass:
    def test_has_required_fields(self):
        i = Insight(
            text="Test insight",
            category="pattern",
            severity="info",
            metric_name="peak_demand",
            metric_value=28000.0,
            persona_relevance=["grid_ops", "trader"],
        )
        assert i.text == "Test insight"
        assert i.category == "pattern"
        assert i.severity == "info"
        assert i.metric_name == "peak_demand"
        assert i.metric_value == 28000.0

    def test_persona_relevance_is_list(self):
        i = Insight(
            text="x",
            category="trend",
            severity="info",
            persona_relevance=["grid_ops"],
        )
        assert isinstance(i.persona_relevance, list)

    def test_defaults(self):
        i = Insight(text="x", category="trend", severity="info")
        assert i.metric_name is None
        assert i.metric_value is None
        assert i.persona_relevance == []


# ── Persona filtering ────────────────────────────────────


class TestFilterForPersona:
    def test_removes_irrelevant_insights(self):
        insights = [
            Insight("A", "pattern", "info", persona_relevance=["grid_ops", "trader"]),
            Insight("B", "performance", "info", persona_relevance=["data_scientist"]),
        ]
        filtered = _filter_for_persona(insights, "grid_ops")
        assert len(filtered) == 1
        assert filtered[0].text == "A"

    def test_ranks_by_persona_position(self):
        insights = [
            Insight("A", "trend", "info", persona_relevance=["trader", "grid_ops"]),
            Insight("B", "anomaly", "info", persona_relevance=["grid_ops", "trader"]),
        ]
        filtered = _filter_for_persona(insights, "grid_ops")
        assert filtered[0].text == "B"  # grid_ops is first in B's list

    def test_severity_breaks_ties(self):
        insights = [
            Insight("info", "pattern", "info", persona_relevance=["grid_ops"]),
            Insight("warning", "anomaly", "warning", persona_relevance=["grid_ops"]),
        ]
        filtered = _filter_for_persona(insights, "grid_ops")
        assert filtered[0].text == "warning"  # warning ranks higher

    def test_truncates_to_max(self):
        insights = [
            Insight(f"Insight {i}", "pattern", "info", persona_relevance=["grid_ops"])
            for i in range(10)
        ]
        filtered = _filter_for_persona(insights, "grid_ops")
        assert len(filtered) == 4  # max for grid_ops

    def test_data_scientist_gets_5(self):
        insights = [
            Insight(f"Insight {i}", "pattern", "info", persona_relevance=["data_scientist"])
            for i in range(10)
        ]
        filtered = _filter_for_persona(insights, "data_scientist")
        assert len(filtered) == 5

    def test_empty_input(self):
        assert _filter_for_persona([], "grid_ops") == []

    def test_no_matching_persona(self):
        insights = [Insight("A", "pattern", "info", persona_relevance=["trader"])]
        assert _filter_for_persona(insights, "grid_ops") == []


# ── Historical stats extraction ───────────────────────────


class TestExtractHistoricalStats:
    def test_returns_expected_keys(self):
        demand_df = _make_demand_df(168)
        stats = _extract_historical_stats(demand_df, None, 168)
        assert "peak_mw" in stats
        assert "avg_mw" in stats
        assert "weekday_avg" in stats
        assert "hours_above_p90" in stats

    def test_peak_is_max(self):
        demand_df = _make_demand_df(168)
        stats = _extract_historical_stats(demand_df, None, 168)
        assert stats["peak_mw"] == float(demand_df["demand_mw"].max())

    def test_handles_empty_df(self):
        stats = _extract_historical_stats(pd.DataFrame(), None, 168)
        assert stats["peak_mw"] is None

    def test_handles_none_df(self):
        stats = _extract_historical_stats(pd.DataFrame({"demand_mw": []}), None, 168)
        assert stats["peak_mw"] is None

    def test_handles_missing_weather(self):
        demand_df = _make_demand_df(168)
        stats = _extract_historical_stats(demand_df, None, 168)
        assert stats["temp_demand_correlation"] is None

    def test_temp_correlation_with_weather(self):
        demand_df = _make_demand_df(168)
        weather_df = _make_weather_df(168)
        stats = _extract_historical_stats(demand_df, weather_df, 168)
        assert stats["temp_demand_correlation"] is not None
        assert -1 <= stats["temp_demand_correlation"] <= 1

    def test_week_over_week_needs_2_weeks(self):
        demand_df = _make_demand_df(100)  # Less than 2 weeks
        stats = _extract_historical_stats(demand_df, None, 100)
        assert stats["week_over_week_pct"] is None

    def test_week_over_week_with_enough_data(self):
        demand_df = _make_demand_df(336)  # 2 weeks
        stats = _extract_historical_stats(demand_df, None, 336)
        assert stats["week_over_week_pct"] is not None


# ── Forecast stats extraction ─────────────────────────────


class TestExtractForecastStats:
    def test_returns_peak_and_min(self):
        predictions = np.array([100, 200, 150, 180])
        timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
        stats = _extract_forecast_stats(predictions, timestamps, None)
        assert stats["peak_mw"] == 200
        assert stats["min_mw"] == 100

    def test_range_calculation(self):
        predictions = np.array([100, 200, 150])
        timestamps = pd.date_range("2024-01-01", periods=3, freq="h")
        stats = _extract_forecast_stats(predictions, timestamps, None)
        assert stats["range_mw"] == 100

    def test_ramp_rate_calculation(self):
        predictions = np.array([100, 200, 250, 150])
        timestamps = pd.date_range("2024-01-01", periods=4, freq="h")
        stats = _extract_forecast_stats(predictions, timestamps, None)
        assert stats["max_hourly_ramp"] == 100

    def test_handles_none_predictions(self):
        stats = _extract_forecast_stats(None, None, None)
        assert stats["peak_mw"] is None

    def test_handles_empty_predictions(self):
        stats = _extract_forecast_stats(np.array([]), None, None)
        assert stats["peak_mw"] is None


# ── Backtest stats extraction ─────────────────────────────


class TestExtractBacktestStats:
    def test_identifies_best_model(self):
        metrics = {
            "xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974},
            "prophet": {"mape": 2.8, "rmse": 450, "mae": 320, "r2": 0.967},
            "ensemble": {"mape": 1.9, "rmse": 340, "mae": 250, "r2": 0.979},
        }
        stats = _extract_backtest_stats(metrics, None, None, None)
        assert stats["best_model"] == "ensemble"
        assert stats["worst_model"] == "prophet"

    def test_mape_spread(self):
        metrics = {
            "xgboost": {"mape": 2.0, "rmse": 300, "mae": 200, "r2": 0.98},
            "prophet": {"mape": 4.0, "rmse": 600, "mae": 400, "r2": 0.96},
        }
        stats = _extract_backtest_stats(metrics, None, None, None)
        assert abs(stats["mape_spread"] - 2.0) < 0.01

    def test_bias_detection(self):
        actual = np.array([100, 200, 300, 400, 500])
        predictions = np.array([90, 190, 290, 390, 490])  # Systematic underforecast
        metrics = {"xgboost": {"mape": 3.0, "rmse": 10, "mae": 10, "r2": 0.99}}
        stats = _extract_backtest_stats(metrics, actual, predictions, None)
        assert stats["mean_bias"] == 10.0  # actual - predicted = +10

    def test_handles_empty_metrics(self):
        stats = _extract_backtest_stats({}, None, None, None)
        assert stats["best_model"] is None

    def test_handles_none_metrics(self):
        stats = _extract_backtest_stats(None, None, None, None)
        assert stats["best_model"] is None


# ── Tab 1 insight generation ──────────────────────────────


class TestGenerateTab1Insights:
    def test_returns_list_of_insights(self):
        demand_df = _make_demand_df(168)
        insights = generate_tab1_insights("grid_ops", "FPL", demand_df, None)
        assert isinstance(insights, list)
        assert all(isinstance(i, Insight) for i in insights)

    def test_returns_nonempty_for_valid_data(self):
        demand_df = _make_demand_df(168)
        insights = generate_tab1_insights("grid_ops", "FPL", demand_df, None)
        assert len(insights) > 0

    def test_handles_none_data(self):
        insights = generate_tab1_insights("grid_ops", "FPL", None, None)
        assert insights == []

    def test_handles_empty_data(self):
        insights = generate_tab1_insights("grid_ops", "FPL", pd.DataFrame(), None)
        assert insights == []

    def test_all_personas_get_insights(self):
        demand_df = _make_demand_df(168)
        for persona in ["grid_ops", "renewables", "trader", "data_scientist"]:
            insights = generate_tab1_insights(persona, "FPL", demand_df, None)
            assert isinstance(insights, list)
            assert len(insights) > 0

    def test_with_weather(self):
        demand_df = _make_demand_df(168)
        weather_df = _make_weather_df(168)
        insights = generate_tab1_insights("renewables", "FPL", demand_df, weather_df)
        assert len(insights) > 0

    def test_max_insights_respected(self):
        demand_df = _make_demand_df(336)
        weather_df = _make_weather_df(336)
        insights = generate_tab1_insights("grid_ops", "FPL", demand_df, weather_df, 336)
        assert len(insights) <= 4


# ── Tab 2 insight generation ──────────────────────────────


class TestGenerateTab2Insights:
    def test_returns_insights_for_predictions(self):
        np.random.seed(42)
        predictions = np.random.normal(25000, 2000, 168)
        timestamps = pd.date_range("2024-01-01", periods=168, freq="h")
        insights = generate_tab2_insights("grid_ops", "FPL", predictions, timestamps)
        assert len(insights) > 0

    def test_handles_none_predictions(self):
        insights = generate_tab2_insights("grid_ops", "FPL", None, None)
        assert insights == []

    def test_handles_empty_predictions(self):
        insights = generate_tab2_insights("grid_ops", "FPL", np.array([]), None)
        assert insights == []

    def test_all_personas_get_insights(self):
        predictions = np.random.normal(25000, 2000, 168)
        timestamps = pd.date_range("2024-01-01", periods=168, freq="h")
        for persona in ["grid_ops", "renewables", "trader", "data_scientist"]:
            insights = generate_tab2_insights(persona, "FPL", predictions, timestamps)
            assert len(insights) > 0

    def test_capacity_risk_for_high_demand(self):
        # FPL capacity is 32,000 MW
        predictions = np.full(24, 30000)  # ~94% utilization
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
        insights = generate_tab2_insights("grid_ops", "FPL", predictions, timestamps)
        categories = [i.category for i in insights]
        assert "risk" in categories

    def test_model_info_included(self):
        predictions = np.full(24, 25000)
        timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
        insights = generate_tab2_insights(
            "data_scientist", "FPL", predictions, timestamps, model_name="ensemble"
        )
        texts = " ".join([i.text for i in insights])
        assert "Ensemble" in texts


# ── Tab 3 insight generation ──────────────────────────────


class TestGenerateTab3Insights:
    def test_returns_insights_for_metrics(self):
        metrics = {"xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974}}
        insights = generate_tab3_insights("data_scientist", "FPL", metrics)
        assert len(insights) > 0

    def test_handles_none_metrics(self):
        insights = generate_tab3_insights("grid_ops", "FPL", None)
        assert insights == []

    def test_handles_empty_metrics(self):
        insights = generate_tab3_insights("grid_ops", "FPL", {})
        assert insights == []

    def test_governance_grade_included(self):
        metrics = {"xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974}}
        insights = generate_tab3_insights("grid_ops", "FPL", metrics, model_name="xgboost")
        texts = " ".join([i.text for i in insights])
        assert any(word in texts.lower() for word in ["mape", "grade"])

    def test_bias_detection(self):
        metrics = {"xgboost": {"mape": 3.0, "rmse": 300, "mae": 250, "r2": 0.97}}
        actual = np.array([1000, 2000, 3000, 4000, 5000])
        predictions = np.array([700, 1700, 2700, 3700, 4700])  # bias of +300
        insights = generate_tab3_insights(
            "grid_ops",
            "FPL",
            metrics,
            actual=actual,
            predictions=predictions,
        )
        texts = " ".join([i.text for i in insights])
        assert "underforecast" in texts.lower() or "bias" in texts.lower()

    def test_ensemble_weights_shown(self):
        metrics = {"ensemble": {"mape": 1.9, "rmse": 340, "mae": 250, "r2": 0.979}}
        weights = {"xgboost": 0.50, "prophet": 0.30, "arima": 0.20}
        insights = generate_tab3_insights(
            "data_scientist",
            "FPL",
            metrics,
            model_name="ensemble",
            ensemble_weights=weights,
        )
        texts = " ".join([i.text for i in insights])
        assert "XGBoost" in texts and "50%" in texts

    def test_all_personas_get_insights(self):
        metrics = {"xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974}}
        for persona in ["grid_ops", "renewables", "trader", "data_scientist"]:
            insights = generate_tab3_insights(persona, "FPL", metrics, model_name="xgboost")
            assert len(insights) > 0


# ── UI card builder ───────────────────────────────────────


class TestBuildInsightCard:
    def test_returns_div(self):
        insights = [Insight("Test", "pattern", "info", persona_relevance=["grid_ops"])]
        card = build_insight_card(insights, "grid_ops", "tab-forecast")
        assert isinstance(card, html.Div)

    def test_empty_insights_returns_empty_div(self):
        card = build_insight_card([], "grid_ops", "tab-forecast")
        assert isinstance(card, html.Div)
        # Should have no children
        assert card.children is None or card.children == ""

    def test_limits_to_max_insights(self):
        insights = [
            Insight(f"Insight {i}", "pattern", "info", persona_relevance=["grid_ops"])
            for i in range(10)
        ]
        card = build_insight_card(insights, "grid_ops", "tab-forecast", max_insights=3)
        assert isinstance(card, html.Div)

    def test_has_persona_color_border(self):
        insights = [Insight("Test", "pattern", "info", persona_relevance=["grid_ops"])]
        card = build_insight_card(insights, "grid_ops", "tab-forecast")
        # Grid ops color is #1f77b4
        assert "#1f77b4" in card.style.get("borderLeft", "")

    def test_invalid_persona_graceful(self):
        insights = [Insight("Test", "pattern", "info", persona_relevance=["grid_ops"])]
        card = build_insight_card(insights, "nonexistent_persona", "tab-forecast")
        assert isinstance(card, html.Div)

    def test_all_severity_colors(self):
        insights = [
            Insight("Info", "pattern", "info", persona_relevance=["grid_ops"]),
            Insight("Notable", "anomaly", "notable", persona_relevance=["grid_ops"]),
            Insight("Warning", "risk", "warning", persona_relevance=["grid_ops"]),
        ]
        card = build_insight_card(insights, "grid_ops", "tab-forecast")
        assert isinstance(card, html.Div)
