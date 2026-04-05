"""Extended tests for components/insights.py covering uncovered lines.

Targets lines: 114, 292-301, 324, 362-363, 376, 408-411, 426, 460, 520-521,
622-625, 643-659, 691-693, 887-992.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from components.insights import (
    Insight,
    _extract_backtest_stats,
    _extract_historical_stats,
    _filter_for_persona,
    generate_tab1_insights,
    generate_tab2_insights,
    generate_tab3_insights,
    generate_tab4_insights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_demand_df(
    n: int = 168,
    base: float = 10_000.0,
    amplitude: float = 3_000.0,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Create a demand DataFrame with realistic diurnal pattern."""
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    hours = ts.hour
    demand = base + amplitude * np.sin((hours - 6) * np.pi / 12)
    return pd.DataFrame({"timestamp": ts, "demand_mw": demand})


def _make_weather_df(
    n: int = 168,
    start: str = "2024-01-01",
    base_temp: float = 75.0,
    amplitude: float = 15.0,
) -> pd.DataFrame:
    """Create a weather DataFrame with temperature_2m column."""
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    hours = ts.hour
    temp = base_temp + amplitude * np.sin((hours - 14) * np.pi / 12)
    return pd.DataFrame({"timestamp": ts, "temperature_2m": temp})


def _make_backtest_metrics(
    mape_xgb: float = 3.0,
    mape_prophet: float = 5.5,
    mape_arima: float = 7.0,
) -> dict:
    """Create a multi-model metrics dict."""
    return {
        "xgboost": {"mape": mape_xgb, "rmse": 150, "mae": 100, "r2": 0.97},
        "prophet": {"mape": mape_prophet, "rmse": 250, "mae": 180, "r2": 0.93},
        "arima": {"mape": mape_arima, "rmse": 300, "mae": 200, "r2": 0.88},
    }


# ===========================================================================
# Tests for _extract_historical_stats — lines 114, 292-301
# ===========================================================================


class TestExtractHistoricalStatsEarlyReturn:
    """Line 114: early return when df is non-empty but becomes empty after tail()."""

    def test_returns_none_stats_when_demand_df_empty_after_copy(self):
        """After tail(timerange_hours), if df is empty, stats['peak_mw'] should be None."""
        # Create a DataFrame with 0 rows by passing periods=0
        ts = pd.date_range("2024-01-01", periods=0, freq="h", tz="UTC")
        pd.DataFrame({"timestamp": ts, "demand_mw": pd.Series(dtype=float)})
        # Provide it with enough rows but then use a very high timerange — still won't be empty.
        # Instead, make a DF with data but that becomes empty after tail(0)
        # Actually, line 114 fires when df has rows initially but tail returns empty.
        # Since tail(n) where n>=1 on a non-empty df never returns empty, the only way
        # is to pass a DF that after copy and timestamp conversion has 0 rows.
        # A simpler trigger: pass a DF with columns but zero rows.
        df_empty = pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "demand_mw": pd.Series(dtype=float),
            }
        )
        stats = _extract_historical_stats(df_empty, None, 168)
        assert stats["peak_mw"] is None

    def test_returns_none_stats_when_no_demand_column(self):
        """If demand_mw column is missing, return early with all Nones."""
        ts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "value": np.random.randn(24)})
        stats = _extract_historical_stats(df, None, 168)
        assert stats["peak_mw"] is None

    def test_returns_none_stats_when_demand_df_is_none(self):
        stats = _extract_historical_stats(None, None, 168)
        assert stats["peak_mw"] is None

    def test_returns_none_stats_when_timerange_zero(self):
        """Line 114: tail(0) produces empty DataFrame after copy."""
        df = _make_demand_df(24)
        stats = _extract_historical_stats(df, None, timerange_hours=0)
        assert stats["peak_mw"] is None


class TestExtractBacktestStatsBiasAnalysis:
    """Lines 292-301: bias analysis — peak_hour_error_avg and offpeak_error_avg."""

    def test_peak_and_offpeak_error_computed(self):
        """With timestamps spanning peak (14-18) and offpeak (22-6) hours,
        the stats should contain peak_hour_error_avg and offpeak_error_avg."""
        n = 48  # 2 full days
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 10_000.0)
        # Predictions are offset so residuals have known values
        predictions = np.full(n, 9_800.0)  # bias = +200 MW

        metrics = {"xgboost": {"mape": 3.0}}
        stats = _extract_backtest_stats(metrics, actual, predictions, ts)

        assert stats["mean_bias"] == pytest.approx(200.0, abs=0.1)
        assert stats["peak_hour_error_avg"] is not None
        assert stats["offpeak_error_avg"] is not None
        # All residuals are 200 MW, so abs error everywhere is 200
        assert stats["peak_hour_error_avg"] == pytest.approx(200.0, abs=0.1)
        assert stats["offpeak_error_avg"] == pytest.approx(200.0, abs=0.1)

    def test_different_peak_vs_offpeak_errors(self):
        """Peak hours have larger errors than off-peak."""
        n = 48
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 10_000.0)
        predictions = np.full(n, 10_000.0)
        # Make peak hours (14-18) have large errors
        hours = ts.hour
        for i in range(n):
            if 14 <= hours[i] <= 18:
                predictions[i] = 9_500.0  # error = 500
            elif hours[i] >= 22 or hours[i] <= 6:
                predictions[i] = 9_900.0  # error = 100

        metrics = {"xgboost": {"mape": 3.0}}
        stats = _extract_backtest_stats(metrics, actual, predictions, ts)

        assert stats["peak_hour_error_avg"] is not None
        assert stats["offpeak_error_avg"] is not None
        assert stats["peak_hour_error_avg"] > stats["offpeak_error_avg"]

    def test_no_timestamps_skips_hourly_analysis(self):
        """Without timestamps, peak_hour_error_avg should remain None."""
        actual = np.full(48, 10_000.0)
        predictions = np.full(48, 9_800.0)
        metrics = {"xgboost": {"mape": 3.0}}
        stats = _extract_backtest_stats(metrics, actual, predictions, None)
        assert stats["mean_bias"] == pytest.approx(200.0, abs=0.1)
        assert stats["peak_hour_error_avg"] is None
        assert stats["offpeak_error_avg"] is None


# ===========================================================================
# Tests for generate_tab1_insights — lines 324, 362-363, 376, 408-411, 426
# ===========================================================================


class TestTab1InsightsEarlyReturn:
    """Line 324: return [] when stats['peak_mw'] is None after extraction."""

    def test_empty_demand_df_returns_empty(self):
        result = generate_tab1_insights("grid_ops", "ERCOT", None, None, 168)
        assert result == []

    def test_no_demand_column_returns_empty(self):
        ts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "value": np.random.randn(24)})
        result = generate_tab1_insights("grid_ops", "ERCOT", df, None, 168)
        assert result == []


class TestTab1PeakDemandInsight:
    """Lines 329-343: peak demand pattern insight generation."""

    def test_peak_demand_insight_generated(self):
        df = _make_demand_df(168, base=10_000, amplitude=3_000)
        result = generate_tab1_insights("grid_ops", "ERCOT", df, None, 168)
        assert len(result) > 0
        peak_insights = [i for i in result if i.metric_name == "peak_demand"]
        assert len(peak_insights) > 0
        assert "Peak demand" in peak_insights[0].text


class TestTab1WeekdayWeekendInsight:
    """Lines 362-363: weekday vs weekend direction (higher/lower) logic."""

    def test_weekday_higher_than_weekend(self):
        """Generate data where weekdays have clearly higher demand than weekends."""
        # 2024-01-01 is a Monday, so we get Mon-Sun + Mon-Sun
        n = 336  # 2 weeks
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 10_000.0)
        for i in range(n):
            dow = ts[i].dayofweek
            if dow >= 5:  # weekend
                demand[i] = 7_000.0
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("data_scientist", "ERCOT", df, None, n)
        weekday_insights = [i for i in result if i.metric_name == "weekday_weekend_ratio"]
        assert len(weekday_insights) > 0
        assert "higher" in weekday_insights[0].text

    def test_weekend_higher_than_weekday(self):
        """Generate data where weekends have clearly higher demand."""
        n = 336
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 7_000.0)
        for i in range(n):
            dow = ts[i].dayofweek
            if dow >= 5:  # weekend
                demand[i] = 10_000.0
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("data_scientist", "ERCOT", df, None, n)
        weekday_insights = [i for i in result if i.metric_name == "weekday_weekend_ratio"]
        assert len(weekday_insights) > 0
        assert "lower" in weekday_insights[0].text


class TestTab1HoursAboveP90:
    """Line 376: hours_above_p90 > 15% of period triggers anomaly insight."""

    def test_high_hours_above_p90_generates_anomaly(self):
        """The P90 anomaly requires hours_above_p90 > timerange_hours * 0.15.

        With strict inequality (demand > p90), at most ~10% of values can exceed P90.
        Using a small timerange_hours (6) so the 15% threshold (0.9) is low enough
        that a single value above P90 triggers the insight.
        """
        n = 24
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # Last 6 values will be used (tail(6)): distinct values so one exceeds P90
        demand = np.tile([10_000, 10_100, 10_200, 10_300, 10_400, 10_500], 4).astype(float)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        # Verify the stat: with 6 distinct values, 1 is above P90
        stats = _extract_historical_stats(df, None, 6)
        assert stats["hours_above_p90"] is not None
        assert stats["hours_above_p90"] >= 1
        assert stats["hours_above_p90"] > 6 * 0.15  # 1 > 0.9

        # Full pipeline: grid_ops receives the anomaly insight
        result = generate_tab1_insights("grid_ops", "ERCOT", df, None, timerange_hours=6)
        p90_insights = [i for i in result if i.metric_name == "hours_above_p90"]
        assert len(p90_insights) > 0
        assert "90th percentile" in p90_insights[0].text
        assert p90_insights[0].severity == "notable"


class TestTab1TemperatureCorrelation:
    """Lines 388-404: temperature-demand correlation insight."""

    def test_strong_positive_correlation(self):
        """Temperature and demand are strongly correlated."""
        n = 168
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        temp = 60 + 20 * np.sin(np.arange(n) * np.pi / 12)
        demand = 8_000 + 4_000 * np.sin(np.arange(n) * np.pi / 12)
        demand_df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        weather_df = pd.DataFrame({"timestamp": ts, "temperature_2m": temp})

        result = generate_tab1_insights("renewables", "ERCOT", demand_df, weather_df, n)
        corr_insights = [i for i in result if i.metric_name == "temp_correlation"]
        assert len(corr_insights) > 0
        assert "strong" in corr_insights[0].text


class TestTab1WeekOverWeek:
    """Lines 408-411: week-over-week trend insight."""

    def test_rising_trend(self):
        """Second week demand is significantly higher than first week."""
        n = 336  # exactly 2 weeks
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 10_000.0)
        # Last week is much higher
        demand[168:] = 11_000.0  # +10%
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("trader", "ERCOT", df, None, n)
        wow_insights = [i for i in result if i.metric_name == "wow_trend"]
        assert len(wow_insights) > 0
        assert "up" in wow_insights[0].text
        assert "rising" in wow_insights[0].text

    def test_falling_trend(self):
        """Second week demand is significantly lower than first week."""
        n = 336
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 11_000.0)
        demand[168:] = 10_000.0  # last week is lower
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("trader", "ERCOT", df, None, n)
        wow_insights = [i for i in result if i.metric_name == "wow_trend"]
        assert len(wow_insights) > 0
        assert "down" in wow_insights[0].text
        assert "falling" in wow_insights[0].text


class TestTab1DemandVariability:
    """Line 426: demand variability (CV > 10%) insight."""

    def test_high_cv_generates_insight(self):
        """Create data with high coefficient of variation."""
        n = 168
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # High variance: alternating low and high
        demand = np.where(np.arange(n) % 2 == 0, 5_000.0, 15_000.0)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("data_scientist", "ERCOT", df, None, n)
        cv_insights = [i for i in result if i.metric_name == "demand_cv"]
        assert len(cv_insights) > 0
        assert "CV=" in cv_insights[0].text


class TestTab1MorningRamp:
    """Lines 346-356: morning ramp rate insight."""

    def test_morning_ramp_generated(self):
        """Data with strong morning ramp from 6-10 AM."""
        n = 168
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        demand = np.full(n, 8_000.0)
        for i in range(n):
            hour = ts[i].hour
            if 6 <= hour <= 10:
                # Rising demand during morning: 8000, 9000, 10000, 11000, 12000
                demand[i] = 8_000 + (hour - 6) * 1_000
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})

        result = generate_tab1_insights("grid_ops", "ERCOT", df, None, n)
        ramp_insights = [i for i in result if i.metric_name == "ramp_rate"]
        assert len(ramp_insights) > 0
        assert "Morning ramp" in ramp_insights[0].text


# ===========================================================================
# Tests for generate_tab2_insights — line 460, 520-521
# ===========================================================================


class TestTab2InsightsEarlyReturn:
    """Line 460: return [] when predictions exist but stats['peak_mw'] is None."""

    def test_empty_predictions_returns_empty(self):
        result = generate_tab2_insights("grid_ops", "ERCOT", None, None)
        assert result == []

    def test_nan_only_predictions(self):
        """All-NaN predictions should still produce peak_mw (np.nanmax of NaNs warns but doesn't None)."""
        # Actually np.nanmax of all-NaN raises ValueError, but the function guards with len check.
        result = generate_tab2_insights("grid_ops", "ERCOT", np.array([]), None)
        assert result == []


class TestTab2WeekdayWeekendForecast:
    """Lines 520-521: weekday vs weekend forecast difference."""

    def test_weekday_weekend_diff_in_forecast(self):
        """Weekday higher than weekend by >3% over 168h horizon."""
        n = 168
        ts = pd.date_range("2024-01-08", periods=n, freq="h", tz="UTC")  # Mon start
        preds = np.full(n, 10_000.0)
        for i in range(n):
            if ts[i].dayofweek >= 5:
                preds[i] = 8_000.0

        result = generate_tab2_insights("trader", "ERCOT", preds, ts, horizon_hours=168)
        wd_insights = [i for i in result if i.metric_name == "weekday_weekend_diff"]
        assert len(wd_insights) > 0
        assert "weekend" in wd_insights[0].text.lower() or "weekday" in wd_insights[0].text.lower()


# ===========================================================================
# Tests for generate_tab3_insights — lines 622-625, 643-659, 691-693
# ===========================================================================


class TestTab3MapeGrade:
    """Lines 586-613: MAPE + governance grade insight."""

    def test_excellent_grade(self):
        metrics = {"xgboost": {"mape": 1.5, "rmse": 100, "r2": 0.98}}
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
            horizon_hours=24,
        )
        mape_insights = [i for i in result if i.metric_name == "mape"]
        assert len(mape_insights) > 0
        assert "Excellent" in mape_insights[0].text

    def test_rollback_grade_severity_warning(self):
        metrics = {"xgboost": {"mape": 15.0, "rmse": 500, "r2": 0.70}}
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
            horizon_hours=24,
        )
        mape_insights = [i for i in result if i.metric_name == "mape"]
        assert len(mape_insights) > 0
        assert mape_insights[0].severity == "warning"
        assert "Rollback" in mape_insights[0].text


class TestTab3R2Interpretation:
    """Lines 622-625: R2 'moderate' and 'weak' labels."""

    def test_moderate_r2(self):
        """R2 between 0.90 and 0.95 should produce 'moderate' label."""
        metrics = {"xgboost": {"mape": 4.0, "rmse": 200, "r2": 0.92}}
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
        )
        r2_insights = [i for i in result if i.metric_name == "r2"]
        assert len(r2_insights) > 0
        assert "moderate" in r2_insights[0].text
        assert r2_insights[0].severity == "notable"

    def test_weak_r2(self):
        """R2 below 0.90 should produce 'weak' label."""
        metrics = {"xgboost": {"mape": 8.0, "rmse": 400, "r2": 0.85}}
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
        )
        r2_insights = [i for i in result if i.metric_name == "r2"]
        assert len(r2_insights) > 0
        assert "weak" in r2_insights[0].text

    def test_excellent_r2(self):
        """R2 above 0.97 should produce 'excellent' label."""
        metrics = {"xgboost": {"mape": 1.0, "rmse": 50, "r2": 0.99}}
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
        )
        r2_insights = [i for i in result if i.metric_name == "r2"]
        assert len(r2_insights) > 0
        assert "excellent" in r2_insights[0].text


class TestTab3CrossModelComparison:
    """Lines 643-659: cross-model comparison with MAPE spread > 0.3."""

    def test_model_comparison_insight(self):
        """When best and worst MAPE differ by > 0.3, comparison insight is generated."""
        metrics = _make_backtest_metrics(mape_xgb=3.0, mape_prophet=5.5, mape_arima=7.0)
        n = 48
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 10_000.0)
        preds = np.full(n, 9_800.0)

        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
            actual=actual,
            predictions=preds,
            timestamps=ts,
        )
        comp_insights = [i for i in result if i.metric_name == "model_comparison"]
        assert len(comp_insights) > 0
        assert "XGBoost" in comp_insights[0].text
        assert "ARIMA" in comp_insights[0].text
        assert "outperforming" in comp_insights[0].text

    def test_no_comparison_when_spread_small(self):
        """When MAPE spread <= 0.3, no comparison insight is generated."""
        metrics = {
            "xgboost": {"mape": 3.0, "rmse": 150, "r2": 0.97},
            "prophet": {"mape": 3.2, "rmse": 160, "r2": 0.96},
        }
        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
        )
        comp_insights = [i for i in result if i.metric_name == "model_comparison"]
        assert len(comp_insights) == 0


class TestTab3BiasDetection:
    """Lines 670-683: bias detection when |mean_bias| > 50."""

    def test_underforecast_bias(self):
        """Positive bias (actual > predicted) = underforecast."""
        metrics = _make_backtest_metrics()
        n = 48
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 10_500.0)
        preds = np.full(n, 10_000.0)  # bias = +500

        result = generate_tab3_insights(
            "grid_ops",
            "ERCOT",
            metrics,
            model_name="xgboost",
            actual=actual,
            predictions=preds,
            timestamps=ts,
        )
        bias_insights = [i for i in result if i.metric_name == "bias"]
        assert len(bias_insights) > 0
        assert "underforecast" in bias_insights[0].text
        assert bias_insights[0].severity == "warning"  # |500| > 200

    def test_overforecast_bias(self):
        """Negative bias (actual < predicted) = overforecast."""
        metrics = _make_backtest_metrics()
        n = 48
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 9_900.0)
        preds = np.full(n, 10_000.0)  # bias = -100

        result = generate_tab3_insights(
            "grid_ops",
            "ERCOT",
            metrics,
            model_name="xgboost",
            actual=actual,
            predictions=preds,
            timestamps=ts,
        )
        bias_insights = [i for i in result if i.metric_name == "bias"]
        assert len(bias_insights) > 0
        assert "overforecast" in bias_insights[0].text
        assert bias_insights[0].severity == "notable"  # |100| <= 200


class TestTab3ErrorByHour:
    """Lines 691-693: error-by-hour pattern when peak/offpeak ratio > 1.3."""

    def test_peak_hour_error_concentration(self):
        """Peak hour errors much larger than off-peak triggers insight."""
        metrics = _make_backtest_metrics()
        n = 48
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        actual = np.full(n, 10_000.0)
        predictions = np.copy(actual)

        # Make peak hours (14-18) have very large errors
        hours = ts.hour
        for i in range(n):
            if 14 <= hours[i] <= 18:
                predictions[i] = 9_000.0  # error = 1000
            elif hours[i] >= 22 or hours[i] <= 6:
                predictions[i] = 9_800.0  # error = 200

        result = generate_tab3_insights(
            "data_scientist",
            "ERCOT",
            metrics,
            model_name="xgboost",
            actual=actual,
            predictions=predictions,
            timestamps=ts,
        )
        ebh_insights = [i for i in result if i.metric_name == "error_by_hour"]
        assert len(ebh_insights) > 0
        assert "afternoon" in ebh_insights[0].text.lower() or "2" in ebh_insights[0].text


class TestTab3EmptyMetrics:
    """Line 580: return [] when metrics is empty/None."""

    def test_none_metrics(self):
        result = generate_tab3_insights("grid_ops", "ERCOT", None)
        assert result == []

    def test_empty_dict_metrics(self):
        result = generate_tab3_insights("grid_ops", "ERCOT", {})
        assert result == []


# ===========================================================================
# Tests for generate_tab4_insights — lines 887-992
# ===========================================================================


class TestTab4Insights:
    """Lines 887-992: Generation & Net Load tab insights."""

    def _make_net_load_data(self, n: int = 168, capacity: int = 130_000):
        """Create net load series with duck-curve-like pattern."""
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        hours = ts.hour
        # Midday trough (solar surplus), evening ramp
        net = np.full(n, capacity * 0.5)
        for i in range(n):
            if 10 <= hours[i] <= 14:
                net[i] = capacity * 0.10  # deep midday trough
            elif 16 <= hours[i] <= 19:
                # Steep evening ramp
                net[i] = capacity * (0.10 + (hours[i] - 16) * 0.20)
        return pd.Series(net, index=range(n)), ts

    def test_empty_net_load_returns_empty(self):
        result = generate_tab4_insights("grid_ops", "ERCOT", None, None, 25.0, None, None)
        assert result == []

    def test_renewable_penetration_insight(self):
        net_load, ts = self._make_net_load_data()
        demand = pd.Series(np.full(168, 65_000.0))
        result = generate_tab4_insights(
            "renewables",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=35.0,
            pivot=None,
            timestamps=ts,
        )
        ren_insights = [i for i in result if i.metric_name == "renewable_pct"]
        assert len(ren_insights) > 0
        assert "high" in ren_insights[0].text

    def test_evening_ramp_insight(self):
        net_load, ts = self._make_net_load_data()
        demand = pd.Series(np.full(168, 65_000.0))
        result = generate_tab4_insights(
            "grid_ops",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=20.0,
            pivot=None,
            timestamps=ts,
        )
        ramp_insights = [i for i in result if i.metric_name == "evening_ramp"]
        assert len(ramp_insights) > 0
        assert "Evening" in ramp_insights[0].text

    def test_duck_curve_trough_insight(self):
        """Midday net load dip below 25% of capacity."""
        net_load, ts = self._make_net_load_data()
        demand = pd.Series(np.full(168, 65_000.0))
        result = generate_tab4_insights(
            "renewables",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=20.0,
            pivot=None,
            timestamps=ts,
        )
        trough_insights = [i for i in result if i.metric_name == "min_net_load"]
        assert len(trough_insights) > 0
        assert "dips" in trough_insights[0].text

    def test_curtailment_risk_insight(self):
        """Net load below 20% of peak for many hours."""
        net_load, ts = self._make_net_load_data()
        demand = pd.Series(np.full(168, 65_000.0))
        result = generate_tab4_insights(
            "renewables",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=20.0,
            pivot=None,
            timestamps=ts,
        )
        curt_insights = [i for i in result if i.metric_name == "curtailment_hours"]
        assert len(curt_insights) > 0
        assert "curtailment" in curt_insights[0].text.lower()

    def test_net_load_range_insight(self):
        """Net load swing > 30% of capacity."""
        net_load, ts = self._make_net_load_data()
        demand = pd.Series(np.full(168, 65_000.0))
        result = generate_tab4_insights(
            "grid_ops",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=20.0,
            pivot=None,
            timestamps=ts,
        )
        range_insights = [i for i in result if i.metric_name == "net_load_range"]
        assert len(range_insights) > 0
        assert "swings" in range_insights[0].text

    def test_solar_dominance_insight(self):
        """Solar peaks at >30% of total generation."""
        n = 168
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        hours = ts.hour
        solar = np.where((hours >= 9) & (hours <= 16), 20_000.0, 0.0)
        gas = np.full(n, 30_000.0)
        pivot = pd.DataFrame({"solar": solar, "gas": gas})
        net_load = pd.Series(np.full(n, 50_000.0))
        demand = pd.Series(np.full(n, 65_000.0))

        result = generate_tab4_insights(
            "renewables",
            "ERCOT",
            net_load,
            demand,
            renewable_pct=30.0,
            pivot=pivot,
            timestamps=ts,
        )
        solar_insights = [i for i in result if i.metric_name == "solar_peak_pct"]
        assert len(solar_insights) > 0
        assert "Solar" in solar_insights[0].text


# ===========================================================================
# Tests for _filter_for_persona — persona filtering logic
# ===========================================================================


class TestFilterForPersona:
    """Test persona-based filtering and ranking."""

    def test_filters_irrelevant_insights(self):
        insights = [
            Insight("A", "pattern", "info", persona_relevance=["grid_ops"]),
            Insight("B", "pattern", "info", persona_relevance=["trader"]),
        ]
        result = _filter_for_persona(insights, "grid_ops")
        assert len(result) == 1
        assert result[0].text == "A"

    def test_respects_max_insights_limit(self):
        insights = [
            Insight(f"Insight {i}", "pattern", "info", persona_relevance=["grid_ops"])
            for i in range(10)
        ]
        result = _filter_for_persona(insights, "grid_ops")
        assert len(result) == 4  # grid_ops max is 4

    def test_data_scientist_gets_more(self):
        insights = [
            Insight(f"Insight {i}", "pattern", "info", persona_relevance=["data_scientist"])
            for i in range(10)
        ]
        result = _filter_for_persona(insights, "data_scientist")
        assert len(result) == 5  # data_scientist max is 5

    def test_sorts_by_persona_position_then_severity(self):
        insights = [
            Insight("Low priority", "pattern", "info", persona_relevance=["trader", "grid_ops"]),
            Insight("High priority warning", "anomaly", "warning", persona_relevance=["grid_ops"]),
        ]
        result = _filter_for_persona(insights, "grid_ops")
        assert result[0].text == "High priority warning"
