"""
Persona-aware insight engine for the Energy Forecast Dashboard.

Generates contextual, data-driven insights for each of the 3 active tabs.
Rule-based: deterministic, zero-latency, testable.

Design principles:
- OBSERVE and EXPLAIN, never RECOMMEND ACTIONS (PRD constraint)
- Persona-aware: different metrics highlighted per role
- Deterministic: same data + persona = same insights (cacheable)
- Follows personas/welcome.py pattern: data extraction -> template rendering
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dash import html

from config import REGION_CAPACITY_MW, mape_grade
from personas.config import get_persona

# ---------------------------------------------------------------------------
# Insight dataclass
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"warning": 0, "notable": 1, "info": 2}


@dataclass
class Insight:
    """A single insight observation."""

    text: str
    category: str  # pattern | anomaly | trend | driver | performance | risk
    severity: str  # info | notable | warning
    metric_name: str | None = None
    metric_value: float | None = None
    persona_relevance: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Persona filtering
# ---------------------------------------------------------------------------

_PERSONA_MAX_INSIGHTS: dict[str, int] = {
    "grid_ops": 4,
    "renewables": 4,
    "trader": 4,
    "data_scientist": 5,
}


def _filter_for_persona(insights: list[Insight], persona_id: str) -> list[Insight]:
    """Filter and rank insights by persona relevance.

    1. Remove insights where persona_id not in persona_relevance
    2. Sort by position of persona_id in each insight's relevance list (lower = better)
    3. Break ties by severity (warning > notable > info)
    4. Truncate to max_insights for the persona
    """
    relevant = [i for i in insights if persona_id in i.persona_relevance]
    relevant.sort(
        key=lambda i: (
            i.persona_relevance.index(persona_id),
            SEVERITY_ORDER.get(i.severity, 2),
        )
    )
    max_n = _PERSONA_MAX_INSIGHTS.get(persona_id, 4)
    return relevant[:max_n]


# ---------------------------------------------------------------------------
# Stats extraction helpers
# ---------------------------------------------------------------------------


def _extract_historical_stats(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    timerange_hours: int,
) -> dict:
    """Extract statistics needed for Tab 1 insight rules."""
    stats: dict = {
        "peak_mw": None,
        "peak_time": None,
        "avg_mw": None,
        "min_mw": None,
        "std_mw": None,
        "pct_above_avg": None,
        "weekday_avg": None,
        "weekend_avg": None,
        "morning_ramp_mw_per_hour": None,
        "week_over_week_pct": None,
        "temp_demand_correlation": None,
        "max_temp": None,
        "temp_at_peak_demand": None,
        "hours_above_p90": None,
    }

    if demand_df is None or len(demand_df) == 0 or "demand_mw" not in demand_df.columns:
        return stats

    df = demand_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Use the requested time range
    df = df.tail(timerange_hours)

    if len(df) == 0:
        return stats

    demand = df["demand_mw"]
    stats["peak_mw"] = float(demand.max())
    stats["min_mw"] = float(demand.min())
    stats["avg_mw"] = float(demand.mean())
    stats["std_mw"] = float(demand.std()) if len(demand) > 1 else 0.0

    if stats["avg_mw"] > 0:
        stats["pct_above_avg"] = (stats["peak_mw"] - stats["avg_mw"]) / stats["avg_mw"] * 100

    # Peak time
    peak_idx = demand.idxmax()
    if "timestamp" in df.columns:
        stats["peak_time"] = df.loc[peak_idx, "timestamp"]

    # Weekday vs weekend
    if "timestamp" in df.columns:
        df["_dow"] = df["timestamp"].dt.dayofweek
        weekday = df[df["_dow"] < 5]["demand_mw"]
        weekend = df[df["_dow"] >= 5]["demand_mw"]
        if len(weekday) > 0:
            stats["weekday_avg"] = float(weekday.mean())
        if len(weekend) > 0:
            stats["weekend_avg"] = float(weekend.mean())

    # Morning ramp (6-10 AM): max hourly increase
    if "timestamp" in df.columns and len(df) >= 24:
        df["_hour"] = df["timestamp"].dt.hour
        morning = df[(df["_hour"] >= 6) & (df["_hour"] <= 10)].copy()
        if len(morning) > 1:
            hourly_diff = morning["demand_mw"].diff()
            stats["morning_ramp_mw_per_hour"] = float(hourly_diff.max()) if not hourly_diff.isna().all() else None

    # Week-over-week trend
    if len(df) >= 336:  # 2 weeks
        this_week = df.tail(168)["demand_mw"].mean()
        last_week = df.iloc[-336:-168]["demand_mw"].mean()
        if last_week > 0:
            stats["week_over_week_pct"] = (this_week - last_week) / last_week * 100

    # Hours above 90th percentile
    p90 = demand.quantile(0.9)
    stats["hours_above_p90"] = int((demand > p90).sum())

    # Temperature-demand correlation
    if weather_df is not None and len(weather_df) > 0 and "temperature_2m" in weather_df.columns:
        wdf = weather_df.copy()
        if "timestamp" in wdf.columns:
            wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], utc=True)
        wdf = wdf.tail(timerange_hours)

        if len(wdf) > 10 and "temperature_2m" in wdf.columns:
            stats["max_temp"] = float(wdf["temperature_2m"].max())
            # Align lengths
            min_len = min(len(df), len(wdf))
            d_vals = df["demand_mw"].tail(min_len).values
            t_vals = wdf["temperature_2m"].tail(min_len).values
            mask = ~(np.isnan(d_vals) | np.isnan(t_vals))
            if mask.sum() > 10:
                stats["temp_demand_correlation"] = float(np.corrcoef(d_vals[mask], t_vals[mask])[0, 1])

            # Temperature at peak demand
            if stats["peak_time"] is not None and "timestamp" in wdf.columns:
                peak_ts = stats["peak_time"]
                closest = (wdf["timestamp"] - peak_ts).abs()
                if closest.min() < pd.Timedelta(hours=2):
                    closest_idx = closest.idxmin()
                    stats["temp_at_peak_demand"] = float(wdf.loc[closest_idx, "temperature_2m"])

    return stats


def _extract_forecast_stats(
    predictions: np.ndarray,
    timestamps: pd.DatetimeIndex,
    weather_df: pd.DataFrame | None,
) -> dict:
    """Extract statistics needed for Tab 2 insight rules."""
    stats: dict = {
        "peak_mw": None,
        "peak_time": None,
        "peak_day_of_week": None,
        "min_mw": None,
        "min_time": None,
        "avg_mw": None,
        "range_mw": None,
        "max_hourly_ramp": None,
        "max_hourly_ramp_time": None,
        "max_hourly_drop": None,
        "weekend_avg": None,
        "weekday_avg": None,
    }

    if predictions is None or len(predictions) == 0:
        return stats

    preds = np.asarray(predictions, dtype=float)
    stats["peak_mw"] = float(np.nanmax(preds))
    stats["min_mw"] = float(np.nanmin(preds))
    stats["avg_mw"] = float(np.nanmean(preds))
    stats["range_mw"] = stats["peak_mw"] - stats["min_mw"]

    peak_idx = int(np.nanargmax(preds))
    min_idx = int(np.nanargmin(preds))

    if timestamps is not None and len(timestamps) > 0:
        ts = pd.DatetimeIndex(timestamps)
        if peak_idx < len(ts):
            stats["peak_time"] = ts[peak_idx]
            stats["peak_day_of_week"] = ts[peak_idx].strftime("%A")
        if min_idx < len(ts):
            stats["min_time"] = ts[min_idx]

        # Weekday vs weekend forecast
        dow = ts.dayofweek
        weekday_mask = dow < 5
        weekend_mask = dow >= 5
        if weekday_mask.any() and len(preds) == len(ts):
            stats["weekday_avg"] = float(np.nanmean(preds[weekday_mask]))
        if weekend_mask.any() and len(preds) == len(ts):
            stats["weekend_avg"] = float(np.nanmean(preds[weekend_mask]))

    # Max hourly ramp and drop
    if len(preds) > 1:
        diffs = np.diff(preds)
        stats["max_hourly_ramp"] = float(np.nanmax(diffs))
        stats["max_hourly_drop"] = float(np.nanmin(diffs))
        ramp_idx = int(np.nanargmax(diffs))
        if timestamps is not None and ramp_idx + 1 < len(timestamps):
            stats["max_hourly_ramp_time"] = pd.DatetimeIndex(timestamps)[ramp_idx + 1]

    return stats


def _extract_backtest_stats(
    metrics: dict,
    actual: np.ndarray | None,
    predictions: np.ndarray | None,
    timestamps: pd.DatetimeIndex | None,
) -> dict:
    """Extract statistics needed for Tab 3 insight rules."""
    stats: dict = {
        "best_model": None,
        "best_mape": None,
        "worst_model": None,
        "worst_mape": None,
        "mape_spread": None,
        "mean_bias": None,
        "peak_hour_error_avg": None,
        "offpeak_error_avg": None,
    }

    if not metrics:
        return stats

    # Best and worst model by MAPE
    mape_scores = {m: v["mape"] for m, v in metrics.items() if isinstance(v, dict) and "mape" in v}
    if mape_scores:
        stats["best_model"] = min(mape_scores, key=mape_scores.get)
        stats["best_mape"] = mape_scores[stats["best_model"]]
        stats["worst_model"] = max(mape_scores, key=mape_scores.get)
        stats["worst_mape"] = mape_scores[stats["worst_model"]]
        stats["mape_spread"] = stats["worst_mape"] - stats["best_mape"]

    # Bias and error-by-hour analysis
    if actual is not None and predictions is not None and len(actual) > 0 and len(predictions) > 0:
        min_len = min(len(actual), len(predictions))
        a = np.asarray(actual[:min_len], dtype=float)
        p = np.asarray(predictions[:min_len], dtype=float)
        residuals = a - p
        stats["mean_bias"] = float(np.nanmean(residuals))

        if timestamps is not None and len(timestamps) >= min_len:
            ts = pd.DatetimeIndex(timestamps[:min_len])
            hours = ts.hour
            abs_err = np.abs(residuals)
            # Peak hours: 14-18 (2 PM - 6 PM)
            peak_mask = (hours >= 14) & (hours <= 18)
            offpeak_mask = (hours >= 22) | (hours <= 6)
            if peak_mask.any():
                stats["peak_hour_error_avg"] = float(np.nanmean(abs_err[peak_mask]))
            if offpeak_mask.any():
                stats["offpeak_error_avg"] = float(np.nanmean(abs_err[offpeak_mask]))

    return stats


# ---------------------------------------------------------------------------
# Tab 1: Historical Demand insights
# ---------------------------------------------------------------------------


def generate_tab1_insights(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
    timerange_hours: int = 168,
) -> list[Insight]:
    """Generate Historical Demand tab insights."""
    if demand_df is None or len(demand_df) == 0:
        return []

    stats = _extract_historical_stats(demand_df, weather_df, timerange_hours)
    if stats["peak_mw"] is None:
        return []

    insights: list[Insight] = []

    # Peak demand pattern
    if stats["peak_mw"] and stats["avg_mw"]:
        peak_str = f"{stats['peak_mw']:,.0f}"
        time_str = ""
        if stats["peak_time"] is not None:
            time_str = f" at {pd.Timestamp(stats['peak_time']).strftime('%a %I %p')}"
        insights.append(Insight(
            text=f"Peak demand reached {peak_str} MW{time_str}, {stats['pct_above_avg']:.0f}% above period average.",
            category="pattern",
            severity="info",
            metric_name="peak_demand",
            metric_value=stats["peak_mw"],
            persona_relevance=["grid_ops", "trader", "renewables", "data_scientist"],
        ))

    # Morning ramp rate
    if stats["morning_ramp_mw_per_hour"] is not None and stats["morning_ramp_mw_per_hour"] > 0:
        insights.append(Insight(
            text=f"Morning ramp rate peaked at {stats['morning_ramp_mw_per_hour']:,.0f} MW/hr (6\u201310 AM window).",
            category="pattern",
            severity="info",
            metric_name="ramp_rate",
            metric_value=stats["morning_ramp_mw_per_hour"],
            persona_relevance=["grid_ops", "trader", "data_scientist", "renewables"],
        ))

    # Weekday vs weekend
    if stats["weekday_avg"] and stats["weekend_avg"] and stats["weekday_avg"] > 0:
        ratio = (stats["weekday_avg"] - stats["weekend_avg"]) / stats["weekday_avg"] * 100
        if abs(ratio) > 3:
            direction = "higher" if ratio > 0 else "lower"
            insights.append(Insight(
                text=f"Weekday demand averages {abs(ratio):.0f}% {direction} than weekend ({stats['weekday_avg']:,.0f} vs {stats['weekend_avg']:,.0f} MW).",
                category="pattern",
                severity="info",
                metric_name="weekday_weekend_ratio",
                metric_value=ratio,
                persona_relevance=["data_scientist", "trader", "grid_ops", "renewables"],
            ))

    # Anomaly: hours above P90
    if stats["hours_above_p90"] is not None and stats["hours_above_p90"] > timerange_hours * 0.15:
        insights.append(Insight(
            text=f"Demand exceeded the 90th percentile for {stats['hours_above_p90']} hours during this period.",
            category="anomaly",
            severity="notable",
            metric_name="hours_above_p90",
            metric_value=float(stats["hours_above_p90"]),
            persona_relevance=["grid_ops", "trader", "renewables", "data_scientist"],
        ))

    # Temperature-demand correlation
    if stats["temp_demand_correlation"] is not None:
        r = stats["temp_demand_correlation"]
        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else None
        if strength:
            temp_str = ""
            if stats["temp_at_peak_demand"] is not None:
                temp_str = f" Peak demand coincided with {stats['temp_at_peak_demand']:.0f}\u00b0F."
            insights.append(Insight(
                text=f"Temperature shows {strength} correlation (r={r:.2f}) with demand.{temp_str}",
                category="driver",
                severity="info" if abs(r) < 0.7 else "notable",
                metric_name="temp_correlation",
                metric_value=r,
                persona_relevance=["renewables", "data_scientist", "grid_ops", "trader"],
            ))

    # Week-over-week trend
    if stats["week_over_week_pct"] is not None and abs(stats["week_over_week_pct"]) > 1.5:
        direction = "up" if stats["week_over_week_pct"] > 0 else "down"
        load_trend = "rising" if direction == "up" else "falling"
        sev = "notable" if abs(stats["week_over_week_pct"]) > 5 else "info"
        insights.append(Insight(
            text=f"Week-over-week demand {direction} {abs(stats['week_over_week_pct']):.1f}%, indicating {load_trend} load trend.",
            category="trend",
            severity=sev,
            metric_name="wow_trend",
            metric_value=stats["week_over_week_pct"],
            persona_relevance=["trader", "grid_ops", "renewables", "data_scientist"],
        ))

    # Demand variability (std)
    if stats["std_mw"] and stats["avg_mw"] and stats["avg_mw"] > 0:
        cv = stats["std_mw"] / stats["avg_mw"] * 100
        if cv > 10:
            insights.append(Insight(
                text=f"Demand variability is elevated (CV={cv:.1f}%), with \u00b1{stats['std_mw']:,.0f} MW standard deviation.",
                category="pattern",
                severity="info",
                metric_name="demand_cv",
                metric_value=cv,
                persona_relevance=["data_scientist", "trader", "grid_ops", "renewables"],
            ))

    return _filter_for_persona(insights, persona_id)


# ---------------------------------------------------------------------------
# Tab 2: Demand Forecast insights
# ---------------------------------------------------------------------------


def generate_tab2_insights(
    persona_id: str,
    region: str,
    predictions: np.ndarray | None,
    timestamps: pd.DatetimeIndex | None,
    model_name: str = "xgboost",
    horizon_hours: int = 168,
    weather_df: pd.DataFrame | None = None,
) -> list[Insight]:
    """Generate Demand Forecast tab insights."""
    if predictions is None or len(predictions) == 0:
        return []

    stats = _extract_forecast_stats(predictions, timestamps, weather_df)
    if stats["peak_mw"] is None:
        return []

    insights: list[Insight] = []
    capacity = REGION_CAPACITY_MW.get(region, 50_000)

    # Peak demand summary
    peak_str = f"{stats['peak_mw']:,.0f}"
    day_str = f" on {stats['peak_day_of_week']}" if stats.get("peak_day_of_week") else ""
    time_str = ""
    if stats["peak_time"] is not None:
        time_str = f" at {pd.Timestamp(stats['peak_time']).strftime('%I %p')}"
    insights.append(Insight(
        text=f"Peak demand of {peak_str} MW forecast{day_str}{time_str}, with {stats['range_mw']:,.0f} MW demand range.",
        category="pattern",
        severity="info",
        metric_name="peak_demand",
        metric_value=stats["peak_mw"],
        persona_relevance=["grid_ops", "trader", "renewables", "data_scientist"],
    ))

    # Ramp rate
    if stats["max_hourly_ramp"] is not None and stats["max_hourly_ramp"] > 0:
        ramp_time_str = ""
        if stats["max_hourly_ramp_time"] is not None:
            ramp_time_str = f" near {pd.Timestamp(stats['max_hourly_ramp_time']).strftime('%a %I %p')}"
        insights.append(Insight(
            text=f"Maximum hourly ramp of {stats['max_hourly_ramp']:,.0f} MW expected{ramp_time_str}.",
            category="driver",
            severity="info",
            metric_name="ramp_rate",
            metric_value=stats["max_hourly_ramp"],
            persona_relevance=["grid_ops", "trader", "data_scientist", "renewables"],
        ))

    # Capacity proximity
    if stats["peak_mw"] and capacity > 0:
        utilization_pct = stats["peak_mw"] / capacity * 100
        if utilization_pct > 75:
            sev = "warning" if utilization_pct > 85 else "notable"
            insights.append(Insight(
                text=f"Peak forecast reaches {utilization_pct:.0f}% of regional capacity ({capacity:,.0f} MW).",
                category="risk",
                severity=sev,
                metric_name="capacity_pct",
                metric_value=utilization_pct,
                persona_relevance=["grid_ops", "trader", "renewables", "data_scientist"],
            ))

    # Weekday vs weekend split
    if stats["weekday_avg"] and stats["weekend_avg"] and stats["weekday_avg"] > 0:
        diff_pct = (stats["weekday_avg"] - stats["weekend_avg"]) / stats["weekday_avg"] * 100
        if abs(diff_pct) > 3 and horizon_hours >= 168:
            lower_period = "weekend" if diff_pct > 0 else "weekday"
            insights.append(Insight(
                text=f"Weekday forecast averages {stats['weekday_avg']:,.0f} MW vs {stats['weekend_avg']:,.0f} MW on weekends ({abs(diff_pct):.0f}% {lower_period} reduction).",
                category="pattern",
                severity="info",
                metric_name="weekday_weekend_diff",
                metric_value=diff_pct,
                persona_relevance=["trader", "grid_ops", "data_scientist", "renewables"],
            ))

    # Min demand (overnight trough)
    if stats["min_mw"] and stats["min_time"] is not None:
        min_time_str = pd.Timestamp(stats["min_time"]).strftime("%a %I %p")
        insights.append(Insight(
            text=f"Minimum demand of {stats['min_mw']:,.0f} MW expected {min_time_str}.",
            category="pattern",
            severity="info",
            metric_name="min_demand",
            metric_value=stats["min_mw"],
            persona_relevance=["renewables", "trader", "grid_ops", "data_scientist"],
        ))

    # Model used
    model_label = {"xgboost": "XGBoost", "ensemble": "Ensemble"}.get(model_name, model_name)
    insights.append(Insight(
        text=f"Forecast generated using {model_label} model over {horizon_hours}-hour horizon.",
        category="performance",
        severity="info",
        metric_name="model",
        metric_value=None,
        persona_relevance=["data_scientist", "grid_ops", "trader", "renewables"],
    ))

    return _filter_for_persona(insights, persona_id)


# ---------------------------------------------------------------------------
# Tab 3: Backtest insights
# ---------------------------------------------------------------------------


def generate_tab3_insights(
    persona_id: str,
    region: str,
    metrics: dict | None,
    model_name: str = "xgboost",
    horizon_hours: int = 24,
    actual: np.ndarray | None = None,
    predictions: np.ndarray | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    ensemble_weights: dict[str, float] | None = None,
) -> list[Insight]:
    """Generate Backtest tab insights."""
    if not metrics:
        return []

    bt_stats = _extract_backtest_stats(metrics, actual, predictions, timestamps)
    insights: list[Insight] = []

    # Current model MAPE + governance grade
    model_metrics = metrics.get(model_name)
    if model_metrics and "mape" in model_metrics:
        mape_val = model_metrics["mape"]
        horizon_key = {24: "24h", 168: "7d", 720: "7d"}.get(horizon_hours, "48h")
        grade = mape_grade(mape_val, horizon_key)
        grade_label = grade.capitalize()
        model_label = {"xgboost": "XGBoost", "ensemble": "Ensemble", "prophet": "Prophet", "arima": "ARIMA"}.get(model_name, model_name)
        insights.append(Insight(
            text=f"{model_label} achieves {mape_val:.2f}% MAPE ({grade_label} grade) on {horizon_hours}-hour ahead backtest.",
            category="performance",
            severity="warning" if grade == "rollback" else "notable" if grade == "acceptable" else "info",
            metric_name="mape",
            metric_value=mape_val,
            persona_relevance=["data_scientist", "grid_ops", "trader", "renewables"],
        ))

    # R-squared interpretation
    if model_metrics and "r2" in model_metrics:
        r2 = model_metrics["r2"]
        if r2 > 0.97:
            r2_label = "excellent"
        elif r2 > 0.95:
            r2_label = "good"
        elif r2 > 0.90:
            r2_label = "moderate"
        else:
            r2_label = "weak"
        insights.append(Insight(
            text=f"R\u00b2 of {r2:.4f} indicates {r2_label} goodness of fit \u2014 model explains {r2 * 100:.1f}% of demand variance.",
            category="performance",
            severity="info" if r2 > 0.95 else "notable",
            metric_name="r2",
            metric_value=r2,
            persona_relevance=["data_scientist", "renewables", "grid_ops", "trader"],
        ))

    # Cross-model comparison
    if bt_stats["best_model"] and bt_stats["worst_model"] and bt_stats["best_model"] != bt_stats["worst_model"]:
        spread = bt_stats["mape_spread"] or 0
        if spread > 0.3:
            best_label = bt_stats["best_model"].replace("xgboost", "XGBoost").replace("prophet", "Prophet").replace("arima", "ARIMA").replace("ensemble", "Ensemble")
            worst_label = bt_stats["worst_model"].replace("xgboost", "XGBoost").replace("prophet", "Prophet").replace("arima", "ARIMA").replace("ensemble", "Ensemble")
            insights.append(Insight(
                text=f"{best_label} leads with {bt_stats['best_mape']:.2f}% MAPE, outperforming {worst_label} by {spread:.1f} percentage points.",
                category="performance",
                severity="info",
                metric_name="model_comparison",
                metric_value=spread,
                persona_relevance=["data_scientist", "grid_ops", "trader", "renewables"],
            ))

    # Bias detection
    if bt_stats["mean_bias"] is not None and abs(bt_stats["mean_bias"]) > 50:
        direction = "underforecast" if bt_stats["mean_bias"] > 0 else "overforecast"
        sev = "warning" if abs(bt_stats["mean_bias"]) > 200 else "notable"
        insights.append(Insight(
            text=f"Systematic {direction} detected: mean bias of {bt_stats['mean_bias']:+,.0f} MW.",
            category="anomaly",
            severity=sev,
            metric_name="bias",
            metric_value=bt_stats["mean_bias"],
            persona_relevance=["grid_ops", "trader", "data_scientist", "renewables"],
        ))

    # Error-by-hour pattern
    if bt_stats["peak_hour_error_avg"] is not None and bt_stats["offpeak_error_avg"] is not None:
        if bt_stats["offpeak_error_avg"] > 0:
            ratio = bt_stats["peak_hour_error_avg"] / bt_stats["offpeak_error_avg"]
            if ratio > 1.3:
                insights.append(Insight(
                    text=f"Errors concentrate in afternoon hours (2\u20136 PM), averaging {bt_stats['peak_hour_error_avg']:,.0f} MW vs {bt_stats['offpeak_error_avg']:,.0f} MW off-peak.",
                    category="pattern",
                    severity="info",
                    metric_name="error_by_hour",
                    metric_value=ratio,
                    persona_relevance=["data_scientist", "grid_ops", "trader", "renewables"],
                ))

    # Ensemble weights
    if ensemble_weights and model_name == "ensemble":
        weight_parts = []
        for m in sorted(ensemble_weights, key=ensemble_weights.get, reverse=True):
            label = m.replace("xgboost", "XGBoost").replace("prophet", "Prophet").replace("arima", "ARIMA")
            weight_parts.append(f"{label} {ensemble_weights[m]:.0%}")
        dominant = max(ensemble_weights, key=ensemble_weights.get)
        dominant_label = dominant.replace("xgboost", "XGBoost").replace("prophet", "Prophet").replace("arima", "ARIMA")
        insights.append(Insight(
            text=f"Ensemble weights: {', '.join(weight_parts)} \u2014 {dominant_label} dominates due to lowest individual MAPE.",
            category="performance",
            severity="info",
            metric_name="ensemble_weights",
            metric_value=None,
            persona_relevance=["data_scientist", "grid_ops", "trader", "renewables"],
        ))

    # RMSE context
    if model_metrics and "rmse" in model_metrics and stats_avg_mw_from_actual(actual):
        rmse = model_metrics["rmse"]
        avg_mw = float(np.nanmean(actual))
        rmse_pct = rmse / avg_mw * 100 if avg_mw > 0 else 0
        insights.append(Insight(
            text=f"RMSE of {rmse:,.0f} MW represents {rmse_pct:.1f}% of average demand ({avg_mw:,.0f} MW).",
            category="performance",
            severity="info",
            metric_name="rmse",
            metric_value=rmse,
            persona_relevance=["data_scientist", "renewables", "grid_ops", "trader"],
        ))

    return _filter_for_persona(insights, persona_id)


def stats_avg_mw_from_actual(actual: np.ndarray | None) -> bool:
    """Check if actual array is usable for statistics."""
    return actual is not None and len(actual) > 0 and not np.all(np.isnan(actual))


# ---------------------------------------------------------------------------
# UI card builder
# ---------------------------------------------------------------------------


def build_insight_card(
    insights: list[Insight],
    persona_id: str,
    tab_name: str,
    max_insights: int = 4,
) -> html.Div:
    """Build a styled insight card for display in a tab.

    Args:
        insights: List of insights (already persona-filtered).
        persona_id: Active persona (for color theming).
        tab_name: Tab identifier for labeling.
        max_insights: Maximum number of insights to show.

    Returns:
        Styled html.Div matching the dashboard dark theme.
    """
    if not insights:
        return html.Div()

    try:
        persona = get_persona(persona_id)
        persona_color = persona.color
        persona_title = persona.title
    except (KeyError, AttributeError):
        persona_color = "#64b5f6"
        persona_title = "Analyst"

    severity_colors = {
        "info": "#64b5f6",
        "notable": "#ffb74d",
        "warning": "#e94560",
    }

    insight_items = []
    for insight in insights[:max_insights]:
        sev_color = severity_colors.get(insight.severity, "#64b5f6")

        category_badge = html.Span(
            insight.category.upper(),
            style={
                "fontSize": "0.6rem",
                "padding": "1px 6px",
                "borderRadius": "3px",
                "background": f"{sev_color}20",
                "color": sev_color,
                "marginRight": "8px",
                "fontWeight": "600",
                "letterSpacing": "0.5px",
            },
        )

        insight_items.append(
            html.Div(
                [category_badge, html.Span(insight.text)],
                style={
                    "padding": "6px 0",
                    "borderBottom": "1px solid #2a2a3e",
                    "fontSize": "0.82rem",
                    "color": "#b0b0c0",
                    "lineHeight": "1.5",
                },
            )
        )

    # Remove bottom border from last item
    if insight_items:
        last_style = dict(insight_items[-1].style)
        last_style.pop("borderBottom", None)
        insight_items[-1].style = last_style

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "Insights",
                        style={
                            "fontWeight": "600",
                            "color": "#ffffff",
                            "fontSize": "0.85rem",
                        },
                    ),
                    html.Span(
                        f"  {persona_title}",
                        style={"color": "#8a8fa8", "fontSize": "0.75rem"},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(insight_items),
        ],
        className="insight-card",
        style={
            "borderLeft": f"4px solid {persona_color}",
        },
    )
