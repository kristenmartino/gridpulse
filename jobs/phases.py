"""
Shared phase functions for the GridPulse scheduled jobs.

Both the hourly scoring job and the daily training job need to:

1. Fetch demand + weather + generation for a region.
2. Engineer features.
3. Write actuals / weather / generation payloads to Redis for the web tier
   to read.

The scoring job additionally predicts forward-looking demand, writes
forecast / alerts / diagnostics / weather-correlation Redis entries, and
refreshes ``wattcast:meta:last_scored``.

The training job additionally trains new model artifacts, persists them to
GCS via :mod:`models.persistence`, recomputes backtests, and refreshes
``wattcast:meta:last_trained``.

Design:
- Every phase returns a structured result (``PhaseResult``) rather than
  raising so a single region's failure can't abort a whole job run.
- No module-level state. Both jobs should be safely invokable from a
  single container without cross-talk.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import (
    EIA_API_KEY,
    PRECOMPUTE_MAX_WORKERS,
    REGION_COORDINATES,
)

log = structlog.get_logger()

# Redis keys + TTL kept in sync with components/callbacks.py consumers.
REDIS_TTL = 86400
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"
BACKTEST_HORIZONS = (24, 168, 720)
FORECAST_HORIZON_HOURS = 720

_EIA_FUEL_MAP = {
    "SUN": "solar",
    "WND": "wind",
    "NG": "gas",
    "NUC": "nuclear",
    "COL": "coal",
    "WAT": "hydro",
    "OTH": "other",
}


# ── Result types ─────────────────────────────────────────────


@dataclass
class PhaseResult:
    """Result of a single-region phase execution."""

    region: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RegionData:
    """Per-region data payload shared across phases."""

    region: str
    demand_df: pd.DataFrame
    weather_df: pd.DataFrame
    featured_df: pd.DataFrame | None = None


# ── Region ordering ──────────────────────────────────────────


def ordered_regions(default_region: str | None = None) -> list[str]:
    """Return all known regions, putting ``default_region`` first when supplied."""
    all_regions = list(REGION_COORDINATES.keys())
    if default_region and default_region in all_regions:
        all_regions.remove(default_region)
        all_regions.insert(0, default_region)
    return all_regions


def _has_eia_key() -> bool:
    return bool(EIA_API_KEY) and EIA_API_KEY != "your_eia_api_key_here"


# ── Phase: data fetch ────────────────────────────────────────


def fetch_region_data(region: str) -> RegionData | None:
    """Fetch demand + weather for a region. Returns ``None`` on failure."""
    if not _has_eia_key():
        log.warning("job_fetch_skipped_no_api_key", region=region)
        return None

    from data.eia_client import fetch_demand
    from data.weather_client import fetch_weather

    try:
        demand_df = fetch_demand(region)
    except Exception as e:
        log.warning("job_fetch_demand_failed", region=region, error=str(e))
        return None

    try:
        weather_df = fetch_weather(region)
    except Exception as e:
        log.warning("job_fetch_weather_failed", region=region, error=str(e))
        return None

    if demand_df is None or weather_df is None or demand_df.empty or weather_df.empty:
        log.warning(
            "job_fetch_partial",
            region=region,
            has_demand=demand_df is not None and not demand_df.empty,
            has_weather=weather_df is not None and not weather_df.empty,
        )
        return None

    log.info(
        "job_data_fetched",
        region=region,
        demand_rows=len(demand_df),
        weather_rows=len(weather_df),
    )
    return RegionData(region=region, demand_df=demand_df, weather_df=weather_df)


def fetch_all_regions(regions: list[str], max_workers: int | None = None) -> dict[str, RegionData]:
    """Fetch data for every region in parallel."""
    workers = max_workers or PRECOMPUTE_MAX_WORKERS
    out: dict[str, RegionData] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_region_data, r): r for r in regions}
        for fut in as_completed(futures):
            region = futures[fut]
            try:
                data = fut.result()
                if data is not None:
                    out[region] = data
            except Exception as e:
                log.warning("job_fetch_error", region=region, error=str(e))
    return out


def engineer_region_features(data: RegionData) -> pd.DataFrame | None:
    """Run feature engineering and store the result on ``data``."""
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather

    try:
        merged = merge_demand_weather(data.demand_df, data.weather_df)
        featured = engineer_features(merged).dropna(subset=["demand_mw"])
        if len(featured) < 168:
            log.warning(
                "job_insufficient_feature_rows",
                region=data.region,
                rows=len(featured),
            )
            return None
        data.featured_df = featured.reset_index(drop=True)
        return data.featured_df
    except Exception as e:
        log.warning("job_feature_engineering_failed", region=data.region, error=str(e))
        return None


# ── Phase: Redis writes (shared by both jobs) ────────────────


def _ts_list(series: Any) -> list[str]:
    return [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in series]


def write_actuals_and_weather(data: RegionData) -> PhaseResult:
    """Write actuals + weather JSON payloads to Redis."""
    from data.redis_client import redis_set

    region = data.region
    try:
        demand_df = data.demand_df
        weather_df = data.weather_df

        actuals_payload = {
            "region": region,
            "timestamps": _ts_list(demand_df["timestamp"]),
            "demand_mw": demand_df["demand_mw"].tolist(),
        }
        redis_set(f"wattcast:actuals:{region}", actuals_payload, ttl=REDIS_TTL)

        weather_payload: dict[str, Any] = {
            "region": region,
            "timestamps": _ts_list(weather_df["timestamp"]),
        }
        for col in weather_df.columns:
            if col == "timestamp":
                continue
            weather_payload[col] = weather_df[col].tolist()
        redis_set(f"wattcast:weather:{region}", weather_payload, ttl=REDIS_TTL)

        return PhaseResult(
            region=region,
            ok=True,
            details={
                "demand_rows": len(demand_df),
                "weather_rows": len(weather_df),
            },
        )
    except Exception as e:
        log.warning("job_write_actuals_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def write_generation(region: str) -> PhaseResult:
    """Fetch generation-by-fuel for a region and write a pivoted payload to Redis."""
    from data.eia_client import fetch_generation_by_fuel
    from data.redis_client import redis_set

    if not _has_eia_key():
        return PhaseResult(region=region, ok=False, error="no_eia_api_key")

    try:
        gen_df = fetch_generation_by_fuel(region)
        if gen_df is None or gen_df.empty:
            log.info("job_generation_empty", region=region)
            return PhaseResult(region=region, ok=False, error="empty")

        gen_df["fuel_type"] = (
            gen_df["fuel_type"].map(_EIA_FUEL_MAP).fillna(gen_df["fuel_type"].str.lower())
        )
        gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])

        pivot = gen_df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        ).fillna(0)

        payload: dict[str, Any] = {
            "region": region,
            "timestamps": _ts_list(pivot.index),
        }
        for col in pivot.columns:
            payload[col] = pivot[col].tolist()

        total = pivot.sum(axis=1)
        ren_cols = [c for c in ("wind", "solar", "hydro") if c in pivot.columns]
        if ren_cols and total.mean() > 0:
            ren_pct = (pivot[ren_cols].sum(axis=1) / total * 100).tolist()
        else:
            ren_pct = [0.0] * len(pivot)
        payload["renewable_pct"] = ren_pct

        redis_set(f"wattcast:generation:{region}", payload, ttl=REDIS_TTL)
        avg_ren = float(np.mean(ren_pct)) if ren_pct else 0.0
        log.info(
            "job_generation_written",
            region=region,
            rows=len(pivot),
            avg_renewable_pct=round(avg_ren, 1),
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={"rows": len(pivot), "avg_renewable_pct": avg_ren},
        )
    except Exception as e:
        log.warning("job_generation_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Phase: forecast (scoring) ────────────────────────────────


def _build_future_feature_frame(featured: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Mirror the climatology-style future-feature builder used by populate_redis.

    Uses per-(hour, dow) historical means for non-time features so the
    forecast has sane exogenous inputs without external weather forecasts.
    """
    last_ts = featured["timestamp"].max()
    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    future_df = pd.DataFrame({"timestamp": future_timestamps})
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
    future_df["month"] = future_df["timestamp"].dt.month
    future_df["day_of_year"] = future_df["timestamp"].dt.dayofyear
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)

    feature_cols = [c for c in featured.columns if c not in ("timestamp", "demand_mw", "region")]

    hist = featured.copy()
    hist["_hour"] = hist["timestamp"].dt.hour
    hist["_dow"] = hist["timestamp"].dt.dayofweek

    non_time_cols = [c for c in feature_cols if c not in future_df.columns]
    numeric_cols = [c for c in non_time_cols if c in hist.columns]
    if numeric_cols:
        group_means = hist.groupby(["_hour", "_dow"])[numeric_cols].mean()
        future_hour = future_df["timestamp"].dt.hour
        future_dow = future_df["timestamp"].dt.dayofweek
        last_row = featured.iloc[-1]
        for col in numeric_cols:
            values = np.empty(horizon, dtype=float)
            for i in range(horizon):
                key = (future_hour.iloc[i], future_dow.iloc[i])
                if key in group_means.index:
                    values[i] = group_means.loc[key, col]
                else:
                    values[i] = float(last_row[col]) if col in last_row.index else 0.0
            future_df[col] = values

    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0

    return future_df


def _predict_one(
    model_name: str,
    model: Any,
    featured: pd.DataFrame,
    future_df: pd.DataFrame,
    horizon: int,
) -> np.ndarray | None:
    """Dispatch a single model to its predict function and return point forecasts.

    XGBoost takes the engineered future-feature frame directly. Prophet builds
    its own future frame internally from the training data + a periods arg —
    so we hand it the training ``featured`` DataFrame instead.

    Returns ``None`` on a per-model failure so the caller can degrade gracefully
    (other models in the dispatch dict still get their predictions written).
    """
    try:
        if model_name == "xgboost":
            from models.xgboost_model import predict_xgboost

            return np.asarray(predict_xgboost(model, future_df), dtype=float)
        if model_name == "prophet":
            from models.prophet_model import predict_prophet

            result = predict_prophet(model, featured, periods=horizon)
            preds = result.get("forecast")
            return np.asarray(preds, dtype=float) if preds is not None else None
    except Exception as exc:  # pragma: no cover — defensive; per-model isolation
        log.warning(
            "scoring_predict_failed",
            model=model_name,
            error=str(exc),
        )
        return None
    return None


def predict_and_write_forecast(data: RegionData, models: dict[str, Any] | None) -> PhaseResult:
    """Run all loaded forward forecasters and write ``wattcast:forecast:{region}:1h``.

    Each model in ``models`` (e.g. ``{"xgboost": <model>, "prophet": <model>}``)
    is dispatched through ``_predict_one``. The per-row Redis payload now
    carries every model that produced a finite prediction under its name
    (``row["xgboost"]``, ``row["prophet"]``, …) plus a ``predicted_demand_mw``
    key set to the primary forecast (XGBoost when available, else first
    successful model). Missing or failing models are skipped silently —
    the phase only fails when **every** model failed.

    Stage 1 of plans/scoring-job-multi-model.md (option B). Stages 2 and 3
    add ARIMA and Ensemble dispatch.
    """
    from data.redis_client import redis_set

    region = data.region
    if not models:
        return PhaseResult(region=region, ok=False, error="no_models")
    if data.featured_df is None:
        return PhaseResult(region=region, ok=False, error="no_features")

    try:
        featured = data.featured_df
        future_df = _build_future_feature_frame(featured, FORECAST_HORIZON_HOURS)
        future_ts = future_df["timestamp"]

        # Run every model defensively — a single per-model failure can't
        # abort the phase. Preserves XGBoost-only behavior when Prophet
        # isn't loaded (e.g. training job hasn't produced a Prophet pickle
        # for this region yet).
        predictions_by_model: dict[str, np.ndarray] = {}
        for name, model in models.items():
            preds = _predict_one(name, model, featured, future_df, FORECAST_HORIZON_HOURS)
            if preds is None or len(preds) < FORECAST_HORIZON_HOURS:
                continue
            predictions_by_model[name] = preds[:FORECAST_HORIZON_HOURS]

        if not predictions_by_model:
            return PhaseResult(region=region, ok=False, error="all_models_failed")

        # Pick the primary that powers ``predicted_demand_mw`` for back-compat.
        # XGBoost when available; otherwise the first successful model.
        primary_name = (
            "xgboost"
            if "xgboost" in predictions_by_model
            else next(iter(predictions_by_model.keys()))
        )
        primary = predictions_by_model[primary_name]

        scored_at = datetime.now(UTC).isoformat()
        fl: list[dict[str, Any]] = []
        for i in range(FORECAST_HORIZON_HOURS):
            row: dict[str, Any] = {
                "timestamp": future_ts.iloc[i].isoformat(),
                "predicted_demand_mw": float(primary[i]),
            }
            for name, preds in predictions_by_model.items():
                row[name] = float(preds[i])
            fl.append(row)

        redis_set(
            f"wattcast:forecast:{region}:1h",
            {
                "region": region,
                "scored_at": scored_at,
                "granularity": "1h",
                "primary_model": primary_name,
                "forecasts": fl,
            },
            ttl=REDIS_TTL,
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "horizon": FORECAST_HORIZON_HOURS,
                "points": FORECAST_HORIZON_HOURS,
                "models": sorted(predictions_by_model.keys()),
            },
        )
    except Exception as e:
        log.warning("job_forecast_write_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Phase: backtests (training) ──────────────────────────────


def write_backtests(data: RegionData) -> PhaseResult:
    """Run walk-forward backtests for the configured horizons and write to Redis.

    Imports ``_run_backtest_for_horizon`` lazily so the Dash callbacks module
    isn't pulled into the job container unless this phase runs.
    """
    from components.callbacks import _run_backtest_for_horizon
    from data.redis_client import redis_set

    region = data.region
    written: list[int] = []
    for horizon in BACKTEST_HORIZONS:
        try:
            bt = _run_backtest_for_horizon(
                data.demand_df,
                data.weather_df,
                horizon,
                "xgboost",
                region,
                DEFAULT_BACKTEST_EXOG_MODE,
                bypass_redis_guard=True,
            )
            if "error" in bt:
                log.warning(
                    "job_backtest_skipped",
                    region=region,
                    horizon=horizon,
                    reason=bt["error"],
                )
                continue

            metrics = bt["metrics"]
            actual = np.asarray(bt["actual"]).tolist()
            preds = np.asarray(bt["predictions"]).tolist()
            timestamps = [pd.Timestamp(t).isoformat() for t in bt["timestamps"]]
            residuals = (np.asarray(bt["actual"]) - np.asarray(bt["predictions"])).tolist()
            redis_set(
                f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}",
                {
                    "horizon": horizon,
                    "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
                    "exog_source": bt.get("exog_source", "climatology/naive baseline"),
                    "metrics": {
                        "xgboost": {
                            "mape": round(float(metrics["mape"]), 2),
                            "rmse": round(float(metrics["rmse"]), 2),
                            "mae": round(float(metrics["mae"]), 2),
                            "r2": round(float(metrics["r2"]), 4),
                        }
                    },
                    "actual": actual,
                    "predictions": {"xgboost": preds},
                    "timestamps": timestamps,
                    "residuals": residuals,
                },
                ttl=REDIS_TTL,
            )
            written.append(horizon)
        except Exception as e:
            log.warning(
                "job_backtest_error",
                region=region,
                horizon=horizon,
                error=str(e),
            )

    return PhaseResult(
        region=region,
        ok=len(written) > 0,
        details={"horizons_written": written},
    )


# ── Phase: diagnostics / weather-correlation / alerts ───────


def write_weather_correlation(data: RegionData) -> PhaseResult:
    """Write the weather-correlation payload consumed by the Weather tab."""
    from data.feature_engineering import compute_solar_capacity_factor, compute_wind_power
    from data.redis_client import redis_set

    region = data.region
    try:
        wc_merged = data.demand_df.merge(data.weather_df, on="timestamp", how="inner")
        corr_cols = [
            c
            for c in (
                "demand_mw",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
                "cloud_cover",
                "surface_pressure",
            )
            if c in wc_merged.columns
        ]
        if len(corr_cols) < 2:
            return PhaseResult(region=region, ok=False, error="insufficient_weather_cols")

        corr = wc_merged[corr_cols].corr()
        importance = corr["demand_mw"].drop("demand_mw").abs().sort_values(ascending=True)

        wp_arr = (
            compute_wind_power(wc_merged["wind_speed_80m"])
            if "wind_speed_80m" in wc_merged.columns
            else []
        )
        scf_arr = (
            compute_solar_capacity_factor(wc_merged["shortwave_radiation"])
            if "shortwave_radiation" in wc_merged.columns
            else []
        )

        demand_ts = wc_merged.set_index("timestamp")["demand_mw"].resample("h").mean().dropna()
        trend = demand_ts.rolling(168, center=True).mean()
        residual = demand_ts - trend

        payload: dict[str, Any] = {
            "region": region,
            "timestamps": _ts_list(wc_merged["timestamp"]),
            "demand_mw": wc_merged["demand_mw"].tolist(),
            "wind_power": wp_arr.tolist() if hasattr(wp_arr, "tolist") else list(wp_arr),
            "solar_cf": scf_arr.tolist() if hasattr(scf_arr, "tolist") else list(scf_arr),
            "correlation_matrix": {
                "cols": corr.columns.tolist(),
                "values": corr.values.tolist(),
            },
            "importance": {
                "names": importance.index.tolist(),
                "values": importance.values.tolist(),
            },
            "seasonal": {
                "timestamps": _ts_list(demand_ts.index),
                "original": demand_ts.values.tolist(),
                "trend": [float(v) if not np.isnan(v) else None for v in trend.values],
                "residual": [float(v) if not np.isnan(v) else None for v in residual.values],
            },
        }
        for col in (
            "temperature_2m",
            "wind_speed_80m",
            "shortwave_radiation",
            "relative_humidity_2m",
            "cloud_cover",
            "surface_pressure",
        ):
            payload[col] = wc_merged[col].tolist() if col in wc_merged.columns else []

        redis_set(f"wattcast:weather-correlation:{region}", payload, ttl=REDIS_TTL)
        return PhaseResult(region=region, ok=True, details={"rows": len(wc_merged)})
    except Exception as e:
        log.warning("job_weather_correlation_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def write_diagnostics(data: RegionData, xgb_model: dict | None) -> PhaseResult:
    """Write the model-diagnostics payload (actuals vs ensemble, residuals, importance)."""
    from data.redis_client import redis_set
    from models.model_service import get_forecasts

    region = data.region
    try:
        diag = get_forecasts(region, data.demand_df)
        diag_metrics = diag.get("metrics", {})
        diag_ensemble = diag.get("ensemble", data.demand_df["demand_mw"].values)
        if not isinstance(diag_ensemble, np.ndarray):
            diag_ensemble = (
                np.array(diag_ensemble)
                if diag_ensemble is not None
                else data.demand_df["demand_mw"].values
            )
        diag_actual = data.demand_df["demand_mw"].values[: len(diag_ensemble)]
        diag_residuals = diag_actual - diag_ensemble
        diag_ts = data.demand_df["timestamp"].iloc[: len(diag_ensemble)]
        hours_of_day = diag_ts.dt.hour
        error_by_hour = pd.DataFrame({"hour": hours_of_day, "abs_error": np.abs(diag_residuals)})
        hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()

        fi_names = [
            "temperature_2m",
            "demand_lag_24h",
            "hour_sin",
            "cooling_degree_days",
            "wind_speed_80m",
            "demand_roll_24h_mean",
            "heating_degree_days",
            "solar_capacity_factor",
            "day_of_week",
            "cloud_cover",
        ]
        fi_vals: list[float] = list(range(10, 0, -1))
        if xgb_model and isinstance(xgb_model, dict) and "feature_importances" in xgb_model:
            sorted_feats = sorted(
                xgb_model["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            fi_names = [f[0] for f in sorted_feats]
            fi_vals = [f[1] for f in sorted_feats]

        redis_set(
            f"wattcast:diagnostics:{region}",
            {
                "region": region,
                "timestamps": _ts_list(diag_ts),
                "actual": diag_actual.tolist(),
                "ensemble": diag_ensemble.tolist(),
                "residuals": diag_residuals.tolist(),
                "metrics": dict(diag_metrics),
                "hourly_error": {
                    "hours": hourly_error.index.tolist(),
                    "values": hourly_error.values.tolist(),
                },
                "feature_importance": {"names": fi_names, "values": fi_vals},
            },
            ttl=REDIS_TTL,
        )
        return PhaseResult(
            region=region,
            ok=True,
            details={"metrics": list(diag_metrics), "points": len(diag_ensemble)},
        )
    except Exception as e:
        log.warning("job_diagnostics_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


def write_alerts(data: RegionData) -> PhaseResult:
    """Write the alerts / stress / anomaly payload for the Risk tab."""
    from data.demo_data import generate_demo_alerts
    from data.redis_client import redis_set

    region = data.region
    try:
        alerts = generate_demo_alerts(region)
        n_crit = sum(1 for a in alerts if a["severity"] == "critical")
        n_warn = sum(1 for a in alerts if a["severity"] == "warning")
        n_info = sum(1 for a in alerts if a["severity"] == "info")
        stress = min(100, n_crit * 30 + n_warn * 15 + 20)
        stress_label = "Normal" if stress < 30 else ("Elevated" if stress < 60 else "Critical")

        recent = data.demand_df.tail(168).copy()
        rolling_mean = recent["demand_mw"].rolling(24).mean()
        rolling_std = recent["demand_mw"].rolling(24).std()
        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std
        anomalies = recent[recent["demand_mw"] > upper]

        recent_w = (
            data.weather_df.tail(168).copy()
            if data.weather_df is not None and not data.weather_df.empty
            else pd.DataFrame()
        )

        payload: dict[str, Any] = {
            "region": region,
            "alerts": alerts,
            "stress_score": stress,
            "stress_label": stress_label,
            "alert_counts": {"critical": n_crit, "warning": n_warn, "info": n_info},
            "anomaly": {
                "timestamps": _ts_list(recent["timestamp"]),
                "demand": recent["demand_mw"].tolist(),
                "upper": [float(v) if not np.isnan(v) else None for v in upper.values],
                "lower": [float(v) if not np.isnan(v) else None for v in lower.values],
                "anomaly_timestamps": _ts_list(anomalies["timestamp"])
                if not anomalies.empty
                else [],
                "anomaly_values": anomalies["demand_mw"].tolist() if not anomalies.empty else [],
            },
        }
        if not recent_w.empty and "temperature_2m" in recent_w.columns:
            payload["temperature"] = {
                "timestamps": _ts_list(recent_w["timestamp"]),
                "values": recent_w["temperature_2m"].tolist(),
            }

        redis_set(f"wattcast:alerts:{region}", payload, ttl=REDIS_TTL)
        return PhaseResult(
            region=region,
            ok=True,
            details={
                "n_critical": n_crit,
                "n_warning": n_warn,
                "n_info": n_info,
                "stress": stress,
            },
        )
    except Exception as e:
        log.warning("job_alerts_failed", region=region, error=str(e))
        return PhaseResult(region=region, ok=False, error=str(e))


# ── Meta keys ────────────────────────────────────────────────


def write_meta(key: str, extra: dict[str, Any] | None = None) -> None:
    """Write a ``wattcast:meta:{key}`` marker with current UTC timestamp."""
    from data.redis_client import redis_set

    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    redis_set(f"wattcast:meta:{key}", payload, ttl=REDIS_TTL)


# ── Orchestration helpers ────────────────────────────────────


def summarize(results: list[PhaseResult], phase: str) -> None:
    """Log a per-phase summary line with pass/fail counts."""
    ok = [r.region for r in results if r.ok]
    failed = [(r.region, r.error) for r in results if not r.ok]
    log.info(
        "job_phase_summary",
        phase=phase,
        ok_count=len(ok),
        fail_count=len(failed),
        failed=failed or None,
    )


def safe_phase(name: str, region: str, fn, *args, **kwargs) -> PhaseResult:
    """Invoke a phase function and wrap any exception as a failed ``PhaseResult``."""
    t0 = time.time()
    try:
        res = fn(*args, **kwargs)
    except Exception as e:
        log.warning(
            "job_phase_crashed",
            phase=name,
            region=region,
            error=str(e),
            tb=traceback.format_exc()[-400:],
        )
        return PhaseResult(region=region, ok=False, error=str(e))
    elapsed = time.time() - t0
    if isinstance(res, PhaseResult):
        res.details.setdefault("elapsed_s", round(elapsed, 2))
        return res
    # Allow phase functions to return ``bool`` / primitives for simplicity.
    return PhaseResult(
        region=region,
        ok=bool(res),
        details={"elapsed_s": round(elapsed, 2), "result": str(res)},
    )


__all__ = [
    "BACKTEST_HORIZONS",
    "DEFAULT_BACKTEST_EXOG_MODE",
    "FORECAST_HORIZON_HOURS",
    "PhaseResult",
    "REDIS_TTL",
    "RegionData",
    "engineer_region_features",
    "fetch_all_regions",
    "fetch_region_data",
    "ordered_regions",
    "predict_and_write_forecast",
    "safe_phase",
    "summarize",
    "write_actuals_and_weather",
    "write_alerts",
    "write_backtests",
    "write_diagnostics",
    "write_generation",
    "write_meta",
    "write_weather_correlation",
]


# ── Backwards-compat helpers (unused by callers, kept for readability) ──


def _unused_json_dumps_placeholder() -> None:  # pragma: no cover
    json.dumps({})  # keeps ``json`` import in use if redis_set paths change
