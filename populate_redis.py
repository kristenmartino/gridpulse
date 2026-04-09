"""Populate Redis cache with precomputed forecast and backtest data."""

import datetime
import json
import os
import traceback
import warnings

import numpy as np
import pandas as pd
import redis

from config import REGION_COORDINATES
from data.demo_data import generate_demo_alerts
from data.eia_client import fetch_demand
from data.feature_engineering import (
    compute_solar_capacity_factor,
    compute_wind_power,
    engineer_features,
)
from data.preprocessing import merge_demand_weather
from data.weather_client import fetch_weather
from models.model_service import get_forecasts
from models.xgboost_model import predict_xgboost, train_xgboost

warnings.filterwarnings("ignore")

REGIONS = list(REGION_COORDINATES.keys())
r = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ.get("REDIS_PORT", "6379")),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=10,
)
print(f"Connected: {r.ping()}")
TTL = 86400
DEFAULT_BACKTEST_EXOG_MODE = "forecast_exog"

for region in REGIONS:
    print(f"\n=== {region} ===")
    try:
        demand_df = fetch_demand(region)
        weather_df = fetch_weather(region)
        print(f"  Demand: {len(demand_df)}, Weather: {len(weather_df)}")
        if demand_df.empty or weather_df.empty:
            continue
        merged = merge_demand_weather(demand_df, weather_df)
        if merged.empty:
            continue
        featured = engineer_features(merged)
        if featured.empty or len(featured) < 48:
            continue
        r.set(
            f"wattcast:actuals:{region}",
            json.dumps(
                {
                    "region": region,
                    "timestamps": [
                        t.isoformat() if hasattr(t, "isoformat") else str(t)
                        for t in demand_df["timestamp"]
                    ],
                    "demand_mw": demand_df["demand_mw"].tolist(),
                }
            ),
            ex=TTL,
        )
        weather_cols = [c for c in weather_df.columns if c != "timestamp"]
        wp = {
            "region": region,
            "timestamps": [
                t.isoformat() if hasattr(t, "isoformat") else str(t)
                for t in weather_df["timestamp"]
            ],
        }
        for col in weather_cols:
            wp[col] = weather_df[col].tolist()
        r.set(f"wattcast:weather:{region}", json.dumps(wp), ex=TTL)
        model_dict = train_xgboost(featured, n_splits=3)
        print(f"  MAPE: {np.mean(model_dict['cv_scores']):.2f}%")
        last_ts = featured["timestamp"].max()
        scored_at = datetime.datetime.utcnow().isoformat()

        # Generate 720h forward-looking forecast (covers all horizon options: 24h, 7d, 30d)
        max_horizon = 720
        future_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=max_horizon,
            freq="h",
        )

        # Build future feature dataframe (mirrors _create_future_features in callbacks.py)
        feature_cols = [
            c for c in featured.columns if c not in ["timestamp", "demand_mw", "region"]
        ]
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

        # Use historical (hour, day_of_week) averages for weather/lag features
        hist = featured.copy()
        hist["_hour"] = hist["timestamp"].dt.hour
        hist["_dow"] = hist["timestamp"].dt.dayofweek
        non_time_cols = [c for c in feature_cols if c not in future_df.columns]
        numeric_cols = [c for c in non_time_cols if c in hist.columns]
        group_means = hist.groupby(["_hour", "_dow"])[numeric_cols].mean()
        future_hour = future_df["timestamp"].dt.hour
        future_dow = future_df["timestamp"].dt.dayofweek
        last_row = featured.iloc[-1]
        for col in numeric_cols:
            values = np.empty(max_horizon)
            for i in range(max_horizon):
                key = (future_hour.iloc[i], future_dow.iloc[i])
                if key in group_means.index:
                    values[i] = group_means.loc[key, col]
                else:
                    values[i] = last_row[col] if col in last_row.index else 0
            future_df[col] = values
        for col in feature_cols:
            if col not in future_df.columns:
                future_df[col] = 0

        preds = predict_xgboost(model_dict, future_df)
        fl = [
            {
                "timestamp": future_timestamps[i].isoformat(),
                "predicted_demand_mw": float(preds[i]),
                "xgboost": float(preds[i]),
            }
            for i in range(len(preds))
        ]
        r.set(
            f"wattcast:forecast:{region}:1h",
            json.dumps(
                {
                    "region": region,
                    "scored_at": scored_at,
                    "granularity": "1h",
                    "forecasts": fl,
                }
            ),
            ex=TTL,
        )
        print(f"  Forecast: {len(preds)} forward-looking points")
        for horizon in [24, 168, 720]:
            n = len(featured)
            ts = min(horizon, int(n * 0.2), n - 48)
            if ts < 24:
                continue
            td, ted = featured.iloc[:-ts], featured.iloc[-ts:]
            bm = train_xgboost(td, n_splits=min(3, max(2, len(td) // 100)))
            bp = predict_xgboost(bm, ted)
            a = ted["demand_mw"].values
            mape = float(np.mean(np.abs((a - bp) / np.where(a == 0, 1, a))) * 100)
            rmse = float(np.sqrt(np.mean((a - bp) ** 2)))
            mae = float(np.mean(np.abs(a - bp)))
            sr, st2 = np.sum((a - bp) ** 2), np.sum((a - np.mean(a)) ** 2)
            r2 = float(1 - sr / st2) if st2 > 0 else 0.0
            tss = [t.isoformat() for t in ted["timestamp"]]
            r.set(
                f"wattcast:backtest:{DEFAULT_BACKTEST_EXOG_MODE}:{region}:{horizon}",
                json.dumps(
                    {
                        "horizon": horizon,
                        "exog_mode": DEFAULT_BACKTEST_EXOG_MODE,
                        "exog_source": "climatology/naive baseline",
                        "metrics": {
                            "xgboost": {
                                "mape": round(mape, 2),
                                "rmse": round(rmse, 2),
                                "mae": round(mae, 2),
                                "r2": round(r2, 4),
                            }
                        },
                        "actual": a.tolist(),
                        "predictions": {"xgboost": bp.tolist()},
                        "timestamps": tss,
                        "residuals": (a - bp).tolist(),
                    }
                ),
                ex=TTL,
            )
            print(f"  BT h={horizon}: MAPE={mape:.2f}%")
        ng = min(len(demand_df), 2160)
        gt = [t.isoformat() for t in demand_df["timestamp"].iloc[-ng:]]
        dv = demand_df["demand_mw"].iloc[-ng:].values
        r.set(
            f"wattcast:generation:{region}",
            json.dumps(
                {
                    "region": region,
                    "timestamps": gt,
                    "coal": (dv * 0.18).tolist(),
                    "gas": (dv * 0.38).tolist(),
                    "hydro": (dv * 0.06).tolist(),
                    "nuclear": (dv * 0.19).tolist(),
                    "other": (dv * 0.02).tolist(),
                    "solar": (dv * 0.08).tolist(),
                    "wind": (dv * 0.09).tolist(),
                    "renewable_pct": [17.0] * ng,
                }
            ),
            ex=TTL,
        )
        # ── Weather Correlation ──────────────────────────────
        wc_merged = demand_df.merge(weather_df, on="timestamp", how="inner")
        corr_cols = [
            c
            for c in [
                "demand_mw",
                "temperature_2m",
                "wind_speed_80m",
                "shortwave_radiation",
                "relative_humidity_2m",
                "cloud_cover",
                "surface_pressure",
            ]
            if c in wc_merged.columns
        ]
        if len(corr_cols) >= 2:
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
            # Seasonal decomposition
            demand_ts = wc_merged.set_index("timestamp")["demand_mw"].resample("h").mean().dropna()
            trend = demand_ts.rolling(168, center=True).mean()
            residual = demand_ts - trend
            r.set(
                f"wattcast:weather-correlation:{region}",
                json.dumps(
                    {
                        "region": region,
                        "timestamps": [t.isoformat() for t in wc_merged["timestamp"]],
                        "demand_mw": wc_merged["demand_mw"].tolist(),
                        "temperature_2m": wc_merged["temperature_2m"].tolist()
                        if "temperature_2m" in wc_merged.columns
                        else [],
                        "wind_speed_80m": wc_merged["wind_speed_80m"].tolist()
                        if "wind_speed_80m" in wc_merged.columns
                        else [],
                        "shortwave_radiation": wc_merged["shortwave_radiation"].tolist()
                        if "shortwave_radiation" in wc_merged.columns
                        else [],
                        "relative_humidity_2m": wc_merged["relative_humidity_2m"].tolist()
                        if "relative_humidity_2m" in wc_merged.columns
                        else [],
                        "cloud_cover": wc_merged["cloud_cover"].tolist()
                        if "cloud_cover" in wc_merged.columns
                        else [],
                        "surface_pressure": wc_merged["surface_pressure"].tolist()
                        if "surface_pressure" in wc_merged.columns
                        else [],
                        "wind_power": wp_arr.tolist()
                        if hasattr(wp_arr, "tolist")
                        else list(wp_arr),
                        "solar_cf": scf_arr.tolist()
                        if hasattr(scf_arr, "tolist")
                        else list(scf_arr),
                        "correlation_matrix": {
                            "cols": corr.columns.tolist(),
                            "values": corr.values.tolist(),
                        },
                        "importance": {
                            "names": importance.index.tolist(),
                            "values": importance.values.tolist(),
                        },
                        "seasonal": {
                            "timestamps": [t.isoformat() for t in demand_ts.index],
                            "original": demand_ts.values.tolist(),
                            "trend": [float(v) if not np.isnan(v) else None for v in trend.values],
                            "residual": [
                                float(v) if not np.isnan(v) else None for v in residual.values
                            ],
                        },
                    }
                ),
                ex=TTL,
            )
            print(f"  Weather correlation: {len(wc_merged)} merged points")

        # ── Model Diagnostics ──────────────────────────────
        try:
            diag_forecasts = get_forecasts(region, demand_df)
            diag_metrics = diag_forecasts.get("metrics", {})
            diag_ensemble = diag_forecasts.get("ensemble", demand_df["demand_mw"].values)
            if not isinstance(diag_ensemble, np.ndarray):
                diag_ensemble = (
                    np.array(diag_ensemble)
                    if diag_ensemble is not None
                    else demand_df["demand_mw"].values
                )
            diag_actual = demand_df["demand_mw"].values[: len(diag_ensemble)]
            diag_residuals = diag_actual - diag_ensemble
            diag_ts = demand_df["timestamp"].iloc[: len(diag_ensemble)]
            hours_of_day = diag_ts.dt.hour
            error_by_hour = pd.DataFrame(
                {"hour": hours_of_day, "abs_error": np.abs(diag_residuals)}
            )
            hourly_error = error_by_hour.groupby("hour")["abs_error"].mean()
            # Feature importance from trained model
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
            fi_vals = list(range(10, 0, -1))  # defaults
            if model_dict and isinstance(model_dict, dict) and "feature_importances" in model_dict:
                imp = model_dict["feature_importances"]
                sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
                fi_names = [f[0] for f in sorted_feats]
                fi_vals = [f[1] for f in sorted_feats]
            r.set(
                f"wattcast:diagnostics:{region}",
                json.dumps(
                    {
                        "region": region,
                        "timestamps": [t.isoformat() for t in diag_ts],
                        "actual": diag_actual.tolist(),
                        "ensemble": diag_ensemble.tolist(),
                        "residuals": diag_residuals.tolist(),
                        "metrics": {k: v for k, v in diag_metrics.items()},
                        "hourly_error": {
                            "hours": hourly_error.index.tolist(),
                            "values": hourly_error.values.tolist(),
                        },
                        "feature_importance": {
                            "names": fi_names,
                            "values": fi_vals,
                        },
                    }
                ),
                ex=TTL,
            )
            print(f"  Diagnostics: {len(diag_metrics)} models, {len(diag_ensemble)} points")
        except Exception as diag_err:
            print(f"  Diagnostics SKIP: {diag_err}")

        # ── Alerts ──────────────────────────────
        alerts = generate_demo_alerts(region)
        n_crit = sum(1 for a in alerts if a["severity"] == "critical")
        n_warn = sum(1 for a in alerts if a["severity"] == "warning")
        n_info = sum(1 for a in alerts if a["severity"] == "info")
        stress = min(100, n_crit * 30 + n_warn * 15 + 20)
        stress_label = "Normal" if stress < 30 else ("Elevated" if stress < 60 else "Critical")
        # Anomaly detection from demand data
        recent = demand_df.tail(168).copy()
        rolling_mean = recent["demand_mw"].rolling(24).mean()
        rolling_std = recent["demand_mw"].rolling(24).std()
        upper = rolling_mean + 2 * rolling_std
        lower = rolling_mean - 2 * rolling_std
        anomalies = recent[recent["demand_mw"] > upper]
        # Temperature data
        recent_w = weather_df.tail(168).copy() if not weather_df.empty else pd.DataFrame()
        alert_payload = {
            "region": region,
            "alerts": alerts,
            "stress_score": stress,
            "stress_label": stress_label,
            "alert_counts": {"critical": n_crit, "warning": n_warn, "info": n_info},
            "anomaly": {
                "timestamps": [t.isoformat() for t in recent["timestamp"]],
                "demand": recent["demand_mw"].tolist(),
                "upper": [float(v) if not np.isnan(v) else None for v in upper.values],
                "lower": [float(v) if not np.isnan(v) else None for v in lower.values],
                "anomaly_timestamps": [t.isoformat() for t in anomalies["timestamp"]]
                if not anomalies.empty
                else [],
                "anomaly_values": anomalies["demand_mw"].tolist() if not anomalies.empty else [],
            },
        }
        if not recent_w.empty and "temperature_2m" in recent_w.columns:
            alert_payload["temperature"] = {
                "timestamps": [t.isoformat() for t in recent_w["timestamp"]],
                "values": recent_w["temperature_2m"].tolist(),
            }
        r.set(f"wattcast:alerts:{region}", json.dumps(alert_payload), ex=TTL)
        print(f"  Alerts: {len(alerts)} alerts, stress={stress}")

        print(f"  Done {region}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

r.set(
    "wattcast:meta:last_scored",
    json.dumps(
        {
            "scored_at": datetime.datetime.utcnow().isoformat(),
            "regions_scored": len(REGIONS),
            "mode": "populate-job",
        }
    ),
    ex=TTL,
)
print(f"\nDone! Keys: {len(r.keys('wattcast:*'))}")
