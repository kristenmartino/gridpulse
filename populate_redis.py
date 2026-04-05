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
from data.eia_client import fetch_demand
from data.feature_engineering import engineer_features
from data.preprocessing import merge_demand_weather
from data.weather_client import fetch_weather
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
                f"wattcast:backtest:{region}:{horizon}",
                json.dumps(
                    {
                        "horizon": horizon,
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
