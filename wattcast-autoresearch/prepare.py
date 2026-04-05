#!/usr/bin/env python3
"""
prepare.py — Data Download & Feature Store Builder
====================================================
READ-ONLY: The agent must NOT modify this file.

Downloads hourly demand data from EIA API and weather data from Open-Meteo,
builds the feature store, and splits into train/val sets.

Usage:
    python prepare.py

Requires EIA_API_KEY environment variable for real data.
Falls back to synthetic data generation for development/testing.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Grid Region Configuration
# ─────────────────────────────────────────────

REGIONS = {
    "ERCOT":  {"lat": 31.0, "lon": -97.0,  "eia_id": "ERCO", "base_demand_mw": 45000, "temp_sensitivity": 1.8},
    "PJM":    {"lat": 39.5, "lon": -77.0,  "eia_id": "PJM",  "base_demand_mw": 85000, "temp_sensitivity": 1.4},
    "CAISO":  {"lat": 37.0, "lon": -120.0, "eia_id": "CISO", "base_demand_mw": 28000, "temp_sensitivity": 1.2},
    "MISO":   {"lat": 41.0, "lon": -90.0,  "eia_id": "MISO", "base_demand_mw": 60000, "temp_sensitivity": 1.5},
    "SPP":    {"lat": 35.0, "lon": -98.0,  "eia_id": "SWPP", "base_demand_mw": 32000, "temp_sensitivity": 1.6},
    "NYISO":  {"lat": 42.5, "lon": -74.0,  "eia_id": "NYIS", "base_demand_mw": 18000, "temp_sensitivity": 1.3},
    "ISONE":  {"lat": 42.0, "lon": -72.0,  "eia_id": "ISNE", "base_demand_mw": 14000, "temp_sensitivity": 1.4},
    "SECO":   {"lat": 33.5, "lon": -84.0,  "eia_id": "SOCO", "base_demand_mw": 35000, "temp_sensitivity": 1.7},
}

# Weather features from Open-Meteo
WEATHER_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
    "apparent_temperature", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "precipitation", "rain", "snowfall",
]

# ─────────────────────────────────────────────
# Synthetic Data Generator
# (Used when EIA/Open-Meteo APIs unavailable)
# ─────────────────────────────────────────────

def generate_synthetic_data(n_days: int = 730) -> pd.DataFrame:
    """
    Generate realistic synthetic energy demand + weather data.
    Models daily/weekly/seasonal patterns, weather correlation,
    and region-specific characteristics.
    """
    np.random.seed(42)
    rows = []
    
    start_date = datetime(2023, 1, 1)
    hours = n_days * 24
    
    for region_name, cfg in REGIONS.items():
        base = cfg["base_demand_mw"]
        temp_sens = cfg["temp_sensitivity"]
        lat = cfg["lat"]
        
        for h in range(hours):
            ts = start_date + timedelta(hours=h)
            hour = ts.hour
            dow = ts.weekday()  # 0=Mon, 6=Sun
            month = ts.month
            day_of_year = ts.timetuple().tm_yday
            
            # Seasonal temperature curve (varies by latitude)
            seasonal_temp = 15 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            seasonal_temp -= (lat - 35) * 0.4  # latitude adjustment
            
            # Diurnal temperature cycle
            diurnal = -5 + 10 * np.sin(2 * np.pi * (hour - 6) / 24)
            temperature = seasonal_temp + diurnal + np.random.normal(0, 3)
            
            # Derived weather
            humidity = np.clip(60 + 20 * np.sin(2 * np.pi * (hour - 18) / 24) + np.random.normal(0, 10), 20, 100)
            dewpoint = temperature - ((100 - humidity) / 5)
            apparent_temp = temperature + 0.33 * (humidity / 100 * 6.105 * np.exp(17.27 * temperature / (237.7 + temperature))) - 4.0
            pressure = 1013 + np.random.normal(0, 5) - 0.12 * (lat - 35)
            cloud_cover = np.clip(np.random.beta(2, 3) * 100, 0, 100)
            wind_speed = np.abs(np.random.normal(10, 5))
            wind_dir = np.random.uniform(0, 360)
            wind_gust = wind_speed * (1.5 + np.random.exponential(0.3))
            solar_factor = max(0, np.sin(2 * np.pi * (hour - 6) / 24)) * (1 - cloud_cover / 150)
            shortwave = solar_factor * (800 + np.random.normal(0, 50))
            direct_rad = shortwave * 0.7
            diffuse_rad = shortwave * 0.3
            precip = np.random.exponential(0.5) if np.random.random() < 0.15 else 0
            rain = precip if temperature > 2 else 0
            snow = precip if temperature <= 2 else 0
            
            # Demand model
            # Base load
            demand = base * 0.6
            
            # Daily pattern (peaks at 9am and 7pm)
            hour_factor = 0.7 + 0.3 * (
                0.6 * np.exp(-((hour - 9) ** 2) / 8) +
                0.8 * np.exp(-((hour - 19) ** 2) / 10) +
                0.2 * np.exp(-((hour - 14) ** 2) / 12)
            )
            demand *= hour_factor
            
            # Weekend reduction
            if dow >= 5:
                demand *= 0.88
            
            # Temperature response (U-shaped: heating + cooling)
            comfort_temp = 18  # Celsius
            temp_delta = abs(temperature - comfort_temp)
            if temperature > comfort_temp:
                demand += base * 0.004 * temp_delta * temp_sens  # cooling load
            else:
                demand += base * 0.002 * temp_delta * temp_sens  # heating load
            
            # Seasonal baseline shift
            demand *= 1.0 + 0.08 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
            
            # Humidity impact on cooling
            if temperature > 25:
                demand *= 1.0 + 0.001 * max(0, humidity - 50)
            
            # Wind chill effect on heating
            if temperature < 10:
                demand *= 1.0 + 0.0005 * wind_speed
            
            # Random noise
            demand *= (1 + np.random.normal(0, 0.02))
            demand = max(demand, base * 0.3)
            
            rows.append({
                "timestamp": ts,
                "region": region_name,
                "demand_mw": round(demand, 1),
                "temperature_2m": round(temperature, 1),
                "relative_humidity_2m": round(humidity, 1),
                "dewpoint_2m": round(dewpoint, 1),
                "apparent_temperature": round(apparent_temp, 1),
                "surface_pressure": round(pressure, 1),
                "cloud_cover": round(cloud_cover, 1),
                "wind_speed_10m": round(wind_speed, 1),
                "wind_direction_10m": round(wind_dir, 1),
                "wind_gusts_10m": round(wind_gust, 1),
                "shortwave_radiation": round(max(0, shortwave), 1),
                "direct_radiation": round(max(0, direct_rad), 1),
                "diffuse_radiation": round(max(0, diffuse_rad), 1),
                "precipitation": round(precip, 2),
                "rain": round(rain, 2),
                "snowfall": round(snow, 2),
            })
    
    return pd.DataFrame(rows)


def try_fetch_real_data() -> pd.DataFrame | None:
    """Attempt to fetch real data from EIA + Open-Meteo APIs."""
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        return None
    
    try:
        import requests
        print("  Fetching real data from EIA API...")
        # Implementation for real API calls would go here
        # For now, return None to fall back to synthetic
        return None
    except Exception as e:
        print(f"  API fetch failed: {e}")
        return None


def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature store from raw demand + weather data.
    Adds temporal features, lag features, rolling statistics,
    and interaction terms.
    """
    print("  Building feature store...")
    
    df = df.sort_values(["region", "timestamp"]).copy()
    
    # ── Temporal features ──
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["timestamp"].dt.quarter
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    # ── Lag features (per region) ──
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # hours
        df[f"demand_lag_{lag}h"] = df.groupby("region")["demand_mw"].shift(lag)
        if lag <= 24:
            df[f"temp_lag_{lag}h"] = df.groupby("region")["temperature_2m"].shift(lag)
    
    # ── Rolling statistics (per region) ──
    for window in [6, 12, 24, 48, 168]:
        grp = df.groupby("region")["demand_mw"]
        df[f"demand_rolling_mean_{window}h"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"demand_rolling_std_{window}h"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # ── Temperature interaction features ──
    df["temp_squared"] = df["temperature_2m"] ** 2
    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["cooling_degree_hours"] = np.maximum(0, df["temperature_2m"] - 18)
    df["heating_degree_hours"] = np.maximum(0, 18 - df["temperature_2m"])
    df["wind_chill_factor"] = np.where(
        df["temperature_2m"] < 10,
        df["wind_speed_10m"] * (10 - df["temperature_2m"]) / 10,
        0
    )
    df["heat_index_proxy"] = np.where(
        df["temperature_2m"] > 25,
        df["temperature_2m"] + 0.5 * (df["relative_humidity_2m"] - 50) * 0.1,
        df["temperature_2m"]
    )
    
    # ── Solar features ──
    df["solar_is_up"] = (df["shortwave_radiation"] > 10).astype(int)
    df["solar_net_radiation"] = df["shortwave_radiation"] - df["diffuse_radiation"]
    
    # Drop rows with NaN from lag features
    df = df.dropna().reset_index(drop=True)
    
    return df


def split_data(df: pd.DataFrame, val_ratio: float = 0.15):
    """
    Temporal split — last val_ratio of data is validation.
    Ensures no future leakage.
    """
    print("  Splitting train/val...")
    
    splits = {}
    for region in df["region"].unique():
        region_df = df[df["region"] == region].sort_values("timestamp")
        n = len(region_df)
        split_idx = int(n * (1 - val_ratio))
        splits[region] = {
            "train": region_df.iloc[:split_idx],
            "val": region_df.iloc[split_idx:],
        }
    
    train = pd.concat([s["train"] for s in splits.values()])
    val = pd.concat([s["val"] for s in splits.values()])
    
    return train, val


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  ⚡ WattCast Data Preparation                ║")
    print("╚══════════════════════════════════════════════╝\n")
    
    # Try real data first, fall back to synthetic
    raw_df = try_fetch_real_data()
    if raw_df is None:
        print("  Using synthetic data generator (set EIA_API_KEY for real data)")
        raw_df = generate_synthetic_data(n_days=730)  # 2 years
    
    print(f"  Raw data: {len(raw_df):,} rows across {raw_df['region'].nunique()} regions")
    
    # Build features
    features_df = build_feature_store(raw_df)
    print(f"  Feature store: {len(features_df):,} rows, {len(features_df.columns)} columns")
    
    # Split
    train_df, val_df = split_data(features_df)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    
    # Save (parquet if pyarrow available, CSV fallback)
    try:
        features_df.to_parquet(DATA_DIR / "features.parquet", index=False)
        train_df.to_parquet(DATA_DIR / "train.parquet", index=False)
        val_df.to_parquet(DATA_DIR / "val.parquet", index=False)
        fmt = "parquet"
    except ImportError:
        features_df.to_csv(DATA_DIR / "features.csv", index=False)
        train_df.to_csv(DATA_DIR / "train.csv", index=False)
        val_df.to_csv(DATA_DIR / "val.csv", index=False)
        fmt = "csv"
    
    # Save metadata
    meta = {
        "regions": list(REGIONS.keys()),
        "n_features": len(features_df.columns),
        "feature_names": list(features_df.columns),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "date_range": {
            "start": str(features_df["timestamp"].min()),
            "end": str(features_df["timestamp"].max()),
        },
        "created": datetime.now().isoformat(),
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\n  ✅ Data saved to {DATA_DIR}/ ({fmt})")
    print(f"     features.{fmt}  — full feature store")
    print(f"     train.{fmt}     — training set")
    print(f"     val.{fmt}       — validation set")
    print(f"     metadata.json     — dataset metadata\n")


if __name__ == "__main__":
    main()
