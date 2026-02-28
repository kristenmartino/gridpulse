#!/usr/bin/env python3
"""
Backtest Script: Evaluate forecast accuracy with 21-day holdout.

This script:
1. Fetches historical demand data from EIA API (90 days)
2. Splits into training (first 69 days) and test (last 21 days = 504 hours)
3. Trains Prophet, ARIMA, and XGBoost models on training data
4. Generates predictions for the test period
5. Compares predictions to actual demand and reports metrics

Usage:
    python scripts/backtest.py [--region ERCOT] [--holdout-days 21]
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import structlog

from config import REGION_COORDINATES
from data.eia_client import fetch_demand
from data.weather_client import fetch_weather
from data.preprocessing import merge_demand_weather
from data.feature_engineering import engineer_features
from models.evaluation import compute_all_metrics, compute_error_by_hour

log = structlog.get_logger()


def run_backtest(region: str, holdout_days: int = 21) -> dict:
    """
    Run backtest for a region with specified holdout period.

    Args:
        region: Balancing authority code (e.g., "ERCOT", "FPL")
        holdout_days: Number of days to hold out for testing (default: 21)

    Returns:
        Dict with metrics for each model and summary statistics
    """
    holdout_hours = holdout_days * 24

    print(f"\n{'='*60}")
    print(f"BACKTEST: {region}")
    print(f"Holdout period: {holdout_days} days ({holdout_hours} hours)")
    print(f"{'='*60}\n")

    # Fetch 90 days of historical data
    print("Step 1: Fetching historical demand data...")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=90)

    demand_df = fetch_demand(
        region=region,
        start=start_date.strftime("%Y-%m-%dT%H"),
        end=end_date.strftime("%Y-%m-%dT%H"),
        use_cache=True,
    )

    if demand_df.empty:
        print(f"ERROR: No demand data available for {region}")
        return {"error": "No demand data"}

    print(f"   Retrieved {len(demand_df)} hourly records")
    print(f"   Date range: {demand_df['timestamp'].min()} to {demand_df['timestamp'].max()}")

    # Fetch weather data (uses past_days parameter, not start/end)
    print("\nStep 2: Fetching weather data...")
    weather_df = fetch_weather(
        region=region,
        past_days=92,  # Fetch ~3 months of historical weather
        forecast_days=0,  # No forecast needed for backtest
        use_cache=True,
    )

    if weather_df.empty:
        print("   WARNING: No weather data, proceeding with demand-only features")
    else:
        print(f"   Retrieved {len(weather_df)} weather records")

    # Merge and engineer features
    print("\nStep 3: Preprocessing and feature engineering...")
    merged_df = merge_demand_weather(demand_df, weather_df)
    featured_df = engineer_features(merged_df)

    # Remove rows with missing demand
    featured_df = featured_df.dropna(subset=["demand_mw"])
    print(f"   Final dataset: {len(featured_df)} records with features")

    if len(featured_df) < holdout_hours + 168:  # Need at least 1 week of training
        print(f"ERROR: Insufficient data for backtest. Need at least {holdout_hours + 168} hours.")
        return {"error": "Insufficient data"}

    # Split into train and test
    print(f"\nStep 4: Splitting data (holdout = last {holdout_days} days)...")
    cutoff_idx = len(featured_df) - holdout_hours
    train_df = featured_df.iloc[:cutoff_idx].copy()
    test_df = featured_df.iloc[cutoff_idx:].copy()

    print(f"   Training: {len(train_df)} hours ({len(train_df) // 24} days)")
    print(f"   Test:     {len(test_df)} hours ({len(test_df) // 24} days)")
    print(f"   Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"   Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    actual = test_df["demand_mw"].values
    timestamps = test_df["timestamp"]

    # Train and evaluate each model
    results = {}

    # Prophet
    print("\nStep 5a: Training Prophet model...")
    try:
        from models.prophet_model import train_prophet, predict_prophet
        prophet_model = train_prophet(train_df)
        prophet_result = predict_prophet(prophet_model, test_df, periods=len(test_df))
        prophet_pred = prophet_result["forecast"][:len(actual)]
        results["prophet"] = {
            "predictions": prophet_pred,
            "metrics": compute_all_metrics(actual, prophet_pred),
        }
        print(f"   Prophet MAPE: {results['prophet']['metrics']['mape']:.2f}%")
    except Exception as e:
        print(f"   Prophet failed: {e}")
        results["prophet"] = {"error": str(e)}

    # ARIMA
    print("\nStep 5b: Training ARIMA model...")
    try:
        from models.arima_model import train_arima, predict_arima
        arima_model = train_arima(train_df)
        arima_pred = predict_arima(arima_model, test_df, periods=len(test_df))[:len(actual)]
        results["arima"] = {
            "predictions": arima_pred,
            "metrics": compute_all_metrics(actual, arima_pred),
        }
        print(f"   ARIMA MAPE: {results['arima']['metrics']['mape']:.2f}%")
    except Exception as e:
        print(f"   ARIMA failed: {e}")
        results["arima"] = {"error": str(e)}

    # XGBoost
    print("\nStep 5c: Training XGBoost model...")
    try:
        from models.xgboost_model import train_xgboost, predict_xgboost
        xgb_model = train_xgboost(train_df)
        xgb_pred = predict_xgboost(xgb_model, test_df)[:len(actual)]
        results["xgboost"] = {
            "predictions": xgb_pred,
            "metrics": compute_all_metrics(actual, xgb_pred),
        }
        print(f"   XGBoost MAPE: {results['xgboost']['metrics']['mape']:.2f}%")
    except Exception as e:
        print(f"   XGBoost failed: {e}")
        results["xgboost"] = {"error": str(e)}

    # Ensemble (1/MAPE weighted)
    print("\nStep 5d: Computing ensemble forecast...")
    valid_models = {k: v for k, v in results.items() if "predictions" in v}

    if valid_models:
        # Compute weights from validation MAPE
        weights = {}
        total_inv_mape = 0
        for name, data in valid_models.items():
            mape = data["metrics"]["mape"]
            if mape > 0:
                inv_mape = 1.0 / mape
                weights[name] = inv_mape
                total_inv_mape += inv_mape

        # Normalize weights
        if total_inv_mape > 0:
            for name in weights:
                weights[name] /= total_inv_mape

        print(f"   Ensemble weights: {', '.join(f'{k}={v:.2f}' for k, v in weights.items())}")

        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(actual))
        for name, w in weights.items():
            ensemble_pred += valid_models[name]["predictions"] * w

        results["ensemble"] = {
            "predictions": ensemble_pred,
            "metrics": compute_all_metrics(actual, ensemble_pred),
            "weights": weights,
        }
        print(f"   Ensemble MAPE: {results['ensemble']['metrics']['mape']:.2f}%")

    # Summary report
    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)
    print(f"\nRegion: {region}")
    print(f"Test period: {test_df['timestamp'].min().strftime('%Y-%m-%d')} to {test_df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Hours evaluated: {len(actual)}")
    print(f"Mean actual demand: {np.mean(actual):,.0f} MW")
    print(f"Std actual demand: {np.std(actual):,.0f} MW")

    print("\n{:12} {:>10} {:>10} {:>10} {:>8}".format("Model", "MAPE %", "RMSE MW", "MAE MW", "R²"))
    print("-" * 54)

    for name in ["prophet", "arima", "xgboost", "ensemble"]:
        if name in results and "metrics" in results[name]:
            m = results[name]["metrics"]
            print("{:12} {:>10.2f} {:>10.0f} {:>10.0f} {:>8.4f}".format(
                name.upper(), m["mape"], m["rmse"], m["mae"], m["r2"]
            ))
        elif name in results and "error" in results[name]:
            print("{:12} {:>10}".format(name.upper(), "FAILED"))

    # Error by hour analysis
    if "ensemble" in results and "predictions" in results["ensemble"]:
        print("\n" + "-"*60)
        print("ERROR BY HOUR OF DAY (Ensemble)")
        print("-"*60)
        error_by_hour = compute_error_by_hour(
            timestamps, actual, results["ensemble"]["predictions"]
        )

        # Show hours with highest/lowest errors
        error_by_hour_sorted = error_by_hour.sort_values("mean_abs_error", ascending=False)
        print("\nHighest error hours:")
        for _, row in error_by_hour_sorted.head(5).iterrows():
            print(f"   Hour {int(row['hour']):02d}:00 - Mean abs error: {row['mean_abs_error']:,.0f} MW")

        print("\nLowest error hours:")
        for _, row in error_by_hour_sorted.tail(5).iterrows():
            print(f"   Hour {int(row['hour']):02d}:00 - Mean abs error: {row['mean_abs_error']:,.0f} MW")

    # Best/worst days
    if "ensemble" in results and "predictions" in results["ensemble"]:
        print("\n" + "-"*60)
        print("DAILY MAPE ANALYSIS")
        print("-"*60)

        test_df = test_df.copy()
        test_df["prediction"] = results["ensemble"]["predictions"]
        test_df["date"] = test_df["timestamp"].dt.date

        daily_mape = []
        for date, group in test_df.groupby("date"):
            act = group["demand_mw"].values
            pred = group["prediction"].values
            mape = np.mean(np.abs((act - pred) / act)) * 100
            daily_mape.append({"date": date, "mape": mape})

        daily_mape_df = pd.DataFrame(daily_mape).sort_values("mape")

        print("\nBest forecast days (lowest MAPE):")
        for _, row in daily_mape_df.head(5).iterrows():
            print(f"   {row['date']}: {row['mape']:.2f}%")

        print("\nWorst forecast days (highest MAPE):")
        for _, row in daily_mape_df.tail(5).iterrows():
            print(f"   {row['date']}: {row['mape']:.2f}%")

    print("\n" + "="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run forecast backtest with holdout period")
    parser.add_argument("--region", default="FPL", choices=list(REGION_COORDINATES.keys()),
                        help="Balancing authority to backtest (default: FPL)")
    parser.add_argument("--holdout-days", type=int, default=21,
                        help="Number of days to hold out for testing (default: 21)")

    args = parser.parse_args()

    results = run_backtest(args.region, args.holdout_days)

    # Return appropriate exit code
    if "error" in results:
        sys.exit(1)

    # Check if ensemble MAPE meets threshold (< 5% is good)
    if "ensemble" in results and "metrics" in results["ensemble"]:
        ensemble_mape = results["ensemble"]["metrics"]["mape"]
        if ensemble_mape > 10:
            print(f"\nWARNING: Ensemble MAPE ({ensemble_mape:.2f}%) exceeds 10% threshold")
            sys.exit(2)
        elif ensemble_mape > 5:
            print(f"\nNOTE: Ensemble MAPE ({ensemble_mape:.2f}%) is acceptable but above 5% target")

    sys.exit(0)


if __name__ == "__main__":
    main()
