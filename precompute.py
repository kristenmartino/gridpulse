"""
Startup precomputation for the Energy Forecast Dashboard.

Runs in a background thread per gunicorn worker (--preload is omitted so
workers can serve /health immediately). Each worker warms its own in-memory
caches. SQLite cache provides cross-restart persistence.

Only XGBoost forecasts and backtests are precomputed (the default view).
Ensemble forecasts/backtests train 3 models each (~15-20s) and are deferred
to on-demand computation + cache after first user request.

Phase 1: Fetch data for all 8 regions in parallel
Phase 2: Train XGBoost + generate XGBoost predictions for all regions
Phase 3: Run XGBoost backtests for all regions

Never raises — failures log warnings and fall back to on-demand computation.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import structlog

from config import (
    EIA_API_KEY,
    PRECOMPUTE_ALL_REGIONS,
    PRECOMPUTE_DEFAULT_REGION,
    PRECOMPUTE_MAX_WORKERS,
    REGION_COORDINATES,
)

log = structlog.get_logger()

# Store fetched data so backtests can reuse without re-fetching
_region_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def precompute_all() -> None:
    """Entry point. Called from app.py at startup. Never raises."""
    t0 = time.time()
    log.info(
        "precompute_start",
        default_region=PRECOMPUTE_DEFAULT_REGION,
        all_regions=PRECOMPUTE_ALL_REGIONS,
    )

    try:
        all_regions = list(REGION_COORDINATES.keys())

        # Phase 0: Warm generation data cache (Tab 4: Generation & Net Load)
        _fetch_generation_all_parallel(all_regions)

        # Phase 1: Fetch demand + weather data for all regions
        _fetch_all_data_parallel(all_regions)

        # Phase 2: Train models + predictions for all regions
        _train_all_models_parallel(all_regions)

        # Phase 3: Backtests for all regions (24h, 168h, 720h)
        _backtest_all_parallel(all_regions)

    except Exception as e:
        log.error("precompute_failed", error=str(e))

    elapsed = time.time() - t0
    log.info("precompute_complete", elapsed_seconds=round(elapsed, 1))


def _fetch_all_data_parallel(regions: list[str]) -> None:
    """Fetch data for all regions in parallel, populating SQLite cache."""
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_data, r): r for r in regions}
        for future in as_completed(futures):
            region = futures[future]
            try:
                result = future.result()
                if result[0] is not None:
                    _region_data[region] = result
            except Exception as e:
                log.warning("precompute_fetch_failed", region=region, error=str(e))


def _train_all_models_parallel(regions: list[str]) -> None:
    """Train XGBoost + generate predictions for all regions in parallel."""
    regions_with_data = [r for r in regions if r in _region_data]
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {pool.submit(_train_region, r): r for r in regions_with_data}
        for future in as_completed(futures):
            region = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning("precompute_train_failed", region=region, error=str(e))


def _backtest_all_parallel(regions: list[str]) -> None:
    """Run XGBoost backtests for all regions in parallel.

    Only XGBoost backtests are precomputed (the default view).
    Ensemble backtests require training all 3 models (~15-20s each)
    and are deferred to on-demand computation + cache.
    """
    regions_with_data = [r for r in regions if r in _region_data]
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {}
        for r in regions_with_data:
            for horizon in [24, 168, 720]:
                futures[pool.submit(_precompute_backtest, r, horizon, "xgboost")] = (
                    r,
                    horizon,
                    "xgboost",
                )
        for future in as_completed(futures):
            region, horizon, model = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning(
                    "precompute_backtest_failed",
                    region=region,
                    horizon=horizon,
                    model=model,
                    error=str(e),
                )


def _train_region(region: str) -> None:
    """Train XGBoost model and generate predictions for one region."""
    if region not in _region_data:
        return

    demand_df, weather_df = _region_data[region]

    try:
        from data.feature_engineering import engineer_features
        from data.preprocessing import merge_demand_weather

        merged_df = merge_demand_weather(demand_df, weather_df)
        featured_df = engineer_features(merged_df)
        featured_df = featured_df.dropna(subset=["demand_mw"])

        if len(featured_df) < 168:
            log.warning("precompute_insufficient_data", region=region, rows=len(featured_df))
            return

        _precompute_model_and_predictions(region, demand_df, weather_df, featured_df)

    except Exception as e:
        log.warning("precompute_region_failed", region=region, error=str(e))


def _fetch_data(region: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch demand + weather data, falling back to demo data."""
    try:
        if EIA_API_KEY and EIA_API_KEY != "your_eia_api_key_here":
            from data.eia_client import fetch_demand
            from data.weather_client import fetch_weather

            try:
                demand_df = fetch_demand(region)
            except Exception as e:
                log.warning("precompute_demand_fallback", region=region, error=str(e))
                from data.demo_data import generate_demo_demand

                demand_df = generate_demo_demand(region)
            try:
                weather_df = fetch_weather(region)
            except Exception as e:
                log.warning("precompute_weather_fallback", region=region, error=str(e))
                from data.demo_data import generate_demo_weather

                weather_df = generate_demo_weather(region)
        else:
            from data.demo_data import generate_demo_demand, generate_demo_weather

            demand_df = generate_demo_demand(region)
            weather_df = generate_demo_weather(region)

        log.info(
            "precompute_data_fetched",
            region=region,
            demand_rows=len(demand_df),
            weather_rows=len(weather_df),
        )
        return demand_df, weather_df

    except Exception as e:
        log.error("precompute_fetch_failed", region=region, error=str(e))
        return None, None


def _precompute_model_and_predictions(
    region: str,
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    featured_df: pd.DataFrame,
) -> None:
    """Train XGBoost model and generate predictions for all horizons."""
    from components.callbacks import _MODEL_CACHE, _compute_data_hash, _run_forecast_outlook

    data_hash = _compute_data_hash(demand_df, weather_df, region)

    try:
        from models.xgboost_model import train_xgboost

        train_df = featured_df.copy()
        xgb_model = train_xgboost(train_df)
        _MODEL_CACHE[(region, "xgboost", 0)] = (xgb_model, data_hash, time.time())
        log.info("precompute_model_trained", region=region)

        # Only precompute XGBoost forecasts (fast, single model already trained).
        # Ensemble forecasts require training Prophet + ARIMA from scratch each time
        # (~15-20s each) and are deferred to on-demand computation + cache.
        # Order: default horizon (168h) first for fastest time-to-interactive.
        for horizon in [168, 24, 720]:
            try:
                result = _run_forecast_outlook(demand_df, weather_df, horizon, "xgboost", region)
                if "error" not in result:
                    log.info(
                        "precompute_predictions_cached",
                        region=region,
                        horizon=horizon,
                        model="xgboost",
                    )
            except Exception as e:
                log.warning(
                    "precompute_prediction_error",
                    region=region,
                    horizon=horizon,
                    model="xgboost",
                    error=str(e),
                )

    except Exception as e:
        log.warning("precompute_model_training_failed", region=region, error=str(e))


def _precompute_backtest(region: str, horizon: int, model: str = "xgboost") -> None:
    """Run backtest for a specific region, horizon, and model."""
    if region not in _region_data:
        return

    demand_df, weather_df = _region_data[region]
    from components.callbacks import _run_backtest_for_horizon

    try:
        result = _run_backtest_for_horizon(demand_df, weather_df, horizon, model, region)
        if "error" not in result:
            log.info("precompute_backtest_cached", region=region, horizon=horizon, model=model)
        else:
            log.warning(
                "precompute_backtest_error",
                region=region,
                horizon=horizon,
                model=model,
                error=result.get("error"),
            )
    except Exception as e:
        log.warning(
            "precompute_backtest_error",
            region=region,
            horizon=horizon,
            model=model,
            error=str(e),
        )


def _fetch_generation_all_parallel(regions: list[str]) -> None:
    """Fetch generation-by-fuel data for all regions, warming SQLite cache.

    Makes Tab 4 (Generation & Net Load) instant on first load.
    Never raises — failures fall back to demo data at callback time.
    """
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_generation_single, r): r for r in regions}
        for future in as_completed(futures):
            region = futures[future]
            try:
                future.result()
            except Exception as e:
                log.warning("precompute_generation_failed", region=region, error=str(e))


def _fetch_generation_single(region: str) -> None:
    """Fetch generation data for one region into SQLite + in-memory cache."""
    try:
        from components.callbacks import _fetch_generation_cached

        gen_df = _fetch_generation_cached(region)
        if gen_df is not None and not gen_df.empty:
            log.info("precompute_generation_cached", region=region, rows=len(gen_df))
        else:
            log.warning("precompute_generation_empty", region=region)
    except Exception as e:
        log.warning("precompute_generation_error", region=region, error=str(e))
