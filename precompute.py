"""
Startup precomputation for the Energy Forecast Dashboard.

Runs in a background daemon thread. On startup, warms all in-memory caches
(models, predictions, backtests) for every region/model/horizon combination.
Then re-runs every PRECOMPUTE_INTERVAL_HOURS (default 8h) to keep caches
fresh. SQLite cache provides cross-restart persistence.

Optimized for Cloud Run container lifecycle — completes the most impactful
work first so partial runs still yield usable caches:

Phase 1a: Fetch demand + weather data for all 8 regions in parallel
Phase 1b: Fetch generation data + write to Redis (fast, must run early)
Phase 2a: XGBoost pass — train + predict for ALL regions (fast, ~3s each)
Phase 2b: Prophet pass — train + predict for ALL regions (~10s each)
Phase 2c: Ensemble predictions (requires 2a + 2b cached)
Phase 3: XGBoost backtests for default region, then remaining regions

Default region (FPL) is always processed first in each phase.
Never raises — failures log warnings and fall back to on-demand computation.
"""

from __future__ import annotations

import fcntl
import gc
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import structlog

from config import (
    EIA_API_KEY,
    PRECOMPUTE_ALL_MODELS,
    PRECOMPUTE_ALL_REGIONS,
    PRECOMPUTE_DEFAULT_REGION,
    PRECOMPUTE_INTERVAL_HOURS,
    PRECOMPUTE_MAX_WORKERS,
    REGION_COORDINATES,
)

log = structlog.get_logger()

# Store fetched data so backtests can reuse without re-fetching
_region_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

# Store featured DataFrames so we don't re-engineer features per model pass
_region_featured: dict[str, pd.DataFrame] = {}

# All horizons to precompute (order: longest first — 720h predictions serve
# all shorter horizons by slicing, maximizing cache utility per training pass)
_ALL_HORIZONS = [720, 168, 24]


def _ordered_regions() -> list[str]:
    """Return all regions with the default region first."""
    all_regions = list(REGION_COORDINATES.keys())
    if PRECOMPUTE_DEFAULT_REGION in all_regions:
        all_regions.remove(PRECOMPUTE_DEFAULT_REGION)
        all_regions.insert(0, PRECOMPUTE_DEFAULT_REGION)
    return all_regions


def precompute_all() -> None:
    """Entry point. Called from app.py at startup. Never raises."""
    t0 = time.time()
    log.info(
        "precompute_start",
        default_region=PRECOMPUTE_DEFAULT_REGION,
        all_regions=PRECOMPUTE_ALL_REGIONS,
        all_models=PRECOMPUTE_ALL_MODELS,
    )
    sys.stdout.flush()

    try:
        all_regions = _ordered_regions()

        # Phase 1a: Fetch demand + weather data for all regions in parallel
        _fetch_all_data_parallel(all_regions)
        log.info("precompute_phase1_complete", regions_with_data=len(_region_data))
        sys.stdout.flush()

        # Phase 1b: Fetch generation data + write to Redis (fast API call, not compute)
        # Must run early — workers may get recycled before later phases finish.
        _fetch_generation_all_parallel(all_regions)
        log.info("precompute_phase1b_generation_complete")
        sys.stdout.flush()

        # Prepare featured DataFrames for all regions (shared across model passes)
        _prepare_features_all(all_regions)

        # Phase 2a: XGBoost pass — fast (~3s/region), gets all regions covered
        _train_model_all_regions("xgboost", all_regions)
        log.info("precompute_phase2a_complete", model="xgboost")
        sys.stdout.flush()

        # Phase 2b: Prophet pass — slower (~10s/region), adds ensemble capability
        if PRECOMPUTE_ALL_MODELS:
            _train_model_all_regions("prophet", all_regions)
            log.info("precompute_phase2b_complete", model="prophet")
            sys.stdout.flush()

            # Free model objects + features before ensemble — ensemble only
            # needs cached predictions, not model objects. This reclaims
            # ~1-2GB to prevent OOM during ensemble generation.
            _free_precompute_memory(all_regions)

            # Phase 2c: Ensemble predictions (uses cached XGBoost + Prophet)
            _generate_ensemble_predictions(all_regions)
            log.info("precompute_phase2c_complete", model="ensemble")
            sys.stdout.flush()
        else:
            _free_precompute_memory(all_regions)

        # Phase 3: XGBoost backtests (default region first, then rest)
        _backtest_all_parallel(all_regions)
        log.info("precompute_phase3_complete")
        sys.stdout.flush()

        log.info("precompute_all_phases_complete")
        sys.stdout.flush()

    except Exception as e:
        log.error("precompute_failed", error=str(e))
        sys.stdout.flush()

    elapsed = time.time() - t0
    log.info("precompute_complete", elapsed_seconds=round(elapsed, 1))


_LOCK_PATH = "/tmp/precompute.lock"


def start_background_scheduler() -> threading.Thread | None:
    """Launch a daemon thread that runs precompute_all() on a recurring interval.

    First run is immediate. Subsequent runs every PRECOMPUTE_INTERVAL_HOURS.
    Uses an OS-level file lock so only one gunicorn worker runs precompute
    across all worker processes (avoids OOM from concurrent training).
    Returns the daemon thread, or None if another worker already holds the lock.
    """
    # Non-blocking file lock — if another worker holds it, skip immediately.
    try:
        lock_fd = open(_LOCK_PATH, "w")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("precompute_scheduler_locked_by_another_worker")
        return None

    log.info("precompute_scheduler_acquired_lock", pid=threading.get_native_id())

    interval_seconds = PRECOMPUTE_INTERVAL_HOURS * 3600

    def _loop() -> None:
        try:
            # Keep lock_fd alive for the lifetime of this thread (holds the flock)
            _ = lock_fd
            log.info("precompute_scheduler_loop_starting")
            sys.stdout.flush()

            # First run
            precompute_all()

            while True:
                log.info(
                    "precompute_scheduler_sleeping",
                    next_run_hours=PRECOMPUTE_INTERVAL_HOURS,
                )
                sys.stdout.flush()
                time.sleep(interval_seconds)
                log.info("precompute_scheduler_wakeup")
                # Re-fetch fresh data and retrain
                _region_data.clear()
                _region_featured.clear()
                precompute_all()
        except Exception:
            log.exception("precompute_scheduler_crashed")
            sys.stdout.flush()

    t = threading.Thread(target=_loop, daemon=True, name="precompute-scheduler")
    t.start()
    return t


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


def _prepare_features_all(regions: list[str]) -> None:
    """Pre-compute feature-engineered DataFrames for all regions with data."""
    from data.feature_engineering import engineer_features
    from data.preprocessing import merge_demand_weather

    for region in regions:
        if region not in _region_data or region in _region_featured:
            continue
        try:
            demand_df, weather_df = _region_data[region]
            merged_df = merge_demand_weather(demand_df, weather_df)
            featured_df = engineer_features(merged_df)
            featured_df = featured_df.dropna(subset=["demand_mw"])
            if len(featured_df) < 168:
                log.warning("precompute_insufficient_data", region=region, rows=len(featured_df))
                continue
            _region_featured[region] = featured_df
        except Exception as e:
            log.warning("precompute_features_failed", region=region, error=str(e))


def _train_model_all_regions(model_name: str, regions: list[str]) -> None:
    """Train a single model type for all regions, then generate predictions.

    Processes regions sequentially to bound memory. Each region: train model,
    generate predictions for all horizons, then GC before next region.
    This ensures maximum region coverage before container recycle.
    """
    from components.callbacks import _MODEL_CACHE, _compute_data_hash, _run_forecast_outlook

    regions_with_features = [r for r in regions if r in _region_featured]

    for region in regions_with_features:
        try:
            demand_df, weather_df = _region_data[region]
            featured_df = _region_featured[region]
            data_hash = _compute_data_hash(demand_df, weather_df, region)
            train_df = featured_df.copy()

            # Train model (reduced CV folds for faster precompute; accuracy
            # is validated at backtest time, not here)
            mck = (region, model_name, 0)
            if model_name == "xgboost":
                from models.xgboost_model import train_xgboost

                model_obj = train_xgboost(train_df, n_splits=2)
            elif model_name == "prophet":
                from models.prophet_model import train_prophet

                model_obj = train_prophet(train_df)
            else:
                continue

            _MODEL_CACHE[mck] = (model_obj, data_hash, time.time())
            log.info("precompute_model_trained", region=region, model=model_name)

            # Generate predictions for all horizons immediately
            for horizon in _ALL_HORIZONS:
                try:
                    result = _run_forecast_outlook(
                        demand_df, weather_df, horizon, model_name, region
                    )
                    if "error" not in result:
                        log.info(
                            "precompute_predictions_cached",
                            region=region,
                            horizon=horizon,
                            model=model_name,
                        )
                except Exception as e:
                    log.warning(
                        "precompute_prediction_error",
                        region=region,
                        horizon=horizon,
                        model=model_name,
                        error=str(e),
                    )

        except Exception as e:
            log.warning(
                "precompute_train_failed",
                region=region,
                model=model_name,
                error=str(e),
            )

        # Reclaim peak memory before next region
        gc.collect()


def _free_precompute_memory(regions: list[str]) -> None:
    """Free model objects and feature DataFrames to reclaim memory.

    After Phase 2 (predictions cached), we no longer need the trained model
    objects or feature DataFrames in memory. This reclaims ~1-2GB before
    backtests run. Predictions remain cached in _PREDICTION_CACHE + SQLite.
    Models are re-trained by backtests (per fold) and by on-demand callbacks.
    """
    from components.callbacks import _MODEL_CACHE

    model_keys_freed = 0
    for region in regions:
        for model_name in ("xgboost", "prophet", "arima"):
            mck = (region, model_name, 0)
            if mck in _MODEL_CACHE:
                del _MODEL_CACHE[mck]
                model_keys_freed += 1

    _region_featured.clear()
    gc.collect()
    log.info("precompute_memory_freed", model_keys_freed=model_keys_freed)


def _generate_ensemble_predictions(regions: list[str]) -> None:
    """Generate ensemble predictions for all regions (requires individual models cached)."""
    from components.callbacks import _run_forecast_outlook

    # NOTE: _region_featured is cleared by _free_precompute_memory() before
    # this phase runs.  Ensemble only needs cached sub-model predictions
    # (stored in _PREDICTION_CACHE), not the feature DataFrames, so we gate
    # on _region_data instead.
    regions_with_data = [r for r in regions if r in _region_data]

    for region in regions_with_data:
        demand_df, weather_df = _region_data[region]
        for horizon in _ALL_HORIZONS:
            try:
                result = _run_forecast_outlook(demand_df, weather_df, horizon, "ensemble", region)
                if "error" not in result:
                    log.info(
                        "precompute_predictions_cached",
                        region=region,
                        horizon=horizon,
                        model="ensemble",
                    )
            except Exception as e:
                log.warning(
                    "precompute_prediction_error",
                    region=region,
                    horizon=horizon,
                    model="ensemble",
                    error=str(e),
                )


def _backtest_all_parallel(regions: list[str]) -> None:
    """Run backtests for all regions in parallel.

    Precomputes XGBoost backtests for all horizons (the default view).
    Prophet/ARIMA backtests are deferred to on-demand + cache since backtest
    requires multiple train/predict cycles and is very expensive for those models.
    """
    regions_with_data = [r for r in regions if r in _region_data]
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {}
        for r in regions_with_data:
            for horizon in _ALL_HORIZONS:
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


def _fetch_data(region: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch demand + weather data from real APIs. Returns None on failure."""
    try:
        if not EIA_API_KEY or EIA_API_KEY == "your_eia_api_key_here":
            log.warning("precompute_skipped_no_api_key", region=region)
            return None, None

        from data.eia_client import fetch_demand
        from data.weather_client import fetch_weather

        demand_df = None
        weather_df = None

        try:
            demand_df = fetch_demand(region)
        except Exception as e:
            log.warning("precompute_demand_failed", region=region, error=str(e))

        try:
            weather_df = fetch_weather(region)
        except Exception as e:
            log.warning("precompute_weather_failed", region=region, error=str(e))

        if demand_df is None or weather_df is None:
            log.warning(
                "precompute_partial_data",
                region=region,
                has_demand=demand_df is not None,
                has_weather=weather_df is not None,
            )
            return None, None

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
    """Fetch real EIA generation data for one region into cache + Redis.

    Only writes to Redis when real EIA data is obtained. If the EIA API
    is down, leaves existing Redis data as-is (no demo overwrite).
    """
    try:
        import time as _time

        from components.callbacks import _EIA_FUEL_MAP, _GENERATION_CACHE
        from config import EIA_API_KEY
        from data.eia_client import fetch_generation_by_fuel

        if not EIA_API_KEY or EIA_API_KEY == "your_eia_api_key_here":
            log.warning("precompute_generation_skipped_no_key", region=region)
            return

        gen_df = fetch_generation_by_fuel(region)
        if gen_df is None or gen_df.empty:
            log.warning("precompute_generation_empty", region=region)
            return

        # Normalize fuel type codes
        gen_df["fuel_type"] = (
            gen_df["fuel_type"].map(_EIA_FUEL_MAP).fillna(gen_df["fuel_type"].str.lower())
        )
        # Update in-memory cache
        _GENERATION_CACHE[region] = (gen_df, _time.time())
        log.info("precompute_generation_cached", region=region, rows=len(gen_df))

        # Write real data to Redis
        _write_generation_to_redis(region, gen_df)

    except Exception as e:
        log.warning("precompute_generation_error", region=region, error=str(e))


def _write_generation_to_redis(region: str, gen_df: pd.DataFrame) -> None:
    """Write generation data to Redis so the fast path serves real EIA data."""
    from data.redis_client import redis_set

    try:
        gen_df = gen_df.copy()
        gen_df["timestamp"] = pd.to_datetime(gen_df["timestamp"])
        pivot = gen_df.pivot_table(
            index="timestamp",
            columns="fuel_type",
            values="generation_mw",
            aggfunc="sum",
        ).fillna(0)

        payload: dict = {
            "region": region,
            "timestamps": [t.isoformat() for t in pivot.index],
        }
        for col in pivot.columns:
            payload[col] = pivot[col].tolist()

        # Compute per-row renewable %
        total = pivot.sum(axis=1)
        ren_cols = [c for c in ["wind", "solar", "hydro"] if c in pivot.columns]
        if ren_cols and total.mean() > 0:
            ren_pct = (pivot[ren_cols].sum(axis=1) / total * 100).tolist()
        else:
            ren_pct = [0.0] * len(pivot)
        payload["renewable_pct"] = ren_pct

        if redis_set(f"wattcast:generation:{region}", payload, ttl=86400):
            avg_ren = sum(ren_pct) / len(ren_pct) if ren_pct else 0
            log.info(
                "precompute_generation_redis_written",
                region=region,
                rows=len(pivot),
                avg_renewable_pct=round(avg_ren, 1),
            )
    except Exception as e:
        log.warning("precompute_generation_redis_error", region=region, error=str(e))
