# Precompute Pipeline — Design, Debugging & Performance

Documents the background precomputation system that warms all in-memory
and SQLite caches at startup so dashboard tabs load instantly.

## Problem

The Demand Forecast tab loaded instantly for 24h/XGBoost (pre-cached) but
spun for 5-45 seconds on 168h and 720h horizons, and for Prophet/ARIMA
models. The Backtest tab had the same issue with longer horizons.

**Root cause:** Prophet and ARIMA models were trained from scratch on every
request. XGBoost was the only model cached in `_MODEL_CACHE`. Even though
the callback code implemented cache-check patterns for all three models,
the precompute pipeline only ran XGBoost, so Prophet/ARIMA always missed
the cache on first load.

| Model | Train Time | Was Cached? |
|-------|-----------|-------------|
| XGBoost | 1-3s | Yes |
| Prophet | 5-10s | No |
| ARIMA | 15-45s | No |

## Architecture

### Pipeline Phases

The precompute pipeline runs in a daemon thread started by
`start_background_scheduler()`. It is triggered once on the first HTTP
request via Flask's `before_request` hook, then re-runs every
`PRECOMPUTE_INTERVAL_HOURS` (default 8h).

```
Phase 1:  Fetch demand + weather data for all 8 regions (parallel I/O)
Phase 2a: XGBoost — train + predict for ALL regions (~3s/region)
Phase 2b: Prophet — train + predict for ALL regions (~10s/region)
          Memory freed (model objects + feature DataFrames)
Phase 2c: Ensemble predictions (reuses cached XGBoost + Prophet)
Phase 3:  XGBoost backtests for all regions (parallel)
Phase 4:  Generation data fetch (dormant tab — lowest priority)
```

Default region (FPL) is always processed first in every phase via
`_ordered_regions()`.

### Two-Pass Model Training

Instead of processing one region at a time (all models + horizons +
backtests), the pipeline uses a two-pass approach:

1. **XGBoost pass**: Trains XGBoost for all 8 regions first. This is
   fast (~3s/region) and ensures every region has at least basic
   predictions cached even if the container recycles early.

2. **Prophet pass**: Trains Prophet for all 8 regions. Slower (~10s/region)
   but still completes within the container lifecycle.

This design maximizes region coverage before container recycle. If Cloud
Run kills the container after 2 minutes, all 8 regions have XGBoost
predictions. The old approach would have completed only 1 region with
all models.

### Caching Layers

```
Request → _PREDICTION_CACHE (in-memory, per-worker)
       → SQLite cache (filesystem, shared across workers)
       → Full pipeline (train + predict)
```

| Cache | Key Format | Durability | Latency |
|-------|-----------|------------|---------|
| `_PREDICTION_CACHE` | `(region, horizon, model)` | Lost on restart | O(1) dict |
| `_MODEL_CACHE` | `(region, model, 0)` | Lost on restart | O(1) dict |
| `_BACKTEST_CACHE` | `(region, horizon, model)` | Lost on restart | O(1) dict |
| SQLite | `forecast:{region}:{horizon}:{model}` | Survives restarts* | O(log N) |

*SQLite is on Cloud Run ephemeral disk — survives across requests but
lost on container recycle.

### Cache Invalidation

- **TTL**: 24 hours (`CACHE_TTL_SECONDS`)
- **Data hash**: `_compute_data_hash()` hashes demand/weather timestamp
  ranges + region. If fresh API data shifts the date range, the hash
  changes and triggers a cache miss → retrain.
- **Container recycle**: All caches are lost. Precompute re-runs from
  scratch on the new container's first request.

### Cross-Worker Coordination

- **File lock** (`fcntl.flock` on `/tmp/precompute.lock`): Only one
  gunicorn worker runs precompute. Prevents OOM from concurrent training.
- **SQLite**: Shared across workers. The non-precompute worker gets cache
  hits via `forecast_sqlite_cache_hit`, which then populates its
  in-memory `_PREDICTION_CACHE`.
- **Boolean flag** (`_precompute_triggered`): Thread-safe guard in the
  `before_request` hook. Replaced an earlier `list.remove()` approach
  that caused `ValueError` under gthread concurrency.

### Memory Management

Peak memory during precompute:
- 8 XGBoost models + 8 Prophet models in `_MODEL_CACHE` ≈ 1.5-2GB
- Feature DataFrames for 8 regions ≈ 0.5GB
- Dash runtime + Python overhead ≈ 1GB
- **Total: ~3.5-4GB** (close to 4Gi limit)

The pipeline frees model objects and feature DataFrames via
`_free_precompute_memory()` between Phase 2b and 2c. Ensemble
predictions only need cached prediction arrays, not model objects.
This drops memory by ~1.5GB before backtests run.

ARIMA is excluded from precompute entirely. `auto_arima` with seasonal
`m=24` + SARIMAX fitting exceeds 4Gi when combined with XGBoost/Prophet
models in memory. ARIMA trains on-demand with `_MODEL_CACHE` caching.

## Bugs Found & Fixed

### 1. XGBoost `early_stopping_rounds` TypeError

**Symptom**: `XGBModel.fit() got an unexpected keyword argument 'early_stopping_rounds'`

**Root cause**: `requirements.txt` had `xgboost>=2.1.0`. Docker installed
3.2.0 where `early_stopping_rounds` is only valid in the constructor
(`XGBRegressor(early_stopping_rounds=100)`), not in `.fit()`.

| XGBoost Version | Constructor | `.fit()` |
|----------------|------------|---------|
| 2.x | Accepts | Accepts |
| 3.x | Accepts | **Rejects** |

**Fix**: Kept `early_stopping_rounds` in the constructor (works on both
2.x and 3.x). Pinned `xgboost>=3.2.0,<4` in `requirements.txt`.

**Commit**: `bf4c19c`

### 2. `before_request` Hook Race Condition

**Symptom**: `ValueError` when two gthread threads concurrently call
`server.before_request_funcs[None].remove(_trigger_precompute)`.

**Fix**: Replaced `list.remove()` with a boolean flag
`_precompute_triggered` guarded by `global` statement.

**Commit**: `04244cd`

### 3. ARIMA OOM Kills

**Symptom**: `Memory limit of 4096 MiB exceeded with 4165 MiB used`
during ARIMA training.

**Fix**: Excluded ARIMA from `_ALL_MODELS` in precompute. ARIMA trains
on-demand with `_MODEL_CACHE` caching (first request trains, subsequent
requests hit cache).

**Commit**: `3498540`

### 4. Cloud Run CPU Throttling

**Symptom**: Precompute background thread barely progressed between HTTP
requests. Only 1 region cached before container appeared to "recycle"
(actually just stalled from CPU starvation).

**Root cause**: Cloud Run throttles CPU to near-zero between requests by
default. The daemon thread received almost no CPU time.

**Fix**: Added `--no-cpu-throttling` and `--cpu 2` to the Cloud Run
deploy command. This keeps CPU allocated even when no requests are
being processed, allowing the background thread to run continuously.

**Commit**: `78b3b5a`

### 5. OOM During Ensemble Phase

**Symptom**: `Memory limit of 4096 MiB exceeded` after Phase 2b completed
but before Phase 2c (ensemble) finished.

**Root cause**: All 16 trained models (8 regions x 2 model types) held
in `_MODEL_CACHE` simultaneously, plus prediction arrays and feature
DataFrames.

**Fix**: Moved `_free_precompute_memory()` to between Phase 2b and 2c.
Frees all model objects and feature DataFrames before ensemble
generation. Ensemble only needs cached prediction arrays, not models.

**Commit**: `39ab6a2`

### 6. Pipeline Ordering (Performance)

**Symptom**: Container recycled after caching only 1 region (ERCOT).
Other 7 regions had no cached predictions.

**Root cause**: The old pipeline processed regions sequentially —
all models + all horizons + backtests for one region before moving
to the next. Each region took ~7 minutes.

**Fix**: Two-pass architecture. XGBoost for all 8 regions first (~24s
total), then Prophet for all 8 regions (~80s total). This ensures
maximum region coverage before container recycle. FPL (default region)
is processed first in every phase.

**Commit**: `200bb31`

## Cloud Run Configuration

```yaml
cpu: 2
memory: 4Gi
min-instances: 1
max-instances: 4
timeout: 300
cpu-boost: true          # Full CPU during startup
no-cpu-throttling: true  # CPU allocated between requests
```

The `--no-cpu-throttling` flag is critical. Without it, the background
precompute thread receives near-zero CPU between requests and cannot
complete training in any reasonable timeframe.

`min-instances: 1` ensures at least one warm container is always
available, avoiding cold starts for the first user request.

## Gunicorn Configuration

```
workers: 2
threads: 2
worker-class: gthread
timeout: 300
```

- 2 gthread workers: one serves requests, one runs precompute
- 2 threads per worker: allows concurrent request handling
- 300s timeout: accommodates long-running model training callbacks
- `--preload` intentionally omitted to avoid import lock deadlock

## Performance Results

### Before

| Tab / Action | Latency | Cause |
|-------------|---------|-------|
| Forecast (24h, XGBoost) | ~200ms | Pre-cached |
| Forecast (168h, XGBoost) | 5-10s | Feature engineering + prediction |
| Forecast (any, Prophet) | 10-20s | Full Prophet training |
| Forecast (any, ARIMA) | 15-45s | Full ARIMA auto-order + training |
| Forecast (any, Ensemble) | 30-60s | Trains all 3 models |
| Backtest (168h) | 15-60s | Multiple train/predict cycles |

### After

| Tab / Action | Latency | Source |
|-------------|---------|--------|
| Dashboard HTML | 82-98ms | Static Dash layout |
| Forecast chart (predictions) | 209-230ms | `_PREDICTION_CACHE` hit |
| Historical demand chart (755KB) | 32-59ms | SQLite / in-memory |
| KPI cards | 49-51ms | Computed from cached data |
| All other callbacks | 3-115ms | Various caches |

### Cache Hit Rate

Over an 8+ hour observation window:
- **202 total forecast requests served**
- **194 from in-memory cache** (`forecast_cache_hit`)
- **8 from SQLite cache** (`forecast_sqlite_cache_hit`)
- **0 model retraining events** (`model_training_start`)
- **100% cache hit rate**

### Pipeline Completion Times

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 (data fetch) | ~10s | Parallel I/O, all 8 regions |
| Phase 2a (XGBoost) | ~3 min | 8 regions, 2-fold CV, sequential |
| Phase 2b (Prophet) | ~2 min | 8 regions, sequential + gc.collect |
| Phase 2c (Ensemble) | ~1 min | Reuses cached predictions |
| Phase 3 (Backtests) | ~10 min | 8 regions x 3 horizons, parallel |
| Phase 4 (Generation) | ~12s | Parallel I/O, dormant tab |
| **Total** | **~71-91 min** | Varies by API response times |

### Cache Inventory After Full Pipeline

```json
{
  "predictions_cached": 48,
  "backtests_cached": 16,
  "models_cached": 0
}
```

- **48 predictions** = 8 regions x 3 horizons x 2 models (XGBoost + Prophet)
- **16 backtests** = 8 regions x 2 horizons of XGBoost
- **0 models** = Freed after predictions cached (memory optimization)

## Key Files

| File | Role |
|------|------|
| `precompute.py` | Pipeline orchestration, scheduler, memory management |
| `components/callbacks.py` | `_PREDICTION_CACHE`, `_MODEL_CACHE`, `_run_forecast_outlook()` |
| `data/cache.py` | SQLite cache singleton with TTL + stale fallback |
| `models/xgboost_model.py` | XGBoost training with TimeSeriesSplit CV |
| `models/prophet_model.py` | Prophet training with weather regressors |
| `app.py` | `before_request` hook that triggers precompute |
| `config.py` | `PRECOMPUTE_*` configuration variables |
| `.github/workflows/deploy-prod.yml` | Cloud Run deploy with `--no-cpu-throttling` |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PRECOMPUTE_ENABLED` | `true` | Enable/disable precompute pipeline |
| `PRECOMPUTE_DEFAULT_REGION` | `FPL` | Region processed first in every phase |
| `PRECOMPUTE_ALL_REGIONS` | `true` | Process all 8 regions (vs default only) |
| `PRECOMPUTE_ALL_MODELS` | `true` | Train Prophet + XGBoost (vs XGBoost only) |
| `PRECOMPUTE_MAX_WORKERS` | `4` | ThreadPoolExecutor workers for I/O phases |
| `PRECOMPUTE_INTERVAL_HOURS` | `8` | Re-run interval for the scheduler loop |
| `CACHE_TTL_SECONDS` | `86400` | Cache entry TTL (24 hours) |
