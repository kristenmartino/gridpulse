# WattCast v1 → v2 Integration Plan
## What stays, what moves, what gets replaced

---

## Executive Summary

The current GridPulse/WattCast v1 is a **Dash monolith** — data fetching, model training,
feature engineering, and serving all happen in the same process. When a user clicks the
Demand Forecast tab, `_run_forecast_outlook()` in `callbacks.py` trains XGBoost on the spot
(lines 68-170). The `precompute.py` module warms caches at startup, but cache misses
trigger on-demand training during requests.

**v2 separates this into two independent systems:**
1. **Compute plane** (Airflow + Kafka): fetches data, builds features, scores models on a schedule
2. **Serving plane** (FastAPI + Redis): reads pre-computed results, serves to frontend in <5ms

---

## Architecture Comparison

```
V1 (Current — Monolith)                    V2 (Target — Separated)
========================                    =======================

Browser                                     Browser (React)
  ↓                                           ↓
Dash Callbacks (callbacks.py)               FastAPI (read-only)
  ↓                                           ↓
_run_forecast_outlook()                     Redis Cache
  ↓  ← TRAINS MODEL HERE                     ↑ (pre-populated)
feature_engineering.py                      Airflow DAG (every 30 min)
  ↓                                           ↓
merge_demand_weather()                      batch_scorer.py
  ↓                                           ↓
eia_client.py / weather_client.py           feature_builder.py
  ↓                                           ↓
SQLite Cache                                kafka_consumer.py
  ↓                                           ↓
EIA API / Open-Meteo API                    Kafka Topics
                                              ↓
                                            weather_producer.py / grid_producer.py
                                              ↓
                                            EIA API / Open-Meteo API
```

---

## File-by-File Migration Map

### ✅ KEEP AS-IS (reuse directly in v2)

These files are compute-plane logic that works perfectly — just needs to run
inside Airflow tasks instead of Dash callbacks.

| v1 File | v2 Location | Notes |
|---------|-------------|-------|
| `data/feature_engineering.py` | `src/processing/feature_builder.py` | **Your 43-feature pipeline is gold.** The v2 `feature_builder.py` I wrote is a simplified version. USE YOUR EXISTING ONE — it's more complete (handles holidays, has all interaction terms, rolling stats at 3 windows). Just rename and import into the batch scorer. |
| `data/preprocessing.py` | `src/processing/preprocessing.py` | `merge_demand_weather()`, `handle_missing_values()` — reuse exactly. Called by Airflow, not callbacks. |
| `data/eia_client.py` | `src/ingestion/grid_producer.py` | Your EIA client already has pagination, backoff, region mapping, and GCS fallback. The Kafka producer wraps it — `fetch_demand()` feeds `producer.produce()`. |
| `data/weather_client.py` | `src/ingestion/weather_producer.py` | Same pattern. Your Open-Meteo client is production-grade. Kafka producer wraps `fetch_weather()`. |
| `models/xgboost_model.py` | `src/processing/batch_scorer.py` | `train_xgboost()` and `predict_xgboost()` are reused inside the batch scorer. The scorer calls them on schedule, writes results to Redis. |
| `models/prophet_model.py` | `src/processing/batch_scorer.py` | Same — called by batch scorer, not callbacks. |
| `models/arima_model.py` | `src/processing/batch_scorer.py` | Same. |
| `models/ensemble.py` | `src/processing/batch_scorer.py` | `compute_ensemble_weights()` and `ensemble_combine()` — called after all 3 models score. |
| `models/evaluation.py` | `src/processing/batch_scorer.py` | `compute_all_metrics()` — called post-scoring, results cached in Redis alongside predictions. |
| `models/pricing.py` | `src/processing/batch_scorer.py` | `estimate_price_impact()` — pre-computed per scenario, cached. |
| `data/demo_data.py` | `src/processing/demo_data.py` | Keep as fallback for when Kafka/APIs are unavailable. |
| `config.py` | `src/config.py` | Region coordinates, capacity, weather variables, thresholds — all reusable. Add Redis/Kafka config. |
| `data/audit.py` | `src/processing/audit.py` | Audit trail moves to the compute plane — records are written during batch scoring. |
| `observability.py` | `src/observability.py` | `PipelineLogger` is perfect for Airflow task logging. |

### 🔄 REFACTOR (logic stays, wrapper changes)

| v1 File | What Changes | Why |
|---------|-------------|-----|
| `precompute.py` | **Becomes the Airflow DAG body.** The `_train_region()`, `_fetch_data()`, `_precompute_model_and_predictions()` functions map almost 1:1 to Airflow task functions. The threading/parallel logic is replaced by Airflow's task parallelism. | Airflow handles scheduling, retries, monitoring. No more `threading.Thread(target=precompute_all, daemon=True).start()`. |
| `data/cache.py` | **SQLite cache becomes Redis.** Your `Cache.get()` / `Cache.set()` interface stays — just swap the backend from SQLite to Redis. Keep SQLite as a fallback layer (Redis miss → SQLite → GCS → demo data). | Redis is the fast serving path. SQLite is the durable fallback. |
| `data/gcs_store.py` | **Keeps working as-is** — GCS Parquet persistence is the cold storage tier. Batch scorer writes here after writing to Redis. | Three-tier: Redis (hot) → SQLite (warm) → GCS (cold) → demo (fallback). |
| `callbacks.py` | **Gutted.** The 900-line callbacks file becomes ~200 lines. Every callback that currently calls `_run_forecast_outlook()` or `_run_backtest_for_horizon()` now reads from Redis instead. No model training, no feature engineering, no merge operations at request time. | This is the entire point of v2. |
| `models/model_service.py` | **Simplified.** Currently handles trained→simulated fallback. In v2, it just reads from Redis. If Redis is empty, it reads from SQLite. If SQLite is empty, it returns demo data. No model objects in memory. | The serving layer doesn't know models exist. |
| `models/training.py` | **Moves into Airflow DAG.** `train_all_models()` is called by the batch scorer task, not by the app process. | Training happens on Airflow's schedule, not on user request. |

### ❌ REPLACED (v2 has new equivalents)

| v1 File | Replaced By | Why |
|---------|------------|-----|
| `app.py` (Dash entry point) | `src/api/server.py` (FastAPI) | Dash served both UI and data. v2 separates: FastAPI serves data, React serves UI. |
| `components/layout.py` | React dashboard (`wattcast-v2-dashboard.jsx`) | Dash layout → React components. The tab structure, persona switcher, and KPI cards are rebuilt in React. |
| `components/callbacks.py` | FastAPI route handlers | Dash callbacks → REST endpoints. `GET /forecasts/PJM` replaces the `update_demand_outlook()` callback. |
| `components/tab_*.py` | React components | Each tab becomes a React component that fetches from FastAPI. |
| `components/cards.py` | React components | KPI cards, welcome cards → React. |
| N/A (new) | `docker-compose.yml` | New: Kafka, Redis, Airflow, Postgres containers. |
| N/A (new) | `dags/wattcast_scoring_dag.py` | New: Airflow DAG orchestrating the pipeline. |
| N/A (new) | `db/init.sql` | New: Postgres schema for raw data + forecasts. |

### 🗄️ DORMANT (carry forward, reactivate later)

| v1 File | Status |
|---------|--------|
| `simulation/scenario_engine.py` | Keep. Scenario simulator can pre-compute common scenarios via Airflow. |
| `simulation/presets.py` | Keep. 6 preset scenarios are static config. |
| `personas/config.py` | Keep. Persona logic moves to React frontend. |
| `personas/welcome.py` | Keep. Welcome message generation can be an API endpoint. |
| `components/tab_weather.py` | Dormant in v1, dormant in v2. |
| `components/tab_models.py` | Dormant in v1, dormant in v2. |
| `components/tab_generation.py` | Dormant in v1, dormant in v2. |
| `components/tab_alerts.py` | Dormant in v1, dormant in v2. |
| `components/tab_simulator.py` | Dormant in v1, dormant in v2. |

---

## The Critical Refactor: callbacks.py

This is where 80% of the work lives. Here's what each callback becomes:

### Before (v1 — callbacks.py)

```python
# Line 68-170: _run_forecast_outlook()
# Called on EVERY tab click. Trains XGBoost, builds features, runs inference.
# Takes 2-10 seconds.

def update_demand_outlook(horizon, model_name, active_tab, demand_json, weather_json, region):
    demand_df = pd.read_json(io.StringIO(demand_json))
    weather_df = pd.read_json(io.StringIO(weather_json))
    result = _run_forecast_outlook(demand_df, weather_df, horizon_hours, model_name, region)
    # _run_forecast_outlook → merge → feature engineering → train XGBoost → predict
    # ALL AT REQUEST TIME
```

### After (v2 — FastAPI endpoint)

```python
# GET /forecasts/PJM?granularity=1h
# Reads from Redis. Sub-millisecond. Zero computation.

@app.get("/forecasts/{region}")
async def get_forecast(region: str, granularity: str = "15min"):
    result = cache.get_forecast(region.upper(), granularity)
    if result is None:
        raise HTTPException(503, "Pipeline hasn't run yet. Check /health.")
    return result
```

### Callback-by-Callback Migration

| v1 Callback | v1 Behavior | v2 Replacement |
|-------------|-------------|----------------|
| `load_data()` | Fetches EIA + weather on region change | **Eliminated.** Airflow fetches on schedule. Frontend reads cached results. |
| `update_forecast_chart()` | Queries demand store, builds chart | `GET /forecasts/{region}?granularity=1h` → React chart |
| `update_demand_outlook()` | TRAINS MODEL, generates predictions | `GET /forecasts/{region}` → pre-computed in Redis |
| `update_backtest_chart()` | TRAINS MODEL, runs backtest | `GET /backtests/{region}?horizon=24&model=xgboost` → pre-computed |
| `switch_persona()` | Reconfigures KPIs, welcome card | React state change (no API call needed) |
| `update_fallback_banner()` | Shows data freshness warnings | `GET /health` → pipeline status from Redis metadata |
| `update_widget_confidence()` | Per-source confidence badges | `GET /health` → includes per-source status |
| `toggle_meeting_mode()` | Strips chrome for projection | React state toggle (pure frontend) |

---

## Migration Order (for Claude Code)

### Phase 1: Infrastructure (Day 1)
1. Add `docker-compose.yml` with Kafka, Redis, Postgres, Airflow
2. Add `db/init.sql` Postgres schema
3. Add `Dockerfile.api` for FastAPI
4. Verify: `docker compose up -d` starts all services

### Phase 2: Compute Plane (Day 1-2)
1. Copy `data/feature_engineering.py` → `src/processing/feature_builder.py`
2. Copy `data/preprocessing.py` → `src/processing/preprocessing.py`
3. Copy `data/eia_client.py` → wrap in Kafka producer
4. Copy `data/weather_client.py` → wrap in Kafka producer
5. Create `src/processing/kafka_consumer.py` (Kafka → Postgres)
6. Create `src/processing/batch_scorer.py` (features → model → Redis)
7. Create `dags/wattcast_scoring_dag.py` from `precompute.py` logic
8. Verify: Airflow DAG runs, Redis populated

### Phase 3: Serving Plane (Day 2)
1. Create `src/api/server.py` (FastAPI, read-only)
2. Create `src/api/cache.py` (Redis read helper)
3. Add endpoints: `/forecasts/{region}`, `/health`, `/regions`
4. Verify: `curl localhost:8000/forecasts/PJM` returns cached data

### Phase 4: Frontend (Day 2-3)
1. Use `wattcast-v2-dashboard.jsx` (already built)
2. Wire `fetch()` calls to FastAPI endpoints
3. Add persona switcher (React state, no API)
4. Add granularity selector (changes API query param)
5. Verify: Dashboard loads, switches regions, shows pre-computed data

### Phase 5: Testing & CI (Day 3)
1. Copy relevant tests from `tests/unit/` (feature engineering, evaluation, etc.)
2. Add new tests for Redis cache, FastAPI endpoints, Airflow DAG
3. Update `ci.yml` to test v2 components
4. Verify: `pytest` passes, Docker health check green

---

## What You Tell Claude Code

```
Read the integration plan in INTEGRATION_PLAN.md.

We're migrating GridPulse (energy-forecast/) from a Dash monolith to a
pre-computation architecture. The key change: nothing computes at request time.

Phase 1: Set up docker-compose with Kafka, Redis, Postgres, Airflow.
Phase 2: Move the compute logic from callbacks.py and precompute.py
         into Airflow tasks. Reuse existing feature_engineering.py,
         preprocessing.py, and all model files (xgboost, prophet, arima,
         ensemble). The batch_scorer.py writes to Redis.
Phase 3: Create FastAPI serving layer that ONLY reads from Redis.
Phase 4: Wire the React dashboard to FastAPI endpoints.

CRITICAL RULES:
- NEVER import or call any model training code from the serving layer
- The FastAPI server.py must not import anything from models/
- If Redis is empty, return 503 with "pipeline not ready" — do NOT fall back to training
- Reuse data/feature_engineering.py exactly — it's 400 lines of tested code
- Reuse all model files exactly — they're tested and validated
- Keep the existing tests working
```

---

## Files to Give Claude Code

From the v2 scaffold (already built):
- `docker-compose.yml`
- `db/init.sql`
- `Dockerfile.api`
- `requirements.txt`
- `src/config.py`
- `src/api/server.py`
- `src/api/cache.py`
- `dags/wattcast_scoring_dag.py`
- `README.md`
- `wattcast-v2-dashboard.jsx`

From v1 (to be migrated):
- `data/feature_engineering.py` → copy to `src/processing/`
- `data/preprocessing.py` → copy to `src/processing/`
- `data/eia_client.py` → wrap in Kafka producer
- `data/weather_client.py` → wrap in Kafka producer
- `models/*.py` → import into batch scorer
- `config.py` → merge with v2 config
- `precompute.py` → logic moves to Airflow DAG
- All tests in `tests/unit/` → keep and extend

---

## Risk Callouts

1. **Kafka is for forward-compatibility, not current necessity.** The pipeline runs
   every 30 minutes because that matches the data cadence (EIA hourly, Open-Meteo hourly).
   Kafka earns its keep the moment you integrate a real-time feed — SCADA telemetry,
   sub-hourly ERCOT SCED pricing, streaming weather nowcasts. At that point, tightening
   to 5-minute cycles requires only changing Airflow's `schedule_interval`. Below 1 minute,
   swap Airflow for Kafka Streams or Flink; the ingestion and serving layers stay identical.

2. **Prophet is heavy — use fast_mode at tight cadences.** Training Prophet for 8 regions
   every 30 minutes takes ~8 min (fine — leaves 22 min of headroom). At 5-minute cycles,
   Prophet is too slow. The config's `fast_mode` flag (auto-set when `SCORING_INTERVAL_MINUTES < 15`)
   tells the batch scorer to run XGBoost only — it's your best model anyway (3.13% MAPE).
   Prophet/ARIMA can run on a separate slower DAG (every 6h) for ensemble weight updates.

3. **Redis TTL must exceed scoring interval.** Config auto-computes this as `2 × interval`.
   At 30-min cycles, TTL = 1 hour. If a scoring run fails, the previous results survive
   for one more cycle. At 5-min cycles, TTL = 10 minutes.

4. **Keep the GCS fallback.** Your existing `data/gcs_store.py` is a perfect cold storage
   tier. Redis (hot) → Postgres (warm) → GCS Parquet (cold) → demo data (fallback).

5. **Your 361 tests are an asset.** The unit tests for feature engineering, evaluation,
   preprocessing, and models all still apply — they test pure functions that don't change.
   Only the callback tests need rewriting (they test Dash-specific wiring).
