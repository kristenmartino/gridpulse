# WattCast v2 — Pre-Computation Architecture

> Weather-aware energy demand forecasting for 8 U.S. grid regions.  
> **Zero computation at request time.** Every dashboard load is a cache read.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ⑤ AIRFLOW (every 30 min)                     │
│  Orchestrates the entire pipeline — the ONLY thing that         │
│  triggers computation. The frontend never does.                 │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────┐          ┌──────────────────────────────┐
│ ① KAFKA INGESTION    │          │ ② BATCH PROCESSING           │
│                      │          │                              │
│ weather_producer.py  │─── ──────│ kafka_consumer.py            │
│ grid_producer.py     │  topics  │ feature_builder.py (43 feat) │
│                      │─── ──────│ batch_scorer.py (XGBoost)    │
│ Weather APIs → Kafka │          │                              │
│ EIA API → Kafka      │          │ Kafka → Postgres → Model     │
└──────────────────────┘          │              → Redis          │
                                  └──────────────────────────────┘
                                              │
                                              ▼
                                  ┌──────────────────────┐
                                  │ ③ SERVING LAYER      │
                                  │                      │
                                  │ FastAPI (read-only)   │
                                  │ Redis (sub-ms reads)  │
                                  │ Postgres (historical) │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ ④ FRONTEND           │
                                  │                      │
                                  │ React Dashboard      │
                                  │ GET /forecasts/PJM   │
                                  │ (cache read only)    │
                                  └──────────────────────┘
```

## The Core Principle

**The frontend reads. Airflow writes. Nothing else.**

| Layer | Responsibility | When it runs |
|-------|---------------|--------------|
| Kafka Producers | Pull external API data | Every 30 min (Airflow) |
| Kafka Consumer | Normalize → Postgres | Every 30 min (Airflow) |
| Batch Scorer | 43-feature model → Redis | Every 30 min (Airflow) |
| FastAPI | Read from Redis/Postgres | On every request |
| React Dashboard | Display cached data | User-initiated |

## Why 30-Minute Cycles (Not Faster)

The pipeline runs every 30 minutes because **that matches the upstream data cadence.** EIA publishes hourly demand data. Open-Meteo updates weather forecasts hourly. Running the pipeline more frequently would re-score the same input data without new information.

Kafka is in the architecture not because 30-minute batch requires it, but because **it supports scaling to 5-minute cycles without a rewrite** — the moment we integrate a real-time data source like a SCADA feed, sub-hourly pricing data from ERCOT's SCED, or a streaming weather nowcast, Kafka is already handling ingestion decoupling. The producers, consumers, and topic structure don't change — only the Airflow schedule interval tightens. Below 1 minute, Airflow's scheduler is the bottleneck and you'd switch to Kafka Streams or Flink for the orchestration layer, but the ingestion and serving tiers stay identical.

## What Gets Pre-Computed

Every 30 minutes, the batch scorer generates:

- **8 regions** × **96 intervals** (15-min) = **768 predictions**
- Rolled up to 3 granularities: `15min`, `1h`, `1d`
- Cached in Redis with 1-hour TTL
- Persisted to Postgres for model monitoring

Total Redis footprint: ~50KB. Response time: <1ms.

## Quick Start

### Prerequisites
- Docker + Docker Compose
- EIA API key ([free registration](https://www.eia.gov/opendata/register.php))

### Run Locally

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your EIA_API_KEY

# 2. Start the full stack
docker compose up -d

# 3. Verify services
curl http://localhost:8000/health    # API
curl http://localhost:8080           # Airflow UI (admin/admin)

# 4. Trigger initial pipeline run
# In Airflow UI: enable and trigger "wattcast_scoring_pipeline"

# 5. Query forecasts
curl "http://localhost:8000/forecasts/PJM?granularity=1h"
curl "http://localhost:8000/forecasts"
```

## API Endpoints

| Endpoint | Description | Data Source |
|----------|-------------|-------------|
| `GET /health` | Pipeline freshness check | Redis metadata |
| `GET /forecasts/{region}` | Single region forecast | Redis |
| `GET /forecasts` | All regions | Redis (pipeline read) |
| `GET /regions` | Available regions | Static config |
| `GET /granularities` | Available time granularities | Static config |

## 43-Feature Pipeline

| Category | Count | Features |
|----------|-------|----------|
| Weather | 9 | temp, humidity, dewpoint, precip, cloud, wind speed/dir, pressure, solar |
| Temporal | 8 | hour sin/cos, DOW sin/cos, month sin/cos, weekend, holiday |
| Demand lags | 10 | 1h-168h lags + 6h/24h rolling means |
| Weather lags | 8 | 24h lags, 3h deltas, 6h rolling means |
| Interactions | 4 | temp×hour, solar×cloud, wind×pressure, humidity×temp |
| Calendar | 4 | day-of-year, HDD, CDD, daylight hours |

## v1 → v2 Migration

| Concern | v1 (Current) | v2 (This) |
|---------|-------------|-----------|
| Model execution | On page load | Batch (every 30 min) |
| Data freshness | Real-time (slow) | Near-real-time (fast) |
| API response time | 2-10 seconds | <5 milliseconds |
| Scalability | Single user | Thousands concurrent |
| Infrastructure | Monolithic | Separated concerns |

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Ingestion | Apache Kafka | Decouples data sources from processing; supports tightening to 5-min cycles or streaming if real-time feeds (SCADA, SCED pricing) are added |
| Orchestration | Apache Airflow | Industry-standard DAG scheduling; 30-min default matches data cadence, configurable down to 1-min |
| Feature Store | PostgreSQL | Reliable, SQL-native, battle-tested |
| ML Model | XGBoost (primary) | 3.13% MAPE on ERCOT holdout; fast inference, strong tabular performance |
| Serving Cache | Redis | Sub-millisecond reads, TTL-based expiry, 1h TTL > 30-min scoring cycle ensures freshness |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Containers | Docker Compose | One-command local dev environment |

## Scaling the Cadence

| Cycle | Orchestration | Ingestion | What Changes |
|-------|--------------|-----------|-------------|
| 30 min (default) | Airflow cron | Kafka (batch produce) | Nothing — current setup |
| 15 min | Airflow cron | Kafka (batch produce) | Change `schedule_interval` in DAG |
| 5 min | Airflow cron | Kafka (batch produce) | Change `schedule_interval`; XGBoost-only scoring (skip Prophet/ARIMA for speed) |
| <1 min | Kafka Streams / Flink | Kafka (continuous) | Replace Airflow with stream processor; Kafka already in place |
