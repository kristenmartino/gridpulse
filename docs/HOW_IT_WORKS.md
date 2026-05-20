# How GridPulse Works

> **Status: STUB.** Real content lands in PR-C1 (recall artifacts, next
> session per the 2026-05-20 wider replan). Mermaid diagrams will live
> alongside this file under `docs/diagrams/` once that PR ships.

## TODO sections

The PR-C1 plan calls for ~600 words across the following sections, each
with an accompanying Mermaid diagram in `docs/diagrams/`:

- **System architecture** — Cloud Run Service ↔ Memorystore Redis ↔ Cloud
  Run Jobs ↔ GCS ↔ EIA / Open-Meteo / NOAA. Why each piece, what flows
  where.
- **Request lifecycle** — what happens when a user loads the Forecast tab.
  Cold cache → warming state. The Redis-only-reads boundary.
- **Scoring + training pipelines** — Cloud Scheduler → Cloud Run Jobs →
  GCS models + Redis writes. The hourly/daily schedule and what each
  phase does.
- **Model architecture** — Features → 3 base models (Prophet / SARIMAX /
  XGBoost) → inverse-MAPE weighted ensemble. The training-vs-scoring
  split.
- **UI structure** — 5 visible tabs, persona selector, region selector,
  role-adapted views.

## Anchor: what to memorize verbatim

Pulled from [`docs/CANONICAL_FACTS.md`](CANONICAL_FACTS.md) — these are
the numbers that show up in every conversation about GridPulse.

- 51 US balancing authorities (~100% of contiguous-US lower-48 load)
- 3 base ML models (Prophet, SARIMAX, XGBoost) + 1 weighted ensemble
- 17 raw weather variables, 43 total features
- Hourly scoring + daily training (04:00 UTC)
- Redis-only reads in the web tier; degraded warming state when cold
