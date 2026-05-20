# Pitch — GridPulse

> Three lengths. Pick by context: 30 seconds for networking, 2 minutes
> for a portfolio open / recruiter screen, 5 minutes for a technical
> walkthrough. Numbers anchored in [`CANONICAL_FACTS.md`](CANONICAL_FACTS.md).

---

## 30-second elevator (~75 words)

> GridPulse is an **energy intelligence platform** I built solo for weather-aware electricity demand forecasting across **51 US balancing authorities** — roughly 100% of contiguous-US lower-48 load. It uses three ML models — Prophet, SARIMAX, and XGBoost — combined via an inverse-MAPE weighted ensemble, deployed on Google Cloud Run with hourly scoring and daily training jobs. Production URL is gridpulse.kristenmartino.ai. The interesting design decision was the role-adapted UI: traders and grid operators see the same data presented differently.

**Practice notes:** Target 30s. Stumble point: don't list everything in the stack — pick three things and stop. The "role-adapted UI" line is the conversation-opener; if they ask one follow-up, that's where it'll go.

---

## 2-minute walkthrough (~300 words)

> **The problem:** Power traders and grid operators routinely manage decisions across six or more disconnected tools — forecasting models, scheduling systems, weather services, telemetry feeds, trading platforms. Each holds a different view of the same underlying data, and reconciliation depends on practitioner expertise rather than process. By the time a unified picture emerges, the trading window has typically closed.
>
> **What I built:** GridPulse is an integrated operating layer for forecast confidence, grid visibility, and decision support. It covers 51 US balancing authorities — ~100% of contiguous-US lower-48 demand — and runs three forecasting models (Prophet, SARIMAX, XGBoost) combined via inverse-MAPE weighted ensemble. The user surface is a Dash/Plotly web app with five tabs (Overview, US Grid, Forecast, Risk, Models) plus a role-based persona system that adapts presentation without changing the underlying data: a trader sees ramp-risk KPIs and pricing bands; a grid operator sees reserve margins and 7-day curtailment outlooks; the demand series itself is identical.
>
> **What's interesting:** Two decisions matter. First, the runtime split: the web tier reads from Redis only — it never fetches from EIA / Open-Meteo / NOAA or trains models in the request path. All expensive work happens in scheduled Cloud Run Jobs that write to Redis. When the cache is cold, the UI renders a "warming" state rather than blocking the user. Second, the ensemble: 1/MAPE weighted (not stacking) because it's self-correcting — a degrading model automatically down-weights — and bounded: the ensemble can never be worse than the worst individual model.
>
> **What's next:** The biggest gap I'd close with another week is continuous model drift monitoring. Holdout MAPE is computed at training time; between trainings, individual models can drift relative to live actuals and the ensemble silently weights them as if they hadn't. Closing that gap is what separates portfolio ML from production ML.

**Practice notes:** Target 2:00. Stumble points: the "53 balancing authorities" — don't say "53," it's 51 (the EIA-930 *total* is 63). The runtime-split paragraph is dense; slow down on "the web tier reads from Redis only" — that's the key architectural choice and listeners need a beat to absorb it.

---

## 5-minute architecture (~750 words)

> **The problem space.** Power traders, grid operators, and forecasting teams in the US wholesale electricity market routinely manage decisions across six or more disconnected tools. Forecasters consult one weather source; operators consult another; traders run their own scenario analysis in Excel. Reconciliation happens informally during the decision itself, often under time pressure. By the time anyone has a unified picture, the trading window has closed. The structural problem isn't bad forecasting — the individual tools are competent. It's that nothing **converges** the inputs before a decision.
>
> **The product.** GridPulse is an integrated decision layer for power-market operations. It covers **51 US balancing authorities** — the IS Os and RTOs (PJM, MISO, CAISO, ERCOT, NYISO, SPP, ISONE) plus utility BAs and federal marketers, totaling roughly 100% of contiguous-US lower-48 demand. The interface adapts by role: four personas (Grid Ops, Renewables, Trader, Data Scientist) get different default KPIs, scenario presets, and alert thresholds, while looking at the same underlying demand series and ensemble forecast. The hypothesis: role-specific interfaces match how decisions are actually made — a trader at 6:30am can't evaluate model selection. The platform makes the selection and surfaces alternatives as secondary signals.
>
> **The ML pipeline.** Three base models — Prophet for additive seasonality, SARIMAX for stationary structure, XGBoost for non-linear weather-demand response — combined via inverse-MAPE weighted ensemble. The ensemble is `weight_i = 1/MAPE_i` normalized to sum to one, computed daily from holdout backtests. I chose this over stacking for three reasons: it's simpler, self-correcting (a degrading model down-weights automatically), and bounded — the ensemble can never be worse than the worst individual model. A real example from a recent FPL training run: `{xgboost: 0.578, prophet: 0.293, arima: 0.130}`. XGBoost wins because it captures the weather-to-cooling-load relationship Florida is dominated by. The feature matrix is 17 raw weather variables from Open-Meteo plus 26 derived features (CDD, HDD, wind power estimate, solar capacity factor, lags, rolling stats) for 43 total.
>
> **The runtime architecture.** GCP-native, three tiers. A **Cloud Run Service** (the web tier) is stateless and reads from Memorystore Redis only — it never fetches from EIA / Open-Meteo / NOAA or trains models in the request path. Two **Cloud Run Jobs** triggered by Cloud Scheduler do the expensive work: an hourly scoring job that fetches data, loads models from GCS, computes forecasts, and writes to Redis; and a daily training job at 04:00 UTC that retrains all three base models on 90 days of holdout data and persists to GCS with an atomic `latest.json` pointer for rollback. When Redis is cold (first deploy, post-flush), the UI renders a "warming" degraded state rather than blocking the user on a 30-second model load.
>
> **What I'd build with more time.** Continuous model drift monitoring. The current holdout MAPE is computed at training time — the daily training run sets the ensemble weights, and they stay frozen until the next training. Between trainings, individual models can drift relative to live actuals, and the ensemble silently weights them as if they hadn't. I saw exactly this symptom in a recent walkthrough: PJM's four forecasts spanned a 47 GW range at the same horizon, with XGBoost predicting a sharp drop and ARIMA predicting we'd keep climbing. That's roughly a week of work to close — a scoring-job-side rolling comparison against live actuals, with the result surfaced as a drift indicator in the Models tab and as alert annotations on confidence badges. It's the strongest argument that the system handles change over time, which is the biggest gap between portfolio ML and production ML.
>
> **The stack, briefly.** Python, Dash/Plotly for the UI, Prophet / pmdarima / xgboost for models, scikit-learn for evaluation, structlog for observability, pytest for ~1,600 tests. Hosting is Google Cloud Run + Memorystore Redis + GCS. CI/CD via GitHub Actions. Repo at github.com/kristenmartino/gridpulse; live at gridpulse.kristenmartino.ai.

**Practice notes:** Target 5:00. Hardest moment: the model-drift section — it's a long single thought. Practice landing the "I saw exactly this symptom" beat and the 47 GW number deliberately. The stack-briefly section at the end is recovery time — if you're ahead of schedule, slow down here; if behind, abbreviate to "Python, Dash, GCP — see the repo."

---

## When to use which

| Context | Use |
|---|---|
| Networking event, casual ask, hallway introduction | 30s |
| Portfolio review open, recruiter screen, intro round of a panel | 2min |
| Technical interviewer's "walk me through your project," deep dive setup, founder pitch | 5min |
| "Tell me about [specific thing]" follow-up | Pivot to relevant story in [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) |

## Practice schedule

- **Before any external use** — re-read whichever variant you'll need. Time it. If you're >10% over target, trim.
- **Weekly** — pick one variant; record on Loom; review for verbal stumbles + filler words
- **After any meaningful new PR** — re-check the 5-minute version. If a fact has moved, update [`CANONICAL_FACTS.md`](CANONICAL_FACTS.md) and the pitch propagates.
