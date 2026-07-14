# GridPulse — Critical Code Review (2026-07)

> **Scope:** exhaustive critical review of the working tree at commit `5000d6a` (frozen 2026-07-01).
> **Method:** 35-agent multi-phase workflow — 17 parallel finders/specialists/seed-verifiers → 14 adversarial verifiers → 4 synthesis writers. Every finding was either demonstrated (verbatim repro output / airtight static trace) or refuted; there are no unverified "PLAUSIBLE" claims in this report.
> **Axes reviewed:** flawed logic · loopholes & missing connections · duplication · over-engineering · half-finished work · forced/illogical design · claims without empirical/mathematical/statistical proof.
> **Deduplicated against** the prior elegance audit (#181–#189) and the live issue tracker: findings are tagged `[NEW]` or `[EXT of #N]`.

---

## Executive summary

The codebase's leaf-level engineering is genuinely strong (pure metric functions, the SARIMAX lean-pickle, the EIA circuit breaker, the stateless web/jobs/Redis split). The problems this review surfaces are almost all at the **seams** — where honest components are wired together in ways that quietly defeat the honesty. Two findings rise to **P0 (integrity)** and were each independently confirmed by multiple verifiers with executable repros:

1. **The Risk/Extreme-Events tab is fabricated end-to-end.** The hourly production scoring job calls `generate_demo_alerts()` and writes canned "Heat Advisory / Wind Advisory" content to `gridpulse:alerts:{region}` with no environment gate, and the UI stamps it **"NOAA Alerts · LIVE."** The real NOAA client exists but has no caller. This is a strictly-worse sibling of the tracked #166 (which is only *honest-empty*), and the #189 audit explicitly recorded that only two correctness defects touched fake data — this is the missed third.
2. **Two of the three models are forecasting for the wrong clock.** Prophet and SARIMAX predictions are anchored at the *pickled model's training-end* but written into Redis rows timestamped from the *current* scoring tick. With daily training and hourly scoring, the diurnal cycle is rotated by up to ~23 hours before the next retrain. Only XGBoost is aligned. This plausibly re-explains the extreme live ARIMA drift that #170 attributes to "genuine model weakness."

Beneath those, a recurring **P1 theme is trust-signal theater**: the "80% empirical prediction interval" is calibrated from XGBoost residuals no matter which model you select; freshness is hardcoded to `'fresh'` rather than measured; `scored_at` is discarded so a stale forecast narrates as current; missing metrics render as `MAPE 0.0%` (fabricated perfection); an sMAPE value is labeled "MAPE." Several of these share the exact shape of the PR #130 bug the project already fixed once. The single most consequential *statistical* finding is that the published holdout numbers are **incommensurable** — XGBoost is scored teacher-forced one-step-ahead while Prophet/ARIMA are scored multi-step — which means #181's "ensemble trails best-base" analysis rests on a contaminated comparison.

### Counts

| Severity | Confirmed (canonical, after merge) | Notes |
|---|---|---|
| **P0 — integrity** | 2 | fabricated alerts; forecast time-misalignment |
| **P1 — user-visible correctness** | 10 | trust-signal theater, holdout methodology, request-path I/O |
| **P2 — latent bugs & material debt** | 52 | grouped into 8 themes below |
| **P3 — polish & doc-only** | 57 | table + claims ledger |
| **Refuted leads** | 13 | see Appendix A — several seeds correctly died against shipped fixes |

Raw verifier tally before duplicate-merging: 184 CONFIRMED / 13 REFUTED across 197 verdicts; ~104 NEW, ~28 extensions of tracked issues, ~26 duplicates of seeds or intra-review. Coverage: all 72 in-scope production files were attested read-in-full by their assigned reviewer (the 3 deploy-workflow YAMLs were gap-filled in Phase 2 — clean).

**What to do first:** the two P0s and the freshness/interval/`scored_at` P1 cluster are all "stop presenting fabricated or mislabeled data as real" — they are the highest trust-per-fix and several share code. Drafted, ready-to-file issues for all 12 P0/P1 findings are in the *Drafted issues* section below.

---

## Scoreboard

<details><summary>All 197 verdicts (P0/P1 first). Canonical write-ups follow; IDs here map to the finder/verifier that raised each.</summary>

| Finding | Sev | Verdict | Dedupe | File |
|---|---|---|---|---|
| F1-001 | P0 | CONFIRMED | NEW — distinct mechanism from #129 (fore | `models/prophet_model.py` |
| F3-001 | P0 | CONFIRMED | NEW — untracked; distinct from #166 (wri | `jobs/phases.py` |
| F3-003-revised | P0 | CONFIRMED | NEW — no issue in the registry covers th | `jobs/phases.py` |
| F4-001 | P0 | CONFIRMED | intra-dup:F3-001 | `jobs/phases.py` |
| F4-002 | P0 | CONFIRMED | NEW — no known issue tracks forecast-ori | `jobs/phases.py` |
| F7-001 | P0 | CONFIRMED | NEW — class-sibling of #166 (scoring job | `components/_callbacks_alerts.py` |
| X3-001 | P0 | CONFIRMED | intra-dup:F3-001 (adds the verified NOAA | `jobs/phases.py` |
| X3-001 | P0 | CONFIRMED | NEW — checked known-issue registry: #166 | `jobs/phases.py` |
| X3-002 | P0 | CONFIRMED | NEW — no registry issue tracks the alert | `components/_callbacks_alerts.py` |
| F1-002 | P1 | CONFIRMED | NEW — #186 is implementation parity of t | `jobs/training_job.py` |
| F2-002 | P1 | CONFIRMED | intra-dup:F1-002 (same evaluation-protoc | `jobs/training_job.py` |
| F3-003 | P1 | CONFIRMED | NEW — not in known issues (#188's warmin | `components/_callbacks_alerts.py` |
| F5-001 | P1 | CONFIRMED | NEW — no known issue covers the Forecast | `components/_callbacks_overview.py` |
| F5-002 | P1 | CONFIRMED | NEW — badge/provenance mismatch class of | `components/_callbacks_shared.py` |
| F5-003 | P1 | CONFIRMED | #127-extension — the panel is #127's sub | `components/_callbacks_overview.py` |
| F5-005 | P1 | CONFIRMED | NEW — #142/#155 introduced the sMAPE pre | `components/_callbacks_overview.py` |
| F5-006 | P1 | CONFIRMED | NEW — not #129 (that was the forecast-ST | `components/_callbacks_overview.py` |
| F5-010 | P1 | CONFIRMED | NEW — distinct from #131 (that was simul | `components/_callbacks_overview.py` |
| F6-003 | P1 | CONFIRMED | NEW — not in known_issues.json (#150/#16 | `components/_callbacks_forecast.py` |
| F7-004 | P1 | CONFIRMED | NEW — #188/#153 adjacent but neither tra | `components/callbacks.py` |
| F7-005 | P1 | CONFIRMED | NEW — no registry issue or seed covers t | `components/_callbacks_us_grid.py` |
| X3-003 | P1 | CONFIRMED | NEW — no registry issue tracks the orpha | `components/_callbacks_overview.py` |
| X3-004 | P1 | CONFIRMED | NEW — adjacent to #153 (typed payloads w | `components/callbacks.py` |
| F1-003 | P2 | CONFIRMED | NEW — no known issue or seed covers the  | `models/arima_model.py` |
| F1-004 | P2 | CONFIRMED | NEW — beyond drift.py's admitted no-back | `models/drift.py` |
| F1-005 | P2 | CONFIRMED | NEW — seed S1-07 covers only the assert  | `models/xgboost_model.py` |
| F1-006 | P2 | CONFIRMED | NEW — distinct from seed S1-01 (compute_ | `models/ensemble.py` |
| F2-001 | P2 | CONFIRMED | NEW — #176/#179 fixed the all-models-hol | `jobs/training_job.py` |
| F2-003 | P2 | CONFIRMED | seed-dup:S5-06 for the GCS-in-request-pa | `models/model_service.py` |
| F2-004 | P2 | CONFIRMED | NEW — not in known_issues.json; distinct | `models/persistence.py` |
| F2-005 | P2 | CONFIRMED | seed-dup:S2-02 (with S2-04b covering the | `models/model_service.py` |
| F2-010 | P2 | CONFIRMED | NEW — training-job sibling of already-ve | `jobs/training_job.py` |
| F2-013 | P2 | CONFIRMED | NEW — persistence.py appears in no known | `models/persistence.py` |
| F3-002 | P2 | CONFIRMED | NEW — the fabricated-alerts pipeline it  | `data/noaa_client.py` |
| F3-004 | P2 | CONFIRMED | NEW — the exit-0 aggregation half overla | `data/redis_client.py` |
| F3-005 | P2 | CONFIRMED | NEW — systemic version of the 'fresh'-mi | `data/cache.py` |
| F3-006 | P2 | CONFIRMED | NEW — distinct mechanism from seed S3-01 | `data/eia_client.py` |
| F3-009 | P2 | CONFIRMED | NEW — #185 covers the three-endpoint fet | `data/eia_client.py` |
| F3-010 | P2 | CONFIRMED | seed-dup:S3-03 — verdict copied from the | `data/gcs_store.py` |
| F4-003 | P2 | CONFIRMED | seed-dup:S4-04 | `jobs/scoring_job.py` |
| F4-005 | P2 | CONFIRMED | NEW — untracked; unrelated to #186 (auto | `jobs/phases.py` |
| F4-006 | P2 | CONFIRMED | NEW — no known issue or seed touches ai_ | `data/ai_briefing.py` |
| F4-007 | P2 | CONFIRMED | NEW — adjacent to #184 (Prophet regresso | `jobs/phases.py` |
| F4-009 | P2 | CONFIRMED | NEW — no issue or seed covers ai_briefin | `data/ai_briefing.py` |
| F5-004 | P2 | CONFIRMED | NEW — #142/#155 (closed) fixed the metri | `components/_callbacks_overview.py` |
| F5-007 | P2 | CONFIRMED | #154-extension — #154 tracks moving the  | `components/_callbacks_overview.py` |
| F5-008 | P2 | CONFIRMED | NEW — the #131 hardcoded-fake-metrics cl | `components/_callbacks_overview.py` |
| F5-009 | P2 | CONFIRMED | NEW — H2 convention violation not tracke | `components/_callbacks_overview.py` |
| F5-011 | P2 | CONFIRMED | NEW — no known issue covers the Net Load | `components/_callbacks_overview.py` |
| F5-012 | P2 | CONFIRMED | NEW — not in known_issues.json (#188 con | `components/_callbacks_overview.py` |
| F6-001 | P2 | CONFIRMED | #166-extension — #166 tracks the diagnos | `components/_callbacks_models.py` |
| F6-004 | P2 | CONFIRMED | NEW — the dead surface + preserved integ | `components/_callbacks_backtest.py` |
| F6-005 | P2 | CONFIRMED | NEW — no known issue tracks interval met | `components/_callbacks_backtest.py` |
| F6-007 | P2 | CONFIRMED | NEW — no known issue tracks the replay s | `components/_callbacks_forecast.py` |
| F6-008 | P2 | CONFIRMED | NEW — same badge-vs-data class as the fi | `components/_callbacks_forecast.py` |
| F6-009 | P2 | CONFIRMED | NEW — no issue tracks the ratio semantic | `components/_callbacks_models.py` |
| F6-012 | P2 | CONFIRMED | #188-extension — #188's warming-state it | `components/_callbacks_forecast.py` |
| F7-002 | P2 | CONFIRMED | seed-dup:S5-06 (verdict copied from the  | `components/_callbacks_us_grid.py` |
| F7-003 | P2 | CONFIRMED | #154-extension — the orphaned-fast-path  | `components/_callbacks_weather.py` |
| F7-009 | P2 | CONFIRMED | intra-dup:X1-001 (same defect reported b | `components/_callbacks_alerts.py` |
| F7-010 | P2 | CONFIRMED | NEW — same tab as F7-001 but a distinct  | `components/_callbacks_alerts.py` |
| F7-011 | P2 | CONFIRMED | NEW — #185 is the closest registry item  | `components/callbacks.py` |
| F8-001 | P2 | CONFIRMED | NEW — extends seed S5-03 (whose reachabi | `components/insights.py` |
| F8-002 | P2 | CONFIRMED | NEW — no registry issue or seed covers a | `components/accessibility.py` |
| F9-001 | P2 | CONFIRMED | seed-dup:S4-04 | `health.py` |
| F9-003 | P2 | CONFIRMED | #187-extension (value-contradiction + de | `config.py` |
| F9-004 | P2 | CONFIRMED | NEW | `app.py` |
| F9-005 | P2 | CONFIRMED | NEW (#127-adjacent — engine wiring track | `simulation/scenario_engine.py` |
| F9-006 | P2 | CONFIRMED | seed-dup:S5-06 | `app.py` |
| F9-009 | P2 | CONFIRMED | #187-extension — #187 tracks Cloud Run f | `Dockerfile` |
| S1-04 | P2 | CONFIRMED | NEW — not in any known issue; adjacent t | `models/pricing.py` |
| S1-05 | P2 | CONFIRMED | #181-dup — the run-to-run variance, the  | `jobs/training_job.py` |
| S2-02 | P2 | CONFIRMED | #184-extension (#184 flags the sibling a | `models/model_service.py` |
| S2-04a | P2 | CONFIRMED | #166-dup (the zero-residual write is doc | `jobs/phases.py` |
| S3-01b | P2 | CONFIRMED | NEW (#185 refactors the three fallback b | `data/eia_client.py` |
| S3-03 | P2 | CONFIRMED | NEW (#174 added the GCS fallback reads/w | `data/gcs_store.py` |
| S4-04 | P2 | CONFIRMED | NEW (STATUS.md:53 acknowledges the parti | `jobs/scoring_job.py` |
| S5-06 | P2 | CONFIRMED | NEW (not in known_issues.json; adjacent  | `components/layout.py` |
| X1-001 | P2 | CONFIRMED | NEW — #184/#185 are models/data-layer cl | `components/_callbacks_alerts.py` |
| X1-002 | P2 | CONFIRMED | seed-dup:S2-02 (verdict copied; the reus | `components/_callbacks_overview.py` |
| X1-004 | P2 | CONFIRMED | #184-extension (2 unlisted copies in com | `components/_callbacks_backtest.py` |
| X1-005 | P2 | CONFIRMED | NEW (fabricated-metrics class adjacent t | `components/_callbacks_overview.py` |
| X1-006 | P2 | CONFIRMED | #188-extension (same dead-config/dead-co | `components/_callbacks_overview.py` |
| X2-006 | P2 | CONFIRMED | NEW — not in known_issues.json (#150 was | `components/_callbacks_backtest.py` |
| X3-005 | P2 | CONFIRMED | NEW — the write-only census is untracked | `jobs/phases.py` |
| X3-005 | P2 | CONFIRMED | NEW — the in-code comments acknowledge t | `jobs/phases.py` |
| X3-006 | P2 | CONFIRMED | intra-dup:F3-002 — same defect verified  | `data/noaa_client.py` |
| X3-006 | P2 | CONFIRMED | NEW — companion/fix-path to X3-001 (shou | `data/noaa_client.py` |
| X3-010 | P2 | CONFIRMED | NEW — #187 reconciles _ENV_DEFAULTS doc- | `config.py` |
| X3-011 | P2 | CONFIRMED | NEW — #187 reconciles the matrix content | `config.py` |
| X3-012 | P2 | CONFIRMED | NEW — no registry issue covers the v1-pa | `components/callbacks.py` |
| X3-013 | P2 | CONFIRMED | seed-dup:S5-06 (verdict copied per proto | `models/model_service.py` |
| X3-014 | P2 | CONFIRMED | NEW — closed issue #150/#169 covered the fabri | `components/_callbacks_backtest.py` |
| F1-007 | P3 | CONFIRMED | NEW — seed S1-04 covers the 0.90-tier di | `models/pricing.py` |
| F1-008 | P3 | CONFIRMED | #184-extension / #188-extension — both i | `models/training.py` |
| F1-009 | P3 | CONFIRMED | NEW — seed S1-06 covers the 1.5x-of-90-d | `models/prophet_model.py` |
| F2-006 | P3 | CONFIRMED | NEW | `scripts/audit/verify_overview_metrics.py` |
| F2-007 | P3 | CONFIRMED | #183-extension — the ladder refactor is  | `models/model_service.py` |
| F2-008 | P3 | CONFIRMED | NEW — not in #183 (metrics resolver), #1 | `jobs/training_job.py` |
| F2-009 | P3 | CONFIRMED | seed-dup:S2-04b (part ii; the seed marke | `models/model_service.py` |
| F2-011 | P3 | CONFIRMED | #181-extension | `scripts/backtest.py` |
| F2-012 | P3 | CONFIRMED | NEW — not covered by #176/#178 (ensemble | `jobs/training_job.py` |
| F2-014 | P3 | CONFIRMED | NEW | `scripts/audit/extended_holdout_check.py` |
| F3-007 | P3 | CONFIRMED | #185-extension — #185's planned 'one fal | `data/eia_client.py` |
| F3-008 | P3 | CONFIRMED | NEW — no known issue or seed covers the  | `data/eia_client.py` |
| F3-012 | P3 | CONFIRMED | #187-extension (dead _ENV_DEFAULTS entry | `config.py` |
| F4-004 | P3 | CONFIRMED | NEW — adjacent to seed S4-01 (CONFIRMED: | `data/preprocessing.py` |
| F4-008 | P3 | CONFIRMED | NEW — no issue or seed covers session_di | `data/session_diff.py` |
| F4-010 | P3 | CONFIRMED | NEW — 422-LOC dead subsystem with GCS ma | `data/forecast_history.py` |
| F4-011 | P3 | CONFIRMED | NEW — untracked; becomes user-relevant o | `jobs/phases.py` |
| F6-002 | P3 | CONFIRMED | seed-dup:S2-01b — same defect, same file | `components/_callbacks_forecast.py` |
| F6-006 | P3 | CONFIRMED | #184-extension — #184 owns the ensemble- | `components/_callbacks_backtest.py` |
| F6-010 | P3 | CONFIRMED | NEW — not in #183/#184/#185/#186/#187/#1 | `components/_callbacks_forecast.py` |
| F6-013 | P3 | CONFIRMED | #166-extension — #166 fixes the diagnost | `components/_callbacks_models.py` |
| F7-006 | P3 | CONFIRMED | seed-dup:S5-03 (verdict copied; S5-03 al | `components/callbacks.py` |
| F7-007 | P3 | CONFIRMED | NEW — extends seed S3-04 (which covered  | `components/callbacks.py` |
| F7-008 | P3 | CONFIRMED | NEW — no registry issue or seed covers t | `components/_callbacks_us_grid.py` |
| F8-003 | P3 | CONFIRMED | #188-extension — same dead-9-tab-era-con | `personas/welcome.py` |
| F8-004 | P3 | CONFIRMED | #188-extension — #188 tracks only routin | `components/error_handling.py` |
| F8-005 | P3 | CONFIRMED | seed-dup:S5-07a (verdict copied; the sec | `components/layout.py` |
| F8-006 | P3 | CONFIRMED | NEW | `components/error_handling.py` |
| F8-007 | P3 | CONFIRMED | NEW (the surface it lives on is S5-03's  | `components/insights.py` |
| F8-008 | P3 | CONFIRMED | #188-extension — #188 files dead persona | `personas/config.py` |
| F8-009 | P3 | CONFIRMED | seed-dup:S5-03 (verdict copied; F8-009 a | `components/insights.py` |
| F8-010 | P3 | CONFIRMED | NEW (adjacent #188's dead-config theme a | `components/cards.py` |
| F8-011 | P3 | CONFIRMED | NEW (the '4 visible tabs' stale-comment  | `components/layout.py` |
| F8-012 | P3 | CONFIRMED | NEW (violates the CLAUDE.md demo-vs-real | `components/callbacks.py` |
| F8-013 | P3 | CONFIRMED | NEW — no known issue or seed covers insi | `components/insights.py` |
| F8-014 | P3 | CONFIRMED | #188-extension (test-breadcrumb/stale-te | `tests/e2e/test_dashboard_render.py` |
| F9-002 | P3 | CONFIRMED | seed-dup:S5-04a (ai_briefing kill-switch | `config.py` |
| F9-007 | P3 | CONFIRMED | #187-extension (dead-config class; sibli | `config.py` |
| F9-008 | P3 | CONFIRMED | NEW (#185/#187 adjacent; the config-vs-c | `config.py` |
| F9-010 | P3 | CONFIRMED | NEW — seed S1-06 covers the unmeasured p | `config.py` |
| F9-011 | P3 | CONFIRMED | NEW — no known issue or seed verdict tou | `app.py` |
| F9-012 | P3 | CONFIRMED | NEW — #152 covers the separate mypy-inst | `.github/workflows/ci.yml` |
| F9-013 | P3 | CONFIRMED | NEW — seed S5-05 covers the different de | `observability.py` |
| F9-014 | P3 | CONFIRMED | NEW — seed S1-04 (CONFIRMED) covers the  | `config.py` |
| F9-015 | P3 | CONFIRMED | NEW — #127 tracks the simulator heuristi | `simulation/presets.py` |
| S1-01 | P3 | CONFIRMED | NEW — #181 is weighting strategy, #184 i | `models/ensemble.py` |
| S1-02 | P3 | CONFIRMED | #184-extension — same 'modeling-layer lo | `models/evaluation.py` |
| S1-03 | P3 | CONFIRMED | NEW — holdout-path counterpart of the tr | `models/evaluation.py` |
| S1-06 | P3 | CONFIRMED | NEW — #189's audit covers duplication/st | `models/prophet_model.py` |
| S1-08 | P3 | CONFIRMED | #181-extension — the weighting strategy  | `models/ensemble.py` |
| S2-01a | P3 | CONFIRMED | NEW (adjacent to #184's model_service co | `models/model_service.py` |
| S2-01b | P3 | CONFIRMED | NEW (#150 closed by #169 covered only th | `components/_callbacks_forecast.py` |
| S2-04b | P3 | CONFIRMED | #184-dup for (i) ('Flag (do not fix here | `models/model_service.py` |
| S3-04 | P3 | CONFIRMED | NEW (#188's consistency sweep covers sta | `data/audit.py` |
| S3-05 | P3 | CONFIRMED | #185-dup (no consequence beyond what #18 | `data/weather_client.py` |
| S4-01 | P3 | CONFIRMED | NEW (adjacent to #188's consistency-swee | `data/preprocessing.py` |
| S4-02 | P3 | CONFIRMED | NEW (same stale-docstring class as #188  | `jobs/phases.py` |
| S5-01 | P3 | CONFIRMED | NEW (same consistency-sweep class as #18 | `README.md` |
| S5-02 | P3 | CONFIRMED | #131-extension (closed issue #131 fixed the pr | `components/_callbacks_forecast.py` |
| S5-03 | P3 | CONFIRMED | #188-extension (#188 files the identical | `components/insights.py` |
| S5-04a | P3 | CONFIRMED | #188-extension (same dead-config class a | `config.py` |
| S5-05 | P3 | CONFIRMED | #187-extension (item 2 covers _ENV_DEFAU | `config.py` |
| S5-07a | P3 | CONFIRMED | NEW | `components/layout.py` |
| X1-003 | P3 | CONFIRMED | NEW (#142/#155 are the CLOSED fix that i | `components/_callbacks_models.py` |
| X1-007 | P3 | CONFIRMED | NEW (seed S1-04's verdict is the 0.90 no | `components/_callbacks_overview.py` |
| X1-008 | P3 | CONFIRMED | NEW (#185 covers EIA client fetch tripli | `components/_callbacks_shared.py` |
| X1-009 | P3 | CONFIRMED | NEW — #183 is the metrics fallback ladde | `components/_callbacks_overview.py` |
| X1-010 | P3 | CONFIRMED | #186-extension (same train/serve-parity- | `components/_callbacks_forecast.py` |
| X1-011 | P3 | CONFIRMED | #184-extension — #184 item 2 covers only | `components/_callbacks_forecast.py` |
| X1-012 | P3 | CONFIRMED | NEW — not enumerated in #184 (models lay | `components/_callbacks_shared.py` |
| X1-013 | P3 | CONFIRMED | NEW — no known issue or seed covers the  | `components/insights.py` |
| X1-014 | P3 | CONFIRMED | NEW — same fast-path/fallback duplicatio | `components/_callbacks_models.py` |
| X1-015 | P3 | CONFIRMED | #166-extension (consolidation belongs wi | `jobs/phases.py` |
| X2-001 | P3 | CONFIRMED | seed-dup:S5-01 | `README.md` |
| X2-002 | P3 | CONFIRMED | S5-01-extension — seed confirmed the two | `README.md` |
| X2-003 | P3 | CONFIRMED | seed-dup:S1-08 (finder's own hint to #18 | `models/ensemble.py` |
| X2-004 | P3 | CONFIRMED | NEW — not in known_issues.json (#188/#18 | `models/xgboost_model.py` |
| X2-005 | P3 | CONFIRMED | NEW — no known issue or seed covers the  | `components/_callbacks_us_grid.py` |
| X2-007 | P3 | CONFIRMED | seed-dup:S2-01b | `components/_callbacks_forecast.py` |
| X2-008 | P3 | CONFIRMED | NEW — no known issue or seed covers READ | `README.md` |
| X2-009 | P3 | CONFIRMED | NEW — not in known_issues.json or any se | `docs/CANONICAL_FACTS.md` |
| X2-010 | P3 | CONFIRMED | #184-extension — #184 tracks the code-si | `PRD.md` |
| X2-011 | P3 | CONFIRMED | NEW — #171 tracks the runtime-headroom e | `docs/HOW_IT_WORKS.md` |
| X2-012 | P3 | CONFIRMED | NEW — no known issue or seed covers this | `docs/CANONICAL_FACTS.md` |
| X2-013 | P3 | CONFIRMED | #188-extension — same 9-tab-era stalenes | `CLAUDE.md` |
| X2-014 | P3 | CONFIRMED | NEW — #188's consistency sweep enumerate | `TECHNICAL_SPEC.md` |
| X2-015 | P3 | CONFIRMED | #188-extension — same stale-comment clas | `config.py` |
| X2-016 | P3 | CONFIRMED | seed-dup:S1-06 for the provenance half ( | `config.py` |
| X3-007 | P3 | CONFIRMED | NEW — no registry issue tracks the dead  | `components/_callbacks_backtest.py` |
| X3-008 | P3 | CONFIRMED | seed-dup:S3-04 for the D2-audit-trail-ha | `components/callbacks.py` |
| X3-009-revised | P3 | CONFIRMED | #185-extension — #185 names only the dat | `components/_callbacks_overview.py` |
| F3-011 | P3 | REFUTED | #185-dup for the surviving residue (news | `components/_callbacks_overview.py` |
| F6-011 | P3 | REFUTED | #184-dup for the surviving residue — the | `components/_callbacks_forecast.py` |
| GAP-00 | P3 | REFUTED | NEW | `.github/workflows/deploy-prod.yml` |
| S1-07 | P3 | REFUTED | NEW (as a style note only) — no known is | `models/xgboost_model.py` |
| S2-03 | P3 | REFUTED | #183-dup (dev-only baseline dict deletio | `models/model_service.py` |
| S2-05 | P3 | REFUTED | NEW (no existing issue covers the gate T | `models/model_service.py` |
| S3-01 | P3 | REFUTED | NEW (not filed anywhere; #174/#185 descr | `data/eia_client.py` |
| S3-02 | P3 | REFUTED | NEW-none-reported (#174 is the issue thi | `data/eia_client.py` |
| S4-03 | P3 | REFUTED | NEW (claim refuted by shipped fixes issue #130 | `components/_callbacks_overview.py` |
| S5-04b | P3 | REFUTED | NEW for the CLAUDE.md:423 staleness (the | `config.py` |
| S5-07b | P3 | REFUTED | NEW (the dead .meeting-mode class sliver | `components/callbacks.py` |
| X2-017 | P3 | REFUTED | NEW as stated (no issue tracks it; refut | `README.md` |
| X3-009 | P3 | REFUTED | #185-extension (the claim itself is neut | `components/_callbacks_overview.py` |

</details>

---


## P0 — Integrity

### P0-1. [NEW] Production scoring job fabricates the Risk tab's alerts: `write_alerts` publishes `generate_demo_alerts()` output to Redis every hour, rendered as real and attributed to NOAA

**CONFIRMED (static + empirical). Independently confirmed by 4 verifiers — finding_ids F3-001, F4-001, X3-001 (verified twice, `jobs` and overflow passes), F7-001, F3-003-revised.**

The hourly production scoring job calls `write_alerts` for every region (`jobs/scoring_job.py:259` at 5000d6a), and that phase unconditionally imports and calls the demo generator: `jobs/phases.py:1342` `from data.demo_data import generate_demo_alerts`, `:1347` `alerts = generate_demo_alerts(region)` — the full function body (`:1340-1403`) contains no `ENVIRONMENT`, `REQUIRE_REDIS`, or feature-flag gate. `data/demo_data.py:195-240` hardcodes a "Heat Advisory … Heat index values up to 105°F expected" for ERCOT/CAISO/FPL and a "Wind Advisory: gusts up to 45 mph" for ERCOT/SPP, year-round, with now-relative expiry timestamps so they always look current. The stress score is arithmetic over the fake counts (`jobs/phases.py:1351` `stress = min(100, n_crit * 30 + n_warn * 15 + 20)` — a permanent 35/"Elevated" for FPL/ERCOT/CAISO, a fake constant 20/"Normal" elsewhere), and everything is written to `gridpulse:alerts:{region}` at `:1390`, the sole production data source for the visible Risk tab (`components/layout.py:25,33`). The renderer (`components/_callbacks_alerts.py:60-91`) attaches no demo label; worse, the provenance is actively mislabeled: `components/callbacks.py:153-156` hardcodes freshness `"alerts": "fresh"`, `components/_callbacks_overview.py:1666` badges the source as "NOAA Alerts" (rendered "LIVE" when fresh), `:1144` tells the user to "Check the timeline above for active NOAA alerts," and the Risk-tab footer credits "EIA · Open-Meteo · NOAA." Meanwhile the real NOAA client (`data/noaa_client.py`) has zero callers outside the `data/__init__.py` re-export and tests — `git log -S fetch_alerts_for_region -- jobs/ components/` is empty — despite `docs/HOW_IT_WORKS.md:41,85` diagramming a scoring-job NOAA fetch and PRD R1.3 listing NOAA alert context as Must Have. `tests/integration/test_scoring_job.py:203-226` pins the write as a change-detector.

Empirical confirmation ran the real `phases.write_alerts` with a stubbed `redis_set` (repro: `/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-energy-forecast-final/e55d5d4a-af14-4217-b412-69238186a7a8/scratchpad/repro2_ui1/repro_f3_003.py`):

```
PhaseResult: True {'n_critical': 0, 'n_warning': 1, 'n_info': 0, 'stress': 35}
Redis key written: gridpulse:alerts:FPL
alert: demo-alert-FPL-1 | Heat Advisory | Heat Advisory for FPL region until 8 PM local time
stress_score: 35 | stress_label: Elevated
```

NEW re-verified against the known-issue registry by quote-match: issue #166's body names only `jobs/phases.write_diagnostics`; #185's demo-data note covers only `news_client`; #189's audit asserts verbatim "Only two findings touch correctness … `news_client` demo-data fallback (#185) and `_predict_from_trained`'s noise fabrication (#184)" — this defect is the third and was missed. Class-sibling of #166 but strictly worse: #166's post-#149 symptom is honest-empty, while alerts are affirmatively fabricated, NOAA-attributed, and stamped fresh.

**Fix direction:** decide the honest v1 behavior first — either wire the already-written `data/noaa_client.fetch_alerts_for_region` into `write_alerts` (job-side, respecting the existing client fallback conventions), or ship an empty/`"unavailable"` alerts payload plus a "no alert feed connected" UI state — and in the same change remove the "NOAA Alerts / LIVE" attribution until a real feed backs it, correct `docs/HOW_IT_WORKS.md`/`TECHNICAL_SPEC.md` §2.3, and re-point the integration test at the honest payload. Any interim demo output must carry an explicit demo label end-to-end and be gated out of `ENVIRONMENT=production`.

---

### P0-2. [NEW] Prophet and SARIMAX forward forecasts are time-mislabeled by up to ~24h: predictions anchored at the pickled model's training end are written into Redis rows timestamped from the current scoring tick

**CONFIRMED (empirical). Independently confirmed by 2 verifiers — finding_ids F1-001, F4-002.**

Both predict functions anchor their forecast window at the *training-time* end of data: `models/prophet_model.py:161` builds `make_future_dataframe` from the pickled model's history (the `featured` argument only supplies regressor values via a `ds`-join, `:168-199`) and `:209` returns `forecast.tail(periods)`; `models/arima_model.py:238-250` reconstructs SARIMAX from the pickle's frozen `tail_y` and `fitted.forecast(steps)` necessarily continues from training end. The scoring job then discards Prophet's returned `timestamps` key (`jobs/phases.py:771-773` keeps only `result.get("forecast")`) and writes `preds[i]` at `future_ts.iloc[i]` anchored at the *current* scoring-time `_resolve_forecast_start` (`:856-867`, `:929-937`). With daily 04:00 UTC training and hourly scoring (`jobs/scoring_job.py:179-181` loads the daily GCS pickles), the label offset grows from 0 right after training toward ~23h before the next train — rotating the diurnal cycle to the wrong hours. XGBoost is unaffected (per-row prediction over the scoring-time `future_df`). Consumption is real: the Overview hero prefers the `ensemble` row key (`components/_callbacks_overview.py:127-128`), which blends aligned XGBoost with the phase-shifted Prophet/ARIMA members; the Forecast tab renders per-model row keys (`components/_callbacks_forecast.py:753-754`); and drift records score each model's row value against the labeled hour's actual — so the extreme live ARIMA drift #170 attributes to "genuine model weakness" (LDWP 188%, AZPS 266%) has an untracked alternative mechanism: wrong-hour comparison.

Verbatim repro (deployed library versions, prophet 1.3.0 + statsmodels 0.14.6; scripts: `/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-energy-forecast-final/e55d5d4a-af14-4217-b412-69238186a7a8/scratchpad/repro2_models/f1_001_timestamp_misalignment.py` and `…/scratchpad/repro2_jobs/f4_002_time_shift.py`):

```
predict_prophet returned timestamps[0]: 2026-06-10 05:00:00 | row-writer would label preds[0] as: 2026-06-10 20:00:00 | MISLABEL OFFSET: 15 hours
pred[0] = 953.3 | true demand at its REAL hour (05:00) = 1289.8 | true at the LABELED hour (20:00) = 740.2
[ARIMA] MAE vs truth at train_end+1..+24 (true origin): 0.0 MW | MAE vs truth at forecast_start..+24 (as labeled): 379.8 MW
```

NEW re-verified: distinct mechanism from the tracked/retired #129 (forecast-start anchor vs the EIA actuals gap — its body is about `featured["timestamp"].max()` vs demand-NaN rows, not the model's internal origin); no commit since 3ce0416 (which addressed Prophet *regressor* misalignment only) touches the output-window anchor. Relates to #170 and #181.

**Fix direction:** make the scoring job honor the model's own forecast origin rather than assuming alignment — either re-anchor per tick (extend `periods` by the training-end→forecast-start gap and trim to the requested window, using Prophet's returned `timestamps` and an explicit SARIMAX origin) or join predictions to row timestamps by the timestamps the model actually emitted, dropping/flagging any model whose window cannot cover `forecast_start`. Add a regression test asserting per-model prediction timestamps equal the row timestamps they are written to, and re-examine #170's per-model drift numbers after the alignment lands.

---

## P1 — User-visible correctness

### P1-1. [NEW] Published holdout metrics are incommensurable and flattered: XGBoost's "168h holdout" is 168 teacher-forced one-hour-ahead predictions (with actual observed weather for all three models), while production inference is recursive with forecast/climatology weather

**CONFIRMED (static, with empirical premise check). Finding_ids F1-002, F2-002 (intra-corroborating).**

`jobs/training_job.py:60-64` slices `val_df = featured_df.iloc[-_HOLDOUT_HOURS:]` and calls `predict_xgboost(holdout_model, val_df)` directly; `featured_df` is engineered over the full actual series before the slice (`:468, 502-505`), so every holdout row's `demand_lag_1h` is the *actual* demand one hour earlier — from inside the holdout window. The repro confirms the premise (`/private/tmp/…/scratchpad/repro2_jobs/f4_005_011_f1_002.py`): "lag feature == actual in-window demand? True | that source hour is inside the holdout window? True". Prophet (`:103`) and ARIMA (`:157-177`) holdouts are genuine 168-step forecasts (no demand features in `PROPHET_REGRESSORS`/`ARIMA_EXOG_COLS`), so the three numbers feeding the 1/MAPE ensemble weights and `docs/BACKTEST_RESULTS.md`'s like-for-like table are measured under different protocols. Additionally, all three holdouts score against *actual observed* weather, while production serves forecast-overlay + climatology weather (`jobs/phases.py:394-419, 861-866`) and recursive predicted-lag XGBoost (`:675-741`); the honest recursive holdout exists at `models/training.py:112-122` but `train_all_models` has no non-test caller, and the comment at `jobs/phases.py:688-691` claiming the inference behavior "matches … the training holdout" is wrong for the persisted path. `docs/BACKTEST_RESULTS.md` says verbatim "for *what production serves by default*, read the ensemble column" — a claim the protocol does not support. The #135 leakage change covered same-row leakage only.

NEW re-verified: #186's body is about the two *feature-coder implementations* agreeing ("two parallel implementations of the autoregressive features that must agree row-for-row"), not the evaluation protocol; #181 consumes these numbers ("this compares **168h holdout MAPE**; the live-served accuracy is the 1h-ahead drift metric") without identifying the teacher-forcing asymmetry. Relates to #181 and #186 — note #181's weighting analysis (48/51 BAs "won" by XGBoost) is computed on these incommensurable bases and should be revisited after the protocol change.

**Fix direction:** persist a recursive (autoregressive-snapshot) XGBoost holdout as the metric of record — `models/training.py:112-122` and `scripts/audit/extended_holdout_check.py` already contain the machinery — so all three models are scored multi-step under production-like inputs; recompute ensemble weights and regenerate `docs/BACKTEST_RESULTS.md` from the commensurable numbers, and relabel or caveat any surface that presents teacher-forced values as "what production serves."

### P1-2. [NEW] The "80% empirical prediction interval" on both the Overview hero and the Forecast tab is calibrated from XGBoost residuals regardless of the model displayed

**CONFIRMED (empirical). Independently confirmed by 2 verifiers on two surfaces — finding_ids F5-002 (Overview), F6-003 (Forecast tab); same underlying defect in the shared collector.**

The only Redis backtest writer stores `"predictions": {"xgboost": preds}` exclusively (`jobs/phases.py:1149-1166`). The shared collector `_collect_backtest_residuals` (`components/_callbacks_shared.py:357-377`) then substitutes any available model when the requested one is absent (`elif preds_map: next(iter(preds_map.values()))`), so a request for `"ensemble"` (Overview hero, `components/_callbacks_overview.py:433-445`, label "80% prediction interval (empirical, n=…)") or for prophet/arima/ensemble on the Forecast tab (`components/_callbacks_forecast.py:224-241, 839-844`, legend "80% empirical prediction interval") silently returns XGBoost-derived quantiles. Repro verbatim (`/private/tmp/…/scratchpad/repro2_ui2/r2_band_substitution.py`, also `…/repro2_ui1/repro_ui1.py`): `model=ensemble available=True lower=50.0 upper=50.0 … model=prophet … lower=50.0 upper=50.0 … model=arima … lower=50.0 upper=50.0` — identical xgboost residuals for every selection. Given #181's own measurement (ensemble median MAPE 3.48% vs XGBoost 2.32%), the band systematically understates the plotted ensemble's error. Secondary defects verified on the Forecast surface: the "calibration window: last {N}h" caption reports a pooled sample count, not a recency window, and can double-count when a stale legacy `backtest:{region}:{horizon}` key coexists (`sample_size: 336` from the same 168 residuals); a single pooled (q10,q90) is applied constant-width across the whole horizon (`models/evaluation.py:164-171`).

NEW re-verified: the retired #150 covered the Prophet `lower_80*0.95` heuristic band, a different branch; #181 is weighting strategy; #153 is payload typing. Relates to #181 (its "step 0" — persisting per-model prediction vectors — would serve both).

**Fix direction:** make the collector honest about identity — return residuals only for the requested model (or return a `substituted_from` marker the caller must surface), and have the training job persist per-model prediction vectors in the backtest payload (the same infrastructure #181 needs) so ensemble/prophet/arima bands are genuinely self-calibrated; until then, either label the band "calibrated on XGBoost residuals" or fall back to the labeled heuristic band, and correct the caption to describe pooled folds rather than a chronological window.

### P1-3. [NEW] Data freshness is asserted, never measured: the Redis fast path hardcodes every source to `'fresh'` with a render-time timestamp, so up-to-24h-stale data renders with no degradation signal and the E2 staleness thresholds are dead config

**CONFIRMED (empirical). Independently confirmed by 2 verifiers — finding_ids F7-004, X3-004.**

`_load_data_from_redis` builds `freshness = {"demand": "fresh", "weather": "fresh", "alerts": "fresh", "timestamp": datetime.now(UTC)…}` (`components/callbacks.py:153-158`, re-asserted `:181-182`) with no comparison of payload age to any threshold; `config.STALENESS_THRESHOLDS_SECONDS` (`config.py:633-639`, Backlog E2) has zero non-test consumers; `update_widget_confidence` (`callbacks.py:890-914`) computes `age_seconds` from the callback-run timestamp (restamped every 300s by the refresh interval), so the 7200s stale check in `error_handling.py:300-323` mathematically cannot fire on old pipeline data. The honest age signal `gridpulse:meta:last_scored` is read only by `health.py:87`, never by any UI callback, and payloads carry no write timestamp, so EIA stale-cache/GCS-fallback-sourced data is indistinguishable from live. `REDIS_TTL=86400` (`jobs/phases.py:49`) bounds the unmarked-stale window at 24h. Repro verbatim with a 23h-old payload (`/private/tmp/…/scratchpad/repro2_ui5/x3_004_freshness_hardcoded.py`):

```
payload data age: ~23h; freshness store: {demand: fresh, weather: fresh, alerts: fresh}
age_seconds computed by update_widget_confidence: 0.0s; confidence level: high -> Live data from verified source
header badge: all fresh -> GREEN Live
```

One verifier correction: the header badge and per-widget confidence bar render into `display:none` carriers (`layout.py:123, 164`), so during a stalled scoring job the user-visible symptom is *no staleness indication anywhere* (the fallback banner only fires on states the Redis path never sets) — the same integrity failure with a quieter face. The 2026-06-01 (~4.5h stall) and 2026-06-04 incidents are the documented occurrence mode. NEW re-verified: #188's warming-state item is about routing degraded output through `warming_state()`; #153 is typed payloads; neither tracks age-based grading.

**Fix direction:** make freshness measured, not asserted — stamp `scored_at`/`fetched_at` into every job-written payload (a natural rider on #153's typed contracts), grade it on read against `STALENESS_THRESHOLDS_SECONDS` (finally giving E2's config a consumer), drive the fallback banner from the graded state, and have the header consume `gridpulse:meta:last_scored` the way `/health` already does. Delete or wire the hidden badge carriers so the signal is actually visible.

### P1-4. [NEW] The Overview hero and insight discard `scored_at` entirely: a stale forecast renders as "Next-24h forecast…" with past timestamps, and in a partial-failure split the forecast bridge is drawn backwards over the actuals

**CONFIRMED (empirical). Finding_id F5-006.** (Sibling of P1-3, distinct mechanism: the staleness field exists in this payload and is discarded.)

`components/_callbacks_overview.py:412-417` and `:587-589` unpack `..., _scored_at = forecast_payload` and never use it, and `update_overview_tab` (~`:2380-2381`) executes `del weather_json, freshness_data` — the flagship surface has no staleness cue of any kind; only the Forecast tab consumes `scored_at` (`components/_callbacks_forecast.py:771` "FORECAST AS OF"). Repro verbatim (payload scored 20h ago, actuals 20h fresher — the reachable shape where the fetch phase succeeds but the predict phase fails; script `/private/tmp/…/scratchpad/repro2_ui1/repro_ui1.py`):

```
hero-chart traces: ['Actual', '±3% indicative range', 'Forecast (24h)']
first forecast point: 2026-07-01 13:00 (19h BEFORE the bridge anchor)
forecast points strictly before the last actual: 19/24 (bridge drawn backwards over the actuals)
no trace, label, or badge reflects scored_at (=20h ago)
```

In the uniform-stall variant (both stale) the "next-24h" narrative and peak time silently describe a partially elapsed window — precisely the 2026-06-01 incident, during which the 08:00 forecast presented as current until 12:00+. NEW re-verified: not #129 (retired; that was the forecast-*start* gap from EIA publishing lag); untracked.

**Fix direction:** consume `scored_at` on the Overview — render a "forecast as of" cue (matching the Forecast tab's existing pattern), suppress or visually mark forecast points that fall before the last actual, and past a staleness threshold degrade to the warming/stale presentation rather than presenting elapsed hours as a forward forecast. Coordinates naturally with P1-3's graded freshness.

### P1-5. [NEW] The Forecast tab's Generation panel performs a live EIA API fetch in the stateless web tier's request path, ignoring the `gridpulse:generation:{region}` payload the scoring job writes hourly

**CONFIRMED (static, every hop read verbatim). Independently confirmed by 2 verifiers — finding_ids F5-001, X3-003.**

Registered callback `update_forecast_generation_panel` (`components/_callbacks_forecast.py:970-986`, fires when the Generation collapse opens on tab-outlook) → `_build_generation_panel` → `_fetch_generation_cached` (`components/_callbacks_overview.py:861`, body `:667-707`), which at `:689-692` does `from data.eia_client import fetch_generation_by_fuel; gen_df = fetch_generation_by_fuel(region)` — a live EIA HTTP call with the client's full retry budget (~150s per failing call pre-circuit-breaker trip) plus stale-cache→GCS reads, inside a Dash callback on the web tier. `grep REQUIRE_REDIS components/_callbacks_overview.py` returns zero hits; the deploy workflow mounts `EIA_API_KEY` on the same service that sets `REQUIRE_REDIS=true`, so the key gate passes in prod. Meanwhile the scoring job writes `gridpulse:generation:{region}` hourly (`jobs/phases.py:286`) and the existing Redis reader `_generation_tab_from_redis` (`components/_callbacks_generation.py:55`) is orphaned — `components/callbacks.py:64` says verbatim the "fast path is currently orphaned in register_callbacks". This is the exact violation class of the CLAUDE.md post-PR#130 web-tier I/O guardrail; the guardrail test (`tests/integration/test_callbacks_redis_only.py`) covers only three other gates. Bonus staleness: the docstring at `:668` claims a demo third tier that `:704-706` contradicts.

NEW re-verified: #185 is EIA client-code DRY, #153 is payload typing; PR #130's changes covered the Overview hero and model badge, not this panel; the orphaned fast path is untracked.

**Fix direction:** convert the panel to read `gridpulse:generation:{region}` (un-orphaning `_generation_tab_from_redis` or a shared reader), render the warming state on a Redis miss under `REQUIRE_REDIS`, delete the request-path EIA import, and extend `test_callbacks_redis_only.py` to pin this callback so the guardrail is structural rather than conventional.

### P1-6. [NEW] `update_alerts_tab` has no `REQUIRE_REDIS` warming gate: any production Redis miss falls straight to an inline `generate_demo_alerts()` render

**CONFIRMED (empirical). Independently confirmed by 2 verifiers — finding_ids F3-003, X3-002.** (Marginal user impact is bounded by P0-1 — on a Redis miss users see the same fabricated alerts the scoring job would have written — which is why this lands P1 despite one verifier proposing P0; it must be closed alongside P0-1 or the fabrication survives that cleanup.)

`components/_callbacks_alerts.py:343-358`: on a `gridpulse:alerts:{region}` miss the callback calls `generate_demo_alerts(region)` inline with no gate — `grep REQUIRE_REDIS components/_callbacks_alerts.py` exits 1, while the sibling gates exist at `components/callbacks.py:272`, `components/_callbacks_forecast.py:354`, `components/_callbacks_backtest.py:561`. The module docstring (`:12-17`) claims "There's no fallback compute path," contradicted by its own registration docstring (`:274-276`). Repro verbatim with `ENVIRONMENT=production` and `redis_get -> None` (`/private/tmp/…/scratchpad/repro2_ui5/x3_002_alerts_no_gate.py`, also `…/repro2_ui1/repro_f3_003.py`):

```
returned 8 outputs (no warming early-return fired); stress score rendered: 35 | label: Elevated
rendered card text: Heat Advisory / Heat Advisory for ERCOT region until 8 PM local time / Wind Advisory ...
any demo/sample/synthetic label in rendered output: False
```

The trigger window is real: the key's 24h TTL means cold-on-fresh-deploy, after a Redis flush, or after a >24h scoring stall (incident classes documented in #171/#174). NEW re-verified: #188's warming item covers overview/forecast hand-rolled degraded output, not this fabrication path.

**Fix direction:** add the same `REQUIRE_REDIS` warming gate the other three callbacks use (routing through `error_handling.warming_state()` per #188's direction), restrict the demo fallback to dev with an explicit demo label, and fix the module docstring; add this callback to the redis-only guardrail test. Should ship with (or immediately after) the P0-1 alerts fix.

### P1-7. [NEW] Absent metric fields render as fabricated perfection: the Overview model card and Models leaderboard format missing values as "MAPE 0.0%" / "RMSE 0 MW" / "R² 0.000", toned positive

**CONFIRMED (empirical). Finding_id F5-010.**

`components/_callbacks_overview.py:528-533` formats `m.get('mape', 0.0)` (and siblings) into the model card; the leaderboard loop (`:1061-1072`) renders any model lacking `mape` as "0.0%" with the positive tone class. Partial dicts are a *supported* prod payload state: `models/model_service.py:121-123` documents "Empty fields omitted; the UI tolerates partial dicts," and both `jobs/scoring_job.py:39-84` (`_extract_holdout_metrics`, per-field None/non-finite drops, mape-only legacy fallback) and `jobs/phases.py:958-977` (per-field sanitizer) produce them by design — the #176/#179 NaN-holdout era is a concrete precedent where `mape` would be dropped while `rmse` survives, yielding a positive-toned "MAPE 0.0%". Repro verbatim (`/private/tmp/…/scratchpad/repro2_ui1/repro_ui1.py`):

```
get_model_metrics -> {'ensemble': {'rmse': 900}} (no mape/mae/r2)
model card text: Model Ensemble simulated MAPE 0.0% RMSE 900 MW MAE 0 MW R² 0.000
leaderboard classes containing 'positive' for the fabricated 0.0%: ['gp-metric-value tabular gp-metric-value--positive']
```

The honest pattern already exists in the same file — the Overview metrics bar renders "—" for missing values (`:219-229`). NEW re-verified: distinct from the retired #131 (simulated values sourced upstream; this is render-side fabrication from honest partial data) and from #183 (resolver structure).

**Fix direction:** treat missing as missing — adopt the metrics bar's "—" pattern in `build_model_metrics_card` and the leaderboard (no default-0.0 `get`, no tone class for absent values), and add a unit test feeding a partial dict that asserts no "0.0%" is rendered.

### P1-8. [NEW] The Overview insight labels an sMAPE value "MAPE": an artifact-prone region can display "live 7d MAPE 18.0%" while its true rolling MAPE is ~190%

**CONFIRMED (empirical). Finding_id F5-005.**

`_resolve_forecast_mape` prefers `rolling_smape_7d` (per the #142/PR-G9 robustness change; `components/_callbacks_overview.py:181-186`) but returns no metric name, and the render string hardcodes the label: `:632` `mape_clause = f" ({mape_source} MAPE {mape_value:.1f}%)"`. Repro verbatim using the project's own fixture magnitudes (`/private/tmp/…/scratchpad/repro2_ui1/repro_ui1.py`): `resolver returned: (18.0, 'live 7d') … insight text: …(live 7d MAPE 18.0%). -> renders 'live 7d MAPE 18.0%' while the true rolling MAPE is 190.0%`. No user-facing "sMAPE" string exists anywhere in `components/` (grep), and the Models tab's live drift panel displays `rolling_mape_7d` (`components/_callbacks_models.py:291-292, 333-336`) — so for an LDWP-class BA the two live surfaces show 18.0 vs ~190 for what both present as MAPE, a 10x cross-surface discrepancy exactly where the sMAPE substitution was adopted because the statistics diverge. NEW re-verified: #142/#155 (both settled) introduced the preference; the mislabel is untracked; adjacent to but distinct from #170.

**Fix direction:** have the resolver return the metric name alongside the value and render it truthfully ("live 7d sMAPE 18.0%"), align the Models tab drift column on the same statistic (or label both explicitly), and pin the rendered label in the honest-signals unit test.

### P1-9. [EXT of #127] The Forecast tab's Scenarios panel baseline is prod-dead: it sources from `model_service.get_forecasts`, which under `REQUIRE_REDIS` always returns `unavailable` on the web tier, so the panel permanently renders "Awaiting baseline forecast" — while the real ensemble baseline sits in Redis in the same module

**CONFIRMED (static). Finding_id F5-003.**

Overlapping sentence from tracked issue #127: *"The Forecast tab's scenario simulator currently uses a hand-tuned **analytical heuristic** in `components/_callbacks_overview._scenario_demand_factor` rather than the full-fidelity physics engine in `simulation/scenario_engine.py`."* **Delta:** #127 debates heuristic-vs-physics and even sketches `scenario_y = baseline_forecast * demand_factor` *assuming a working baseline* — but in production there is no baseline at all. `_build_scenarios_panel` (`components/_callbacks_overview.py:1198-1228`, callback at `components/_callbacks_forecast.py:1026-1050`) calls `get_forecasts(region, demand_df, models_shown=['ensemble'])`; `models/model_service.py:54-74` has no Redis forecast read, and with no local pickle under `REQUIRE_REDIS` (the web tier's permanent state) returns `{'source': 'unavailable', …}` — so `base_y` stays `None` and the panel renders the 4-dash KPI bar plus `_empty_figure('Awaiting baseline forecast')` forever. Meanwhile `_read_ensemble_forecast_from_redis` in the *same file* (`:94-141`) already reads the real ensemble baseline for the hero chart. In dev, the baseline is `_simulate_forecasts` output charted as a trace literally named "Baseline" (`:1314`) with no simulated label — `_build_scenarios_panel` never inspects `forecasts['source']`.

**Fix direction:** point the panel's baseline at the Redis ensemble payload via the existing `_read_ensemble_forecast_from_redis` (a small change independent of #127's heuristic-vs-physics question), render warming on a miss, and label any dev-simulated baseline as simulated; fold this into #127's scope note so the eventual physics upgrade builds on a live baseline.

### P1-10. [NEW] The US Grid "National Peak (24h)" is the max of per-region maxima (the largest single BA's peak), not a national peak

**CONFIRMED (empirical). Finding_id F7-005.**

`components/_callbacks_us_grid.py:183-189` computes `peak_24h_mw` as an outer max over regions of each region's own `today_mw` max, while `total_mw` (`:180`) is a genuine cross-region sum; the two render side by side in the same MetricsBar (`:230-237`). Repro verbatim (`/private/tmp/…/scratchpad/repro2_ui3/f7_005_peak.py`): finder case `{'A': 50000, 'B': 70000} -> 'Total Demand = 120.0 GW' / 'National Peak (24h) = 70.0 GW'`; scaled case, 51 BAs each peaking 9.5 GW at the same hour → `'Total Demand = 459.0 GW' / 'National Peak (24h) = 9.5 GW'` (true simultaneous national peak: 484.5 GW). With real data the "National Peak" necessarily displays several times below the adjacent Total Demand. No test pins either semantic (the NaN-guard test uses a single region where the two coincide). NEW re-verified: no registry issue covers the US Grid metrics bar.

**Fix direction:** compute the peak of the summed cross-region series (handling non-aligned `today_mw` windows across BAs — sum on aligned timestamps, then max), or relabel the slot "Largest BA Peak (24h)" if the single-BA semantic is intended; add a multi-region test pinning the chosen semantic.

---

---

## P2 — Latent bugs & material debt

All file:line citations reference SHA `5000d6a267d2becbfb7f9faf2e99c406641b1b36`. Repro script paths are relative to the review scratchpad (`/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-energy-forecast-final/e55d5d4a-af14-4217-b412-69238186a7a8/scratchpad/`). The 70 CONFIRMED P2 verdicts merge to **52 canonical findings** (18 duplicate reports folded in); duplicates are noted as "independently confirmed by N verifiers."

---

### 1. Pipeline silent-failure & monitoring blind spots

**P2-01 · Scoring job reports success at 1-of-51 regions, and a region counts as "scored" if any phase ran** — `jobs/scoring_job.py:265` sets `summary["ok"] = any(p.get("ok") ...)` and `:313-315` returns `0 if ok_count > 0`, so a run where the forecast phase failed for all 51 BAs (or where every Redis write failed but the fetch succeeded) exits 0, refreshes `last_scored` unconditionally (`:296-300`), and never trips the PR #165/#148-era failed-execution alert; `/health?deep=1` cannot catch it either, since `health.py:100-113` checks only `last_scored` age and `:129,133` samples only the first-scored default region (FPL, ordered first by `jobs/phases.py:105-111`). [NEW] — STATUS.md acknowledges the partial-write class informally but no issue tracks it; independently confirmed by 3 verifiers (S4-04, F4-003, F9-001). Empirical (`repro_S4/repro_s4_04_exitcode.py`): `1-of-51 regions ok -> run() exit code: 0 ... region where EVERY Redis write failed (only fetch ok): summary['ok'] = True`. Fix direction: define region-ok as "forecast phase ok and its Redis write ok," fail the run (or emit a distinct alertable signal) below a threshold fraction of regions, write `last_scored` only from successful regions, and have deep-health inspect `regions_failed`.

**P2-02 · Training job exits 0 when 16 of 17 regions in a task fail** — `jobs/training_job.py:712` returns `0 if ok_count > 0 else 1` and `:668-677` swallows every per-region exception into `{'ok': False}` with no re-raise, so Cloud Run sees success and never retries, defeating the resume logic (`:448-453`) that was built specifically to make retries cheap. [NEW] — training-side sibling of P2-01, distinct entry point and aggregate rule (F2-010). Fix direction: same threshold-based exit policy as P2-01, plus a `regions_failed`-driven alert, so a mostly-failed training task triggers the Cloud Run retry the resume logic already supports.

**P2-03 · Redis failure is structurally invisible to the scoring job** — `data/redis_client.py:21-51` locks in a failed first ping for the process lifetime (`_redis_init_attempted` is never reset), `redis_set` swallows all errors into a bool (`:96-106`) that every caller in `jobs/phases.py` discards (`:220,230,286`), and `redis_available()` is stale in both directions — so a Redis blip at job start yields a run that writes nothing yet exits 0 and logs `scoring_job_complete`. [NEW] (F3-004); adjacent to #148 and #153, neither of which names the client mechanisms. Empirical (`repro2_data/f3_004_redis_silent.py`): `after recovery, call2 _get_redis(): None ... write_actuals_and_weather -> ok: True | error: None ... (no exception raised; Redis received nothing)`. Fix direction: allow re-init after failure (or TTL the lockout), make `redis_set` failures raise or propagate into PhaseResult ok-flags, and count Redis-write failures in the run summary that P2-01's exit policy consumes.

**P2-04 · `latest.json` pointer-race exhaustion is silent success** — `models/persistence.py:228-322`: `_write_latest`'s PreconditionFailed retry loop has no backoff (grep: no `sleep` in the file) and returns `None` on all paths, so after exhaustion `save_model:395-406` still logs `model_saved` and returns the version — the training summary reports success while `latest.json` points at yesterday's model and `write_extra_to_meta`'s ensemble metrics land on a blob nothing references; the docstring's claimed `get_model_metadata` legacy-meta fallback does not exist. [NEW] (F2-013) — `persistence.py` appears in no tracked issue. Empirical (`repro2_overflow/f2_013_write_latest_silent.py`): five `model_latest_race_retrying` lines within one second, then `model_latest_race_exhausted` followed by `model_saved ... version=20260702T082829Z (caller sees SUCCESS: not None)`. Fix direction: return a success/failure result from `_write_latest`, add jittered backoff, and make `save_model` propagate pointer-write failure into the training summary (and P2-02's exit policy).

**P2-05 · GCS parquet backups ride unjoined daemon threads killed at job exit** — `data/gcs_store.py:112` starts each `write_parquet` upload as `daemon=True` with no join/atexit anywhere in `jobs/`, and `scoring_job.run()` returns immediately after `write_meta`, so tail-region `latest.parquet` refreshes are silently lost with zero log signal; the only failure signal (`gcs_write_failed`) has no metric or alert consumer. This ages the #174 outage-fallback-of-record invisibly until the next EIA outage exposes it. [NEW]; independently confirmed by 2 verifiers (S3-03, F3-010). Empirical (`repro_S3/s3_03_child.py`): `child: upload threads alive=1 daemon_flags=[True] ... marker_exit.txt exists: NO` (in-flight upload killed at exit; nothing logged). Fix direction: track upload futures and join them (bounded) before job exit, and surface upload failure counts in the run summary.

### 2. Ingestion & fallback data-integrity gaps

**P2-06 · Mid-pagination truncation is cached for 24h and overwrites the GCS last-known-good** — `data/eia_client.py:421-423` breaks out of `_paginated_fetch` when a page hard-fails, returning partial records; the caller's only guard is on `all_records` being falsy, so the truncated frame is `cache.set` (TTL 86400) and written over `latest.parquet` (`:244,257-261`), corrupting the durable fallback during a partial outage. [NEW] (F3-006) — distinct mechanism from P2-07; relates to #174/#185 but neither covers partial-fetch integrity. Empirical (`repro2_data/f3_006_truncation.py`): `returned rows: 5000 (truncated: server said total=15000, we kept 5000) ... => truncated dataset overwrote gs://.../generation/ERCOT/latest.parquet: True`. Fix direction: compare row count against the server-reported total, and on shortfall either raise into the stale→GCS fallback or mark the frame incomplete and skip cache/GCS writes.

**P2-07 · HTTP-200 responses that parse to zero demand rows bypass the fallback and poison the cache** — the fallback guard at `data/eia_client.py:171` tests the raw `all_records`, not the parsed frame, so a 200 with only type-DF rows produces a 0-row frame that is cached for 86400s (`:184-189`); subsequent calls serve the poisoned empty frame without ever consulting the stale/GCS data the #174 machinery holds. [NEW] (S3-01b). Empirical (`repro_S3/s3_01b_partial_records.py`): `[info] eia_demand_cached region=ERCOT rows=0 ... call2 rows: 0 GCS reads (unchanged means fallback skipped): []`. Fix direction: move the fallback decision after parsing (empty parsed frame → stale→GCS chain) and never cache an empty frame over prior real data.

**P2-08 · Generation/interchange parsers coerce EIA nulls to 0.0 MW, contradicting the demand parser's documented policy in the same file** — `_parse_generation_records` (`data/eia_client.py:553`) and `_parse_interchange_records` (`:570`) do `float(r.get("value", 0) or 0)`, while `_parse_demand_records:499-513` documents and implements null/0→NaN; missing fuel observations therefore render as real 0 MW in fuel-mix, renewable-share, and net-load views, and unit tests lock the zero-fill in. [NEW] (F3-009) — #185 covers fetch-skeleton duplication, not value semantics. Fix direction: apply the demand parser's null→NaN policy to both parsers (with downstream NaN handling in `jobs/phases.write_generation`) and update the tests that pin the wrong behavior.

**P2-09 · No tier bounds or propagates data age; the web tier hardcodes `fresh` on any Redis hit** — the chain is unbroken end to end: `data/cache.py:114-120` serves stale with no age ceiling, `gcs_store.read_parquet:131-143` never checks blob age, EIA/weather fallbacks return unmarked frames (`eia_client.py:173-181`, `weather_client.py:181-193`), scoring payloads carry no `generated_at` (`jobs/phases.py:215-219`), and `components/callbacks.py:153-158,181-182` asserts `freshness='fresh'` unconditionally on every Redis hit while the age-aware `freshness_badge` helper (`components/error_handling.py:202-238`) has zero product callers. A sustained outage republished from GCS would therefore display as fresh. [NEW] (F3-005) — systemic; adjacent to #153/#188, which name neither age propagation nor the orphaned badge. Fix direction: add `generated_at` to scoring payloads (fits the #153 typed-contract work), compute freshness from it in the web tier, and enforce an age ceiling on the stale-cache and GCS fallback paths.

### 3. Model store & web-tier quality gate

**P2-10 · The forecast quality gate runs live GCS reads in web-tier request paths, is frozen by a process-lifetime pointer cache, and fails open** — `app.py:129` builds the layout once at import, running `is_forecast_quality_acceptable` for all 51 BAs (`components/layout.py:71-80`), each resolving to real GCS `latest.json` + `meta.json` reads (`models/persistence.py:475-501`); `_callbacks_us_grid.py:76-80,125` repeats the sweep per render. Worse, `persistence.py:201-202` pins `_latest_cache` at first read for the process lifetime and no production caller ever invalidates it, so even the callback path's 600s TTL re-reads the same pinned version forever and never reflects daily retraining; and since `get_xgboost_holdout_mape` returns None on any exception and None means pass (`models/model_service.py:343-366,389-392`), the gate silently disables when GCS is unreachable. [NEW] — violates the CLAUDE.md web-tier Redis-only guardrail from a surface its watchlist doesn't cover; independently confirmed by 5 verifiers (S5-06, F2-003, F9-006, F7-002, X3-013). Fix direction: have the scoring job publish per-region gate status to a `gridpulse:*` key, make the dropdown/US-Grid read Redis, and make gate evaluation fail closed (or visibly degraded) rather than silently open.

**P2-11 · One transient GCS blip negative-caches the entire model store for the process lifetime** — `models/persistence.py:222-225` stores `{}` in `_latest_cache` on any `latest.json` read failure, and the guard at `:201-202` then returns it to every caller with no retry; no production code calls `invalidate_latest_cache` or passes `force=True` (grep: only tests and `scripts/export_holdout_metrics.py`). One failed read at the start of a scoring run makes all 51 regions × 3 models unloadable for the whole run; on the web tier it blanks the quality gate and meta layers until restart. [NEW] (F2-004) — distinct from P2-10 (failure-pinning vs success-staleness); #183's resolver refactor is adjacent but doesn't name it. Fix direction: never cache the failure sentinel — cache only successful reads, and add a short TTL or retry-on-empty so a blip self-heals.

### 4. Model & training methodology

**P2-12 · `auto_arima` order selection silently ignores the weather exogenous matrix** — `models/arima_model.py:323-347` passes `exogenous=exog_sub`, a keyword removed in pmdarima 2.x; in the pinned pmdarima 2.1.1 the kwarg lands in `**fit_args` and every candidate SARIMAX is constructed with `exog=None` (pmdarima `auto.py:330-379`), so orders are selected for a no-exog model while the final fit includes exog, with only a swallowed FutureWarning. [NEW] (F1-003). Fix direction: pass `X=exog_sub` per the pmdarima 2.x API and add a unit test asserting order selection actually consumes the exog matrix (e.g. via a mock or a synthetic series where exog changes the selected order).

**P2-13 · Prophet serves weather-blind while XGBoost/ARIMA got the weather-forecast fix in the same function** — `jobs/phases.py:771` calls `predict_prophet(model, featured, ...)` with only the historical frame, so `models/prophet_model.py:194-199` forward-fills the last observed regressor values across the whole horizon (temperature, radiation, wind, CDD/HDD frozen at the final historical hour), while the sibling code paths at `phases.py:861-866,779` thread the real weather forecast to the other two models. [NEW] (F4-007) — a call-site data-flow gap distinct from #184's Prophet-mode item. Fix direction: pass the future weather frame (already built for XGBoost/ARIMA) into `predict_prophet` so its regressors carry forecast values instead of a frozen tail.

**P2-14 · Future `is_holiday` is imputed from (hour, dow) historical means, never 1** — `_build_future_feature_frame` explicitly constructs only hour/dow/month/doy/sin/cos/is_weekend (`jobs/phases.py:630-638`); `is_holiday` falls into `non_time_cols` and is filled from `hist.groupby(['_hour','_dow'])` means (`:646-661`), so the models never see a holiday flag inside the 720h horizon and holiday load drops are systematically over-forecast. [NEW] (F4-005). Empirical (`repro2_jobs/f4_005_011_f1_002.py`, horizon spanning July 4 2026): `is_holiday values written by _build_future_feature_frame: min=0.0000 max=0.0000 | correct flag per compute_holiday_flag: min=1.0 max=1.0`. Fix direction: compute `is_holiday` (and any other calendar-derivable feature) directly from the future timestamps via `compute_holiday_flag` instead of the group-mean imputer.

**P2-15 · XGBoost CV MAPE is optimistically biased and measures a model that is never shipped** — each fold at `models/xgboost_model.py:87-92` uses the validation fold both to drive early stopping (best_iteration) and to produce the reported fold MAPE — textbook selection-on-the-test-set — while the persisted artifact (`:95-97`) is a full 6000-tree fit with early stopping removed that no fold ever scored; via `jobs/training_job.py:268-291`, `cv_mape` can become `meta.mape` and thus the ensemble-weight/quality-gate basis when the holdout is unavailable. [NEW] (F1-005) — relates to #181 (weighting) but the CV-protocol defect is untracked. Fix direction: split an inner early-stopping eval set from the scored fold (or fix `n_estimators` from CV and refit), and persist which protocol produced `meta.mape`.

**P2-16 · On partial holdout failure, the persisted ensemble metric and the served ensemble diverge in both composition and weights** — when one model's holdout returns None but its full training still saves (`jobs/training_job.py:328`), training persists an inverse-MAPE ensemble metric over the surviving models only (`:213-236`), while scoring serves an equal-weight blend over ALL loaded models because `jobs/phases.py:890-901` falls back to equal weights when `mape_input` doesn't cover every prediction (`scoring_job.py:208` passes the None through). The displayed "ensemble holdout" then describes a formula and membership the served ensemble doesn't use. [NEW] (F2-001) — #176/#179 handled the all-models crash; this partial-failure divergence is untracked. Fix direction: make training and scoring share one membership/weight rule (skip unweighted models in both, or persist the metric only for the composition scoring will actually serve).

**P2-17 · A single unsmoothed 168h holdout drives weights and quality gates, with documented run-to-run flap** — `jobs/training_job.py:40` (`_HOLDOUT_HOURS = 168`) plus `jobs/scoring_job.py:187-216` and `jobs/phases.py:898-899` means one window with no CV/smoothing feeds `compute_ensemble_weights` and the 22% quality gate. [EXT of #181] — #181 already states: "**Run-to-run variance is real:** AZPS best-base went 11.90% (2026-06-17) → 26.68% (2026-06-19)." Delta: #181 frames this as a *weighting-strategy* question; this finding adds that the quality gate (`models/model_service.py:394`, 22% threshold) sits inside AZPS's observed swing range, so region visibility itself can flap run-to-run, not just weights (S1-05). Fix direction: fold a smoothed/rolling-holdout requirement into #181's methodology so both weights and the gate consume a stabilized metric.

**P2-18 · `ensemble_combine` propagates NaN and its bounds self-check is dead for exactly that case** — `models/ensemble.py:94-108` has no non-finite handling, and the min/max invariant check silently passes on NaN because NaN comparisons are False; `_predict_one` finite-guards ARIMA but not Prophet (`jobs/phases.py:771-785`), so a NaN-bearing Prophet yhat reaches the served payload and Python's `json` emits a non-standard `NaN` token. [NEW] (F1-006) — latent since the known Prophet NaN source was eliminated by PR #179. Empirical (`repro2_models/f1_004_006_007_purepy.py`): `result: [1.5 nan] | 'ensemble_out_of_bounds' warning fired: False | json.dumps ... {"ensemble": NaN}`. Fix direction: add a non-finite guard in `ensemble_combine` (drop or renormalize over finite members, warn loudly), which #184's one-ensemble-path consolidation makes a single-site change.

**P2-19 · The "1-hour-ahead" drift signal doesn't enforce a 1-hour lead and permanently drops catch-up hours** — after an EIA publishing catch-up, `models/drift.py:344-369` records only the most recent matchable hour (an N-hour-lead prediction) into the same rolling series as true 1h-lead records, and the skipped hours' observations can never match any future payload. [NEW] (F1-004) — beyond drift.py's admitted no-backfill scope; distinct from the completed #142/#155 robustness work and from #170. Empirical (same script): `-> single record at H+4 (a 4h-lead prediction), H+1..H+3 never recorded: True ... next tick ... {} (permanently dropped)`. Fix direction: record all matchable hours from the previous payload (they carry known leads) and store a lead-hours field on `DriftRecord` so the 7d statistic can filter or stratify by lead.

**P2-20 · The scenario engine zeroes `temperature_deviation` for any constant temperature override** — `simulation/scenario_engine.py:86` broadcasts the override as a constant, then `:145-148` recomputes deviation as the series minus its own 720h rolling mean (`data/feature_engineering.py:308-322`), which is identically 0 for a constant — so the "unusual weather" signal reads exactly 0 during the most extreme scenarios, and `tests/unit/test_scenario_extended.py:176-184` enshrines it. [NEW] (F9-005); relates to #127 (wiring) but this is internal engine correctness. Empirical (`repro2_platform1`, inline python): `constant-112F override -> temperature_deviation unique values = [0.] ... varying 75F→112F onset -> last-value deviation = 31.71`. Fix direction: compute deviation against the *baseline* history's rolling mean (pre-override) rather than the overridden series' self-mean, and correct the test.

### 5. Misleading numbers on live surfaces

**P2-21 · The Overview headline "live 7d" MAPE can display from a single in-window observation** — `components/_callbacks_overview.py:176-190` gates on `n_records`, the TOTAL merged drift-record count (`models/drift.py:496`), not the 7d-window sample count, and both the 7d and 30d branches test the same total so the documented thin-window fallback can never fire. [NEW] (F5-004) — the completed #142/#155 work covered metric robustness, not window gating. Empirical (`repro2_ui1/repro_ui1.py`): `total records ...: 102 / records inside the 7d window: 2 / _resolve_forecast_mape -> (35.29, 'live 7d') <- 'live 7d' from 2 samples`. Fix direction: have the drift payload expose per-window sample counts and gate the headline (and the 30d fallback) on those.

**P2-22 · "7d Peak / 7d Low / Average" KPIs are computed over the last 168 non-zero rows, not the last 7 days** — `_callbacks_overview.py:240-241` filters zeros then tails 168 rows, so EIA gaps silently stretch the window past 7 calendar days, while the hero chart above uses raw `df.tail(168)` (`:371`) — a max in the stretched-only portion appears in "7d Peak" but not on the chart. [NEW] (F5-012). Fix direction: window both surfaces by timestamp (`>= last_ts - 7d`) and share the frame between KPI and chart.

**P2-23 · The "Net Load (avg)" hero KPI silently falls back to average total generation** — `_callbacks_overview.py:917` seeds `net_load_avg = avg_total` (all fuels, including the wind+solar net load subtracts); the true computation (`:928`) only overwrites it when demand JSON parses and ≥2 timestamps align, exceptions are swallowed (`:930-931`), and the label never changes (`:935-940`) — overstating net load by the renewable share. [NEW] (F5-011). Empirical (`repro2_ui1/repro_ui1.py`): 40%-renewable mix with `demand_json=None` renders `Net Load (avg) 1,000 MW` (the total). Fix direction: on alignment failure render an explicit degraded state (or relabel "Total Generation") instead of substituting a differently-defined quantity.

**P2-24 · The Models-tab leaderboard tones MAPE with raw hardcoded thresholds, violating the H2 governance rule for the very metric shown** — `_callbacks_overview.py:1063-1068` paints ≤2.5 positive / ≤5.0 secondary / else negative, but the values are 168h-holdout MAPEs for which `mape_grade` (config.py:651-671, 7d row) grades 6.0% "excellent" — so a 5.5% model renders red on a live surface while governance calls it excellent (duplicate thresholds at `:2212`). [NEW] (F5-009) — the #189 audit lists `mape_grade` as "genuinely elegant, keep" without noting this live bypass. Fix direction: route the toning through `mape_grade(mape, horizon)` with the correct horizon and delete the inline thresholds.

**P2-25 · The Models-tab drift panel labels a cross-horizon ratio "within ±10% of expected"** — `_callbacks_models.py:310-321` computes `live_7d / holdout` where live_7d is a 1-hour-ahead rolling MAPE (`models/drift.py:12-27`) and holdout is the 168h-horizon training metric (`training_job.py:40`), then labels ratio ≤1.10 "On track" — a systematically lenient calibration claim since 1h-ahead errors are structurally smaller; the headline "All models tracking within ±10 % of holdout baseline" (`:365-369`) is asserted even when every row is Warming/Live-only. [NEW] (F6-009); #181 independently notes the two metrics may rank models differently but doesn't cover this status logic. Fix direction: either compare like-for-like (persist a 1h-ahead holdout figure, or a horizon-matched drift metric) or relabel the panel as an indicative ratio and suppress the headline when rows lack both terms.

**P2-26 · The forecast fast path can chart one model's data under another model's name** — `_callbacks_forecast.py:748` exempts `xgboost` from the payload-availability check and `:753` falls back to `predicted_demand_mw` (the primary = whichever model succeeded first), while the trace and title still say "XGBOOST Forecast" (`:796`); the trigger (xgboost missing from a payload that still has a primary) is a designed-for state in `jobs/phases.py:869-877,918-938`. [NEW] (F6-008) — same badge-vs-data class as the PR #130 c2d6c20 bug but a different, still-live mechanism. Empirical (`repro2_ui2/r2_band_substitution.py`): `title: 24-Hour XGBOOST Demand Forecast — FPL / forecast trace value == prophet/primary value (777.0)? True`. Fix direction: remove the xgboost exemption (return None on model miss like every other model) or relabel the trace by the key actually plotted.

**P2-27 · The backtest prediction-interval "coverage monitor" is self-validating by construction and write-only in production** — in `_run_backtest_for_horizon` (`components/_callbacks_backtest.py:680-732`) the calibration window always equals the entire pooled residual array (n = folds×h ≤ max(5h,120)), so overall coverage is ~80% identically and "recent coverage" is measured on a subset of its own calibration sample; meanwhile `jobs/phases.write_backtests` drops the interval dict (`jobs/phases.py:1148-1166`), so the computation is dead weight for 51 BAs × 3 horizons daily. [NEW] (F6-005). Empirical (`repro2_ui2/r3_interval_tautology.py`): `h=24 ... calib_window=120 (covers whole array: True) overall=0.800 recent=0.800 drift=+0.0pp`. Fix direction: calibrate on a leading window and measure coverage strictly out-of-sample, and either persist the result for a consumer or stop computing it.

**P2-28 · The Backtest renderer's "Recent coverage" statistic is circular and its caption renders "last 0h"** — `_callbacks_backtest.py:228-246` fits quantiles on the same displayed-holdout residuals coverage is then measured on (the recent window is fully contained in the calibration window for h=168/720), so it reads ≈80% for arbitrarily bad forecasts, and the caption at `:364-367` is emitted unconditionally, printing "calibration window: last 0h" when no interval was computed. [NEW] (X2-006) — sibling of P2-27 on the renderer side (this function is currently orphaned; see P2-42). Empirical (`repro2_claims/x2_006_circular_coverage.py`): `garbage-forecast MAPE=38.7% -> recent_coverage ≈ 0.80`. Fix direction: same out-of-sample split as P2-27, and gate the caption on `interval_available`.

**P2-29 · The Risk tab presents invented constants as data** — `_callbacks_alerts.py:191-196` hardcodes event severity scores (Uri=95, CA Heat Wave=80, Heat Dome=85, Eclipse=40) on an unsourced 0-100 "Severity Score" axis rendered for all 51 BAs; fixed 95/100/105°F exceedance lines apply identically to Seattle City Light and AZPS (`:179-186,499-505`); the stress score uses unprovenanced 30/15/+20 weights and 30/60 cutoffs; and a ±2σ band is drawn while only high-side excursions are flagged. [NEW] (F7-010) — distinct from the fabricated-alert-content P0/P1 findings on the same tab. Fix direction: parameterize thresholds per region (from climatology or config), source or drop the severity scores, and document or replace the stress-score weights.

**P2-30 · The grid-ops briefing asserts "nominal" with zero data and computes reserve margin with the wrong denominator** — `data/ai_briefing.py:190` emits "System status for {region} is nominal." before any data check, and `:197-201` computes (capacity−peak)/capacity instead of the industry-standard (capacity−peak)/peak, understating the figure compared to its 15% threshold. [NEW] (F4-009). Empirical (`repro2_data/f4_008_009_briefing_diff.py`): `no-data briefing: 'System status for ERCOT is nominal.' | observations: []` and `@peak=43000: code says -> 'Reserve margin at 14% — below 15% threshold.'; standard = 16.3% (comfortable)` — the two conventions disagree on which side of the threshold the region falls for peak/capacity ≈0.85–0.87. Fix direction: gate the nominal claim on having observations, and use the (capacity−peak)/peak convention (or rename the metric).

### 6. Fallback paths that bypass the honesty gates

**P2-31 · `get_forecasts(models_shown=["ensemble"])` returns actuals relabeled as a forecast in dev and a permanent dead-end in prod** — the loop at `models/model_service.py:481-482` skips every base model because "ensemble" never matches, so `:528` returns `ensemble = actual.copy()` under `source='trained'`; in production (no monolithic pickle exists) the #149 strict gate returns `unavailable` and the Overview Scenarios panel renders "Awaiting baseline forecast" forever — even though the correct Redis reader `_read_ensemble_forecast_from_redis` is defined and used twice in the same module (`components/_callbacks_overview.py:94` vs the disk-touching call at `:1210`). [EXT of #184] — #184 says: "**Flag (do not fix here):** `_predict_from_trained`'s per-model fallback fabricates `actual*(1+noise)` — an integrity concern tracked with #149/#166; note it during consolidation." Delta: #184 flags the per-model noise fallback but not the `ensemble = actual.copy()` branch, its deterministic trigger via `models_shown=["ensemble"]`, or the prod dead-end while real ensemble values sit unread in Redis. Independently confirmed by 3 verifiers (S2-02, F2-005, X1-002). Empirical (`repro_S2/repro_model_service.py`): `source = trained / ensemble == actual verbatim? True / max|residual| = 0.0`. Fix direction: point the Scenarios baseline at the existing Redis reader and make `get_forecasts` treat an all-skipped model list as unavailable rather than echoing actuals.

**P2-32 · `write_diagnostics` converts the honest `unavailable` marker into identically-zero residuals every prod tick** — `jobs/phases.py:1277`'s `diag.get("ensemble", demand)` default turns `{'source':'unavailable'}` into ensemble=actuals, writing zero residuals to `gridpulse:diagnostics:{region}` hourly, which `_callbacks_models.py:87-90` renders as real ensemble diagnostics on Ensemble-only selection. [EXT of #166] — #166 says: "After #149, `get_forecasts` returns `{\"source\": \"unavailable\", ...}` in prod (REQUIRE_REDIS), so `write_diagnostics` now writes **empty metrics + zero residuals** (it has an `or actual` fallback for the ensemble, so no crash)." Delta: the zero residuals are not merely an empty-honest panel — they render as a *perfect* ensemble on Ensemble-only selection, which is fabricated precision (S2-04a). Fix direction: implement #166's derive-from-the-real-forecast approach and, until then, write an explicit unavailable marker instead of zeros.

**P2-33 · The Models tab's default selection has no working diagnostics path at all** — the fast-path gate at `_callbacks_models.py:87` uses an identity check (`selected_models is not default_models`) that a Dash-deserialized default list can never satisfy, so Redis is not even read for the default 4-model view (only exactly `["ensemble"]` passes), and the v1 fallback is strict-gated under REQUIRE_REDIS — residual/histogram/heatmap/SHAP charts are permanently empty by default, with a nonsensical SHAP empty-state (`:545-564,620-629`). [EXT of #166] — #166 says: "That's more honest than fake, but the diagnostics panel now shows flat/empty in prod." Delta: a different mechanism (identity-check gate + strict-gated v1) keeps the *default* view empty even after #166's payload change lands (F6-001). Empirical (`repro2_ui2/r1_models_gate.py`): `A) default 4-model selection -> fast path result: None / redis_get even called? False`. Fix direction: replace the identity check with a set comparison and render per-model diagnostics from the Redis payload for any selection it covers.

**P2-34 · Demo data is mislabeled as "stale" cached data, and the audit trail records it as `api`** — on EIA/Open-Meteo failure or empty result, `components/callbacks.py:302-317` (weather twins `:322-335`) serves `generate_demo_*` output but sets `freshness='stale'`, so the banner reads "serving cached data (API unavailable)" (`:564-567`), the confidence badge shows Medium instead of the demo tier, and the D2 audit record claims `forecast_source='api'` (`:378`) — synthetic curves wearing a real-data label, contrary to the G2 vocabulary and the CLAUDE.md no-fake-data rule. [NEW]; independently confirmed by 2 verifiers (F7-011, X3-012). Empirical (`repro2_ui5/x3_012_demo_labeled_stale.py`): `freshness: {demand: stale, weather: stale}; demand rows returned: 2160 (synthetic demo data ...)`. Fix direction: set `freshness='demo'` on every demo-serving branch so the existing banner/badge/audit plumbing tells the truth, and align with #185's one-fallback-contract work.

**P2-35 · Permanent failure states render as "warming up — forecast will appear shortly" forever** — any Redis model miss or REQUIRE_REDIS cache miss funnels through `_callbacks_forecast.py:748-750 → 350-365 → 1214-1219` to the same transient copy with no age check or escalation anywhere on the path, so a model that never trains for a region claims "shortly" indefinitely. [EXT of #188] — #188's acceptance item is: "Warming/degraded output goes through `warming_state()`". Delta: routing through the helper is cosmetic consolidation; the missing transient-vs-persistent *semantics* (payload/`last_scored` age check, escalation to an error state) are not in #188's scope and are the substance here (F6-012). Fix direction: thread `last_scored`/payload age into the warming decision and escalate to a distinct "unavailable" state past a threshold.

### 7. Dead machinery, duplication & latent defects in dormant surfaces

**P2-36 · The scoring job computes and writes generation + weather-correlation payloads hourly for 51 BAs that nothing reads** — `jobs/scoring_job.py:156,253` run `write_generation`/`write_weather_correlation` (full correlation matrix + seasonal decomposition per region, `jobs/phases.py:1189-1261,286`) every tick, but the only readers (`components/_callbacks_generation.py:55`, `_callbacks_weather.py:58`) are orphaned fast paths never registered as callbacks (`components/callbacks.py:64,121` self-annotate "fast path is currently orphaned"). Pure wasted compute inside the runtime-constrained ~855s job (relates to #171), kept green by tests, with docstrings still claiming the web tier reads them. [NEW]; reported three times (X3-005 ×2, F7-003). Fix direction: decide re-wire vs delete per the in-code TODO — if the tabs stay removed, drop the two phases (buying #171 headroom) and the orphaned readers; if not, register the callbacks.

**P2-37 · The real NOAA alerts pipeline is fully built, tested, and dead — and its failure path would cache blanks over real alerts** — repo-wide grep shows the only non-test importer of `data/noaa_client.py` is the `data/__init__.py:12` re-export; no job phase or callback calls `fetch_alerts_for_region`/`fetch_all_alerts` (the fabricated alert path ships instead), and `STATE_TO_BA`/`BA_FOR_STATE` (config.py:494-557) are maintained for nothing at runtime. Latent within it: `_fetch_state_alerts` returns `[]` on request errors (`noaa_client.py:139-141`) and the caller unconditionally `cache.set`s the result (`:106-110`), so a transient NWS failure would replace the last real alert list with an empty one the moment anyone wires it in; the "not all 51 BAs mapped" docstring (`:117-119`) is also stale. [NEW]; reported three times (F3-002, X3-006 ×2) — this is the natural replacement path for the fabricated-alerts P0/P1 cluster. Fix direction: wire the client into the scoring job's alert phase (writing to the existing Redis alerts key), and make the failure path serve stale cache instead of caching blanks.

**P2-38 · A ~600-850-line Overview briefing surface is dead code pinned alive by tests** — the R2 layout removed every target ID (`components/tab_overview.py:25-62` defines exactly 5 dynamic IDs; `test_layout_no_legacy_ids` pins the removal), yet `_build_overview_spotlight/_digest/_briefing/_news/_build_persona_kpis` and the `_spotlight_*` trio (`components/_callbacks_overview.py:1378-2330`) remain re-exported (`callbacks.py:74-91`) and heavily unit-tested, preserving drifted pricing/metrics/renewable logic and rewire hazards (sync Anthropic call, unlabeled demo news, wrong Wind CF). [EXT of #154] — #154's scope reads: "Sub-step 3: Move Overview briefing surface (sparklines / briefing / digest / spotlights / weather / data-health / changes / news / persona KPIs) into a separate module". Delta: the surface is *dead*, not live product code — the right action is delete or quarantine (with the hazards fixed first if resurrection is intended), not move. Independently confirmed by 2 verifiers (X1-006, F5-007). Fix direction: make the product call (GP-P1-04 resurrection vs deletion), then delete the builders + their change-detector tests or quarantine behind an explicit flag.

**P2-39 · The dead spotlight fabricates MAPE bars from the model name's length, and the digest scans a cache with the wrong key shape** — `_spotlight_model_accuracy` probes `"mape" in result_dict` but both `_BACKTEST_CACHE` writers nest it under `"metrics"` (`_callbacks_backtest.py:690-748`), so the fallback `4.5 + len(model_name)*0.3` always renders even with a warm real cache — the chart can never show a real number — while `_build_overview_digest` scans `_PREDICTION_CACHE` with 2-tuple keys against 3-tuple writers (`_callbacks_overview.py:1904-1911, 1977-1994`); `_build_persona_kpis` reads the same cache with the correct shape (`:2149-2159`). [NEW]; independently confirmed by 2 verifiers (F5-008, X1-005) — the #131 fabricated-metrics class at untracked sites. Empirical (`repro2_ui1/repro_ui1.py`): `bars rendered: {'Prophet': 6.6, 'Arima': 6.0, 'Xgboost': 6.6} ... -> cached real 2.0% ignored`. Fix direction: delete with P2-38; if any part is resurrected, read `result["metrics"]["mape"]` and the 3-tuple keys, and never render an invented fallback number — #153's typed contracts would prevent recurrence.

**P2-40 · `_backtest_tab_from_redis` is a dead surface preserving live-looking integrity defects, including a fabricated placeholder statistic** — the Backtest tab was removed from the 5-tab shell and grep finds no callback registration (only tests and the `callbacks.py:45` re-export), yet the function (`components/_callbacks_backtest.py:183-370`) still: serves XGBoost data labeled "ENSEMBLE Forecast" (`:200`), seeds `interval_monitor` with `{'recent_coverage': 0.0, 'drift': -0.8}` at `:229` which the exception path (`:247-248`) leaves intact so the explanation renders a precise-looking "Recent coverage: 0.0% (drift vs 80% target: -80.0 pp)" (`:338-339,364-368`), charts predictions=actual as a perfect forecast, and formats missing metrics as a flawless 0.00%. [NEW]; independently confirmed by 2 verifiers (F6-004, X3-014). Fix direction: delete or quarantine the renderer with its orphaned tests; any resurrection must gate the coverage sentence on `interval_available` and remove the placeholder/perfect-forecast fallbacks.

**P2-41 · Three of four insight generators are unreachable, and the Models tab ships a slot no callback ever fills** — caller census shows `generate_tab1/tab3/tab4_insights` (`components/insights.py:313-1035`, ~600 LOC) are reachable only from the dead digest, the dead backtest renderer, and the orphaned generation module respectively; only `generate_tab2_insights` is live. `tab-models` renders a `tab3-insight-card` slot that nothing populates, contradicting the `tab_models.py` docstring. [NEW] (F8-001) — extends the dead-code census beyond what #154/#188 enumerate. Fix direction: delete the dead generators (or rewire tab3's into the live Models callback and fill the slot), and correct the docstring.

**P2-42 · Keyboard-shortcut and ARIA machinery is dead in Python and wrong in JS** — `components/accessibility.py:150-169` carries an 8-tab `TAB_KEY_MAP` and "Alt+1 through Alt+8" claims with zero importers beyond the palette constants, while `assets/accessibility.js` declares a 4-tab map it never uses, clicking positionally (`visibleLinks[parseInt(e.key)-1].click()`) — so in the 5-tab shell Alt+2/3/4 activate different tabs than the map declares, the Models tab has no shortcut, and chart ARIA labeling targets a class only dead code emits. [NEW] (F8-002). Fix direction: drive the JS from a single 5-tab map keyed by tab id (not position), and delete or rewire the dead Python helpers.

**P2-43 · The Forecast-Replay subsystem (NEXD-14) is permanently inert with a circular re-enable condition** — `config.py:733` hard-sets `forecast_replay: False` with no env override; both callbacks and the snapshot-save hook are gated on it (`components/_callbacks_forecast.py:643-660,1393-1506`), the panel is `display:none`, and no job produces snapshots — so the flag's "re-enable once the snapshot pipeline is producing fresh data" comment can never be satisfied because the only producer is behind the flag. [NEW] (F6-007). Fix direction: either move snapshot production into the scoring job (breaking the circularity) and then flip the flag, or delete the subsystem (`data/forecast_history.py` + both callbacks).

**P2-44 · The Risk tab is implemented twice with visual/data drift, and the stress formula exists in three places** — the Redis fast path still renders legacy emoji stress rows and a lone temperature KPI (`components/_callbacks_alerts.py:99-127, 225-246`) while only the dev fallback received the v2 `gp-stress-row` markup and `_build_weather_context` (`:388-434, 545`) — production shows the *older* design; the stress formula `min(100, n_crit*30 + n_warn*15 + 20)` is duplicated at `:382-383` and `jobs/phases.py:1351-1359`, and the 4-event timeline and temp chart are copy-pasted between paths (`:191-196` vs `:510-515`). [NEW]; independently confirmed by 2 verifiers (X1-001, F7-009). Fix direction: extract one render function both paths call (Redis payload vs computed inputs), and make the job-computed stress score the single source the UI reads.

**P2-45 · Two more ensemble implementations live in components/, with genuine policy drift and a false parity docstring** — `_ensemble_fold` (`components/_callbacks_backtest.py:436-459`) uses equal weights while its docstring asserts production parity, and `_run_forecast_outlook` (`_callbacks_forecast.py:458-468,594-597`) uses equal weights and excludes ARIMA beyond 168h — whereas production weights by inverse holdout MAPE and includes ARIMA for the full 720h (`jobs/phases.py:887-912`). [EXT of #184] — #184 says: "**Inverse-MAPE ensemble in 3 places.**" Delta: two additional, unlisted copies in the components layer, with membership/weighting drift from production and a docstring that misinforms both #184 and #181 (X1-004). Fix direction: fold these two call sites into #184's one-ensemble-path consolidation and delete the false docstring claim.

**P2-46 · The "claude" briefing source is permanently unreachable behind an invalid model id** — `data/ai_briefing.py:117` hardcodes `claude-haiku-4-20250414`, an id that does not exist in the Anthropic catalog (no claude-haiku-4 generation was ever released), so every call 404s, is caught at `:76-83`, and silently degrades to the rule-based briefing after a wasted network round-trip per cache miss whenever `ANTHROPIC_API_KEY` is set. [NEW] (F4-006). Fix direction: pin a real current Haiku id (e.g. `claude-haiku-4-5-20251001`), add a startup log when the API path is active, and cover the model id with a test against a stubbed client.

**P2-47 · Merit-order pricing drops 28.6% as demand rises through 90% of capacity** — `models/pricing.py:47-59`: the linear tier ends at 1.4× base just below u=0.90 and the exponential tier restarts at 1.0× base, a −40 $/MWh discontinuity that directly contradicts the module docstring's "demand > 90% capacity: exponential spike"; price stays below the u=0.8999 level until u≈0.9224. The surface consuming it is currently hidden (Scenarios), making this a latent bug that goes live with the planned P1 Scenarios polish. [NEW] (S1-04). Empirical (`repro_S1/s1_04_pricing_discontinuity.py`): `u=0.8999 -> 139.9800; u=0.9000 -> 100.0000 <-- PRICE DROPS AS DEMAND RISES`. Fix direction: anchor the exponential tier to the linear tier's endpoint (continuity at 1.4× base) and add a monotonicity property test across tier boundaries.

### 8. Configuration, deployment & endpoint hardening

**P2-48 · An unrecognized `ENVIRONMENT` value fails open to full development defaults, silently** — `config.py:76` does `_ENV_DEFAULTS.get(ENVIRONMENT, _ENV_DEFAULTS["development"])` with no validation or warning, so a typo like `ENVIRONMENT=prod` (or `Production`, or a trailing space) sets `require_redis=False, demo=True, precompute_enabled=True, gcs_enabled=False` — removing every honesty gate at once — while `feature_enabled` (`:757-768`) warns on unknown flags, showing the fail-loud pattern already exists in the file. [NEW] (X3-011). Empirical (`repro2_platform2/x3_011_env_fail_open.py`): `ENVIRONMENT='prod' → {REQUIRE_REDIS: false, USE_DEMO_DATA: true, PRECOMPUTE_ENABLED: true, GCS_ENABLED: false} ... warnings about unknown ENVIRONMENT: NONE`. Fix direction: validate against the known tier set at import, failing hard (or at minimum warning loudly and defaulting to the *strictest* tier, not development).

**P2-49 · The Docker image bakes a dev-tier `PRECOMPUTE_ENABLED=true` that the four job deploys inherit** — `Dockerfile:43` sets the env var, which beats the staging/production matrix defaults (`config.py:61,72,700-702`); the two web-service deploy blocks remember a per-service override to false (`deploy-prod.yml:88`, `deploy-dev.yml:75`) but the four Cloud Run Job blocks do not (`deploy-prod.yml:113,148`, `deploy-dev.yml:90,106`), so any deploy surface that forgets the override runs the inline scoring pipeline inside the web container (`app.py:143-145`); `Dockerfile:44`'s `PRECOMPUTE_ALL_REGIONS` is read by nothing. [EXT of #187] — #187 says: "**Cloud Run flags duplicated across ~4 `gcloud run jobs deploy` blocks** (`deploy-prod.yml:107-156`, `deploy-dev.yml:84-112`) kept in sync by a literal `# keep in sync with production` comment; the stale `wattcast-connector` (old project name) lingers in every block." Delta: the drift #187 warns about already has a live instance — a baked wrong-tier image ENV that only some deploy surfaces remember to override (F9-009). Fix direction: remove tier-specific ENV values from the image entirely and let the J1 matrix (plus explicit deploy overrides) own them, per #187's single-owner principle.

**P2-50 · `LOG_LEVEL` is a dead knob with contradictory values on four surfaces, and production emits DEBUG logs** — `configure_logging()` applies no level filtering, and no code reads `LOG_LEVEL` (grep: only `config.py:18-26` and one test); meanwhile the J1 matrix comment documents production=WARNING, the default is INFO, and `deploy-dev.yml:75,90,106` sets DEBUG under `ENVIRONMENT=staging`. [EXT of #187] — #187 says: "The header ASCII table (~`:23-34`) lists `LOG_LEVEL`/`MAX_INSTANCES` rows that aren't in the dict; `cache_ttl` is identical across all tiers; staging ≈ production." Delta: not just doc-vs-dict drift — the knob has *no consumer at all*, so every tier emits DEBUG regardless of any setting (F9-003). Empirical (`repro2_platform1`, inline python, `LOG_LEVEL=WARNING`): `{"note": "LOG_LEVEL=WARNING was set", "event": "debug_leaks_in_prod", "level": "debug", ...}`. Fix direction: wire a structlog level filter driven by `LOG_LEVEL` in `configure_logging`, then reconcile the four surfaces to one owner under #187.

**P2-51 · `USE_DEMO_DATA` is documented and tested but read by nothing** — grep at HEAD shows the only readers are its own definition (`config.py:78`), the matrix header row (`:30`), and `tests/unit/test_sprint4_features.py:29-96` (which assert only env→flag plumbing); actual demo serving is controlled by `EIA_API_KEY` presence and `REQUIRE_REDIS` (`components/callbacks.py:296-333`), so setting `USE_DEMO_DATA=false` does not prevent demo data. [NEW] (X3-010) — #187's matrix trim wouldn't catch it since the row genuinely differs per tier. Fix direction: either make the demo branches consult the flag (giving the J1 matrix real control) or delete the flag, its matrix row, and its change-detector tests.

**P2-52 · The `/metrics` IP allowlist is bypassable with a client-supplied `X-Forwarded-For: 127.0.0.1`** — `app.py:203` trusts the *leftmost* XFF entry, but Cloud Run's edge appends the real client IP to the *end*, so the leftmost element is attacker-controlled; `:206` then admits the spoofed 127.0.0.1 while rejecting honest external traffic. [NEW] (F9-004); exposure is limited to internal timing stats. Empirical (`repro2_platform1`, inline python, faithful copy of `:199-208`): `attacker sets XFF=127.0.0.1, edge appends real IP 203.0.113.7 -> 200` / `honest external client XFF=203.0.113.7 -> 403`. Fix direction: take the rightmost untrusted-hop entry (or Cloud Run's dedicated client-IP handling), or replace IP allowlisting with an auth header for this internal endpoint.

---

**Count: 52 canonical P2 findings (43 [NEW], 9 [EXT]) from 70 CONFIRMED verdicts after merging 18 duplicate reports across 11 clusters.**

---

## Unsupported-claims ledger (axis 7 — first-class section)

All entries verified at SHA `5000d6a`. Individually P3; the section exists because the *pattern* is the finding. Duplicate detections are merged; corroborating finding_ids listed together.

| # | Claim | Where (file:line) | Reality / grounding found | Verdict | finding_id(s) |
|---|---|---|---|---|---|
| 1 | "3.13% MAPE on ERCOT 21-day holdout (Feb 2026)" headline | `README.md:36` | The figure was measured under the pre-#135 leakage regime that the project's own `docs/BACKTEST_RESULTS.md:13-24` explicitly invalidated; it survived the #177 stale-claims sweep. | STALE | X2-001, S5-01 (2 verifiers) [NEW] |
| 2 | "43 engineered features" / "~43 total" / "43 (17 raw + 26 derived)" | `README.md:36,119`; `PRD.md:125`; `docs/HOW_IT_WORKS.md:192` | Mechanical count of `get_feature_names()` this review: `total=49 raw_weather=17 derived=32` (repro: `scratchpad/repro_S5/s5_01_feature_count.py`). HOW_IT_WORKS:192 sits inside a "verbatim recall" section that contradicts both `CANONICAL_FACTS.md` (49) and the same doc's §4 diagram. | STALE | X2-002, S5-01 (2 verifiers) [EXT #188] |
| 3 | "Combining models almost always beats individual models"; "can never be worse than the worst individual model"; ADR-004 "self-correcting" | `models/ensemble.py:6`; `docs/HOW_IT_WORKS.md:142`; `CLAUDE.md:215`; `docs/CANONICAL_FACTS.md:111` | Falsified by the project's own current measurement: the served ensemble beats XGBoost-alone on 4 of 51 BAs; median 3.48% vs 2.32% (`CANONICAL_FACTS.md:75-90`, `BACKTEST_RESULTS.md:51-53`). | CONTRADICTED | X2-003, S1-08 (2 verifiers) [EXT #181] |
| 4 | "tuned via autoresearch: 30 experiments, 16.4% MAPE improvement" | `models/xgboost_model.py:23` | No measurement artifact anywhere in the repo (the git-tracked `wattcast-autoresearch.tar.gz` contains no experiment journal); the sole provenance is a commit message, and the figure predates the #135 leakage repair. | UNGROUNDED | X2-004 [NEW] |
| 5 | Coverage: "~100% of contiguous-US lower-48 load" vs "~80% of demand" vs "~99%" | `docs/CANONICAL_FACTS.md:17` vs `components/_callbacks_us_grid.py:700` vs `docs/internal/NEXT_UP.md:184` | Three mutually contradicting figures for the flagship coverage claim; no derivation artifact exists for any of them. | CONTRADICTED | X2-005 [NEW] |
| 6 | Legend "80% indicative range" (+ `interval_meta target_coverage: 0.80`) on the heuristic forecast band | `components/_callbacks_forecast.py:221, 236-241` | The band is an uncalibrated ±3/6/10% envelope whose own docstring (`:92-100`) says it "should not be interpreted as probabilistic coverage guarantees"; the Overview labels the identical fallback "±3% indicative range" (`_callbacks_overview.py:447-449`). Independently confirmed by 3 verifiers. | UNGROUNDED | X2-007, F6-002, S2-01b [NEW] |
| 7 | `upper_80`/`lower_80` payload keys on the model-service band | `models/model_service.py:533-534, 571-572` | The keys imply 80% coverage over a fixed ±3% multiplier; empirical repro: `upper_80 == ensemble*1.03? True` (`scratchpad/repro_S2/repro_model_service.py`). PR #169's fabricated-interval purge touched only `prophet_model.py` and never reached this band. | UNGROUNDED | S2-01a [NEW] |
| 8 | Training job retrains "on the last 60 days" | `README.md:79` | Traced path (`training_job.py:462` → `phases.py:131` → `eia_client.py:129`) defaults to ~90 days, matching HOW_IT_WORKS:94, TECHNICAL_SPEC, and published holdout `train_rows`. | STALE | X2-008 [NEW] |
| 9 | Test count "1,589 passing" / "1681 tests" | `docs/CANONICAL_FACTS.md:56`; `README.md:134` | `pytest --collect-only -q` at HEAD: `1981 tests collected in 1.64s`. The registry and its supposed consumer disagree with each other and with the tree. | STALE | X2-009 [NEW] |
| 10 | Must-Have R3.1: "Prophet with weather regressors and multiplicative seasonality" | `PRD.md:131` | Implementation is deliberately additive everywhere: `prophet_model.py:63` `seasonality_mode="additive"`, and `:66-70` forces additive per-regressor with a written rationale. | CONTRADICTED | X2-010 [EXT #184] |
| 11 | Scoring "~5 min", "5-hour timeouts", "4 horizons × 4 models" | `docs/HOW_IT_WORKS.md:52, 161` | Contradicted by the same doc (`:104` "~14 minutes"), `CANONICAL_FACTS.md:39` (855s runtime, 1800s timeout), STATUS.md's measured 1083–1333s, and the UI's 3 horizons. | STALE | X2-011 [NEW] |
| 12 | "Confidence interval \| 80% empirical, last 120h calibration window \| models/evaluation.py" | `docs/CANONICAL_FACTS.md:31` | Actual window is `min(available, max(horizon*5, 120))` — 840h at the default 7d horizon (120h is only the floor) — and the logic lives in `components/_callbacks_shared.py:398`, not `models/evaluation.py`. | CONTRADICTED | X2-012 [NEW] |
| 13 | "Active top-level tabs" ×9 + Module Map listing `tab_forecast/tab_backtest/tab_generation/tab_weather/tab_simulator.py` | `CLAUDE.md:200-209, 234-241` | The shipped shell is 5 tabs; the five listed modules were deleted in PR #63 and do not exist at HEAD. Agent-facing, so the highest-leverage staleness instance. | STALE | X2-013 [EXT #188] |
| 14 | TECHNICAL_SPEC §9/§12 present the same five deleted modules as current surfaces; §4.2 claims an `is_weekend` engineered feature | `TECHNICAL_SPEC.md:396-409, 461-469, 207` | None of the five files exist at HEAD; the engineered feature set carries `is_holiday`, not `is_weekend`. | STALE | X2-014 [NEW] |
| 15 | "every healthy 16-BA primary model is below 5% MAPE, so the gate only fires when a tiny BA … produces nonsense" | `config.py:736-739` | The fleet is 51 BAs; seven exceed 5% XGBoost holdout MAPE in the current published table; the BA the gate actually hides (AZPS, 33.97%) is a major Arizona utility. The comment justifies live gating behavior. | STALE | X2-015 [EXT #188] |
| 16 | Governance/calibration constants: `MAPE_BY_HORIZON` cutoffs, ARIMA drift threshold `std(y)*0.5`, Prophet cap 1.5× the 90-day max | `config.py:651-656`; `models/arima_model.py:140`; `models/prophet_model.py:106` | Cutoffs appear verbatim in the initial commit with no derivation; no `.md` file even mentions `MAPE_BY_HORIZON`; git messages cite outcomes for adjacent changes but never for these constants. Worse, `TECHNICAL_SPEC.md:317-326` publishes a *competing* flat grading scale (<3/3–5/5–10/>10, no rollback tier) for the same decision. | UNGROUNDED | X2-016, S1-06 (2 verifiers) [NEW] |
| 17 | Prophet logistic-growth cap silently defaults to 50,000 MW | `models/prophet_model.py:164` | The constant matches nothing in config, is below the load scale of PJM (184,202 MW), MISO (186,986), and ERCOT (153,000) per `config.REGION_COORDINATES`-adjacent capacity table, and engages with no log line. | UNGROUNDED | F1-009 [NEW] |
| 18 | Pricing model constants (base $50/MWh, tiers 0.70/0.90/1.00, emergency ×20) presented as decision support | `config.py:621-628`; `models/pricing.py:44-59` | Zero citations (in pointed contrast to the per-BA-cited `REGION_CAPACITY_MW` block) and the tier math produces a ~4.5× price discontinuity at the 100% boundary (repro: `scratchpad/repro2_platform2/f9_014_pricing_boundary.py`); no "illustrative" label anywhere. | UNGROUNDED | F9-014 [NEW] |
| 19 | Presets define "the weather conditions during a real historical event" | `simulation/presets.py:5-6, 113-114` | Values are uncited hand-picks applied as flat constants across the whole horizon; the 2024 eclipse preset encodes the wrong physical mechanism (`cloud_cover=100` rather than irradiance attenuation). | UNGROUNDED | F9-015 [NEW] |
| 20 | "models exceeding ROLLBACK threshold are auto-disabled … model disabled, fallback to next-best" | `config.py:643, 648` | No disable mechanism exists anywhere in the repo; the only rollback-grade enforcement hides regions from the UI while rollback-grade models keep contributing to the served ensemble; the four `MAPE_THRESHOLD_*` constants are consumed exclusively by tests. | CONTRADICTED | F9-010 [NEW] |
| 21 | get_model_metrics docstring + CLAUDE.md watchlist: layers 1–3 "require meta.json on local disk" and "all fail in production" | `models/model_service.py:97-102` (impl `:161-205`); CLAUDE.md watchlist | Layers 1–3 call `persistence.get_model_metadata`, a **GCS network read** that works on the web tier — so a layer-0 Redis miss performs undeclared per-render GCS I/O rather than the documented warming state. | CONTRADICTED | F2-007 [EXT #183] |
| 22 | Training-job comment: the gate/weights use "the latest XGBoost training-holdout MAPE" on a "consistent metric basis" with prophet/arima | `jobs/training_job.py:265-277` | `saved_mape = (holdout_metrics or {}).get("mape") or cv_mape` silently substitutes the CV-mean MAPE whenever the XGBoost holdout fails, then feeds it into inverse-MAPE weighting alongside true holdout MAPEs — a mislabeled statistical quantity. | CONTRADICTED | F2-008 [NEW] |
| 23 | Published ensemble backtest/holdout numbers | `scripts/backtest.py:168-200`; `jobs/training_job.py:230-238` | Both computations fit ensemble weights on the same window they score (in-sample), while production applies previous-window weights to future data — a directionally favorable bias in every published ensemble figure. | UNGROUNDED | F2-011 [EXT #181] |
| 24 | Ship/no-ship criterion: "if MAPE(169-384) / MAPE(1-168) < 1.5, ship PR-E with the 384h cap" | `scripts/audit/extended_holdout_check.py:20-22` vs `:129-145` | The code divides by MAPE(72–168h) — typically the worst short window — and never computes MAPE(1–168), systematically shrinking the ratio and biasing toward "ship". | CONTRADICTED | F2-014 [NEW] |
| 25 | `_ensemble_fold` docstring: "Forward forecasts already use equal weights … keeps backtest and production behaviour consistent" | `components/_callbacks_backtest.py:440-442` | Production computes inverse-MAPE weights at scoring time (`jobs/phases.py:898-912`), so the "ensemble" backtest evaluates a different combiner than production serves. | CONTRADICTED | F6-006 [EXT #184] |
| 26 | Models tab presents its metrics table and diagnostics charts as one evaluation | `components/_callbacks_models.py:99-141` vs `:145-223` | The table comes from `get_model_metrics` (168h training holdout); the charts directly below come from `gridpulse:diagnostics:{region}` (a different evaluation, empty in prod per #166) — with no badge, caption, or cue distinguishing the two provenances. | UNGROUNDED | F6-013 [EXT #166] |
| 27 | Documented gap policy: "interpolate gaps <6h, flag ≥6h via data_quality" | CLAUDE.md module map; `data/preprocessing.py:73-137` | `handle_missing_values`/`validate_dataframe` have zero production callers; the live pipeline is `engineer_features(merged).dropna(subset=["demand_mw"])` (`jobs/phases.py:177-196`); `data_quality` is write-only where produced. The policy is documented but nowhere enforced. | CONTRADICTED | F4-004 [NEW] |
| 28 | D2 "Forecast Model Input Audit Trail" listed as implemented, purposed for post-event lineage / FERC-NERC defensibility | CLAUDE.md Sprint-5 list; `data/audit.py:9, 65-164` | It is a per-process, memory-only 1000-record ring buffer with no persistence; the query API has zero production consumers; the scoring job — the thing that actually produces forecasts — writes no audit records at all. | CONTRADICTED | S3-04 [NEW] |

**Pattern.** Three claim classes recur. First, **era drift**: counts and figures published once and never re-synced across the project's regime changes (16→51 BAs, 9→5 tabs, pre/post-leakage-repair, 60→90-day windows) — and the registry built to prevent exactly this, `CANONICAL_FACTS.md`, itself carries two wrong rows (#9, #12). Second, **ungrounded calibration**: essentially every user-facing grading, pricing, cap, and threshold constant traces to the initial commit or a hand-pick with no measurement artifact, in sharp contrast to the meticulously cited `REGION_CAPACITY_MW` block, which proves the team knows how to do provenance when it tries. Third, **statistical inflation**: quantities labeled stronger than their construction (heuristic bands wearing "80%", CV MAPE wearing "holdout", in-sample weights inside published ensemble numbers). Fix direction is uniform: correct or delete each claim at its source in one sweep, route every recurring number through `CANONICAL_FACTS.md` with a freshness date, and require a one-line provenance comment (source, date, measurement) for any constant that gates or grades user-visible output.

## P3 — Polish & doc-only

Remaining non-claims P3 findings, merged and grouped. All locations at SHA `5000d6a`.

### Models & metrics layer

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F1-007 | `models/pricing.py:45-59` | NaN utilization or 0/0 capacity falls through every `np.where` condition to the $1000/MWh emergency price — no isfinite guard | [NEW] |
| S1-01 | `models/ensemble.py:32, 41-43, 87-92` | Exact-0 MAPE silently drops the *best* model from weighting; a denormal MAPE (1e-320) passes `v>0` but overflows `1/v` → NaN weights → all-NaN ensemble | [NEW] |
| S1-02 | `models/evaluation.py:19-42` (+`xgboost_model.py:193-198`, `drift.py:91-141`) | MAPE implemented 3× with divergent zero/near-zero handling — same data, different numbers | [EXT #184] |
| S1-03 | `models/evaluation.py:37-42` | `compute_mape` silently excludes near-zero-actual rows with no exclusion count logged or returned | [NEW] |
| F2-009, S2-04b(ii) | `models/model_service.py:268-275` | `get_ensemble_weights` is production-dead API returning fabricated hardcoded weights {0.30/0.20/0.50}, ungated by #149; its only callers are unit tests that cement the values | [NEW] |
| S2-04b(i) | `models/model_service.py:506-512` | `_predict_from_trained`'s per-model exception fallback fabricates `actual*(1+noise)` under `source="trained"`, outside the REQUIRE_REDIS gate | [EXT #184] |
| F1-008 | `models/training.py:34-58` | `train_all_models` is production-dead yet its docstring claims scheduled use; `len(df)==validation_hours` yields an empty training frame with only a warning; re-engineering the 168-row validation slice skews `temperature_deviation` | [EXT #184/#188] |

### Jobs / scoring & training pipeline

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F2-012 | `jobs/training_job.py:478-492` | Data-hash resume path returns before `write_backtests`, so a resumed region's Redis backtest keys (TTL 24h) can expire while the job reports `ok=True/resumed` | [NEW] |
| F4-011 | `jobs/phases.py:278-284` | `renewable_pct` division guarded only on the aggregate mean; an individual all-zero fuel hour writes NaN into the Redis list | [NEW] |
| S4-02 | `jobs/phases.py:532-568` | `_resolve_forecast_start` docstring promises a 3-level fallback chain; level 3 does not exist in the implementation | [NEW] |
| X1-015 | `jobs/phases.py:1291-1303` vs `components/_callbacks_models.py:425-438` | Two *disagreeing* fabricated XGBoost feature-importance lists feed the same chart (different feature names and weight shapes) | [EXT #166] |

### Data layer

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F3-007 | `data/eia_client.py:214-219, 246-248, 281-314` | Stale-SQLite fallback tier is structurally dead: hour-embedded cache keys can never match a prior fetch, and job containers start with a fresh cache | [EXT #185] |
| F3-008 | `data/eia_client.py:161, 528-542` | EIA day-ahead (DF) facet fetched on every demand call (~2× records), its forward rows dropped by the left-merge, and `forecast_mw` stripped before Redis — all downstream consumers inert | [NEW] |
| S4-01 | `data/preprocessing.py:9-10, 81-135` | Exactly-6h gap boundary contradicts the docstring/spec (>6h vs ≥6h); a 6-NaN weather run is fully interpolated while a 6-NaN demand run is flagged; the promised `data_gap` column is never created | [NEW] |
| F4-008 | `data/session_diff.py:63-65, 145-208` | `alert_count`/`renewable_pct` are never populated, so the alert-diff and renewable-share "What Changed" branches can never fire | [NEW] |
| F4-010 | `data/forecast_history.py` (+`config.py:733`) | Time-Scrub Replay (422 LOC) is dark behind `forecast_replay=False` with a circular re-enable condition — the snapshot pipeline that must "produce fresh data" *is* the flag-gated call | [NEW] |
| S3-05 | `data/weather_client.py:66-68, 90-92, 298-311` | Open-Meteo client makes a single un-retried GET per endpoint vs EIA's 5-retry + circuit breaker — asymmetric failure policy across clients | [EXT #185] |
| X3-009-revised | `components/_callbacks_overview.py:2068-2078` (+`data/news_client.py:61,103,111`) | Dormant second demo-news layer above the client fallback fabricates perpetually fresh timestamps attributed to EIA/DOE with no demo marker — would silently defeat the #185 typed-empty change if news is re-wired | [EXT #185] |

### Config / infra / CI / observability

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F3-012 | `config.py:78` (+`components/callbacks.py:296`) | `USE_DEMO_DATA` is a declared-then-ignored flag (real gating is an API-key string compare + REQUIRE_REDIS); the header "safety property" table row is inert | [EXT #187] |
| F9-002, S5-04a | `config.py:712-741` | 12 of 18 `FEATURE_FLAGS` gate nothing in production — including `ai_briefing`, a decorative kill switch on a paid Anthropic API call | [EXT #188] |
| F9-007 | `config.py:630-639` | `STALENESS_THRESHOLDS_SECONDS` is write-only (the freshness UI hardcodes its own 7200s default) and uncalibratable ('generation' 300s vs hourly EIA-930; 'pricing' has no live feed) | [EXT #187] |
| F9-008 | `config.py:680-684` vs `data/eia_client.py:35-36` | D3 rate-limit constants shadowed by the client's own divergent values (5 vs 4 retries; 2.0 vs 1.0s backoff); `RATE_LIMIT_ALERT_THRESHOLD` wired to nothing | [NEW] |
| S5-05 | `config.py:41-42, 82-87`; `Dockerfile:63` | `_ENV_DEFAULTS` `profile`/`workers` entries dead: `ENABLE_PROFILING` gates nothing; `GUNICORN_WORKERS` ignored because the Dockerfile hardcodes `--workers 2` | [EXT #187] |
| F9-012 | `.github/workflows/ci.yml:62-63` | "v2 Pipeline Tests" step is permanently green (`scaling-analytics/` doesn't exist) and its `2>/dev/null \|\| echo` construction would mask real failures if the directory returned | [NEW] |
| F9-013 | `observability.py:135-258` | `PipelineLogger` (I1) instruments none of the real ETL — jobs never import it; the web tier logs Redis-read pseudo-steps — and `TAB_LOAD_P95_SECONDS`, `MODEL_REFRESH_INTERVAL`, `PRECOMPUTE_*` are dead config | [NEW] |
| F9-011 | `app.py:110, 117` | `og:image`/`twitter:image` use a relative URL; the OG protocol requires absolute, so the generated 1200×630 card never appears in link previews | [NEW] |
| F2-006 | `scripts/audit/verify_overview_metrics.py:110-114, 253-256, 274-286` | Audit script crashes (TypeError) formatting `trend_24h_pct=None` (≥90-min EIA gap at the 24h anchor); docstring/comment describe the retired methodology | [NEW] |

### UI / components

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F6-010 | `components/_callbacks_shared.py:41-44, 540` (+`callbacks.py:112`) | `_cache_lock` documented as protecting multi-step cache mutations and never acquired anywhere; ThreadPool workers mutate `_MODEL_CACHE`/`_PREDICTION_CACHE` unlocked; the "app.py uses for cache stats" comment is false | [NEW] |
| F8-013 | `components/insights.py:193-252` (caller `_callbacks_forecast.py:871-880`) | `_extract_forecast_stats` never reads `weather_df`, yet the production Forecast fast path pays a full `pd.read_json` of the weather store per render to feed the ignored argument | [NEW] |
| X1-009 | `components/_callbacks_overview.py:1063-1068` (+`_callbacks_us_grid.py:138-141`) | Live leaderboard tones on 2.5%/5.0% literals matching no `MAPE_BY_HORIZON` row; the 22% rollback value hardcoded as tooltip prose — thresholds outside the H2 `mape_grade` framework | [NEW] |
| X1-010 | `components/_callbacks_forecast.py:665-725` vs `jobs/phases.py:571-672` | Future-feature-frame builder duplicated near-verbatim between web and job, with the weather overlay + #129 anchoring landed only in the job copy; a third partial copy in the backtest fold | [EXT #186] |
| X1-011 | `components/_callbacks_forecast.py:552-560`; `_callbacks_backtest.py:414-423` | ARIMA exog column list duplicated by value from `ARIMA_EXOG_COLS`, plus a redundant pre-fill that `predict_arima` already performs (outputs byte-identical) | [EXT #184] |
| X1-012 | `components/_callbacks_shared.py:356-360, 395-401` | Dual-key Redis backtest fallback triplicated across 3 modules; empirical-interval calibration parameters (window formula, q=0.10/0.90, 0.80 target) triplicated | [NEW] |
| X1-014 | `components/_callbacks_models.py:112-140` vs `:569-597` | Models-tab metrics table duplicated token-identically between the Redis fast path and the v1 fallback in the same file | [NEW] |
| X1-003 | `components/_callbacks_models.py:290-292` vs `_callbacks_overview.py:181-186` | The two `gridpulse:drift:{region}` consumers read different statistics (sMAPE-preferred vs raw MAPE) — same region can show a healthy Overview headline and a Degraded Models-tab chip | [NEW] |
| X1-007 | `components/_callbacks_overview.py:2194-2205, 1862-1864` | Merit-order pricing triplicated with contradictory math (2× vs 5× slope; exponential vs linear scarcity) plus a third tier definition ("High tier (85%)" vs config 0.90) | [NEW] |
| X1-008 | `components/_callbacks_shared.py:70-85` vs job side | Two same-named `_EIA_FUEL_MAP` dicts drifted (14 vs 7 entries; "Natural Gas"→`gas` vs `natural gas`); a stack order iterates a `natural_gas` key no normalizer can produce; three color/label systems | [NEW] |
| F7-006, F8-009, S5-03 | `components/callbacks.py:839-882`; `insights.py:441, 761, 1032, 833-848` | NEXD-11 cross-tab links fully inert (3 of 4 targets are deleted 9-tab-era ids; the one valid target self-suppresses; quick-nav components deleted) while `handle_cross_tab_link` applies unvalidated indices to `active_tab`. Independently confirmed by 3 verifiers | [EXT #188] |
| F7-007, X3-008 | `components/callbacks.py:453-522`; `layout.py:195, 199` | `changes-store`/`pipeline-log-store` written but never read; `audit-store`'s only consumer is that dead chain — `compute_session_changes` burns a demand-DataFrame parse + diff every tick for nothing (extends the S3-04 audit census) | [NEW] |
| F8-011 | `components/layout.py:186-196` | Five `dcc.Store`s (news/models/features/alerts/briefing) have no reader and no writer anywhere — declare-only state slots | [NEW] |
| F8-005, S5-07a | `components/layout.py:122-124, 164`; `callbacks.py:946-981` | Both freshness surfaces are `display:none` carriers with live per-update compute, and exiting Briefing Mode sets the confidence bar's style to `{}`, resurrecting the retired element | [NEW] |
| F8-006 | `components/error_handling.py:462-468` | `widget_confidence_bar`'s skip-list misses `latest_data`, fabricating a green high-confidence "Latest_Data" badge from a timestamp string | [NEW] |
| F8-004 | `components/error_handling.py:25-253` | The module's advertised surface (safe_callback, api_error_state, spinners, empty/error states, freshness_badge — 8 symbols) has zero app callers; only `widget_confidence_bar` is reachable | [EXT #188] |
| F8-010, X1-013 | `components/cards.py:11-277`; `insights.py:777-782` | Six cards builders unreachable from the app (each pinned green by dedicated tests), and two same-named `build_insight_card` functions with incompatible signatures are both imported by `_callbacks_overview.py` | [NEW] |
| F8-003 | `personas/welcome.py:22-194` | "Data-driven welcome" subsystem has zero app callers; its latent copy mislabels EIA's own day-ahead accuracy as "Ensemble MAPE" and the observed historical peak as a forecast | [EXT #188] |
| F8-008 | `personas/config.py:30-148`; `callbacks.py:413-427` | Docstring promises "4 personas with distinct default tabs" but all four set `tab-overview`, so the live callback yanks any persona switch to Overview from any tab | [EXT #188] |
| F8-007 | `components/insights.py:377-387` | "Hours above P90" anomaly rule is mathematically unreachable (at most ~10% of a window can exceed its own 0.9-quantile; the trigger needs >15%) — repro searched 200 random + constructed series per window size | [NEW] |
| F7-008 | `components/_callbacks_us_grid.py:703-708` | Choropleth caption counts regions with live Redis data as "mapped", so a warming, polygon-covered BA renders as "not yet covered here"; the companion test pins the wrong semantic | [NEW] |
| F8-012 | `components/callbacks.py:301-343` | Dev fallback substitutes demo data but marks freshness `stale`, making the banner claim "serving cached data (API unavailable)" about fabricated data; the G2 `demo` status fires only in the no-key branch | [NEW] |
| X3-007 | `components/_callbacks_backtest.py:129-133` | The backtest exog path reads three `weather-forecast*` Redis keys that no code ever writes, so the priority-1 "archived forecast snapshot" source can never engage and every `forecast_exog` backtest silently uses climatology | [NEW] |
| S5-02 | `components/_callbacks_forecast.py:1105`; `_callbacks_overview.py:535` | Model-card badge trusts `is_trained()` (forecast-payload presence), not metric provenance — dev mode demonstrably labels hardcoded simulated metrics "trained" | [EXT #131] |

### Tests

| finding_id(s) | file | one-line defect | status |
|---|---|---|---|
| F8-014 | `tests/e2e/test_dashboard_render.py:187-194` | AC-7.7 "default persona" test ends in `assert True` (verifies nothing); the file docstring claims persona-switch KPI/welcome verification the tests don't perform; no US-Grid e2e render test | [EXT #188] |

**Fix direction (collective).** Everything marked [EXT] should be folded into the acceptance criteria of its tracked issue rather than re-filed; the [NEW] dead-code and duplication rows are natural additions to the #188/#189 sweep sequence (delete-or-wire decision per row); the handful of latent logic defects (F1-007, S1-01, F2-012, F4-011, F8-006) are one-guard-clause changes best batched into a single hardening PR with the existing repro scripts converted to regression tests.

### Tracked-issue overlap notes (quotes for [EXT] rows)

- **#131** (S5-02) — overlap: "the Overview model card can now correctly say 'trained' while still displaying MAPE 1.6%." Delta: the badge still keys on payload presence rather than metric provenance; the dev-mode divergence remains demonstrable today.
- **#166** (X1-015, F6-013) — overlap: "`write_diagnostics` should derive residuals + error-by-hour from the **real forecast the scoring job already produced**". Delta: two *disagreeing* fabricated feature-importance lists must be consolidated in that work, and even with real diagnostics the Models tab still flattens holdout-table vs diagnostics provenance with no visual cue.
- **#181** (X2-003·S1-08; F2-011) — overlap: "Fitting `w` to the same 168h then *reporting* on it is leakage — a great in-sample number that won't hold live." Delta: #181 flags this as a trap for the *future* methodology; F2-011 shows both *current* published ensemble computations already commit it, and the stale "almost always beats"/"never worse" claim sites are not enumerated there.
- **#183** (F2-007) — overlap: "`models/model_service.get_model_metrics` (`model_service.py:80-265`) is a ~185-line, six-layer fallback function". Delta: the documented layer semantics are factually wrong (layers 1–3 are GCS network reads that work on the web tier), so the resolver refactor must also correct the docstring and the CLAUDE.md watchlist.
- **#184** (X2-010; F6-006; S1-02; S2-04b(i); X1-011; F1-008) — overlap: "**Inverse-MAPE ensemble in 3 places.**" and "**Prophet regressor-mode tuple lies** — `PROPHET_REGRESSORS` carries a per-regressor `\"multiplicative\"/\"additive\"` mode … that the loop then ignores, hardcoding `mode=\"additive\"`". Deltas: a fourth combiner with a false consistency docstring lives in the backtest tab; MAPE itself is also implemented 3×; the `actual*(1+noise)` fabrication #184 flags remains ungated; two component-side by-value `ARIMA_EXOG_COLS` copies and the production-dead `train_all_models` orchestrator are outside its checklist; PRD R3.1 is the doc-side counterpart of the mode-tuple item.
- **#185** (F3-007; S3-05; X3-009-revised) — overlap: "Decide + document ONE fallback contract for read clients (stale → GCS where applicable → typed-empty) and apply it consistently". Deltas: the stale tier that contract assumes cannot fire at all (hour-embedded keys, fresh job containers); retry/breaker policy is asymmetric across clients; and a callback-layer second demo-news fallback would silently defeat the planned typed-empty change.
- **#186** (X1-010) — overlap: "They're written in different idioms and kept in lockstep **by hand**." Delta: the future-feature-frame builder is a second hand-synced web/job pair, already divergent (overlay and anchoring landed only in the job copy).
- **#187** (F3-012; F9-007; S5-05) — overlap: "**`_ENV_DEFAULTS` over-stated** (`config.py:35-75`)." Deltas: `USE_DEMO_DATA` is fully unread (its `stale`-vs-`demo` mislabel sub-claim is new), `STALENESS_THRESHOLDS_SECONDS` is write-only with unsatisfiable values, and the `profile`/`workers` rows are dead against a hardcoded Dockerfile.
- **#188** (X2-002; X2-013; X2-015; F9-002·S5-04a; F7-006 cluster; F8-003; F8-004; F8-008; F8-014) — overlap: "**Dead persona config.** `personas/config.py` `priority_tabs`/`kpi_metrics` reference 9-tab-era tab IDs (`tab-forecast`/`tab-backtest`/…) that don't exist in the 5-tab shell" (and, for the count claims, "Feature-count docstrings match `get_feature_names()` (32/49)"). Deltas: the same 9-tab-era/dead-config classes recur in CLAUDE.md's tab list and Module Map, the published docs' feature counts, the `config.py` gate-justification comment, the feature-flag census, the cross-tab-link machinery, the welcome subsystem, `error_handling.py`'s dead surface, the constant `default_tab`, and the vacuous e2e assertion — none enumerated in #188's checklist.

### Empirical repro receipts (decisive lines)

- F1-007 — `estimate_price_impact(nan, 100000.0) = 1000.0 | estimate_price_impact(0.0, 0.0) = 1000.0` — `scratchpad/repro2_models/f1_004_006_007_purepy.py`
- S1-01 — `compute_ensemble_weights({'a': 0.0, 'b': 5.0}) -> {'b': 1.0}` … `{'a': 1e-320, 'b': 5.0} -> {'a': nan, 'b': 0.0}; ensemble_combine -> [nan nan]` — `scratchpad/repro_S1/s1_01_ensemble_edge.py`
- S1-02/S1-03 — `evaluation.compute_mape = 10.0` vs `drift raw mape_over_records = 666666640.0` on the same vectors — `scratchpad/repro_S1/s1_02_03_mape_three_ways.py`
- F2-009/S2-04b — `get_ensemble_weights prod, no pickle: {'prophet': 0.3, 'arima': 0.2, 'xgboost': 0.5} (hardcoded default, ungated)`; S2-01a — `upper_80 == ensemble*1.03? True` — `scratchpad/repro_S2/repro_model_service.py`
- F4-011 — `renewable_pct written to Redis: [50.0, 50.0, nan, 50.0] | NaN entries: 1` — `scratchpad/repro2_jobs/f4_005_011_f1_002.py`
- F3-007 — `prior-hour entry still in cache (fresh, ttl 86400): True / stale lookup under CURRENT key: None` — `scratchpad/repro2_data/f3_007_stale_dead.py`
- F4-008 — `compute_snapshot: alert_count = None | renewable_pct = None` — `scratchpad/repro2_data/f4_008_009_briefing_diff.py`
- F2-006 — `CRASH at report line 253: TypeError - unsupported format string passed to NoneType.__format__` — inline repro under `scratchpad/repro2_platform1`
- F9-012 — raw `pytest scaling-analytics/tests/` → `exit=4`; the ci.yml construction → `'v2 tests skipped (optional)'` with `step exit=0` — `scratchpad/repro2_platform2/f9_012_ci_dead_step.sh`
- F9-014 — boundary discontinuity — `scratchpad/repro2_platform2/f9_014_pricing_boundary.py`
- F8-006 — `badge label='Latest_Data' color='#34d399' tooltip='Live data from verified source'` — `scratchpad/repro2_ui4/f8_006_latest_data_badge.py`
- F8-007 — `T=168 max_count_above_p90=17 threshold=25.20 can_trigger=False` — `scratchpad/repro2_ui4/f8_007_p90_unreachable.py`
- X1-007 — three divergent price curves from one utilization input — `scratchpad/repro2_ui4/x1_007_pricing_divergence2.py`
- X1-008 — `raw "Natural Gas": web -> "gas" | job -> "natural gas"` — `scratchpad/repro2_ui4/x1_008_fuel_map_divergence.py`
- X1-011 — `exog arrays byte-identical (pre-fill is redundant): True; max |diff|: 0.0` — `scratchpad/repro2_ui5/x1_011_exog_fill_redundancy.py`
- S4-01 — `5-NaN run: ['interpolated'] | 6-NaN run: ['gap'] | 6-NaN weather run: NaN remaining=0/6 | 'data_gap' col: absent` — `scratchpad/repro_S4/repro_s4_01_gaps.py`
- S4-02 — all degenerate inputs → `featured.max()+1h` — `scratchpad/repro_S4/repro_s4_02_resolve_start.py`
- X2-002/S5-01 — `total=49 raw_weather=17 derived=32` — `scratchpad/repro_S5/s5_01_feature_count.py`
- X2-009 — `1981 tests collected in 1.64s` — `pytest tests/ --collect-only -q` (venv Python 3.13)
- S5-02 — `is_trained('FPL') = True` while metrics resolve to the simulated dict — `scratchpad/repro_S5/s5_02_badge_divergence.py`
- F7-006/S5-03 — registered tabs vs `related_tab` census (3 of 4 DEAD; handler validates nothing) — `scratchpad/repro_S5/s5_03_related_tab_census.py`
- F9-002/S5-04a — wired-flag census (6 of 18 wired) — `scratchpad/repro_S5/s5_04_flag_census.py`

(Scratchpad root: `/private/tmp/claude-501/-Users-rootk-nextera-portfolio-energy-forecast-energy-forecast-final/e55d5d4a-af14-4217-b412-69238186a7a8/scratchpad`)

---

## Appendix A — Refuted leads

These 13 leads were investigated to a REFUTED verdict. Each row names the specific gate, code path, or artifact that neutralizes the claim so the next audit does not re-chase it. Where a refutation carries a caveat or a surviving residue, it is noted in the last column — residues are tracked separately and do not resurrect the refuted claim.

| finding_id | Claimed defect | Refuting mechanism | Caveat / residue |
|---|---|---|---|
| GAP-00 | Deploy/CI workflows + Dockerfile suspected of REQUIRE_REDIS drift, `PYTHONOPTIMIZE`/`-O` assert-stripping, unpinned deploys, secrets mishandling | Checked, clean: `Dockerfile:37-44` ENV and `:61-68` CMD (plain gunicorn, no `-O`) and all three workflow YAMLs contain no `PYTHONOPTIMIZE`; deploy is SHA-pinned | Clean **with the caveat** that this was a coverage gap-fill, not a finding hunt; already-tracked items (LOG_LEVEL / F9-003, Dockerfile-flag DRY on #187) remain open |
| X2-017 | `README.md:38` "~10× speedup" for the SARIMAX cached-order path has no measurement artifact anywhere in the repo | The artifact exists: PR #84 (merged as commit `753aee1`, 2026-05-06) records run-anchored before/after timings from the 2026-05-05 run (12 min/BA × 51 = 10.2h baseline) | The warm-path multiplier in PR #84's impact table was a pre-merge projection never re-timed in isolation; linking PR #84 from the README is optional polish |
| F3-011 | Overview news widget performs a live RSS fetch in the prod request path and falls back to unlabeled fabricated demo headlines | `_build_overview_news` has no live caller: the R2 Overview layout (`components/tab_overview.py:25-62`) exposes only 5 dynamic IDs, and the sole registered callback (`components/_callbacks_overview.py:2347-2400`) never invokes it; a no-legacy-ids test pins this | Dead-helper retention is the residue, tracked via F5-007 (extends #154); the news-client demo fallback itself is an acceptance criterion on #185 |
| F6-011 | Direct single-model "arima" branch lacks the exog NaN-fill guard present in the ensemble worker, so ARIMA-alone can crash on NaN exog | `models/arima_model.py:232` — `predict_arima`'s first line routes exog through `_get_exog` (`:260-299`), which coerces to numeric and ffill/bfill/zero-fills every NaN unconditionally (the #176 change); demonstrated empirically (`repro2_ui2/r3_interval_tautology.py`) | The duplicated NaN-fill implementations across layers are real and already on #184's checklist |
| X3-009 | Overview news callback converts even an empty successfully-fetched article list into fabricated demo articles, defeating the #185 remediation | Same unreachability as F3-011: the code at `_callbacks_overview.py:2068-2078` exists as described, but repo-wide grep shows zero callers in any registered callback at SHA 5000d6a | Dead-code residue re-filed as X3-009-revised; the claimed prod surface is provably absent from the layout + callback graph |
| S1-07 | Bare `assert` leakage guard at `models/xgboost_model.py:82` is stripped by `python -O`, silently disabling leakage protection in prod | Prod never strips asserts: grep of `Dockerfile` + `.github/workflows/*.yml` for `PYTHONOPTIMIZE`/`-O`/`-OO` returns nothing (exit 1); web tier runs plain gunicorn, jobs run `python -m jobs`. Repro: `repro_S1/s1_07_assert_strip.py` (confirms the `-O` mechanism exists in principle — `__debug__ = False` — but is never engaged) | Style note only; a future operator adding `PYTHONOPTIMIZE` would still be stripping a tautological check |
| S2-03 | Hardcoded baseline metrics dict (prophet 2.8 / xgboost 2.1 / ensemble 1.9 MAPE) at `models/model_service.py:260-265` is reachable in production | The strict gate at `model_service.py:222-227` (commit `2751994`, shipped via PR #167, relates to #149) returns real layers 0–3 only when `REQUIRE_REDIS` is true, which is the staging+prod default (`config.py` `_ENV_DEFAULTS`). Repro: `repro_S2/repro_model_service.py` — `get_model_metrics prod (REQUIRE_REDIS=True, redis+meta empty): {}` | Dev-tier unlabeled baseline is real but its deletion is an explicit acceptance criterion on #183 |
| S2-05 | 600s quality-gate TTL is incoherent with daily training cadence; a stale gate can hide a newly-failing model | Timing arithmetic: the gate's input (xgboost meta MAPE) changes at most once per day (`jobs/training_job.py:283-291` writes it at ~04:00 UTC), so a 600s TTL bounds staleness to 10 minutes against a 24-hour signal | None |
| S3-01 | HTTP 200 with an empty data payload (not an error) skips the stale-cache→GCS fallback chain in the EIA/Open-Meteo clients | Empirical: both clients engaged the full chain under stubbed 200-empty responses — `eia_no_data` warning followed by `eia_demand_gcs_fallback ... rows=1` returning the GCS marker frame. Repro: `repro_S3/s3_01_empty_200.py` | A neighboring gap for 200-with-*unusable-records* is real and reported separately as S3-01b |
| S3-02 | `_EIACircuitBreaker` probe cadence (PROBE_INTERVAL=30) does not match docstring/intent for a ~153-call scoring run | Empirical simulation of a 51-BA × 3-endpoint sustained outage: 3 full-retry calls before trip, 145 fail-fast, probes at call indices 33/63/93/123/153, worst-case ~600s vs the 1800s task timeout — matching #174's intent. Repro: `repro_S3/s3_02_breaker_cadence.py` | Two benign observations: docstring cadence wording is off by one (29 suppressed calls between probes), and unlocked int counters under the 4-worker ThreadPool can shift a probe by ~30s — neither changes the runtime bound |
| S4-03 | With `REQUIRE_REDIS` set, a Redis key that exists but carries an empty/partial payload causes web callbacks to fall through to synthetic/demo data | Empirical, against every claimed surface: existing-but-degraded payloads yield `None` or the explicit warming state (`{'error': 'warming', 'status': 'warming', ...}`), never synthetic — the shipped gates from PRs relating to #130/#149/#167 hold. Repro: `repro_S4/repro_s4_03_empty_payload.py` | Latent-contract residues (a `'fresh'` label on empty stores, a KeyError path, `.get`-default-0 forecast points) would activate only if a *future* writer emits partial payloads — thematically covered by the typed-payload work on #153 |
| S5-04b | `feature_enabled()` returns True for unknown flag names, so a typo silently enables a feature | `config.py:757-769` — unknown flags log `feature_flag_unknown` and return **False**; fail-closed since 2026-05-29 (PR-G8, relates to #145). Empirically: `feature_enabled("no_such_flag") = False`. Repro: `repro_S5/s5_04_flag_census.py` | Surviving sliver: `CLAUDE.md:423` still documents the old fail-open behavior ("unknown flags default to True") — a stale doc line, not covered by #188's CLAUDE.md item |
| S5-07b | Meeting-Ready Mode (C9, documented as implemented) is functionally inert | The clientside callback at `components/callbacks.py:920-931` toggles `body.briefing`, and `assets/custom.css:367-480` attaches ~110 lines of real projection chrome to that class (hides nav-tabs/badges/modebar, scales hero KPI to 56px, stamps a watermark) | Residue: a dead `.meeting-mode` class sliver remains in CSS — adjacent to the consistency sweep on #188 |

## Appendix B — Methodology & coverage

**Scope.** The review examined the working tree at SHA `5000d6a267d2becbfb7f9faf2e99c406641b1b36`, frozen on 2026-07-01. Exclusions: `.claude/worktrees/**`, `specs/archive/**`, and asset/binary files.

**Architecture: territories × lenses.** The review was structured as a grid of code territories crossed with review lenses, executed in three phases:

- **Phase 0 (prep).** Built an issue registry from the 29 live GitHub issue bodies (so every finding could be dedupe-checked against tracked work) and a coverage manifest of the 72 production files.
- **Phase 1 (find).** 17 parallel agents: 9 territory finders, each applying all 7 review axes to its assigned territory; 3 cross-cutting specialists (duplication, claims verification, reachability tracing); and 5 seed verifiers working 28 exploration leads. Output: 157 candidate findings plus 34 seed verdicts.
- **Phase 2 (verify).** 14 adversarial verifiers, module-bucketed with at most 13 findings each. Empirical repro scripts ran sandboxed under the session scratchpad with network stubbed and no repository writes. Output: 163 verdicts. **Zero PLAUSIBLE verdicts survived — every claim was either demonstrated or refuted.**

**Totals.**

| Metric | Count |
|---|---|
| Total verdicts | 197 |
| CONFIRMED | 184 (9 P0, 14 P1, 70 P2, 91 P3, pre-merge of duplicates) |
| REFUTED | 13 (Appendix A) |
| PLAUSIBLE | 0 |
| Dedupe vs prior audit (#181–#189 et al.): NEW | ~104 |
| Dedupe: extensions of tracked issues | ~28 |
| Dedupe: duplicates of verified seeds or intra-review duplicates | ~26 |

**Coverage proof.** All 72 files in the coverage manifest were attested read-in-full by their assigned territory owner. The 3 GitHub workflow YAMLs, initially unattested in Phase 1, were gap-filled in Phase 2 as GAP-00 and came back clean (see Appendix A for the caveat).

**Verdict discipline.** The verifiers ran under an adversarial, refute-first charter. A CONFIRMED verdict required either verbatim repro-script output or an airtight static trace of the failing path. A REFUTED verdict had to name the specific neutralizing mechanism — the gate, test, commit, or unreachability proof that kills the claim — rather than merely expressing doubt.

**The 7 review axes.** Every territory finder applied all seven:

1. Flawed logic
2. Loopholes / missing connections
3. Duplication
4. Over-engineering
5. Half-finished work
6. Forced / illogical constructs
7. Claims without empirical proof

---

## Drafted issues (P0/P1, NEW only)

### Draft 1 — `bug(jobs): scoring job publishes fabricated demo alerts to gridpulse:alerts:* — Risk tab renders them as real NOAA data`

**Labels:** bug, prod-readiness

> **Context.** `jobs/phases.write_alerts` unconditionally calls `data.demo_data.generate_demo_alerts(region)` and writes the result (plus a stress score computed from the fake alert counts) to `gridpulse:alerts:{region}` every hourly tick for all 51 BAs. The visible Risk tab renders these as real active alerts with no demo label; the Overview data-health strip attributes them to "NOAA Alerts" with status LIVE; the real `data/noaa_client.py` has zero non-test callers despite PRD R1.3 (Must Have) and `docs/HOW_IT_WORKS.md` diagramming a scoring-job NOAA fetch. The 2026-06-19 elegance audit (#189) explicitly asserted only two fake-data prod-path findings existed — this is a third it missed.
>
> **Evidence.** At 5000d6a: `jobs/phases.py:1342,1347,1351,1390` (ungated fabrication + write); `jobs/scoring_job.py:259` (hourly invocation); `data/demo_data.py:195-240` (hardcoded year-round Heat/Wind advisories for ERCOT/CAISO/FPL/SPP); `components/_callbacks_alerts.py:60-91` (unlabeled render); `components/callbacks.py:153-156` + `components/_callbacks_overview.py:1666` (hardcoded fresh + "NOAA Alerts" badge); `tests/integration/test_scoring_job.py:203-226` pins the write. Repro (executed real `write_alerts`, stubbed `redis_set`): `Redis key written: gridpulse:alerts:FPL / alert: demo-alert-FPL-1 | Heat Advisory | … until 8 PM local time / stress_score: 35 | stress_label: Elevated`.
>
> **Why it matters.** This is the exact "no fake data on the prod path" rule the project treats as its core credibility guarantee (#131/#149 lineage) — violated on a visible tab, hourly, with false NOAA attribution. Every stress score users see is arithmetic over fabricated counts.
>
> **Suggested fix direction.** Wire `noaa_client.fetch_alerts_for_region` into `write_alerts` (or ship an honest empty/"unavailable" alerts payload plus a "no alert feed connected" UI state); remove the NOAA/LIVE attribution until a real feed backs it; correct HOW_IT_WORKS.md and TECHNICAL_SPEC §2.3; re-point the integration test at the honest payload. Relates to #166 (class-sibling), #189, #185.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 2 — `bug(models): Prophet/SARIMAX forward forecasts time-mislabeled up to ~24h — predictions anchored at training end, labeled from scoring-time forecast_start`

**Labels:** bug, prod-readiness

> **Context.** `predict_prophet` anchors its future window on the pickled model's fit-time history (`models/prophet_model.py:161,209`) and `predict_arima` continues from the training-time `tail_y` (`models/arima_model.py:238-250`), but `jobs/phases.py:771-773` discards Prophet's returned timestamps and `:929-937` labels `preds[i]` with scoring-time `forecast_start + i`. With daily 04:00 training and hourly scoring, the offset grows to ~23h — rotating the diurnal cycle to the wrong hours in the served per-model traces, the blended ensemble, and per-model drift records. XGBoost is unaffected.
>
> **Evidence.** At 5000d6a: files/lines above plus `jobs/scoring_job.py:179-181` (daily GCS pickles). Empirical repro on deployed lib versions: `MISLABEL OFFSET: 15 hours`; ARIMA on a diurnal sine: `MAE vs truth at train_end+1..+24 (true origin): 0.0 MW | vs labeled hours: 379.8 MW`. Scripts archived with the review.
>
> **Why it matters.** Two of three served models are wrong-hour in steady state; drift metrics score them against the wrong hour, so #170's extreme per-model drift (LDWP 188%, AZPS 266%) has an untracked alternative mechanism, and #181's model-ranking analysis is partially contaminated.
>
> **Suggested fix direction.** Honor the model's own forecast origin: extend `periods` by the training-end→forecast-start gap and trim, or join predictions to rows by the timestamps the model emitted; drop/flag models whose window can't cover `forecast_start`. Regression test: per-model prediction timestamps == row timestamps. Re-check #170's numbers afterwards. Relates to #170, #181.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 3 — `bug(models): holdout protocol incommensurable — XGBoost teacher-forced one-step vs Prophet/ARIMA multi-step, all with actual weather; drives ensemble weights and published accuracy`

**Labels:** bug, prod-readiness

> **Context.** `jobs/training_job.py:60-64` scores XGBoost's "168h holdout" on `featured_df.iloc[-168:]`, where every row's demand lags are actual in-window demand — 168 teacher-forced one-hour-ahead predictions. Prophet/ARIMA holdouts are genuine 168-step forecasts. All three use actual observed weather, while production is recursive predicted-lag XGBoost (`jobs/phases.py:675-741`) with forecast/climatology weather. The honest recursive holdout (`models/training.py:112-122`) has no production caller, and `docs/BACKTEST_RESULTS.md` labels the resulting ensemble column "what production serves by default."
>
> **Evidence.** At 5000d6a: lines above; repro confirms the leak premise: "lag feature == actual in-window demand? True | source hour inside the holdout window? True". The comment at `jobs/phases.py:688-691` claiming inference matches the training holdout is wrong for the persisted path.
>
> **Why it matters.** The 1/MAPE ensemble weights and the headline accuracy table compare numbers measured under different protocols; XGBoost's advantage (and #181's 48/51 analysis) is overstated by an unquantified amount.
>
> **Suggested fix direction.** Persist a recursive autoregressive-snapshot XGBoost holdout as the metric of record (machinery exists in `models/training.py` and `scripts/audit/extended_holdout_check.py`); recompute weights and regenerate BACKTEST_RESULTS.md on commensurable bases; caveat teacher-forced numbers wherever they remain. Relates to #181, #186.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 4 — `bug(ui): "80% empirical prediction interval" calibrated from XGBoost residuals for every model selection (Overview hero + Forecast tab)`

**Labels:** bug, prod-readiness

> **Context.** The only backtest writer persists `predictions={"xgboost": …}` (`jobs/phases.py:1149-1166`); `_collect_backtest_residuals` (`components/_callbacks_shared.py:357-377`) substitutes any available model when the requested one is absent. So the Overview hero's ensemble band (`components/_callbacks_overview.py:433-445`) and the Forecast tab's ensemble/prophet/arima bands (`components/_callbacks_forecast.py:224-241, 839-844`) are all XGBoost-calibrated while labeled model-specific "empirical." Secondary: the "calibration window: last {N}h" caption reports a pooled (double-countable) sample count, and one pooled quantile pair is applied constant-width across the horizon.
>
> **Evidence.** Repro with the prod payload shape: identical `lower=50.0 upper=50.0` quantiles returned for ensemble, prophet, and arima; legacy-key coexistence yields `sample_size: 336` from 168 residuals. Given #181 (ensemble median MAPE 3.48% vs XGBoost 2.32%), the band systematically understates ensemble error.
>
> **Why it matters.** Uncertainty is a headline trust feature; the label claims a provenance the pipeline doesn't have — same class as the retired #131/#150, on a new surface.
>
> **Suggested fix direction.** Return residuals only for the requested model (or surface a substitution marker); persist per-model prediction vectors in the backtest payload (the same "step 0" #181 needs); until then relabel the band honestly. Fix the caption semantics and consider lead-time-dependent widths. Relates to #181, #153.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 5 — `bug(ui): data freshness asserted, never measured — Redis fast path hardcodes 'fresh' with render-time timestamps; STALENESS_THRESHOLDS_SECONDS is dead config`

**Labels:** bug, prod-readiness

> **Context.** `_load_data_from_redis` hardcodes `{"demand": "fresh", "weather": "fresh", "alerts": "fresh", "timestamp": now()}` (`components/callbacks.py:153-158, 181-182`); widget-confidence age derives from the 5-minute-refreshed callback timestamp so the 7200s stale check can never fire; `config.STALENESS_THRESHOLDS_SECONDS` (Backlog E2) has zero non-test consumers; `gridpulse:meta:last_scored` is read only by `/health`. Payloads carry no write timestamp, so GCS-fallback-sourced data is indistinguishable from live. With REDIS_TTL=24h, a stalled scoring job serves up-to-24h-old data with no degradation signal (the header badge and confidence bar are additionally in `display:none` carriers).
>
> **Evidence.** Repro with a 23h-old payload: freshness all `fresh`, `age_seconds: 0.0`, confidence `high -> Live data from verified source`. Incident precedent: 2026-06-01 (~4.5h stall), 2026-06-04 EIA outage.
>
> **Why it matters.** The trust-indicator system is self-referential: it reports on the callback run, not the data. During exactly the incident modes it exists for, it says nothing.
>
> **Suggested fix direction.** Stamp `scored_at`/`fetched_at` into job-written payloads (rider on #153's typed contracts), grade on read against the E2 thresholds, drive the fallback banner from the graded state, surface `last_scored` in the header, and un-hide (or remove) the badge carriers. Relates to #153, #188, #171, #174.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 6 — `bug(ui): Overview discards scored_at — stale forecast renders as current "next-24h" narrative, bridge can draw backwards over actuals`

**Labels:** bug, prod-readiness

> **Context.** The Overview hero and insight unpack `_scored_at` from the forecast payload and never use it (`components/_callbacks_overview.py:412-417, 587-589`), and `update_overview_tab` deletes `freshness_data` unused (~`:2380-2381`). Only the Forecast tab shows "FORECAST AS OF". A payload up to 24h stale renders as "Forecast (24h)" / "Next-24h forecast peaks…" with no cue; when actuals are fresher than the forecast (partial-phase-failure shape), the forecast bridge draws backwards over the actuals.
>
> **Evidence.** Repro (payload scored 20h ago, actuals fresh): `first forecast point … 19h BEFORE the bridge anchor; forecast points strictly before the last actual: 19/24; no trace, label, or badge reflects scored_at`. Uniform-stall variant matches the 2026-06-01 incident.
>
> **Why it matters.** The flagship surface presents elapsed hours as a forward forecast during exactly the failure modes operators need to notice.
>
> **Suggested fix direction.** Render a "forecast as of" cue on the Overview (reuse the Forecast tab pattern), mark/suppress forecast points before the last actual, and degrade to warming/stale presentation past a threshold. Coordinates with the freshness-grading issue (Draft 5). Relates to #153.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 7 — `bug(ui): Forecast-tab Generation panel performs live EIA fetch in the web request path, ignoring the scoring job's gridpulse:generation:* payload`

**Labels:** bug, prod-readiness, area:infra

> **Context.** `update_forecast_generation_panel` → `_build_generation_panel` → `_fetch_generation_cached` (`components/_callbacks_overview.py:667-707`, call at `:861`, callback at `components/_callbacks_forecast.py:970-986`) executes `fetch_generation_by_fuel(region)` — a live EIA HTTP call with retry budget (~150s per failing call pre-breaker) plus GCS fallback reads — inside a Dash callback on the stateless web tier. No `REQUIRE_REDIS` gate exists anywhere in the module; the prod service mounts `EIA_API_KEY`, so the path is live. The scoring job already writes `gridpulse:generation:{region}` hourly (`jobs/phases.py:286`) and the Redis reader `_generation_tab_from_redis` is self-documented as orphaned (`components/callbacks.py:64`).
>
> **Evidence.** Full static chain verified at 5000d6a (every hop read verbatim); `grep REQUIRE_REDIS components/_callbacks_overview.py` → zero hits; deploy workflow mounts the EIA secret and `REQUIRE_REDIS=true` on the same service. Not covered by `tests/integration/test_callbacks_redis_only.py`.
>
> **Why it matters.** Direct violation of the CLAUDE.md post-PR#130 web-tier I/O guardrail — the class that produced two real bugs on 2026-05-20 — and a worker-blocking hazard during EIA outages (#174's incident mode, now in the request path).
>
> **Suggested fix direction.** Read `gridpulse:generation:{region}` (un-orphan the existing reader), warming state on miss under `REQUIRE_REDIS`, delete the request-path EIA import, and add this callback to the redis-only guardrail test. Relates to #185, #153, #174.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 8 — `bug(ui): update_alerts_tab has no REQUIRE_REDIS warming gate — prod Redis miss renders inline generate_demo_alerts() output`

**Labels:** bug, prod-readiness

> **Context.** `components/_callbacks_alerts.py:343-358` falls straight to `generate_demo_alerts(region)` on a `gridpulse:alerts:{region}` miss, in every environment — the module contains no `REQUIRE_REDIS` reference, while sibling gates exist at `callbacks.py:272`, `_callbacks_forecast.py:354`, `_callbacks_backtest.py:561`. The module docstring claims "no fallback compute path." The trigger window (24h TTL key) is real: fresh deploy pre-first-scoring-run, Redis flush, or >24h scoring stall.
>
> **Evidence.** Repro with `ENVIRONMENT=production`, `redis_get -> None`: 8 outputs returned (no warming early-return), rendered "Heat Advisory"/"Wind Advisory" cards, stress 35/"Elevated", no demo/sample/synthetic label anywhere.
>
> **Why it matters.** Once the scoring-job alert fabrication (Draft 1) is cleaned up, this path would silently reintroduce fabricated alerts in prod on any cold-Redis window — it must be closed in the same effort or the P0 survives.
>
> **Suggested fix direction.** Add the standard `REQUIRE_REDIS` warming gate routed through `error_handling.warming_state()` (per #188's consolidation direction), restrict the demo fallback to dev with an explicit demo label, correct the docstring, and extend the redis-only guardrail test. Ship with Draft 1. Relates to #188.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 9 — `bug(ui): absent metric fields render as fabricated perfection — "MAPE 0.0%" / "RMSE 0 MW" / "R² 0.000" toned positive`

**Labels:** bug

> **Context.** The Overview model card formats `m.get('mape', 0.0)` and siblings (`components/_callbacks_overview.py:528-533`) and the Models leaderboard renders models lacking `mape` as "0.0%" with the positive tone class (`:1061-1072`) — while partial metric dicts are an explicitly supported prod payload state (`models/model_service.py:121-123`; per-field drops in `jobs/scoring_job.py:39-84` and `jobs/phases.py:958-977`; the #176/#179 NaN-holdout era is a concrete precedent).
>
> **Evidence.** Repro: `{'ensemble': {'rmse': 900}}` → "MAPE 0.0% RMSE 900 MW MAE 0 MW R² 0.000"; leaderboard applies `gp-metric-value--positive` to the fabricated 0.0%. The honest "—" pattern already exists in the same file (`:219-229`).
>
> **Why it matters.** Render-side fabrication of perfect scores from honest partial data — the same credibility class as the retired #131, generated in the UI instead of upstream.
>
> **Suggested fix direction.** Render "—" (no tone class) for absent fields in `build_model_metrics_card` and the leaderboard; unit test that a partial dict never renders "0.0%". Relates to #183, #188.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 10 — `bug(ui): Overview insight labels sMAPE as "MAPE" — artifact-prone BAs show 18% where rolling MAPE is ~190%`

**Labels:** bug

> **Context.** `_resolve_forecast_mape` prefers `rolling_smape_7d` (the #142/PR-G9 robustness choice) but returns only a value+source, and the render string hardcodes the label: `mape_clause = f" ({mape_source} MAPE {mape_value:.1f}%)"` (`components/_callbacks_overview.py:628-636`, preference `:181-186`). No user-facing "sMAPE" string exists in `components/`. The Models tab drift panel shows `rolling_mape_7d` for the same regions, so the two live surfaces can disagree ~10x on what both present as MAPE.
>
> **Evidence.** Repro with fixture magnitudes (smape 18.0, mape 190.0): insight renders "(live 7d MAPE 18.0%)". `tests/unit/test_overview_honest_signals.py:176-192` pins the preference but not the label.
>
> **Why it matters.** The mislabel is wrong precisely where the two statistics diverge — the artifact-prone BAs the sMAPE switch was made for.
>
> **Suggested fix direction.** Resolver returns the metric name; render "sMAPE" when sMAPE is used; align or explicitly label the Models tab drift column; pin the rendered label in the honest-signals test. Relates to #170.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md

### Draft 11 — `bug(ui): US Grid "National Peak (24h)" is the max of per-BA maxima, not a national peak`

**Labels:** bug

> **Context.** `components/_callbacks_us_grid.py:183-189` computes the KPI as the outer max over regions of each region's own 24h max, next to a Total Demand that is a genuine cross-region sum (`:180`) in the same MetricsBar (`:230-237`) — so the "national peak" renders several times smaller than the adjacent total.
>
> **Evidence.** Repro on the real function: 51 BAs each peaking 9.5 GW simultaneously → "Total Demand = 459.0 GW / National Peak (24h) = 9.5 GW" (true simultaneous peak 484.5 GW). Existing tests only cover single-region or the Total cell, so the semantic is unpinned.
>
> **Why it matters.** A flagship national-rollup number is off by an order of magnitude relative to its label.
>
> **Suggested fix direction.** Sum aligned per-BA `today_mw` series and take the max (handling non-aligned windows), or relabel to "Largest BA Peak (24h)"; add a multi-region test pinning the chosen semantic.
>
> Found by critical review 2026-07, see docs/internal/CRITICAL_REVIEW_2026-07.md