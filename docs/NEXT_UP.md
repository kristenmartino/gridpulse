# Next Up — Post-Redesign Roadmap

_Living punchlist for work queued after the [shell redesign](.claude/plans/shell-redesign-v2.md) (R1→R6 + R7 emoji sweep) and the [multi-model scoring extension](.claude/plans/scoring-job-multi-model.md) (option A + Stages 1–3) shipped to `main`._

Each item below has explicit **acceptance criteria** and a **rough effort estimate** so it can be picked up cold.

## V0 — Production verification (do first, before everything else)

Both option-B PRs are merged. Verification is the only thing left between the user-visible Forecast tab being honest end-to-end. **Don't ship UI-facing V1+ work (V1.β/γ) until V0.2/V0.3 close.** V1.α was config-only and shipped ahead in [#61](https://github.com/kristenmartino/gridpulse/pull/61) — see status board.

### V0.1 — Trigger scoring run + verify

`.github/workflows/deploy-prod.yml` auto-deploys both the Cloud Run **service** (`gridpulse`) and the Cloud Run **Jobs** (`gridpulse-scoring-job`, `gridpulse-training-job`) on every push to `main` — see `docs/SCHEDULED_JOBS.md`. The only user-driven step is to trigger a manual scoring run instead of waiting for the next hourly tick, then confirm all four models land in the logs.

```bash
# 1. Confirm the CI deploy succeeded for the latest main commit
gh run list --workflow deploy-prod.yml --branch main --limit 1

# 2. Manual run — don't wait for the hourly tick
gcloud run jobs execute gridpulse-scoring-job --region=us-east1 --wait

# 3. Read recent logs for per-region phase results
gcloud logging read 'resource.labels.job_name="gridpulse-scoring-job"' \
  --limit=30 --format='value(textPayload, jsonPayload)'
```

**Acceptance** (all from the `gcloud logging read` output above, post-V1.α):
- `scoring_job_complete ok_count=16 fail_count=0 failed_regions=[]` — every region's pipeline finished cleanly.
- Three `model_loaded` lines per region (one each for `xgboost`, `prophet`, `arima`) — 48 total. Ensemble is computed in-process and never loaded from disk.
- One `scoring_ensemble_*` line per region confirms the ensemble path executed. Until the first training tick after [afd6a49](https://github.com/kristenmartino/gridpulse/commit/afd6a49) writes prophet/arima holdout MAPEs into the trained-model `.meta.json`, expect `scoring_ensemble_equal_weights_fallback`; afterward, expect MAPE-derived weights with no fallback log.

**Effort**: 5 min.

### V0.2 — Verify Redis carries all four model keys

```bash
redis-cli GET wattcast:forecast:FPL:1h | python -c "
import sys, json
d = json.loads(sys.stdin.read())
row0 = d['forecasts'][0]
print('keys:', sorted(row0.keys()))
print('ensemble_weights:', d.get('ensemble_weights'))
print('primary_model:', d.get('primary_model'))
"
```

**Pass**:
- `keys` includes `xgboost`, `prophet`, `arima`, `ensemble` (alongside `predicted_demand_mw`, `timestamp`)
- `ensemble_weights` is a dict summing to ~1.0
- `primary_model` is `"xgboost"`

**Fail modes + diagnostics**:
- `keys` missing `prophet` → `load_model(region, "prophet")` returned `None`. Check `gs://nextera-portfolio-energy-cache/models/{region}/prophet/latest.json` exists.
- `keys` missing `arima` → same story, `arima` pickle path.
- `ensemble` missing but ≥2 models present → `compute_ensemble_weights` failed. Check scoring-job logs for `scoring_ensemble_failed`.
- `ensemble_weights` is `None` but `ensemble` is present → all model MAPEs were `None` so the equal-weights fallback kicked in (still correct, just informational that we couldn't use real weights).

**Effort**: 2 min.

### V0.3 — UI walkthrough

Open the deployed Forecast tab; click each of the four model options in the segmented control. Each should render a forecast (not the "Pipeline is warming up" empty state). Sanity-check that XGBoost / Prophet / ARIMA forecasts are within ±15% of each other on stable demand windows (they're forecasting the same thing); Ensemble should sit between them.

**Effort**: 5 min.

---

## V1 — Region expansion

Plan: [`~/.claude/plans/us-grid-expansion.md`](../.claude/plans/us-grid-expansion.md). Three sequenced PRs (α / β / γ).

### V1.α — Add 8 BAs to reach ~98% US load coverage (shipped)

Shipped in [#61](https://github.com/kristenmartino/gridpulse/pull/61). Adds `SOCO`, `TVA`, `DUK`, `CPLE`, `BPAT`, `AZPS`, `NEVP`, `PSCO` (8 → 16 total). The memory-bump step in the original spec was unnecessary — `deploy-prod.yml` already runs scoring at 4 Gi / training at 8 Gi. First 16-region training runs at 04:00 UTC on 2026-05-01.

**Open follow-up**: NEVP's 8,000 MW capacity is a first-pass estimate (NV Energy doesn't separately disclose Nevada Power's BA-specific fleet). See V3 for the verification follow-up.

### V1.β — US Grid tab (small-multiples)

**Files**: `components/tab_us_grid.py` (new), `components/layout.py` (5th visible tab), `config.py` (`TAB_LABELS` + `TAB_IDS`), `components/callbacks.py` (`update_us_grid_snapshot`), `assets/custom.css` (`.gp-region-card` + `.gp-region-grid`), `tests/unit/test_tab_us_grid.py` (new), `tests/unit/test_redesign_smoke.py` (bump `test_visible_tab_count_is_four` → `_five`).

**Layout**:
```
US Grid tab
├─ Title (region count + national now-demand)
├─ 4-up MetricsBar (Total Demand · National Peak Today · Highest-Stress Region · Lowest Reserve)
├─ Responsive grid of region cards (16 cards on desktop, 4-col)
│   each card: name + demand (hero) + delta chip + 24h sparkline + stress chip
└─ Footer
```

**Click → drill-down**: any region card sets `region-selector.value` AND `dashboard-tabs.active_tab=tab-outlook`, landing the user on Forecast for that region.

**Acceptance**:
- New tab between Overview and Forecast
- All regions render cards (or `—` placeholders when Redis cold)
- Click any card → Forecast tab opens with that region selected
- Tests: `pytest tests/unit/test_tab_us_grid.py tests/unit/test_redesign_smoke.py -q` clean

**Effort**: 1–2 days. Reuses every v2 component already shipped.

**Status**: ready to start. V1.α shipped in [#61](https://github.com/kristenmartino/gridpulse/pull/61); cards for the new BAs will populate as their Redis rows fill in over the first few post-V1.α scoring cycles.

### V1.γ — US Grid map overlay

**Files**: extend `tab_us_grid.py` + `callbacks.py` + `custom.css`.

**Toggle**: `Cards | Map` segmented control above the grid. Map view replaces cards with a Plotly `scatter_geo` of BA centroids — sized by current demand, colored by stress score.

**Why centroid scatter, not choropleth**: real BA territories don't follow state lines. Open-source BA-polygon GeoJSON exists but cleanup is ~2 days alone. Centroid scatter is 80% of the visual punch for 10% of the build cost; document choropleth as a follow-up plan.

**Acceptance**:
- Toggle preserves data when flipping between views
- Click on a map point → same drill-down as a card click
- Map theme reads as v2-native (not default Plotly chrome) — modebar trimmed via `PLOT_CONFIG`, paper bg = `--bg-base`, axis fonts = `--text-tertiary`

**Effort**: ~1 day.

**Blocks on**: V1.β (the toggle UX needs the cards path to exist).

---

## V2 — Outstanding craft cuts (carried over from REDESIGN_CHECKLIST.md)

Low pressure, no specific deadline. Pick up when there's a quiet hour.

### V2.1 — Hidden-tab file deletion

R3 hid 5 tabs (`tab_forecast.py` Historical, `tab_backtest.py`, `tab_generation.py`, `tab_weather.py`, `tab_simulator.py`) via `tab_class_name="d-none"` so callbacks would still resolve. R4a–c absorbed their content into the visible tabs. The hidden modules now render dead trees in the DOM that nothing reads.

**Process**:
1. For each hidden tab, list every callback Output that targets one of its IDs
2. Confirm none are still functional / wired (most should be dead since R4 absorbed them)
3. Delete the module + the callbacks + the corresponding `_tab(...)` line in `layout.py`
4. Run `pytest tests/unit -q` — anything that depends on the deleted IDs surfaces as a failing test
5. Update `tests/unit/test_redesign_smoke.py::TestTabStructure` — visible-tab count and the hidden-tab list both shrink

**Acceptance**: `tab_class_name="d-none"` no longer appears in `layout.py`; `_VISIBLE_TABS` and `dbc.Tabs.children` are the same length.

**Effort**: 4–8 hr (careful audit; one missed callback breaks production).

### V2.2 — Brand spec doc addendum

`docs/gridpulse_brand_system_spec.md` still references cyan / blue / teal. R1 swapped the production palette to v2 blue (`#3b82f6`) + orange (`#f97316`). Add a 1-paragraph addendum at the top noting the swap; cite [PR #36 (R1)](https://github.com/kristenmartino/gridpulse/pull/36).

**Effort**: 15 min.

### V2.3 — PRD tab-list addendum

`PRD.md` lists the original 9 tabs. R3 reduced to 4 visible (Overview / Forecast / Risk / Models). Add an addendum to the IA section; cite [PR #38 (R3)](https://github.com/kristenmartino/gridpulse/pull/38).

**Effort**: 15 min.

### V2.4 — PJM scoring failure investigation

User reported during the post-R6 walkthrough that PJM specifically returned `warming` for non-XGBoost models. After option B Stages 1–3 ship Prophet + ARIMA + Ensemble to Redis, this is automatically fixed _if_ the daily training job has produced trained pickles for PJM. If not, the issue is upstream: the training job is failing for that region.

**To diagnose**:
```bash
# Check whether Prophet pickle exists in GCS
gsutil ls gs://nextera-portfolio-energy-cache/models/PJM/prophet/

# If empty / missing — read the daily training job logs
gcloud logging read 'resource.labels.job_name="gridpulse-training-job"' \
  --limit=100 --format='value(jsonPayload)' | grep -i "PJM\|prophet"
```

If `train_prophet` is failing for PJM specifically, the failure mode is usually data quality (NaN runs, insufficient samples, regressor misalignment). Out of scope for the redesign / option B work; needs its own debugging session.

**Effort**: 1–2 hr (data-quality debugging). Defer until V0 confirms the rest of the regions work.

---

## V3 — Operator-flagged future ideas (not yet scoped)

Captured for future planning sessions, not actionable today:

- **BA-to-BA flow analysis** — interconnection-level data via FERC 714 / OASIS. Adds new data dependency.
- **Real BA-polygon choropleth** — replace V1.γ's centroid scatter with actual service-territory polygons. Requires open-source GeoJSON cleanup (~2 days).
- **Hawaii / Alaska coverage** — EIA-930 doesn't report HEI / Alaska at meaningful resolution. Different data path needed.
- **Multi-tenant / per-user views** — currently single-deployment, no auth. Would need user accounts + per-tenant Redis namespace.
- **NEVP capacity verification** — current value (8,000 MW in [config.py](../config.py)) is a first-pass estimate from V1.α since NV Energy bundles Nevada Power south + Sierra Pacific Power north in its IRP. Verify against the next EIA-860 Form Schedule 6 refresh and update.

---

## Status board (updated when items complete)

_All V0–V2 items shipped as of 2026-05-01. Verified live: the 2026-05-01 04:00 UTC training run wrote real holdout MAPEs for prophet (FPL: 7.88, PJM: 11.04) and ARIMA (FPL: 5.55, PJM: 5.19); the 2026-05-01 09:00 UTC scoring run produced skill-weighted ensembles like `{xgboost: 0.578, prophet: 0.293, arima: 0.130}` instead of the equal-weights fallback._

| Item | Status | PR / Doc |
|---|---|---|
| V0.1 Trigger scoring run + verify | ✅ shipped + verified live | [#57](https://github.com/kristenmartino/gridpulse/pull/57) [#58](https://github.com/kristenmartino/gridpulse/pull/58) [#59](https://github.com/kristenmartino/gridpulse/pull/59) [#60](https://github.com/kristenmartino/gridpulse/pull/60) |
| V0.2 Verify Redis carries 4 model keys | ✅ shipped + verified via scoring logs | [#57](https://github.com/kristenmartino/gridpulse/pull/57) [#58](https://github.com/kristenmartino/gridpulse/pull/58) |
| V0.3 UI walkthrough | ✅ shipped + manual walkthrough confirmed | [#57](https://github.com/kristenmartino/gridpulse/pull/57) [#58](https://github.com/kristenmartino/gridpulse/pull/58) |
| V1.α Region expansion (16 BAs) | ✅ shipped | [#61](https://github.com/kristenmartino/gridpulse/pull/61) |
| V1.β US Grid small-multiples tab | ✅ shipped | [#64](https://github.com/kristenmartino/gridpulse/pull/64) |
| V1.γ US Grid map overlay | ✅ shipped | [#64](https://github.com/kristenmartino/gridpulse/pull/64) |
| V2.1 Hidden-tab deletion | ✅ shipped | [#63](https://github.com/kristenmartino/gridpulse/pull/63) |
| V2.2 Brand spec addendum | ✅ shipped | [`docs/gridpulse_brand_system_spec.md`](./gridpulse_brand_system_spec.md) |
| V2.3 PRD tab-list addendum | ✅ shipped | [`PRD.md`](../PRD.md) |
| V2.4 PJM scoring investigation | ✅ auto-resolved by V0 — PJM now loads all 3 models hourly | [#57](https://github.com/kristenmartino/gridpulse/pull/57) [#58](https://github.com/kristenmartino/gridpulse/pull/58) |

## Next: V3 candidates

Captured in §V3 above. Not yet scoped — pick one to take to a planning session if/when ready:

- **BA-to-BA flow analysis** — interconnection-level data via FERC 714 / OASIS. New data dependency.
- **Real BA-polygon choropleth** — replace V1.γ's centroid scatter with actual service-territory polygons. ~2 days of GeoJSON cleanup.
- **Hawaii / Alaska coverage** — EIA-930 doesn't report HEI / Alaska at meaningful resolution; needs a different data path.
- **Multi-tenant / per-user views** — currently single-deployment, no auth. Needs user accounts + per-tenant Redis namespace.
