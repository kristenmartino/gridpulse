# Next Up — Post-Redesign Roadmap

_Living punchlist for work queued after the [shell redesign](.claude/plans/shell-redesign-v2.md) (R1→R6 + R7 emoji sweep) and the [multi-model scoring extension](.claude/plans/scoring-job-multi-model.md) (option A + Stages 1–3) shipped to `main`._

Each item below has explicit **acceptance criteria** and a **rough effort estimate** so it can be picked up cold.

## V0 — Production verification (do first, before everything else)

Both option-B PRs are merged. Verification is the only thing left between the user-visible Forecast tab being honest end-to-end. **Until V0 closes, don't ship V1+.**

### V0.1 — Redeploy scoring Cloud Run Job

The Cloud Run **service** (`gridpulse`) auto-deploys from `main`. The Cloud Run **Job** (`gridpulse-scoring-job`) does **not** — it needs an explicit image push. This caught us once with Stage 1 ([context](https://github.com/kristenmartino/gridpulse/pull/53#issuecomment)); easy to miss again.

```bash
# 1. Build + push image
gcloud builds submit --tag gcr.io/<project>/gridpulse:multi-model

# 2. Update the Cloud Run Job
gcloud run jobs update gridpulse-scoring-job \
  --image=gcr.io/<project>/gridpulse:multi-model \
  --region=<region>

# 3. Memory bump if needed (loaded models grow ~3× since Stage 3)
gcloud run jobs update gridpulse-scoring-job --memory=1Gi --region=<region>

# 4. Manual run — don't wait for hourly tick
gcloud run jobs execute gridpulse-scoring-job --region=<region>

# 5. Watch logs for per-region phase results
gcloud logging read 'resource.labels.job_name="gridpulse-scoring-job"' \
  --limit=30 --format='value(textPayload, jsonPayload)'
```

**Acceptance**: log line `forecast {ok=True, models=['arima', 'ensemble', 'prophet', 'xgboost']}` for each region. **Effort**: 5 min.

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

### V1.α — Add 8 BAs to reach ~98% US load coverage

**Files**: `config.py` (`REGION_COORDINATES`, `REGION_CAPACITY_MW`), Cloud Run Job memory bump.

**Adds**: `SOCO`, `TVA`, `DUK`, `CPLE`, `BPAT`, `AZPS`, `NEVP`, `PSCO` (8 → 16 total).

**Capacity numbers should come from authoritative sources** (each BA's most recent IRP / CDR / 10-K). The plan file proposes order-of-magnitude defaults; treat them as placeholders until verified.

**Acceptance**:
- Region picker in the header shows 16 entries
- `python -m jobs scoring` completes for every region without OOM
- 16 distinct `wattcast:forecast:{region}:1h` keys exist in Redis

**Effort**: ~2 hr (mostly capacity-value research).

**Risk**: Cloud Run memory. Bump to 1 GB scoring + 2 GB training in the same PR; rollback is reverting the config commit.

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

**Blocks on**: V1.α (cards for the new BAs need their Redis rows to populate).

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

---

## Status board (updated when items complete)

| Item | Status | PR / Doc |
|---|---|---|
| V0.1 Redeploy scoring Cloud Run Job | open | n/a |
| V0.2 Verify Redis carries 4 model keys | open | n/a |
| V0.3 UI walkthrough | open | n/a |
| V1.α Region expansion (16 BAs) | scoped | [`scoring-job-multi-model.md`](../.claude/plans/us-grid-expansion.md) |
| V1.β US Grid small-multiples tab | scoped | [`us-grid-expansion.md`](../.claude/plans/us-grid-expansion.md) |
| V1.γ US Grid map overlay | scoped | [`us-grid-expansion.md`](../.claude/plans/us-grid-expansion.md) |
| V2.1 Hidden-tab deletion | open | n/a |
| V2.2 Brand spec addendum | open | n/a |
| V2.3 PRD tab-list addendum | open | n/a |
| V2.4 PJM scoring investigation | blocked on V0 | n/a |
