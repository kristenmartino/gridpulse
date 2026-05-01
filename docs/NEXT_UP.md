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

## V3 — Scoped backlog (post-V2)

_Scoped 2026-05-01. Each item now has explicit acceptance criteria, files, and effort so it can be picked up cold._

**Recommended order** (highest leverage first, lowest blast radius first):

1. ~~**V3.ε** NEVP capacity verification~~ — ✅ shipped 2026-05-01 ([config.py](../config.py); 8,000 → 15,445 MW per EIA-860M Feb 2026)
2. **V3.α** Interchange flow visualization — ~1.5 days, mostly UI (data already plumbed)
3. **V3.β** Real BA-polygon choropleth — ~3 days, GeoJSON cleanup + map upgrade
4. **V3.γ** Hawaii / Alaska coverage — 3–5 days, data-path investigation
5. **V3.δ** Multi-tenant / per-user views — deferred (weeks; awaits product-market signal)

### V3.ε — NEVP capacity verification — ✅ shipped 2026-05-01

**What landed**: `REGION_CAPACITY_MW["NEVP"]` updated from 8,000 → **15,445 MW** in [config.py](../config.py).

**Source**: EIA-860M February 2026, retrieved via the EIA API v2 (`/electricity/operating-generator-capacity/data/`) on 2026-05-01: 261 generators in the NEVP balancing authority, summed at the `nameplate-capacity-mw` field, filtered to `statusDescription == "Operating"`.

**Why the V1.α estimate was so wrong**: I conflated NV Energy's utility-owned fleet (~6–8 GW) with the BA-level total. EIA's BA capacity counts every generator in the territory — utility-owned + IPPs + wholesale sellers. The 15,445 MW figure aligns with NEVP's Las Vegas summer peak (~7 GW) at typical IOU reserve margins.

**Follow-up worth noting**: the other 7 V1.α BAs were sourced from utility 10-Ks / IRPs, not EIA-860M. A future cleanup could re-source all 16 against EIA-860M for uniformity; that's not in this scope but worth a one-line note in [config.py](../config.py) if it ever happens.

---

### V3.α — Interchange flow visualization

**Why**: GridPulse already pulls hourly tie-line interchanges via [`fetch_interchange`](../data/eia_client.py) (returns `[timestamp, from_ba, to_ba, interchange_mw]`), but nothing in the UI surfaces them. Operators in PJM/MISO/SPP routinely care about net imports/exports — currently a blind spot.

**Goal**: Add an interchange overlay to the US Grid tab. For the active region (or all 16), show net import/export rate and the top 3 counterparty BAs.

**Files**:
- [`components/tab_us_grid.py`](../components/tab_us_grid.py) — new section + container IDs
- [`components/callbacks.py`](../components/callbacks.py) — `update_us_grid_interchange` callback reading `wattcast:interchange:{region}:1h` (new Redis key)
- [`jobs/phases.py`](../jobs/phases.py) — new phase `write_interchange` invoking `fetch_interchange` and serializing to Redis (mirror the existing `write_generation` shape)
- [`jobs/scoring_job.py`](../jobs/scoring_job.py) — invoke the new phase per region
- [`assets/custom.css`](../assets/custom.css) — interchange chip / panel styling
- New `tests/unit/test_tab_us_grid_interchange.py` — fake Redis row, assert UI shape

**Acceptance**:
- Per-region net interchange (MW signed) renders on every region card.
- Top 3 counterparties listed (e.g. PJM ↔ MISO −1,200 MW).
- `wattcast:interchange:{region}:1h` populated by the next hourly scoring run.
- New unit test green; existing 1377 tests still pass.

**Effort**: ~1.5 days. Most work is UI + the new scoring phase; data plumbing already exists.

**Risk**: EIA's interchange endpoint sometimes returns sparse data for smaller BAs (NEVP, AZPS). Handle empty payloads gracefully in the UI.

**Open questions**:
- Sankey across all 16 BAs vs per-region detail panel? Sankey is a stronger story but harder to read at 16 nodes; per-region detail is safer.
- Show as a header chip on the US Grid card or as a separate full-width panel?

---

### V3.β — Real BA-polygon choropleth

**Why**: V1.γ's map view ([deferral noted in `~/.claude/plans/us-grid-expansion.md:252-254`](../.claude/plans/us-grid-expansion.md)) renders centroid scatter because real BA territories don't follow state lines (PJM spans 13 states; SOCO is parts of AL/GA/MS). Centroids were "80% of the visual punch for 10% of the cost." V3.β closes the remaining 20%.

**Goal**: Replace `scatter_geo` with `choropleth_mapbox` keyed on actual BA service-area polygons.

**Files**:
- New `assets/ba_polygons.geojson` — cleaned BA service-territory polygons (16 features)
- [`components/tab_us_grid.py`](../components/tab_us_grid.py) — toggle wires to the polygon path
- [`components/callbacks.py`](../components/callbacks.py) — `update_us_grid_map_choropleth` (replaces / extends current scatter callback)
- New `scripts/build_ba_polygons.py` — one-shot GeoJSON ingest + simplification script
- `tests/unit/test_tab_us_grid_map.py` — assert each polygon has a region code matching `REGION_COORDINATES` keys

**Source**: HIFLD (Homeland Infrastructure Foundation-Level Data) electric retail service territory polygons + manual aggregation for multi-utility BAs (SOCO = Alabama Power + Georgia Power + Mississippi Power, BPAT = federal hydro service area, etc.). EIA Energy Atlas KMZ is an alternate source but heavier cleanup.

**Acceptance**:
- 16 polygons cover ~98% of US contiguous load (matching V1.α's coverage claim).
- Map view defaults to choropleth; centroid scatter survives as a fallback toggle.
- Polygon file <500 KB after `mapshaper -simplify 5%` to keep page-load fast.
- Click on a polygon → same drill-down as the current centroid click.
- `pytest tests/unit/test_tab_us_grid_map.py` clean.

**Effort**: ~3 days. GeoJSON cleanup is the bulk; the Plotly swap is straightforward.

**Risk**:
- BA boundaries vs retail service territories don't always align cleanly. Document the aggregation methodology in the GeoJSON's `properties` field.
- Page weight: if the simplified file lands above 500 KB, consider lazy-loading on toggle click rather than at initial render.

**Open questions**:
- HIFLD vs EIA Atlas as the primary source? HIFLD is structured for retail utilities; EIA Atlas is structured for BAs. EIA Atlas is the better fit if the cleanup is tractable.
- Color the polygon by current demand, by stress, or by demand delta vs forecast? V1.γ's scatter colors by stress — keep that for consistency unless there's a reason to change.

---

### V3.γ — Hawaii / Alaska coverage

**Why**: GridPulse claims "~98% of US load coverage" after V1.α, but contiguous-only. Adding HI/AK is the natural completion.

**Data path challenges**:
- **Hawaii**: EIA-930 reports HECO at hourly granularity, but data quality is inconsistent compared to ISO/RTO BAs (more gaps, occasional sentinel zeros). NOAA NWS has weather coverage for the islands.
- **Alaska**: The major systems (Anchorage M&LP, Golden Valley, Chugach Electric) are NOT in EIA-930 hourly. EIA-861 reports them annually. Real-time data would need vendor-specific APIs or scraping. The Alaska Railbelt grid is also electrically isolated from the contiguous US — different operational context.

**Goal**: Add HI to the dashboard via standard EIA path. Defer AK pending a viable hourly data source.

**Files** (HI-only):
- [`config.py`](../config.py) — `REGION_COORDINATES["HECO"]` + `REGION_CAPACITY_MW["HECO"]` + `STATE_TO_BA["HECO"] = ["HI"]`
- [`data/eia_client.py`](../data/eia_client.py) — verify EIA-930 returns valid data for `HECO` respondent code
- New `tests/unit/test_heco_data_quality.py` — assert recent HECO data has <10% gap rate

**Acceptance**:
- Region picker shows HECO; clicking it loads forecast/actuals/weather without errors.
- HECO scoring run completes ok in the next hourly tick.
- Data-quality gate: if HECO's gap rate exceeds threshold, log a warning but don't fail the scoring job for other regions.

**Effort**: 3–5 days for HI alone (mostly data-quality QA + edge-case handling). AK is a separate planning effort; not in scope here.

**Risk**:
- HECO data may be too sparse to train a usable model. Worst case: the region appears in the picker but the forecast tab shows perpetual "warming" state. Add a feature flag to hide HECO if data quality is below threshold.
- HECO peak demand is ~1,200 MW (vs ERCOT's 80,000+) — visualizations that scale by demand magnitude will under-represent it.

**Open questions**:
- Is the visual under-representation acceptable, or do we need a normalized view for small BAs?
- Treat HI as a 17th region or as a separate "non-contiguous" category? Separate category protects the "98% US coverage" narrative for the existing 16.

---

### V3.δ — Multi-tenant / per-user views (deferred)

**Why**: Currently single-deployment, no auth. Each browser sees the same data. For GridPulse to graduate from a portfolio piece to a real SaaS, this is required.

**Defer rationale**: Weeks of architectural work without product-market validation. Don't build until there's evidence multiple operators want their own scoped views.

**Sketch** (when revived):
- Auth via Auth0 or Clerk fronting Cloud Run with IAP, or a Supabase auth path matching the existing Supabase MCP.
- Per-tenant Redis namespace: `wattcast:{tenant_id}:forecast:{region}:1h` instead of `wattcast:forecast:{region}:1h`.
- Tenant-scoped region access list (some operators may only want the BAs they cover).
- User profile / preferences scoped to the (user_id, tenant_id) pair, replacing the current localStorage-only `user-prefs-store`.
- Scoring/training jobs need to either run per-tenant or remain shared with tenant-scoped Redis writes.

**Effort**: 2–4 weeks. Major architectural change. Affects every Redis-touching surface and the Cloud Run cost model (per-tenant min-instances vs shared min-instances).

**Risk**: This change touches the entire pipeline. Don't take it on without a concrete tenant pipeline or paying user.

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

## Next: V3 backlog

V3 was scoped on 2026-05-01. See §V3 above for the full plan per item. Recommended order:

1. ~~**V3.ε** NEVP capacity verification~~ — ✅ shipped 2026-05-01
2. **V3.α** Interchange flow visualization (~1.5 days)
3. **V3.β** Real BA-polygon choropleth (~3 days)
4. **V3.γ** Hawaii coverage (3–5 days)
5. **V3.δ** Multi-tenant — deferred
