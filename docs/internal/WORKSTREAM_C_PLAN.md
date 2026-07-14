# Workstream C — Implementation Plan (#194 + #195)

## 1. Executive summary

- **What's broken:** Two measurement-seam defects on `main @ 647b8d2`. **#194 (P0-2):** Prophet/SARIMAX forward forecasts are written at scoring-tick timestamps but their *values* are anchored to the frozen daily-training end — an offset that grows to ~23h before each 04:00 UTC retrain, mislabeling served rows and corrupting per-model drift. **#195 (P1-1):** XGBoost's 168h holdout is teacher-forced one-step (rows see true previous-hour demand) while Prophet/ARIMA are honest 168-step forecasts — incommensurable MAPEs that drive inverse-MAPE ensemble weights and every published accuracy table.
- **The two fixes:** #194 — re-anchor Prophet/ARIMA per tick (predict `gap + horizon`, slice the horizon window starting at `forecast_start`) with a timestamp-join guard so no value is written to a mismatched row. #195 — score XGBoost's holdout recursively via a shared `recursive_autoregressive_forecast` helper reused by production inference, keeping observed weather for all three.
- **Sequencing:** Two PRs, **#194 first** (it changes served forecasts and must land before any #170 drift re-measurement is trustworthy), **#195 second** (its payoff is re-measuring #181 on aligned, commensurable data).
- **Headline risk:** #194 self-heals on the next scoring tick (stateless web tier, pure revert); **#195 latches recomputed weights into GCS `meta.json` behind `latest.json`** — rollback requires a retrain or a `latest.json` re-point, and a `force=True` first run is needed or unchanged-data BAs keep stale weights.
- **Scope discipline:** Neither PR touches the weather-optimism asymmetry (all three holdouts still use observed weather vs production's forecast/climatology) — that stays an explicit follow-up, not bundled.

---

## 2. Fix 1: #194 time-alignment

### Recommended approach
**Option 1 (re-anchor per tick) with a timestamp-join guard borrowed from Option 2.** Predict `gap + horizon` steps from the frozen training end, slice the `horizon`-long window starting at `forecast_start`, keyed by explicit per-step timestamps; reject any model whose window can't reach `forecast_start`. Rejected alternatives: **Option 3 (refit-at-scoring)** — retraining SARIMAX + Prophet hourly × 51 BAs blows the hourly job budget (the whole lean-pickle/runtime-split design exists to avoid this). **Pure Option 2 (join-only)** — insufficient because ARIMA emits *no* timestamps and Prophet's window *ends before* `forecast_start` once the gap exceeds horizon slack; there'd be nothing to join to. Option 1 generates coverage; Option 2's join is folded in as the safety guard. Cost is bounded: `gap ≤ ~23h`, so ≤~23 extra forecast steps on one already-per-tick predict call.

### Exact code changes

**`models/arima_model.py` — persist a training-end anchor, surface an origin + timestamps**
- `train_arima` (payload dict, `:200-207`): add `"train_end": str(pd.Timestamp(train_df["timestamp"].iloc[-1]))` — the anchor `tail_y` currently loses.
- `predict_arima` (`:238-250`): add keyword-only `start_ts: pd.Timestamp | None = None`. When `start_ts > train_end + 1h`: compute `gap`, forecast `steps = gap + periods`, slice `[gap : gap+periods]`, return `{"forecast": <sliced>, "timestamps": date_range(start_ts, periods, "h")}`. **Return-shape rule (resolves an A-internal ambiguity):** return a **dict only on the anchored path (`start_ts` given)**; keep the **legacy array return when `start_ts is None`** — this minimizes blast radius on the ~10 refit-then-predict callers. Legacy pickles lacking `train_end` → treat `gap=0` (today's behavior), log `arima_predict_no_anchor`.
- Exog for the gap (main implementation subtlety): build ARIMA's exog frame from `featured.tail(gap)` (real historical weather for the now-past gap hours) concatenated with `future_df` (forecast exog ahead), so the value window and exog window agree. Do **not** rely on `_get_exog`'s pad-last-row fallback for the gap region.

**`models/prophet_model.py` — accept an explicit anchor (already returns timestamps)**
- `predict_prophet` (`:136-140`): add `start_ts: pd.Timestamp | None = None`. When given: compute `gap` from `model.history["ds"].max()` → `start_ts`; call `make_future_dataframe(periods=gap+periods, freq="h")`; after `predict`, select the `periods` rows with `ds ≥ start_ts` (tz-normalized) instead of blind `.tail(periods)` (`:209`). The existing `"timestamps"` key (`:217`) becomes authoritative. New kwarg defaults to `None` → current behavior, so no churn for existing callers.

**`jobs/phases.py` — stop discarding timestamps; join-write by timestamp**
- `_predict_one` (`:749-798`): change return to `tuple[preds, timestamps] | None`. Pass `start_ts=forecast_start` into both predicts. Prophet branch (`:776-778`) → `(result["forecast"], result["timestamps"])`. ARIMA branch (`:784`) → read dict `(res["forecast"], res["timestamps"])`. **XGBoost branch (`:770`) → `(preds, future_ts.to_numpy())` pass-through — XGBoost math is untouched.**
- `predict_and_write_forecast` (`:877-943`): store `predictions_by_model[name] = (preds, ts)`. **Fast path:** since all three now re-anchor to the same `future_ts`, assert each returned `timestamps == future_ts` and write positionally as today. **Guard path:** if they differ, write `row[name]` only where a model has a prediction at `future_ts.iloc[i]`; otherwise omit the key. Recompute the ensemble (`:892-921`) over only models present at each hour (align by timestamp before `ensemble_combine`); `primary` selection (`:925-930`) stays XGBoost-first, so the hero's primary line is safe even if a base model is dropped.

### Consumers corrected
- **Drift records (highest impact)** — `write_drift_metrics` (`:1029`) → `build_records_from_actuals` (`models/drift.py:317-363`) now pairs each `row[model]` with the actual at the *matching* hour; prophet/arima/ensemble drift MAPE becomes true 1h-ahead error. XGBoost drift unchanged.
- **Overview hero** (`components/_callbacks_overview.py:127-128`) and **Forecast-tab per-model traces** (`_callbacks_forecast.py:753-754`) — Prophet/ARIMA and the ensemble blend re-phase to the correct diurnal hour (user-visible correction).
- **Not affected:** the backtest path (`phases.py:1128-1146`, XGBoost-only, refit-then-predict with `train_end == anchor`); all ~10 refit-then-predict callers in `models/training.py`, `jobs/training_job.py`, `components/_callbacks_forecast.py`, `models/model_service.py`, `scripts/backtest.py`, `simulation/scenario_engine.py`, `components/_callbacks_backtest.py` — all call with `start_ts=None` → today's anchor is correct. Because `predict_arima` keeps the array return when `start_ts is None`, these need **no edits**.

### Tests (`tests/unit/test_scoring_time_alignment.py`, pure/mocked)
1. **Core regression:** fake Prophet `history["ds"].max()=T_train`, lean ARIMA `train_end=T_train`, `forecast_start=T_train+20h`; mock `predict` to return `yhat=f(ds)` (known monotone fn); assert `row["prophet"] == f(row["timestamp"])` and same for ARIMA — value matches its row's label for every model key present.
2. **Zero-gap invariance:** `forecast_start == T_train+1h` → output byte-identical to pre-fix positional path.
3. **Coverage guard:** mock a model returning too few steps to reach `forecast_start`; assert its key is **absent** from rows, phase still `ok=True`, ensemble recomputed without it.
4. **`predict_prophet` unit:** `start_ts=T+10h` → `timestamps[0]==T+10h`, length==`periods`.
5. **`predict_arima` unit:** `train_end=T`, `start_ts=T+10h`, monkeypatch `SARIMAX.forecast`→`arange(steps)`; assert returned == `arange(gap, gap+periods)`, `timestamps[0]==T+10h`; legacy payload (no `train_end`) → `gap=0`, warns.
6. **Drift end-to-end (integration):** two-tick sim; assert prophet/arima drift pairs each prediction with the matching-hour actual.

---

## 3. Fix 2: #195 holdout commensurability

### Recommended approach
**Score XGBoost's holdout with the same recursive-autoregressive protocol production uses, via one shared helper, keeping observed weather for all three.** Commensurability first; weather-realism is a separate follow-up (switching to forecast-weather would move all three models' numbers at once and entangle two problems). Extract a shared helper rather than calling `train_all_models` wholesale (which re-trains all three on its own split and returns a heavy dict) — lifting just the recursive-scoring loop is far less blast radius and simultaneously advances #186's structural-parity goal.

### Exact code changes

**`data/feature_engineering.py` — new shared helper** (co-located with `compute_autoregressive_snapshot` to avoid an import cycle; `predict_fn` injected):
```python
def recursive_autoregressive_forecast(model, seed_demand, future_df, predict_fn) -> np.ndarray:
    history = [float(v) for v in seed_demand if v is not None and not pd.isna(v) and v > 0]  # reuse prod zero/NaN filter (#129)
    preds = []
    for i in range(len(future_df)):
        row = future_df.iloc[[i]].copy()
        for col, val in compute_autoregressive_snapshot(history).items():
            if col in row.columns:
                row[col] = val
        row = row.ffill().bfill().fillna(0)
        p = float(predict_fn(model, row)[0])
        preds.append(p); history.append(p)
    return np.asarray(preds, dtype=float)
```
Then refactor `jobs/phases._predict_xgboost_with_recursive_autoregressive`'s recursive zone (`:722-735`) and `models/training.py:112-122` to call it — single source of truth (#186 lever).

**`jobs/training_job._holdout_metrics_xgboost` (`:44-82`)** — regenerate `val_df`'s autoregressive columns recursively instead of reading leaked in-window actuals:
```python
train_df = featured_df.iloc[:-_HOLDOUT_HOURS]
val_df   = featured_df.iloc[-_HOLDOUT_HOURS:]
holdout_model = train_xgboost(train_df, n_splits=3)
seed = train_df["demand_mw"].tolist()   # history strictly before the window
forecast = recursive_autoregressive_forecast(holdout_model, seed, val_df, predict_xgboost)[:len(val_df)]
y_val = np.asarray(val_df["demand_mw"].values, dtype=float)
```
Downstream (`compute_all_metrics`, the `{metrics, forecast, actual}` return shape, finite/positive guards) is unchanged, so `_ensemble_holdout_metrics` (`:230-233`), `_train_xgboost` (`:277`), and the ensemble-persist block need **no edits**. **Do not touch** `add_autoregressive_demand_features` — backward-looking features on real history is correct for *training*; the bug is only in *evaluation*. Also correct the in-code comment at `phases.py:688-691` that claims parity with "the training holdout" — true of dead `train_all_models`, false of the persisted `training_job` path.

### Numbers that move
- **XGBoost per-BA holdout MAPE rises** (multi-step error accumulation replaces near-perfect 1h-ahead) — median plausibly from ~2.2–2.3% toward 3–6%, worse on the volatile tail.
- **Ensemble weights shift toward Prophet/ARIMA** but XGBoost likely still dominates (Prophet/ARIMA run 3–5× worse per `BACKTEST_RESULTS.md`). The ensemble may now beat best-base on **more than 4/51 BAs** — the substantive interaction with #181.
- **Models tab** live `model_metrics` (Redis) — MAPE/RMSE/MAE/R² jump; must move in lockstep with the doc tables.

### Tests
1. **Protocol assertion (recursive, not observed lags):** perturb `val_df["demand_mw"]` in-window → forecast **unchanged**; perturb seed history → forecast **changes**. The exact inverse of current behavior.
2. **Monotonicity:** `mape_recursive ≥ mape_teacher_forced − ε` on a controlled fixture.
3. **Helper parity (#186):** `recursive_autoregressive_forecast` output == `phases._predict_xgboost_with_recursive_autoregressive` recursive zone on identical seed/future_df/model.
4. **Ensemble-path regression:** `_holdout_metrics_xgboost` still returns `{metrics, forecast, actual}`; XGBoost `actual` array equals Prophet/ARIMA `actual` (shared window, already asserted at `training_job.py:220-228`).
5. **Integration:** persisted `meta.extra["holdout_metrics"]["mape"]` for XGBoost is the recursive value.

---

## 4. Sequencing & PRs

**Two PRs — separable code paths (#194 = forecast write/model predict; #195 = holdout scoring). Coupling them makes the diff and the re-measurement story hard to reason about.**

### PR 1 — #194 (first)
- **Why first:** changes what the product serves (higher integrity), and must land before any #170 drift re-measurement is trustworthy. No dependency on #195.
- **Validation gate:** (a) unit/regression per §2 — per-model prediction ts == row ts, no NaN rows, ensemble finite when a model is flagged out; (b) run the scoring job against staging Redis for a handful of BAs, diff served Prophet/ARIMA rows before/after to confirm the diurnal cycle re-phases; (c) render Overview + Forecast tabs against the new payload (Preview) — no gaps, no backwards bridges; (d) open a drift re-measurement window (watch `rolling_smape_7d` for LDWP/AZPS over ~7 days or backfill).
- **Rollback:** pure code revert. Web tier is stateless; the next scoring tick overwrites `gridpulse:forecast:*` with reverted logic. No model artifacts change, `latest.json` untouched. One revert + one tick.

### PR 2 — #195 (second, after #194 alignment confirmed)
- **Why second:** its payoff is re-measuring #181 on a commensurable basis; you want served forecasts already aligned so recomputed weights blend correctly-timed members.
- **Validation gate:** (a) recursive holdout yields finite MAPE for all 51 BAs (reuse #176 self-heal diagnostics); (b) before/after holdout table side-by-side — confirm XGBoost MAPE rises and weights shift as predicted (log **both** recursive and teacher-forced MAPE for one release so the shift is observable before it drives gates); (c) training-job wall-clock under task-timeout on a full 51-BA run (recursive = 168-iter Python loop × 51 BAs; measure a few BAs first); (d) **deploy → one training run with `force=True` → re-run `scripts/export_holdout_metrics.py`** (pure metadata reader; **must run post-deploy with `ENVIRONMENT=production` — requires prod GCS ADC, not safe locally**), then land doc edits (§5) same PR or immediate fast-follow.
- **Rollback (asymmetric — the headline risk):** code revert stops persisting recursive holdouts, but weights/metrics already written to GCS `meta.json` are pointed to by `latest.json`. To fully roll back the served ensemble, re-run training on the reverted image (repoints `latest.json` atomically) or re-point `latest.json` to the prior version. #195's effect is **latched into GCS artifacts**, unlike #194.
- **Force-run caveat:** the resume short-circuit (`_skip_if_data_hash_matches`) keys on `data_hash`, not code version — so unchanged-data BAs would keep stale teacher-forced weights. A one-time `force=True` run (or a code-version stamp in the hash) is required on deploy.

**Scripts note:** `scripts/backtest.py` / `backtest_all.py` are **not** the doc source of record (separate 21-day from-scratch methodology; `backtest.py:158` is itself teacher-forced). Don't re-run them to refresh docs — the doc source is `export_holdout_metrics.py` surfacing `meta.json`. File their teacher-forced bug as a follow-up.

---

## 5. Doc-update checklist (same-PR per CLAUDE.md §2/§3)

Every number below derives from the contaminated holdout and moves once #195 lands + training re-runs. These are **#195-PR** edits (or an immediate fast-follow once re-measured numbers exist):

- `docs/BACKTEST_RESULTS.md:42-49` — min/median/mean/p90/max distribution table (XGBoost 2.32%/3.79%, best-base 2.30%/3.61%, ensemble 3.48%/4.92%).
- `docs/BACKTEST_RESULTS.md:51-59` — "ensemble trails best base… beats XGBoost on only 4/51" + the "for what production serves by default, read the ensemble column" claim (review flags this as unsupported).
- `docs/BACKTEST_RESULTS.md:61-69` — worst-5, best-base 48/51/ARIMA-2/Prophet-1 counts.
- `docs/BACKTEST_RESULTS.md:77-129` — full 51-row per-BA MAPE table (regenerate via exporter).
- `docs/BACKTEST_RESULTS.md:26-33, 141-143` — Methodology + "No-leakage validation": state XGBoost is now recursive multi-step; add "pre-#195 numbers were teacher-forced one-step and not comparable."
- `docs/_holdout_table.md:1-54` — entire table (regenerate, verbatim exporter output).
- `docs/CANONICAL_FACTS.md:73-90` — mirrored distribution table + "48/51 / 4/51 / ensemble trails best-base / AZPS 33.97%→27.40%" narrative.
- `docs/CANONICAL_FACTS.md:101-102` — FPL ensemble-weights example `{xgboost:0.578, prophet:0.293, arima:0.130}` (weights shift after recompute).
- `README.md:36` — "Best base model for 48 of 51 BAs; reference run 0.98% MAPE on ERCOT (168h holdout)".
- `README.md:41` — "Real holdout metrics… last 168-hour holdout" — add recursive-vs-teacher-forced clarification.
- `docs/HOW_IT_WORKS.md` / `TECHNICAL_SPEC.md §2.x` — any holdout-methodology prose (grep "168" / "teacher" / "one-step").
- `jobs/phases.py:688-691` — in-code comment claiming inference "matches the training holdout" (wrong for the persisted path) — fix in the #195 PR.
- `docs/INTERVIEW_PREP.md:81-95` — the drift/ensemble STAR story (both fixes; see §6).
- `docs/INTERVIEW_PREP.md:95` — the "LDWP ARIMA 188% MAPE reflects genuine model weakness" line — revisit after #194 re-measurement.
- `STATUS.md:51-52` — already lists workstream C; update active-focus/decisions when each PR lands (CLAUDE.md end-of-PR check #4).

---

## 6. Post-merge re-measurement (what this unblocks)

- **#170 (drift observability + "genuine ARIMA weakness")** — the observability half (log the ensemble headline, not `sorted(models)[0]` at `phases.py:1088-1090`) is orthogonal and can proceed anytime. The *interpretive* conclusion (`INTERVIEW_PREP.md:95`) is contaminated by **#194**: ARIMA's forward series was scored against an actual up to ~23h off. Repro beat: **ARIMA MAE 0.0 MW at the true origin vs 379.8 MW as-labeled.** After #194, re-read `rolling_smape_7d`/`rolling_mape_7d` for LDWP/AZPS — but the drift window is 7–30 days of hourly points, so **the number decays over the window, it does not correct at merge**; don't read a mid-transition number as the verdict. If ARIMA/Prophet drift drops sharply while XGBoost holds, #194 was the cause; if it stays extreme, #170's "weakness" reading survives.
- **#181 (inverse-MAPE ensemble trails best-base; XGBoost wins 48/51)** — rests on the *holdout* table, contaminated by **#195, not #194**. After #195 rescores XGBoost recursively: re-run training → `export_holdout_metrics.py` → recompute the distribution; produce the **before/after per-BA table on a commensurable basis**. If XGBoost still wins ≥N BAs, #181 stands with honest magnitudes; if it flips, its premise dissolves. **Do not close #181 — it becomes a re-evaluation;** add "supersede prior analysis after #195" to the issue. #195 must land before #181 is re-analyzed.
- **INTERVIEW_PREP capture (same PRs):**
  - The money quote: "XGBoost's 168h holdout was 168 teacher-forced one-hour-ahead predictions while Prophet/ARIMA were genuine 168-step forecasts — and all three saw actual weather while production serves forecast/climatology weather — so the 1/MAPE weights and the headline table compared apples to oranges."
  - The before/after distribution table (old teacher-forced vs new recursive) — the single most persuasive artifact.
  - The #181 resolution (survived / narrowed / reversed) with honest magnitude.
  - The #194/#170 correction: a chunk of the "extreme live ARIMA drift = genuine weakness" reading was a **wrong-hour comparison artifact** (train-origin values labeled with scoring-tick timestamps rotated the diurnal cycle up to ~23h) — the 0.0 vs 379.8 MW repro. Sharpens the existing "a verification pass should find a more precise truth" lesson.
  - The seam thesis (trade-off/recovery framing per the user's incident-PR preference): the leaf components were honest — pure metric functions, clean drift math; the defect lived at the *seam* where they were wired. **A metric is only as honest as the measurement protocol feeding it; two "MAPE" columns under different protocols aren't comparable no matter how clean each function is.**
  - The rollback asymmetry as a production-readiness beat: #194 self-heals on the next tick; #195 latches weights into GCS behind `latest.json`.

---

## 7. Open questions / risks (human decision before coding)

1. **`train_end` provenance / off-by-one.** Must be captured at training time and match Prophet's `history` end and the demand frame's clock (all UTC). If training's last-timestamp convention differs from scoring's `last_real_demand`, `gap` is off by 1h. Pin down with an explicit test asserting `predict_arima`/`predict_prophet` return `timestamps[0] == forecast_start`. **[#194 blocker]**
2. **ARIMA gap exog — confirm approach (a).** Build exog from `featured.tail(gap)` + `future_df` (real historical exog for the now-past gap, forecast ahead) rather than pad-last-row. Verify D=1 seasonal differencing isn't sensitive to a garbage lead-in even though gap steps are sliced off. **[#194 main subtlety]**
3. **Legacy-pickle window.** For one daily cycle post-#194-deploy, old ARIMA pickles lack `train_end` → `gap=0` keeps today's buggy behavior for those BAs until the 04:00 retrain. Prophet corrects immediately (live `model.history`). **Report the fix as fully effective only after one training cycle.** Human decision: accept the one-cycle window or force an off-cycle retrain.
4. **Run-to-run variance amplification (#195).** Recursive scoring compounds error over 168 steps; a single unsmoothed 168h window already feeds both weights and the 22% region-visibility gate (`model_service.py:394`; AZPS documented 11.9%→33.97%). Recursive scoring can flip a BA across the gate run-to-run. **Recommend scoping the smoothing fix (rolling/multi-window holdout, P2-17/#181) as a fast-follow, not bundled;** at minimum log both MAPEs for one release. Human decision: accept variance for now vs. block #195 on smoothing.
5. **Training-job wall-clock headroom.** Recursive holdout adds a 168-iter Python loop × 51 BAs. Measure on a few BAs before rolling to all 51; confirm against the task timeout with the circuit-breaker/outage budget in mind. **[#195 gate]**
6. **Force-run policy on #195 deploy.** `_skip_if_data_hash_matches` keys on `data_hash`, not code version → unchanged-data BAs keep stale weights. Decide: one-time `force=True` run vs. adding a code-version stamp to the resume hash (durable, prevents recurrence).
7. **Weather-optimism scope (explicit non-goal).** All three holdouts still score against observed weather vs production's forecast/climatology. #195 makes the three commensurable with *each other* but not with production. File as a follow-up; do not silently fix in either PR.

**Investigator disagreement resolved:** A left the `predict_arima` return shape ambiguous ("keep array when `start_ts is None`" vs "dict everywhere"). **Adopted: dict only on the anchored path, array when `start_ts is None`** — this keeps all refit-then-predict callers (including #195's `_holdout_metrics_xgboost`, which calls `predict_xgboost`, not `predict_arima`, so it's unaffected either way) edit-free and is the minimal-blast-radius choice both A and C favor.