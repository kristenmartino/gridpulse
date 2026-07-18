# Forecast-dive diagnosis — the training-day fit lottery

**Date**: 2026-07-18 · **Instrument**: `scripts/forecast_dive_diagnosis.py` ·
**Subject**: LDWP XGBoost, generalizes fleet-wide

## The incident

The live LDWP XGBoost forecast scored 2026-07-18 02:11Z launched sanely off
the conditioned anchor (4,005 MW) then dove to **1,302 MW by 07Z** — deep in
EIA's partial band, ~50% under any settled overnight trough — and ran the
whole first day at ~70% of reality. Every input was verified clean before
this study ran: settled history (hour-by-hour guard-log proof), sane
mirrored weather (LA July night, 68–75 °F, zero NaNs, 381 h forecast
coverage), gap-free 90-day EIA view, conditioned day-ahead anchor.
Live 7d sMAPE 24% vs holdout 8.8%.

## The ladder (one dimension per rung)

| rung | question | result |
|---|---|---|
| 1 | Is the model miscalibrated on real rows? | **No** — teacher-forced 1h-ahead on settled featured rows: **0.53% MAPE** (overnight 0.44%) |
| 2 | Does the serve-style future frame corrupt features? | **No** — holdout-style slice vs `_build_future_feature_frame` over the same window: 1.97% vs 1.94%, only `temperature_deviation` differs (0.13 sd) |
| 4 | Does anchor conditioning contribute? | **No** — settled / conditioned-3h / single-hour seeds: 1.94 / 1.91 / 1.94% |
| 0b | Which pickle reproduces the live dive? | **Exactly one.** On the identical rebuilt tick frame: 0713→1,903 trough · 0714→2,272 · 0715→2,669 · 0716→3,092 · **0717→1,286 (matches prod 1,302 within 4.3% across the full 24h curve)** · 0718→2,634 |
| 3 | Is 0717 uniquely bad? | **No — it's a lottery.** All 67 persisted vintages replayed over one fixed window (Jul 16 01Z + 48h): **18/67 (27%) dive** (trough < 75% of settled truth), in runs — May 19–22, Jun 12–13, Jun 16–17, Jun 22–24, **Jul 6–10 (troughs 461–928 MW)** — separated by stretches of sane fits |
| 3b | Does anything at training time flag a diver? | **Nothing available.** Holdout MAPE of divers: 1.79–8.99% — indistinguishable from sane vintages. Trailing-partial contamination before training: zero for the Jun/Jul divers, **8 for the sane 0718**. `train_rows` constant |

Control: PNM through rungs 0–2 — no dive, frames identical (harness sound).

## Verdict

**Per-training-day fit instability, expressed only in the recursive serve
regime.** Each daily retrain is an independent draw; roughly a quarter of
draws produce a model whose recursive extrapolation collapses overnight
demand into a phantom regime — condition-dependent (0717 dives on the
Jul-18 frame but not the Jul-16 window; the Jul 6–10 draws the reverse), so
any single backtest window undercounts. The recursion feeds the model its
own predictions as lag features, walking it into feature territory no
training row covers; what a boosted ensemble does out there is decided by
fit noise, and some fits extrapolate hard toward zero.

Nothing in the training job can currently see this: the published holdout
scores a *freshly retrained* model on a sliced historical frame (observed
weather, real AR features, one fixed seed date) — **it never runs the
deployed pickle through the serve path**, and rung 3b shows its MAPE
carries zero signal about diving.

Hypotheses killed by this study: serve-frame feature corruption (rung 2),
weather values/NaN handling (pre-study verification + rung 2), anchor
conditioning (rung 4), train/serve AR-semantics skew (code parity diff +
rung 5 per-step forensics: recursion rows match settled rows to <1%),
training-data partial contamination as the dive driver (rung 3b),
historical partial-band rows in EIA's settled view (90-day audit: none).

## Implication — the fix

1. **Persist-time acceptance gate** (the durable fix): after each retrain,
   replay the *candidate pickle* through the real serve path
   (`_build_future_feature_frame` + `recursive_autoregressive_forecast`) on
   the current frame and refuse the `latest.json` repoint when the curve is
   degenerate (trough vs recent settled troughs, level-ratio bounds). A bad
   draw is discarded; yesterday's accepted model keeps serving — stale-but-
   sane over fresh-but-insane, the same principle as the data-fallback
   policy. Rung 0b is the gate's exact prototype.
2. **Close the holdout blindness**: score the published holdout through the
   serve-style frame so the metric class that hid this defect can't hide
   the next one.
3. Training-frame quality guard stays worthwhile as hygiene (holdout ground
   truth should never include unsettled partials) but is explicitly **not**
   the dive's cause.

## Reproduction

```bash
GCS_ENABLED=true GCS_BUCKET_NAME=nextera-portfolio-energy-cache \
  .venv/bin/python scripts/forecast_dive_diagnosis.py
```

Reads only GCS mirrors (vintage, weather, model store) via ADC. The live
curve it must reproduce is pinned in `PROD_CURVE_0718` with provenance.
