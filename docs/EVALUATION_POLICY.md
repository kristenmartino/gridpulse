# How GridPulse decides an experiment

**One optimising metric, two satisficing constraints, and a rolling origin.**
Implemented in [`models/rolling_eval.py`](../models/rolling_eval.py); every A/B
study in `scripts/` should route its verdict through it.

Written 2026-07-29, after an experiment shipped on numbers that turned out to
be noise.

---

## Why this exists

The ARIMA order study (`ARIMA_ORDER_EXOG_STUDY.md`) ran a single 168-hour
holdout on 2026-07-28, then again on 07-29 — same code, window one day apart.

| BA | Δ on 07-28 | Δ on 07-29 |
|---|---:|---:|
| CAISO | **−7.24** | **+3.87** |
| WALC | +4.59 | +0.04 |
| MISO | −2.90 | −3.86 (control error halved, 7.62 → 3.91) |

Re-run under this harness across **8 rolling windows**, CAISO's per-window
deltas are:

```
-0.43  +2.31  -0.20  0.00  -6.93  +4.30  +1.57  -0.30
```

Mean **+0.04**, median **−0.10** — a wash. The two numbers previously
published were **the two extremes of that distribution**. The decision they
supported was right for other reasons, but the evidence was noise, and nothing
in the process could have told us.

## 1. Rolling origin, never a single window

`rolling_origin_splits` produces walk-forward `(train, test)` slices, newest
first. Train always ends exactly where test begins, so no window sees its own
future. Default 8 windows × 168 h, non-overlapping.

If history runs short it returns **fewer windows and says so** rather than
padding — a short study must not be able to pass for a thorough one.

## 2. One optimising metric: **WAPE**

`Σ|error| / Σ|actual|`, as a percentage.

**Not MAPE**, and the reason is not taste. For one hour, `|A−F|/A` grows
without bound as the forecast exceeds actual but caps at 100% as it approaches
zero. A MAPE-minimising model is therefore biased toward **under-forecasting
demand** — the expensive direction for a grid, where under-procured reserves
mean scarcity risk. MAPE also explodes on low denominators (SEC is a ~300 MW
co-op with quiet overnight hours) and cannot be aggregated across BAs in any
way that reflects total MW error: 20% of SEC is 60 MW, 2% of PJM is ~2,000 MW.

WAPE has none of those properties: scale-robust, direction-neutral, peak hours
dominate (correct for grid operations), and it sums meaningfully across BAs.

**MAPE remains the published number.** Comparability with EIA, the ISOs and
every vendor scorecard is a genuine requirement — it just should not be the
thing we optimise. It is protected as a constraint instead.

## 3. Two satisficing constraints

A win on WAPE is vetoed unless both hold:

| Constraint | Default | Why |
|---|---|---|
| `\|bias_pct\|` ≤ `MAX_ABS_BIAS_PCT` | 2.0% | An arm must not buy a WAPE win with a systematic under-forecast. This is the guard MAPE lacks. |
| MAPE regression ≤ `MAX_MAPE_REGRESSION_PTS` | 0.5 pts | The published metric may not quietly degrade. |

An **unmeasurable** constraint counts as failed. An unchecked constraint is not
a satisfied one.

## 4. The verdict rule

`verdict(deltas)` returns `decisive` only when all of:

- **≥ 4 windows.** One window is what produced the CAISO reversal.
- **Magnitude** — `|mean| ≥ 2 × stderr`.
- **Sign consistency** — the winner wins ≥ 75% of windows.
- **Mean and median agree in sign.** This is the specific signature of outlier
  domination: one catastrophic window can drag a mean across zero while most
  windows disagree with it. Reported as tail risk, not laundered into an
  average. (The fleet ARIMA run had a BA at −19.18 against a typical −0.2.)

`worst_window` and `best_window` ship with every verdict, decisive or not — a
passing mean is not the whole story.

The t-statistic is a decision rule, **not a significance claim**: rolling
windows share training data and are not independent draws, so a nominal
p-value would be optimistic. It is used as a conservative filter.

## Worked example

[`EVAL_HARNESS_CAISO_DEMO.json`](EVAL_HARNESS_CAISO_DEMO.json) — the CAISO run
above. Verdict: **inconclusive**, `ship: false`, reason *"mean +0.041 and median
−0.099 disagree in sign — outlier window(s) dominate"*. The order changed in 7
of 8 windows, so this is a real difference in behaviour that nonetheless does
not add up to a real difference in accuracy.

**An incidental finding worth its own look:** across those 8 windows the
*control* arm — what production serves — under-forecasts CAISO by **−2.83%**
on average. That is the operationally dangerous direction, and it is what the
bias constraint exists to catch. It is consistent with the concern about
MAPE-shaped selection, though one BA does not establish the mechanism, and
AIC rather than MAPE drives order choice directly.

## Limits

1. Rolling windows overlap in training data — not independent samples.
2. Defaults (8 windows, 168 h, 2σ, 75%) are judgement calls, not derived.
3. The harness compares two arms on one BA. Fleet-level roll-up is the
   caller's job, and 51 single-BA verdicts are not one fleet verdict.
4. A WAPE-optimising experiment can disagree with the MAPE-based serving gate.
   That gap is real; the fix is migrating the gate, not changing the metric
   back.
