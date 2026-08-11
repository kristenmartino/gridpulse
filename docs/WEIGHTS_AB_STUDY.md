# Ensemble-weight smoothing — pre-registered A/B (#451)

**Question.** ADR-004 weights each model by `(1/MAPE)³`, where MAPE is the single
most recent 168-hour holdout. That estimator flaps — median 12% run-to-run, p90
43% ([`HOLDOUT_STABILITY_STUDY.md`](HOLDOUT_STABILITY_STUDY.md)) — and the
resulting weights move 0.220 L1 per day, about 11% of the blend mass relocating
between models daily. **Do weights computed from a smoothed MAPE forecast
better?** Stability was never the objective; accuracy is.

**Answer.** Yes, decisively, at α=0.3 — and not for the reason anyone would
guess. Shipped **off** pending a fleet re-measure; see *Why the flag is off*.

Reproduce: `python scripts/weights_ab_study.py --end 2026-08-10`

---

## Design, pre-registered in #451 before any of it was run

| | |
|---|---|
| **Arms** | `raw` (control, today) · `ewma_0.3` · `ewma_0.5` |
| **Optimising metric** | WAPE. **Reported:** MAPE |
| **Windows** | 8 rolling origins × 168h, non-overlapping (2026-06-15 → 2026-08-10) |
| **Satisficing** | \|bias\| ≤ 2.0%, MAPE regression ≤ 0.5 pts; unmeasurable counts as failed |
| **Decisive** | ≥4 windows, \|mean\| ≥ 2×stderr, ≥75% sign consistency |
| **Ship criterion** | a decisive WAPE win with both constraints met. Anything else ships nothing |

12 BAs — the same set as the stability study (SPA, IID, AZPS, SEC, HST, PSCO,
ERCOT, PJM, MISO, FPL, CAISO, BPAT), deliberately weighted toward the hard tail
— giving 96 (BA, window) cases. Every verdict comes from
[`models/rolling_eval.py`](../models/rolling_eval.py).

## It needed no training runs

#451 priced this at ~3h of training per arm and parked it as a scheduled job.
That was wrong, and the reason matters: **the arms differ only in the weight
vector.** The per-model forecasts are identical across arms, so the expensive
half is paid once.

GCS holds every daily vintage back to 2026-04-19 (3,495 for these 12 BAs). For
each origin, the study loads the vintage that was *live at that moment* and
predicts the 168h forward window through the production serve path
(`recursive_autoregressive_forecast`). It is a **replay, not a retrain** — and
being forward-looking, it is the serve regime production actually operates in,
not the holdout regime. Total: ~15 minutes, cached thereafter.

Weights for each arm are computed only from vintages trained **strictly before**
the origin. An arm that could see the window it is scored on would be the
in-sample reasoning `ledger-23` was fixed for.

## Result

| Arm | mean Δ WAPE | t | windows won | decisive | bias | MAPE | ships |
|---|---|---|---|---|---|---|---|
| `raw` (control) | — | — | — | — | — | 13.07 | — |
| **`ewma_0.3`** | **+0.524** | 2.21 | **7 of 8** | ✅ | +0.08% | **12.54** | ✅ |
| `ewma_0.5` | +0.271 | 2.14 | 6 of 8 | ✅ | +0.01% | 12.78 | ✅ |

Positive Δ = smoothing better. Both satisficing constraints pass for both arms.
**α=0.3 wins by roughly twice as much as α=0.5**, so it is the value carried
forward. It was not tuned — sweeping α on this data would be choosing the
conclusion from what suggested it.

Per-BA: **10 of 12 favour `ewma_0.3`** on the mean (FPL +1.27, ERCOT +0.94, HST
+0.84 lead; only AZPS −0.13 and CAISO −0.22 are negative), and 66 of 96
individual cases.

## The confound, refuted — and backwards from the guess

Smoothing compresses the spread between models' MAPEs. Cubed, that *should*
flatten the blend, so the obvious hypothesis is that the gain is really "less
concentrated weights help" — an argument about **ADR-004's exponent**, not about
smoothing, with a completely different action attached.

Both halves of that hypothesis are false:

| Arm | concentration (HHI) | mean WAPE | Δ vs control |
|---|---|---|---|
| `raw`, k=3 (control) | 0.603 | 11.93 | — |
| `ewma_0.3`, k=3 | **0.617** | 11.40 | **+0.524** |
| `raw`, k=2.5 | 0.566 | 12.13 | −0.206 |
| `raw`, k=2.0 | 0.522 | 12.42 | −0.494 |
| `raw`, k=1.5 | 0.471 | 12.83 | −0.899 |
| `raw`, k=1.0 | 0.413 | 13.42 | −1.495 |

Smoothing makes the blend **more** concentrated, not less. And every flatter
exponent loses *decisively* to k=3, monotonically in how flat it is. Two
conclusions: the gain is not a concentration artifact, and **ADR-004's k=3 is
re-validated** as a side effect, independently of the #181 sweep that set it.

What is left as the mechanism is that an EWMA is a **better estimate of a model's
next-week quality** than one noisy draw, so the mass lands on the right model
more often. The dose-response supports it: gain correlates **+0.40** with how far
the weights actually moved (L1), averaging **+0.897** where they moved most
against **+0.152** where they barely moved. Noise would show no such gradient.

## Why the flag is off

The pre-registered criterion is met and is not being relitigated. The flag is off
because of a question the pre-registration did not cover — **scope** — plus a
fragility a reader is owed:

- **Measured on 12 of 51 BAs**, chosen for the hard tail. Flipping the flag moves
  served forecasts fleet-wide.
- **The verdict sits near its own threshold.** t = 2.21 against a 2.0 filter, and
  it moved 2.26 → 2.21 between two runs hours apart (EIA data settling).
- **Leave-one-out** (not a ship criterion — disclosure): `ewma_0.3` stops being
  decisive if **FPL alone** is dropped, or if any of 3 of the 8 windows is.
  `ewma_0.5` is worse — 2 BAs and 6 of 8 windows.

`update_smoothed_mape` persists the series **regardless of the flag**, so a later
flip finds real history instead of starting from a single observation. Flipping
is a one-line change plus a fleet re-measure.

## Limitations

- **Weather regressors are partly imputed on deep history.** The ERA5 archive
  endpoint lacks `wind_speed_80m/120m` and `soil_temperature_0cm` (documented in
  `data/weather_client`), where production's forward run had real forecast wind.
  Identical across arms, so the *paired* comparison holds — but the absolute
  error levels (13.07% control MAPE) are **not** production's ~4.35%, and this BA
  set is the hard tail besides.
- **Rolling origins share training data**, so the t-statistic is a decision rule,
  not a significance claim. `rolling_eval.verdict` says so itself.
- **Weights come from production's real holdout metas; predictions come from the
  replay.** The two halves are of different data quality. Again identical across
  arms.
