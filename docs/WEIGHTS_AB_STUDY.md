# Ensemble-weight smoothing — pre-registered A/B (#451)

**Question.** ADR-004 weights each model by `(1/MAPE)³`, where MAPE is the single
most recent 168-hour holdout. That estimator flaps — median 12% run-to-run, p90
43% ([`HOLDOUT_STABILITY_STUDY.md`](HOLDOUT_STABILITY_STUDY.md)) — and the
resulting weights move 0.220 L1 per day, about 11% of the blend mass relocating
between models daily. **Do weights computed from a smoothed MAPE forecast
better?** Stability was never the objective; accuracy is.

**Answer.** The WAPE win is real and survives every re-cut at 51 BAs. **But the
ship criterion is not met**, because the bias constraint cannot be evaluated in a
replay harness — the *control* arm breaches it by 3×. Per the pre-registered rule
(an unmeasurable constraint counts as failed), **nothing ships**. Details in
*The fleet re-measure*.

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

## The fleet re-measure — 51 BAs, 408 cases

The 12-BA cut above was the hard tail. Re-run across **all 51 BAs**, the two
halves of the result move in *opposite* directions, which is why the fleet run
was worth doing.

| | 12 BAs | 51 BAs |
|---|---|---|
| `ewma_0.3` Δ WAPE | +0.524 | **+0.355** |
| t | 2.21 | **2.62** |
| windows won | 7 of 8 | 7 of 8 |
| breaks if one BA is dropped | **FPL** | **none** |
| breaks if one window is dropped | **3 of 8** | **none** |
| treatment bias | +0.08% | **+6.01%** ❌ |
| **ships** | ✅ | ❌ |

**The WAPE win got more robust, not less.** At 51 BAs it survives dropping any
single BA and any single window — the fragility that qualified the 12-BA result
was a small-sample artifact.

**And the ship criterion still fails, on the constraint.** `|bias| ≤ 2.0%` is
breached at +6.01%.

### The bias belongs to the harness, not to smoothing

`satisficing_check` only ever sees the *treatment's* bias, so the number above
invites the reading "smoothing over-forecasts". It does not:

| arm | mean bias |
|---|---|
| `raw` (control) | **+6.042%** |
| `ewma_0.3` | +6.013% |
| `ewma_0.5` | +6.023% |

**Treatment minus control: −0.029 pts.** Smoothing very slightly *reduces* bias.
**26 of 51 BAs breach ±2% in the control arm alone** — TIDC +43%, PGE +31%,
ISONE +31%, NYISO +30%. This is a replay artifact: a vintage carried forward 168h
against partly-imputed weather over-forecasts, and the 12-BA subset simply
happened to average near zero.

That last part is a correction to this document's earlier reading. The 12-BA run
recorded bias +0.08% and passed the constraint — **by luck of the subset, not
because the harness could measure it.**

### So the honest verdict

**Ships nothing**, and for a stronger reason than "the number was too small":

> A replay whose control arm is +6% biased cannot certify that the treatment
> holds bias within ±2%. The constraint is **unmeasurable in this harness**, and
> `EVALUATION_POLICY.md` is explicit that an unmeasurable constraint counts as
> failed.

**This retroactively justifies #451's own instinct — for a reason the issue did
not state.** It called for real training runs per arm. That is not needed to
compare *weights* (the replay does that fine, and better). It **is** needed to
check the bias constraint, because only production-quality forecasts have
production-quality bias. The scheduled run was the right call about the wrong
half of the experiment.

### What would settle it

Arms measured on production-grade forecasts — either a shadow scoring pass that
writes both weightings, or the per-arm training runs #451 originally specified.
The WAPE half is already answered and would not need re-running.

## Why the flag is off

Because the ship criterion is not met. `update_smoothed_mape` persists the series
**regardless of the flag**, so whenever the bias half is settled the flip finds
real history rather than starting from a single observation.

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
