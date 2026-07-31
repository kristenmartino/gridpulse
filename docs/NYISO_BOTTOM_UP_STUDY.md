# Bottom-up beats top-down on NYISO — and the gain splits in half

**The first decisive, shipping result in this line of work.** Eleven zonal
models with zonal weather, summed, beat one BA-level model by **+0.729 WAPE
pts — 100% of 6 windows, 3.7× stderr, satisficing clean.**

It also lands where [`NYISO_ZONAL_PROBE.md`](NYISO_ZONAL_PROBE.md) predicted:
that probe isolated a 0.5–0.8 pt effect from zonal weather diversity, and the
experiment delivered 0.729.

Run 2026-07-31, 4 months, 6 rolling windows, 168h horizon.
Reproduce: `python -m scripts.nyiso_bottom_up_study --months 4 --windows 6`.
Raw: [`NYISO_BOTTOM_UP_STUDY.json`](NYISO_BOTTOM_UP_STUDY.json).

---

## Result

| arm | WAPE | MAPE | bias | gain vs top-down |
|---|---:|---:|---:|---:|
| **top-down** (1 model, BA weather) | 3.958 | 3.798 | −0.003% | — |
| bottom-up, **BA** weather (ablation) | 3.609 | — | — | **+0.349** |
| **bottom-up, zonal weather** | **3.229** | 3.124 | +0.523% | **+0.729** |

Verdict: `decisive`, winner `treatment`, 100% sign consistency, t = 3.74,
worst window +0.194, best +1.332. MDE at 6 windows is 0.390 and the effect is
0.729, so this one is **detectable** — unlike most per-BA results in the #230
fleet run.

## The attribution matters more than the headline

Bottom-up gains two things at once: eleven load histories *and* eleven weather
points. The probe's hypothesis was specifically about weather. Without an
ablation the win could not be attributed, and the recommendation differs
sharply — zonal weather means 12 archive calls per BA per run, zonal load
alone is just the ISO feed.

**The split is almost exactly even:**

* zonal **load** decomposition alone: **+0.349 pts** (48% of the gain), itself
  decisive at ~3.9× stderr
* adding zonal **weather**: **+0.380 pts** more (52%)

Both ingredients are real and neither dominates. That makes a staged path
possible rather than all-or-nothing.

## The caveat that governs any adoption

**The target here is the sum of NYISO's zones, not EIA's `D`.** That was a
deliberate design choice, made before the run, because the two quantities
differ by **2.70% WAPE hour-by-hour** — means agree (ratio 1.0003) but the
hourly ratio ranges 0.94 to 1.07.

Our top-down error on NYISO is ~4%, so that definitional gap is more than half
the error budget. Scoring bottom-up against `D` would have charged it a ~2.7%
floor it cannot control, measuring a data-definition mismatch rather than the
forecasting question.

**Production forecasts `D`.** So this result does not transfer directly: a
real adoption has to either reconcile the zone sum to `D`, or accept that the
published benchmark — scored against EIA settled `D` — moves for definitional
reasons rather than accuracy ones. That reconciliation is unsolved here and is
the first thing any implementation must address.

## Recommendation

**A staged path, not a rewrite, and not seven integrations.**

1. **Reconcile first.** Until the 2.70% zone-sum-vs-`D` gap is understood, no
   amount of forecasting gain is bankable. This is a data question, cheap, and
   strictly prior to everything else.
2. **Then zonal load alone** for the ISO BAs — half the gain, one new data
   source per ISO, no extra weather cost.
3. **Then zonal weather**, if step 2 holds up in production, for the rest.

Coverage stays ~7 of 51 BAs (though those carry ~62% of fleet MW error), and
this is **one BA, one season**. Before building, the same experiment should run
on at least PJM and ISONE — the #230 fleet run is a standing reminder that a
result on a handful of BAs changed character at 51.

## Limits

1. **One BA, one season** (4 months, summer-weighted), 6 windows.
2. **Target is the zone sum, not EIA `D`** — see the caveat above; this is the
   binding limitation on adoption, not a footnote.
3. **XGBoost day-ahead**, not the served recursive ensemble.
4. **Coarse zone centres** — approximate load centres, not real centroids, so
   the zonal-weather half of the gain is if anything understated.
5. Bottom-up bias is +0.523% against top-down's −0.003%. Inside the ±2% band
   and satisficing passed, but it is a real drift toward over-forecasting that
   would want watching at fleet scale.
6. Eleven models per BA per run is 11× the training cost, against a ~0.7 pt
   gain. Not free, and unmeasured here in wall-clock terms.
