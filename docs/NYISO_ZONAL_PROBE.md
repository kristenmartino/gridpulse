# NYISO zonal structure: one prediction of three survives

**Verdict: the hypothesis is not dead, and not yet worth seven integrations.**
Zonal *weather diversity* predicts our BA-level error and survives a
temperature control. Zonal *load mix* does not, in either form tested.

NYISO, 985 joined hours, 11 zones, 6 rolling windows, 4 months of archive.
Reproduce: `python -m scripts.nyiso_zonal_probe --months 4`.
Raw: [`NYISO_ZONAL_PROBE.json`](NYISO_ZONAL_PROBE.json).

---

## Why this and not a connector

[`ISO_REALTIME_FEEDS.md`](ISO_REALTIME_FEEDS.md) killed the anchor case for ISO
ingestion — EIA's lag is 1.7h, and a 2h stale anchor costs +0.014 WAPE pts.
What survived was a different claim: EIA-930 publishes one number per BA where
NYISO publishes eleven zones, and sub-BA structure is information the model has
never had.

That claim was stated, not tested. This tests it before any connector exists —
the pattern that has been working, and the one the cooling pack violated.

## Prediction 1 — weather diversity. **Supported**

Our absolute residual against the temperature spread across the 11 zone
centres (range 3.9–26.4 °F):

| spread quintile | mean abs residual |
|---|---:|
| 1 (most uniform) | 3.367% |
| 2 | 3.352% |
| 3 | 3.656% |
| 4 | 3.268% |
| **5 (most divergent)** | **4.674%** |

`corr = +0.123`, which *understates* it — the relationship is a threshold, not
a line. Quintiles 1–4 are flat; only the top one moves, and it moves ~40%.

**It survives the control.** Spread and temperature level are correlated
(`corr = +0.257`), so this could have been temperature wearing a diversity
costume — the trap the BTM probe's prediction 2 fell into. Within temperature
bands, the high-spread minus low-spread effect is:

| band | effect |
|---|---:|
| cool | **+0.512 pts** |
| mid | +0.061 pts |
| hot | **+0.804 pts** |

The mid band showing nothing is what makes this credible rather than
suspicious: when temperatures are mild, load barely responds to weather, so it
should not matter that zones disagree. When the response is steep — cold or
hot — zones sitting at different points on that curve is exactly when a single
BA-level temperature misrepresents the aggregate.

The top-spread quintile carries **25.6% of total absolute error on 20.1% of
hours — 1.27×**.

## Prediction 2 — load mix. **Fails**

`corr(downstate share, signed residual) = +0.085`, and the quintile pattern is
non-monotonic (+0.61, −0.34, −2.10, −0.31, +1.77). No usable relationship.

## Prediction 3 — mix instability. **Fails**

`corr(mix departure from its 168h norm, absolute residual) = −0.008` — zero.
The quintiles are U-shaped (4.32, 3.50, 3.23, 3.04, 4.22), highest at *both*
extremes, which is not the predicted shape and is more consistent with the
extremes of any noisy series being unusual for other reasons.

## The caution this probe owes the reader

**A 1.27× error concentration is the same shape that produced two consecutive
failures.** `ERROR_ANALYSIS.md` found hot hours carrying 1.43× their share of
error; the cooling pack built to address it was inconclusive on 8 of 8 BAs,
and the BTM hypothesis that followed failed its own sign test.

Concentration says where error *is*. It does not say the error is
*addressable*, and this project now has two well-documented cases where it
wasn't. This finding is one BA, one season, 985 hours, and a coarser version
of the same argument.

## Recommendation

**Do not build seven ISO integrations.** The evidence supports exactly one
next step, and it is cheap:

> **Bottom-up vs top-down on NYISO alone.** Eleven zone-level models with
> zone-level weather, summed, against one BA-level model — the existing
> rolling-eval harness, one BA, no connector in production. If zonal weather
> diversity is really costing us, a bottom-up forecast should recover the
> 0.5–0.8 pts the control isolated.

That experiment is decisive, reuses machinery that already exists, and costs
one BA rather than seven. Only if it wins does connector work become
justified — and even then, note that zonal load is available for ~7 of 51 BAs
and would need zonal weather to match, since
`assets/multipoint_coordinates.json` is BA-level.

## Limits

1. **One BA, one season** (4 months, summer-weighted), 985 joined hours.
2. **Coarse zone centres** — approximate load centres, not real centroids.
   Adequate for a diversity test, not for a bottom-up forecast.
3. **XGBoost day-ahead proxy**, not the served recursive ensemble.
4. **Temperature only.** Humidity, irradiance and wind also vary zonally and
   are not in the spread measure.
5. The threshold shape is read off quintiles, not fitted; the exact breakpoint
   is not established.
6. Zonal load is aggregated 5-minute → hourly by mean, and NYISO's local
   timestamps are localized with `ambiguous="NaT"`, so DST-repeat hours are
   dropped rather than resolved.
