# Cooling-response features: measured, and rejected

**Verdict: the feature pack does nothing.** 8 of 8 BAs inconclusive, mean
effect **−0.0033 WAPE pts**, 6 of 8 pointing slightly the wrong way. The flag
`cooling_response_features` stays **off**.

Run 2026-07-30. Reproduce: `python -m scripts.cooling_response_study --windows 6`.
Raw results: [`COOLING_RESPONSE_STUDY.json`](COOLING_RESPONSE_STUDY.json).

---

## Why it was worth trying

[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) measured the hottest temperature
quintile carrying a mean **34.7%** of our forecast error against **11.9%** for
the coldest, monotone in 7 of 8 BAs. The existing representation of cooling
load is a single linear `cooling_degree_days` against a fixed 65°F baseline,
which cannot express three things cooling load actually does:

* **accumulate** — the third consecutive 95°F day draws more than the first,
  because structures carry heat overnight (`cdd_accum_24h`, `cdd_accum_72h`);
* **curve** — load rises faster than linearly once plant nears capacity
  (`cdd_squared`);
* **respond to humidity** — 95°F at 70% RH is not 95°F at 20% (`heat_index`
  via the NWS Rothfusz regression, and `cdd_x_humidity`).

All five are built from weather variables we already fetch, so nothing here
touches the fetch path.

## What the measurement said

Two arms, identical except the flag. Six rolling windows per BA through
`models/rolling_eval.py`; both arms forecast day-ahead with known future
weather, so the comparison isolates the features.

| BA | control WAPE | treatment WAPE | Δ pts | sign consistency | worst window | best window |
|---|---:|---:|---:|---:|---:|---:|
| ISONE | 4.221 | **3.893** | **+0.328** | 83% | −0.15 | +1.15 |
| FPL | 2.599 | **2.519** | +0.080 | 50% | −0.13 | +0.51 |
| PJM | 2.836 | 2.881 | −0.045 | 67% | −0.21 | +0.22 |
| TVA | 2.577 | 2.625 | −0.048 | 67% | −0.45 | +0.29 |
| SOCO | 2.432 | 2.489 | −0.056 | 67% | −0.18 | +0.15 |
| MISO | 2.798 | 2.878 | −0.080 | 67% | −1.06 | +0.68 |
| NYISO | 3.669 | 3.767 | −0.098 | 67% | −0.96 | +0.59 |
| ERCOT | 1.729 | 1.835 | −0.106 | 67% | −0.23 | +0.01 |

**Mean across BAs: −0.0033 pts.** Every row is `inconclusive` under
[`EVALUATION_POLICY.md`](EVALUATION_POLICY.md) — none clears 2× stderr, and
none reaches 75% sign consistency. Nothing was vetoed by the satisficing
constraints, because nothing got far enough to be vetoed.

Worth noting explicitly: **ISONE is the only row with a real-looking effect**
(+0.328, 83% consistent), and ISONE is exactly where the error analysis
predicted the largest payoff — highest hot-quintile concentration (41.7%) and
highest addressable share (58.7%). It still does not clear the bar, and
picking it out of eight would be exactly the cherry-pick the harness exists to
prevent. It is a follow-up, not a result.

## What this actually tells us — the useful part

The error analysis was **right about where** the error is and **wrong about
what it is**. The hot-hour error is not a temperature-representation problem.

The strongest evidence is what the arms were given: both had **perfect future
weather** from the ERA5 archive. If explicit accumulation, convexity and
humidity terms cannot reduce hot-hour error when the temperature is known
exactly, then the missing ingredient is not a better function of temperature.

Two hypotheses survive that, neither tested here:

1. **Behind-the-meter solar.** Hot afternoons are also peak-irradiance
   afternoons. Rooftop PV suppresses *net* load exactly when cooling load
   peaks, its penetration grows quarterly, and nothing in the feature set
   knows it exists — `solar_capacity_factor` is derived from irradiance, not
   from installed capacity. This would look precisely like unexplained
   hot-afternoon error that better temperature features cannot touch.
2. **Demand response and price-driven curtailment.** Large loads shed at peak
   under programmes that fire on conditions the model never sees.

A third, more prosaic possibility deserves stating: gradient-boosted trees
approximate interactions natively given enough data, so explicit interaction
terms may simply be redundant with what the model already learns from
`temperature_2m`, `relative_humidity_2m` and `cooling_degree_days`. If that is
the whole story, the lesson is about feature engineering for tree models
generally, not about cooling.

## What shipped

Nothing, to the served path. The flag is off and the features are absent from
production frames.

The code is **kept, dormant and documented**, rather than deleted, for two
reasons: the ISONE signal deserves a longer run, and the error analysis was a
**summer** window, so its own re-run in January is already outstanding — at
which point the same machinery answers the heating-side question without being
rebuilt. `tests/unit/test_cooling_response_features.py` pins the pack's
correctness (NWS spot values, backward-only accumulation) and its two
invariants: weather-only, so a day-ahead forecast can have it, and no NaN
introduced into the training frame.

## Limits

1. **Six windows.** ISONE's +0.328 at 83% consistency might clear the bar with
   12; the policy's thresholds are deliberately conservative and this is the
   case where that costs something.
2. **XGBoost only**, not the served ensemble — the same proxy as
   `ERROR_ANALYSIS.md`, credible there because CV tracked production's live
   numbers, but still a proxy. Prophet and SARIMAX consume features
   differently and might use these terms where trees do not.
3. **Summer only**, June–July 2026. This says nothing about heating response.
4. **Eight BAs**, chosen as 82% of the addressable gap.
5. **The pack was tested together**, so a single useful feature could be
   masked by four useless ones. An ablation was not run because the combined
   arm did not clear the bar — there was nothing to attribute.
