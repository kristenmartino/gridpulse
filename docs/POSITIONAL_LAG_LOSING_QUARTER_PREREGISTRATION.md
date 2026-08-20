# Pre-registration — #559, where the losing windows sit

**Committed before the analysis is written. Nothing below was chosen after
seeing a breakdown.**

## 0. Status: this analysis is EXPLORATORY and cannot confirm anything

It re-cuts a dataset that has already produced a verdict
([`POSITIONAL_LAG_INJECTION_RERUN_STUDY.md`](POSITIONAL_LAG_INJECTION_RERUN_STUDY.md)),
so any pattern it finds is a **hypothesis for a future confirmatory study on new
data**, never grounds for shipping a rule. Saying so first is the point: the
temptation with a result like "better on average, worse on a quarter of windows"
is to find the quarter, carve it out, and ship the remainder — which is fitting a
rule on the same data that suggested it.

Nothing in this document can turn `temporal_ar_seed` on.

## 1. What was already seen, and what was not

Honesty requires separating these, because one of the hypotheses below is
derived from data already inspected.

**Already seen** (published in the re-run study): per-BA mean WAPE for both arms
in both strata; pooled means, medians, MDEs, sign consistency; the paired
before/after swing. From that: **PSCO is the one BA still consistently worse
(−0.46)**, and PSCO is separately known — from the #537 origin work — to have
real gaps that are *clock-aligned* (six regressions at exactly 10:00 UTC on
consecutive days) rather than randomly placed.

**Not seen**: any breakdown of the paired deltas by `gap_len`, `gap_hour_utc` or
`gap_lead_h`. No per-window cut of any kind has been inspected.

So hypothesis H1 below is genuinely pre-specified against unseen data. **H3 is
post-hoc**, derived from an inspected per-BA result, and is labelled as such
wherever it is reported.

## 2. Hypotheses, fixed now

* **H1 — gap length.** Losses concentrate on long gaps. A 13-24h hole is filled
  by the seasonal branch of the absent-hour policy (same clock hour, previous
  day), which is a weaker estimate than interpolation across a 1h hole.
* **H2 — gap recency.** Losses concentrate where the gap sits close to the
  origin (`gap_lead_h` small), because a recent hole corrupts `demand_lag_1h`
  and `demand_lag_3h`, which the model weights most heavily.
* **H3 — gap timing (POST-HOC).** Losses concentrate at particular hours of day,
  where a same-clock-hour fill lands on a different part of the load shape.
  Suggested by PSCO, which was already seen to be worse and is known to have
  clock-aligned real gaps.

## 3. The cut, fixed now

Unit: one paired window. Outcome: `Δ = wape_control − wape_treatment`, positive
= treatment better. Both strata analysed separately and never pooled.

Covariate bins, fixed here so they cannot be tuned to a result:

| covariate | bins |
|---|---|
| `gap_len` | `1`, `2-3`, `13-24` (the empirical distribution's own three clusters) |
| `gap_lead_h` | `1-24`, `25-72`, `73-168` (inside `lag_24h`'s reach / inside `lag_72h` rolling / only `lag_168h`) |
| `gap_hour_utc` | `0-5`, `6-11`, `12-17`, `18-23` |

Reported per cell: n, mean Δ, median Δ, **win rate** (share of windows with
Δ > 0), and a bootstrap 95% interval on the mean.

## 4. What counts as a signal worth carrying forward

A cell is flagged only if **all three** hold:

1. **n ≥ 30** — below that the win rate is noise.
2. **Win rate < 50%** — the treatment loses more often than it wins there.
3. **The bootstrap 95% interval on mean Δ excludes 0** on the losing side.

**Multiplicity is real and is not corrected away.** Three covariates x up to four
bins x two strata is up to 22 cells; at a 5% level roughly one false flag is
expected by chance alone. So: **every cell is reported, flagged or not**, and no
cell is presented without the count of cells examined beside it. A single flagged
cell in 22 is reported as "consistent with chance"; a coherent pattern across
adjacent bins of the same covariate is what would be worth a confirmatory study.

## 5. What follows from any outcome, fixed now

* **A coherent, monotone pattern in H1 or H2** → a hypothesis for a new
  pre-registered confirmatory study on **fresh** windows (different seed, so
  different injected holes). Not a rule, not a flag flip.
* **Isolated flagged cells with no pattern** → chance; the losing quarter is
  diffuse, and there is nothing to carve out. This is a legitimate and likely
  outcome, and it would mean the flag question is settled as "not worth it" until
  the effect itself changes.
* **H3 flagged** → still post-hoc, and its only legitimate use is choosing what a
  future study injects, since it was suggested by data already seen.
* **Nothing flagged anywhere** → same as the second case.

## 6. Stopping rule

**One pass over the existing artifacts.** No re-binning, no additional
covariates, no per-BA fishing beyond the pre-specified PSCO look. If the cut
shows nothing, that is the answer.

## 7. Known limits

* Injected gaps are placed **uniformly at random**, so `gap_hour_utc` is uniform
  by construction and H3 tests only whether the *effect* varies by hour — not
  whether real gaps cluster there. Real clustering (PSCO's 10:00 UTC) is
  therefore **not** represented in this dataset at its true frequency.
* `gap_len` bins inherit the empirical distribution's imbalance: ~81% of windows
  carry a 1h gap, so the long-gap bins will be small and their intervals wide.
* One gap per window; this cannot speak to pile-ups.
* Stratum A carries real gaps in addition to injected ones, so its `gap_len`
  label describes only the injected hole.
