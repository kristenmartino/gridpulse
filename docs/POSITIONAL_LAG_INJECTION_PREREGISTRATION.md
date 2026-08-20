# Pre-registration — #559, settling the seed question by injecting gaps

**Committed before the run. Nothing below was chosen after seeing results.**

The observational study (`POSITIONAL_LAG_SEED_STUDY.md`) was **inconclusive at
both horizons** and cannot be rescued by waiting: the defect only produces an
observation when EIA drops an hour, so a decisive verdict is 1.2-6.6 years out.
This run removes the rate limit by creating the gaps instead of waiting for them.

The earlier study was **not** pre-registered — arms and metric were fixed in
advance, the per-BA cut was not, which is why its TIDC result was reported as
hypothesis-generating only. This one is pre-registered so its result can count.

## 1. Why injection is legitimate here

Usually a synthetic manipulation buys power at the cost of realism. The trade is
unusually cheap in this case, and the reason is mechanical:

`dropna` deletes every row whose lag source was null, and
`compute_autoregressive_snapshot` then indexes the surviving list **by
position**. That is deterministic. A hole is a hole: an injected null hour and an
EIA-dropped null hour produce *the same deletion* and therefore the same index
shift. Nothing about the defect can distinguish them.

What injection cannot reproduce is *where and when* real gaps fall. §6 treats
that as the live confound.

## 2. The hypothesis, stated in advance

> On origins whose 168-hour lookback contains a gap, resolving autoregressive
> lags by timestamp produces lower WAPE than resolving them by position.

Direction is predicted, not merely "differs": the training features are
temporally correct, so the temporal seed moves inference *toward* the convention
the model was fit under.

## 3. Injection procedure, fixed now

Gaps are NaN **values on present rows** — the shape EIA actually produces (7
absent rows fleet-wide against 78 null ones), so this reproduces the dominant
mechanism rather than a convenient one.

Run lengths are drawn from the **empirical distribution measured 2026-08-20**
across all 51 mirrors over 90 days, frozen here:

| run length (h) | 1 | 2 | 3 | 13 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| runs observed | 25 | 2 | 1 | 1 | 1 | 1 |

31 runs, 85 missing hours, 7 BAs. Median run is 1 hour; the tail is what moves
`lag_168h` furthest.

**One gap run per window**, placed uniformly at random inside the window's
168-hour lookback, seeded per (BA, window) so the run is reproducible.

## 4. Population and unit of analysis

Two strata, reported separately and never pooled into one headline:

* **A — naturally gapped**: LGEE, PSCO, TIDC, IID, NWMT, NEVP, SPA. Preserves the
  population the defect actually affects.
* **B — never gapped**: MISO, PJM, ERCOT, CAISO, SPP, DUK. Tests whether the
  effect survives on larger, smoother demand, where it might not.

The unit is one **paired 48-hour window** (control and treatment on the same
origin, same vintage, same weather, same injected gap). 48h matches the horizon
where the observational study had most windows.

## 5. Power, fixed before the run

Per-window sd implied by the observational study (n=85, stderr 0.203) is
**1.872**. Minimum detectable effect is 2 x stderr:

| n | MDE |
|---:|---:|
| 240 | 0.242 |
| **480** | **0.171** |
| 720 | 0.139 |

**Target n = 480 per stratum.** That clears both thresholds worth clearing: the
+0.181 the observational study estimated, and 0.25 pts — half of
`MAX_MAPE_REGRESSION_PTS`, which is this repo's existing notion of a material
move in the published metric. The realised MDE is reported whatever it is; if
the run yields fewer scoreable windows than planned, that is stated rather than
quietly absorbed.

### Amendment, 2026-08-20 — before any result was computed

Target n was set before checking it was reachable, and for stratum A it is not.
Only **7** BAs are ever naturally gapped, and a 90-day mirror yields **40**
non-overlapping 48h windows each after warm-up and truth requirements:

| stratum | BAs | windows | MDE |
|---|---:|---:|---:|
| **A — naturally gapped** | 7 (structural ceiling) | **280** | **0.224** |
| **B — never gapped** | 12 (expanded from 6) | **480** | **0.171** |

Stratum A cannot be enlarged. Extending history does not help: archived model
vintages only reach 2026-05-07, which is a tighter bound than the mirror, and
without a vintage there is nothing to replay. So:

* **Stratum A is powered for a 0.25-pt effect but NOT for the +0.181 the
  observational study estimated.** If the true effect is that size, A is
  expected to return inconclusive, and that is a limit of the available data,
  not a finding.
* **Stratum B is expanded to 12 BAs** — MISO, PJM, ERCOT, CAISO, SPP, DUK,
  ISONE, NYISO, FPL, TVA, BPAT, PACE — reaching the original target and the
  original MDE.

Reading fixed now, so it cannot be chosen later: if **A is inconclusive and B is
decisive**, that is evidence the mechanism is real at a size B can see and A
cannot, and the honest summary is "smaller than 0.224 pts on the affected
population." It is **not** licence to quote B's number as A's.

This amendment is recorded rather than folded in silently, and it is committed
in its own commit before the runner exists — the ordering is checkable in git.

## 6. What counts as confirmation

The hypothesis is **confirmed** only if all three hold:

1. **Stratum A is decisive and positive** — `models.rolling_eval.verdict()`
   returns `decisive` with `winner == "treatment"` on pooled paired deltas.
2. **The null control is exact** — windows with **no** injected gap show Δ of
   exactly 0. A nonzero delta there means the harness is measuring something
   other than the seed convention, and voids the run.
3. **Satisficing holds** — `satisficing_check` passes on bias and MAPE
   regression, with the control arm's own bias inside `MAX_ABS_BIAS_PCT` first.
   A harness whose control fails a constraint cannot certify a treatment
   against it.

Anything less is **not confirmed**. `verdict()` returning inconclusive at
n=480 is a real and publishable outcome: it would mean the effect is smaller
than 0.171 WAPE points, which is itself an answer about whether the flag matters.

## 7. Stopping rule

**One run**, at the parameters above. No re-runs with adjusted gap placement,
horizon, BA set or window count. If it fails, it fails, and any follow-up is a
new pre-registration.

## 8. Known confounds, acknowledged in advance

* **Placement is uniform; real gaps may not be.** PSCO's ran at exactly 10:00 UTC
  on consecutive days. If gap timing correlates with demand dynamics, uniform
  placement under- or over-states the effect. Not corrected for; hour-of-day is
  recorded so it can be examined later, and any such cut is descriptive.
* **Injection destroys real information.** Both arms lose the same rows to
  `dropna`, so the *comparison* stays fair, but absolute accuracy in both arms is
  worse than an ungapped baseline. Deltas are the unit; levels are not.
* **Archived vintages and mirrored weather**, as in the observational study, so
  absolute levels are not production's. Common-mode across arms.
* **XGBoost only.** Says nothing about Prophet, SARIMAX, or the served ensemble
  beyond `xgboost_weight × Δ`.
* One gap per window, so this measures the effect of a *typical* gap, not of the
  pile-up a bad week produces.
