# Per-horizon champion-chasing (#478 side study): rejected

**Verdict: chasing the previous window's per-horizon champion model does not
clear.** It loses to a fixed model at all three horizons — 24h, 48h, and 72h.
**The reason is that champion-vs-runner-up margins are near-ties, not that the
champion flips often.** The raw flip-rate numbers (90–98%) look alarming and
are not evidence of anything; read past them to §2 before drawing a
conclusion from them.

Measured 2026-08-18 from the preserved drift-horizon buffer. Reproduce:
`scripts/analyze_horizon_champion_flips.py` against
`gs://nextera-portfolio-energy-cache/cache/adhoc/drift_horizon_dump_20260818T035339Z.json`
(41.6MB, 51 BAs, dumped by `scripts/dump_drift_horizon.py`, `{arima, ensemble,
prophet, xgboost}` × `{24h, 48h, 72h}`, 720 raw records per BA-model-horizon).
**No GCS lifecycle rule on that object, so it will not age out — do not
re-dump it.**

---

## 1. The strategy comparison

Four rolling, non-overlapping weekly windows, four candidate models, WAPE:

| strategy | 24h | 48h | 72h |
|---|---:|---:|---:|
| follow previous window's champion | 4.980 | 5.452 | 5.634 |
| **always xgboost** | **4.909** | 5.378 | 5.626 |
| **always ensemble** | 5.000 | **5.326** | **5.625** |
| always prophet | 6.357 | 6.743 | 7.070 |
| always arima | 6.758 | 7.354 | 7.851 |
| oracle (perfect hindsight) | 4.619 | 4.917 | 5.125 |

Chasing the champion loses to the best fixed model at **every** horizon:
+0.071 pts at 24h, +0.126 at 48h, +0.009 at 72h. It isn't close to
competitive with oracle either — chasing captures none of the ~0.4-pt gap
between the best fixed policy and perfect hindsight (§5).

**Why chasing loses despite always following a recent winner:** median
champion-vs-runner-up margin is only **0.212 / 0.316 / 0.307 WAPE pts** at
24/48/72h. A margin that small is close to measurement noise, so "last week's
winner" carries little information about "this week's winner" — the premise
the strategy depends on. See §2 for the quantitative version of this.

## 2. The flip-rate numbers need a null baseline, and read backwards without one

The instinct on seeing a champion flip 90–98% of the time is "unstable,
therefore chasing is a bad idea." That instinct is right here, but not for
that reason — with 4 candidate models, flipping is what a **random** champion
does too, so the raw rate carries no signal on its own.

Expected flip rate if the champion were chosen at random among 4 models,
independently each window (`P(different) = 1 − 1/4 = 75%` pairwise; compounding
across more comparisons pushes it higher):

| windows compared | P(identical if random) | expected flip rate |
|---|---:|---:|
| 2 | 25.0% | **75.0%** |
| 4 (non-overlapping) | 1.6% | **98.4%** |

Measured 24/48/72h flip rates — **90.2% / 98.0% / 94.1%** — sit **at** the
98.4% null for a 4-window comparison. They carry **no signal**: this is
indistinguishable from a champion chosen at random each week.

The 1h figure is the one that actually says something, and it says the
opposite of what "high flip rate" suggests: **18/51 = 35.3%**, measured
against a **2-window** comparison, so its null is 75.0%. **35.3% is far below
75%** — champions at 1h are *more stable than chance*, not less.
**Anyone reading 35.3% as "unstable" has it backwards.** Comparing flip rates
across different window counts without their respective nulls is invalid;
every figure in this section is reported against its own null for that
reason.

## 3. Mutual validation against an independent measurement

This result reproduces a sign flip measured independently, on a different
metric and different windows, in the parallel fleet-baseline work: xgboost
ahead at 24h, ensemble ahead at 48h/72h. That measurement's paired xgboost-
vs-ensemble delta moved **−0.0635 → +0.0554** across horizons — the same
crossover this study's strategy table shows (xgboost wins 24h by 0.091,
ensemble wins 48h by 0.052 and 72h by 0.001). Two independent measurements
agreeing on the crossover, on different data, is stronger evidence for it
than either alone — and it's the same underlying fact as §1/§2: the models
are close enough that which one is "ahead" depends on the window.

## 4. Champion counts — "best fixed policy" is not "usually wins per BA"

At 24h, across 204 BA-weeks: xgboost 82 (40.2%), ensemble 82 (40.2%), prophet
24 (11.8%), arima 16 (7.8%).

XGBoost is the best fixed policy at 24h (§1), but it does not dominate
per-BA: it won **all four** windows on only **3 BAs** (CHPD, IID, SPP), and
**none** on **9 BAs** (AZPS, BPAT, FPL, PACW, PGE, PNM, SEC, TIDC, TVA). A
policy can be the best *fleet-aggregate* choice while losing consistently on
a fifth of the fleet — those two claims are not in tension, but they answer
different questions, and only the fleet-aggregate one is what "always
xgboost" in §1 measures.

## 5. Where the oracle headroom sits — and why it's unreachable this way

Oracle (perfect hindsight) beats the best fixed policy by roughly **0.4 pts**
at every horizon: 4.619 vs 4.909 (24h), 4.917 vs 5.326 (48h), 5.125 vs 5.625
(72h). That headroom is real, but §1–§2 show why chasing can't reach it: the
oracle knows *this* window's winner, and the near-tie margins mean last
window's winner is a weak predictor of it. Closing this gap needs information
oracle has and chasing doesn't — not a better chasing rule.

## 6. Data completeness — episodic and BA-specific, and window comparability was not checked

Windows carry a median of **161–162 of a possible 168** records, not 168.
This is not uniform sampling noise: **#537** now explains the mechanism.
Each BA's forecast origin should advance one hour per tick; when it stalls,
the horizon path re-derives a target it already holds for that stalled
window, and `seen` correctly drops the duplicate — so missing records
concentrate wherever a BA's origin froze, not randomly across the fleet.
**LGEE** is the clearest case: 44 of 140 ticks sat on a frozen origin in the
week examined for #537, one BA carrying a large share of the shortfall by
itself.

**What this study can and cannot rule out.** Missingness is known to be
BA-specific (LGEE vs. the rest) and episodic (clustered in the ticks a given
BA's origin was frozen, not spread evenly). Whether the missingness rate
**correlates with which of the four weekly windows a BA falls in** — i.e.,
whether some windows are systematically thinner than others across the
fleet, which would make cross-window WAPE comparisons in §1 not strictly
comparable — was **not checked** as part of this measurement. This is an open
question, not a ruled-out one: the strategy table in §1 should be read as
measured on the available records per window, without a verified guarantee
that each window's coverage is equivalent to the others'.

## 7. #542 does not touch this study

Measured directly: **442,537 horizon records across 51 BAs, zero carry a
lead.** `_horizon_rollup_block` never lead-filters — it has no code path that
reads `lead_hours` at all. The lead-erasure defect in `regrade_records`
(#542) and the shadow-eval instrument failure (#541) both live in the 1h
drift/shadow paths; the 24/48/72h records behind this study come from
`drift_horizon`, a separate code path neither one touches. This study's
numbers are unaffected by either fix.

## 8. Limitation: fixed per-BA assignment was not tested

This study tested three things: champion-**chasing** (dynamic, switches to
last window's winner), **fleet-fixed** (one model for the whole fleet, held
constant), and **oracle** (perfect hindsight). It did **not** test a
**per-BA fixed assignment learned from history** — e.g., "serve XGBoost for
CHPD/IID/SPP permanently, ensemble for the rest."

ADR-004 (`PRD.md:227`) already rejected that idea under "Alternatives
considered (2) Winner-take-all," on grounds that don't depend on this
study's data: it forgoes the ensemble's error-decorrelation wins, and "a
model that wins one week can lose the next" — which §1/§2 above independently
corroborate (near-tie margins, no meaningful persistence of a window's
winner). **This study does not close the door on a learned per-BA fixed
policy, and it does not open it either** — it simply never tested that
specific strategy. Do not read this document as evidence for or against
per-BA fixed assignment.

## Recommendation

**Keep the current fixed ensemble weighting (ADR-004) as-is.** Per-horizon
champion-chasing does not clear on its own operational test (§1), the
mechanism for why is understood and corroborated two independent ways (§2–3),
and the residual headroom (§5) is not reachable by this strategy. No follow-up
experiment is proposed here: the evidence doesn't point at a specific next
test worth running, and this negative is worth recording as-is rather than
reaching for one.
