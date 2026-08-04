# Pre-registration — winter replication of the zonal bottom-up test

**Committed before the run.** The last of the two "different evidence" routes
[`COMPONENT_VIABILITY_STUDY.md`](COMPONENT_VIABILITY_STUDY.md) left open; the
other (a third ISO) needs credentials only a human can create.

This is **not** another regrouping — the stopping rule forbade those, and this
changes the season, not the decomposition.

## What is being tested

Every zonal result so far is summer-weighted (four months ending late July).
The NYISO bottom-up effect could be structural, or it could be a property of
the cooling season — NY's load is strongly air-conditioning-driven downstate
and much less so upstate, which is exactly the kind of thing that could make
zonal decomposition pay off in July and not in January.

**Window: 4 months ending 2026-03-01** — November 2025 through February 2026.
Heating season in both territories. Everything else (6 windows, 168h horizon,
harness, target = zone sum, XGBoost day-ahead) is unchanged from the summer
runs, so the comparison is like-for-like.

## Arms

Both ISOs at their original decompositions, the two that produced the headline
summer results:

* **NYISO** — 11 zones. Summer: **+0.729**, decisive, 100% sign consistency.
* **CAISO** — 5 TAC areas. Summer: **+0.283**, inconclusive.

## Predictions

1. **Structural** — NYISO winter is decisive and of similar magnitude
   (≥ +0.40), CAISO winter stays inconclusive. The effect is real and
   season-independent; the cross-ISO puzzle stands unchanged.
2. **Summer artifact** — NYISO winter falls materially (< +0.40, or
   inconclusive). The effect is cooling-driven, which would finally explain
   the CAISO difference in terms of climate rather than structure, and would
   be the first mechanism to survive.
3. **Something else** — e.g. CAISO becomes decisive in winter. Would mean the
   effect is seasonal in *both*, in opposite directions, and that nothing so
   far generalises at all.

## What each outcome licenses

| outcome | conclusion |
|---|---|
| NYISO decisive, CAISO not | effect is structural and still unexplained; **stop** — the remaining route is a third ISO |
| NYISO falls | seasonality is a live mechanism; worth one follow-up isolating heating vs cooling |
| CAISO decisive | nothing generalises; abandon the zonal line |

## Stopping rule

One run per ISO. No re-windowing to find a season that works — if winter and
summer disagree, that disagreement is the finding.

## Prior

Six mechanisms proposed for this effect, six refuted, while the effect itself
survived every attempt to explain it away. Prediction 1 (nothing changes) is
therefore the one I expect, and I am recording that expectation now so that a
surprise cannot be narrated afterwards as what I thought all along.
