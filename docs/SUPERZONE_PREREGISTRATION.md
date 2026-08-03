# Pre-registration — NYISO 11 → 5 super-zone test

**Committed before the run.** Nothing below was chosen after seeing results.

## The question

`ZONAL_GENERALISATION.md` found NYISO's bottom-up win (+0.729 pts, decisive)
did not replicate on CAISO (+0.283, inconclusive), and that the proposed
mechanism — zonal **weather diversity** — is contradicted: CAISO has 19.0 °F
mean hourly zonal temperature spread against NYISO's 12.6 °F, 51% *more*, and
gets less than half the gain.

The surviving hypothesis is **granularity**: 11 zones including a dense metro
pocket and rural upstate, versus 5 large utility territories that each already
average over diverse geography.

This tests it on data already in hand: aggregate NYISO's 11 zones into 5 and
re-run. No new data source, no connector.

## The grouping, fixed now

By **geographic contiguity**, to mirror CAISO's large contiguous utility
territories. Chosen from the map, not from any result:

| super-zone | members |
|---|---|
| WEST | WEST, GENESE |
| CENTRAL | CENTRL, MHK VL |
| NORTH_CAPITAL | NORTH, CAPITL |
| LOWER_HUDSON | HUD VL, MILLWD, DUNWOD |
| METRO | N.Y.C., LONGIL |

Each super-zone's weather is taken at the mean of its members' coordinates.

## The confound, and how the existing ablation resolves it

Aggregating 11 → 5 reduces load granularity **and** weather resolution
together, so the headline comparison alone cannot attribute. The study already
runs an ablation arm — bottom-up with a single BA-level weather point for every
zone — which is **pure load decomposition, no zonal weather at all**. Comparing
that arm across zone counts isolates the load channel:

| arm | 11 zones (measured) | 5 super-zones (this run) |
|---|---:|---|
| bottom-up, zonal weather | **+0.729** | ? |
| bottom-up, BA weather (pure load) | **+0.349** | ? |

## Predictions

1. **If granularity drives the load channel**, the BA-weather arm falls
   materially below +0.349 — toward CAISO's +0.023 equivalent.
2. **If granularity does not drive it**, that arm stays near +0.349 despite
   halving the zone count, and the difference between NYISO and CAISO lies
   somewhere else entirely (load heterogeneity, sample, or chance).
3. The full arm falls below +0.729 under either story, since it loses weather
   resolution too — so **the full arm alone decides nothing**, and is reported
   only for completeness.

## Decision rule

Through `models/rolling_eval.py` per `docs/EVALUATION_POLICY.md` — WAPE,
satisficing, `verdict()` may refuse. Same 4 months, 6 windows, 168h horizon as
both prior runs, so the comparison is like-for-like.

## Stopping rule

One run. No re-grouping to fit a result; if the grouping turns out to matter,
that is itself a finding and needs its own pre-registration.
