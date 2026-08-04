# Pre-registration — CAISO component-viability test

**Committed before the run.** Required by
[`SUPERZONE_PREREGISTRATION.md`](SUPERZONE_PREREGISTRATION.md)'s stopping
rule, which forbade re-grouping CAISO after seeing it lose.

## The hypothesis

`SUPERZONE_STUDY.md` left one observation standing after three mechanisms were
refuted: CAISO's five zones are not five comparable components.

| CAISO | share | | NYISO 11 | share |
|---|---:|---|---|---:|
| SCE-TAC | 0.480 | | N.Y.C. | 0.354 |
| PGE-TAC | 0.420 | | … | … |
| SDGE-TAC | 0.090 | | NORTH | 0.030 |
| **MWD-TAC** | **0.010** | | **MILLWD** | **0.0165** ← smallest |
| **VEA-TAC** | **0.004** | | | |

> Fitting a separate model to a component carrying 0.4% of load adds its own
> error to the sum for almost no signal. Bottom-up may help only when the
> decomposition is into comparably-sized, well-populated components.

**Pre-check, run before this document and reported honestly:** NYISO's
11-zone split — which gained +0.729 decisively — has a smallest zone of
**1.65%** and **nothing below 1%**. CAISO has two components below 1.1%, one
of them 4× smaller than anything in NYISO. So the hypothesis is not refuted by
NYISO's own tail, which is why the test is worth running.

## The regrouping, fixed now

**CAISO-3**: fold the two negligible areas into SCE-TAC, their geographic
neighbour (MWD is Southern California pumping load; VEA sits on CAISO's
southeastern edge adjacent to SCE's footprint).

| CAISO-3 component | members | share |
|---|---|---:|
| SCE_PLUS | SCE-TAC, MWD-TAC, VEA-TAC | 0.494 |
| PGE | PGE-TAC | 0.420 |
| SDGE | SDGE-TAC | 0.090 |

Weather for `SCE_PLUS` is SCE-TAC's own point — MWD and VEA are folded into
its load, not averaged into its coordinate, because the point of the test is
to stop modelling them separately, not to move SCE's weather.

Everything else — months, windows, horizon, harness, target — is unchanged
from the CAISO-5 run, so the comparison is like-for-like.

## Predictions

1. **If component viability drives it**, CAISO-3's gain rises materially above
   CAISO-5's **+0.283** and becomes decisive.
2. **If it does not**, CAISO-3 ≈ CAISO-5, and the NYISO/CAISO difference stays
   unexplained after four mechanisms.

## Confirmation criteria — both required

* CAISO-3 `verdict` is **decisive** with `winner == "treatment"`, **and**
* gain ≥ **+0.50 pts** — at least half the distance from CAISO-5's +0.283
  toward NYISO5's +0.745.

Anything less is **not confirmed**. A decisive but small gain (say +0.35) would
mean removing dead components helps a little, not that viability explains the
cross-ISO difference — and the criteria are set now so that result cannot be
narrated as a win later.

## Prior, stated for honesty

Four mechanisms have now been proposed for this effect — anchor staleness,
weather diversity, granularity, component heterogeneity — and measurement has
refuted all four, while the NYISO effect survived every attempt to explain it.
My prior on this fifth one is correspondingly low, and a marginal result should
be read against that record rather than as the explanation finally landing.

## Stopping rule

One run. No further regrouping. If this fails, the honest position is that the
NYISO bottom-up effect is real, replicated, and **unexplained** — and that
further mechanism-hunting needs a different kind of evidence than another
regrouping of the same two ISOs.
