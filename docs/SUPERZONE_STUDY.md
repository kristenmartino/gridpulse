# Granularity is not the driver either — three mechanisms, three refutations

**The pre-registered prediction failed.** Halving NYISO's zone count from 11 to
5 left the bottom-up gain unchanged: **+0.745 pts against +0.729**, both
decisive, both 100% sign consistency. Zone count is not what separates NYISO
from CAISO.

Run 2026-07-31 against
[`SUPERZONE_PREREGISTRATION.md`](SUPERZONE_PREREGISTRATION.md), committed
before the run. Raw:
[`NYISO_SUPERZONE_STUDY.json`](NYISO_SUPERZONE_STUDY.json).
Reproduce: `python -m scripts.nyiso_bottom_up_study --iso NYISO5 --months 4`.

---

## Result

| arm | NYISO 11 zones | **NYISO 5 super-zones** | CAISO 5 zones |
|---|---:|---:|---:|
| top-down WAPE | 3.958 | 3.874 | 3.406 |
| bottom-up WAPE | 3.229 | **3.129** | 3.123 |
| **gain** | **+0.729** | **+0.745** | +0.283 |
| verdict | decisive, 3.7× se | **decisive, 3.4× se** | inconclusive |
| BA-weather arm (pure load) | +0.349 | +0.261 (inconclusive) | +0.023 |

**Prediction 1 failed.** The load channel was supposed to fall materially if
granularity drove it; it went from +0.349 to +0.261, a drop well inside its own
noise (stderr 0.257, t = 1.02).

**Prediction 3 also failed, and more interestingly.** The full arm was expected
to fall under *either* story, because aggregating loses weather resolution as
well as load granularity. It did not fall at all. Five weather points did as
much as eleven.

**What is now established:** at *equal zone count*, NYISO gains +0.745
decisively and CAISO gains +0.283 inconclusively. The difference is not the
number of zones.

## Three mechanisms proposed, three contradicted

| mechanism | prediction | measured | status |
|---|---|---|---|
| zonal **weather diversity** | more spread → more gain | CAISO 19.0 °F vs NYISO 12.6 °F, and CAISO gains *less* | **contradicted** |
| **granularity** (zone count) | fewer zones → less gain | 5 zones = 11 zones on NYISO | **contradicted** |
| component **heterogeneity** | less-correlated zones → more gain | CAISO inter-zone shape corr **0.412** vs NYISO5 **0.826**, and CAISO gains *less* | **contradicted** |

The NYISO effect itself is robust — it replicates across two independent zone
groupings of the same data, decisive both times. It is the *explanation* that
keeps failing.

## The observation that survives, untested

CAISO's five zones are not five comparable components:

| | share of load |
|---|---:|
| SCE-TAC | 0.48 |
| PGE-TAC | 0.42 |
| SDGE-TAC | 0.09 |
| **MWD-TAC** | **0.01** |
| **VEA-TAC** | **0.004** |

Two of the five are rounding errors. CAISO is effectively a 2½-component
decomposition with two negligible zones attached, while NYISO5 is METRO at
0.50 plus four genuine zones at 0.10–0.15.

Fitting a separate model to a zone carrying 0.4% of load contributes its own
error to the sum for almost no signal. **Bottom-up may help only when the
decomposition is into comparably-sized, well-populated components.**

That is a hypothesis and it is **not tested here**. The pre-registration's
stopping rule was explicit — *"no re-grouping to fit a result; if the grouping
turns out to matter, that is itself a finding and needs its own
pre-registration"* — and dropping MWD and VEA after seeing CAISO lose is
exactly the re-grouping it forbids. It is the obvious next test, under its own
pre-registration.

## Where this leaves the zonal line of work

* NYISO bottom-up: **real, replicated, decisive** (+0.729 / +0.745).
* CAISO: **inconclusive and half the size**.
* Why: **unknown**. Three candidate mechanisms tested, all contradicted.
* PJM and ISO-NE: still gated behind registration.

**Still do not build zonal ingestion.** A robust effect on one ISO with no
working explanation and one failed replication is not a foundation. The value
of this run is negative-knowledge: it removes zone count from the candidate
list and stops anyone shipping a granularity-based rationale.

Next, in order: the component-viability test above (own pre-registration);
PJM/ISO-NE keys, which need a human; then a decision.

## Limits

1. **One grouping.** Geographic contiguity, fixed in advance. A different
   5-way split might behave differently, and that is untested by design.
2. Six windows per arm; the BA-weather arm's drop (+0.349 → +0.261) is inside
   its own MDE, so "unchanged" is the honest reading rather than "slightly
   lower".
3. Four months, summer-weighted; XGBoost day-ahead, not the served ensemble.
4. Super-zone weather is the unweighted mean of member coordinates — a
   load-weighted centroid might preserve more signal, untested.
