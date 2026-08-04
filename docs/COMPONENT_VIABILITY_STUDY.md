# Component viability fails too — the effect is real, replicated, and unexplained

**Not confirmed on either pre-registered criterion.** Folding CAISO's two
negligible areas into their neighbour moved the bottom-up gain from **+0.283**
to **+0.348** — a nudge well inside noise, still inconclusive, and nowhere near
the +0.50 the pre-registration required.

Run 2026-08-03 against
[`COMPONENT_VIABILITY_PREREGISTRATION.md`](COMPONENT_VIABILITY_PREREGISTRATION.md),
committed before the run. Raw: [`CAISO3_STUDY.json`](CAISO3_STUDY.json).
Reproduce: `python -m scripts.nyiso_bottom_up_study --iso CAISO3 --months 4`.

---

## Result against the pre-registered criteria

| criterion | required | result | |
|---|---|---|:--|
| verdict decisive, winner `treatment` | yes | **inconclusive** (t = 1.375) | **FAIL** |
| gain ≥ +0.50 pts | yes | **+0.348** | **FAIL** |

| | CAISO 5 | **CAISO 3** | NYISO 5 | NYISO 11 |
|---|---:|---:|---:|---:|
| components | 5 | **3** | 5 | 11 |
| smallest share | 0.004 | **0.090** | 0.10 | 0.0165 |
| **gain** | +0.283 | **+0.348** | **+0.745** | **+0.729** |
| verdict | inconclusive | **inconclusive** | decisive | decisive |
| MDE | 0.334 | 0.507 | 0.437 | 0.390 |
| pure-load channel | +0.023 | **+0.039** | +0.261 | +0.349 |

Removing every component below 9% of load bought **+0.065 pts**, against a
detection floor of 0.507. The mean also sits far above the median (+0.348 vs
+0.068), the outlier signature the harness flags.

## The sharpest statement of the difference

**The pure-load-decomposition channel works on NYISO and is dead on CAISO, and
no regrouping of CAISO revives it.**

* NYISO: +0.349 (11 zones), +0.261 (5 zones)
* CAISO: +0.023 (5 zones), +0.039 (3 zones)

An order of magnitude apart, stable across every grouping tried on both sides.

## Six explanations, six refutations

| # | explanation | measured | |
|---|---|---|---|
| 1 | anchor staleness | 2h stale anchor costs +0.014 pts | refuted |
| 2 | zonal weather diversity | CAISO 19.0 °F spread vs NYISO 12.6 °F, gains *less* | refuted |
| 3 | granularity / zone count | 5 zones = 11 zones on NYISO | refuted |
| 4 | component heterogeneity | CAISO inter-zone corr 0.412 vs NYISO 0.826, gains *less* | refuted |
| 5 | component viability | this run — not confirmed on both criteria | refuted |
| 6 | lossy zonal data | CAISO zone sum vs CAISO's own total: **0.000% WAPE**, ratio 1.0000, std 0.0000 | refuted |

Explanation 6 was checked because it was *different* evidence rather than
another regrouping, and it is the cleanest negative of the set: CAISO's
`CA ISO-TAC` is arithmetically the exact sum of its five areas. The
decomposition loses nothing at source.

## The honest position

The NYISO bottom-up effect is:

* **real** — +0.729 and +0.745, decisive, 100% sign consistency both times;
* **replicated** — across two independent zone groupings of the same data;
* **not generalised** — CAISO is inconclusive at every grouping tried;
* **unexplained** — six candidate mechanisms tested, all refuted.

The pre-registration's stopping rule anticipated exactly this: *"further
mechanism-hunting needs a different kind of evidence than another regrouping of
the same two ISOs."* That is now binding. **No more regroupings.**

## What different evidence looks like

1. **A third ISO.** PJM and ISO-NE both gate zonal load behind free
   registration (HTTP 401, verified). That needs a human with an account; it is
   the single highest-value unblock left in this line.
2. **A winter run.** Everything here is summer-weighted. If the effect is
   cooling-driven in a way that happens to differ between NY and CA, a January
   run would show it.
3. **Nothing else on these two ISOs.** Six mechanisms is enough to conclude the
   available evidence cannot distinguish them.

**Do not build zonal ingestion.** One ISO, unexplained, one failed
replication.

## A note on the record

Six mechanisms proposed across this line of work, six refuted by measurement,
while the underlying effect survived every attempt to explain it away. The
apparatus is doing its job — each refutation cost hours, not weeks, and each
one removed a wrong rationale that could otherwise have justified a build.

The pattern also says something about the hypotheses: they were all *plausible*
and all *wrong*, which is a reason to weight the next one lower still, and to
prefer evidence that discriminates (a third ISO) over evidence that elaborates
(a seventh regrouping).

## Limits

1. Six windows per arm; CAISO-3's MDE is **0.507**, larger than the effect
   being measured, so "inconclusive" here means the test cannot see a +0.35
   effect — not that one is absent.
2. Four months, summer-weighted; XGBoost day-ahead, not the served ensemble.
3. One regrouping, fixed in advance. A different fold (MWD into SDGE, say)
   is untested by design.
4. The pure-load channel comparison across ISOs is descriptive, not a
   controlled contrast — the two ISOs differ in many ways at once.
