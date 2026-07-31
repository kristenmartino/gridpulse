# Bottom-up does not generalise cleanly — and the mechanism is contradicted

**NYISO's +0.729 pt bottom-up win does not replicate on CAISO**, and the
cross-ISO comparison breaks the explanation I had offered for it.

Run 2026-07-31. Raw:
[`CAISO_BOTTOM_UP_STUDY.json`](CAISO_BOTTOM_UP_STUDY.json) ·
[`NYISO_BOTTOM_UP_STUDY.json`](NYISO_BOTTOM_UP_STUDY.json).
Reproduce: `python -m scripts.nyiso_bottom_up_study --iso CAISO --months 4`.

---

## PJM and ISO-NE could not be tested — both are gated

The recommendation was to test PJM and ISONE. **Neither publishes zonal load
without credentials**, verified rather than assumed:

| endpoint | result |
|---|---|
| PJM Data Miner 2 API (`api.pjm.com`) | **HTTP 401** without a subscription key |
| ISO-NE web services (`webservices.iso-ne.com`) | **HTTP 401** |
| ISO-NE static report paths (4 tried) | **HTTP 404** |
| NYISO monthly archive | HTTP 200, open |
| CAISO OASIS | HTTP 200, open |

Registration is free for both but requires creating accounts, which is not
something I can do. **CAISO was used instead** — a genuinely different case
rather than a convenience substitute: five TAC areas against eleven zones,
western rather than northeastern geography, and one very unusual zone (MWD is
pumping load, not population load).

## The result

| | NYISO | CAISO |
|---|---:|---:|
| zones | 11 | 5 |
| top-down WAPE | 3.958 | 3.406 |
| bottom-up WAPE | **3.229** | 3.123 |
| **gain** | **+0.729** | **+0.283** |
| verdict | **decisive**, 3.7× stderr | **inconclusive**, t = 1.69 |
| MDE at 6 windows | 0.390 | 0.334 |
| detectable | yes | **no** (0.283 < 0.334) |
| sign consistency | 100% | 83% |
| share of gain from load decomposition | 48% | **8%** |

Same direction, less than half the magnitude, below its own detection floor,
and a different internal composition.

## The mechanism is contradicted

`NYISO_ZONAL_PROBE.md` proposed that the gain comes from **zonal weather
diversity** — zones sitting at different points on a steep load-temperature
response, which one BA-level temperature cannot represent. The probe supported
it within NYISO, and the bottom-up win was consistent with it.

The cross-ISO comparison breaks that story:

| | mean hourly zonal temp spread | bottom-up gain |
|---|---:|---:|
| NYISO | 12.6 °F | **+0.729** |
| **CAISO** | **19.0 °F** | +0.283 |

**CAISO has ~51% more zonal temperature spread and gets less than half the
gain.** If weather diversity were the driver, this should run the other way.

The ablation says the same thing from the other side: on NYISO, 48% of the
gain came from **load** decomposition alone (BA weather for every zone); on
CAISO that channel contributes **8%**. What differs between the two is not
weather spread but **zone count and load heterogeneity** — NYISO splits into
eleven zones including a dense metropolitan pocket and rural upstate, while
CAISO's five TAC areas are large utility territories that each already average
over diverse geography.

That is a hypothesis, not a finding. What is established is only that the
weather-diversity explanation does not survive the second ISO.

**Caveat on the spread numbers:** both sets of zone centres are my own
approximations, and CAISO's five span a much larger area (San Francisco to San
Diego to Pahrump) than NYISO's eleven within one state. Some of the spread
difference is an artifact of that placement. It would have to be a large
artifact to reverse the conclusion, since the direction is opposite to the
prediction rather than merely weaker.

## What this means for building

**Do not build zonal ingestion yet.** The evidence is now:

* one ISO with a decisive win (NYISO, +0.729, replicated in ablation),
* one ISO where it is inconclusive and half the size (CAISO),
* the proposed mechanism contradicted by the comparison between them,
* and the two ISOs that would settle it locked behind registration.

This is the #230 pattern again: a result that looked clean on a small sample
changed character when the sample widened. The difference is that here it
widened by one ISO rather than 41 BAs, so the evidence is thinner in both
directions.

**If this is pursued**, the order is:

1. **Get PJM and ISO-NE keys** — free registration, and they are the two
   remaining large-ISO tests. Without them this cannot be settled.
2. **Test the zone-count hypothesis** directly: aggregate NYISO's 11 zones into
   5 super-zones and re-run. If the gain drops toward CAISO's, granularity is
   the driver and the weather story is dead.
3. Only then consider connectors.

## The reconciliation gap replicates

Both ISOs' zone sums differ materially from EIA's `D` for the same BA:

| | zone sum vs EIA `D` |
|---|---:|
| NYISO | **2.70% WAPE** |
| CAISO | **3.34% WAPE** |

Means agree in both cases (ratios 1.0003 and 1.0017), so this is hourly
disagreement, not a scale error. It is a general property of ISO-vs-EIA
definitions rather than a NYISO quirk, and it remains strictly prior to any
adoption: production forecasts `D`.

## Limits

1. **Two ISOs, one season** (4 months, summer-weighted), 6 windows each.
2. CAISO's effect is **below its detection floor** — "inconclusive" here means
   the test cannot see it, not that it is zero.
3. Zone centres are approximate in both cases; see the spread caveat.
4. XGBoost day-ahead, not the served recursive ensemble.
5. MWD-TAC is pumping load and behaves unlike the other four CAISO zones; it
   was kept because excluding it after seeing results would be selection.
