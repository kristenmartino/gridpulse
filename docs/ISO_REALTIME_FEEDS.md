# ISO real-time feeds: the anchor case is dead, the zonal case is untested

**Verdict on the reason I gave for wanting them: wrong.** EIA-930's publishing
lag is ~1.7h, not the multi-hour gap I asserted, and forecast accuracy is
insensitive to anchor age anyway — a **2h stale anchor costs +0.014 WAPE pts
(median −0.036)**, and even a **24h** stale anchor costs under 0.1.

A different case survives, but it is not the one I was making, and it is a
much larger integration.

Run 2026-07-31. Raw: [`ANCHOR_STALENESS_PROBE.json`](ANCHOR_STALENESS_PROBE.json).
Reproduce: `python -m scripts.anchor_staleness_probe --windows 6`.

---

## What I claimed, and what is true

Across several turns I recommended ISO real-time feeds on this reasoning:
*"EIA-930 is hourly and lagged; ISO feeds are 5-minute and near-real-time.
That matters because SEC's error was flat across horizon — wrong at hour 1 —
which is an anchor/timeliness problem."*

Measured:

| | claimed | measured |
|---|---|---|
| EIA publication lag | "lagged" | **1.7h, identical across all 51 BAs** |
| trailing stub hours | implied material | **median 0**; only ERCOT has any (6h) |
| effective anchor staleness | multi-hour | **1.7h** for 48 of 49 BAs, 7.7h worst |

**A correction inside the correction.** My first staleness probe reported a
19.7h median. That was a bug in my own script: EIA's `D` and `DF` series cover
different ranges — `DF` extends into the future — so every forecast-only
future hour was counted through an `isna(D)` branch as a stale actual. The
18-hour "staleness" was the forecast horizon. Fixed, and the real number is 0.

## The ceiling: what a perfect real-time anchor would buy

One direct multi-horizon model per (BA, window); staleness is then just a
shift in `horizon_h`, so the same model answers every level. Scored hours held
fixed at `[origin, origin+24)`, so only the anchor age varies. `s = 0` is a
perfect real-time anchor — the best any ISO feed could deliver.

5 BAs × 6 windows:

| anchor age | mean cost (pts) | median |
|---:|---:|---:|
| 0h (perfect) | 0.000 | 0.000 |
| 1h | +0.017 | −0.037 |
| **2h ← today's lag** | **+0.014** | **−0.036** |
| 4h | −0.025 | −0.033 |
| 8h | −0.017 | −0.025 |
| 16h | +0.126 | +0.204 |
| 24h | +0.087 | +0.076 |

The 0–8h range is noise around zero — several entries are *negative*, meaning
the staler anchor scored slightly better, which is what no-effect looks like.
A signal only appears around 16h, and 24h is lower than 16h, so even that is
not a clean curve.

**At a 24h operating horizon the forecast barely uses recent demand** — it is
driven by weather and calendar. That is coherent: the autoregressive signal
decays long before the target hour. It is also why production's hourly
re-anchoring already delivers low live 1h drift: **median 2.08% sMAPE, ensemble,
7-day window, across 51 BAs** (2026-08-18T07:06Z).

*The unqualified "~1.5%" that stood here was stale and carried no metric,
window or statistic — the format `docs/BENCHMARK_METHODOLOGY.md` §8 rules out.
Re-measured during the #542 lead-filter work, which moved this median by
**+0.010 pts** (2.077 → 2.087); the gap to 1.5% predates that change and is
not caused by it. The **mean** over the same population is 3.63%, dragged by a
handful of small or broken-feed BAs — one fleet number needs its statistic
named.*

## What ISO feeds do offer — verified, not asserted

Since the anchor case failed, I checked what else is actually there rather
than repeating the claim from memory. Fetched live, auth-free:

| source | status | content |
|---|---|---|
| NYISO real-time load (`pal.csv`) | HTTP 200, 170 KB | 5-minute load **by zone** |
| NYISO load forecast (`isolf.csv`) | HTTP 200, 11 KB | day-ahead forecast **per zone** — Capitl, Centrl, Dunwod, Genese, Hud Vl, Longil, Mhk Vl, Millwd, N.Y.C., North… |
| CAISO OASIS `SLD_FCST` | HTTP 200 | day-ahead system load forecast |

**The real differentiator is zonal decomposition, not freshness.** EIA-930
gives one number per BA; NYISO publishes eleven. Sub-BA structure is
information the model has never had, and it is plausibly relevant — a BA's
load shape is a sum of zones with different weather and different mixes.

That is a genuinely different hypothesis from the one I was arguing, and it is
untested here.

## Honest cost of that path

* **Coverage is ~7 of 51 BAs.** Only ISO/RTO territories publish this;
  PJM, MISO, ERCOT, SPP, CAISO, ISONE, NYISO. They do carry roughly 62% of
  fleet MW error (`ERROR_ANALYSIS.md`), so coverage is concentrated where it
  matters — but 44 BAs get nothing.
* **Seven different integrations.** Each ISO has its own schema, auth model,
  zone definitions and rate limits. This is not one connector.
* **The weather side would have to follow.** Zonal load without zonal weather
  is half a feature, and `assets/multipoint_coordinates.json` is BA-level.
* **ERCOT now gates its modern API behind registration**, unlike the two
  verified above — so "auth-free" does not generalise across all seven.

## Recommendation

**Do not build the real-time ingestion I was recommending.** Its stated
purpose — a fresher anchor — is worth approximately zero, measured.

If ISO data is pursued, it should be for **zonal decomposition**, tested the
way everything else in this project now is: a probe on one ISO (NYISO is the
cheapest — auth-free, 11 zones, both load and forecast) asking whether zonal
structure predicts BA-level residuals, *before* any connector is written.

Note the pattern this closes: five consecutive investigations — cooling
features, BTM solar, ARIMA order selection, direct multi-horizon, and now
anchor freshness — have returned negative or inconclusive. The apparatus is
trustworthy enough that those negatives are worth something, but the
accumulated evidence increasingly says the remaining error is not reachable
from the inputs and structures tried so far.

## Limits

1. **5 BAs × 6 windows** for the staleness curve; the per-BA MDE caveat from
   `DIRECT_MULTIHORIZON_PREREGISTRATION.md` applies, though an effect this
   close to zero would need a very large sample to be hiding something.
2. **24h horizon only.** Staleness plausibly matters more at 1–6h horizons,
   which production serves by re-anchoring hourly and where live drift is
   already ~1.5%.
3. **XGBoost, direct formulation**, not the served recursive ensemble.
4. **Lag measured at one instant** (2026-07-31 12:39 UTC). EIA publishes
   hourly, so the lag oscillates roughly 1–2h; it was identical across all 51
   BAs at that instant, which is the informative part.
5. The zonal hypothesis is **stated, not tested**.
