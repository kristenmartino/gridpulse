# Market position — what GridPulse could sell, and what it can't

**Internal.** Written 2026-07-28, after the E0 benchmark produced its first real
numbers and a competitive review of [orreryhq.com](https://orreryhq.com).

Every figure here is measured and re-runnable unless marked otherwise. Where the
research behind this doc overclaimed, the correction is kept in place rather than the
original — several claims that sounded good did not survive checking, and the ones
that did are worth more for it.

> **Posture note.** Nothing in this doc is public, and nothing derived from it should
> be published as a commercial claim while the current interview process is live. The
> public surfaces stay on the neutral "Forecast Benchmark" framing.

---

## 1. The finding that sets everything else

The benchmark measures GridPulse against each balancing authority's own day-ahead
forecast — the free incumbent, published by EIA-930. First real result:

| | |
|---|---|
| Scoreable BAs | **44** of 51 (7 excluded, each with a published reason) |
| Head-to-head at 24h | **operator closer on 28**, GridPulse on 15 |
| Median mean MAPE | **theirs 3.73%**, ours 4.74% |
| Error spread across BAs | **theirs 23.4×**, ours **8.1×** |

**"We forecast demand better" is not a sellable claim.** It is contradicted by our own
published scorecard, on a public page, computed from a method we wrote down first.

The 48h conservative arm does not rescue it (14–30, 3.82% vs 5.16%), and the loss
concentrates exactly where the money is:

| ISO | GridPulse | Operator |
|---|---|---|
| ERCOT | 2.48% | **1.44%** |
| MISO | 3.47% | **2.43%** |
| PJM | 4.44% | **3.69%** |
| NYISO | 5.25% | **2.06%** |
| ISONE | 7.90% | **3.43%** |

We lose on five of six major ISOs. We win, by large margins, on smaller BAs whose own
forecasting is weak:

| BA | GridPulse | Operator |
|---|---|---|
| PSEI | **3.59%** | 40.9% |
| FMPP | **5.52%** | 28.15% |
| FPC | **6.41%** | 22.0% |
| CAISO | **4.88%** | 8.77% |

**The wedge is a floor, not a ceiling.** Not "better than your forecast" but "no BA
left catastrophically unforecast." That is a real product for someone holding a
portfolio spanning many small BAs. It is not a product for a desk trading ERCOT.

Sharpening the same point: `docs/PERSISTENCE_SKILL.md` shows **35 of 44** BAs beat a
seasonal-naive baseline (median +0.83 pts), and **9 do not**. SEC was −6.36 and is now
served the baseline instead of the model (ADR-pending, flag `baseline_substitution`).
A forecasting product with regions that lose to "yesterday, same hour" has a floor
problem before it has a differentiation problem.

## 2. Is Orrery a competitor?

**Not today.** [orreryhq.com](https://orreryhq.com) (New Earth Technologies) sells
derived *weather* forecasts by the call. Verified by calling the live API:

- Serves **NOAA GFS 0.25°** only — temperature, 10m/80m wind, GHI — plus NWS station
  observations and a derived hub-height wind product.
- The advertised ECMWF IFS / HRRR / GEFS / NBM / CAMS are **not served yet**. They gate
  serving on published verification, which is a discipline worth borrowing.
- **No dollar figures anywhere**, despite marketing "published prices, metered per
  call". The metering *model* is specified; the prices are not.
- Maturity: ~30-day-old domain, apparently one engineer, no funding, no customers, no
  press. The copy reads bigger than the company is.

They sit at the **weather-ingestion layer** — where we use Open-Meteo + NBM +
multipoint — so they are closer to an Open-Meteo substitute than a rival. Currently a
worse one for us: 0.25° GFS is coarser than the NBM-class inputs we already run.

**The signal that matters is forward-looking.** `load`, `lmp`, `wind_generation` and
`solar_generation` are already first-class members of their canonical schema, and
**`/v1/energy/load` is a live, documented, BA-keyed route returning `series: null`** —
declared surface, no data behind it. Adding demand is a connector job for them, not a
redesign.

**Watch item:** if a load connector lands with real data, they become adjacent
immediately, and they will arrive with commercially-licensed weather inputs and
working metering — the two things we lack.

**They independently converged on our honesty standard.** Their baseline is *"24-hour
persistence — yesterday's observed value. Positive means the model beats it"* — the
same construct as `models/skill.py`, shipped the same week. Two parties arriving at
the same standard is evidence it is the right one, and evidence it will not
differentiate for long.

Their verification is genuinely live with real MAE/RMSE/bias/skill numbers — more than
most vendors do — but it is 16 days deep, covers GFS/ECMWF only, and **explicitly does
not score the hub-height wind that is their flagship product.**

## 3. The competitive set, and what actually differentiates

The market is stratified: enterprise incumbents (Enverus, Yes Energy, Ascend, Itron,
Hitachi, Wood Mackenzie/Genscape, DNV, UL), a funded US startup tier (Amperon,
Gridmatic, Tyba, Modo, Camus), a European API tier (Dexter, rebase.energy, Jua,
Solcast, Meteomatics), and a **free substrate** — EIA-930, ISO forecasts, gridstatus,
Open-Meteo, NOAA NBM — that already gives the named buyers a serviceable forecast at
zero cost.

**Publishing accuracy against the free incumbent is table stakes, not a
differentiator.** Enverus and Amperon both publish head-to-head MAPE-vs-ISO figures as
marketing. *(Caveat, from checking: the specific Amperon example is a self-selected
four-day window quoting only relative improvement, and a widely-repeated Enverus
"market share" figure is vendor marketing about client-attributed capacity. The
pattern holds; the individual numbers should not be repeated.)*

**Skill against a naive baseline is standard in evaluation, rare in marketing.** DOE/EPRI's
Solar Forecast Arbiter generates persistence benchmarks; IEA Wind Task 36 built a
recommended practice around skill scores. A technical buyer will recognise it
instantly and will not be impressed by it as an innovation — but its absence would be
noticed.

**The genuine white space is narrower than "we publish our accuracy":**

> Publishing **continuously, pre-sale, across every region, including the regions where
> we lose.** Every vendor example found was a selected window showing a win.

*Inferred, not established:* the market's real verification norm appears to be private
anonymised trials — EPRI ran a multi-vendor comparison designed around anonymity — with
public numbers serving as top-of-funnel marketing rather than the decision mechanism.
If that holds, public transparency wins *attention*, not *deals*.

## 4. Assets, ranked by how hard they are to copy

| Asset | Verdict | Why |
|---|---|---|
| **Vintage instrument** — what EIA first published vs what settled, per hour, per BA | **Real barrier, small moat** | EIA does not republish its own revision history, so this genuinely cannot be backfilled. But ours is a rolling 30-day window, not an accumulating archive — a competitor reaches parity 30 days after they start. Head start, not a data moat. *(A production anomaly wiped parts of this window on 07-16/17; it stopped, the defence is armed, and the window has rebuilt cleanly since — see #313.)* |
| **ADR-010 serve-path acceptance gate** | **Head start** | Hardest thing here to arrive at independently — it took an incident plus replaying 67 persisted model vintages. No IP, no lock-in, and the whole diagnosis is published in this repo. Most teams ship holdout MAPE and never discover the failure mode. |
| **Published benchmark methodology** | **Institutionally expensive to copy** | Technically 2–4 weeks of work. What a competitor will not do is publish a scorecard on which they lose 28 of 43, with the exclusions and limits attached. |
| **Skill-vs-naive across the fleet** | **Textbook** | The measurement is standard. *Acting* on it in production — serving the baseline where the model loses — is not. |
| **51-BA coverage + weather integration** | **Commodity** | Weeks of work for a funded team. Nothing proprietary in the inputs. The measured A/B studies (NBM +0.92 pts, multipoint +1.14 pts) are a modest head start. |

**The pattern:** what is differentiated is the *measurement apparatus*, not the
forecast. That inverts the natural pitch, and it is the honest read.

## 5. What blocks selling the API today

Two are hard blockers. Neither is about product quality.

**1. Licensing — `assets/ba_polygons.geojson` is AGPL-3.0**
([#357](https://github.com/kristenmartino/gridpulse/issues/357)). Established, not
speculative: upstream relicensed 2023-01-30, and commit `83cfc4fe` ("Changes borders
of El Paso and ERCOT") edited geometries in our 51-BA set *after* that date. AGPL §13
extends copyleft to network-served works. No confirmed replacement source yet — EIA
publishes no BA boundary layer (all 79 of its public feature services enumerated).

**2. Licensing — Open-Meteo's free tier is non-commercial.** Verbatim: *"You may only
use the free API services for non-commercial purposes."* We re-serve derived values
with `Access-Control-Allow-Origin: *`. Fixable with a paid tier; note Orrery avoided
this entirely by building on ECMWF/NOAA/Copernicus.

**3. There is no customer.** No API keys, no auth, no accounts, no usage records, no
billing substrate. The only per-caller artifact is a 60-second-TTL per-IP rate-limit
counter that fails open by design, and request logs omit the caller — so we cannot
reconstruct usage to *price* a plan, let alone invoice one.

**4. No SLA furniture** — no OpenAPI schema, changelog, deprecation policy, status
page, support contact, or uptime target. A build list, not a blocker.

**What survives diligence intact:** the honesty layer — allow-listed export, 503
warming instead of fabricated data, provenance on every payload, published
methodology, contract tests. That is the expensive half of a data API and it is done
to a standard most vendors do not reach.

## 6. The strategic read

Three options, in descending order of how well the evidence supports them.

**A. Sell the instrumentation, not the forecast.** The vintage instrument answers
"which hours of EIA-930 can I trust?" for anyone building on that feed — including
Orrery, whose `/v1/energy/load` stub implies they will need exactly this. This is the
only offer where our measured position is a strength rather than something to explain.
Weakness: small market, and the asset is a 30-day head start rather than a moat.

**B. Sell the floor, to portfolio holders across many small BAs.** Supported by the
data (PSEI 40.9% → 3.59%, FMPP 28.15% → 5.52%) and honestly framed as consistency
rather than superiority. Weakness: the buyers with budget are on the ISOs, where we
lose.

**C. Sell demand forecasts against the incumbent.** **Not supported.** Our own
published benchmark contradicts it on five of six major ISOs.

**The near-term recommendation is neither A nor B: it is to keep measuring.** The
benchmark is 1–2 days old, the skill layer is 1 day old, and the substitution flipped
today. The most valuable thing available right now is a longer, stable run of the
instrument we just built — which also happens to be the asset with the shortest shelf
life if a competitor starts accumulating their own.

## 7. Open questions worth resolving before any commercial conversation

- [#357](https://github.com/kristenmartino/gridpulse/issues/357) — AGPL asset; needs a
  replacement source or a compliance decision.
- ~~[#313](https://github.com/kristenmartino/gridpulse/issues/313) — vintage windows
  destructively re-pinned in production~~ — **closed 2026-07-28, and this doc
  overstated it.** The anomaly stopped on 2026-07-17 and the instrument has
  accumulated cleanly since: the tombstone has fired **zero** times in 13 days, as
  have both drift read-failure events, so it is not "defended and ongoing" — it is
  gone. A log audit did confirm the damage was real while it lasted (8 reset ticks,
  15 drift-window wipes across CAISO/ERCOT/FPL/PJM, 07-16 to 07-17), and found that
  the resets hit *several regions at the same instant* — a property of the tick, not
  the region, which is the thread to pull if it ever recurs. The tombstone stays armed
  as the tripwire.
- [#349](https://github.com/kristenmartino/gridpulse/issues/349) — the quality gate
  judges the holdout against the 7-day band, so a BA failing at 24h passes silently.
- [#358](https://github.com/kristenmartino/gridpulse/issues/358) — backfilled hours
  scored on the as-issued benchmark arm.
- Open-Meteo paid tier — cost unknown, and needed on volume grounds independent of the
  licence question.
