# Error analysis: where our forecast error actually is

Run 2026-07-30. Reproduce: `python -m scripts.error_analysis --windows 6`.
Raw results: [`ERROR_ANALYSIS.json`](ERROR_ANALYSIS.json).

The measurement apparatus has been good for months and work has still been
chosen from an issue queue. This is the missing step: bucket the errors by
cause and size each bucket, so the next change is picked by evidence about
where the error *is*.

Two parts. Part 1 needs no modelling and already changes the picture.

---

## Part 1 — the fleet's error is not where the scorecard points

Percentages weight every BA equally. Operations does not: 20% of a 300 MW
co-op is 60 MW, 2% of PJM is ~2,000 MW. Converting the live benchmark to **MW
of error** (MAE × hours, normalised to a common 720 h month so unequal paired
hours cannot tilt it):

| | ours | operators |
|---|---:|---:|
| Fleet total absolute error | **16.46M MWh** | 17.99M MWh |

**In MW across the fleet we are 8.5% better than the incumbents** — the
opposite of the published headline, which counts BAs and reports the operator
closer on 27 of 43. Both are true. They answer different questions, and only
one of them is about megawatts.

**Error is heavily concentrated.** The top 10 BAs carry **77.5%** of all fleet
MW error (PJM 19.8%, MISO 13.0%, ERCOT 7.5%, SPP 7.2%, CAISO 6.8%, SOCO 5.8%,
ISONE 4.7%, TVA 4.4%, NYISO 4.3%, FPL 4.1%).

**And the BAs that look worst are nearly irrelevant in MW:**

| BA | MAPE (what the scorecard shows) | share of fleet MW error |
|---|---:|---:|
| SEC | 17.93% | **0.30%** |
| WALC | 8.16% | 0.52% |
| JEA | 6.42% | 0.66% |
| SRP | 7.98% | 2.43% |

SEC consumed multiple sessions of work. It is three tenths of one percent of
the megawatts.

**The addressable gap.** Splitting BAs by whether we beat the operator:

- 16 BAs where we are **better**, saving 5.15M MWh
- 28 BAs where we are **worse**, costing 3.62M MWh

and 82% of that 3.62M sits in **eight** BAs:

| BA | ours | operator | cost | share of gap |
|---|---:|---:|---:|---:|
| MISO | 3.41% | 2.41% | 538k MWh/mo | 14.9% |
| ERCOT | 2.43% | 1.43% | 500k | 13.8% |
| ISONE | 7.59% | 3.29% | 442k | 12.2% |
| NYISO | 5.04% | 2.07% | 407k | 11.2% |
| PJM | 4.10% | 3.53% | 407k | 11.2% |
| TVA | 4.34% | 2.29% | 331k | 9.1% |
| FPL | 4.35% | 3.04% | 192k | 5.3% |
| SOCO | 3.95% | 3.25% | 170k | 4.7% |

Part 2 analyses exactly those eight.

## Part 2 — what the errors are made of

Per hour, three arms: **ours** (reconstructed locally, day-ahead), the
**operator's** own EIA-930 forecast, and **seasonal-naive**. Six rolling
168-hour windows per BA via `models/rolling_eval.py`.

### The addressable share is large

An hour is "missed" at >5% absolute error. Shares are of our **total MW
error**, not of hours:

| BA | ours | oper | naive | both missed | **only we missed** | only oper | neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| MISO | 2.85 | 2.91 | 4.68 | 9.1% | **39.9%** | 8.1% | 42.8% |
| ERCOT | 1.74 | 1.64 | 3.22 | 1.4% | **19.1%** | 3.4% | 76.2% |
| ISONE | 4.30 | 3.07 | 10.66 | 6.8% | **58.7%** | 8.0% | 26.4% |
| NYISO | 3.77 | 2.39 | 8.69 | 8.1% | **52.5%** | 3.5% | 35.9% |
| PJM | 2.72 | 4.07 | 6.02 | 15.5% | **29.1%** | 17.8% | 37.7% |
| TVA | 2.62 | 2.69 | 5.88 | 8.4% | **28.2%** | 8.1% | 55.3% |
| FPL | 2.59 | 2.99 | 4.40 | 7.9% | **26.7%** | 10.2% | 55.2% |
| SOCO | 2.46 | 1.83 | 5.06 | 4.6% | **35.5%** | 3.6% | 56.3% |

*(WAPE %, the optimising metric per `EVALUATION_POLICY.md`.)*

**19–59% of our error is on hours the operator got right** — and our arm was
given *perfect* future weather. We had better information than they did and
still missed. That bucket is not weather-forecast error, and it is not
inherent difficulty. It is addressable.

The `neither missed` column (26–76%) is the ordinary grind: many small errors
that never cross 5% but sum to a lot of MW. No single fix touches it.

### Which axis explains it

**Temperature — the one clear signal.** Share of error in each temperature
quintile (flat would be 20%):

| BA | cold | cool | mild | warm | **hot** |
|---|---:|---:|---:|---:|---:|
| ISONE | 8.7 | 11.7 | 16.5 | 21.5 | **41.7** |
| PJM | 11.4 | 13.0 | 18.4 | 19.1 | **38.2** |
| MISO | 9.7 | 13.0 | 15.1 | 25.0 | **37.3** |
| SOCO | 11.9 | 12.2 | 17.3 | 21.8 | **36.7** |
| NYISO | 10.1 | 12.0 | 18.5 | 23.9 | **35.5** |
| ERCOT | 13.2 | 14.4 | 18.3 | 22.3 | **31.8** |
| TVA | 13.0 | 16.6 | 19.5 | 19.6 | **31.3** |
| FPL | 17.4 | 16.8 | 16.0 | 24.6 | **25.2** |

Mean **34.7%** in the hot quintile against **11.9%** in the cold. Monotone in
7 of 8 BAs. Cooling response is where the error is.

**Holidays — specific and actionable.** Ratio of error share to hour share
(>1.0 = over-represented):

| BA | holiday ratio | weekend ratio |
|---|---:|---:|
| ISONE | **1.87** | 1.19 |
| NYISO | **1.69** | 1.06 |
| TVA | 1.34 | 0.99 |
| SOCO | 1.35 | 1.15 |
| PJM | 1.16 | 0.93 |
| FPL | 0.72 | 0.92 |
| MISO | 0.66 | 0.92 |
| ERCOT | 0.63 | 0.85 |

The four BAs where holidays hurt most are four of the top seven in the
addressable gap. `is_holiday` is a single binary flag — no day-before/after,
no regional variation, no bridge days.

**What does *not* explain it.** Ramp magnitude is flat (Q1 19–29%, Q4 21–30%
against 25% expected) — "we are bad at ramps" is not supported. Hour of day is
flat too (top hours 6–7% against 4.2% uniform), and the mild peak at 19–23h
UTC is the same afternoon/evening heat, not a separate cause. Weekends are
broadly fine.

## What this says to do next

1. **Cooling response**, the biggest bucket in every BA analysed. CDD uses a
   fixed 65°F baseline fleet-wide; there is no humidity×temperature
   interaction and no saturation term for very hot hours. Cheap to test now
   that the harness exists.
2. **Holiday features** for ISONE / NYISO / TVA / SOCO. A binary flag against
   a 1.87× over-representation is a small change against a measured target.
3. **ISONE** deserves its own look — worst on every axis here: highest
   addressable share (58.7%), highest hot concentration (41.7%), worst holiday
   ratio (1.87×), and the only BA where ramps show anything (Q4 30.1%).
4. **Reconsider direct multi-horizon ([#230])**, which I had argued against on
   the grounds that model work was not where the error was. This
   reconstruction is a *direct* day-ahead model and beat production's live
   number on several BAs (PJM 2.72 vs 4.10 MAPE, MISO 2.85 vs 3.41). Confounded
   by perfect weather — but the gap is large enough to be worth a controlled
   test rather than a dismissal.

**Not supported by this analysis:** ramp-specific modelling, hour-of-day
corrections, and further per-BA work on small BAs like SEC.

## Limits

1. **Perfect future weather.** Our arm gets the ERA5 archive; production gets
   a forecast. This flatters our arm against both production and the operator.
   It does *not* soften the addressable finding — an error made with perfect
   weather cannot be blamed on weather — but every head-to-head WAPE here is
   optimistic.
2. **XGBoost alone, not the served ensemble** (ADR-005's primary model, used
   as a tractable proxy). CV MAPE tracked production's live numbers closely
   (MISO ~3.0–3.4 vs live 3.41), which is why the proxy is credible, but it is
   still a proxy.
3. **Direct day-ahead, not recursive.** `make_day_ahead_safe` drops sub-24h
   lags and shifts rolling windows to issue time. Production instead predicts
   recursively through `_build_future_feature_frame`. These are *different
   forecasters*, which is exactly why finding 4 above is worth testing.
4. **Summer only.** Six 168-hour windows ending ~7 days back covers June–July
   2026. The hot-quintile dominance may be partly seasonal: a winter run could
   put the mass in the cold quintile. **Re-run in January before treating
   "cooling response" as the year-round answer.**
5. **Eight BAs, not 51** — chosen as 82% of the addressable gap, so this says
   nothing about the other 43.
6. The 5% miss threshold is a judgement call; the bucket *shares* move with it
   even though the ranking does not.

[#230]: https://github.com/kristenmartino/gridpulse/issues/230
