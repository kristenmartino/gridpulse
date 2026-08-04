# The zonal effect is a cooling-season phenomenon — the first mechanism to survive

**Prediction 2 confirmed.** NYISO's bottom-up gain falls from **+0.729
(decisive)** in summer to **+0.149 (inconclusive)** in winter — below the
pre-registered "< +0.40 or inconclusive" threshold on both counts. CAISO falls
from +0.283 to **+0.015**.

After six refuted mechanisms, seasonality is the first to hold.

Run 2026-08-04 against
[`WINTER_RUN_PREREGISTRATION.md`](WINTER_RUN_PREREGISTRATION.md), committed
before the run. Raw: [`WINTER_NYISO_STUDY.json`](WINTER_NYISO_STUDY.json) ·
[`WINTER_CAISO_STUDY.json`](WINTER_CAISO_STUDY.json).
Reproduce: `python -m scripts.nyiso_bottom_up_study --iso NYISO --end 2026-03-01`.

---

## Result

| | NYISO summer | **NYISO winter** | CAISO summer | **CAISO winter** |
|---|---:|---:|---:|---:|
| **gain** | **+0.729** | **+0.149** | +0.283 | **+0.015** |
| verdict | **decisive** (3.7× se) | inconclusive (t=1.32) | inconclusive | inconclusive |
| windows | 6 | 4 | 6 | 6 |
| MDE | 0.390 | 0.226 | 0.334 | 0.202 |
| top-down WAPE | 3.958 | **2.237** | 3.406 | **2.326** |
| pure-load channel | +0.349 | +0.142 | +0.023 | +0.069 |

**The point estimate is what matters here, not the verdict.** NYISO winter's
+0.149 is one fifth of summer's +0.729. Even granting winter fewer windows and
a lower detection floor, +0.149 is nowhere near +0.729 — this is a real
collapse, not a power artifact.

## The mechanism, and why it is coherent

**Winter load is far easier to forecast, and there is correspondingly less for
zonal decomposition to recover.** Top-down WAPE falls from 3.958 to 2.237 on
NYISO and 3.406 to 2.326 on CAISO — winter error is roughly 40% lower before
any decomposition is attempted.

The reason is structural: New York heats largely with gas and oil, so winter
*electric* load is much less temperature-sensitive than summer load, which is
air-conditioning-driven. In summer the load–temperature response is steep,
zones sit at different points on it, and splitting the forecast pays. In
winter the response is shallow, the zones behave alike, and there is nothing
to exploit.

This is the same shape as the `NYISO_ZONAL_PROBE` finding that the
weather-diversity effect appeared in the cool and hot temperature bands but
vanished in the mid band — a steep response is the precondition.

## What it does *not* explain

**Seasonality does not close the cross-ISO gap.** At the same season, NYISO
gains +0.729 and CAISO +0.283. Both fall toward zero in winter, so season is a
real driver of the *variation* — but at matched season the two ISOs still
differ by more than 2×.

So the honest state is:

* **Seasonality: supported.** First surviving mechanism of seven.
* **The NYISO/CAISO difference: still unexplained.** Six mechanisms refuted,
  and seasonality accounts for the within-ISO variation rather than the
  between-ISO one.

## What this licenses

The pre-registration said: *NYISO falls → seasonality is a live mechanism,
worth one follow-up isolating heating vs cooling.*

But the practical reading is blunter. The effect that motivated this entire
line — a decisive +0.729 — exists **only in the cooling season, on one ISO,
and remains unexplained across ISOs**. That is:

* not a foundation for seven integrations,
* not a foundation for one integration,
* and its realistic ceiling is now visibly smaller: whatever a zonal build
  bought would be seasonal, so the annualised gain is well under the headline.

**Recommendation: close the zonal line.** Not because it was worthless — the
NYISO effect is real and this is now the best-characterised negative in the
project — but because the remaining upside is a seasonal fraction of a
one-ISO effect with no working cross-ISO explanation, and the only evidence
that could change that (PJM/ISO-NE) needs credentials a human must create.

If it is ever reopened, the entry point is a third ISO in *summer*, not more
work on these two.

## Limits

1. **NYISO winter scored 4 windows, not 6.** NYISO's archive fetch grabs whole
   calendar months while the weather fetch uses an exact span, so the winter
   overlap was 2011 rows against summer's 2588. MDE 0.226. The conclusion
   rests on the point estimate (+0.149 vs +0.729), not on the verdict.
2. **"Summer" is March–July and "winter" is October–March** for CAISO — the
   OASIS chunking follows the exact span while NYISO follows calendar months,
   so the two ISOs' windows are not identically aligned.
3. Four months per season, one year. No multi-year replication.
4. XGBoost day-ahead, not the served recursive ensemble; target is the zone
   sum, not EIA `D` (see `NYISO_BOTTOM_UP_STUDY.md` for that 2.70%/3.34% gap).
5. The heating-vs-cooling mechanism is inferred from the WAPE levels and NY's
   fuel mix, not measured directly. A heating-degree-day interaction test would
   confirm it; it is not run here because the recommendation is to close the
   line either way.
