# Behind-the-meter solar: not supported

**Verdict: the BTM hypothesis fails on its own sign test.** Two of three
falsifiable predictions fail, and the third points the other way once read
alongside the levels. No feature work follows from this.

Run 2026-07-30, 8 BAs × 6 rolling windows. Reproduce:
`python -m scripts.btm_solar_probe --windows 6`. Raw:
[`BTM_SOLAR_PROBE.json`](BTM_SOLAR_PROBE.json).

---

## The hypothesis, and why it was already weak

[`COOLING_RESPONSE_STUDY.md`](COOLING_RESPONSE_STUDY.md) established what the
hot-hour error is *not*: with perfect future weather, better temperature
features moved nothing. Rooftop PV was the surviving explanation — hot
afternoons are peak-irradiance afternoons, PV suppresses net load exactly at
cooling peak, and installed capacity grows quarterly while the model never
sees it.

Two facts weaken that before any measurement, and the probe was written to
say so up front:

* EIA-930 `D` is **metered grid load**, already net of BTM generation.
* The model is already given `solar_capacity_factor`, raw
  `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation` and
  `cloud_cover`.

So the model can already learn average suppression. The hypothesis survives
only if **residual** structure remains after that.

This probed residuals rather than building features — the explicit lesson from
the cooling pack, which was built first, measured second, and failed.

## The three predictions, and what happened

### 1. Sign — **FAILS**, and this is the decisive one

Unmodelled PV suppression means we forecast load that PV then removes:
**over**-forecasting, positive signed error, concentrated at high irradiance.

| BA | BTM rank | mean signed % (daylight) | Q1 irradiance | Q5 irradiance | Q5 positive? |
|---|---:|---:|---:|---:|:--|
| CAISO | 5 | −1.11 | −1.14 | +0.72 | yes |
| FPL | 4 | −1.06 | −0.49 | −1.42 | no |
| SOCO | 3 | −0.44 | −0.12 | −1.15 | no |
| ISONE | 3 | +0.07 | −0.95 | +2.15 | yes |
| TVA | 2 | −0.67 | −1.18 | −0.61 | no |
| NYISO | 2 | −0.08 | −1.94 | +1.11 | yes |
| ERCOT | 2 | −0.86 | −0.34 | −1.21 | no |
| MISO | 1 | −0.94 | −0.41 | −0.84 | no |

Signed error at the highest irradiance quintile is positive in **3 of 8** BAs.
BTM predicts 8 of 8.

**Mean signed error across all daylight hours is −0.636%.** We systematically
**under**-forecast in daylight. That is the opposite of the hypothesis, and it
needs no external data or assumption to read.

### 2. Survives a temperature control — holds, but is not diagnostic

Within the hot tercile the irradiance gradient averages **+2.76 pts** across
the 7 BAs where it is measurable. So real residual structure vs irradiance
does exist beyond temperature.

But with prediction 1 failed, that structure reads the other way: the levels
are negative, so the finding is **we under-forecast most on hot,
low-irradiance hours** — hot and overcast. That is muggy, high-cooling-load,
low-solar-relief weather, not a PV signature.

The control is also thin: hot *and* dark hours barely exist, so one BA had no
measurable within-hot gradient at all.

### 3. Dose-response with penetration — **FAILS**

`corr(BTM rank, gradient) = +0.14` — essentially nothing. The ordering is
wrong in detail too: the highest-penetration BA (CAISO) shows +1.85 while
FPL, the second highest, shows **−0.93**; the two largest gradients belong to
ISONE (+3.10) and NYISO (+3.05), neither a high-penetration system.

**Caveat that limits how much this test is worth:** `BTM_RANK` is my own
ordinal guess, not EIA-861M capacity data. A failed dose-response against a
guessed ranking is weak evidence. It is reported for completeness, and the
verdict does not rest on it — prediction 1 does, and prediction 1 needs no
ranking at all.

## What the data actually says

1. **The daylight bias is negative, not positive.** −0.636% mean. We under-
   forecast during the day, and under-forecasting demand is the operationally
   expensive direction — the same asymmetry that made WAPE-plus-bias-band the
   evaluation policy rather than MAPE.
2. **The residual structure is hot-and-cloudy, not hot-and-sunny.** Whatever
   is missing shows up worst when it is hot with low irradiance.
3. That is a humidity/latent-load-shaped story — which is uncomfortable,
   because `cdd_x_humidity` and the NWS heat index were in the cooling pack
   and that pack did nothing. Either those particular encodings are wrong for
   it, or the cause is not weather-functional at all.

## Where that leaves the hot-hour error

Three explanations have now been tested or set aside:

| explanation | status |
|---|---|
| temperature representation | **rejected** — cooling pack, 8/8 inconclusive with perfect weather |
| behind-the-meter solar | **rejected** — sign test fails 5/8, no dose-response |
| trees already learn the interactions | untested, and now the most economical explanation |

### The premise itself, checked — and it holds

Before hunting further causes it was worth asking whether the hot-hour
concentration is simply where the *load* is: more MW at peak produces more MW
of error at the same percentage, with no extra difficulty involved. That check
is cheap (demand and temperature only, no model), so it was run rather than
left as a suggestion:

| | hot quintile |
|---|---:|
| share of MW **served** | **24.3%** |
| share of MW **error** | **34.7%** |
| ratio | **1.43×** |

Hot hours are **genuinely harder**, not merely bigger, in 7 of 8 BAs (MISO
1.53, ISONE 1.71, PJM 1.58, NYISO 1.49, SOCO 1.48, ERCOT 1.33, TVA 1.26). The
error analysis premise survives.

**FPL is the exception at 1.04** — there the "hot-hour concentration" really
was just load size, and FPL should be dropped from any future hot-hour work.

So the remaining untested candidates are demand response / price-driven
curtailment at peak, and the economical possibility that gradient-boosted
trees already extract what these hand-built encodings offer.

## Limits

1. **Six windows, summer only** (June–July 2026). A winter probe would test
   the mirror-image prediction on heating.
2. **XGBoost proxy**, not the served ensemble — same limit as
   `ERROR_ANALYSIS.md`.
3. **`BTM_RANK` is a guess.** Prediction 3 is therefore weak evidence either
   way; the verdict rests on prediction 1.
4. **The within-hot control has thin overlap** — hot and low-irradiance hours
   are rare by construction, and one BA had none.
5. Daylight defined as irradiance > 50 W/m², which folds dawn/dusk into the
   lowest quintile.
