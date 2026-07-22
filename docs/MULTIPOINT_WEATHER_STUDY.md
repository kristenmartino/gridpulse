# Multi-point / population-weighted weather study

Sample: MISO, PJM, SPP, SOCO, ERCOT + GVL (control) · K=12 cells · 10 windows/BA · arms: A single-point, B unweighted mean, C population-weighted.

## Summary

**ADOPT multi-point weather.** Aggregating several points across a BA's footprint beats the single representative point by a mean **+1.14 sMAPE pts** across the large multi-state sample — largest on the geographically-spread BAs (MISO +1.77, PJM +1.41, SPP +1.45), smallest on the more compact ones, and the GVL control (one county) reads zero.

**Population weighting adds essentially nothing** (weighting lift C−B = +0.03 pts): unweighted averaging (arm B, +1.11) captures nearly the entire gain. The benefit is spatial averaging, not load-weighting — so a production adoption can skip the census/population machinery and simple-average N footprint points. (This contradicts the literature's utility-ZONAL finding that weighting beats averaging; at BA-aggregate scale the demand series has already integrated the load distribution.)

## C (pop-weighted) vs single-point A (positive = beats A)

Mean Δ **+1.141** sMAPE pts over 50 paired (BA, window) cells; win rate 90%; big wins (≥1.0): 3 BA(s); worst BA +0.306.

| ba | mean_delta_pts |
|---|---|
| ERCOT | 0.757 |
| MISO | 1.773 |
| PJM | 1.414 |
| SOCO | 0.306 |
| SPP | 1.453 |

**Verdict: ADOPT**

## B (unweighted multi-point) vs single-point A (positive = beats A)

Mean Δ **+1.107** sMAPE pts over 50 paired (BA, window) cells; win rate 88%; big wins (≥1.0): 3 BA(s); worst BA +0.234.

| ba | mean_delta_pts |
|---|---|
| ERCOT | 0.819 |
| MISO | 1.745 |
| PJM | 1.395 |
| SOCO | 0.234 |
| SPP | 1.341 |

**Verdict: ADOPT**

## Weighting effect (C − B, positive = weighting helps)

| ba | mean_delta_pts |
|---|---|
| ERCOT | -0.062 |
| MISO | 0.028 |
| PJM | 0.019 |
| SOCO | 0.072 |
| SPP | 0.112 |

## Control (GVL, ≈one county)

Mean Δ(A−C) **+0.000** pts (gate |Δ| ≤ 0.15). ✓ within noise — harness sound.
