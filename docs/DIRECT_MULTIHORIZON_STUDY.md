# Direct multi-horizon vs recursive (#230): not a rewrite, possibly a per-BA choice

**Verdict: the prototype does not clear as a general change.** 10 BAs, 5
rolling windows each, 168h horizon. One decisive win and one decisive loss
that cancel; 5 better, 5 worse; only 1 of 10 ships.

**But there is a real conditional signal:** direct helps where recursion is
already struggling and hurts where it is fine. That is a per-BA strategy
question, not a rewrite — and it needs a fleet run with a pre-registered
threshold before anyone acts on it.

Run 2026-07-30. Reproduce: `python -m scripts.direct_multihorizon_study`.
Raw: [`DIRECT_MULTIHORIZON_STUDY.json`](DIRECT_MULTIHORIZON_STUDY.json).

---

## Arms

Identical data, features and weather. Only the horizon strategy differs.

* **recursive (control)** — production's own
  `recursive_autoregressive_forecast`, the single source of truth for both
  production scoring and holdout evaluation. Chains its own predictions
  forward across 168 steps.
* **direct (treatment)** — one model on
  `(features known at origin, horizon h) → demand@origin+h`, autoregressive
  block frozen at the origin, `horizon_h` as a feature. Nothing compounds.

## Results

| BA | recursive WAPE | direct WAPE | Δ pts | consistency | ×stderr | verdict |
|---|---:|---:|---:|---:|---:|---|
| NYISO | 10.164 | **5.872** | **+4.292** | 60% | 1.5 | inconclusive |
| ISONE | 8.453 | **6.186** | **+2.267** | 80% | 1.5 | inconclusive |
| PJM | 5.622 | **4.498** | **+1.124** | 100% | 2.8 | **DIRECT** |
| TVA | 4.535 | 4.052 | +0.483 | 60% | 0.8 | inconclusive |
| CAISO | 6.101 | 5.827 | +0.274 | 60% | 0.2 | inconclusive |
| ERCOT | 2.475 | 2.568 | −0.093 | 80% | 0.4 | inconclusive |
| SOCO | 3.478 | 3.611 | −0.133 | 60% | 0.6 | inconclusive |
| MISO | 4.979 | 5.116 | −0.136 | 60% | 0.3 | inconclusive |
| FPL | 3.517 | 3.930 | −0.413 | 60% | 1.0 | inconclusive |
| SPP | 4.917 | **6.122** | **−1.206** | 100% | 4.6 | **RECURSIVE** |

Mean **+0.646**, median **+0.091**. The gap between them is the
outlier-domination signature `models/rolling_eval.py` exists to flag: the mean
is carried by NYISO and ISONE.

## The conditional signal

Direct's advantage tracks how badly recursion is doing:

`corr(mean-of-arms WAPE, Δ) = **+0.737**`

Split at the median difficulty (5.05 WAPE):

| | n | mean Δ | better |
|---|---:|---:|---:|
| harder BAs (PJM, ISONE, SPP, CAISO, NYISO) | 5 | **+1.350** | 4/5 |
| easier BAs (MISO, ERCOT, SOCO, TVA, FPL) | 5 | −0.058 | 1/5 |

Mechanistically this is what error accumulation predicts: recursion's penalty
grows with how wrong each step is, so it costs most where the model is
already struggling. Where the one-step model is accurate, chaining it is
nearly free and the direct arm gives up its sharper autoregressive signal for
nothing.

**Two honesty notes on this number.** First, `Δ = recursive − direct` shares a
term with `recursive`, so the obvious `corr(recursive, Δ) = +0.872` is
mechanically inflated; +0.737 above uses the mean of both arms, which is
symmetric. Second, **the threshold was chosen after seeing the data**. This is
hypothesis-generating, not a confirmed rule.

**And SPP breaks it.** SPP sits in the harder group and is the study's single
decisive loss (−1.206 at 4.6× stderr, control winning 100% of windows). A rule
that fits 4 of 5 with a decisive counterexample is not a rule yet.

## Bias — the reason this is not a free option

| | mean bias |
|---|---:|
| recursive | **−0.636%** |
| direct | **−1.848%** |

Direct under-forecasts roughly three times as much, and under-forecasting
demand is the operationally expensive direction — the whole reason
`EVALUATION_POLICY.md` optimises WAPE with a bias band rather than MAPE. Both
arms breach the ±2% band in 3 of 10 BAs, but direct sits far closer to it on
average (MISO −4.51%, SPP −3.44%, CAISO −2.94%). Any per-BA adoption has to
carry that constraint, not just the WAPE comparison.

## Recommendation

1. **Do not rewrite the forecasting strategy.** #230's own gate was "prototype
   on 3–4 BAs and measure before committing." Measured on 10, it does not
   clear.
2. **Do not discard it either.** The conditional signal is strong enough, and
   the upside where it lands is large (NYISO 10.16 → 5.87 is a 42% error
   reduction; ISONE 27%).
3. **Next test, if this is picked up:** a full 51-BA run with the threshold
   **pre-registered** before the run, and SPP's counterexample specifically
   probed. If the rule survives that, the change is not a rewrite — it is
   per-BA strategy selection, which the serve-path acceptance gate (ADR-010)
   is already shaped to carry.

## A process note worth keeping

The first version of this study sampled **14 horizons** for training and let
`horizon_h` interpolate across all 168. Gradient-boosted trees do not
interpolate, they split, so unsampled horizons were served by whichever bucket
they fell into. That version measured direct as **−0.757 pts on PJM**.
Training on all 168 horizons instead flipped it to **+0.754** — a 1.5-point
swing from my own sampling choice, in the arm being tested.

Had that first number shipped, it would have rejected #230 on an
implementation artifact and reported the opposite conclusion.

## Limits

1. **10 BAs, 5 windows.** Chosen as the BAs carrying 77.5% of fleet MW error
   (`ERROR_ANALYSIS.md`) — not a random sample, and deliberately weighted to
   where a gain would matter.
2. **The difficulty threshold is post-hoc.** Stated above; it is the main
   reason the recommendation is "test it properly" rather than "adopt it".
3. **XGBoost only.** Prophet and SARIMAX are natively multi-step and this
   comparison says nothing about them, nor about the served ensemble.
4. **Perfect future weather** for both arms. Fair as a controlled comparison,
   but it removes the weather-forecast error that grows with horizon — which
   plausibly *favours* the direct arm, since recursion at least re-anchors on
   its own trajectory.
5. **Origins strided at 12h** to keep the direct training frame near ~31k rows.
   Denser origins might change the direct arm's ceiling.
6. Production re-anchors hourly, so as #230 itself notes, the 168h number
   measures cold-forecast capability rather than lived near-term accuracy.
