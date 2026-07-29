"""Rolling-origin evaluation and a single declared decision metric.

Two problems this exists to fix, both self-inflicted and both documented.

**1. One holdout window cannot rank anything.** The ARIMA order study
(`docs/ARIMA_ORDER_EXOG_STUDY.md`) ran a 168-hour holdout on 2026-07-28 and
again on 07-29 — the same code, a window one day apart. CAISO's verdict
**reversed sign** (−7.24 → +3.87 pts), WALC's +4.59 gain became +0.04, and
MISO's *control* error nearly halved (7.62 → 3.91) without anything changing.
A correct decision shipped on four numbers, one of which was noise. Every
future A/B inherits that variance unless the harness handles it.

**2. Too many metrics decide things.** Serving decisions use mean MAPE
(`config.mape_grade`, the ADR-004 weights, the benchmark winner, the gate);
the A/B studies used sMAPE. Mixing them already produced a median-vs-mean
mix-up on a published page. So one metric decides — but *not* MAPE, and the
reason matters:

**MAPE is asymmetric against over-forecasting.** For one hour, ``|A-F|/A``
grows without bound as F exceeds A but caps at 100% as F approaches 0, so a
MAPE-minimising model is biased toward **under**-forecasting demand — the
expensive direction in grid operations, where under-procured reserves mean
scarcity risk. It also explodes on low denominators (SEC is a ~300 MW
co-op) and does not aggregate across BAs in any way that reflects total MW
error: 20% of SEC is 60 MW, 2% of PJM is ~2,000 MW.

So this module follows the optimizing/satisficing split rather than pretending
one number can do every job:

* **Optimizing — WAPE.** Scale-robust, no low-denominator blowup, aggregates
  to total MW error, peak hours dominate (correct for grid ops). Already
  computed in the benchmark payload.
* **Satisficing — bias band.** An arm cannot win by systematically
  under-forecasting; mean signed percentage error must stay inside
  ``MAX_ABS_BIAS_PCT``. This is precisely the guard MAPE lacks.
* **Satisficing — no MAPE regression** beyond ``MAX_MAPE_REGRESSION_PTS``.
  MAPE stays the *published* number, because comparability with EIA, the ISOs
  and every vendor scorecard is a real requirement — it just should not be the
  thing we optimise.

Known cost: a WAPE-optimising experiment can disagree with the MAPE-based
serving gate. That gap is real and the fix is to migrate the gate, not to
optimise a worse metric because the gate happens to read it.
"""

from __future__ import annotations

import numpy as np

#: The one metric an experiment is optimised on. WAPE, not MAPE — see the
#: module docstring for why MAPE's asymmetry points the wrong way for a grid.
DECISION_METRIC = "wape"

#: Published for external comparability, and constrained rather than optimised.
REPORTED_METRIC = "mape"

#: Satisficing: |mean signed percentage error| must stay inside this. Stops an
#: arm buying a WAPE win with a systematic under-forecast.
MAX_ABS_BIAS_PCT = 2.0

#: Satisficing: the published metric may not degrade by more than this, however
#: good the optimising metric looks.
MAX_MAPE_REGRESSION_PTS = 0.5

#: An arm must win in at least this fraction of windows. Guards against a
#: single catastrophic window driving the mean: the fleet ARIMA run had one BA
#: at −19.18 against a typical −0.2, which is a tail-risk finding, not evidence
#: that the average window favours either arm. Both facts matter and they are
#: different facts.
MIN_SIGN_CONSISTENCY = 0.75

#: |mean| must exceed this many standard errors. 2.0 ≈ the conventional 95%
#: paired threshold; with the window counts realistic here (8–12) this is
#: deliberately a rule of thumb, not an exact test — see `verdict`.
MIN_T_STATISTIC = 2.0

#: Below this, `verdict` refuses to call anything. One window is what produced
#: the CAISO reversal; two cannot distinguish a trend from a coin flip.
MIN_WINDOWS_FOR_A_VERDICT = 4


def rolling_origin_splits(
    n_rows: int,
    *,
    n_windows: int,
    holdout_h: int,
    stride_h: int | None = None,
    min_train_h: int,
) -> list[tuple[slice, slice]]:
    """Walk-forward (train, test) index slices, newest window first.

    Each window holds out ``holdout_h`` rows and trains on everything before
    them — never after, so no window can see its own future. Windows step back
    by ``stride_h`` (default: ``holdout_h``, i.e. adjacent non-overlapping
    holdouts).

    Returns fewer than ``n_windows`` splits — possibly zero — when history runs
    out. That is deliberate: a caller that silently accepted a short list would
    reproduce the failure mode this module exists to prevent, so callers should
    check ``len()`` and say what they got.

    Args:
        n_rows: Rows available, after feature engineering has dropped its warmup.
        n_windows: Maximum number of windows to produce.
        holdout_h: Rows held out per window.
        stride_h: Rows between consecutive window origins. Defaults to
            ``holdout_h`` (non-overlapping holdouts).
        min_train_h: A window is dropped if its train slice is shorter than this.

    Returns:
        ``[(train_slice, test_slice), ...]``, most recent window first.
    """
    if holdout_h <= 0 or n_windows <= 0:
        return []
    step = holdout_h if stride_h is None else stride_h
    if step <= 0:
        return []

    splits: list[tuple[slice, slice]] = []
    for k in range(n_windows):
        test_end = n_rows - k * step
        test_start = test_end - holdout_h
        if test_start < min_train_h:
            break
        splits.append((slice(0, test_start), slice(test_start, test_end)))
    return splits


def paired_deltas(
    control: list[float] | np.ndarray, treatment: list[float] | np.ndarray
) -> np.ndarray:
    """Per-window ``control - treatment`` for an error metric.

    Positive means the treatment is **better** (lower error), matching the
    ``delta_*`` convention already used by the benchmark payload. Windows where
    either arm failed to score (``None``/NaN) are dropped pairwise, so one
    arm's blow-up cannot silently shorten only its own series.
    """
    a = np.asarray(control, dtype=float)
    b = np.asarray(treatment, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms must be the same length: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok] - b[ok]


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """``sum|error| / sum|actual|`` as a percentage — the optimising metric.

    Unlike MAPE this is a ratio of totals, so a low-demand hour cannot
    dominate, and summing across BAs stays meaningful (it is MW error
    normalised by MW served).
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(f)
    denom = np.abs(a[ok]).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(a[ok] - f[ok]).sum() / denom * 100)


def bias_pct(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean **signed** percentage error: ``mean((F - A) / A)`` as a percent.

    Negative means the forecast runs low. This is the number the satisficing
    constraint watches, because the optimising metric is blind to direction
    and under-forecasting demand is the operationally expensive error.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(predicted, dtype=float)
    ok = np.isfinite(a) & np.isfinite(f) & (a != 0)
    if not ok.any():
        return float("nan")
    return float(np.mean((f[ok] - a[ok]) / a[ok]) * 100)


def satisficing_check(
    *,
    treatment_bias_pct: float | None,
    control_mape: float | None,
    treatment_mape: float | None,
    max_abs_bias_pct: float = MAX_ABS_BIAS_PCT,
    max_mape_regression_pts: float = MAX_MAPE_REGRESSION_PTS,
) -> dict:
    """Do the constraints hold? A failure vetoes a win on the optimising metric.

    Returns ``{"passed": bool, "failures": [str, ...]}``. Unmeasurable inputs
    (``None``/NaN) are reported as failures rather than passes — an unchecked
    constraint is not a satisfied one.
    """
    failures: list[str] = []

    if treatment_bias_pct is None or not np.isfinite(treatment_bias_pct):
        failures.append("bias not measured")
    elif abs(treatment_bias_pct) > max_abs_bias_pct:
        direction = "under" if treatment_bias_pct < 0 else "over"
        failures.append(
            f"bias {treatment_bias_pct:+.2f}% exceeds ±{max_abs_bias_pct}% "
            f"({direction}-forecasting systematically)"
        )

    if (
        control_mape is None
        or treatment_mape is None
        or not np.isfinite(control_mape)
        or not np.isfinite(treatment_mape)
    ):
        failures.append("published metric (MAPE) not measured")
    else:
        regression = treatment_mape - control_mape
        if regression > max_mape_regression_pts:
            failures.append(
                f"published MAPE regresses {regression:+.2f} pts "
                f"(> {max_mape_regression_pts} allowed)"
            )

    return {"passed": not failures, "failures": failures}


def verdict(
    deltas: list[float] | np.ndarray,
    *,
    min_sign_consistency: float = MIN_SIGN_CONSISTENCY,
    min_t: float = MIN_T_STATISTIC,
    min_windows: int = MIN_WINDOWS_FOR_A_VERDICT,
) -> dict:
    """Decide whether a paired A/B difference is real, or refuse to decide.

    Two conditions, both required, because they catch different failures:

    * **Magnitude** — ``|mean| >= min_t * stderr``. Rejects differences
      indistinguishable from window-to-window noise. This alone would have
      failed CAISO, whose two observed deltas were −7.24 and +3.87.
    * **Sign consistency** — the winning arm must win in at least
      ``min_sign_consistency`` of windows. Rejects a mean driven by one
      catastrophic window, which is a *tail-risk* finding and deserves to be
      reported as one rather than laundered into an average.

    Fewer than ``min_windows`` windows is always ``inconclusive``, whatever the
    numbers look like.

    The t-statistic here is a decision rule, not a significance claim: windows
    from a rolling origin overlap in training data and are not independent
    draws, so the nominal p-value would be optimistic. It is used as a
    conservative filter, and `worst_window` is reported alongside precisely
    because a passing mean is not the whole story.

    Returns:
        ``{"decisive", "winner", "n", "mean", "median", "stderr", "t",
        "sign_consistency", "worst_window", "best_window", "reason"}`` —
        ``winner`` is ``"treatment"``, ``"control"``, or ``None``.
    """
    d = np.asarray([x for x in np.asarray(deltas, dtype=float) if np.isfinite(x)], dtype=float)
    n = int(d.size)
    out: dict = {
        "decisive": False,
        "winner": None,
        "n": n,
        "mean": None,
        "median": None,
        "stderr": None,
        "t": None,
        "sign_consistency": None,
        "worst_window": None,
        "best_window": None,
        "reason": "",
    }
    if n == 0:
        out["reason"] = "no scored windows"
        return out

    mean = float(d.mean())
    out.update(
        mean=round(mean, 4),
        median=round(float(np.median(d)), 4),
        worst_window=round(float(d.min()), 4),
        best_window=round(float(d.max()), 4),
    )
    if n < min_windows:
        out["reason"] = f"{n} window(s) < {min_windows} required for a verdict"
        return out

    # ddof=1: we are estimating the spread of window outcomes, not describing
    # these particular windows.
    stderr = float(d.std(ddof=1) / np.sqrt(n))
    out["stderr"] = round(stderr, 4)

    if stderr == 0:
        # Identical deltas in every window — a real effect (often an exact
        # no-op, delta 0) with no spread to test against.
        out["t"] = None
        consistency = 1.0 if mean != 0 else 0.0
    else:
        out["t"] = round(mean / stderr, 3)
        favoured = np.sign(mean)
        consistency = float((np.sign(d) == favoured).sum() / n)
    out["sign_consistency"] = round(consistency, 3)

    if mean == 0:
        out["reason"] = "no difference"
        return out

    # Outlier domination has its own signature: the mean points one way and
    # the MEDIAN points the other. One catastrophic window can drag a mean
    # across zero while most windows disagree with it. Checked before the
    # noise test because that test would also reject this input — such an
    # outlier inflates the variance it is measured against — but "within
    # noise" is the less useful thing to tell a reader. This is the ISONE
    # shape: three small wins and one -19.18.
    median = float(np.median(d))
    if median != 0 and np.sign(mean) != np.sign(median):
        out["reason"] = (
            f"mean {mean:+.3f} and median {median:+.3f} disagree in sign — "
            f"outlier window(s) dominate; report as tail risk, not as a win"
        )
        return out

    if stderr > 0 and abs(mean) < min_t * stderr:
        out["reason"] = (
            f"|mean| {abs(mean):.3f} < {min_t}x stderr {stderr:.3f} — within window-to-window noise"
        )
        return out
    if consistency < min_sign_consistency:
        out["reason"] = (
            f"wins only {consistency:.0%} of windows (< {min_sign_consistency:.0%}) — "
            f"mean is driven by outlier windows, report as tail risk not as a win"
        )
        return out

    out["decisive"] = True
    out["winner"] = "treatment" if mean > 0 else "control"
    out["reason"] = (
        f"{out['winner']} wins {consistency:.0%} of {n} windows, "
        f"mean {mean:+.3f} pts ({abs(mean) / stderr:.1f}x stderr)"
    )
    return out
