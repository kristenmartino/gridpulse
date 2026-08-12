"""
Weighted ensemble combiner for demand forecasting.

Per spec §Model 4:
- Weighted average where weights are proportional to a power of inverse recent
  MAPE — ``(1/MAPE_i)^k``, ``k = config.ENSEMBLE_WEIGHT_EXPONENT`` (ADR-004)
- The value is error DECORRELATION, not dominance: measured on the recursive
  51-BA holdout the ensemble beats XGBoost-alone on 17 of 51 BAs (see
  docs/BACKTEST_RESULTS.md "Ensemble weighting") — the k=3 sharpening exists
  precisely because plain blending often trails the best single model
- Ensemble forecast is bounded pointwise between the min and max of the
  individual forecasts (per hour; this does NOT bound aggregate MAPE)
"""

import numpy as np
import structlog

from config import ENSEMBLE_WEIGHT_EXPONENT

log = structlog.get_logger()


def compute_ensemble_weights(mape_scores: dict[str, float]) -> dict[str, float]:
    """
    Compute ensemble weights from a power of each model's inverse MAPE.

    weight_i = (1/MAPE_i)^k / sum_j (1/MAPE_j)^k,  k = ENSEMBLE_WEIGHT_EXPONENT

    ``k=1`` is plain inverse-MAPE; ``k>1`` sharpens the blend toward the best
    model (ADR-004 refinement, #181). ``k=3`` is the validated default — plain
    inverse-MAPE over-weighted models running 3–5× worse than the leader and
    trailed the best single model on the recursive holdout.

    Args:
        mape_scores: Dict mapping model name → recent MAPE (%).

    Returns:
        Dict mapping model name → weight (sums to 1.0).
    """
    if not mape_scores:
        raise ValueError("No MAPE scores provided")

    # Filter out models with zero or invalid MAPE
    valid = {k: v for k, v in mape_scores.items() if v > 0 and np.isfinite(v)}

    if not valid:
        # Equal weights fallback
        n = len(mape_scores)
        weights = {k: 1.0 / n for k in mape_scores}
        log.warning("ensemble_equal_weights_fallback", reason="no valid MAPE scores")
        return weights

    inverse = {k: (1.0 / v) ** ENSEMBLE_WEIGHT_EXPONENT for k, v in valid.items()}
    total = sum(inverse.values())
    weights = {k: v / total for k, v in inverse.items()}

    log.info(
        "ensemble_weights_computed",
        weights={k: round(v, 3) for k, v in weights.items()},
        exponent=ENSEMBLE_WEIGHT_EXPONENT,
    )
    return weights


def resolve_ensemble_weights(
    members: list[str] | set[str],
    mape_scores: dict[str, float | None] | None,
) -> tuple[dict[str, float], str]:
    """Decide the weights for a given ensemble membership. One rule, two callers.

    Returns ``(weights, rule)`` where ``rule`` is ``"inverse_mape_cubed"`` or
    ``"equal"``, normalized to sum to 1 over exactly ``members``.

    **P2-16 (#273): this exists so training and scoring cannot disagree.**
    They previously each decided membership *and* weighting independently —
    training always applied inverse-MAPE³ over whichever models produced a
    holdout, while scoring applied it only when *every* predicting model had a
    MAPE and otherwise fell back to equal weights. So the persisted ensemble
    metric could describe an MAPE³-weighted blend of one membership while
    production served an equal-weighted blend of another, under the same name.
    Membership still differs by necessity — training has holdout payloads,
    scoring has forecast arrays — but the *rule* applied to a membership is now
    shared, and the caller records which membership it used.

    Cubed weights require a usable MAPE for **every** member. Partial coverage
    falls back to equal weights rather than silently concentrating the blend on
    whichever models happen to have been measured.
    """
    names = sorted(members)
    if not names:
        return {}, "equal"
    usable = {
        n: float(v)
        for n, v in (mape_scores or {}).items()
        if n in names and v is not None and np.isfinite(v) and v > 0
    }
    if len(usable) == len(names):
        weights = compute_ensemble_weights(usable)
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}, "inverse_mape_cubed"
    return {n: 1.0 / len(names) for n in names}, "equal"


def ensemble_combine(
    forecasts: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Combine multiple model forecasts using weighted average.

    Args:
        forecasts: Dict mapping model name → forecast array.
        weights: Dict mapping model name → weight. If None, equal weights.

    Returns:
        Ensemble forecast array.
    """
    if not forecasts:
        raise ValueError("No forecasts provided")

    model_names = list(forecasts.keys())
    arrays = [forecasts[name] for name in model_names]

    # Validate all arrays have same length
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        log.warning(
            "ensemble_length_mismatch",
            lengths=dict(zip(model_names, lengths, strict=False)),
            truncating_to=min_len,
        )
        arrays = [a[:min_len] for a in arrays]

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    # Renormalize weights to available models
    available_weights = {k: weights.get(k, 0) for k in model_names}
    total = sum(available_weights.values())
    if total == 0:
        available_weights = {k: 1.0 / len(model_names) for k in model_names}
        total = 1.0
    normalized = {k: v / total for k, v in available_weights.items()}

    # Weighted average
    result = np.zeros(len(arrays[0]))
    for name, arr in zip(model_names, arrays, strict=False):
        result += normalized[name] * arr

    # Verify ensemble is bounded by individual forecasts
    stacked = np.stack(arrays)
    individual_min = stacked.min(axis=0)
    individual_max = stacked.max(axis=0)

    out_of_bounds = ((result < individual_min - 1e-6) | (result > individual_max + 1e-6)).sum()
    if out_of_bounds > 0:
        log.warning("ensemble_out_of_bounds", count=int(out_of_bounds))

    return result


def update_smoothed_mape(
    previous_smoothed: float | None,
    latest_mape: float | None,
    alpha: float | None = None,
) -> float | None:
    """One recursive EWMA step over a model's holdout-MAPE history (#451).

    ``new = alpha * latest + (1 - alpha) * previous``, seeded by the first usable
    observation. Recursive rather than a windowed mean so the training job carries
    one number forward in its own meta instead of re-reading the vintage history
    every night — the whole series is already summarised by the previous value.

    Returns ``None`` when nothing usable is available, which
    :func:`resolve_ensemble_weights` already treats as "not measured" and answers
    with equal weights. A smoothed MAPE must never invent a number for a model
    that has not been scored.

    Evidence: docs/WEIGHTS_AB_STUDY.md. The daily holdout estimator flaps (median
    12% run-to-run) and weights computed from a single draw of it partly chase
    noise; an EWMA at alpha=0.3 won the WAPE half of a pre-registered A/B over 8
    rolling origins x 51 BAs, robustly. It is nonetheless NOT enabled: the bias
    constraint could not be evaluated in that harness (the control arm itself
    breaches it), and an unmeasurable constraint counts as failed. This series is
    persisted anyway so the question stays answerable later.
    """
    from config import ENSEMBLE_MAPE_EWMA_ALPHA

    a = ENSEMBLE_MAPE_EWMA_ALPHA if alpha is None else float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {a}")

    def usable(v: float | None) -> float | None:
        return float(v) if v is not None and np.isfinite(v) and v > 0 else None

    latest, prev = usable(latest_mape), usable(previous_smoothed)
    if latest is None:
        return prev
    if prev is None:
        return latest
    return a * latest + (1.0 - a) * prev


#: Weight-input bases, published so a consumer can tell which number a set of
#: weights was derived from. See :func:`weighting_input`.
WEIGHT_INPUT_HOLDOUT = "holdout_mape"
WEIGHT_INPUT_EWMA = "mape_ewma"


def _weighting_choice(meta_mape: float | None, meta_extra: dict | None) -> tuple[float | None, str]:
    """The weight input for one model, and the name of the basis it came from.

    One branch, two readers. :func:`weighting_mape` and :func:`weighting_input`
    both derive from this rather than each re-deciding, because a label that
    can disagree with the value it describes is worse than no label — it is the
    same failure this pair exists to detect, one level down.
    """
    from config import feature_enabled

    if not feature_enabled("smoothed_ensemble_weights"):
        return meta_mape, WEIGHT_INPUT_HOLDOUT
    smoothed = (meta_extra or {}).get("mape_ewma")
    if smoothed is not None and np.isfinite(smoothed) and smoothed > 0:
        return float(smoothed), WEIGHT_INPUT_EWMA
    return meta_mape, WEIGHT_INPUT_HOLDOUT


def weighting_mape(meta_mape: float | None, meta_extra: dict | None) -> float | None:
    """The MAPE that drives the SERVED ensemble weights for one model (#451).

    Flag-gated: with ``smoothed_ensemble_weights`` off this is exactly today's
    behaviour — the latest holdout MAPE. With it on, the persisted EWMA is used
    when the training job has published one.

    Falls back to the raw MAPE whenever the smoothed value is absent, so the
    first run after the flag flips — and any model whose meta predates the field
    — still weights on a real measurement rather than dropping to equal weights.

    **There is exactly ONE caller: the scoring job** (#514). An earlier version
    of this docstring claimed "both weight callers route through here so the two
    cannot drift apart on *which number* they weight by" — the parallel to
    :func:`resolve_ensemble_weights`, which really does bind both sides to one
    *rule*. That claim was never true, and a false guarantee is worse than none,
    because it is the reason nobody checks.

    The training job deliberately does **not** route through here, and should
    not. Its persisted ensemble metric is scored strictly out-of-sample
    (ledger-23 / #404): weights are fitted on the leading slice of the holdout
    from MAPEs computed *on that slice*. An EWMA is a series across training
    *runs*, so it has no within-window analogue — feeding one in would either
    reintroduce the in-sample bias #404 removed or weight the scored half by a
    number drawn from outside it.

    So the two sides legitimately weight by different inputs, and under a
    flipped flag they *will*. What must not happen is their doing so silently,
    which is what :func:`weighting_input` and the ``weight_input`` field on both
    records exist to prevent.
    """
    return _weighting_choice(meta_mape, meta_extra)[0]


def weighting_input(meta_mape: float | None, meta_extra: dict | None) -> str:
    """Which basis :func:`weighting_mape` used for this model.

    Published on both the served payload and the persisted metric so the two
    can be compared. Training always reports ``holdout_mape``; scoring reports
    whatever the flag and the available history produced, per model — the
    fallback means a fleet can be genuinely mixed on the first run after a flip.
    """
    return _weighting_choice(meta_mape, meta_extra)[1]


def shadow_weighting_mape(meta_mape: float | None, meta_extra: dict | None) -> float | None:
    """The MAPE of the arm that is NOT being served — the shadow arm (#478).

    Exact mirror of :func:`weighting_mape`: whatever that returns, this returns
    the other one. With ``smoothed_ensemble_weights`` off (today) the served arm
    is the raw holdout MAPE and the shadow is the EWMA; flip the flag and the two
    swap, so the shadow always measures the alternative to whatever ships.

    Returns ``None`` when the alternative is not available — a model with no
    persisted ``mape_ewma`` yet has no shadow, and inventing one by falling back
    to the raw value would make the two arms *identical* and the comparison
    vacuously "no difference". That is the opposite of what ``weighting_mape``
    should do, which is why this is a separate function rather than a flag on it.
    """
    from config import feature_enabled

    smoothed = (meta_extra or {}).get("mape_ewma")
    smoothed_ok = smoothed is not None and np.isfinite(smoothed) and smoothed > 0

    if feature_enabled("smoothed_ensemble_weights"):
        return meta_mape
    return float(smoothed) if smoothed_ok else None
