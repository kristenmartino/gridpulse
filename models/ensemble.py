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
