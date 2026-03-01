"""
Weighted ensemble combiner for demand forecasting.

Per spec §Model 4:
- Weighted average where weights are inversely proportional to recent MAPE
- Combining models almost always beats individual models
- Ensemble forecast is bounded between min and max of individual forecasts
"""

import numpy as np
import structlog

log = structlog.get_logger()


def compute_ensemble_weights(mape_scores: dict[str, float]) -> dict[str, float]:
    """
    Compute ensemble weights inversely proportional to each model's MAPE.

    weight_i = (1/MAPE_i) / sum(1/MAPE_j)

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

    inverse = {k: 1.0 / v for k, v in valid.items()}
    total = sum(inverse.values())
    weights = {k: v / total for k, v in inverse.items()}

    log.info(
        "ensemble_weights_computed",
        weights={k: round(v, 3) for k, v in weights.items()},
    )
    return weights


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
