"""Pure statistics for the #478 shadow-weight evaluation.

Lives here rather than inside ``scripts/shadow_weights_eval.py`` because that
script sets ``ENVIRONMENT=production`` at import time so it can reach the real
Redis. A test that imported the script to reach these functions leaked that env
var into the whole pytest session and flipped unrelated tests into the strict
production gate — ``test_sprint4`` started seeing ``get_forecasts_unavailable_prod``.
It passed locally, where ENVIRONMENT was already set, and failed in CI, where it
was not.

The rule this encodes: **logic worth testing does not sit behind an import side
effect.** The script keeps the I/O, the CLI and the printing; everything a test
needs lives here and imports cleanly.
"""

from __future__ import annotations

import numpy as np

#: #451's pre-registered thresholds, reused verbatim rather than re-chosen.
#: Relaxing either makes this a different experiment; a test pins them.
MAX_ABS_BIAS_PCT = 2.0
MAX_MAPE_REGRESSION_PTS = 0.5
MIN_DAYS_DEFAULT = 14


def arm_stats(records: list[dict], arm: str) -> dict[str, float] | None:
    """Bias and MAPE for one arm over graded records."""
    pairs = [
        (float(r["actual"]), float(r[f"{arm}_predicted"]))
        for r in records
        if r.get("actual") and r.get(f"{arm}_predicted") is not None
    ]
    pairs = [(a, p) for a, p in pairs if np.isfinite(a) and np.isfinite(p) and a > 0]
    if not pairs:
        return None
    actual = np.array([a for a, _ in pairs])
    pred = np.array([p for _, p in pairs])
    return {
        "n": len(pairs),
        # Signed: the constraint is about systematic DIRECTION. MAPE's asymmetry
        # biases optimisation toward under-forecasting, which is the expensive
        # direction for a grid, so the sign is the point.
        "bias_pct": float(np.mean((pred - actual) / actual) * 100.0),
        "mape": float(np.mean(np.abs(pred - actual) / actual) * 100.0),
        "wape": float(np.sum(np.abs(pred - actual)) / np.sum(actual) * 100.0),
    }


def fleet_stats(per_region: dict[str, dict], arm: str) -> dict[str, float] | None:
    """Fleet figure with **each BA counted once**, whatever its record count.

    The pooled alternative (concatenate every record, then average) silently
    weights well-fed BAs more heavily — and coverage here is very uneven by
    construction. The #309 quality guard NaNs unreliable hours, so the
    broken-feed BAs grade far fewer records than the rest: measured on
    2026-08-11, LDWP graded **0 records in 10 ticks** while 46 of 51 BAs graded
    one per tick. Pooling would have given LDWP no voice at all and ERCOT ten
    times AZPS's.

    That matters because #451's whole bias finding turned on *which* BAs were in
    the average — its 12-BA cut passed the ±2% constraint and the 51-BA cut
    failed it, from the same code. Repeating that with an implicit weighting
    would be the same mistake wearing a different hat.

    Both figures are reported; this is the headline because a BA is the unit the
    product is served in, matching the convention
    ``scripts/export_holdout_metrics.py`` already states — accuracy is per-BA,
    and one pooled number hides the tail.
    """
    vals = [v[arm] for v in per_region.values() if v.get(arm)]
    if not vals:
        return None
    return {
        "n_bas": len(vals),
        "n": int(sum(v["n"] for v in vals)),
        "bias_pct": float(np.mean([v["bias_pct"] for v in vals])),
        "mape": float(np.mean([v["mape"] for v in vals])),
        "wape": float(np.mean([v["wape"] for v in vals])),
    }


def coverage_rows(per_region: dict[str, dict]) -> list[tuple[str, int, float]]:
    """``(region, n_records, days)`` sorted sparsest first — the tail is the point."""
    return sorted(((r, v["n"], v["days"]) for r, v in per_region.items()), key=lambda t: t[1])
