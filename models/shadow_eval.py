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

from models.drift import DriftRecord, filter_by_lead, filter_low_actuals

#: #451's pre-registered thresholds, reused verbatim rather than re-chosen.
#: Relaxing either makes this a different experiment; a test pins them.
MAX_ABS_BIAS_PCT = 2.0
MAX_MAPE_REGRESSION_PTS = 0.5
MIN_DAYS_DEFAULT = 14


def filter_records(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply the drift path's quality gates before either arm is scored.

    **This closes the defect that made the #478 evaluation unable to decide.**
    ``compute_drift_payload`` runs every record through :func:`filter_by_lead`
    then :func:`filter_low_actuals` before it averages anything
    (``models/drift.py``). The shadow path reused the *grading* primitive
    (``build_records_from_actuals``) and none of the *filtering* — so the two
    paths graded identically and filtered differently, and only one of them was
    protected from the artifacts the filters exist to remove.

    Measured on production 2026-08-18, served arm, before → after this gate:
    per-BA bias **+9.421% → +3.264%**, pooled **+3.656% → +2.412%** over 51 BAs.
    Almost all of that is :func:`filter_by_lead` removing **415 records whose
    known lead exceeded 1h** from a window whose whole name is "1-hour-ahead"
    (production carried leads out to 63h). :func:`filter_low_actuals` dropped
    only **2 records fleet-wide** — it is region-*relative*, so when a BA's
    artifact hours are a large enough share of a short window they set the
    median and no longer read as outliers against it.

    **This gate does NOT clear the ±2% bound, and must not be described as
    doing so.** Post-filter the control arm still breaches, dominated by a
    single BA: **IID +86.49% over 126 clean lead-1 records**. That is not a
    forecasting property — the drift path grades IID over the same window at a
    *longer* 24h lead and reads **+1.65%**, on a feed whose actuals (339–960 MW)
    and predictions (397–882 MW) are both sane. A shorter lead cannot be 52×
    worse than a longer one, so the residual is a defect in how the shadow
    record stream is written, tracked separately. Filtering is necessary here
    and is not sufficient.

    **Filtering is on the SHARED ``actual``, so both arms keep exactly the same
    hours.** That is what makes it safe: a gate applied per-arm could keep
    different hours for each and turn a weighting comparison into a coverage
    comparison — the mislabelling class this experiment already refuses
    elsewhere by skipping partial shadow arms rather than equal-weighting them.

    **Call this once per region, not inside :func:`arm_stats`.** It is
    deliberately not idempotent: :func:`filter_low_actuals` thresholds on the
    median of what it is given, so filtering an already-filtered window raises
    the threshold and drops more. One explicit call at the top of the
    per-region flow is the only correct shape.

    Returns ``(kept_records, counts)``.
    """
    empty = {
        "n_in": 0,
        "n_kept": 0,
        "n_lead_dropped": 0,
        "n_unknown_lead": 0,
        "n_low_actual_dropped": 0,
    }
    if not records:
        return [], empty

    # Reuse the real filters rather than reimplementing their rules here. A
    # second copy of "what counts as a usable hour" is how these two paths
    # diverged in the first place.
    stubs: list[DriftRecord] = []
    origin: dict[int, dict] = {}
    for r in records:
        try:
            actual = float(r.get("actual"))
        except (TypeError, ValueError):
            actual = float("nan")
        lead = r.get("lead_hours")
        stub = DriftRecord(
            timestamp=str(r.get("timestamp") or ""),
            # Neither filter reads ``predicted``; mirroring ``actual`` keeps the
            # auto-derived sMAPE finite instead of seeding a NaN nobody reads.
            predicted=actual,
            actual=actual,
            abs_pct_error=0.0,
            lead_hours=int(lead) if isinstance(lead, (int, float)) else None,
        )
        stubs.append(stub)
        origin[id(stub)] = r

    lead_kept, n_lead_dropped, n_unknown_lead = filter_by_lead(stubs)
    kept_stubs, n_low_actual_dropped = filter_low_actuals(lead_kept)
    kept = [origin[id(s)] for s in kept_stubs]
    return kept, {
        "n_in": len(records),
        "n_kept": len(kept),
        "n_lead_dropped": n_lead_dropped,
        "n_unknown_lead": n_unknown_lead,
        "n_low_actual_dropped": n_low_actual_dropped,
    }


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
