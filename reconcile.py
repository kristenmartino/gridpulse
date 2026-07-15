"""Reconciliation — does the number on the screen match reality? (#304)

Every serious bug in this project's recent history has been in the **meta-layer**
— the system measuring itself — and every one was caught by a human's eye on
prod, never by a test or an alert (#131 simulated MAPE shown as real, #220
permanently-empty residual panels, #296 forecasts decaying to 0 MW, #304 the
Live Drift panel). The unit suite pins *shapes and contracts*; nothing pins
**correspondence to reality**. This module is that missing check.

## What it checks (Check A)

The Models tab's Live Drift panel displays ``rolling_mape_7d`` from
``gridpulse:drift:{region}``. Each underlying ``DriftRecord`` was scored at tick
time against the actual **EIA had published then** — a *preliminary* value. EIA
later revises those hours, sometimes enormously.

Measured in prod across all 51 BAs (2026-07-15):

    corr(EIA revision magnitude, settled forecast error) = 0.88

    LDWP  70.7% mean revision   panel 147.91   settled  53.22
    AZPS  66.7% mean revision   panel 338.67   settled  15.64   <- 21x overstated
    PSCO  14.3% mean revision   panel   6.28   settled  13.26   <- UNDERSTATED
    BPAT  14.2% mean revision   panel  10.72   settled   8.67
    PNM    0.7% mean revision   panel   2.06   settled   1.82

So the panel is scored against a **moving target** and errs in *both*
directions. Understatement (PSCO) is the more dangerous half: the model looks
healthier than it is.

Two signals, both fleet-invisible today:

* **A1 — displayed-vs-settled divergence.** Rescore the *same* stored predictions
  against settled demand and diff the result against the number the panel shows.
* **A2 — revision magnitude.** The upstream cause (corr 0.88). Bad preliminary
  data poisons the forecast (via the ``demand_lag_1h`` anchor) *and* the metric
  that grades it (via the scored actual) — one defect, two victims.

## Independence — or this is just another thing that's wrong

A checker that reuses the code it checks agrees with it by construction (the
#217 circular-verdict trap). So:

* Reuse only **leaf math** (``absolute_pct_error``) and **parsing**
  (``deserialize_records``, ``_normalize_ts``). Shared *thresholds* are a spec,
  not circular logic.
* **Independently re-implement the windowing and aggregation.** That is exactly
  where this class of bug lives — cf. the ``n_records`` (total, count-trimmed)
  vs ``n_7d`` (in-window) trap in ``models/drift.py``. Reusing
  ``compute_drift_payload`` would re-derive the displayed number using the
  displayed number's own code.
* Truth comes from GCS parquet (settled), not from the serving path's own state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from models.drift import _normalize_ts, absolute_pct_error, deserialize_records

# The window the panel displays. Defined here rather than imported so a change
# to the producer's window can't silently move the checker's goalposts too.
RECONCILE_WINDOW_HOURS = 24 * 7

#: Mirrors ``models.drift.LOW_ACTUAL_FRACTION``. Deliberately duplicated: this
#: is a *threshold spec* both sides must honour, and the checker must apply the
#: same rule to compare like with like. If the producer changes its filter, this
#: check SHOULD diverge — that is a real finding, not a bug here.
RECONCILE_LOW_ACTUAL_FRACTION = 0.10


@dataclass(frozen=True)
class ReconcileFinding:
    """One (region, model) verdict.

    ``ok`` is False only when a threshold was breached with enough samples to
    judge; ``skipped`` carries the reason when we declined to judge at all
    (never silently counted as a pass).
    """

    region: str
    model: str
    ok: bool
    displayed_mape: float | None = None
    settled_mape: float | None = None
    divergence_pct_points: float | None = None
    mean_abs_revision_pct: float | None = None
    n_compared: int = 0
    reasons: list[str] = field(default_factory=list)
    skipped: str | None = None

    def as_log_fields(self) -> dict[str, Any]:
        """Flat kwargs for a structlog event (jsonPayload.* after #306)."""
        out: dict[str, Any] = {
            "region": self.region,
            "model": self.model,
            "n_compared": self.n_compared,
        }
        for name in (
            "displayed_mape",
            "settled_mape",
            "divergence_pct_points",
            "mean_abs_revision_pct",
        ):
            val = getattr(self, name)
            if val is not None:
                out[name] = round(float(val), 4)
        if self.reasons:
            out["reasons"] = ",".join(self.reasons)
        if self.skipped:
            out["skipped"] = self.skipped
        return out


def settled_actuals_from_demand(demand_df: Any) -> dict[str, float]:
    """``{normalized_iso_hour -> settled_mw}`` from a demand frame.

    Source of truth is the GCS parquet (``gcs_store.read_parquet("demand",
    region)``), which carries EIA's **current** view of each hour — i.e. settled
    where revisions have landed. Non-finite and non-positive values are dropped:
    a zero/NaN demand row is a reporting artifact, and it cannot form a
    percentage denominator.
    """
    import pandas as pd

    if demand_df is None or len(demand_df) == 0:
        return {}
    if "timestamp" not in demand_df.columns or "demand_mw" not in demand_df.columns:
        return {}

    ts = pd.to_datetime(demand_df["timestamp"], utc=True)
    vals = pd.to_numeric(demand_df["demand_mw"], errors="coerce")
    out: dict[str, float] = {}
    for t, v in zip(ts, vals, strict=False):
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        out[_normalize_ts(t.isoformat())] = float(v)
    return out


def _within_window(records: list[Any], window_hours: int, now: datetime) -> list[Any]:
    """Records whose target hour falls inside the window.

    Independently implemented (not ``models.drift._within_window``) on purpose:
    the window boundary is precisely where the ``n_records`` vs ``n_7d`` class of
    bug lives, so the checker must not inherit the producer's notion of it.
    """
    cutoff = now - timedelta(hours=window_hours)
    keep = []
    for r in records:
        try:
            when = datetime.fromisoformat(r.timestamp)
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        if when >= cutoff:
            keep.append(r)
    return keep


def _drop_low_actuals(
    pairs: list[tuple[float, float]], min_fraction: float
) -> list[tuple[float, float]]:
    """Drop (predicted, actual) pairs whose actual is a near-zero outlier.

    Mirrors the producer's region-relative rule so the two means are comparable:
    threshold = ``min_fraction × median(|actual|)``. Independently implemented
    (median + compare) rather than reusing ``filter_low_actuals``, which is typed
    to ``DriftRecord`` and belongs to the producer.
    """
    if not pairs or min_fraction <= 0:
        return pairs
    actuals = [abs(a) for _, a in pairs]
    median = float(np.median(actuals))
    if median <= 0:
        return pairs
    threshold = min_fraction * median
    return [(p, a) for p, a in pairs if abs(a) >= threshold]


def recompute_settled_mape(
    records: list[Any],
    settled: dict[str, float],
    *,
    window_hours: int = RECONCILE_WINDOW_HOURS,
    now: datetime | None = None,
    min_actual_fraction: float = RECONCILE_LOW_ACTUAL_FRACTION,
) -> tuple[float | None, float | None, int]:
    """Rescore the stored predictions against settled demand.

    Returns ``(settled_mape, mean_abs_revision_pct, n_compared)``. Hours absent
    from ``settled`` are **skipped, not counted as agreement**: the GCS parquet
    is overwritten fire-and-forget and EIA's last-known-good chain can route
    around a write, so it may legitimately lag Redis. Returns ``(None, None, 0)``
    when nothing is comparable.
    """
    now = now or datetime.now(UTC)
    in_window = _within_window(records, window_hours, now)

    pairs: list[tuple[float, float]] = []
    revisions: list[float] = []
    for r in in_window:
        settled_mw = settled.get(_normalize_ts(r.timestamp))
        if settled_mw is None or not np.isfinite(settled_mw) or settled_mw <= 0:
            continue
        if not np.isfinite(r.predicted) or not np.isfinite(r.actual) or r.actual <= 0:
            continue
        pairs.append((float(r.predicted), float(settled_mw)))
        revisions.append(abs(settled_mw - float(r.actual)) / settled_mw * 100.0)

    if not pairs:
        return None, None, 0

    kept = _drop_low_actuals(pairs, min_actual_fraction)
    errors = [e for e in (absolute_pct_error(p, a) for p, a in kept) if e is not None]
    if not errors:
        return None, float(np.mean(revisions)) if revisions else None, 0
    return float(np.mean(errors)), float(np.mean(revisions)), len(errors)


def check_drift_against_settled(
    region: str,
    drift_payload: dict[str, Any] | None,
    settled: dict[str, float],
    *,
    now: datetime | None = None,
    min_samples: int = 24,
    max_divergence_pct_points: float = 2.0,
    max_revision_pct: float = 5.0,
) -> list[ReconcileFinding]:
    """Check A: is the displayed drift metric true against settled demand?

    A1 fires when ``|displayed − settled|`` exceeds ``max_divergence_pct_points``
    — the panel is grading against a moving target (AZPS: 338.67 displayed vs
    15.64 settled; PSCO: 6.28 displayed vs 13.26 settled, i.e. *flattering*).

    A2 fires when mean revision exceeds ``max_revision_pct`` — the upstream data
    -quality cause (corr 0.88 with settled error fleet-wide), which also poisons
    the forecast through its ``demand_lag_1h`` anchor.

    Thresholds are generous by design (the #217 cry-wolf lesson); paging
    hysteresis lives in the job, not here. Never raises.
    """
    if not isinstance(drift_payload, dict):
        return []
    models = drift_payload.get("models")
    if not isinstance(models, dict):
        return []

    findings: list[ReconcileFinding] = []
    for model, block in sorted(models.items()):
        if not isinstance(block, dict):
            continue
        records = deserialize_records(block.get("records"))
        if not records:
            findings.append(
                ReconcileFinding(region=region, model=model, ok=True, skipped="no_records")
            )
            continue

        settled_mape, revision, n = recompute_settled_mape(records, settled, now=now)
        displayed = block.get("rolling_mape_7d")
        displayed = float(displayed) if isinstance(displayed, int | float) else None

        if settled_mape is None or n < min_samples:
            findings.append(
                ReconcileFinding(
                    region=region,
                    model=model,
                    ok=True,
                    displayed_mape=displayed,
                    settled_mape=settled_mape,
                    mean_abs_revision_pct=revision,
                    n_compared=n,
                    skipped="insufficient_settled_overlap",
                )
            )
            continue

        reasons: list[str] = []
        divergence = None
        if displayed is not None and np.isfinite(displayed):
            divergence = abs(displayed - settled_mape)
            if divergence > max_divergence_pct_points:
                # Name the direction: understatement is the dangerous half.
                reasons.append(
                    "displayed_overstates" if displayed > settled_mape else "displayed_understates"
                )
        if revision is not None and revision > max_revision_pct:
            reasons.append("high_eia_revision")

        findings.append(
            ReconcileFinding(
                region=region,
                model=model,
                ok=not reasons,
                displayed_mape=displayed,
                settled_mape=settled_mape,
                divergence_pct_points=divergence,
                mean_abs_revision_pct=revision,
                n_compared=n,
                reasons=reasons,
            )
        )
    return findings
