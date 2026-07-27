"""Public Forecast Benchmark — GridPulse vs. each BA's own day-ahead forecast.

The product competes against a *free* incumbent: EIA-930 publishes each
balancing authority's own day-ahead forecast (the ``DF`` column). This
module scores the two head-to-head, continuously, riding instrumentation
that already exists rather than a one-shot replay:

* **Official arm** — ``first_seen_df`` from the vintage instrument.
* **GridPulse arm** — the resolved 24h/48h snapshots the horizon-drift
  pipeline already keeps, settled-regraded (#318).
* **Truth** — the vintage ``last_d``, EIA's settled value, used as the
  SINGLE truth for both arms so neither is scored against its own yardstick.

## Two provenance limits that MUST be disclosed before publishing

Both were found by design review after the first implementation, and both
constrain what this module is allowed to claim:

1. **``first_seen_df`` is not "the day-ahead value as published."**
   ``data.vintage`` only admits an hour once EIA publishes a metered ``D``,
   so the DF we store was re-read 0–3h *after* the target hour, not at
   day-ahead time. The *value* originates from the BA's day-ahead
   submission, but nothing in this repo yet measures whether EIA revises DF
   in between. Until that is measured, the honest phrasing is "the earliest
   day-ahead forecast we observed," never "their day-ahead forecast." The
   bias direction is at least conservative for us: any revision that landed
   before our capture makes *their* number better, not ours.
2. **Our lead is nominal, not realized.** ``_resolve_forecast_start``
   anchors the forecast at the last *real* demand hour, so with EIA's 1–4h
   publishing lag a "24h" record is a realized ~20–24h wall-clock lead. The
   resolved drift records discard ``made_at``, so this module cannot yet
   compute the realized lead — it therefore labels leads as **nominal** and
   must not publish a "24 hours ahead" claim.

## The traps this module does defend against

1. **The stub trap.** For not-yet-reported hours EIA publishes the official
   forecast *as* the actual (``D == DF``). Scoring those hours would credit
   the official forecast with a perfect prediction on hours it never made.
   Two predicates drop them: the first-sight ``was_placeholder`` flag, and —
   the sharper one — any hour whose *settled* value still equals the
   day-ahead forecast, where the official arm scores 0% by construction and
   our arm is being graded against their forecast rather than reality.
2. **The preliminary-actuals trap.** First-published actuals run up to 70%
   wrong on high-revision feeds. Only settled ``last_d`` is truth.

## Exclusions are a feature, not an omission

A BA is excluded when it cannot be scored *fairly*, and the reason is
published. Two classes:

* **broken-feed** — the feed revises so heavily that intraday scoring is
  meaningless, AND (ADR-009) GridPulse's own anchor is seeded from that
  BA's day-ahead forecast, which would make the comparison partly
  self-referential. Saying so is the transparency product.
* **df-coverage** — the BA publishes a day-ahead forecast too sparsely to
  score (<80% of hours).

## Lead-time handling

Ours is a *nominal* 24h snapshot (realized ~20–24h — see limit 2 above).
Theirs is a day-ahead submission documented at **17–41h** depending on
hour-of-day, per the Form EIA-930 instructions — documented, not observed
by us. The payload therefore also carries a comparison at our nominal
**48h** snapshot, which even after the lag shortfall exceeds their
documented maximum. That variant may only be *labelled* conservative once
realized leads are measured; until then it is published as a second data
point, not as a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fraction of hours a BA must publish a day-ahead forecast for to be
#: scoreable at all.
MIN_DF_COVERAGE = 0.80
#: Minimum paired hours before a per-BA verdict is published. Measured
#: sample sizes: 30-day windows give median 649 / min 500 paired hours per
#: BA, while 7-day windows give median 133 / min 46 — too thin for a
#: per-BA call, which is why verdicts use the 30-day window and the 7-day
#: number is only ever shown as a trend.
MIN_PAIRED_HOURS = 200

#: Feed classes that cannot be scored fairly (see module docstring).
UNSCOREABLE_CLASSES = frozenset({"broken"})

EXCLUDE_BROKEN_FEED = "broken-feed"
EXCLUDE_DF_COVERAGE = "df-coverage"
EXCLUDE_INSUFFICIENT = "insufficient-paired-hours"

#: Human-readable exclusion rationale, published verbatim on the benchmark
#: surfaces — the reason is the product.
EXCLUSION_REASONS: dict[str, str] = {
    EXCLUDE_BROKEN_FEED: (
        "This feed's provisional readings revise heavily before settling, so "
        "intraday scoring is not meaningful — and GridPulse anchors its own "
        "forecast on this BA's day-ahead value (ADR-009), which would make the "
        "comparison partly self-referential."
    ),
    EXCLUDE_DF_COVERAGE: (
        f"The BA publishes a day-ahead forecast for under "
        f"{MIN_DF_COVERAGE:.0%} of hours — too sparse to score fairly."
    ),
    EXCLUDE_INSUFFICIENT: (
        f"Fewer than {MIN_PAIRED_HOURS} comparable hours in the window "
        "(after excluding placeholder hours and unsettled values)."
    ),
}

#: The headline lead, and the conservative one (see module docstring).
HEADLINE_LEAD = "24h"
CONSERVATIVE_LEAD = "48h"

#: EIA's documented day-ahead submission window (Form EIA-930
#: instructions) — documented, not observed by us. The conservative arm may
#: only be labelled as such while our realized lead exceeds the upper
#: bound; measured at 46.80h minimum, so it currently does
#: (docs/BENCHMARK_PROVENANCE.md).
OFFICIAL_DOCUMENTED_LEAD_H = (17.0, 41.0)


@dataclass(frozen=True)
class PairedHour:
    """One target hour where both forecasts and a settled actual exist.

    Two official values, because the probe found EIA revises the day-ahead
    forecast for some BAs (PSEI 26%, SOCO 24%; the big ISOs never):

    * ``official`` — as-issued, the earliest DF we observed. The fair
      comparison, and the primary.
    * ``official_revised`` — EIA's current DF for the hour, which for a
      revised BA carries hindsight. Scored as the *conservative* arm so no
      reader can object that we graded a stale number. Equals ``official``
      wherever the BA never revises.
    """

    timestamp: str
    actual: float  # settled truth — the SAME value scores every arm
    official: float
    gridpulse: float
    official_revised: float


# ── metrics ──────────────────────────────────────────────────


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute percentage error: ``Σ|a−p| / Σ|a| × 100``.

    Reported alongside MAPE because MAPE is dominated by small-denominator
    hours, which systematically misrepresents small BAs — the ones where
    this benchmark's most striking results live.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = float(np.sum(np.abs(a)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / denom * 100.0)


def score_arm(pairs: list[PairedHour], arm: str) -> dict[str, float]:
    """MAPE / MAE / WAPE for one arm over the paired hours."""
    actual = np.array([p.actual for p in pairs], dtype=float)
    pred = np.array([getattr(p, arm) for p in pairs], dtype=float)
    if actual.size == 0:
        return {"mape": float("nan"), "mae": float("nan"), "wape": float("nan"), "n": 0}
    return {
        "mape": round(float(np.mean(np.abs(actual - pred) / actual * 100.0)), 3),
        "mae": round(float(np.mean(np.abs(actual - pred))), 1),
        "wape": round(wape(actual, pred), 3),
        "n": int(actual.size),
    }


# ── scoreability + pairing ───────────────────────────────────


def scoreability(vintage_records: list[Any], revision_class: str | None) -> dict[str, Any]:
    """Can this BA be scored, and if not, exactly why?

    Returns ``{"scoreable", "reason", "reason_detail", "df_coverage",
    "placeholder_pct", "n_hours"}``. The reason travels with the verdict so
    every exclusion is publishable.
    """
    n = len(vintage_records)
    if n == 0:
        return {
            "scoreable": False,
            "reason": EXCLUDE_INSUFFICIENT,
            "reason_detail": EXCLUSION_REASONS[EXCLUDE_INSUFFICIENT],
            "df_coverage": 0.0,
            "placeholder_pct": 0.0,
            "n_hours": 0,
        }

    has_df = sum(1 for r in vintage_records if np.isfinite(r.first_seen_df))
    placeholders = sum(1 for r in vintage_records if r.was_placeholder)
    coverage = has_df / n

    reason: str | None = None
    if revision_class in UNSCOREABLE_CLASSES:
        reason = EXCLUDE_BROKEN_FEED
    elif coverage < MIN_DF_COVERAGE:
        reason = EXCLUDE_DF_COVERAGE

    return {
        "scoreable": reason is None,
        "reason": reason,
        "reason_detail": EXCLUSION_REASONS[reason] if reason else None,
        "df_coverage": round(coverage, 4),
        "placeholder_pct": round(placeholders / n * 100, 2),
        "n_hours": n,
    }


#: An hour is an unresolved stub when settled truth still equals the
#: day-ahead forecast to within this many MW. Compared on the 2dp values
#: EIA publishes, so the tolerance only absorbs float noise.
STUB_EPSILON_MW = 0.01


def pair_hours(
    vintage_records: list[Any],
    gridpulse_by_ts: dict[str, float],
    revised_df_by_ts: dict[str, float] | None = None,
) -> tuple[list[PairedHour], dict[str, int]]:
    """Join the two arms on target hour, dropping every unfair hour.

    Returns ``(pairs, drop_counts)`` — the per-reason counts are published,
    because the exclusions are not neutral across BAs (a stub-heavy BA like
    MISO loses ~20% of its hours while a clean one loses none) and a reader
    who cannot see that will assume the worst.

    Drop reasons, in order:

    * ``unresolved_stub`` — settled value still equals the day-ahead
      forecast. The official arm would score exactly 0% by construction,
      *and* our arm would be graded against their forecast rather than
      reality. This is the sharper of the two stub predicates.
    * ``first_seen_placeholder`` — flagged ``D == DF`` at first sight.
      Dropped conservatively even when later corrected.
    * ``unsettled`` — no finite, positive settled actual yet.
    * ``no_df`` — the BA published no day-ahead forecast for the hour.
      Checked independently of the stub predicates, which both read as
      "not a stub" when DF is simply absent.
    * ``no_gridpulse`` — we have no matured prediction for the hour. Both
      arms are always scored on the SAME hour set; a one-sided score would
      compare a 30-day official record against a 1-day GridPulse one.
    """
    out: list[PairedHour] = []
    drops = {
        "unresolved_stub": 0,
        "first_seen_placeholder": 0,
        "unsettled": 0,
        "no_df": 0,
        "no_gridpulse": 0,
    }
    for r in vintage_records:
        official = r.first_seen_df
        if not np.isfinite(official):
            drops["no_df"] += 1
            continue
        actual = r.last_d
        if not np.isfinite(actual) or actual <= 0:
            drops["unsettled"] += 1
            continue
        if abs(actual - official) < STUB_EPSILON_MW:
            drops["unresolved_stub"] += 1
            continue
        if r.was_placeholder:
            drops["first_seen_placeholder"] += 1
            continue
        gp = gridpulse_by_ts.get(_normalize_ts(r.timestamp))
        if gp is None or not np.isfinite(gp):
            drops["no_gridpulse"] += 1
            continue
        # Falls back to the as-issued value, so a BA that never revises
        # (7 of 10 sampled) scores identically on both arms.
        revised = (revised_df_by_ts or {}).get(_normalize_ts(r.timestamp))
        if revised is None or not np.isfinite(revised):
            revised = official
        out.append(
            PairedHour(
                timestamp=str(r.timestamp),
                actual=float(actual),
                official=float(official),
                gridpulse=float(gp),
                official_revised=float(revised),
            )
        )
    return out, drops


def revised_df_from_frame(demand_df: Any) -> dict[str, float]:
    """EIA's CURRENT day-ahead forecast per hour, from the live demand frame.

    ``forecast_mw`` on the freshly fetched frame *is* the post-revision DF,
    so the conservative official arm needs no new capture — the value is
    already in the frame the scoring job holds.
    """
    if demand_df is None or getattr(demand_df, "empty", True):
        return {}
    cols = getattr(demand_df, "columns", [])
    if "forecast_mw" not in cols or "timestamp" not in cols:
        return {}
    out: dict[str, float] = {}
    for ts, value in zip(demand_df["timestamp"], demand_df["forecast_mw"], strict=False):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out[_normalize_ts(pd_ts_iso(ts))] = v
    return out


def pd_ts_iso(ts: Any) -> str:
    """ISO-8601 for a pandas Timestamp or anything str()-able."""
    iso = getattr(ts, "isoformat", None)
    return iso() if callable(iso) else str(ts)


def _normalize_ts(ts: Any) -> str:
    """Compare timestamps as ISO strings to the hour, tz-normalized."""
    s = str(ts)
    return s.replace("+00:00", "Z").replace(" ", "T")


def gridpulse_predictions(
    horizon_payload: dict[str, Any] | None, model: str, lead: str
) -> dict[str, float]:
    """Extract ``{target_hour: predicted}`` from the horizon-drift payload.

    Reads the resolved records the horizon pipeline already keeps for
    ``models.{model}.{lead}.records`` — these are settled-regraded, so the
    GridPulse arm inherits the same truth discipline as the official arm.
    """
    if not horizon_payload:
        return {}
    block = ((horizon_payload.get("models") or {}).get(model) or {}).get(lead) or {}
    out: dict[str, float] = {}
    for row in block.get("records") or []:
        ts = row.get("ts") or row.get("timestamp")
        pred = row.get("p", row.get("predicted"))
        if ts is None or pred is None:
            continue
        try:
            out[_normalize_ts(ts)] = float(pred)
        except (TypeError, ValueError):
            continue
    return out


# ── payload ──────────────────────────────────────────────────


def compute_benchmark_payload(
    region: str,
    vintage_records: list[Any],
    horizon_payload: dict[str, Any] | None,
    revision_class: str | None,
    *,
    model: str = "ensemble",
    mean_revision_pct: float | None = None,
    revised_df_by_ts: dict[str, float] | None = None,
    observed_lead_h: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build ``gridpulse:benchmark:{region}`` for one tick.

    Always returns a payload — an excluded BA carries its reason rather than
    silently vanishing, because the exclusion list is part of what gets
    published.

    ``revised_df_by_ts`` supplies EIA's current (post-revision) day-ahead
    values so the conservative official arm can be scored; ``observed_lead_h``
    carries this tick's measured lead per nominal horizon, which decides
    whether the conservative label may be applied at all.
    """
    score = scoreability(vintage_records, revision_class)
    payload: dict[str, Any] = {
        "region": region,
        "revision_class": revision_class,
        "mean_revision_pct": mean_revision_pct,
        **score,
        "leads": {},
    }
    if not score["scoreable"]:
        return payload

    for lead in (HEADLINE_LEAD, CONSERVATIVE_LEAD):
        pairs, drops = pair_hours(
            vintage_records,
            gridpulse_predictions(horizon_payload, model, lead),
            revised_df_by_ts,
        )
        if len(pairs) < MIN_PAIRED_HOURS:
            payload["leads"][lead] = {
                "scoreable": False,
                "reason": EXCLUDE_INSUFFICIENT,
                "n": len(pairs),
                "excluded_hours": drops,
            }
            continue
        official = score_arm(pairs, "official")
        official_revised = score_arm(pairs, "official_revised")
        gridpulse = score_arm(pairs, "gridpulse")

        observed = (observed_lead_h or {}).get(lead)
        block = {
            "scoreable": True,
            "n": len(pairs),
            "official": official,
            # EIA revises DF for some BAs, so the conservative arm grades
            # them on the value that carries hindsight. Identical to
            # `official` wherever the BA never revises.
            "official_revised": official_revised,
            "gridpulse": gridpulse,
            # Positive delta = GridPulse is more accurate on that metric.
            "delta_mape": round(official["mape"] - gridpulse["mape"], 3),
            "delta_wape": round(official["wape"] - gridpulse["wape"], 3),
            "delta_mape_vs_revised": round(official_revised["mape"] - gridpulse["mape"], 3),
            "winner": "gridpulse" if gridpulse["mape"] < official["mape"] else "official",
            "winner_vs_revised": (
                "gridpulse" if gridpulse["mape"] < official_revised["mape"] else "official"
            ),
            "excluded_hours": drops,
            "observed_lead_h": None if observed is None else round(observed, 2),
            "lead_basis": "observed" if observed is not None else "nominal",
        }
        # The conservative label is EARNED, not assumed: it holds only while
        # our realized lead exceeds the operators' documented maximum.
        if lead == CONSERVATIVE_LEAD:
            block["conservative"] = bool(
                observed is not None and observed > OFFICIAL_DOCUMENTED_LEAD_H[1]
            )
            block["conservative_basis"] = (
                f"observed lead {observed:.2f}h > documented official max "
                f"{OFFICIAL_DOCUMENTED_LEAD_H[1]:.0f}h"
                if block["conservative"]
                else "withheld — observed lead does not exceed the documented official maximum"
            )
        payload["leads"][lead] = block

    headline = payload["leads"].get(HEADLINE_LEAD) or {}
    payload["scoreable"] = bool(headline.get("scoreable"))
    if not payload["scoreable"] and payload.get("reason") is None:
        payload["reason"] = EXCLUDE_INSUFFICIENT
        payload["reason_detail"] = EXCLUSION_REASONS[EXCLUDE_INSUFFICIENT]
    return payload


def fleet_rollup(
    region_payloads: list[dict[str, Any]], *, isolate: tuple[str, ...] = ("ERCOT",)
) -> dict[str, Any]:
    """Aggregate per-BA results into the fleet headline.

    ``isolate`` regions are reported separately rather than folded into the
    aggregate (ERCOT by epic requirement — its official forecast is the
    fleet's best at ~1.2% and the trader story there rests on quantiles,
    not the point scorecard).
    """
    scored = [p for p in region_payloads if p.get("scoreable")]
    excluded = [p for p in region_payloads if not p.get("scoreable")]
    fleet = [p for p in scored if p["region"] not in isolate]

    def _headline(p: dict[str, Any]) -> dict[str, Any]:
        return p["leads"][HEADLINE_LEAD]

    wins = sum(1 for p in fleet if _headline(p)["winner"] == "gridpulse")
    gp_mapes = [_headline(p)["gridpulse"]["mape"] for p in fleet]
    off_mapes = [_headline(p)["official"]["mape"] for p in fleet]

    return {
        "n_scoreable": len(scored),
        "n_excluded": len(excluded),
        "excluded": [{"region": p["region"], "reason": p.get("reason")} for p in excluded],
        "fleet": {
            "n": len(fleet),
            "wins": wins,
            "losses": len(fleet) - wins,
            "median_gridpulse_mape": round(float(np.median(gp_mapes)), 3) if gp_mapes else None,
            "median_official_mape": round(float(np.median(off_mapes)), 3) if off_mapes else None,
            # The consistency story: our spread vs theirs. This is the
            # durable claim even in windows where the win count is a coin
            # flip — measured 41x spread in the operators' own accuracy.
            "gridpulse_spread": _spread(gp_mapes),
            "official_spread": _spread(off_mapes),
        },
        "isolated": {p["region"]: _headline(p) for p in scored if p["region"] in isolate},
    }


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    lo, hi = float(min(values)), float(max(values))
    return {
        "min": round(lo, 3),
        "max": round(hi, 3),
        "ratio": round(hi / lo, 1) if lo > 0 else None,
    }
