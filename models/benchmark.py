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

## Two provenance limits, now measured (``docs/BENCHMARK_PROVENANCE.md``)

Both were found by design review after the first implementation. Neither is
an open question any more, and what each one still forbids is stated here:

1. **``first_seen_df`` is not "the day-ahead value as published."**
   ``data.vintage`` only admits an hour once EIA publishes a metered ``D``,
   so the DF we store was re-read 0–3h *after* the target hour, not at
   day-ahead time. **Measured:** EIA does revise DF, very unevenly — 7 of 10
   sampled BAs never revise at all (PJM, MISO, ERCOT, CAISO, GVL, SPP,
   NYISO), while PSEI revises 26.4% and SOCO 24.2%. The largest movement in
   any sampled operator's own *median* APE is 1.43 points — which bounds
   nothing about a head-to-head result, since verdicts are decided on *mean*
   MAPE and the probe never measures it. So rather than pick a side, the
   official arm is scored **both ways** (``official`` as-issued,
   ``official_revised`` as EIA's current view) and both verdicts publish. Still forbidden: calling
   either one "their day-ahead forecast" — a revision that landed *before*
   our first capture is invisible to us, so the phrasing stays "the earliest
   day-ahead forecast we observed."
2. **The lead is measured per tick, not assumed.**
   ``_resolve_forecast_start`` anchors the forecast at the last *real*
   demand hour, so with EIA's publishing lag a nominal "24h" record carries
   a realized ~23.9h. ``jobs.phases._observed_lead_hours`` computes that
   from the forecast payload each tick and passes it as ``observed_lead_h``;
   the payload publishes the number and sets ``lead_basis`` to ``observed``.
   Still forbidden: an unqualified "24 hours ahead" claim — quote the
   realized figure, which is the shorter one.

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

Ours is a *nominal* 24h snapshot, realized ~23.9h (measured every tick —
see limit 2 above). Theirs is a day-ahead submission documented at
**17–41h** depending on hour-of-day, per the Form EIA-930 instructions —
documented, not observed by us. The payload therefore also carries the
comparison at our nominal **48h** snapshot, realized ~47.9h, which clears
their documented maximum. That arm is *labelled* conservative only while
the tick's own observed lead exceeds 41h, so the label lapses by itself if
EIA's publishing lag ever grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fraction of hours a BA must publish a day-ahead forecast for to be
#: scoreable at all.
MIN_DF_COVERAGE = 0.80
#: Minimum paired hours before a per-BA verdict is published. Sizing came
#: from *officially scoreable* hours — the count after the vintage-side drops
#: but BEFORE the ``no_gridpulse`` join, which is what
#: ``docs/BENCHMARK_SCOREABILITY.md`` reports: median 649 / min 500 per BA
#: over 30 days, against median 133 / min 46 over 7. Paired hours are a
#: subset of those, so the 30-day window is the one that can carry a per-BA
#: verdict and 7-day numbers are only ever a trend. The true paired count is
#: the per-lead ``n`` in the payload.
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
#: bound; measured at 47.80h minimum, so it currently does
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
    """MAPE / median APE / MAE / WAPE for one arm over the paired hours.

    ``mape`` is the **mean** APE — the industry-standard headline, and the
    statistic the winner is decided on. ``median_ape`` is published beside it
    because the offline reports (``docs/BENCHMARK_SCOREABILITY.md``,
    ``docs/BENCHMARK_PROVENANCE.md``) characterise BAs by *median* APE, and a
    reader carrying a figure between artifacts without noticing the statistic
    changed would be comparing two different things. On a feed with a fat
    error tail the two diverge substantially — which is information, not
    noise: a large gap says the BA's error is tail-driven.
    """
    actual = np.array([p.actual for p in pairs], dtype=float)
    pred = np.array([getattr(p, arm) for p in pairs], dtype=float)
    if actual.size == 0:
        return {
            "mape": float("nan"),
            "median_ape": float("nan"),
            "mae": float("nan"),
            "wape": float("nan"),
            "n": 0,
        }
    ape = np.abs(actual - pred) / actual * 100.0
    return {
        "mape": round(float(np.mean(ape)), 3),
        "median_ape": round(float(np.median(ape)), 3),
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
    *,
    exclude_stale_capture: bool = True,
) -> tuple[list[PairedHour], dict[str, int]]:
    """Join the two arms on target hour, dropping every unfair hour.

    Returns ``(pairs, drop_counts)`` — the per-reason counts are published,
    because the exclusions are not neutral across BAs and a reader who cannot
    see that will assume the worst. MISO loses ~20% of its hours to stubs and
    ~10% more to hours with no published DF; the cleanest feeds still lose a
    percent or two. None loses nothing.

    Drop reasons, in evaluation order (the order matters: each hour is
    attributed to the FIRST rule it trips, so these counts are disjoint):

    * ``no_df`` — the BA published no day-ahead forecast for the hour.
      Checked first, because both stub predicates read as "not a stub" when
      DF is simply absent.
    * ``unsettled`` — no finite, positive settled actual yet.
    * ``unresolved_stub`` — settled value still equals the day-ahead
      forecast. The official arm would score exactly 0% by construction,
      *and* our arm would be graded against their forecast rather than
      reality. This is the sharper of the two stub predicates.
    * ``first_seen_placeholder`` — flagged ``D == DF`` at first sight.
      Dropped conservatively even when later corrected.
    * ``stale_capture`` — the hour was first seen more than
      ``FRESH_CAPTURE_LAG_HOURS`` after it passed, so its ``first_seen_df`` is
      a POST-revision value. Putting it on the as-issued arm collapses the
      as-issued/as-revised distinction the dual arm exists to draw (#358).
      Evaluated right after ``no_df`` because it disqualifies the official
      arm's *provenance* — a more fundamental objection than "the settled
      value still equals it". That ordering moves some hours out of the stub
      buckets and into this one; the counts stay disjoint, but they are not
      comparable to payloads published before #358.
    * ``no_gridpulse`` — we have no matured prediction for the hour. Both
      arms are always scored on the SAME hour set; a one-sided score would
      compare a 30-day official record against a 1-day GridPulse one. Note
      the cost: the operator's forecast for that hour exists and goes
      unscored, so the sample is conditioned on OUR availability too.
    """
    out: list[PairedHour] = []
    from data.vintage import FRESH_CAPTURE_LAG_HOURS, capture_lag_hours

    drops = {
        "unresolved_stub": 0,
        "first_seen_placeholder": 0,
        "unsettled": 0,
        "no_df": 0,
        "no_gridpulse": 0,
        "stale_capture": 0,
    }
    for r in vintage_records:
        official = r.first_seen_df
        if not np.isfinite(official):
            drops["no_df"] += 1
            continue
        if exclude_stale_capture:
            lag = capture_lag_hours(r)
            # An unparseable timestamp cannot be shown fresh, and the official
            # arm's whole claim is that its value was seen before the hour
            # settled — so an unmeasurable lag is treated as stale rather than
            # waved through.
            if lag is None or lag > FRESH_CAPTURE_LAG_HOURS:
                drops["stale_capture"] += 1
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
    ``models.{model}.{lead}.records``, and takes **only the prediction** from
    them. The drift record's own ``actual`` is deliberately ignored: this
    module re-grades every prediction against the vintage ``last_d``, so both
    arms are scored by one yardstick computed in one place rather than each
    inheriting its own pipeline's notion of truth.
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


def serve_grade(
    horizon_payload: dict[str, Any] | None, model: str, lead: str
) -> dict[str, Any] | None:
    """Our own live grade for **the exact series this row scores** (#348).

    Read off the same ``models.{model}.{lead}`` block
    :func:`gridpulse_predictions` takes its predictions from — deliberately
    not a champion-across-models or a different horizon. The marker on the
    page must describe the line in that row, not a neighbouring measurement
    that happens to be healthier.

    Returns ``None`` when the block carries no grade, which is a warming
    region, not a passing one.
    """
    if not horizon_payload:
        return None
    block = ((horizon_payload.get("models") or {}).get(model) or {}).get(lead) or {}
    grade = block.get("grade")
    if not grade:
        return None
    # The boundary the number had to cross to earn this grade. `mape_grade`
    # returns `rollback` for anything ABOVE `acceptable` — the band dict's own
    # "rollback" entry (12.0 at 24h) is never used as a threshold, so quoting
    # it on the page would overstate how bad a flagged row has to be.
    from config import MAPE_BY_HORIZON

    bands = MAPE_BY_HORIZON.get(lead) or {}
    return {
        "grade": grade,
        "model": model,
        "horizon": lead,
        "rolling_mape_7d": block.get("rolling_mape_7d"),
        "n_7d": block.get("n_7d"),
        "acceptable_max": bands.get("acceptable"),
    }


def _stale_capture_impact(
    vintage_records: list[Any],
    gridpulse_by_ts: dict[str, float],
    revised_df_by_ts: dict[str, float] | None,
    gridpulse_now: dict[str, Any],
    official_now: dict[str, Any],
) -> dict[str, Any] | None:
    """How the #358 exclusion moved this BA's published numbers.

    Rescoring the same lead WITHOUT the filter is the only honest way to
    answer the methodology's §14 requirement continuously: the direction is
    not uniform across BAs (it favours the operator where its revisions
    improve its forecast and favours us where they worsen it), so a single
    fleet statement would be wrong for roughly half the fleet.

    ``None`` when nothing was excluded — the common case once a vintage window
    has rolled past its seed, and worth distinguishing from "measured zero".
    """
    pairs, drops = pair_hours(
        vintage_records, gridpulse_by_ts, revised_df_by_ts, exclude_stale_capture=False
    )
    n_stale = len(pairs) - int(gridpulse_now.get("n") or 0)
    if n_stale <= 0 or len(pairs) < MIN_PAIRED_HOURS:
        return None
    without = {"gridpulse": score_arm(pairs, "gridpulse"), "official": score_arm(pairs, "official")}
    return {
        "n_hours_excluded": n_stale,
        # Positive = the exclusion IMPROVED that arm's published MAPE.
        "gridpulse_mape_shift_pts": round(
            float(without["gridpulse"]["mape"] - gridpulse_now["mape"]), 3
        ),
        "official_mape_shift_pts": round(
            float(without["official"]["mape"] - official_now["mape"]), 3
        ),
        "note": (
            "same hours rescored without the stale-capture filter; positive "
            "means the exclusion improved that arm's MAPE"
        ),
    }


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
    served_series: str | None = None,
) -> dict[str, Any]:
    """Build ``gridpulse:benchmark:{region}`` for one tick.

    Always returns a payload — an excluded BA carries its reason rather than
    silently vanishing, because the exclusion list is part of what gets
    published.

    ``revised_df_by_ts`` supplies EIA's current (post-revision) day-ahead
    values so the conservative official arm can be scored; ``observed_lead_h``
    carries this tick's measured lead per nominal horizon, which decides
    whether the conservative label may be applied at all.

    ``served_series`` names what the product actually serves for this BA
    (``"model"`` or ``"seasonal-naive"``). It does not change a single score
    — this arm always grades ``model``, by design, so the benchmark keeps
    measuring the forecaster rather than quietly re-basing onto whatever we
    fell back to. But where the two differ the row is no longer describing
    what a user of that BA gets, and #348 is precisely about rows that carry
    context we hold and do not publish.
    """
    score = scoreability(vintage_records, revision_class)
    payload: dict[str, Any] = {
        "region": region,
        "revision_class": revision_class,
        "mean_revision_pct": mean_revision_pct,
        **score,
        "scored_model": model,
        "served_series": served_series,
        "serves_scored_model": None if served_series is None else served_series == "model",
        "leads": {},
    }
    if not score["scoreable"]:
        return payload

    for lead in (HEADLINE_LEAD, CONSERVATIVE_LEAD):
        gridpulse_by_ts_for_lead = gridpulse_predictions(horizon_payload, model, lead)
        pairs, drops = pair_hours(
            vintage_records,
            gridpulse_by_ts_for_lead,
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
            # The winner is decided on mean MAPE; this is published so a
            # reader can see immediately whether that call is metric-
            # dependent, instead of having to take the headline on trust.
            "delta_median_ape": round(official["median_ape"] - gridpulse["median_ape"], 3),
            "delta_mape_vs_revised": round(official_revised["mape"] - gridpulse["mape"], 3),
            "winner": "gridpulse" if gridpulse["mape"] < official["mape"] else "official",
            "winner_vs_revised": (
                "gridpulse" if gridpulse["mape"] < official_revised["mape"] else "official"
            ),
            "excluded_hours": drops,
            "observed_lead_h": None if observed is None else round(observed, 2),
            "lead_basis": "observed" if observed is not None else "nominal",
            # #358 §14: methodology requires stating which direction a rule
            # change moves our own number. Rather than predict it, or measure
            # it once and let it rot, the payload carries it — the same hours
            # rescored WITHOUT the stale-capture filter, so the delta is
            # published per BA, per tick, and stays true as windows roll.
            "stale_capture_impact": _stale_capture_impact(
                vintage_records, gridpulse_by_ts_for_lead, revised_df_by_ts, gridpulse, official
            ),
            # #348: our own rolling grade for THIS row's series. A row we
            # already grade `rollback` was being published as an ordinary
            # comparison — the one unflattering fact on the page that wasn't
            # disclosed deliberately.
            "serve_grade": serve_grade(horizon_payload, model, lead),
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
            # The consistency story: our spread vs theirs, both on MEAN
            # MAPE. Note the 41x figure quoted elsewhere is a *median* APE
            # spread from docs/BENCHMARK_SCOREABILITY.md — a different
            # statistic over a different hour set, not this field.
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
