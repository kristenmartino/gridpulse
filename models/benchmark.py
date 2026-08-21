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
from datetime import datetime
from typing import Any

import numpy as np

#: Fraction of hours a BA publishes a day-ahead forecast for. **Published,
#: never a gate** — see the "Coverage does not gate" section on
#: :func:`scoreability`. Retained only as the figure the exclusion sentences
#: quote; the warning band it once anchored was retired with the rate alert
#: it fed (#587).
MIN_DF_COVERAGE = 0.80

#: The longest stretch of the window a BA's day-ahead forecast may be missing
#: for before we stop scoring it. The gate that replaced the coverage
#: threshold (#549), measured over the whole window rather than only its
#: trailing edge since #587 — see :func:`_df_gaps`.
#:
#: Fitted to the fleet, not chosen: measured 2026-08-18 over the live 719-hour
#: window, the longest DF gap per BA was **SPP 345**, LDWP 48, WALC 48, TEC 30,
#: and every remaining BA at most 24. 168h sits 3.5x above the live fleet's
#: worst and 2x below the one dead feed, so neither side is near it. It is also
#: the repo's existing week unit (``rolling_eval`` windows, the drift 7-day
#: span).
MAX_DF_GAP_HOURS = 168.0

#: Below this many absent hours, ``absent_hours_bias_pct`` is noise and is
#: reported as ``None`` rather than as a number a reader would trust. The same
#: 2026-08-18 sweep: BAs missing 3-4 hours produced apparent load skews of
#: -20% (PACE, NWMT, IPCO) purely from which few hours were missing, against
#: +0.18% for SPP's 341 and +0.83% for TEC's 143.
MIN_ABSENT_HOURS_FOR_BIAS = 20
#: Minimum paired hours before a per-BA verdict is published. Sizing came
#: from *officially scoreable* hours — the count after the vintage-side drops
#: but BEFORE the ``no_gridpulse`` join, which is what
#: ``docs/BENCHMARK_SCOREABILITY.md`` reports: median 649 / min 500 per BA
#: over 30 days, against median 133 / min 46 over 7. Paired hours are a
#: subset of those, so the 30-day window is the one that can carry a per-BA
#: verdict and 7-day numbers are only ever a trend. The true paired count is
#: the per-lead ``n`` in the payload.
MIN_PAIRED_HOURS = 200

#: DF is not guaranteed to be measured on the same basis as D, and this is
#: EIA's own instruction rather than a respondent error. Form EIA-930 tells a
#: BA: "If you do not produce a day-ahead demand forecast in the normal course
#: of business that is directly comparable to actual demand as defined for this
#: collection ... you are not required to produce a consistent demand forecast
#: for the purposes of EIA-930 reporting. Please report the day-ahead demand
#: forecast generated in the normal course of business."
#:
#: The mechanism is the form's physical-vs-commercial split. D is defined
#: PHYSICALLY — everything inside the tie-line boundary, with "ownership and
#: dispatch ... irrelevant" and pseudo-ties and dynamic schedules excluded from
#: adjustment. A utility forecasts COMMERCIALLY: its own load obligation. Where
#: those two footprints differ, DF and D are different quantities and the gap
#: between them is not forecast error.
#:
#: Screen: a forecast off by more than the tolerance in the SAME direction on
#: essentially every hour is a definitional gap, not a miss. Both conditions
#: are required — a genuinely bad forecast is large but two-sided.
DF_SCOPE_RATIO_TOLERANCE = 0.10
DF_SCOPE_ONE_SIDED_MIN = 0.90

#: Feed classes that cannot be scored fairly (see module docstring).
UNSCOREABLE_CLASSES = frozenset({"broken"})

EXCLUDE_BROKEN_FEED = "broken-feed"
#: Retired as a gate by #549 and deliberately still defined: payloads written
#: before that change carry it, and the page groups it as a fairness exclusion.
#: Nothing emits it any more.
EXCLUDE_DF_COVERAGE = "df-coverage"
#: Likewise retired by #587 — it named only half of what the rule catches. A
#: feed that resumed still carries a hole, and calling that "stopped" is the
#: same species of false label #549 was about.
EXCLUDE_DF_FEED_STOPPED = "df-feed-stopped"
EXCLUDE_DF_FEED_GAP = "df-feed-gap"
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
    EXCLUDE_DF_FEED_GAP: (
        "This BA's day-ahead forecast is missing for a long enough stretch of "
        "the window that the hours we could score no longer describe the same "
        "period as every other row."
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


def df_scope(pairs: list[PairedHour]) -> dict[str, Any]:
    """Is the operator's DF measured on the same basis as its D?

    Returns the median DF/D ratio, the share of hours falling on one side of
    parity, and whether the pair passes as comparable. See
    ``DF_SCOPE_RATIO_TOLERANCE`` for why this is a scope question rather than
    an accuracy one.

    A failing row is still scored and still published — hiding it would be
    the opposite of the point. It is flagged, and kept out of the fleet
    aggregate, because a scope gap averaged into a fleet median silently
    becomes an accuracy claim about operators generally.
    """
    actual = np.array([p.actual for p in pairs], dtype=float)
    official = np.array([p.official for p in pairs], dtype=float)
    ok = np.isfinite(actual) & np.isfinite(official) & (actual > 0)
    actual, official = actual[ok], official[ok]
    if actual.size < 50:
        return {"ratio": None, "one_sided": None, "comparable": True, "basis": "too few hours"}

    ratio = float(np.median(official / actual))
    one_sided = float(max(np.mean(official > actual), np.mean(official < actual)))
    off_parity = abs(ratio - 1.0) > DF_SCOPE_RATIO_TOLERANCE
    persistent = one_sided > DF_SCOPE_ONE_SIDED_MIN
    comparable = not (off_parity and persistent)
    return {
        "ratio": round(ratio, 3),
        "one_sided": round(one_sided, 4),
        "comparable": comparable,
        "basis": (
            "comparable"
            if comparable
            else (
                f"DF runs at {ratio:.2f}x D on {one_sided:.0%} of hours in one direction — "
                f"a scope difference, not forecast error (EIA-930 does not require DF to be "
                f"comparable to D)"
            )
        ),
    }


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


def _hour(ts: Any) -> datetime | None:
    """Parse a vintage timestamp, or None if it does not parse.

    Never raises: a single unparseable row must not take down the gate. An
    unparseable timestamp is simply not evidence of anything, so it is skipped
    rather than defaulted — defaulting it would let one bad row decide whether
    a BA is published.
    """
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _df_gaps(vintage_records: list[Any]) -> dict[str, Any]:
    """Where, and how long, this BA's day-ahead forecast is missing.

    Returns ``{"stale_hours", "last_published_at", "longest_gap_hours",
    "longest_gap_end"}``, all ``None``/0 shaped for a BA that published nothing.

    ## Gaps in TIME, not runs of absent records

    An hour EIA never published a positive metered ``D`` for is not a record at
    all, so counting runs of absent *records* silently under-measures a hole
    that contains one. This walks the timestamps of the hours that DO carry a
    DF and measures the distance between them, which is the same number when no
    records are missing and the correct one when some are.

    Three kinds of gap, all treated alike:

    * **leading** — window start to the first published DF
    * **interior** — between two published DFs
    * **trailing** — last published DF to window end. This one is
      ``stale_hours``, i.e. "has the feed stopped?"

    ## Why the gate reads the longest and not the trailing one (#587)

    Trailing-only is right on the way out of a feed outage and wrong on the way
    back in. When a dead feed resumes, the trailing gap collapses to ~0 on the
    first tick while the hole it left is still sitting in the window — so the
    BA would be scored over two disjoint clusters of hours separated by that
    hole, published under a "last 30 days" header. That is precisely the
    condition ``EXCLUDE_DF_FEED_GAP``'s own text describes, and the
    trailing-edge version of this gate could only see it in one direction.

    Taking the max costs nothing on the way out — for a feed that is still dark
    the trailing gap IS the longest — and keeps the BA excluded after it
    resumes until the hole ages out of the window.
    """
    covered = sorted(
        t for r in vintage_records if np.isfinite(r.first_seen_df) and (t := _hour(r.timestamp))
    )
    all_ts = [t for r in vintage_records if (t := _hour(r.timestamp))]
    if not covered or not all_ts:
        return {
            "stale_hours": None,
            "last_published_at": None,
            "longest_gap_hours": None,
            "longest_gap_end": None,
        }

    start, end = min(all_ts), max(all_ts)
    hours = lambda a, b: (b - a).total_seconds() / 3600.0  # noqa: E731

    # (length, last absent hour of the gap). The trailing gap's "last absent
    # hour" is the window edge; a leading gap's is the hour before first cover.
    gaps: list[tuple[float, Any]] = [
        (hours(start, covered[0]), covered[0]),  # leading
        (hours(covered[-1], end), end),  # trailing
    ]
    gaps += [
        (hours(covered[i], covered[i + 1]) - 1.0, covered[i + 1])
        for i in range(len(covered) - 1)
        if hours(covered[i], covered[i + 1]) > 1.0
    ]
    longest, longest_end = max(gaps, key=lambda g: g[0])

    return {
        "stale_hours": hours(covered[-1], end),
        "last_published_at": _normalize_ts(covered[-1].isoformat()),
        "longest_gap_hours": longest,
        "longest_gap_end": _normalize_ts(longest_end.isoformat()) if longest > 0 else None,
    }


def _absent_hour_bias_pct(vintage_records: list[Any]) -> float | None:
    """Do the hours a BA skipped carry different load than the hours it published?

    ``(mean D on absent hours - mean D on covered hours) / mean D on covered``,
    as a percentage. This is the hazard the old coverage gate was a proxy for —
    a BA that goes quiet when the grid gets hard would be graded only on its
    easy hours. Published, and it gates nothing: promoting it to a gate needs a
    disqualifying magnitude, and the fleet has not yet produced one to calibrate
    against (#549).

    ``None`` when it cannot mean anything: no absent hours (no bias is
    *possible* — distinct from unmeasurable), no covered hours, or too few
    absent hours for the mean to be stable. Non-finite and non-positive ``D``
    are dropped from **both** sides, so one bad row cannot move it.

    ## It is confounded wherever the absence is BLOCKED — which here is always

    The statistic compares two sets of hours as though they were drawn from the
    same period. That holds when absences are scattered through the window. It
    does not hold for a contiguous outage, where the difference also carries
    whatever load happened to do during those particular days.

    Measured on the live payload 2026-08-18: **WALC −19.88%** over 49 absent
    hours, which sit in two contiguous ~48h blocks, and **AVA −17.41%** over 24
    hours in one block. A 48h block spans two full diurnal cycles, so neither is
    an off-peak artifact — those days were milder, and the BA's choice of hours
    had nothing to do with it.

    ``MIN_ABSENT_HOURS_FOR_BIAS`` does not address this. It was fitted to drop
    3–4-hour cases, and the confound is a property of the absence's **shape**,
    not its size. Since no BA in this fleet is diffusely absent (#549), every
    value this function returns today carries it to some degree.

    So read it as *"were the dark hours different?"* — never as *"does this BA
    withhold its forecast on hard hours?"* Those coincide only for scattered
    absence. **Separating them is a precondition for promoting this to a gate,**
    on top of finding a disqualifying magnitude: a blocked-absence BA could be
    excluded for a cold snap it did not choose.
    """
    covered = [
        float(r.last_d)
        for r in vintage_records
        if np.isfinite(r.first_seen_df) and np.isfinite(r.last_d) and r.last_d > 0
    ]
    absent = [
        float(r.last_d)
        for r in vintage_records
        if not np.isfinite(r.first_seen_df) and np.isfinite(r.last_d) and r.last_d > 0
    ]
    if not covered or len(absent) < MIN_ABSENT_HOURS_FOR_BIAS:
        return None
    mean_covered = float(np.mean(covered))
    if mean_covered <= 0:
        return None
    return round((float(np.mean(absent)) - mean_covered) / mean_covered * 100.0, 2)


def scoreability(vintage_records: list[Any], revision_class: str | None) -> dict[str, Any]:
    """Can this BA be scored, and if not, exactly why?

    Returns ``{"scoreable", "reason", "reason_detail", "df_coverage",
    "df_asissued_coverage", "placeholder_pct", "n_hours"}``. The reason travels
    with the verdict so every exclusion is publishable.

    ## Two coverages, because one number was answering two questions (#535)

    * ``df_coverage`` — **the BA's publication rate.** Does this balancing
      authority publish a day-ahead forecast at all? A property of the BA, and
      the only one this gate is entitled to act on, because the exclusion text
      it produces makes a claim about the BA.
    * ``df_asissued_coverage`` — **our capture quality.** What share of hours
      did we observe the DF for *before the hour settled*, i.e. in time for the
      as-issued arm? A property of our collector. Published, never a gate.

    Until #535 these were the same number. ``first_seen_df`` was pinned on the
    tick that first admitted the hour and never revisited, so a DF that EIA
    published slightly later was lost permanently — and the gate read that loss
    as sparse publishing. Twenty-six BAs were excluded on it, five of them large
    ISOs, on a public page. Measured against EIA directly on 2026-08-17, exactly
    one of the excluded set (SPP, 53.8%) is genuinely below the threshold.

    ``data.vintage`` now gives the DF a second look, so ``df_coverage`` means
    what it says. The capture-quality half did not disappear — it moved to its
    own field, where a reader can see it and it decides nothing.

    ## Coverage does not gate; feed liveness does (#549)

    #535 fixed *what* the coverage number measured. It did not establish that a
    publication **rate** is the right thing to exclude on, and it is not: a rate
    cannot tell a BA that half-publishes from a BA that published completely and
    then stopped, and the exclusion sentence it produced asserted the first
    shape for both.

    Measured across all 51 BAs on 2026-08-18, the distinction the rate was
    assumed to be drawing does not exist in this fleet. **No BA is diffusely
    sparse.** Every BA with any absence has 92-100% of its absent hours inside
    runs of >=3h. SPP — the one BA the coverage gate ever correctly excluded,
    and which this module previously described as genuinely not publishing — is
    absent in **one contiguous 341-hour block**: its feed stopped at
    ``2026-08-04T06Z`` and did not resume. TEC, at 80.1%, is absent in six
    blocks and publishes 100% of hours on the days it publishes at all. Both
    were confirmed against EIA directly, so neither is a capture artifact.

    What actually separates them is **liveness**, and it separates them
    cleanly: hours since the newest published DF were SPP 341, TEC 30, every
    other BA <=6. So that is what gates.

    The defect in scoring a stopped feed is not thinness — ``MIN_PAIRED_HOURS``
    already owns sample size, and SPP's covered hours are unbiased (+0.18%).
    It is that every hour we could score predates the stop, so the row
    describes a different slice of the window than every other row, under a
    header that says 30 days. That is #535's defect class — a headline
    describing a population the reader does not expect — one level up.

    ``df_coverage`` follows ``df_asissued_coverage`` into the published-but-
    deciding-nothing set, joined by ``absent_hours_bias_pct``: the hazard the
    rate was standing in for, now measured directly instead of assumed.
    """
    n = len(vintage_records)
    if n == 0:
        return {
            "scoreable": False,
            "reason": EXCLUDE_INSUFFICIENT,
            "reason_detail": EXCLUSION_REASONS[EXCLUDE_INSUFFICIENT],
            "df_coverage": 0.0,
            "df_asissued_coverage": 0.0,
            "placeholder_pct": 0.0,
            "n_hours": 0,
            "n_absent_hours": 0,
            "df_stale_hours": None,
            "df_last_published_at": None,
            "df_longest_gap_hours": None,
            "df_longest_gap_end": None,
            "absent_hours_bias_pct": None,
        }

    from data.vintage import FRESH_CAPTURE_LAG_HOURS, df_capture_lag_hours

    has_df = sum(1 for r in vintage_records if np.isfinite(r.first_seen_df))
    as_issued = sum(
        1
        for r in vintage_records
        if np.isfinite(r.first_seen_df)
        and (lag := df_capture_lag_hours(r)) is not None
        and lag <= FRESH_CAPTURE_LAG_HOURS
    )
    placeholders = sum(1 for r in vintage_records if r.was_placeholder)
    coverage = has_df / n

    gaps = _df_gaps(vintage_records)
    stale_h, last_df_at = gaps["stale_hours"], gaps["last_published_at"]
    longest_gap = gaps["longest_gap_hours"]

    reason: str | None = None
    if revision_class in UNSCOREABLE_CLASSES:
        reason = EXCLUDE_BROKEN_FEED
    # A BA that published nothing at all in the window has `longest_gap is
    # None`, and is excluded here rather than falling through as though its
    # feed were fresh — "no DF anywhere" is the strongest possible form of the
    # condition this gate tests, not an absence of evidence about it.
    elif longest_gap is None or longest_gap > MAX_DF_GAP_HOURS:
        reason = EXCLUDE_DF_FEED_GAP

    return {
        "scoreable": reason is None,
        "reason": reason,
        # The measured figures, not just the rule. A reader who sees only
        # "under 80% of hours" cannot tell 79% from 39%, and those are
        # different facts about a BA.
        "reason_detail": _reason_detail(reason, coverage, as_issued / n, n, gaps),
        "df_coverage": round(coverage, 4),
        "df_asissued_coverage": round(as_issued / n, 4),
        "placeholder_pct": round(placeholders / n * 100, 2),
        "n_hours": n,
        "n_absent_hours": n - has_df,
        # The gate's own figure, published whether or not it fired, so a reader
        # can see how far from it a scored BA sits instead of taking the
        # verdict on trust.
        "df_stale_hours": None if stale_h is None else round(stale_h, 1),
        "df_last_published_at": last_df_at,
        # The gate's own figure since #587. `df_stale_hours` is the trailing
        # gap only; this is the worst gap anywhere in the window, so it stays
        # high after a dead feed resumes and the hole is still being scored
        # across.
        "df_longest_gap_hours": None if longest_gap is None else round(longest_gap, 1),
        "df_longest_gap_end": gaps["longest_gap_end"],
        "absent_hours_bias_pct": _absent_hour_bias_pct(vintage_records),
    }


def _reason_detail(
    reason: str | None,
    coverage: float,
    as_issued_coverage: float,
    n_hours: int,
    gaps: dict[str, Any] | None = None,
) -> str | None:
    """The published exclusion rationale, carrying THIS BA's measured numbers.

    ``EXCLUSION_REASONS`` states the rule; this states the case. Both DF-side
    texts name the as-issued share, because #535 was precisely a reader — us —
    unable to tell "the BA barely publishes" from "we barely captured it" from
    the sentence that shipped.

    The gap text names **dates**, which is the whole point of #549: the
    sentence that preceded it asserted a *shape* ("too sparse") that had never
    been measured and was wrong for both BAs it was ever applied to. A date is
    checkable against EIA in one query; a shape adjective is not.

    Two cases, because since #587 a BA can be excluded for a hole it has
    already recovered from, and telling a reader its feed "has stopped" would
    then be false:

    * **still dark** — the longest gap runs to the window edge. Name when it
      last published, and how long ago that was.
    * **resumed, hole still in the window** — name the gap's length and when it
      ended, and say plainly that it is publishing again.
    """
    if reason is None:
        return None
    base = EXCLUSION_REASONS[reason]
    if reason != EXCLUDE_DF_FEED_GAP:
        return base
    gaps = gaps or {}
    last_df_at = gaps.get("last_published_at")
    if last_df_at is None:
        return (
            f"{base} Over the {n_hours}-hour window EIA carried no day-ahead "
            "forecast for this BA at all."
        )
    measured = (
        f"Measured over {n_hours} hours: EIA published a day-ahead forecast "
        f"for {coverage:.1%} of them, of which we captured "
        f"{as_issued_coverage:.1%} in time to score as-issued."
    )
    stale = gaps.get("stale_hours") or 0.0
    longest = gaps.get("longest_gap_hours") or 0.0
    # The feed is still down when the worst hole is the one still open at the
    # window edge; anything else means it came back and the hole is history.
    if stale >= longest:
        return (
            f"{base} It has not published one since {last_df_at}, "
            f"{longest:.0f} hours ago. {measured}"
        )
    return (
        f"{base} It is publishing again — most recently for {last_df_at} — but "
        f"went {longest:.0f} hours without a day-ahead forecast, through "
        f"{gaps.get('longest_gap_end')}, and that gap is still inside the "
        f"scoring window. {measured}"
    )


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
    * ``stale_capture`` — the hour's ``DF`` was first seen more than
      ``FRESH_CAPTURE_LAG_HOURS`` after the hour passed, so its ``first_seen_df``
      is a POST-revision value. Measured on ``df_at`` since #535, because a DF
      filled on a later tick is exactly the case this rule exists to catch. Putting it on the as-issued arm collapses the
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
    from data.vintage import FRESH_CAPTURE_LAG_HOURS, df_capture_lag_hours

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
            # THE DF's lag, not the hour's. Since #535 a DF may be filled on a
            # tick later than the one that admitted the hour, and it is the DF
            # observation that the as-issued claim is about. Reverting this to
            # `capture_lag_hours` would silently put post-revision values on the
            # as-issued arm at scale — the exact #358/#392 defect, and the one
            # way the #535 fix could do more harm than the bug. Pinned by
            # TestStaleCapture::test_late_filled_df_cannot_reach_the_asissued_arm.
            lag = df_capture_lag_hours(r)
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
            # Whether the operator's arm is even the same quantity as the
            # truth it is graded against. EIA does not require that it is.
            "df_scope": df_scope(pairs),
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

    def _flagged(p: dict[str, Any]) -> bool:
        block = (p.get("leads") or {}).get(HEADLINE_LEAD) or {}
        return not bool((block.get("df_scope") or {}).get("comparable", True))

    # Reported, NOT acted on. The screen cannot separate "DF is a different
    # quantity" from "DF is a very bad forecast" — a large error is one-sided
    # either way — and dropping the second from the fleet median would
    # flatter us. So these rows stay in every aggregate and the reader is
    # told which ones carry the question.
    scope_flagged = [p["region"] for p in scored if _flagged(p)]

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
        # Advisory only — these regions ARE in the fleet figures above.
        "scope_flagged": sorted(scope_flagged),
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


# ── scoreability regression detection (#535) ─────────────────


def scoreability_alerts(
    rollup: dict[str, Any],
    region_payloads: list[dict[str, Any]],
    *,
    min_scoreable: int | None = None,
    gap_warn_hours: float | None = None,
) -> list[dict[str, Any]]:
    """Alertable events when the scorecard's population is shrinking (#535).

    Pure: returns ``[{"event", **fields}]`` for the caller to log. The scoring
    job owns the I/O; the rule lives here beside the gate it watches, and is
    unit-testable without Redis.

    ## Why these events and not a threshold on the count we already log

    ``benchmark_fleet_written`` has carried ``scoreable``/``excluded`` since
    E0, and a metric-threshold alert on it is the obvious design and a trap —
    the same one ``docs/monitoring/backtest_recompute_alert.json`` documents.
    It would have to be a **threshold-BELOW** condition, so when the benchmark
    phase stops emitting entirely the logs-based counter has *no data*, the
    condition never evaluates, and the alert goes quiet at precisely the moment
    it should fire.

    Both events below therefore fire on the **failing** direction and increase
    as things get worse, so absence of data can never mask them. A benchmark
    phase that stops running is a different failure, already covered by
    ``cloud_run_job_failure_alert`` and ``scoring_partial_failure``.

    ## Two distances from the same cliff

    * ``benchmark_scoreability_drop`` — the fleet count is below the floor.
      The incident itself: #535 published 25 of 51 for three weeks.
    * ``benchmark_df_gap_at_risk`` — a BA is still scoreable but its longest
      DF gap has entered the warning band **below** the gate. By the time a BA
      falls out, the public page is already wrong; this names it first.

    ``df_asissued_coverage`` is deliberately NOT alerted on. It measures our
    capture, not the BA's publishing, and gating on it is the whole of #535.
    It rides along on the payload so an on-call reader can tell the two apart.

    ## The gate's own warning, and why the ordering is no longer an accident

    ``benchmark_df_gap_at_risk`` — a still-scoreable BA whose longest DF gap
    has passed ``BENCHMARK_DF_GAP_WARN_HOURS`` on its way to
    ``MAX_DF_GAP_HOURS``. This is the one that watches the gate.

    #587 was not that no warning existed; a dying feed's coverage did cross
    the 0.85 band roughly 60h before the 168h gate. It was that **the warning
    measured a rate and the gate measured a duration**, so "warn first" was an
    arithmetic coincidence of two unrelated constants and the window length.
    Change the window to 14 days and the band trips at ~50h; widen the gate
    and the order inverts. Nothing tested it because nothing could: the two
    numbers were not comparable.

    Both are now hours of gap on the same measurement, so the thresholds
    order by construction for any window length. That is necessary and it is
    **not sufficient**, which the first draft of this docstring got wrong:

    * The warning is only *observed* if some tick evaluates the BA while its
      gap is inside the band. ``scoreability_alerts`` is fed payloads read
      back from Redis (``jobs/scoring_job.py``), not the ones just computed,
      so what it sees can lag the truth.
    * The band is therefore only safe while **the widest a payload can lag is
      narrower than the band**. It is: ``REDIS_TTL`` is 24h, the band is 48h,
      so a payload stale enough to skip the band has expired instead — and a
      BA whose key expired leaves ``payloads`` entirely and is counted by
      ``benchmark_scoreability_drop``, a louder alert, rather than slipping
      through quietly.

    Those two constants live in different files and nothing connected them, so
    this relation was itself an accident of the kind #587 is about.
    ``test_the_band_is_wider_than_a_payload_can_go_stale`` now pins it.

    A first observation already past the gate — a newly-tracked BA, a rebuilt
    vintage store — gets no warning, correctly: there is no lead time to give
    when the condition is already true.

    ## What this warning is scoped to

    Only BAs that are **currently scored**. A BA excluded for another reason
    (``broken-feed``, ``insufficient-paired-hours``) is not watched, and that
    is deliberate rather than an oversight: the warning exists to give lead
    time before the published population *changes*, and a BA that is already
    out cannot make the page wrong by staying out. It answers "who might
    leave", never "who might fail to return".

    ## Why ``benchmark_coverage_at_risk`` is gone

    Retired here rather than left running. Once coverage stopped gating (#549)
    it was watching a number that decides nothing, and the measurement showed
    what that costs: it fired on **every tick** from the #580 deploy onward for
    TEC — 0.822 and drifting, feed fully alive, correctly scored, nothing to
    do about it. A permanently-firing alert on a healthy BA is worse than no
    alert, because it trains the reader to ignore the channel that the real
    warning now shares.

    **The gap it leaves, stated rather than papered over:** a BA whose
    publication rate decays *diffusely* — many short holes, no long one — trips
    neither the new warning nor the gate. No BA in this fleet behaves that way
    (#549 measured it: every BA with any absence has 92-100% of those hours in
    runs of >=3h), so this is a shape we have never observed rather than one we
    have decided to tolerate. If it appears, it needs its own measurement, not
    a resurrected rate threshold.
    """
    from config import BENCHMARK_DF_GAP_WARN_HOURS, BENCHMARK_MIN_SCOREABLE

    floor = BENCHMARK_MIN_SCOREABLE if min_scoreable is None else min_scoreable
    warn = BENCHMARK_DF_GAP_WARN_HOURS if gap_warn_hours is None else gap_warn_hours

    out: list[dict[str, Any]] = []
    n_scoreable = rollup.get("n_scoreable")
    if isinstance(n_scoreable, int) and n_scoreable < floor:
        out.append(
            {
                "event": "benchmark_scoreability_drop",
                "n_scoreable": n_scoreable,
                "n_excluded": rollup.get("n_excluded"),
                "floor": floor,
                # WHICH BAs, not just how many — "26 excluded" and "26 excluded
                # including five large ISOs" are different pages.
                "excluded_regions": sorted(
                    str(e.get("region")) for e in (rollup.get("excluded") or [])
                ),
            }
        )

    for p in region_payloads:
        if not p.get("scoreable"):
            continue
        gap = p.get("df_longest_gap_hours")
        if not isinstance(gap, int | float) or gap < warn:
            continue
        out.append(
            {
                "event": "benchmark_df_gap_at_risk",
                "region": p.get("region"),
                # The gate's own quantity, the warning line, and the gate — all
                # in the same unit, which is the whole of #587.
                "df_longest_gap_hours": round(float(gap), 1),
                "warn_above": warn,
                "gate_gap_hours": MAX_DF_GAP_HOURS,
                # Trailing gap beside the longest, so a reader can tell "the
                # feed is down right now" from "it came back and the hole is
                # still in the window" without opening the payload.
                "df_stale_hours": p.get("df_stale_hours"),
                "df_last_published_at": p.get("df_last_published_at"),
                "df_coverage": p.get("df_coverage"),
                "df_asissued_coverage": p.get("df_asissued_coverage"),
            }
        )
    return out
