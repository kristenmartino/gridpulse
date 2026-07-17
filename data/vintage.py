"""Demand-vintage capture — what did EIA say about an hour, and when? (#309)

Every hour, the scoring job fetches a window of EIA demand and anchors the
forecast on the newest reading (``demand_history[-1]`` →
``feature_engineering.py:220`` → ``demand_lag_1h`` and 20 sibling
autoregressive features). EIA then **revises** those readings. Measured live
across 51 BAs on 2026-07-15, ``corr(revision magnitude, settled forecast
error) = 0.88``.

We cannot study that, because **we do not keep what EIA first said.**
``data/gcs_store.py`` writes ``{data_type}/{region}/latest.parquet`` — one blob
per region, overwritten every hour. The parquet always holds EIA's *current*
view. The only preliminary value that survives anywhere is
``DriftRecord.actual``, and only for the single hour that tick happened to
score. That is why the 0.88 correlation had to be measured by a live probe
(#305) rather than from history: **the history does not exist.**

This module is the missing recorder. For each target hour it pins:

* ``first_seen_d``   — the value EIA published when we first saw the hour.
  **This is the number the anchor actually used.** It never changes.
* ``first_seen_df``  — the BA's own day-ahead forecast at that same moment.
* ``last_d``         — EIA's latest value for the hour, i.e. settled.
* ``n_updates``      — how many times it moved.

``(first_seen_d → last_d)`` is exactly the pair a revision study needs, and
``captured_at`` dates the observation.

## Why this ships before any anchor fix

Three hypotheses about this defect have now died on contact with data (#304
twice; and on 2026-07-15 the "day-ahead placeholder poisons the anchor" theory,
below). Each was formed from a snapshot and refuted by a measurement. Without
vintages there is no way to replay what the anchor saw, so there is no honest
way to validate a fourth guess — including a confident one.

## What direct EIA measurement already showed (2026-07-15)

At the newest hour, 12 of 43 BAs publish ``D == DF`` **exactly** — the metered
field carrying the BA's own day-ahead forecast for an hour nobody has reported
yet. It is a placeholder: equality holds at 0-3% of *settled* hours, and ``DF``
is published for hours that have not happened (+1h…+4h) while ``D`` never is,
so the value can only flow forecast → actual.

**It is load-bearing, and must not be "fixed" without measuring.** Removing it
makes the forecast *worse*: persistence proxy over 14d, anchoring on the
placeholder scores 6.55% mean error vs 7.72% when skipped for the last measured
hour — it wins 9/12 BAs. A decent forecast *for the hour you want* beats a real
measurement two hours stale, because demand ramps. It only loses where the BA
forecasts itself badly (NEVP 14.60→8.40, PNM, SC).

It also does **not** explain the revisions it was supposed to:
``corr(day-ahead error, probe revision) = 0.15``. PNM's placeholder is 8.59%
wrong yet PNM revises 0.7%; BPAT's is 1.70% wrong yet BPAT revises 14.2%.
``was_placeholder`` is recorded here so that question can finally be settled
with data rather than another hypothesis.

## Contract

Pure functions, no I/O. The scoring job is the only writer
(``gridpulse:vintage:{region}``); the web tier never reads it. Capture must
never fail a scoring run — it is a measurement, not a critical path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# 30 days × 24 hours = 720 hourly records per region. Each row is ~90 bytes of
# JSON → ~65 KB per region × 51 regions ≈ 3 MB in Redis. Sized to match the
# drift window (``models/drift.py::DEFAULT_MAX_RECORDS``) so the two series
# cover the same span and join cleanly for analysis.
VINTAGE_WINDOW_HOURS = 24 * 30

#: A revision this small is rounding, not a revision. Used only for the
#: ``n_updates`` counter so float noise in EIA's own re-publication doesn't
#: inflate it — the recorded values are never rounded.
REVISION_EPSILON_MW = 0.5


@dataclass(frozen=True)
class VintageRecord:
    """What EIA said about one target hour, first and latest.

    Frozen: ``first_seen_d`` losing its value is the one way this module can
    fail silently and destroy the study, so the type makes it unassignable.
    Updates go through :func:`update_vintage_records`, which rebuilds the
    record rather than mutating it.
    """

    timestamp: str  # ISO-8601 UTC, the target hour
    first_seen_d: float  # what the anchor used
    first_seen_df: float  # BA day-ahead forecast at first sight; NaN if absent
    captured_at: str  # ISO-8601 UTC, when we first saw it
    last_d: float  # EIA's current value — settled, once revisions stop
    n_updates: int = 0

    @property
    def revision_pct(self) -> float | None:
        """``|last_d - first_seen_d| / last_d * 100``, or None if unusable.

        The headline number: how wrong the anchor's seed turned out to be.
        Denominated on ``last_d`` (settled truth), matching
        ``models.drift.absolute_pct_error``'s convention of dividing by the
        actual, so the two are directly comparable.
        """
        if not np.isfinite(self.last_d) or self.last_d <= 0:
            return None
        if not np.isfinite(self.first_seen_d):
            return None
        return abs(self.last_d - self.first_seen_d) / self.last_d * 100.0

    @property
    def was_placeholder(self) -> bool:
        """True when the first-seen reading *was* the BA's day-ahead forecast.

        Exact equality is the fingerprint, not a tolerance: metered demand does
        not land on a day-ahead forecast to the megawatt. Measured base rate on
        settled hours is 0-3%, vs 100% at the newest hour for 12/43 BAs.
        """
        if not np.isfinite(self.first_seen_d) or not np.isfinite(self.first_seen_df):
            return False
        return self.first_seen_d == self.first_seen_df


def canonical_hour(ts: Any) -> str | None:
    """``timestamp -> canonical ISO-8601 UTC string``, or None if unparseable.

    Produces the *same* canonical form as ``models.drift._normalize_ts`` (e.g.
    ``2026-07-15T12:00:00+00:00``) so vintage records join to drift records by
    timestamp. That compatibility is pinned by
    ``tests/unit/test_vintage.py::TestJoinsToDriftRecords`` rather than by
    importing the helper: ``models/`` already imports ``data/``
    (``models/training.py:24``), so a ``data → models`` import would invert the
    layering for six lines of string handling.
    """
    try:
        parsed = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    if parsed is None or pd.isna(parsed):
        return None
    parsed = parsed.tz_localize(UTC) if parsed.tzinfo is None else parsed.tz_convert(UTC)
    return parsed.isoformat()


def _readings(demand_df: Any) -> dict[str, tuple[float, float]]:
    """``{canonical_hour -> (d, df)}`` from a demand frame.

    Hours whose ``D`` is missing/non-positive are **skipped, not recorded as
    zero**: the anchor never sees them either (``jobs/phases.py:209`` drops NaN
    demand, and ``_resolve_forecast_start`` masks ``> 0``), so recording them
    would log a reading the forecast never used. ``forecast_mw`` is absent for
    some hours and rides as NaN — that is a fact about the hour, so it is kept.
    """
    if demand_df is None or len(demand_df) == 0:
        return {}
    if "timestamp" not in demand_df.columns or "demand_mw" not in demand_df.columns:
        return {}

    ts = pd.to_datetime(demand_df["timestamp"], utc=True, errors="coerce")
    d = pd.to_numeric(demand_df["demand_mw"], errors="coerce")
    if "forecast_mw" in demand_df.columns:
        df_ = pd.to_numeric(demand_df["forecast_mw"], errors="coerce")
    else:
        df_ = pd.Series([np.nan] * len(demand_df), index=demand_df.index)

    out: dict[str, tuple[float, float]] = {}
    for t, dv, fv in zip(ts, d, df_, strict=False):
        if pd.isna(t):
            continue
        if dv is None or not np.isfinite(dv) or dv <= 0:
            continue
        hour = canonical_hour(t)
        if hour is None:
            continue
        out[hour] = (float(dv), float(fv) if fv is not None and np.isfinite(fv) else float("nan"))
    return out


def update_vintage_records(
    existing: list[VintageRecord],
    demand_df: Any,
    *,
    now: datetime | None = None,
    window_hours: int = VINTAGE_WINDOW_HOURS,
) -> list[VintageRecord]:
    """Fold this tick's fetched window into the vintage series.

    For each hour present in ``demand_df``:

    * **unseen** → record ``first_seen_d`` / ``first_seen_df`` / ``captured_at``.
    * **seen, value moved** → bump ``n_updates`` and refresh ``last_d``.
      ``first_seen_d`` is never touched — it is the whole point of the record.

    Records whose target hour has aged out of ``window_hours`` are trimmed.
    Returns a chronologically-sorted list, oldest first. Never raises on a
    malformed frame; an unusable frame simply contributes nothing.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    by_ts: dict[str, VintageRecord] = {r.timestamp: r for r in existing}
    captured_at = now.isoformat()

    for hour, (d_val, df_val) in _readings(demand_df).items():
        prior = by_ts.get(hour)
        if prior is None:
            by_ts[hour] = VintageRecord(
                timestamp=hour,
                first_seen_d=d_val,
                first_seen_df=df_val,
                captured_at=captured_at,
                last_d=d_val,
                n_updates=0,
            )
            continue
        if abs(d_val - prior.last_d) <= REVISION_EPSILON_MW:
            continue
        by_ts[hour] = VintageRecord(
            timestamp=prior.timestamp,
            first_seen_d=prior.first_seen_d,
            first_seen_df=prior.first_seen_df,
            captured_at=prior.captured_at,
            last_d=d_val,
            n_updates=prior.n_updates + 1,
        )

    kept = []
    for rec in by_ts.values():
        when = canonical_hour(rec.timestamp)
        if when is None:
            continue
        if datetime.fromisoformat(when) >= cutoff:
            kept.append(rec)
    return sorted(kept, key=lambda r: r.timestamp)


def serialize_records(records: list[VintageRecord]) -> list[dict[str, Any]]:
    """Compact dict form for Redis JSON. Keys deliberately short.

    ``df`` is omitted when non-finite — JSON has no NaN, and a missing key
    round-trips back to NaN on deserialize. Values are rounded to 2dp (the MW
    resolution EIA itself publishes); ``first_seen_d`` and ``last_d`` are
    rounded identically so ``was_placeholder``'s exact-equality test and
    ``revision_pct`` survive the round trip unchanged.
    """
    return [
        {
            "ts": r.timestamp,
            "d": round(r.first_seen_d, 2),
            "at": r.captured_at,
            "ld": round(r.last_d, 2),
            "n": r.n_updates,
            **({"df": round(r.first_seen_df, 2)} if np.isfinite(r.first_seen_df) else {}),
        }
        for r in records
    ]


def deserialize_records(rows: list[dict[str, Any]] | None) -> list[VintageRecord]:
    """Inverse of :func:`serialize_records`. Tolerant of malformed rows.

    A single corrupt row drops out rather than poisoning the window — the same
    contract as ``models.drift.deserialize_records``.
    """
    if not rows:
        return []
    out: list[VintageRecord] = []
    for row in rows:
        try:
            df_raw = row.get("df")
            out.append(
                VintageRecord(
                    timestamp=str(row["ts"]),
                    first_seen_d=float(row["d"]),
                    first_seen_df=float(df_raw) if df_raw is not None else float("nan"),
                    captured_at=str(row["at"]),
                    last_d=float(row["ld"]),
                    n_updates=int(row.get("n", 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def summarize(records: list[VintageRecord], *, region: str) -> dict[str, Any]:
    """Flat structlog kwargs describing the vintage series for one region.

    Scalars only, so the fields land as queryable ``jsonPayload.*`` (#306).
    This is the shadow-mode signal: it answers "how often is the anchor's seed
    a placeholder, and how far do readings move afterwards?" — currently
    unknown, observed once by hand.
    """
    revisions = [r.revision_pct for r in records]
    revised = [v for v in revisions if v is not None]
    out: dict[str, Any] = {
        "region": region,
        "n_records": len(records),
        "n_placeholder": sum(1 for r in records if r.was_placeholder),
        "n_revised": sum(1 for r in records if r.n_updates > 0),
    }
    if revised:
        out["mean_revision_pct"] = round(float(np.mean(revised)), 4)
        out["max_revision_pct"] = round(float(np.max(revised)), 4)
    if records:
        newest = max(records, key=lambda r: r.timestamp)
        out["newest_hour"] = newest.timestamp
        out["newest_was_placeholder"] = newest.was_placeholder
    return out


# ── Revision-class heuristic (v1, provisional) ──────────────────────────────
# Thresholds live here, not config.py: these are instrument-internal study
# parameters expected to be re-tuned as the vintage table matures, not
# governance thresholds. Every cut below traces to measured prod behavior
# (2026-07-15..17): PNM/PJM revise ~never; BPAT revises ~every fresh hour at
# 15-30%; PSCO/FPL take daily-file bulk rewrites of dozens of deep-history
# hours; LDWP/AZPS/IID publish fresh values that revise 60-80%.

#: A record captured within this many hours of its target hour saw the value
#: "fresh" — its revision measures what the anchor/tiles experienced. Records
#: captured later (the first tick's ~700-hour backfill) were born settled.
FRESH_CAPTURE_LAG_HOURS = 3
#: Minimum fresh-captured records before classifying at all.
CLASS_MIN_FRESH = 24
#: broken: fresh readings routinely revise beyond this (gross partials).
CLASS_BROKEN_MEAN_REVISION_PCT = 25.0
#: churn: at least this fraction of fresh-captured records revised.
CLASS_CHURN_REVISED_FRACTION = 0.5
#: bulk: at least this many BACKFILLED (born-settled) records later revised —
#: the daily-file deep-history rewrite signature.
CLASS_BULK_BACKFILLED_REVISIONS = 5
#: clean: fresh revisions rarer than this fraction AND small.
CLASS_CLEAN_REVISED_FRACTION = 0.1
CLASS_CLEAN_MEAN_REVISION_PCT = 2.0


def _capture_lag_hours(record: VintageRecord) -> float | None:
    """Hours between the target hour and when we first saw it."""
    target = canonical_hour(record.timestamp)
    captured = canonical_hour(record.captured_at)
    if target is None or captured is None:
        return None
    delta = datetime.fromisoformat(captured) - datetime.fromisoformat(target)
    return delta.total_seconds() / 3600.0


def classify_region(records: list[VintageRecord]) -> dict[str, Any]:
    """Derive the region's revision class from its vintage window.

    Returns ``{"revision_class": ..., **evidence}`` where the class is one of
    ``clean | churn | bulk | broken | unknown`` and the evidence fields are the
    flat scalars the verdict rests on — auditable in logs and reusable by the
    provenance callouts without re-deriving.

    Precedence (a region can exhibit several signatures — LDWP is broken AND
    bulk): **broken > bulk > churn > clean**, because the labels drive
    user-facing copy and the most consequential caveat must win.

    Heuristic v1, deliberately conservative: with fewer than
    ``CLASS_MIN_FRESH`` fresh-captured records the answer is ``unknown`` —
    a callout that hedges correctly beats one that classifies wrongly.
    """
    fresh: list[VintageRecord] = []
    backfilled_revised = 0
    for r in records:
        lag = _capture_lag_hours(r)
        if lag is None:
            continue
        if lag <= FRESH_CAPTURE_LAG_HOURS:
            fresh.append(r)
        elif r.n_updates > 0:
            backfilled_revised += 1

    evidence: dict[str, Any] = {
        "n_fresh": len(fresh),
        "n_backfilled_revised": backfilled_revised,
    }

    fresh_revised = [r for r in fresh if r.n_updates > 0]
    if fresh:
        evidence["fresh_revised_fraction"] = round(len(fresh_revised) / len(fresh), 4)
    revisions = [r.revision_pct for r in fresh_revised if r.revision_pct is not None]
    if revisions:
        evidence["mean_fresh_revision_pct"] = round(float(np.mean(revisions)), 4)

    if len(fresh) < CLASS_MIN_FRESH:
        return {"revision_class": "unknown", **evidence}

    mean_rev = evidence.get("mean_fresh_revision_pct", 0.0)
    revised_fraction = evidence.get("fresh_revised_fraction", 0.0)

    if revisions and mean_rev >= CLASS_BROKEN_MEAN_REVISION_PCT:
        cls = "broken"
    elif backfilled_revised >= CLASS_BULK_BACKFILLED_REVISIONS:
        cls = "bulk"
    elif revised_fraction >= CLASS_CHURN_REVISED_FRACTION:
        cls = "churn"
    elif (
        revised_fraction <= CLASS_CLEAN_REVISED_FRACTION
        and mean_rev <= CLASS_CLEAN_MEAN_REVISION_PCT
    ):
        cls = "clean"
    else:
        cls = "unknown"
    return {"revision_class": cls, **evidence}
