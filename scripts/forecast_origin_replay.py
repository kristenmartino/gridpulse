"""Replay the forecast origin, tick by tick, against the origin production carried.

#537's open half: LGEE's forecast payload froze at one origin for 15 ticks and
then served an origin 23 hours OLDER than one it had already served. A monotonic
origin should never go backwards.

Method (the repo's standing one): do not reason about candidate mechanisms —
diff two things that should agree, then look for structure in the disagreement.

  COMPUTED  the origin ``_resolve_forecast_start`` returns when replayed against
            the demand frame that BA actually held at that tick
  CARRIED   the origin the payload actually shipped, recovered from the
            ``drift_updated`` log as ``new_record_ts - lead_hours + 1``

Reconstructing the per-tick frame. The mechanism under test is driven by which
hours were MISSING, not by their values, and missing-ness is exactly
recoverable: ``VintageRecord.captured_at`` records when an hour was first seen
carrying a positive ``D``. So for a tick at wall-clock ``T``::

    demand[h] = NaN  if h is in the vintage window and captured_at(h) > T
                cur  otherwise   (cur = the current mirrored value)

Hours that never carried a positive ``D`` are absent from the vintage window and
are already NaN in the mirror, so the same rule covers them.

STATED APPROXIMATION: revision *timing* is not recoverable per hour (the record
keeps ``n_updates``, not a history), so a value revised between ``T`` and now is
reconstructed at its settled value rather than its as-of-``T`` value. That does
not touch this measurement — the origin is set by ``dropna``, which sees
NaN-ness and not magnitude. The independence check below is what makes that
claim testable rather than asserted.

Reads the GCS mirror with ADC; no Redis, no VPC, no job execution. Follows
``scripts/anchor_conditioning_study.py``.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import timedelta

import pandas as pd
import structlog

from data.feature_engineering import engineer_features
from jobs.phases import _resolve_forecast_start

log = structlog.get_logger()

_BUCKET = "nextera-portfolio-energy-cache"
_FETCH_WINDOW_DAYS = 90  # data/eia_client.py:324-329


def _load(kind: str, region: str) -> pd.DataFrame:
    from google.cloud import storage

    blob = storage.Client().bucket(_BUCKET).blob(f"cache/{kind}/{region}/latest.parquet")
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def _current_frame(region: str) -> pd.DataFrame:
    df = _load("demand", region)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _capture_times(region: str) -> pd.Series:
    """``timestamp -> captured_at`` for every hour the vintage window holds."""
    v = _load("vintage", region)
    return pd.Series(
        pd.to_datetime(v["at"], utc=True).to_numpy(dtype="datetime64[ns]"),
        index=pd.to_datetime(v["ts"], utc=True),
    ).dt.tz_localize("UTC")


def frame_as_of(current: pd.DataFrame, captured: pd.Series, tick: pd.Timestamp) -> pd.DataFrame:
    """The demand frame the scoring job held at ``tick``."""
    start = (tick - timedelta(days=_FETCH_WINDOW_DAYS)).floor("D")
    end = tick.floor("D") + timedelta(hours=23)
    df = current[(current["timestamp"] >= start) & (current["timestamp"] <= end)].copy()

    # Hours not yet captured at this tick were NaN in the frame then. Compare on
    # the capture's own HOUR, not its instant: ``captured_at`` is stamped a few
    # minutes into the tick that first saw the hour (12:04:40 for the 12:00
    # tick), so an instant comparison against the tick hour would exclude the
    # very hour that tick captured — an off-by-one that silently cancels the
    # producer/consumer tick offset below and manufactures agreement.
    unseen = captured.reindex(pd.DatetimeIndex(df["timestamp"]))
    not_yet = (unseen.dt.floor("h") > tick).fillna(False).to_numpy()
    df.loc[not_yet, "demand_mw"] = float("nan")
    return df.reset_index(drop=True)


def replay_tick(demand_df: pd.DataFrame) -> dict:
    """Run the real primitives and report both terms of the ``min()``."""
    featured = engineer_features(demand_df).dropna(subset=["demand_mw"])  # jobs/phases.py:315
    if featured.empty:
        return {"origin_computed": None, "last_featured_ts": None, "last_real_demand": None}

    mask = demand_df["demand_mw"].notna() & (demand_df["demand_mw"] > 0)
    last_real = demand_df.loc[mask, "timestamp"].max()
    last_featured = featured["timestamp"].max()
    return {
        "origin_computed": _resolve_forecast_start(featured, demand_df),
        "last_real_demand": last_real,
        "last_featured_ts": last_featured,
        "binding_term": "featured" if last_featured <= last_real else "real_demand",
        "featured_rows": len(featured),
    }


def carried_origins(path: str) -> dict[tuple[str, pd.Timestamp], pd.Timestamp]:
    """``(region, tick hour) -> origin`` from ``drift_updated`` log rows."""
    out: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 4 or not parts[3]:
                continue
            wall, region, target, lead = parts
            origin = pd.Timestamp(target) - pd.Timedelta(hours=int(lead) - 1)
            out[(region, pd.Timestamp(wall).floor("h"))] = origin
    return out


def run(regions: list[str], first: str, last: str, drift_csv: str) -> list[dict]:
    carried = carried_origins(drift_csv)
    ticks = pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="h", tz="UTC")
    rows: list[dict] = []

    for region in regions:
        current = _current_frame(region)
        captured = _capture_times(region)
        log.info("replay_region_loaded", region=region, rows=len(current), vintage=len(captured))

        for tick in ticks:
            frame = frame_as_of(current, captured, tick)
            if frame["demand_mw"].notna().sum() < 200:
                continue
            res = replay_tick(frame)
            # A tick's drift record grades the payload written by the PREVIOUS
            # tick (``read_existing_forecast`` runs before the forecast phase),
            # so the origin recovered at T is the origin computed at T-1.
            got = carried.get((region, tick + pd.Timedelta(hours=1)))
            rows.append(
                {
                    "region": region,
                    "tick": tick.isoformat(),
                    "origin_computed": _iso(res["origin_computed"]),
                    "origin_carried_next_tick": _iso(got),
                    "agree": _iso(res["origin_computed"]) == _iso(got) if got is not None else None,
                    "last_real_demand": _iso(res["last_real_demand"]),
                    "last_featured_ts": _iso(res["last_featured_ts"]),
                    "binding_term": res.get("binding_term"),
                    "newest_demand_hour": _iso(
                        frame.loc[frame["demand_mw"].notna(), "timestamp"].max()
                    ),
                }
            )
    return rows


def _iso(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


if __name__ == "__main__":
    regions = sys.argv[1].split(",")
    rows = run(regions, sys.argv[2], sys.argv[3], sys.argv[4])
    with open(sys.argv[5], "w") as fh:
        json.dump(rows, fh)
