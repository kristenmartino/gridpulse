"""
Forecast snapshot history for Time-Scrub Replay (NEXD-14).

Stores up to 30 forecast snapshots per (region, horizon, model) in SQLite.
Each snapshot captures the predicted timestamps + values at a specific point
in time, enabling users to compare how the forecast evolved.

Pure logic — no Dash dependencies.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import structlog

from config import CACHE_DB_PATH

log = structlog.get_logger()

MAX_SNAPSHOTS = 30

_initialized = False


def _predictions_hash(predictions: list[float]) -> str:
    """Compute a stable hash of predictions for content deduplication."""
    raw = ",".join(f"{p:.2f}" for p in predictions)
    return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324


def _init_db(conn: sqlite3.Connection) -> None:
    """Create the forecast_snapshots table if it doesn't exist."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT NOT NULL,
            horizon_hours INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            scored_at TEXT NOT NULL,
            predictions_json TEXT NOT NULL,
            timestamps_json TEXT NOT NULL,
            peak_mw REAL NOT NULL,
            avg_mw REAL NOT NULL,
            created_at REAL NOT NULL,
            predictions_hash TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
        ON forecast_snapshots (region, horizon_hours, model_name, scored_at)
    """)
    # Backfill: add predictions_hash column if upgrading from older schema
    import contextlib

    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(
            "ALTER TABLE forecast_snapshots ADD COLUMN predictions_hash TEXT NOT NULL DEFAULT ''"
        )
    conn.commit()
    _initialized = True


@contextmanager
def _connect():
    """Context manager for SQLite connections to the cache DB."""
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        _init_db(conn)
        yield conn
    finally:
        conn.close()


def save_forecast_snapshot(
    region: str,
    horizon_hours: int,
    model_name: str,
    timestamps: list[str],
    predictions: list[float],
) -> None:
    """Save a forecast snapshot to SQLite. Enforces FIFO (max 30 per combo).

    Args:
        region: Balancing authority code.
        horizon_hours: Forecast horizon in hours.
        model_name: Model identifier.
        timestamps: List of ISO timestamp strings.
        predictions: List of predicted demand values (MW).
    """
    scored_at = datetime.now(tz=UTC).isoformat()
    peak_mw = max(predictions) if predictions else 0.0
    avg_mw = sum(predictions) / len(predictions) if predictions else 0.0
    pred_hash = _predictions_hash(predictions)

    with _connect() as conn:
        # Content dedup: skip if predictions match the most recent snapshot
        latest = conn.execute(
            """SELECT predictions_hash FROM forecast_snapshots
               WHERE region = ? AND horizon_hours = ? AND model_name = ?
               ORDER BY scored_at DESC LIMIT 1""",
            (region, horizon_hours, model_name),
        ).fetchone()
        if latest and latest["predictions_hash"] and latest["predictions_hash"] == pred_hash:
            log.debug("snapshot_content_duplicate_skipped", region=region, model=model_name)
            return

        conn.execute(
            """INSERT INTO forecast_snapshots
               (region, horizon_hours, model_name, scored_at, predictions_json,
                timestamps_json, peak_mw, avg_mw, created_at, predictions_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                region,
                horizon_hours,
                model_name,
                scored_at,
                json.dumps(predictions),
                json.dumps(timestamps),
                peak_mw,
                avg_mw,
                time.time(),
                pred_hash,
            ),
        )

        # Enforce FIFO: delete oldest beyond MAX_SNAPSHOTS
        count = conn.execute(
            """SELECT COUNT(*) FROM forecast_snapshots
               WHERE region = ? AND horizon_hours = ? AND model_name = ?""",
            (region, horizon_hours, model_name),
        ).fetchone()[0]

        if count > MAX_SNAPSHOTS:
            excess = count - MAX_SNAPSHOTS
            conn.execute(
                """DELETE FROM forecast_snapshots WHERE id IN (
                       SELECT id FROM forecast_snapshots
                       WHERE region = ? AND horizon_hours = ? AND model_name = ?
                       ORDER BY scored_at ASC
                       LIMIT ?
                   )""",
                (region, horizon_hours, model_name, excess),
            )

        conn.commit()
        log.info(
            "forecast_snapshot_saved",
            region=region,
            horizon=horizon_hours,
            model=model_name,
            peak_mw=round(peak_mw),
        )


def list_forecast_snapshots(
    region: str,
    horizon_hours: int,
    model_name: str,
) -> list[dict]:
    """Return metadata for available snapshots, newest first.

    Args:
        region: Balancing authority code.
        horizon_hours: Forecast horizon.
        model_name: Model identifier.

    Returns:
        List of dicts with keys: scored_at, peak_mw, avg_mw.
    """
    with _connect() as conn:
        rows = conn.execute(
            """SELECT scored_at, peak_mw, avg_mw FROM forecast_snapshots
               WHERE region = ? AND horizon_hours = ? AND model_name = ?
               ORDER BY scored_at DESC""",
            (region, horizon_hours, model_name),
        ).fetchall()
        return [
            {"scored_at": row["scored_at"], "peak_mw": row["peak_mw"], "avg_mw": row["avg_mw"]}
            for row in rows
        ]


def get_forecast_snapshot(
    region: str,
    horizon_hours: int,
    model_name: str,
    scored_at: str,
) -> dict | None:
    """Retrieve a single snapshot by its scored_at timestamp.

    Args:
        region: Balancing authority code.
        horizon_hours: Forecast horizon.
        model_name: Model identifier.
        scored_at: ISO timestamp of the snapshot.

    Returns:
        Dict with timestamps, predictions, peak_mw, avg_mw, scored_at — or None.
    """
    with _connect() as conn:
        row = conn.execute(
            """SELECT scored_at, predictions_json, timestamps_json, peak_mw, avg_mw
               FROM forecast_snapshots
               WHERE region = ? AND horizon_hours = ? AND model_name = ? AND scored_at = ?""",
            (region, horizon_hours, model_name, scored_at),
        ).fetchone()
        if row is None:
            return None
        return {
            "scored_at": row["scored_at"],
            "predictions": json.loads(row["predictions_json"]),
            "timestamps": json.loads(row["timestamps_json"]),
            "peak_mw": row["peak_mw"],
            "avg_mw": row["avg_mw"],
        }


def build_replay_options(
    region: str,
    horizon_hours: int,
    model_name: str,
) -> list[dict]:
    """Build Dropdown options for the replay selector.

    Returns:
        List of {"label": str, "value": str} dicts. First entry is always
        {"label": "Current", "value": "current"}. Historical snapshots
        follow in newest-first order.
    """
    options: list[dict] = [{"label": "Current", "value": "current"}]
    with _connect() as conn:
        rows = conn.execute(
            """SELECT scored_at, peak_mw FROM forecast_snapshots
               WHERE region = ? AND horizon_hours = ? AND model_name = ?
               ORDER BY scored_at DESC""",
            (region, horizon_hours, model_name),
        ).fetchall()
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["scored_at"])
            label = dt.strftime("%b %d %H:%M")
            peak = row["peak_mw"]
            options.append(
                {
                    "label": f"{label} (Peak: {peak:,.0f} MW)",
                    "value": row["scored_at"],
                }
            )
        except (ValueError, KeyError):
            continue
    return options
