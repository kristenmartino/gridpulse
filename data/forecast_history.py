"""
Forecast snapshot history for Time-Scrub Replay (NEXD-14).

Stores up to 30 forecast snapshots per (region, horizon, model) in SQLite.
Each snapshot captures the predicted timestamps + values at a specific point
in time, enabling users to compare how the forecast evolved.

GCS replication ensures snapshots survive Cloud Run container recycles.
Writes are fire-and-forget (background thread); reads restore SQLite on
first access when the table is empty.

Pure logic — no Dash dependencies.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import structlog

from config import CACHE_DB_PATH, GCS_BUCKET_NAME, GCS_ENABLED, GCS_PATH_PREFIX

log = structlog.get_logger()

MAX_SNAPSHOTS = 30

_initialized = False
_gcs_restored = False


def _predictions_hash(predictions: list[float]) -> str:
    """Compute a stable hash of predictions for content deduplication."""
    raw = ",".join(f"{p:.2f}" for p in predictions)
    return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324


def _gcs_blob_path(region: str, horizon_hours: int, model_name: str) -> str:
    """Build the GCS object path for a snapshot combo."""
    return f"{GCS_PATH_PREFIX}/snapshots/{region}/{horizon_hours}_{model_name}.json"


def _get_gcs_client():
    """Get the GCS client from gcs_store (lazy, shared singleton)."""
    try:
        from data.gcs_store import _get_client

        return _get_client()
    except Exception:
        return None


def _replicate_to_gcs(
    region: str,
    horizon_hours: int,
    model_name: str,
    conn: sqlite3.Connection,
) -> None:
    """Replicate all snapshots for a combo to GCS in a background thread.

    Serializes all snapshots as a JSON array and uploads to GCS.
    Fire-and-forget — never blocks the caller, never raises.
    """
    if not GCS_ENABLED or not GCS_BUCKET_NAME:
        return

    rows = conn.execute(
        """SELECT region, horizon_hours, model_name, scored_at,
                  predictions_json, timestamps_json, peak_mw, avg_mw,
                  created_at, predictions_hash
           FROM forecast_snapshots
           WHERE region = ? AND horizon_hours = ? AND model_name = ?
           ORDER BY scored_at DESC""",
        (region, horizon_hours, model_name),
    ).fetchall()

    snapshots = [
        {
            "region": row["region"],
            "horizon_hours": row["horizon_hours"],
            "model_name": row["model_name"],
            "scored_at": row["scored_at"],
            "predictions_json": row["predictions_json"],
            "timestamps_json": row["timestamps_json"],
            "peak_mw": row["peak_mw"],
            "avg_mw": row["avg_mw"],
            "created_at": row["created_at"],
            "predictions_hash": row["predictions_hash"],
        }
        for row in rows
    ]

    payload = json.dumps(snapshots).encode()
    blob_path = _gcs_blob_path(region, horizon_hours, model_name)

    def _upload() -> None:
        try:
            client = _get_gcs_client()
            if client is None:
                return
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(payload, content_type="application/json")
            log.debug(
                "gcs_snapshots_replicated",
                region=region,
                horizon=horizon_hours,
                model=model_name,
                count=len(snapshots),
            )
        except Exception as e:
            log.warning(
                "gcs_snapshot_replicate_failed",
                region=region,
                model=model_name,
                error=str(e),
            )

    threading.Thread(target=_upload, daemon=True).start()


def _restore_from_gcs(conn: sqlite3.Connection) -> None:
    """One-time bootstrap: restore snapshots from GCS when SQLite is empty.

    Called during _init_db. Scans the GCS snapshots prefix for all combo
    files and inserts them into the local SQLite table.
    """
    global _gcs_restored  # noqa: PLW0603
    if _gcs_restored or not GCS_ENABLED or not GCS_BUCKET_NAME:
        _gcs_restored = True
        return

    _gcs_restored = True

    # Only restore if the table is empty
    count = conn.execute("SELECT COUNT(*) FROM forecast_snapshots").fetchone()[0]
    if count > 0:
        log.debug("gcs_snapshot_restore_skipped", reason="table_not_empty", count=count)
        return

    try:
        client = _get_gcs_client()
        if client is None:
            return
        bucket = client.bucket(GCS_BUCKET_NAME)
        prefix = f"{GCS_PATH_PREFIX}/snapshots/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        total_restored = 0
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            try:
                data = json.loads(blob.download_as_bytes())
                for snap in data:
                    conn.execute(
                        """INSERT INTO forecast_snapshots
                           (region, horizon_hours, model_name, scored_at,
                            predictions_json, timestamps_json, peak_mw, avg_mw,
                            created_at, predictions_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            snap["region"],
                            snap["horizon_hours"],
                            snap["model_name"],
                            snap["scored_at"],
                            snap["predictions_json"],
                            snap["timestamps_json"],
                            snap["peak_mw"],
                            snap["avg_mw"],
                            snap["created_at"],
                            snap.get("predictions_hash", ""),
                        ),
                    )
                    total_restored += 1
            except Exception as e:
                log.warning("gcs_snapshot_blob_restore_failed", blob=blob.name, error=str(e))
                continue

        if total_restored > 0:
            conn.commit()
            log.info(
                "gcs_snapshots_restored",
                total=total_restored,
                blobs=len(blobs),
            )
    except Exception as e:
        log.warning("gcs_snapshot_restore_failed", error=str(e))


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
    _restore_from_gcs(conn)
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
        _replicate_to_gcs(region, horizon_hours, model_name, conn)
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
