"""
Audit wrapper for batch scoring runs.

Records model versions, data hashes, feature hashes, and ensemble weights
into the Postgres audit_trail table. The latest audit record is also
cached in Redis for the /audit/{region} endpoint.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def read_latest_audit(conn, region: str) -> dict | None:
    """
    Read the most recent audit record for a region from Postgres.

    Args:
        conn: psycopg2 connection.
        region: Grid region code.

    Returns:
        Dict with audit fields or None if no records exist.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT region, scored_at, demand_source, weather_source,
                       demand_rows, weather_rows, model_versions,
                       ensemble_weights, feature_count, feature_hash,
                       mape, peak_forecast_mw, scoring_mode, created_at
                FROM audit_trail
                WHERE region = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (region,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            cols = [desc[0] for desc in cur.description]
            record = dict(zip(cols, row, strict=False))
            # Serialize non-JSON-native fields
            for key in ("model_versions", "ensemble_weights", "mape"):
                if isinstance(record.get(key), str):
                    with contextlib.suppress(json.JSONDecodeError, TypeError):
                        record[key] = json.loads(record[key])
            if record.get("created_at"):
                record["created_at"] = record["created_at"].isoformat()
            return record
    except Exception as e:
        logger.warning("Failed to read audit for %s: %s", region, e)
        return None


def get_data_freshness(conn) -> list[dict]:
    """
    Check data freshness across all sources.

    Returns a list of dicts with source name, last update time, status.
    """
    sources = []
    now = datetime.now(UTC)

    checks = [
        ("raw_demand", "EIA Demand", 3600),
        ("raw_weather", "Open-Meteo Weather", 3600),
        ("forecasts", "Forecast Pipeline", 3600),
        ("audit_trail", "Audit Trail", 7200),
    ]

    for table, label, max_age_seconds in checks:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT MAX(created_at) FROM {table}"  # noqa: S608
                )
                row = cur.fetchone()
                last_update = row[0] if row else None

            if last_update is None:
                status = "no_data"
                age_seconds = None
            else:
                if last_update.tzinfo is None:
                    last_update = last_update.replace(tzinfo=UTC)
                age_seconds = (now - last_update).total_seconds()
                status = "fresh" if age_seconds < max_age_seconds else "stale"

            sources.append(
                {
                    "source": label,
                    "table": table,
                    "last_update": last_update.isoformat() if last_update else None,
                    "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
                    "status": status,
                }
            )
        except Exception as e:
            sources.append(
                {
                    "source": label,
                    "table": table,
                    "last_update": None,
                    "age_seconds": None,
                    "status": "error",
                    "error": str(e),
                }
            )

    return sources
