"""
Freshness Checker — Prevents wasted work by checking data freshness.

Compares the latest timestamp in raw data tables against the last-checked
timestamp in the data_freshness table. If no new data has arrived since
the last successful ingest, the scoring pipeline can skip redundant work.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Maps logical source names to their database tables and timestamp columns
SOURCE_CONFIG = {
    "demand": {"table": "raw_demand", "ts_col": "timestamp"},
    "weather": {"table": "raw_weather", "ts_col": "timestamp"},
}


class FreshnessChecker:
    """
    Checks whether new data has arrived since the last pipeline run.

    Uses the data_freshness table to track the latest timestamp seen
    for each data source. Scoring can be skipped if both demand and
    weather sources report no new data.
    """

    def __init__(self, conn):
        self.conn = conn

    def has_new_data(self, source: str) -> bool:
        """
        Check if a source has new data since the last check.

        Args:
            source: Source name ("demand" or "weather").

        Returns:
            True if new data exists, False otherwise.
        """
        config = SOURCE_CONFIG.get(source)
        if not config:
            logger.warning("Unknown source: %s", source)
            return True  # Assume new data if source is unknown

        table = config["table"]
        ts_col = config["ts_col"]

        try:
            with self.conn.cursor() as cur:
                # Get the latest timestamp in the raw data table
                cur.execute(
                    f"SELECT MAX({ts_col}) FROM {table}"  # noqa: S608
                )
                row = cur.fetchone()
                current_max = row[0] if row else None

                if current_max is None:
                    return False  # No data at all

                # Get the last-checked timestamp from data_freshness
                cur.execute(
                    "SELECT last_timestamp FROM data_freshness WHERE source = %s",
                    (source,),
                )
                row = cur.fetchone()
                last_checked = row[0] if row else None

                if last_checked is None:
                    return True  # Never checked before — treat as new

                # Ensure both are timezone-aware for comparison
                if current_max.tzinfo is None:
                    current_max = current_max.replace(tzinfo=UTC)
                if last_checked.tzinfo is None:
                    last_checked = last_checked.replace(tzinfo=UTC)

                return current_max > last_checked

        except Exception as e:
            logger.warning("Freshness check failed for %s: %s", source, e)
            return True  # On error, assume new data to avoid skipping

    def record_check(self, source: str, last_timestamp: datetime | None = None):
        """
        Record that a source has been checked up to a given timestamp.

        Args:
            source: Source name ("demand" or "weather").
            last_timestamp: The latest timestamp processed. If None,
                queries the source table for MAX(timestamp).
        """
        if last_timestamp is None:
            config = SOURCE_CONFIG.get(source)
            if config:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            f"SELECT MAX({config['ts_col']}) FROM {config['table']}"  # noqa: S608
                        )
                        row = cur.fetchone()
                        last_timestamp = row[0] if row else None
                except Exception as e:
                    logger.warning("Failed to query max timestamp for %s: %s", source, e)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_freshness (source, last_timestamp, last_checked_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (source)
                    DO UPDATE SET last_timestamp = EXCLUDED.last_timestamp,
                                  last_checked_at = NOW()
                    """,
                    (source, last_timestamp),
                )
            self.conn.commit()
        except Exception as e:
            logger.warning("Failed to record freshness for %s: %s", source, e)

    def should_score(self) -> bool:
        """
        Check if either demand or weather has new data.

        Returns True if scoring should proceed, False if both sources
        are unchanged since last check.
        """
        demand_new = self.has_new_data("demand")
        weather_new = self.has_new_data("weather")

        if not demand_new and not weather_new:
            logger.info("No new data since last check — skipping scoring")
            return False

        logger.info(
            "Freshness check: demand_new=%s, weather_new=%s — proceeding",
            demand_new,
            weather_new,
        )
        return True

    def get_all_status(self) -> list[dict]:
        """
        Return freshness status for all tracked sources.

        Used by the /data-freshness API endpoint.
        """
        results = []
        now = datetime.now(UTC)

        for source, config in SOURCE_CONFIG.items():
            try:
                with self.conn.cursor() as cur:
                    # Get current max from source table
                    cur.execute(
                        f"SELECT MAX({config['ts_col']}), COUNT(*) FROM {config['table']}"  # noqa: S608
                    )
                    row = cur.fetchone()
                    current_max = row[0] if row else None
                    row_count = row[1] if row else 0

                    # Get last check from freshness table
                    cur.execute(
                        "SELECT last_timestamp, last_checked_at FROM data_freshness WHERE source = %s",
                        (source,),
                    )
                    fresh_row = cur.fetchone()

                age_seconds = None
                if current_max is not None:
                    if current_max.tzinfo is None:
                        current_max = current_max.replace(tzinfo=UTC)
                    age_seconds = round((now - current_max).total_seconds(), 1)

                results.append(
                    {
                        "source": source,
                        "table": config["table"],
                        "latest_data_at": current_max.isoformat() if current_max else None,
                        "last_checked_at": fresh_row[1].isoformat()
                        if fresh_row and fresh_row[1]
                        else None,
                        "age_seconds": age_seconds,
                        "row_count": row_count,
                        "status": "fresh"
                        if age_seconds is not None and age_seconds < 7200
                        else "stale",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "source": source,
                        "table": config["table"],
                        "status": "error",
                        "error": str(e),
                    }
                )

        return results
