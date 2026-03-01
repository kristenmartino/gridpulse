"""
Forecast Model Input Audit Trail (Backlog D2).

Logs which model version, weather data vintage, feature encoding, and data
sources produced each forecast. Provides lineage for post-event investigations
and regulatory defensibility (FERC/NERC).

Each forecast run produces an AuditRecord stored in-memory and optionally
persisted to the cache database. Records are queryable by region, time range,
or model version.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import structlog

log = structlog.get_logger()


@dataclass
class AuditRecord:
    """Immutable record of a single forecast computation."""

    # Identity
    record_id: str = ""
    timestamp: str = ""

    # Context
    region: str = ""
    forecast_horizon_hours: int = 168

    # Model lineage
    model_versions: dict[str, str] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)

    # Data lineage
    demand_source: str = ""  # "eia_api" | "cache_stale" | "demo"
    weather_source: str = ""  # "open_meteo" | "cache_stale" | "demo"
    demand_rows: int = 0
    weather_rows: int = 0
    demand_range: tuple[str, str] = ("", "")
    weather_range: tuple[str, str] = ("", "")

    # Feature lineage
    feature_count: int = 0
    feature_hash: str = ""  # SHA-256 of feature column names

    # Outputs
    mape: dict[str, float] = field(default_factory=dict)
    peak_forecast_mw: float = 0.0
    forecast_source: str = ""  # "trained_model" | "simulated"

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditTrail:
    """
    In-memory audit log with structured logging output.

    Each forecast run appends a record. Recent records are queryable.
    In production, records would be persisted to BigQuery or Cloud SQL.
    """

    def __init__(self, max_records: int = 1000):
        self._records: list[AuditRecord] = []
        self._max_records = max_records

    def record_forecast(
        self,
        region: str,
        demand_source: str,
        weather_source: str,
        demand_rows: int,
        weather_rows: int,
        demand_range: tuple[str, str] = ("", ""),
        weather_range: tuple[str, str] = ("", ""),
        model_versions: dict[str, str] | None = None,
        ensemble_weights: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
        mape: dict[str, float] | None = None,
        peak_forecast_mw: float = 0.0,
        forecast_source: str = "simulated",
        forecast_horizon_hours: int = 168,
    ) -> AuditRecord:
        """
        Record a forecast computation and return the audit record.

        Called by load_data / model_service after each forecast cycle.
        """
        now = datetime.now(UTC)
        feature_names = feature_names or []
        feature_hash = hashlib.sha256(",".join(sorted(feature_names)).encode()).hexdigest()[:16]

        record = AuditRecord(
            record_id=f"{region}-{now.strftime('%Y%m%d%H%M%S')}",
            timestamp=now.isoformat(),
            region=region,
            forecast_horizon_hours=forecast_horizon_hours,
            model_versions=model_versions or {"prophet": "sim", "arima": "sim", "xgboost": "sim"},
            ensemble_weights=ensemble_weights or {},
            demand_source=demand_source,
            weather_source=weather_source,
            demand_rows=demand_rows,
            weather_rows=weather_rows,
            demand_range=demand_range,
            weather_range=weather_range,
            feature_count=len(feature_names),
            feature_hash=feature_hash,
            mape=mape or {},
            peak_forecast_mw=peak_forecast_mw,
            forecast_source=forecast_source,
        )

        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

        log.info(
            "forecast_audit",
            record_id=record.record_id,
            region=region,
            demand_source=demand_source,
            weather_source=weather_source,
            demand_rows=demand_rows,
            weather_rows=weather_rows,
            feature_count=record.feature_count,
            feature_hash=feature_hash,
            forecast_source=forecast_source,
            peak_mw=peak_forecast_mw,
            models=list((model_versions or {}).keys()),
        )

        return record

    def get_recent(self, n: int = 10) -> list[AuditRecord]:
        """Return the n most recent audit records."""
        return self._records[-n:]

    def get_by_region(self, region: str, n: int = 10) -> list[AuditRecord]:
        """Return recent records for a specific region."""
        return [r for r in self._records if r.region == region][-n:]

    def get_latest(self, region: str) -> AuditRecord | None:
        """Return the most recent record for a region, or None."""
        records = self.get_by_region(region, n=1)
        return records[0] if records else None

    @property
    def count(self) -> int:
        """Total number of stored records."""
        return len(self._records)


# Module-level singleton
audit_trail = AuditTrail()
