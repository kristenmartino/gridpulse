"""
Pipeline observability — step-level timing and logging.

Wraps v1's PipelineLogger pattern and writes completed pipeline
runs to the Postgres pipeline_logs table.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PipelineLogger:
    """
    Tracks pipeline steps with timing and context.

    Usage:
        pl = PipelineLogger("wattcast_scoring", region="ERCOT")
        pl.step("fetch_weather", rows=2160)
        pl.step("engineer_features", features=43)
        summary = pl.done()
    """

    def __init__(self, pipeline_name: str, **context):
        self.pipeline_name = pipeline_name
        self.context = context
        self.steps: list[dict] = []
        self._start_time = time.monotonic()
        self._step_start: float | None = None

    def step(self, name: str, **details) -> PipelineLogger:
        """Record a completed pipeline step with optional details."""
        now = time.monotonic()
        elapsed_ms = round((now - (self._step_start or self._start_time)) * 1000, 1)
        self._step_start = now

        entry = {
            "step": name,
            "elapsed_ms": elapsed_ms,
            **details,
        }
        self.steps.append(entry)
        logger.info(
            "pipeline_step",
            extra={
                "pipeline": self.pipeline_name,
                "step": name,
                "elapsed_ms": elapsed_ms,
                **self.context,
                **details,
            },
        )
        return self

    def done(self) -> dict:
        """Finalize and return the pipeline run summary."""
        total_ms = round((time.monotonic() - self._start_time) * 1000, 1)
        summary = {
            "pipeline_name": self.pipeline_name,
            "total_ms": total_ms,
            "steps": self.steps,
            "context": self.context,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "pipeline_complete",
            extra={
                "pipeline": self.pipeline_name,
                "total_ms": total_ms,
                "step_count": len(self.steps),
                **self.context,
            },
        )
        return summary

    def persist(self, conn) -> None:
        """Write the pipeline log to Postgres."""
        summary = self.done() if not self.steps else None
        if summary is None:
            summary = {
                "pipeline_name": self.pipeline_name,
                "total_ms": round((time.monotonic() - self._start_time) * 1000, 1),
                "steps": self.steps,
                "context": self.context,
            }
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pipeline_logs
                        (pipeline_name, region, scored_at, steps, total_ms)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        self.pipeline_name,
                        self.context.get("region"),
                        self.context.get("scored_at"),
                        json.dumps(self.steps),
                        summary["total_ms"],
                    ),
                )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to persist pipeline log: %s", e)
