"""
Observability: structured logging, request monitoring, performance tracking.

Configures structlog for Cloud Run JSON logging, adds request timing
middleware, and provides performance monitoring utilities.

Per AC-8.8: Structured logging (structlog) outputs JSON to stdout
for Cloud Run log aggregation.
"""

import functools
import os
import time
from collections.abc import Callable
from typing import Any

import structlog


def configure_logging(json_output: bool | None = None) -> None:
    """
    Configure structlog for the application.

    In production (Cloud Run): JSON to stdout for log aggregation.
    In development: human-readable colored output.

    Args:
        json_output: Force JSON output. If None, auto-detect from DASH_DEBUG.
    """
    if json_output is None:
        json_output = os.getenv("DASH_DEBUG", "true").lower() != "true"

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # Production: JSON for Cloud Run / GCP Logging
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: human-readable
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def untrusted_client_ip(xff_header: str | None, remote_addr: str | None) -> str:
    """Resolve the caller IP for allowlist checks, spoof-resistantly.

    ``X-Forwarded-For`` is client-controlled on its LEFT — a caller can prepend
    any IP. The trusted edge (e.g. Cloud Run's load balancer) appends the REAL
    source on the RIGHT, so we take the rightmost hop; a prepended
    ``127.0.0.1`` can't slip past an allowlist that way (2026-07 review P2-52).
    Falls back to ``remote_addr`` (the TCP peer) when there is no XFF.
    """
    hops = [h.strip() for h in (xff_header or "").split(",") if h.strip()]
    if hops:
        return hops[-1]
    return remote_addr or ""


def ip_in_allowlist(ip: str, entries) -> bool:
    """True when ``ip`` matches any allowlist entry — an exact address or a CIDR.

    Entries may be plain addresses (``"1.2.3.4"``, ``"::1"``) for exact match,
    or CIDR networks (``"2600:1700:f890:2740::/64"``, ``"10.0.0.0/8"``) to match
    a whole prefix. The CIDR form is what makes a residential IPv6 usable in an
    allowlist: the host bits rotate (privacy extensions) but the ``/64`` prefix
    is stable, so ``…:2740::/64`` keeps matching across rotations (#253).

    Fail-closed on any parse error — a malformed caller IP or entry never
    matches, so a bad value can't accidentally widen access. Address objects
    (not strings) are compared, so ``"::1"`` and ``"0:0:0:0:0:0:0:1"`` are
    equivalent.
    """
    import ipaddress

    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                if addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif ipaddress.ip_address(entry) == addr:
                return True
        except ValueError:
            continue  # skip a malformed entry rather than fail the whole check
    return False


def add_request_logging(server) -> None:
    """
    Add request logging middleware to the Flask server.

    Logs: method, path, status, duration_ms for every request.
    Excludes: health checks and static assets to reduce noise.
    """
    log = structlog.get_logger()

    @server.before_request
    def start_timer():
        from flask import g

        g.start_time = time.monotonic()

    @server.after_request
    def log_request(response):
        from flask import g, request

        # Skip noisy endpoints
        path = request.path
        if path in ("/health", "/_dash-update-component", "/_dash-dependencies"):
            return response
        if path.startswith("/_dash-component-suites/") or path.startswith("/assets/"):
            return response

        duration_ms = (time.monotonic() - getattr(g, "start_time", time.monotonic())) * 1000

        log.info(
            "http_request",
            method=request.method,
            path=path,
            status=response.status_code,
            duration_ms=round(duration_ms, 1),
        )
        return response


def timed(func: Callable) -> Callable:
    """
    Decorator that logs function execution time.

    Usage:
        @timed
        def expensive_operation():
            ...
    """
    log = structlog.get_logger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.monotonic() - start) * 1000
            log.info(
                "function_completed",
                function=func.__name__,
                duration_ms=round(duration_ms, 1),
            )
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            log.error(
                "function_failed",
                function=func.__name__,
                duration_ms=round(duration_ms, 1),
                error=str(e),
            )
            raise

    return wrapper


class PerformanceTracker:
    """
    Track callback and data pipeline performance metrics.

    Stores rolling metrics in memory (no persistence needed).
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._timings: dict[str, list[float]] = {}

    def record(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(duration_ms)
        # Trim to max entries
        if len(self._timings[name]) > self.max_entries:
            self._timings[name] = self._timings[name][-self.max_entries :]

    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a named operation."""
        timings = self._timings.get(name, [])
        if not timings:
            return {"count": 0, "mean_ms": 0, "p95_ms": 0, "max_ms": 0}

        import numpy as np

        arr = np.array(timings)
        return {
            "count": len(arr),
            "mean_ms": round(float(np.mean(arr)), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
            "max_ms": round(float(np.max(arr)), 1),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {name: self.get_stats(name) for name in self._timings}


# ── Pipeline Transformation Logger (I1) ──────────────────────


class PipelineLogger:
    """
    Step-by-step ETL transformation logger (Backlog I1).

    Tracks each transformation stage with row counts, column diffs,
    duration, and data shape. When a dashboard number looks wrong,
    engineers can trace back through the pipeline.

    Usage:
        pipe = PipelineLogger("demand_etl", region="FPL")
        pipe.step("fetch", rows=168, cols=3, source="eia_api")
        pipe.step("clean", rows=168, cols=3, nulls_filled=4)
        pipe.step("merge", rows=168, cols=20, weather_cols=17)
        pipe.step("features", rows=144, cols=35, dropped_lag=24)
        pipe.done()
    """

    def __init__(self, pipeline_name: str, **context):
        self._name = pipeline_name
        self._context = context
        self._steps: list[dict[str, Any]] = []
        self._start = time.monotonic()
        self._step_start = self._start
        self._log = structlog.get_logger()

    def step(self, name: str, **details) -> "PipelineLogger":
        """Record a pipeline transformation step."""
        now = time.monotonic()
        duration_ms = round((now - self._step_start) * 1000, 1)
        self._step_start = now

        entry = {"step": name, "duration_ms": duration_ms, **details}
        self._steps.append(entry)

        # Filter out non-serializable objects from log output
        safe_details = {
            k: v for k, v in details.items() if isinstance(v, (str, int, float, bool, type(None)))
        }
        self._log.info(
            "pipeline_step",
            pipeline=self._name,
            step=name,
            step_num=len(self._steps),
            duration_ms=duration_ms,
            **safe_details,
            **self._context,
        )
        return self

    def done(self) -> dict[str, Any]:
        """Finalize the pipeline log and return summary."""
        total_ms = round((time.monotonic() - self._start) * 1000, 1)
        summary = {
            "pipeline": self._name,
            "total_steps": len(self._steps),
            "total_ms": total_ms,
            "steps": self._steps,
            **self._context,
        }

        self._log.info(
            "pipeline_complete",
            pipeline=self._name,
            total_steps=len(self._steps),
            total_ms=total_ms,
            **self._context,
        )

        # Record in perf tracker
        perf.record(f"pipeline.{self._name}", total_ms)
        return summary

    @property
    def steps(self) -> list[dict[str, Any]]:
        """Access recorded steps."""
        return list(self._steps)


# Module-level singletons
perf = PerformanceTracker()
