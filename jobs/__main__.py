"""
CLI dispatcher for the GridPulse scheduled jobs.

Usage:
    python -m jobs scoring
    python -m jobs training

Invoked directly by Cloud Run Jobs. Returns the underlying job's exit code
unchanged so Cloud Scheduler retries only on hard failures.
"""

from __future__ import annotations

import sys
from typing import Callable

import structlog

log = structlog.get_logger()


def _scoring() -> int:
    from jobs.scoring_job import run

    return run()


def _training() -> int:
    from jobs.training_job import run

    return run()


_ENTRYPOINTS: dict[str, Callable[[], int]] = {
    "scoring": _scoring,
    "training": _training,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m jobs {scoring|training}", file=sys.stderr)
        return 2

    cmd = args[0].lower()
    entry = _ENTRYPOINTS.get(cmd)
    if entry is None:
        print(
            f"unknown job '{cmd}'. valid options: {', '.join(sorted(_ENTRYPOINTS))}",
            file=sys.stderr,
        )
        return 2

    log.info("job_cli_start", job=cmd)
    try:
        code = entry()
    except Exception:
        log.exception("job_cli_unhandled_error", job=cmd)
        return 1
    log.info("job_cli_exit", job=cmd, code=code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
