"""
CLI dispatcher for the GridPulse scheduled jobs.

Usage:
    python -m jobs scoring
    python -m jobs training [--force]

Invoked directly by Cloud Run Jobs. Returns the underlying job's exit code
unchanged so Cloud Scheduler retries only on hard failures.

``main()`` calls ``observability.configure_logging()`` before dispatching —
see the note there. Without it the jobs emit console-formatted text and every
log-based alert policy silently matches nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import structlog

from observability import configure_logging

log = structlog.get_logger()


def _scoring() -> int:
    from jobs.scoring_job import run

    return run()


def _training(force: bool = False) -> int:
    from jobs.training_job import run

    return run(force=force)


#: Entrypoints taking the ``--force`` flag. ``force`` disables the data-hash
#: resume AND the cached SARIMAX order search — see
#: ``jobs.training_job._read_cached_arima_order``, which otherwise never
#: re-runs. Scoring has no equivalent, so the flag is rejected for it rather
#: than silently ignored.
_FORCEABLE = {"training"}

_ENTRYPOINTS: dict[str, Callable[..., int]] = {
    "scoring": _scoring,
    "training": _training,
}


def main(argv: list[str] | None = None) -> int:
    # Configure structlog BEFORE anything logs. The web tier does this at
    # app.py import time; the jobs never did, so they fell back to structlog's
    # default ConsoleRenderer and every job log landed in Cloud Logging as
    # ``textPayload`` — meaning ``jsonPayload.event`` never existed and BOTH
    # log-based alert policies (scoring_runtime_creep #171, scoring_partial_
    # failure #267) matched nothing and could never fire. The Dockerfile
    # already sets DASH_DEBUG=false, so this emits JSON in Cloud Run and stays
    # human-readable locally. See docs/monitoring/README.md.
    configure_logging()

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m jobs {scoring|training} [--force]", file=sys.stderr)
        return 2

    cmd = args[0].lower()
    entry = _ENTRYPOINTS.get(cmd)
    if entry is None:
        print(
            f"unknown job '{cmd}'. valid options: {', '.join(sorted(_ENTRYPOINTS))}",
            file=sys.stderr,
        )
        return 2

    flags = {a.lower() for a in args[1:]}
    force = "--force" in flags
    unknown = flags - {"--force"}
    if unknown:
        print(f"unknown flag(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if force and cmd not in _FORCEABLE:
        # Rejected, not ignored: silently dropping it would let an operator
        # believe they had forced a retrain when they had not.
        print(f"--force is not supported for '{cmd}'", file=sys.stderr)
        return 2

    log.info("job_cli_start", job=cmd, force=force)
    try:
        code = entry(force=force) if cmd in _FORCEABLE else entry()
    except Exception:
        log.exception("job_cli_unhandled_error", job=cmd)
        return 1
    log.info("job_cli_exit", job=cmd, code=code)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
