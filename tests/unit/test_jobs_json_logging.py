"""The jobs CLI must configure structured logging before it dispatches.

Why this test exists: ``observability.configure_logging()`` was called only by
``app.py`` (the web tier). ``jobs/__main__.py`` never called it, so the jobs ran
on structlog's default ``ConsoleRenderer`` and every job log reached Cloud
Logging as ``textPayload``. ``jsonPayload.event`` therefore never existed, and
BOTH log-based alert policies matched nothing and could never fire:

* ``docs/monitoring/scoring_runtime_creep_alert.json`` (#171)
* ``docs/monitoring/scoring_partial_failure_alert.json`` (#267)

Those are the alerts built to catch the 2026-06-01 timeout and the #267 partial
failure. They were dead. Delete the ``configure_logging()`` call and these tests
fail.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import jobs.__main__ as jobs_main


class TestJobsCliConfiguresLogging:
    """NB: ``_ENTRYPOINTS`` captures direct references to ``_scoring`` /
    ``_training`` at import time, so ``patch.object(jobs_main, "_scoring")``
    does NOT intercept dispatch — the dict still holds the original function
    and ``main()`` runs the real job. Patch the dict itself."""

    def test_configure_logging_called_before_dispatch(self):
        """The call must happen, and must precede the job body — a job that
        logs before configuration emits unstructured text."""
        order: list[str] = []

        with (
            patch.object(
                jobs_main, "configure_logging", side_effect=lambda: order.append("configure")
            ),
            patch.dict(
                jobs_main._ENTRYPOINTS,
                {"scoring": lambda: (order.append("dispatch"), 0)[1]},
                clear=False,
            ),
        ):
            code = jobs_main.main(["scoring"])

        assert code == 0
        assert order == ["configure", "dispatch"]

    def test_configure_logging_called_even_for_unknown_job(self):
        """Usage/exit paths log too, so configuration can't hide behind a
        successful dispatch."""
        with patch.object(jobs_main, "configure_logging") as mock_cfg:
            code = jobs_main.main(["nonsense"])
        assert code == 2
        mock_cfg.assert_called_once()

    def test_configure_logging_called_with_no_args(self):
        with patch.object(jobs_main, "configure_logging") as mock_cfg:
            code = jobs_main.main([])
        assert code == 2
        mock_cfg.assert_called_once()

    @pytest.mark.parametrize("job", ["scoring", "training"])
    def test_both_jobs_configure(self, job):
        with (
            patch.object(jobs_main, "configure_logging") as mock_cfg,
            patch.dict(jobs_main._ENTRYPOINTS, {job: lambda: 0}, clear=False),
        ):
            assert jobs_main.main([job]) == 0
        mock_cfg.assert_called_once()


class TestProductionEmitsJsonPayload:
    """The end-to-end property the alert policies depend on: in the container
    (DASH_DEBUG=false, per Dockerfile) a structlog event must serialize to JSON
    with a top-level ``event`` key — that is what ``jsonPayload.event`` binds to.
    """

    def test_event_renders_as_json_with_event_key(self, capsys):
        import structlog

        from observability import configure_logging

        try:
            configure_logging(json_output=True)
            structlog.get_logger().error("reconcile_divergence", region="BPAT", delta=11.1)
            out = capsys.readouterr().out.strip().splitlines()[-1]
            payload = json.loads(out)  # must be JSON at all
            # The exact fields the alert filter and the triage email rely on.
            assert payload["event"] == "reconcile_divergence"
            assert payload["region"] == "BPAT"
            assert payload["delta"] == 11.1
        finally:
            # Don't leak JSON config into other tests' captured output.
            configure_logging(json_output=False)
