"""Tests for the deploy-divergence check.

The check exists because `deploy-prod.yml`'s staleness guard skips a superseded
deploy on the reasoning that "a newer deploy covers it" — an assumption about
the future that fails whenever the newer commit is red, whenever merges outrun
the pipeline, and whenever a deploy half-lands. In all three cases every
workflow reports success.

What matters in these tests is the *shape of the wrongness*, in both directions:

* **Crying wolf during a normal deploy** would make the alert worthless within a
  week, because a mismatch is the expected state for several minutes after every
  single merge.
* **Going quiet when it cannot answer** is the failure this project has already
  been bitten by — see the note in
  `docs/monitoring/backtest_recompute_alert.json` about a metric that stops
  being emitted exactly when the thing it watches breaks. So "I don't know"
  exits non-zero and is never folded into "fine".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.check_deploy_divergence import DEFAULT_GRACE_MINUTES, evaluate

NOW = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)
GREEN = "d2893c3598fdc93b14ed96c6887ffe5da14ca267"
OLD = "38eb438770342ac85ebc0d96a2bf6f9e1294dfa4"
SURFACES = ["gridpulse", "gridpulse-scoring-job", "gridpulse-training-job"]


def _all(sha: str | None) -> dict[str, str | None]:
    return dict.fromkeys(SURFACES, sha)


def _ago(minutes: float) -> datetime:
    return NOW - timedelta(minutes=minutes)


class TestConverged:
    def test_every_surface_on_the_expected_commit_is_ok(self):
        v = evaluate(GREEN, _ago(120), _all(GREEN), NOW)
        assert v.status == "ok"
        assert v.exit_code == 0
        assert not v.stale

    def test_still_ok_long_after_the_deploy(self):
        """Age is irrelevant when nothing is behind — a commit that went green
        weeks ago and is correctly deployed must never alert."""
        assert evaluate(GREEN, _ago(60 * 24 * 30), _all(GREEN), NOW).status == "ok"


class TestInFlightIsNotAFinding:
    """A mismatch is the NORMAL state for several minutes after every merge.
    Alerting on it would fire on every deploy and be muted immediately."""

    def test_fresh_mismatch_is_in_flight(self):
        v = evaluate(GREEN, _ago(3), _all(OLD), NOW)
        assert v.status == "in_flight"
        assert v.exit_code == 0

    def test_just_inside_the_grace_window(self):
        v = evaluate(GREEN, _ago(DEFAULT_GRACE_MINUTES - 1), _all(OLD), NOW)
        assert v.status == "in_flight"

    def test_just_outside_the_grace_window_is_diverged(self):
        v = evaluate(GREEN, _ago(DEFAULT_GRACE_MINUTES + 1), _all(OLD), NOW)
        assert v.status == "diverged"
        assert v.exit_code == 1

    def test_grace_is_configurable(self):
        assert evaluate(GREEN, _ago(20), _all(OLD), NOW, grace_minutes=10).status == "diverged"
        assert evaluate(GREEN, _ago(20), _all(OLD), NOW, grace_minutes=60).status == "in_flight"


class TestDiverged:
    def test_nothing_deployed_names_every_stale_surface(self):
        v = evaluate(GREEN, _ago(180), _all(OLD), NOW)
        assert v.status == "diverged"
        assert set(v.stale) == set(SURFACES)
        assert "not deployed" in v.reason

    def test_partial_deploy_is_called_out_separately(self):
        """THE #418 SHAPE. `gcloud run jobs deploy` rejected a flag the service
        step accepted, so the service advanced while both jobs froze on a
        12-hour-old image. The distinguishing signal is that the surfaces
        disagree with EACH OTHER, not merely with the expected commit — that
        means a deploy half-landed rather than never starting, which is a
        different investigation."""
        mixed = {
            "gridpulse": GREEN,
            "gridpulse-scoring-job": OLD,
            "gridpulse-training-job": OLD,
        }
        v = evaluate(GREEN, _ago(180), mixed, NOW)
        assert v.status == "diverged"
        assert "partially deployed" in v.reason
        assert set(v.stale) == {"gridpulse-scoring-job", "gridpulse-training-job"}
        assert "gridpulse" not in v.stale

    def test_a_single_stale_surface_still_diverges(self):
        one = dict.fromkeys(SURFACES, GREEN) | {"gridpulse-training-job": OLD}
        v = evaluate(GREEN, _ago(180), one, NOW)
        assert v.status == "diverged"
        assert set(v.stale) == {"gridpulse-training-job"}

    def test_age_is_reported_for_the_runbook(self):
        v = evaluate(GREEN, _ago(200), _all(OLD), NOW)
        assert v.age_minutes == pytest.approx(200, abs=0.5)


class TestUnknownIsNeverSilent:
    """Exit 2, not 0. A check that cannot reach a verdict is not protecting
    anything, and folding that into 'fine' is how an alert goes quiet at exactly
    the moment it should fire."""

    def test_unresolved_expected_commit(self):
        v = evaluate(None, _ago(120), _all(OLD), NOW)
        assert v.status == "unknown"
        assert v.exit_code == 2

    def test_an_unreadable_surface_is_unknown_not_ok(self):
        """Critically NOT 'ok'. If gcloud fails and the other two surfaces happen
        to match, treating the unreadable one as fine would report a healthy
        deploy while a surface is unaccounted for."""
        partial: dict[str, str | None] = dict.fromkeys(SURFACES, GREEN)
        partial["gridpulse-training-job"] = None
        v = evaluate(GREEN, _ago(120), partial, NOW)
        assert v.status == "unknown"
        assert v.exit_code == 2
        assert "gridpulse-training-job" in v.stale

    def test_unreadable_wins_over_a_real_divergence(self):
        """When some surfaces are unreadable the picture is incomplete, so the
        honest answer is 'cannot tell' rather than a confident divergence
        computed from a subset."""
        partial: dict[str, str | None] = {"gridpulse": OLD, "gridpulse-scoring-job": None}
        assert evaluate(GREEN, _ago(999), partial, NOW).status == "unknown"

    def test_missing_ci_timestamp_cannot_distinguish_in_flight_from_stuck(self):
        v = evaluate(GREEN, None, _all(OLD), NOW)
        assert v.status == "unknown"
        assert v.exit_code == 2

    def test_missing_ci_timestamp_is_still_ok_when_converged(self):
        """The timestamp only matters for grading a mismatch. With nothing
        behind there is no question to answer, so a missing one must not
        manufacture an unknown."""
        assert evaluate(GREEN, None, _all(GREEN), NOW).status == "ok"

    def test_no_surfaces_checked(self):
        assert evaluate(GREEN, _ago(120), {}, NOW).status == "unknown"


class TestExitCodeMapping:
    @pytest.mark.parametrize(
        ("status", "code"),
        [("ok", 0), ("in_flight", 0), ("diverged", 1), ("unknown", 2)],
    )
    def test_codes(self, status, code):
        from scripts.check_deploy_divergence import Verdict

        assert Verdict(status, "").exit_code == code
