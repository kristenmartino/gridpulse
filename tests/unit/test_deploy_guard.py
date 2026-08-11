"""Tests for the production deploy guard's skip rule.

The guard decides whether a `workflow_run`-triggered deploy proceeds. It has
gotten this wrong twice in production, in opposite directions, and both
directions are pinned here:

* **Too permissive (2026-08-05).** `cancel-in-progress: true` cancelled by
  arrival time, an older commit shipped over a newer one, and production ran the
  wrong code for an hour with both runs green. The tip comparison fixed it.
* **Too strict (2026-08-11).** The tip comparison skips on the prediction that
  "a newer deploy covers it", which is false when the newer commit is red — its
  deploy never runs, so the skipped commit is stranded forever. Measured: two of
  three deploys skipped in a 12-minute merge burst.

The rule is now "skip only if a strictly-newer commit has ALREADY PASSED CI".
The tests that matter most are the two replays — `test_the_2026_08_05_inversion_
still_skips` (the old bug must not come back) and the CI-failed/CI-unfinished
cases (the new bug must actually be fixed). Everything else is boundary work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.deploy_guard import decide

TIP = "d2893c3598fdc93b14ed96c6887ffe5da14ca267"
MID = "7dffa7fd981ee1deb2e88ddd853377cc0062e599"
OLD = "e78dd5c85aa1f4b2c3d4e5f60718293a4b5c6d7e"


class TestTheTipAlwaysDeploys:
    def test_tip_proceeds(self):
        d = decide(TIP, TIP, [], {})
        assert d.proceed
        assert "main's tip" in d.reason

    def test_tip_proceeds_even_with_stale_ci_data(self):
        """Being the tip is sufficient — no CI lookup should be able to veto it."""
        assert decide(TIP, TIP, [], {TIP: "failure"}).proceed


class TestSupersededByAGreenCommit:
    """The 2026-08-05 protection. These must keep skipping."""

    def test_the_2026_08_05_inversion_still_skips(self):
        """THE REGRESSION THAT MUST NOT COME BACK.

        CI `319b9de3` (newer) finished 06:50:57; CI `345d284a` (older) finished
        06:51:27, so the older commit's deploy ran second and won. Replayed
        under the new rule: when the older commit asks, the newer one has
        already passed CI, so it still skips — which is the outcome that
        mattered. The relaxation must not touch this case.
        """
        newer, older = "319b9de3" + "0" * 32, "345d284a" + "0" * 32
        d = decide(older, newer, [newer], {newer: "success", older: "success"})
        assert not d.proceed
        assert newer[:9] in d.reason

    def test_skips_when_any_newer_commit_is_green(self):
        d = decide(OLD, TIP, [MID, TIP], {MID: "failure", TIP: "success"})
        assert not d.proceed
        assert TIP[:9] in d.reason

    def test_names_the_oldest_green_superseder(self):
        """The one that will deploy first is the useful one to name."""
        d = decide(OLD, TIP, [MID, TIP], {MID: "success", TIP: "success"})
        assert MID[:9] in d.reason

    def test_a_skip_is_a_routine_notice_not_a_warning(self):
        assert decide(OLD, TIP, [TIP], {TIP: "success"}).level == "notice"


class TestTheStrandedCommitFix:
    """The 2026-08-11 bug. These all used to skip and now must deploy."""

    def test_newer_commit_failed_ci_so_deploy_anyway(self):
        """THE CORE BUG. A red commit's deploy never runs — this workflow is
        gated on CI success — so skipping here strands a green commit forever
        while every workflow stays green."""
        d = decide(MID, TIP, [TIP], {TIP: "failure"})
        assert d.proceed
        assert "strand" in d.reason.lower()

    def test_newer_commit_ci_still_running_so_deploy(self):
        """It is not covering anything yet, and it may go red. Deploying is the
        recoverable choice: if it later goes green its deploy supersedes this
        one, costing an extra deploy rather than a stranded commit."""
        assert decide(MID, TIP, [TIP], {}).proceed

    def test_newer_commit_cancelled_so_deploy(self):
        assert decide(MID, TIP, [TIP], {TIP: "cancelled"}).proceed

    def test_the_2026_08_11_burst_deploys_instead_of_stranding(self):
        """Replay: `e78dd5c`'s deploy ran while `7dffa7f` and `d2893c3` were
        both still in CI. Old rule skipped it; there were three commits and only
        the last one ever deployed."""
        d = decide(OLD, TIP, [MID, TIP], {})
        assert d.proceed
        assert "2 newer commit(s)" in d.reason

    def test_mixed_unfinished_and_failed_still_deploys(self):
        assert decide(OLD, TIP, [MID, TIP], {MID: "failure"}).proceed


class TestRefusals:
    """Cases where NOT deploying is right, and loud about it."""

    def test_commit_not_on_main_is_refused_with_a_warning(self):
        """No newer commits AND not the tip means it is not an ancestor of main
        — a rewritten history. Deploying would ship code main does not have."""
        d = decide(OLD, TIP, [], {})
        assert not d.proceed
        assert d.level == "warning"
        assert "rewritten" in d.reason

    def test_empty_deploy_sha_is_refused(self):
        """An unset DEPLOY_SHA must never be read as 'matches nothing, deploy'."""
        d = decide("", TIP, [], {})
        assert not d.proceed
        assert d.level == "warning"

    def test_failed_ci_lookup_refuses_rather_than_deploying(self):
        """`None` is NOT `{}`. If a `gh` outage read as "nothing newer is
        green", every superseded commit would deploy — reinstating the
        2026-08-05 older-over-newer bug at the exact moment nothing can detect
        it. This is the asymmetry that decides the direction: a stranded commit
        is caught within the hour by the divergence check, while shipping an
        older commit over a newer one ran wrong code for an hour and was only
        noticed by hand."""
        d = decide(MID, TIP, [TIP], None)
        assert not d.proceed
        assert d.level == "warning"
        assert "could not be read" in d.reason

    def test_empty_ci_map_is_not_treated_as_a_failure(self):
        """A successful lookup that finds no finished runs genuinely means
        nothing newer is green, so the stranding fix applies."""
        assert decide(MID, TIP, [TIP], {}).proceed

    def test_a_failed_lookup_cannot_veto_the_tip(self):
        """The tip needs no CI cross-check — it is what CI just validated."""
        assert decide(TIP, TIP, [], None).proceed


class TestTheWorkflowStillWiresItUp:
    """Structural guards — the logic being correct is worthless if the workflow
    stops calling it, or if the serialisation it depends on is removed."""

    def _workflow(self) -> str:
        root = Path(__file__).resolve().parents[2]
        return (root / ".github" / "workflows" / "deploy-prod.yml").read_text()

    def test_guard_invokes_the_script(self):
        assert "scripts/deploy_guard.py" in self._workflow(), (
            "the guard has stopped calling deploy_guard.py; the tested logic is "
            "no longer what gates production deploys"
        )

    def test_cancel_in_progress_is_still_false(self):
        """LOAD-BEARING for the new rule. Deploying a superseded commit is only
        safe because `concurrency` SERIALISES deploys, so a newer one queued
        behind it still lands last. Flip this to true and cancellation by
        arrival time returns — the exact 2026-08-05 bug."""
        assert "cancel-in-progress: false" in self._workflow()

    def test_checkout_is_not_shallow(self):
        """fetch-depth: 1 would make every non-tip commit look like it is not an
        ancestor of main, so the guard would refuse every superseded deploy —
        failing closed, but for a bogus reason."""
        assert "fetch-depth: 60" in self._workflow()

    @pytest.mark.parametrize("gate", ["needs.guard.outputs.proceed == 'true'"])
    def test_deploy_job_is_still_gated_on_the_guard(self, gate):
        assert gate in self._workflow()
