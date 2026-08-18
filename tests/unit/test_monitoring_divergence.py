"""Tests for the applied-vs-committed monitoring check.

The check exists because the only other guard on `docs/monitoring/` compares
committed files to a table of **ids**, and an id is the one thing that stayed
correct in every divergence this repo has actually had. `#267`: the JSON was
committed and never applied. `#553`: the runbook was edited, merged, and could
not be applied at all (4035 characters against a 4000 cap), so the console
served the previous copy for four days with every workflow green. On its first
live run this check found a third: `scoring_runtime_creep` had been serving the
pre-#389 runbook since 2026-07-08 while the repo carried the rewrite from
2026-08-04 — edited, merged, never applied, and nothing said so for two weeks.

So the assertions here are about *state and content*, never identity.

Two shapes matter as much as the happy path:

* **A warning must not fail the build.** The headroom check fires at 200
  characters under the cap, and that number is invented — the API enforces
  4000, not 3800. Asserting an invented limit as hard failure is the mistake
  this directory has a standing rule against, so headroom warns and exits 0.
* **"I cannot tell" is never folded into "fine".** An unreadable table or an
  unreachable API exits 2, for the same reason `test_deploy_divergence.py`
  records: a check that cannot run is not protecting anything.
"""

from __future__ import annotations

import pytest

from scripts.check_monitoring_divergence import (
    DOC_CHAR_CAP,
    DOC_HEADROOM_WARN,
    applied_policy_ids,
    evaluate,
)

FILE = "example_alert.json"
PID = "1234567890"
DOC = "**Runbook**\n1. Look at the thing.\n2. Fix the thing."


def policy(**overrides) -> dict:
    """A healthy applied policy; override one field per test."""
    base = {
        "name": f"projects/nextera-portfolio/alertPolicies/{PID}",
        "displayName": "GridPulse — example",
        "enabled": True,
        "notificationChannels": ["projects/nextera-portfolio/notificationChannels/7"],
        "documentation": {"content": DOC},
    }
    base.update(overrides)
    return base


def run(live: dict | None = None, committed: dict | None = None, applied: dict | None = None):
    return evaluate(
        applied if applied is not None else {FILE: PID},
        live if live is not None else {PID: policy()},
        committed if committed is not None else {FILE: DOC},
    )


class TestConverged:
    def test_matching_policy_passes(self):
        report = run()
        assert report.exit_code == 0
        assert report.failures == []
        assert report.checked == 1


class TestDocumentationDrift:
    def test_differing_documentation_fails(self):
        """The #553 / #389 failure: the console serves something the repo doesn't say."""
        report = run(live={PID: policy(documentation={"content": "stale runbook"})})
        assert report.exit_code == 1
        assert len(report.failures) == 1
        assert report.failures[0].subject == FILE

    def test_failure_names_both_directions(self):
        """'They differ' is not actionable — which side has what is the whole point."""
        report = run(live={PID: policy(documentation={"content": "stale"})})
        message = report.failures[0].message
        assert "applied 5 chars" in message
        assert f"committed {len(DOC)} chars" in message
        assert "committed (repo)" in message and "applied (GCP)" in message

    def test_missing_documentation_counts_as_drift(self):
        """A policy with no runbook at all is the worst version of this, not an exemption."""
        report = run(live={PID: policy(documentation={})})
        assert report.exit_code == 1

    def test_whitespace_difference_is_still_drift(self):
        """No normalising. A trailing-newline difference means the apply step did not
        run with this file, and that is the thing being detected."""
        report = run(live={PID: policy(documentation={"content": DOC + "\n"})})
        assert report.exit_code == 1


class TestPolicyState:
    def test_disabled_policy_fails(self):
        """An over-length documentation PATCH disarms a log-match policy while
        returning HTTP 200. In a listing that is one boolean."""
        report = run(live={PID: policy(enabled=False)})
        assert report.exit_code == 1
        assert "DISABLED" in report.failures[0].message

    def test_missing_enabled_field_fails(self):
        """Absent is not the same as true, and defaulting it to true would make the
        check pass on exactly the payload it cannot interpret."""
        live = {PID: policy()}
        del live[PID]["enabled"]
        report = run(live=live)
        assert report.exit_code == 1

    def test_validity_field_fails(self):
        report = run(live={PID: policy(validity={"code": 13, "message": "Recompilation…"})})
        assert report.exit_code == 1
        assert "validity" in report.failures[0].message

    def test_policy_with_no_notification_channels_fails(self):
        """Valid, enabled, and notifies nobody — what creating from a committed file
        alone produces, since the files carry no channel."""
        report = run(live={PID: policy(notificationChannels=[])})
        assert report.exit_code == 1
        assert "notifies nobody" in report.failures[0].message


class TestInventory:
    def test_tabled_policy_missing_from_gcp_fails(self):
        report = run(live={})
        assert report.exit_code == 1
        assert "does not exist in this project" in report.failures[0].message

    def test_untabled_policy_in_gcp_fails(self):
        """The other direction: something applied by hand that nobody wrote down."""
        report = run(live={PID: policy(), "999": policy(displayName="GridPulse — orphan")})
        assert report.exit_code == 1
        assert any(f.subject == "alertPolicies/999" for f in report.failures)

    def test_tabled_file_that_does_not_exist_fails(self):
        report = run(committed={})
        assert report.exit_code == 1
        assert "not in docs/monitoring" in report.failures[0].message

    def test_empty_table_is_unresolved_not_clean(self):
        """A parser that silently stops matching would otherwise report a perfect
        score forever — the failure mode this whole directory exists to prevent."""
        report = run(applied={})
        assert report.exit_code == 2
        assert report.unresolved is not None
        assert report.failures == []


class TestHeadroomIsAWarningOnly:
    def test_runbook_near_the_cap_warns(self):
        near = "x" * (DOC_CHAR_CAP - DOC_HEADROOM_WARN + 1)
        report = run(live={PID: policy(documentation={"content": near})}, committed={FILE: near})
        assert len(report.warnings) == 1
        assert "under the" in report.warnings[0].message

    def test_a_warning_does_not_fail_the_build(self):
        """The 200-character margin is invented; only 4000 is enforced. Asserting the
        declaration as a hard failure is what this directory has a rule against."""
        near = "x" * (DOC_CHAR_CAP - DOC_HEADROOM_WARN + 1)
        report = run(live={PID: policy(documentation={"content": near})}, committed={FILE: near})
        assert report.exit_code == 0
        assert report.failures == []

    def test_comfortable_runbook_does_not_warn(self):
        comfortable = "x" * (DOC_CHAR_CAP - DOC_HEADROOM_WARN - 1)
        report = run(
            live={PID: policy(documentation={"content": comfortable})},
            committed={FILE: comfortable},
        )
        assert report.warnings == []

    def test_over_the_cap_still_warns_rather_than_double_reporting(self):
        """Over the cap is the unit test's job (it fails the build at commit time).
        Here it must not resurface as a second warning about headroom."""
        over = "x" * (DOC_CHAR_CAP + 35)
        report = run(live={PID: policy(documentation={"content": over})}, committed={FILE: over})
        assert report.warnings == []


class TestTableParser:
    def test_parses_filename_and_id(self):
        row = (
            "| scoring-job runtime creep (#171) | `scoring_runtime_creep_alert.json` "
            "| `alertPolicies/5813319064717268577` |"
        )
        assert applied_policy_ids(row) == {
            "scoring_runtime_creep_alert.json": "5813319064717268577"
        }

    @pytest.mark.parametrize(
        "line",
        [
            "| Uptime check config | — | `uptimeCheckConfigs/gridpulse-health-162OIAwsIpE` |",
            "| Monthly budget — $150 | — | `budgets/3363cac4-5a23-46ea-a51f-ddbbadeca827` |",
            "Prose mentioning alertPolicies/123 outside any table.",
        ],
        ids=["uptime-row", "budget-row", "prose"],
    )
    def test_ignores_non_policy_rows(self, line: str):
        assert applied_policy_ids(line) == {}

    def test_agrees_with_the_unit_guard_on_which_files_are_tabled(self):
        """Two parsers read the same table for different questions. If they ever
        disagree about which files appear, one of them is quietly watching a
        subset — so this pins them together rather than letting them drift."""
        from tests.unit.test_monitoring_policies_applied import _applied_rows, _readme

        assert set(applied_policy_ids(_readme())) == set(_applied_rows())

    def test_the_real_table_is_not_empty(self):
        """Guard the guard: every assertion above is vacuous if the live README
        stops parsing."""
        from tests.unit.test_monitoring_policies_applied import _readme

        assert len(applied_policy_ids(_readme())) >= 5
