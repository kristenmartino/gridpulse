"""Detect Cloud Monitoring serving something other than the committed policies.

**The gap this closes.** `tests/unit/test_monitoring_policies_applied.py` is the
only automatic check on `docs/monitoring/`, and it compares committed files to a
table of **ids**. It cannot see what GCP is actually serving. Applying a policy —
or a documentation edit to one — is a manual `gcloud` step outside CI, so the
repo and the console drift the moment somebody merges and forgets to apply.

This is the third level of the same failure. `#267`: landing the JSON is not
landing the alert — `scoring_partial_failure_alert.json` sat committed-and-inert
for a week. `#544`: an alert whose event nothing emits is armed, valid, and
silent. And now `#553`: **landing the edit is not landing the runbook**.
`benchmark_coverage_at_risk_alert.json` shipped at 4035 characters against a
4000-character cap, so it was un-appliable from the moment it merged and the
console kept serving the previous runbook for four days. Every workflow was
green throughout, correctly — the id was never stale.

A unit test now fails the build on any runbook over the cap. That closes one
reason committed and applied can diverge. The likelier one — edited, merged,
never applied — needs to ask GCP, which is why this is a scheduled workflow
rather than another unit test.

**Why the enabled/validity half matters as much as the documentation half.** An
over-length `PATCH ?updateMask=documentation` on a log-match policy returns
**HTTP 200**, reports `validity code 13`, and flips `enabled` to **false** — an
edit intended to improve a runbook disarms the alert instead. That happened
twice in one session on 2026-08-18, and both times it was caught only because
somebody happened to be looking. In a policy listing a disabled alert is
indistinguishable from a healthy one except for a single boolean.

**This script never mutates.** It reads and compares, which is also the only
honest way to check this API: a Monitoring mutation returns 200 on failure, so
a write's own response is not evidence that the write landed. Everything here is
a GET.

**One `gcloud monitoring policies list --format=json` answers all of it.** That
is the GA command group, present in the base Cloud SDK that `setup-gcloud`
installs; `list` returns each policy's full `documentation.content`, `enabled`,
`validity` and `notificationChannels`, so the whole comparison is one call and
no per-policy `describe`. Reach for the GA group deliberately: `gcloud alpha
monitoring policies describe` needs the alpha component, which is absent on a
fresh runner and prompts interactively to install — fatal in CI, and it is
exactly what blocked this check being run by hand on 2026-08-18.

It needs `roles/monitoring.viewer` on the workflow's service account, weaker
than the `run.admin` it already holds.

Exit codes are the alert channel — a non-zero exit fails the scheduled workflow,
which notifies the repo owner:

* ``0`` — applied matches committed (warnings may still be printed)
* ``1`` — DIVERGED: what GCP serves is not what the repo declares
* ``2`` — the check itself could not reach a verdict

``2`` is deliberately not silent, for the reason `check_deploy_divergence.py`
records: a check that cannot run is not protecting anything, and treating that
as a pass is how an alert goes quiet exactly when it should fire.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITORING_DIR = REPO_ROOT / "docs" / "monitoring"
README = MONITORING_DIR / "README.md"

PROJECT = "nextera-portfolio"

#: Cloud Monitoring's hard cap on ``documentation.content``, enforced by the API.
#: Mirrors ``_DOC_CHAR_CAP`` in tests/unit/test_monitoring_policies_applied.py,
#: which fails the build on anything over it.
DOC_CHAR_CAP = 4000

#: Warn once a runbook is within this many characters of the cap, so the ceiling
#: is visible before it is hit. Deliberately a WARNING and deliberately not in
#: the unit test: a hard failure at 3800 would assert a limit the API does not
#: enforce, and the standing rule for this directory is to assert the
#: enforcement, not the declaration.
DOC_HEADROOM_WARN = 200

#: How much of a documentation diff to print. Enough to identify which edit is
#: missing; not so much that the runbook is re-printed in the log twice.
MAX_DIFF_LINES = 24


@dataclass
class Finding:
    """One thing wrong, or one thing worth knowing."""

    severity: str  # "fail" | "warn"
    subject: str  # policy filename, or a bare id when there is no file
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    unresolved: str | None = None  # set when the check could not run at all

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def exit_code(self) -> int:
        if self.unresolved:
            return 2
        return 1 if self.failures else 0


def applied_policy_ids(readme_text: str) -> dict[str, str]:
    """``{policy_filename: policy_id}`` from the README applied-policies table.

    The filename is what makes the table machine-checkable — prose labels drift
    (the uptime row reads "(alert)"), filenames don't. This is the same table
    ``_applied_rows()`` in tests/unit/test_monitoring_policies_applied.py reads,
    and a test asserts the two agree on which files appear; matching on
    ``displayName`` instead would key the check on prose.

    Rows for non-policy resources (`uptimeCheckConfigs/`, `budgets/`) carry no
    ``alertPolicies/`` id and are skipped.
    """
    rows: dict[str, str] = {}
    for line in readme_text.splitlines():
        if not line.startswith("|") or "alertPolicies/" not in line:
            continue
        name = re.search(r"`([\w./-]+_alert\.json)`", line)
        pid = re.search(r"`alertPolicies/(\d+)`", line)
        if name and pid:
            rows[name.group(1)] = pid.group(1)
    return rows


def committed_documentation(paths: list[Path]) -> dict[str, str]:
    """``{filename: documentation.content}`` for every committed policy file."""
    out: dict[str, str] = {}
    for path in paths:
        try:
            doc = json.loads(path.read_text()).get("documentation") or {}
        except (json.JSONDecodeError, OSError):
            continue
        out[path.name] = doc.get("content", "")
    return out


def _doc_of(policy: dict) -> str:
    return (policy.get("documentation") or {}).get("content", "")


def _diff(committed: str, applied: str) -> str:
    """A short unified diff, committed-side first, truncated to stay readable."""
    lines = list(
        difflib.unified_diff(
            committed.splitlines(),
            applied.splitlines(),
            fromfile="committed (repo)",
            tofile="applied (GCP)",
            lineterm="",
            n=1,
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        omitted = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES] + [f"... ({omitted} more diff lines)"]
    return "\n".join(f"      {line}" for line in lines)


def evaluate(
    applied: dict[str, str],
    live: dict[str, dict],
    committed: dict[str, str],
) -> Report:
    """Compare the README's applied table against what Monitoring is serving.

    Args:
        applied: ``{filename: policy_id}`` parsed from the README table.
        live: ``{policy_id: policy_json}`` — every alert policy in the project.
        committed: ``{filename: documentation.content}`` from the repo.

    Returns:
        A Report whose ``exit_code`` is the alert.
    """
    report = Report()

    if not applied:
        report.unresolved = (
            "no applied-policy rows parsed out of docs/monitoring/README.md — the "
            "table format changed, or the file moved. This check is watching "
            "nothing until that is fixed."
        )
        return report

    for filename, policy_id in sorted(applied.items()):
        policy = live.get(policy_id)
        if policy is None:
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    f"the applied table points at alertPolicies/{policy_id}, which "
                    f"does not exist in this project. Either it was deleted (the "
                    f"event it filters on is now emitted into a void) or the table "
                    f"row carries a stale id.",
                )
            )
            continue

        report.checked += 1

        if policy.get("enabled") is not True:
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    f"alertPolicies/{policy_id} is DISABLED (enabled="
                    f"{policy.get('enabled')!r}). It will not notify. A failed "
                    f"documentation PATCH disarms a log-match policy exactly this "
                    f"way, returning HTTP 200 while doing it (#553).",
                )
            )

        if "validity" in policy:
            validity = policy["validity"]
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    f"alertPolicies/{policy_id} carries a validity error: "
                    f"code={validity.get('code')} {validity.get('message', '')!r}. "
                    f"On a log-match policy this is frequently mis-reported as a "
                    f"filter recompilation failure when the real cause is an "
                    f"over-length documentation body.",
                )
            )

        if not policy.get("notificationChannels"):
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    f"alertPolicies/{policy_id} has no notificationChannels — it is "
                    f"valid, enabled, and notifies nobody. Creating from a committed "
                    f"file alone does this, because the files carry no channel.",
                )
            )

        want = committed.get(filename)
        if want is None:
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    "the applied table names this file but it is not in "
                    "docs/monitoring/. Renamed without updating the table?",
                )
            )
            continue

        got = _doc_of(policy)
        if got != want:
            report.findings.append(
                Finding(
                    "fail",
                    filename,
                    f"the runbook GCP serves is NOT the committed one "
                    f"(applied {len(got)} chars, committed {len(want)} chars). An "
                    f"on-call reader following the console is reading something the "
                    f"repo does not say.\n{_diff(want, got)}",
                )
            )

        headroom = DOC_CHAR_CAP - len(want)
        if 0 <= headroom < DOC_HEADROOM_WARN:
            report.findings.append(
                Finding(
                    "warn",
                    filename,
                    f"committed runbook is {len(want)} chars — only {headroom} under "
                    f"the {DOC_CHAR_CAP} cap. Move design rationale into "
                    f"docs/monitoring/README.md before the next edit puts it over; "
                    f"over the cap it becomes un-appliable and the PATCH that fails "
                    f"also disables the alert.",
                )
            )

    tabled_ids = set(applied.values())
    for policy_id, policy in sorted(live.items()):
        if policy_id in tabled_ids:
            continue
        report.findings.append(
            Finding(
                "fail",
                f"alertPolicies/{policy_id}",
                f"exists in GCP ({policy.get('displayName', '<no name>')!r}) with no "
                f"row in the README applied-policies table. Either it was created by "
                f"hand and never written down, or it is a leftover — an untracked "
                f"policy is one nobody maintains.",
            )
        )

    return report


def fetch_policies(project: str) -> dict[str, dict] | None:
    """``{policy_id: policy_json}`` for every alert policy, or None if unreadable.

    One `list` call rather than a `describe` per row, so a policy present in GCP
    but absent from the table is visible too — that direction is half the check
    and a per-row lookup cannot see it.
    """
    try:
        out = subprocess.run(
            [
                "gcloud",
                "monitoring",
                "policies",
                "list",
                f"--project={project}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        policies = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(policies, list):
        return None
    return {p["name"].rsplit("/", 1)[-1]: p for p in policies if "name" in p}


def _remediation() -> str:
    """What to run, in the README's own idiom — a divergence report that does not
    say how to close it just moves the puzzle."""
    return (
        "\nTo close a documentation divergence, PATCH the committed runbook onto the\n"
        "policy. `gcloud monitoring policies update --fields` cannot do this — it\n"
        "accepts only `disabled` and `notificationChannels`, and a full\n"
        "`--policy-from-file` would wipe the notification channel, which the committed\n"
        "files do not carry:\n\n"
        "  POLICY=<id>\n"
        "  FILE=docs/monitoring/<policy>_alert.json\n"
        '  curl -s -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" \\\n'
        '    -H "Content-Type: application/json" \\\n'
        "    -d \"$(python3 -c 'import json,sys; "
        'print(json.dumps({"documentation": json.load(open(sys.argv[1]))["documentation"]}))\' "$FILE")" \\\n'
        f'    "https://monitoring.googleapis.com/v3/projects/{PROJECT}/alertPolicies/'
        '$POLICY?updateMask=documentation"\n\n'
        "Then verify by re-running THIS check. Do not read the PATCH response as\n"
        "confirmation — it returns HTTP 200 on failure, which is the whole reason this\n"
        "script exists:\n\n"
        "  python3 scripts/check_monitoring_divergence.py\n\n"
        'If the PATCH reports `validity code 13` ("Recompilation of log match condition\n'
        "failed\"), check the runbook's length before believing it: over 4000 characters\n"
        "it is un-appliable, the error names the wrong cause, and the failed PATCH also\n"
        "sets enabled=false. A second PATCH under the cap clears both (#553).\n"
        "Full procedure: docs/monitoring/README.md\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare applied alert policies to the repo.")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    applied = applied_policy_ids(README.read_text())
    committed = committed_documentation(sorted(MONITORING_DIR.glob("*_alert.json")))

    live = fetch_policies(args.project)
    if live is None:
        report = Report(
            unresolved=(
                f"could not list alert policies in {args.project} — `gcloud monitoring "
                f"policies list` failed. Most likely the service account is missing "
                f"roles/monitoring.viewer. The check did not run, which is NOT "
                f"evidence that nothing has diverged."
            )
        )
    else:
        report = evaluate(applied, live, committed)

    print(f"applied-table rows: {len(applied)}   policies compared: {report.checked}")

    for finding in report.warnings:
        print(f"\nWARN  {finding.subject}\n      {finding.message}")
    for finding in report.failures:
        print(f"\nFAIL  {finding.subject}\n      {finding.message}")

    if report.unresolved:
        print(f"\nUNKNOWN: {report.unresolved}")
    elif report.failures:
        print(f"\nDIVERGED: {len(report.failures)} problem(s) above.")
        print(_remediation())
    else:
        suffix = f" ({len(report.warnings)} warning(s))" if report.warnings else ""
        print(f"\nOK: applied policies match the repo{suffix}.")

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
