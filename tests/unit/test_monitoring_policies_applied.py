"""Every committed alert policy must be applied — or explicitly declared not to be.

Applying a Cloud Monitoring policy is a **manual `gcloud` step outside CI**, so a
policy JSON can be committed, reviewed, merged, and look completely done while
never existing in GCP. That is not hypothetical: `scoring_partial_failure_alert.json`
(#267) did exactly that — `jobs/scoring_job.py:471` emitted the event into a void
for a week, and it was only caught while planning something unrelated.

**Landing the JSON is not landing the alert.**

This is the same "nothing watches the watcher" failure as #131 / #220 / #296 /
#304 — applied to the alerting config itself. The test can't reach GCP, so it
checks the next best invariant: a committed policy is either

1. recorded in the README's applied-policies table with a live id, or
2. listed in ``_KNOWN_UNAPPLIED`` with a reason.

Either way somebody made an explicit decision. Silent drift fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

MONITORING_DIR = Path(__file__).resolve().parents[2] / "docs" / "monitoring"
README = MONITORING_DIR / "README.md"

#: Policies deliberately NOT applied yet, with the reason. To apply one: run the
#: gcloud recipe in the README, add its row (with live id) to the
#: applied-policies table, then delete it from this dict.
_KNOWN_UNAPPLIED: dict[str, str] = {}

#: Live-id line format in the README table, e.g. `alertPolicies/5813319064717268577`.
_LIVE_ID = re.compile(r"`alertPolicies/\d+`")


def _policy_files() -> list[Path]:
    return sorted(MONITORING_DIR.glob("*_alert.json"))


def _readme() -> str:
    return README.read_text()


def _applied_rows() -> dict[str, str]:
    """``{policy_filename: live_id_cell}`` from the README table.

    Matches rows that name a policy file AND carry an alertPolicies/<id>. The
    filename is what makes the table machine-checkable — prose labels drift
    (the uptime row reads "(alert)"), filenames don't.
    """
    rows: dict[str, str] = {}
    for line in _readme().splitlines():
        if not line.startswith("|") or "alertPolicies/" not in line:
            continue
        m = re.search(r"`([\w./-]+_alert\.json)`", line)
        if m:
            rows[m.group(1)] = line
    return rows


class TestPolicyFilesAreDeclared:
    def test_policy_files_exist(self):
        """Guard the guard: if the glob stops matching, every assertion below
        passes vacuously."""
        assert len(_policy_files()) >= 5

    @pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
    def test_each_policy_is_applied_or_declared_unapplied(self, path: Path):
        applied = _applied_rows()
        name = path.name
        if name in _KNOWN_UNAPPLIED:
            assert name not in applied, (
                f"{name} is listed in _KNOWN_UNAPPLIED but ALSO appears in the "
                f"README applied table. If it's now applied, delete it from "
                f"_KNOWN_UNAPPLIED."
            )
            return
        assert name in applied, (
            f"{name} is committed but has no live id in the README applied-policies "
            f"table — so it may not exist in GCP at all, and the event it filters on "
            f"is being emitted into a void (this is what happened to "
            f"scoring_partial_failure, #267).\n\n"
            f"Either apply it (see docs/monitoring/README.md) and add a row with its "
            f"live id, or add it to _KNOWN_UNAPPLIED in this file with a reason."
        )

    @pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
    def test_applied_rows_carry_a_real_live_id(self, path: Path):
        """A row without a real `alertPolicies/<digits>` id is a placeholder."""
        row = _applied_rows().get(path.name)
        if row is None:
            return  # covered by the declaration test above
        assert _LIVE_ID.search(row), f"{path.name} row has no `alertPolicies/<id>`: {row}"


class TestKnownUnappliedIsHonest:
    def test_entries_reference_real_files(self):
        """A stale allowlist entry would silently exempt nothing — or worse, mask
        a renamed policy."""
        names = {p.name for p in _policy_files()}
        for declared in _KNOWN_UNAPPLIED:
            assert declared in names, (
                f"_KNOWN_UNAPPLIED names {declared!r}, which no longer exists. Remove the entry."
            )

    def test_entries_have_a_reason(self):
        for name, reason in _KNOWN_UNAPPLIED.items():
            assert reason.strip(), f"{name} declared unapplied with no reason"

    def test_readme_flags_every_unapplied_policy(self):
        """The human-facing doc must warn about what the allowlist exempts, or
        the exemption is invisible to anyone not reading tests."""
        readme = _readme()
        for name in _KNOWN_UNAPPLIED:
            stem = name.removesuffix("_alert.json")
            assert stem in readme, (
                f"{name} is declared unapplied but {stem!r} is not mentioned in "
                f"docs/monitoring/README.md — readers won't know the alert is dead."
            )


#: Source trees a job/web event can legitimately be emitted from. Tests and
#: scripts are excluded deliberately — an event only a test emits is exactly
#: the inert case this guards.
_EMITTER_DIRS = ("jobs", "models", "data", "components", "simulation", "personas")

#: `jsonPayload.event="<name>"` inside a conditionMatchedLog filter.
_FILTER_EVENT = re.compile(r'jsonPayload\.event\s*=\s*"([^"]+)"')


def _filter_events(policy: dict) -> list[str]:
    """Every event name a policy's log-based conditions key on."""
    out: list[str] = []
    for cond in policy.get("conditions", []):
        matched = cond.get("conditionMatchedLog")
        if not matched:
            continue
        out.extend(_FILTER_EVENT.findall(matched.get("filter", "")))
    return out


def _emitted_events() -> set[str]:
    """Every event name the source can produce, across both emit idioms.

    Deliberately a *lexical* scan, not an import: the point is to compare the
    policy's filter against what the source says, without executing anything
    or depending on a module's import side effects.

    Two idioms, and missing the second is how this check would quietly stop
    working:

    1. ``log.warning("scoring_runtime_creep", ...)`` — the literal first arg.
    2. ``{"event": "benchmark_scoreability_drop", ...}`` built by a helper and
       emitted later as ``log.error(alert.pop("event"), **alert)``
       (`jobs/scoring_job.py`). The first argument is a *variable*, so idiom 1's
       pattern cannot see it — the first cut of this test failed on exactly
       these two events and looked, briefly, like a real inert alert.

    A dict key named ``event`` that is never logged would be a false negative
    here. That direction is safe: it can only make the test pass when it should
    have failed, never fail a policy that is fine.
    """
    root = Path(__file__).resolve().parents[2]
    direct = re.compile(r"log\.(?:info|warning|error|debug|exception)\(\s*[\"']([a-z0-9_]+)[\"']")
    as_field = re.compile(r"[\"']event[\"']\s*:\s*[\"']([a-z0-9_]+)[\"']")
    found: set[str] = set()
    for d in _EMITTER_DIRS:
        for py in (root / d).rglob("*.py"):
            src = py.read_text(encoding="utf-8")
            found.update(direct.findall(src))
            found.update(as_field.findall(src))
    return found


class TestLogBasedPoliciesAreWellFormed:
    """conditionMatchedLog policies have two footguns that fail silently."""

    @pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
    def test_log_based_policies_have_notification_rate_limit(self, path: Path):
        policy = json.loads(path.read_text())
        log_based = any("conditionMatchedLog" in c for c in policy.get("conditions", []))
        if not log_based:
            return
        rate_limit = (policy.get("alertStrategy") or {}).get("notificationRateLimit")
        assert rate_limit, (
            f"{path.name} uses conditionMatchedLog but has no "
            f"alertStrategy.notificationRateLimit — GCP rejects the policy."
        )

    @pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
    def test_every_log_filter_names_an_event_the_code_emits(self, path: Path):
        """An alert watching an event nothing emits is armed, valid, and silent.

        The module docstring records the inverse — #267 emitted
        ``scoring_partial_failure`` into a void for a week because the policy
        was never applied. This is the mirror: the policy is applied, correct,
        enabled and routed, and no code ever produces the event it waits for.
        Both failures look identical from outside — a green dashboard — and
        neither announces itself, because an alert that never fires is
        indistinguishable from an alert with nothing to report.

        The realistic path in is a rename. ``benchmark_scoreability_drop`` is a
        string in two places that nothing links: a filter in
        ``docs/monitoring/*.json`` and a ``log.warning(...)`` call in
        ``models/``. Renaming the emitter is an ordinary refactor that no test,
        type, or import would object to — and it would silently disarm the
        alarm.

        **Scope, stated honestly.** This proves the event name exists in the
        source. It does NOT prove the policy exists in GCP, is enabled, routes
        to a live channel, still matches this file, or that the emitting branch
        is reachable or deployed. Those need credentials; see
        ``docs/monitoring/README.md`` for what is and is not covered.
        """
        events = _filter_events(json.loads(path.read_text()))
        if not events:
            return
        emitted = _emitted_events()
        for event in events:
            assert event in emitted, (
                f"{path.name} alerts on jsonPayload.event={event!r}, but no "
                f"structlog call in {'/, '.join(_EMITTER_DIRS)}/ emits it. "
                f"Either the emitter was renamed (update the filter) or the "
                f"alert was written against an event that never shipped — "
                f"the policy is armed and permanently silent either way."
            )

    @pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
    def test_log_based_filters_use_jsonpayload_event(self, path: Path):
        """Pins the #306 contract: jobs emit JSON so `jsonPayload.event` exists.
        A filter on textPayload would work today but silently rot if logging is
        ever reconfigured — and the reverse is what made these inert for weeks.
        """
        policy = json.loads(path.read_text())
        for cond in policy.get("conditions", []):
            matched = cond.get("conditionMatchedLog")
            if not matched:
                continue
            filt = matched.get("filter", "")
            assert "jsonPayload.event=" in filt, (
                f"{path.name} log filter does not key on jsonPayload.event — "
                f"see the inertness note in docs/monitoring/README.md: {filt}"
            )
