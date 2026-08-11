"""Detect production running something other than main's newest validated commit.

**The gap this closes.** `deploy-prod.yml`'s staleness guard skips a deploy whose
commit is no longer main's tip, on the reasoning that *"a newer deploy covers
it"*. That reasoning is an assumption about the future, and it fails in at least
three ways — all of which leave every workflow green:

1. **The next commit is red.** Merge A (green), merge B minutes later, B's CI
   fails. A's deploy already skipped because the tip moved to B; B's deploy never
   runs because the workflow is gated on CI success. Production sits on the
   pre-A image indefinitely and nothing says so.
2. **Merges outrun the pipeline.** Observed live 2026-08-11: four commits landed
   in 14 minutes, and each deploy found the tip had moved past it before its
   guard ran. Every run reported success; none deployed.
3. **A deploy half-lands.** 2026-08-04 (#418): `gcloud run jobs deploy` rejected
   a flag the service step accepted, so the *service* advanced while **both jobs
   froze on a 12-hour-old image**. The workflow failed loudly, and the surfaces
   still disagreed for half a day.

Case 3 is why this compares each surface separately rather than asking "did the
last deploy succeed". A partial deploy is the failure mode most likely to be
misread as a healthy one, because the thing you would naturally check — the
workflow's conclusion — is about the *run*, not about what is actually serving.

**Why detection rather than a smarter guard.** Making the guard compare against
"newest green commit" instead of the tip would fix case 1 at the source, and it
would add logic to the exact file that has already produced two production
incidents (the `cancel-in-progress` order inversion, and the sticky-flag deploy
that reported success while changing nothing). Detection is strictly weaker and
strictly safer: it does not prevent the skip, but it makes *every* way a deploy
can silently not happen visible, including ways not enumerated here.

Exit codes are the alert channel — a non-zero exit fails the scheduled workflow,
which notifies the repo owner:

* ``0`` — deployed matches, or a deploy is plausibly still in flight
* ``1`` — DIVERGED: production is not running what it should be
* ``2`` — the check itself could not reach a verdict

``2`` is deliberately not silent. A check that cannot run is a check that is not
protecting anything, and the failure mode worth engineering against is the one
`docs/monitoring/backtest_recompute_alert.json` documents: an alert that goes
quiet at exactly the moment it should fire.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

#: Every Cloud Run surface `deploy-prod.yml` updates, and how to read the image
#: it is currently running. The nesting differs between services and jobs —
#: a job's container sits one `template` deeper.
SURFACES: dict[str, tuple[str, str]] = {
    "gridpulse": ("services", "spec.template.spec.containers[0].image"),
    "gridpulse-scoring-job": ("jobs", "spec.template.spec.template.spec.containers[0].image"),
    "gridpulse-training-job": ("jobs", "spec.template.spec.template.spec.containers[0].image"),
}

#: How long after a commit's CI turns green before divergence is real rather
#: than a deploy still working. Build + push + three deploys + the smoke check
#: runs several minutes, and under rapid merges a commit can be legitimately
#: superseded more than once before one sticks. Generous on purpose: the
#: condition being detected is PERMANENT divergence, so a late alert costs
#: nothing and a premature one trains the reader to ignore it.
DEFAULT_GRACE_MINUTES = 45

PROJECT = "nextera-portfolio"
REGION = "us-east1"


@dataclass
class Verdict:
    status: str  # "ok" | "in_flight" | "diverged" | "unknown"
    reason: str
    stale: dict[str, str | None] = field(default_factory=dict)
    age_minutes: float | None = None

    @property
    def exit_code(self) -> int:
        return {"ok": 0, "in_flight": 0, "diverged": 1, "unknown": 2}[self.status]


def evaluate(
    expected: str | None,
    expected_ci_completed_at: datetime | None,
    deployed: dict[str, str | None],
    now: datetime,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
) -> Verdict:
    """Decide whether production has diverged from main's newest green commit.

    Pure — every input is passed in, so the decision is testable without a
    network. ``deployed`` maps surface name to the image tag it is running, or
    to ``None`` when that surface could not be read.

    Args:
        expected: SHA that should be live; ``None`` if it could not be resolved.
        expected_ci_completed_at: When that commit's CI went green.
        deployed: Surface name -> running image tag (or ``None`` on read failure).
        now: Current time, injected so the grace window is testable.
        grace_minutes: How long a mismatch is attributed to an in-flight deploy.

    Returns:
        A :class:`Verdict` whose ``exit_code`` is what the CLI returns.
    """
    if not expected:
        return Verdict("unknown", "could not resolve the newest green commit on main")

    unreadable = {s: v for s, v in deployed.items() if v is None}
    if unreadable:
        return Verdict(
            "unknown",
            f"could not read the deployed image for: {', '.join(sorted(unreadable))}",
            stale=unreadable,
        )
    if not deployed:
        return Verdict("unknown", "no surfaces were checked")

    stale = {s: v for s, v in deployed.items() if v != expected}
    if not stale:
        return Verdict("ok", f"all {len(deployed)} surfaces are running {expected[:9]}")

    # A mismatch during the deploy window is expected, not a finding. Without a
    # completion time there is no way to tell in-flight from stuck, and calling
    # it stuck would fire on every normal deploy — so treat it as unresolvable
    # rather than guessing in the alarming direction.
    if expected_ci_completed_at is None:
        return Verdict(
            "unknown",
            "surfaces disagree with the expected commit, but its CI completion "
            "time is unknown, so in-flight cannot be distinguished from stuck",
            stale=stale,
        )

    age = (now - expected_ci_completed_at).total_seconds() / 60.0
    if age < grace_minutes:
        return Verdict(
            "in_flight",
            f"{len(stale)} surface(s) behind, but {expected[:9]} only went green "
            f"{age:.0f}m ago (grace {grace_minutes}m)",
            stale=stale,
            age_minutes=age,
        )

    # Surfaces disagreeing with EACH OTHER is the #418 shape and worth naming
    # separately — it means a deploy half-landed rather than never started.
    distinct = {v for v in deployed.values()}
    shape = "partially deployed" if len(distinct) > 1 else "not deployed"
    return Verdict(
        "diverged",
        f"{expected[:9]} went green {age:.0f}m ago and is still {shape}",
        stale=stale,
        age_minutes=age,
    )


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"  ! command failed: {' '.join(cmd[:3])}... ({type(e).__name__})", file=sys.stderr)
        return None
    return out.stdout.strip()


def newest_green_commit(limit: int = 30) -> tuple[str | None, datetime | None]:
    """Newest commit on main whose CI passed, resolved in COMMIT order.

    Deliberately not "the most recently completed successful CI run". CI
    completion order has nothing to do with commit order — a newer commit's run
    can finish first — and confusing the two is precisely the bug that shipped an
    older commit to production on 2026-08-05 and cost `cancel-in-progress`. So
    walk main's history newest-first and take the first commit that has a green
    run, rather than trusting the run list's ordering.
    """
    # `origin/main` exists locally but NOT after actions/checkout, which leaves a
    # detached HEAD with no remote-tracking ref. Try in order so the same script
    # runs in both places rather than silently returning no history on the runner
    # — which would degrade to exit 2 forever and look like a broken check.
    history: list[str] = []
    for ref in ("origin/main", "main", "HEAD"):
        log = _run(["git", "log", "--format=%H", f"-n{limit}", ref])
        if log:
            history = log.splitlines()
            break
    if not history:
        return None, None

    raw = _run(
        [
            "gh",
            "run",
            "list",
            "--workflow=CI",
            "--branch=main",
            "--event=push",
            f"--limit={limit * 2}",
            "--json",
            "headSha,conclusion,updatedAt",
        ]
    )
    if raw is None:
        return None, None
    try:
        runs = json.loads(raw)
    except json.JSONDecodeError:
        return None, None

    green = {r["headSha"]: r["updatedAt"] for r in runs if r.get("conclusion") == "success"}
    for sha in history:
        if sha in green:
            try:
                ts = datetime.fromisoformat(green[sha].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = None
            return sha, ts
    return None, None


def deployed_images() -> dict[str, str | None]:
    """Image tag currently running on each surface, or None where unreadable."""
    out: dict[str, str | None] = {}
    for name, (kind, fmt) in SURFACES.items():
        val = _run(
            [
                "gcloud",
                "run",
                kind,
                "describe",
                name,
                "--region",
                REGION,
                "--project",
                PROJECT,
                "--format",
                f"value({fmt})",
            ]
        )
        out[name] = val.rsplit(":", 1)[-1] if val else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grace-minutes", type=int, default=DEFAULT_GRACE_MINUTES)
    args = ap.parse_args()

    expected, ci_at = newest_green_commit()
    deployed = deployed_images()
    verdict = evaluate(expected, ci_at, deployed, datetime.now(UTC), args.grace_minutes)

    print(f"expected (newest green on main): {expected or '<unresolved>'}")
    for surface, image in sorted(deployed.items()):
        mark = "ok" if image == expected else "STALE"
        print(f"  {surface:<24} {image or '<unreadable>'}  [{mark}]")
    print(f"\n{verdict.status.upper()}: {verdict.reason}")

    if verdict.status == "diverged":
        print(
            "\nProduction is not running main's newest validated commit, and no "
            "deploy is going to fix it on its own.\n"
            "Most likely: every deploy for these commits was skipped by the "
            "staleness guard in deploy-prod.yml because main moved on before each "
            "one ran.\n"
            "Confirm with:  gh run list --workflow=deploy-prod.yml --limit 5\n"
            "A run whose 'Build & Deploy' job is 'skipped' did NOT deploy, even "
            "though the run reports success.\n"
            "To resolve, re-run the deploy for the expected commit, or push a "
            "no-op commit so a fresh CI -> deploy cycle runs while it is the tip."
        )
    return verdict.exit_code


if __name__ == "__main__":
    sys.exit(main())
