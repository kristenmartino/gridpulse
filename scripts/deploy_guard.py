"""Decide whether a `workflow_run`-triggered production deploy should proceed.

Replaces the inline tip comparison in `deploy-prod.yml`. The gate it enforces is
one notch weaker, and that notch is the whole point.

**What was there.** "Deploy only if `DEPLOY_SHA` is still main's tip." That
correctly stopped an older commit shipping over a newer one on 2026-08-05, when
`cancel-in-progress` cancelled by *arrival* time and production ran the wrong
commit for an hour. But it skips on a **prediction** — the notice literally said
*"a newer deploy covers it"* — and a prediction can be wrong:

* **The newer commit is red.** Its deploy never runs, because this workflow is
  gated on CI success. The skipped commit never gets another turn and production
  sits on the pre-merge image indefinitely, every workflow green.
* **Merges outrun the pipeline.** 2026-08-11: three commits in 12 minutes, two of
  three deploys skipped, and #488 reached production 22 minutes after merge via a
  later commit's deploy rather than its own.

**What it is now.** Skip only when a strictly-newer commit on main has
**already passed CI** — because that is exactly the condition under which "a
newer deploy covers it" is a fact rather than a hope. A newer commit whose CI is
still running is not yet covering anything, so this deploys instead of gambling
on it.

**Why that is still safe against the 2026-08-05 inversion.** Replay it: CI
`319b9de3` (newer) finished 06:50:57, CI `345d284a` (older) finished 06:51:27.
When the older commit's deploy asks this question, the newer commit has already
passed CI, so it still skips — the outcome that mattered. The rule only
*loosens* the case where nothing newer has been validated yet, which is precisely
the case the tip comparison got wrong.

Ordering is safe in every interleaving because `concurrency: cancel-in-progress:
false` serialises deploys: whichever runs last wins, and a commit only skips when
something newer is genuinely queued to run after it.

**The price, stated honestly.** During a burst of merges this can deploy two or
three times where the old rule deployed once — each an image build, push, and
three Cloud Run updates. That is the cost of never stranding a green commit. It
is bounded by the Artifact Registry cleanup policy, and Actions minutes are free
on this public repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

#: How far back to look for commits newer than the deploy target. A deploy that
#: has fallen more than this many commits behind is not a race any more.
HISTORY_DEPTH = 50


@dataclass
class Decision:
    proceed: bool
    reason: str
    #: "notice" for routine outcomes, "warning" for the ones a human should read.
    level: str = "notice"


def decide(
    deploy_sha: str,
    tip: str,
    newer_commits: list[str],
    ci_conclusion: dict[str, str] | None,
) -> Decision:
    """Should ``deploy_sha`` be deployed?

    Pure, so every branch is testable without a network or a git tree.

    Args:
        deploy_sha: The commit whose CI triggered this deploy.
        tip: Current tip of main.
        newer_commits: Commits on main strictly newer than ``deploy_sha``.
            Empty when it is the tip. An empty list combined with a differing
            ``tip`` means ``deploy_sha`` is not on main at all.
        ci_conclusion: SHA -> CI conclusion ("success", "failure", ...).
            A SHA absent from this map has no finished CI run. ``None`` means
            the lookup itself failed, which is NOT the same as "nothing is
            green" — see the refusal below.

    Returns:
        A :class:`Decision`.
    """
    if not deploy_sha:
        return Decision(False, "DEPLOY_SHA is empty — refusing to deploy", "warning")

    if deploy_sha == tip:
        return Decision(True, f"{deploy_sha[:9]} is main's tip")

    # Not the tip and nothing newer on main means it is not on main — a rewritten
    # or force-pushed history. Deploying it would ship code main does not have.
    if not newer_commits:
        return Decision(
            False,
            f"{deploy_sha[:9]} is not main's tip ({tip[:9]}) and is not an "
            f"ancestor of it — main's history was rewritten. Refusing to deploy "
            f"a commit that is not on main.",
            "warning",
        )

    # A failed lookup must not read as "nothing newer is green". If it did, a
    # `gh` outage would make every superseded commit deploy — reinstating the
    # 2026-08-05 older-over-newer bug at exactly the moment there is no way to
    # detect it. Of the two failure directions this is the one with no backstop:
    # a stranded commit is caught within the hour by the divergence check
    # (scripts/check_deploy_divergence.py), whereas shipping an older commit
    # over a newer one ran wrong code in production for an hour and was noticed
    # by hand. So refuse, loudly, and let detection handle the consequence.
    if ci_conclusion is None:
        return Decision(
            False,
            f"Refusing {deploy_sha[:9]}: main is at {tip[:9]} and the CI status "
            f"of the {len(newer_commits)} newer commit(s) could not be read, so "
            f"whether one supersedes this is unknown. The hourly divergence "
            f"check will surface it if this leaves production behind.",
            "warning",
        )

    # THE FIX. A newer commit only "covers" this one if it is actually going to
    # deploy, and it will only deploy if its CI passed. Anything else — running,
    # failed, cancelled, never started — means skipping here strands a green
    # commit that nothing will pick up.
    covered_by = [c for c in newer_commits if ci_conclusion.get(c) == "success"]
    if covered_by:
        return Decision(
            False,
            f"Skipping {deploy_sha[:9]} — {covered_by[0][:9]} is newer and has "
            f"already passed CI, so its deploy supersedes this one.",
        )

    unfinished = [c for c in newer_commits if c not in ci_conclusion]
    detail = (
        f"{len(newer_commits)} newer commit(s) exist but none has passed CI "
        f"({len(unfinished)} still unfinished)"
    )
    return Decision(
        True,
        f"Deploying {deploy_sha[:9]} even though main is at {tip[:9]}: {detail}. "
        f"Skipping here would strand it — a newer commit that never goes green "
        f"never deploys, and nothing else would ship this one.",
    )


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"  ! {' '.join(cmd[:3])}... failed ({type(e).__name__})", file=sys.stderr)
        return None


def newer_commits_on_main(deploy_sha: str) -> list[str]:
    """Commits on main strictly newer than ``deploy_sha``, oldest first.

    Empty when ``deploy_sha`` is the tip *or* is not an ancestor of it; the
    caller distinguishes those by comparing against the tip directly.
    """
    if _run(["git", "merge-base", "--is-ancestor", deploy_sha, "HEAD"]) is None:
        return []
    out = _run(["git", "rev-list", "--reverse", f"{deploy_sha}..HEAD"])
    return out.splitlines() if out else []


def ci_conclusions(limit: int = HISTORY_DEPTH * 2) -> dict[str, str] | None:
    """SHA -> CI conclusion for recent pushes to main. Only FINISHED runs.

    ``None`` on lookup failure — the caller must NOT conflate that with an empty
    result, which would mean "nothing newer is green" and let everything deploy.
    """
    raw = _run(
        [
            "gh",
            "run",
            "list",
            "--workflow=CI",
            "--branch=main",
            "--event=push",
            f"--limit={limit}",
            "--json",
            "headSha,conclusion,status",
        ]
    )
    if not raw:
        return None
    try:
        runs = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return {
        r["headSha"]: r["conclusion"]
        for r in runs
        if r.get("status") == "completed" and r.get("conclusion")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploy-sha", default=os.environ.get("DEPLOY_SHA", ""))
    args = ap.parse_args()

    deploy_sha = args.deploy_sha.strip()
    tip = _run(["git", "rev-parse", "HEAD"]) or ""
    newer = newer_commits_on_main(deploy_sha) if deploy_sha and tip else []
    conclusions: dict[str, str] | None = ci_conclusions() if newer else {}

    d = decide(deploy_sha, tip, newer, conclusions)

    print(f"deploy sha: {deploy_sha or '<empty>'}")
    print(f"main tip:   {tip or '<unknown>'}")
    if newer:
        print(f"newer on main ({len(newer)}):")
        for sha in newer:
            state = (
                "<lookup failed>" if conclusions is None else conclusions.get(sha, "<unfinished>")
            )
            print(f"  {sha[:9]}  ci={state}")

    print(f"::{d.level}::{d.reason}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"proceed={'true' if d.proceed else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
