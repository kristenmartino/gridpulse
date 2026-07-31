"""Mutation testing over the decision-critical modules.

Coverage answers "did this line run?". It cannot answer "would anything have
noticed if this line were wrong?" — and that is the question worth asking of a
test suite you did not write by hand. Mutation testing answers it by breaking
the code on purpose and checking whether the suite fails.

A **survivor** is a mutant the suite did not catch: a precise, reproducible
statement that a specific line can be broken with CI still green. Survivors are
the output of this script; the score is just a summary of them.

Scope is the seven modules in ``[tool.mutmut] only_mutate`` (pyproject.toml) —
pure logic where a silently wrong number propagates into a published result.
Rationale, measured baselines and the adjudicated survivor ledger live in
``docs/TEST_QUALITY.md``.

Usage::

    python scripts/mutation_test.py                       # all targets
    python scripts/mutation_test.py --module models/skill.py
    python scripts/mutation_test.py --skip-run            # re-report, no re-run
    python scripts/mutation_test.py --json out.json --markdown out.md

This script never edits tracked files. Mutants are written to ``mutants/``,
a throwaway tree mutmut copies the source into; the working copy is untouched.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MUTANTS_DIR = REPO_ROOT / "mutants"

_MUTANT_DEF_RE = re.compile(r"^def\s+x_\w+?__mutmut_\w+\s*\(", re.MULTILINE)

# Mirrors mutmut's own ``status_by_exit_code`` (mutmut/__main__.py). Copied
# rather than imported: importing that module loads mutmut's config as an
# import-time side effect and hard-fails outside the repo root.
_STATUS_BY_EXIT_CODE: dict[int | None, str] = {
    1: "killed",
    3: "killed",  # internal pytest error still means the mutant was noticed
    0: "survived",
    2: "interrupted",
    5: "no tests",
    33: "no tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    37: "caught by type check",
    24: "timeout",
    -24: "timeout",
    152: "timeout",
    255: "timeout",
    -11: "segfault",
    -9: "segfault",
    None: "not checked",
}

# Statuses that count toward the score. "no tests"/"skipped"/"not checked"
# are excluded from BOTH numerator and denominator: they describe the harness,
# not the suite, and folding them in would silently flatter or punish the score.
_SCORED = {"killed", "survived", "timeout", "suspicious", "segfault"}
_KILLED = {"killed", "timeout", "suspicious", "segfault"}


@dataclass
class Survivor:
    """One mutant the suite failed to notice."""

    key: str
    function: str
    module: str
    kind: str  # "logic" | "observability" | "string-literal"
    diff: str


@dataclass
class ModuleResult:
    module: str
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    survivors: list[Survivor] = field(default_factory=list)

    @property
    def scored(self) -> int:
        return sum(n for s, n in self.counts.items() if s in _SCORED)

    @property
    def killed(self) -> int:
        return sum(n for s, n in self.counts.items() if s in _KILLED)

    @property
    def score(self) -> float | None:
        """Killed / scored, as a percentage. ``None`` when nothing was scored."""
        return None if not self.scored else 100.0 * self.killed / self.scored

    @property
    def logic_survivors(self) -> list[Survivor]:
        return [s for s in self.survivors if s.kind == "logic"]

    @property
    def logic_score(self) -> float | None:
        """Score counting only behavioural mutants — the honest denominator.

        mutmut rewrites string constants (``"x"`` -> ``"XXxXX"``) and structlog
        arguments (``log.info(None, ...)``) prolifically. Nothing asserts on
        either, so they survive by construction and drag the raw score down
        without indicating a test gap. This drops them from BOTH sides of the
        ratio. Reported alongside — never instead of — the raw score, because
        deciding what counts as noise is exactly the kind of judgement a
        quality metric should show its working for.
        """
        noise = len(self.survivors) - len(self.logic_survivors)
        denom = self.scored - noise
        return None if denom <= 0 else 100.0 * self.killed / denom


def _load_targets() -> list[str]:
    """The ``only_mutate`` list from pyproject.toml — the single source of scope."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh).get("tool", {}).get("mutmut", {})
    targets = config.get("only_mutate", [])
    if not targets:
        sys.exit("pyproject.toml [tool.mutmut] has no only_mutate — nothing to do.")
    return list(targets)


def _normalise(source: str) -> str:
    """Rename the mutmut wrapper function to a fixed name.

    Every mutant differs from the original in its ``def`` line
    (``x_f__mutmut_orig`` vs ``x_f__mutmut_3``). Left in, that difference makes
    every mutant look like a logic change and puts a spurious hunk at the top
    of every diff.
    """
    return _MUTANT_DEF_RE.sub("def _f(", source, count=1)


def _strip_string_constants(tree: ast.AST) -> ast.AST:
    """Blank every string constant, so two trees compare equal modulo prose."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return tree


class _StripObservability(ast.NodeTransformer):
    """Blank the payload of ``log.*(...)`` calls and ``raise`` expressions.

    mutmut rewrites structlog arguments enthusiastically — ``log.info(None,
    ...)``, ``reason=None``, whole kwargs dropped — and every one of those
    survives, because no test asserts on log call arguments. Left uncategorised
    they bury the handful of survivors that are real.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in {"log", "logger", "LOG"}
        ):
            node.args = []
            node.keywords = []
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            node.exc.args = []
            node.exc.keywords = []
        return node


def _classify(original: str, mutated: str) -> str:
    """Bucket a survivor by what it actually changed.

    * ``string-literal`` — only rewrites string constants
    * ``observability`` — confined to a log call or an exception message
    * ``logic`` — everything else, i.e. the ones worth reading
    """
    try:
        a, b = ast.parse(original), ast.parse(mutated)
    except SyntaxError:
        return "logic"

    if ast.dump(_strip_string_constants(ast.parse(original))) == ast.dump(
        _strip_string_constants(ast.parse(mutated))
    ):
        return "string-literal"

    stripper = _StripObservability()
    if ast.dump(stripper.visit(a)) == ast.dump(stripper.visit(b)):
        return "observability"
    return "logic"


def _function_sources(mutant_file: Path) -> dict[str, str]:
    """Map every top-level ``x_<fn>__mutmut_<suffix>`` to its source.

    mutmut emits the original as ``x_<fn>__mutmut_orig`` and each mutant as
    ``x_<fn>__mutmut_<N>`` in the same file, which is what makes a local diff
    possible without shelling out to ``mutmut show`` once per survivor.
    """
    source = mutant_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and "__mutmut_" in node.name:
            out[node.name] = ast.get_source_segment(source, node) or ""
    return out


def _diff(original: str, mutated: str, label: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            mutated.splitlines(keepends=True),
            fromfile=f"{label} (original)",
            tofile=f"{label} (mutant)",
            n=1,
        )
    )


def _collect(target: str) -> ModuleResult | None:
    """Read one module's ``.meta`` verdicts and build its survivor diffs."""
    meta_path = MUTANTS_DIR / f"{target}.meta"
    mutant_path = MUTANTS_DIR / target
    if not meta_path.exists():
        return None

    verdicts = json.loads(meta_path.read_text(encoding="utf-8")).get("exit_code_by_key", {})
    sources = _function_sources(mutant_path) if mutant_path.exists() else {}
    result = ModuleResult(module=target)

    for key, exit_code in verdicts.items():
        status = _STATUS_BY_EXIT_CODE.get(exit_code, "suspicious")
        result.counts[status] += 1
        if status != "survived":
            continue

        # "models.skill.x_skill_score__mutmut_16" -> mangled name is the tail
        mangled = key.rsplit(".", 1)[-1]
        original_name = mangled.rsplit("__mutmut_", 1)[0] + "__mutmut_orig"
        original_src = sources.get(original_name, "")
        mutated_src = sources.get(mangled, "")
        function = mangled.removeprefix("x_").rsplit("__mutmut_", 1)[0]

        if original_src and mutated_src:
            original_src = _normalise(original_src)
            mutated_src = _normalise(mutated_src)
            kind = _classify(original_src, mutated_src)
        else:
            # No source to compare — report it rather than silently dropping it.
            kind = "logic"

        result.survivors.append(
            Survivor(
                key=key,
                function=function,
                module=target,
                kind=kind,
                diff=_diff(original_src, mutated_src, function),
            )
        )
    return result


# mutmut forks a worker per mutant. Forking a parent that has already spun up
# a threaded BLAS pool deadlocks the child — the worker sits at 0% CPU forever
# and mutmut's wall-clock timeout never fires because nothing is running. This
# reliably hung the data/feature_engineering.py mutants (its tests build large
# frames, so BLAS threads are live by fork time). Pinning every pool to one
# thread before the fork is the fix; it costs little because the parallelism
# that matters here is across mutants, not within one.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _run_mutmut(patterns: list[str], max_children: int | None) -> None:
    cmd = [sys.executable, "-m", "mutmut", "run"]
    if max_children is not None:
        cmd += ["--max-children", str(max_children)]
    cmd += patterns

    env = {**os.environ, **_SINGLE_THREAD_ENV}
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=env)
    # mutmut exits non-zero when mutants survive, which is the normal case
    # here and must NOT abort the report. A genuine harness failure shows up
    # as missing .meta files, which _collect reports as a skipped module.
    if proc.returncode not in (0, 1, 2):
        print(f"warning: mutmut exited {proc.returncode}", file=sys.stderr)


def _markdown(results: list[ModuleResult]) -> str:
    lines = [
        "# Mutation testing report",
        "",
        "A **survivor** is a mutant the suite did not catch — a line that can be",
        "broken with CI still green. `logic` survivors are the ones worth reading;",
        "`string-literal` and `observability` survivors rewrite prose and log",
        "arguments that nothing asserts on, and survive by construction.",
        "",
        "| module | mutants | killed | logic surv. | noise surv. | score | logic score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        noise = len(r.survivors) - len(r.logic_survivors)
        score = "n/a" if r.score is None else f"{r.score:.1f}%"
        logic_score = "n/a" if r.logic_score is None else f"{r.logic_score:.1f}%"
        lines.append(
            f"| `{r.module}` | {r.scored} | {r.killed} | {len(r.logic_survivors)} "
            f"| {noise} | {score} | {logic_score} |"
        )

    total_scored = sum(r.scored for r in results)
    total_killed = sum(r.killed for r in results)
    total_noise = sum(len(r.survivors) - len(r.logic_survivors) for r in results)
    total_logic = sum(len(r.logic_survivors) for r in results)
    if total_scored:
        denom = total_scored - total_noise
        logic_pct = f"{100.0 * total_killed / denom:.1f}%" if denom > 0 else "n/a"
        lines += [
            "",
            f"**Overall: {total_killed}/{total_scored} killed "
            f"({100.0 * total_killed / total_scored:.1f}%) · "
            f"logic score {logic_pct} · "
            f"{total_logic} logic survivors, {total_noise} noise survivors**",
        ]

    for r in results:
        logic = r.logic_survivors
        if not logic:
            continue
        lines += ["", f"## Logic survivors — `{r.module}` ({len(logic)})", ""]
        for s in logic:
            lines += [f"### `{s.function}` — `{s.key}`", "", "```diff", s.diff.rstrip(), "```", ""]

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        help="Restrict to one target, e.g. models/skill.py (default: all of only_mutate)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Report from the existing mutants/ state without re-running mutmut",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=None,
        help="Cap concurrent mutant workers (default: mutmut's own, os.cpu_count())",
    )
    parser.add_argument("--json", default="mutation-report.json", help="JSON report path")
    parser.add_argument("--markdown", default="mutation-report.md", help="Markdown report path")
    args = parser.parse_args()

    targets = _load_targets()
    if args.module:
        if args.module not in targets:
            sys.exit(
                f"{args.module} is not in [tool.mutmut] only_mutate.\nTargets: "
                + ", ".join(targets)
            )
        targets = [args.module]

    if not args.skip_run:
        # mutmut matches mutant names by fnmatch, so a module maps to a glob
        # over its dotted prefix: models/skill.py -> models.skill.*
        patterns = [args.module.removesuffix(".py").replace("/", ".") + ".*"] if args.module else []
        _run_mutmut(patterns, args.max_children)

    results = []
    for target in targets:
        result = _collect(target)
        if result is None:
            print(f"warning: no results for {target} (mutants/{target}.meta missing)")
            continue
        results.append(result)

    if not results:
        print("No mutation results found. Did mutmut run?", file=sys.stderr)
        return 1

    payload = {
        "targets": [r.module for r in results],
        "overall": {
            "scored": sum(r.scored for r in results),
            "killed": sum(r.killed for r in results),
            "survived": sum(r.counts.get("survived", 0) for r in results),
        },
        "modules": [
            {
                "module": r.module,
                "counts": dict(r.counts),
                "scored": r.scored,
                "killed": r.killed,
                "score_pct": r.score,
                "score_excl_string_only_pct": r.logic_score,
                "survivors": [
                    {
                        "key": s.key,
                        "function": s.function,
                        "kind": s.kind,
                        "diff": s.diff,
                    }
                    for s in r.survivors
                ],
            }
            for r in results
        ],
    }

    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _markdown(results)
    Path(args.markdown).write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"JSON report:     {args.json}")
    print(f"Markdown report: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
