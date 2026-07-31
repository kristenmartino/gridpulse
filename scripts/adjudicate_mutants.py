"""Verify reported mutation survivors against the FULL unit suite.

`scripts/mutation_test.py` reports what mutmut found. mutmut picks the tests to
run per mutant from coverage tracing, which is blind to work done in a
subprocess — so a mutant a subprocess test would kill is reported as a
survivor. That is limitation 1 in ``docs/TEST_QUALITY.md``, and it is not
hypothetical: it produced a false finding on the first batch adjudicated by
hand.

This closes the loop. For each survivor it applies the mutant to the real
source file, runs the **whole** unit suite, and reverts:

* suite **passes**  -> ``confirmed`` — genuinely unnoticed, worth reading
* suite **fails**   -> ``false-survivor`` — mutmut missed the killing test,
  which is named in the output

``confirmed`` still does not mean "bug". An *equivalent* mutant changes the
source without changing behaviour and no test can ever kill it. Separating
those two is a human judgement; this script narrows the field it has to be
applied to.

Usage::

    python scripts/adjudicate_mutants.py --module models/evaluation.py --limit 20
    python scripts/adjudicate_mutants.py --keys models.skill.x_mape__mutmut_3
    python scripts/adjudicate_mutants.py --limit 40 --out adjudication.json

Requires a clean working tree and a populated ``mutants/`` directory (run
``scripts/mutation_test.py`` first).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MUTANTS_DIR = REPO_ROOT / "mutants"

_MUTANT_DEF_RE = re.compile(r"^(\s*)def\s+(x_\w+?__mutmut_\w+)\s*\(", re.MULTILINE)
_FAILED_RE = re.compile(r"^FAILED (\S+)", re.MULTILINE)
_ERROR_RE = re.compile(r"^ERROR (\S+)", re.MULTILINE)

# --fast: the only tests mutmut can have MISSED.
#
# mutmut picks per-mutant tests from coverage tracing during its stats pass, so
# by construction it already ran every test that traces into the mutated
# function — and those did not kill it, or it would not be a survivor. The one
# way a test kills a mutant without mutmut knowing is if tracing could not see
# it, which means the work happened in a child process.
#
# So confirming a survivor does not need the whole suite; it needs exactly the
# tests tracing is blind to. `grep -rln "subprocess|multiprocessing|Popen"
# tests/unit/` returns two files, and that is the entire blind spot.
#
# Measured: 2.9s per mutant instead of ~70s, same verdict on the known false
# survivor (the ensemble_combine length-guard, killed by the hash test below).
# Re-derive this list if tests start spawning processes elsewhere; --full is
# always available as the unconditional check.
_BLIND_SPOT_TESTS = [
    "tests/unit/test_stable_hash_reproducibility.py",
    "tests/unit/test_cache.py",
]


@dataclass
class Verdict:
    key: str
    module: str
    function: str
    status: str  # "confirmed" | "false-survivor" | "skipped"
    killed_by: list[str]
    note: str = ""


def _require_clean_tree() -> None:
    """Refuse to run against uncommitted work.

    This script rewrites source files and restores them from memory. If it is
    killed mid-run the restore never happens, and the fallback is
    ``git checkout`` — which would take any uncommitted work in that file with
    it. Cheaper to refuse.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        sys.exit(
            "Working tree has uncommitted changes:\n"
            f"{dirty}\n\n"
            "This script rewrites source files in place. Commit or stash first."
        )


def _function_sources(mutant_file: Path) -> dict[str, tuple[str, str]]:
    """``{mangled_name: (source, indent)}`` for every mutant in a mutmut file."""
    source = mutant_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    out: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and "__mutmut_" in node.name:
            seg = ast.get_source_segment(source, node)
            if seg:
                out[node.name] = (seg, "")
    return out


def _real_function_span(path: Path, func_name: str) -> tuple[int, int] | None:
    """Line span (0-indexed, end-exclusive) of ``func_name`` in the real file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            return start, node.end_lineno
    return None


def _apply(path: Path, func_name: str, mutant_src: str) -> str:
    """Swap the body of ``func_name`` for the mutant's. Returns the original text."""
    original = path.read_text(encoding="utf-8")
    span = _real_function_span(path, func_name)
    if span is None:
        raise LookupError(f"{func_name} not found in {path}")
    start, end = span
    lines = original.splitlines(keepends=True)

    # Rename the mutmut wrapper back to the real function name, and match the
    # indentation of the definition being replaced (module-level here, but do
    # not assume it).
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    body = _MUTANT_DEF_RE.sub(rf"\g<1>def {func_name}(", mutant_src, count=1)
    body = "\n".join(indent + ln if ln.strip() else ln for ln in body.splitlines())
    if not body.endswith("\n"):
        body += "\n"

    path.write_text("".join(lines[:start]) + body + "".join(lines[end:]), encoding="utf-8")
    return original


def _run_suite(test_path: str | list[str], timeout: int) -> tuple[bool, list[str]]:
    """Run the suite. Returns ``(passed, killing_test_ids)``.

    ``-x`` stops at the first failure: a killed mutant is usually detected in
    seconds, while a genuine survivor pays the full suite runtime. That
    asymmetry is the right way round — survivors are the interesting result and
    deserve the complete check.
    """
    paths = [test_path] if isinstance(test_path, str) else list(test_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-x",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode == 0:
        return True, []
    killers = _FAILED_RE.findall(out) + _ERROR_RE.findall(out)
    return False, killers[:5]


def _select(report: dict, modules: list[str] | None, keys: list[str] | None, limit: int | None):
    picked = []
    for mod in report["modules"]:
        if modules and mod["module"] not in modules:
            continue
        for s in mod["survivors"]:
            if s["kind"] != "logic":
                continue  # noise survivors are noise by construction
            if keys and s["key"] not in keys:
                continue
            picked.append((mod["module"], s))
    return picked[:limit] if limit else picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="mutation-report.json")
    ap.add_argument("--module", action="append", dest="modules")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Run only mutmut's blind-spot tests instead of the whole suite (see _BLIND_SPOT_TESTS)",
    )
    ap.add_argument("--tests", default="tests/unit", help="Test path to run per mutant")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="adjudication.json")
    args = ap.parse_args()

    _require_clean_tree()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    picked = _select(report, args.modules, args.keys, args.limit)
    if not picked:
        sys.exit("Nothing selected.")

    tests = _BLIND_SPOT_TESTS if args.fast else args.tests
    label = f"blind-spot tests ({len(_BLIND_SPOT_TESTS)} files)" if args.fast else f"`{args.tests}`"
    print(f"Adjudicating {len(picked)} logic survivors against {label}\n", flush=True)
    sources_cache: dict[str, dict[str, tuple[str, str]]] = {}
    verdicts: list[Verdict] = []

    for i, (module, s) in enumerate(picked, 1):
        mangled = s["key"].rsplit(".", 1)[-1]
        real_name = mangled.removeprefix("x_").rsplit("__mutmut_", 1)[0]
        path = REPO_ROOT / module

        if module not in sources_cache:
            sources_cache[module] = _function_sources(MUTANTS_DIR / module)
        mutant_src = sources_cache[module].get(mangled, (None, None))[0]
        if not mutant_src:
            verdicts.append(
                Verdict(s["key"], module, real_name, "skipped", [], "mutant source missing")
            )
            print(f"[{i}/{len(picked)}] {s['key']}: SKIPPED (no source)", flush=True)
            continue

        original = None
        try:
            original = _apply(path, real_name, mutant_src)
            passed, killers = _run_suite(tests, args.timeout)
        except Exception as exc:  # noqa: BLE001 - report, never leave a file mutated
            verdicts.append(
                Verdict(s["key"], module, real_name, "skipped", [], f"{type(exc).__name__}: {exc}")
            )
            print(f"[{i}/{len(picked)}] {s['key']}: SKIPPED ({exc})", flush=True)
            continue
        finally:
            if original is not None:
                path.write_text(original, encoding="utf-8")

        status = "confirmed" if passed else "false-survivor"
        verdicts.append(Verdict(s["key"], module, real_name, status, killers))
        tag = (
            "CONFIRMED survivor"
            if passed
            else f"FALSE survivor — killed by {killers[0] if killers else '?'}"
        )
        print(f"[{i}/{len(picked)}] {s['key']}: {tag}", flush=True)

    confirmed = [v for v in verdicts if v.status == "confirmed"]
    false = [v for v in verdicts if v.status == "false-survivor"]
    skipped = [v for v in verdicts if v.status == "skipped"]

    Path(args.out).write_text(
        json.dumps(
            {
                "checked": len(verdicts),
                "confirmed": len(confirmed),
                "false_survivors": len(false),
                "skipped": len(skipped),
                "verdicts": [asdict(v) for v in verdicts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{len(confirmed)} confirmed · {len(false)} false survivors · "
        f"{len(skipped)} skipped  ->  {args.out}"
    )
    subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, check=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
