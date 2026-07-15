#!/usr/bin/env python3
"""Fail if a color literal appears anywhere but the token module.

Why this exists
---------------
GridPulse already had the convention. ``_callbacks_shared`` carried a comment
calling itself a "design-token mirror"; ``accessibility`` documented a
CVD-verified palette. Nothing enforced either, so the codebase drifted to eight
blues, three severity triads for one concept, two fuel palettes (the accessible
one with zero callsites), and ~85 orphaned literals. A convention that is only
a comment is a convention that rots. This is the rule with teeth.

The rule: a color literal may appear in ``components/tokens.py`` and nowhere
else. No per-line escape pragma — an escape hatch is how the last one rotted.
If something genuinely is not a design color, it still lives in the token
module (see ``ICON_MASK_STROKE``), where it is at least visible and named.

Two things a naive ``grep '#[0-9a-f]{6}'`` gets wrong, both real here:

1.  FALSE NEGATIVES. ``rgba(59, 130, 246, 0.08)`` is stock Tailwind blue-500
    and a hex grep sails right past it. Two such literals had already drifted
    off the line they were meant to match: the Overview hero drew an ACCENT
    line over a blue-500 fill, and the Forecast band used the retired #38D0FF
    under a line of some entirely different color.

2.  FALSE POSITIVES. This repo references GitHub issues in prose comments
    ("P2-21 (#273): ...", "#283 Phase 1"). A 3-digit hex pattern matches 49 of
    them. So we parse Python properly with ``tokenize`` and inspect STRING
    tokens only — comments cannot paint anything, and issue refs stop being our
    problem by construction.

CSS is checked with a line scan instead: ``:root`` is where custom.css declares
its values, and every rule below it must go through ``var(--token)``.

Usage:
    python scripts/check_color_tokens.py          # repo-wide, exit 1 on any find
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import sys
import tokenize

REPO = pathlib.Path(__file__).resolve().parents[1]

# The one file allowed to hold color literals.
TOKEN_MODULE = REPO / "components" / "tokens.py"

# Python trees to police — every tree that can put a color on screen.
#
# Tests are included on purpose: a test that hardcodes "#2BD67B" silently
# re-pins a value the token module is supposed to own, which is how
# test_sprint3 ended up asserting a palette the app did not paint.
#
# personas/ is included because leaving it out is how a FIFTH color system
# (matplotlib's tab10 defaults) sat unnoticed in personas/config.py, rendering
# as the insight card's left border. A gate is only worth the trees it walks.
PY_ROOTS = [
    "components",
    "personas",
    "tests",
    "scripts",
    "models",
    "data",
    "simulation",
    "jobs",
    "app.py",
    "api.py",
    "config.py",
    "observability.py",
]

# scripts/color_science.py is the verifier itself; its literals are test
# vectors and matrix coefficients, not design colors.
PY_EXCLUDE = {
    TOKEN_MODULE,
    REPO / "scripts" / "color_science.py",
    REPO / "scripts" / "check_color_tokens.py",
    REPO / "scripts" / "verify_palette.py",
}

# 6- or 8-digit hex, anywhere in a string.
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?\b")

# 3-digit shorthand (#fff) ONLY when it is the entire value. Prose in this repo
# cites GitHub issues inline — including in user-facing copy, e.g. "Prediction
# intervals are omitted until per-model calibration (#196)." — and a loose
# 3-digit pattern flags every one of those. Anchoring to a whole-string match
# still catches a real `color: "#fff"` while ignoring "(#196)".
HEX3_RE = re.compile(r"^#[0-9a-fA-F]{3}$")

RGBA_RE = re.compile(r"\brgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+", re.I)

CSS_VAR_LINE = re.compile(r"^\s*--[\w-]+\s*:")


def _iter_py() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in PY_ROOTS:
        p = REPO / root
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(x for x in p.rglob("*.py") if "__pycache__" not in x.parts)
    return [p for p in sorted(set(out)) if p not in PY_EXCLUDE]


def _docstring_lines(src: str) -> set[int]:
    """Line numbers of real docstrings — prose, which cannot paint anything.

    Only the first statement of a module/class/function counts. An earlier
    version skipped EVERY triple-quoted string, which let app.py's
    ``index_string`` — a triple-quoted HTML template carrying
    ``<link rel="mask-icon" color="#35c6ff">`` — sail straight through the
    gate. Triple quotes mean nothing about whether a string reaches a browser.
    """
    lines: set[int] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            for ln in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                lines.add(ln)
    return lines


def check_python(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, offending literal) for color literals in STRING tokens."""
    hits: list[tuple[int, str]] = []
    try:
        src = path.read_text()
    except OSError:
        return hits
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return hits
    skip = _docstring_lines(src)
    for tok in toks:
        # STRING covers plain strings; FSTRING_MIDDLE covers the literal text
        # between {} in an f-string on 3.12+, so f"1px solid #263556" is caught.
        if tok.type not in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
            continue
        if tok.start[0] in skip:
            continue
        text = tok.string
        for m in HEX_RE.finditer(text):
            hits.append((tok.start[0], m.group(0)))
        if HEX3_RE.match(text.strip("\"'")):
            hits.append((tok.start[0], text.strip("\"'")))
        for m in RGBA_RE.finditer(text):
            hits.append((tok.start[0], m.group(0) + "...)"))
    return hits


def check_css(path: pathlib.Path) -> list[tuple[int, str]]:
    """Color literals outside a custom-property declaration."""
    hits: list[tuple[int, str]] = []
    in_block_comment = False
    for i, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                continue
        while "/*" in line:
            head, rest = line.split("/*", 1)
            if "*/" in rest:
                line = head + rest.split("*/", 1)[1]
            else:
                line = head
                in_block_comment = True
                break
        # A `--token: value;` line is a declaration — that is where values live.
        if CSS_VAR_LINE.match(line):
            continue
        for m in HEX_RE.finditer(line):
            hits.append((i, m.group(0)))
        for m in re.finditer(r"#[0-9a-fA-F]{3}\b(?![0-9a-fA-F])", line):
            hits.append((i, m.group(0)))
        for m in RGBA_RE.finditer(line):
            hits.append((i, m.group(0) + "...)"))
    return hits


def main() -> int:
    failures: list[str] = []

    for path in _iter_py():
        for lineno, lit in check_python(path):
            failures.append(f"{path.relative_to(REPO)}:{lineno}: {lit}")

    css = REPO / "assets" / "custom.css"
    if css.exists():
        for lineno, lit in check_css(css):
            failures.append(f"{css.relative_to(REPO)}:{lineno}: {lit}")

    if failures:
        print("Color literals found outside the token module:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            f"\n{len(failures)} violation(s).\n\n"
            "Every color value belongs in components/tokens.py (mirrored by the\n"
            "assets/custom.css :root block). Import it and reference the token:\n\n"
            "    from components import tokens\n"
            "    line=dict(color=tokens.ACCENT)\n"
            "    fillcolor=tokens.alpha(tokens.ACCENT, 0.08)   # not rgba(...) by hand\n\n"
            "In CSS, reference it: color: var(--accent-base);\n",
            file=sys.stderr,
        )
        return 1

    print("check_color_tokens: OK — no color literals outside components/tokens.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
