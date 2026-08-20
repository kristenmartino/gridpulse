"""Verification harness for the ownership forecast-bias study.

Implements the mechanical half of
``docs/studies/OWNERSHIP_FORECAST_BIAS.md`` §8. Every check here is
deterministic — string matching, set comparison, HTTP status. **No check in
this module may call a language model.** Validating a model's citation with
another model introduces a correlated failure, which §8.2 prohibits.

What each check catches, and what it does not:

``compare_classifications``
    Catches idiosyncratic disagreement between two independently prompted
    classifiers. Blind to systematic error: two agents sharing a wrong prior
    agree, and this reports agreement.

``check_controls``
    Catches an agent that is confidently wrong on entities whose answer is
    documented and unambiguous. This is the check that does bite on
    systematic error, which is why controls are asserted here from primary
    sources rather than supplied by the agent being tested.

``check_urls``
    Catches fabricated or dead citations. Does NOT check that the page
    supports the claim.

``verify_quote``
    Catches fabricated extracted values, because a quote that was never in
    the document cannot be found in it. Does NOT catch misattribution — a
    real quote read out of the wrong table. §8.4's human pass owns that.

``rule_of_three_bound``
    Converts "we checked n and found nothing" into the number that may
    actually be written down.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: Ownership classes the study admits. An agent returning anything else is a
#: schema failure, not a classification to be interpreted.
VALID_CLASSES = frozenset(
    {
        "investor-owned",
        "municipal",
        "cooperative",
        "federal",
        "state-authority",
        "rto-iso",
    }
)

#: Seeded controls: entities whose ownership is documented and not in dispute.
#: Asserted here from primary sources, never supplied by the agent under test.
#: Deliberately EXCLUDES the known-hard cases (SRP, BANC, IID, SC, SEC) —
#: those are what the study is trying to resolve, so scoring an agent on them
#: would be grading it against a guess.
CONTROLS: dict[str, str] = {
    "BPAT": "federal",  # Bonneville Power Administration, US DOE
    "SPA": "federal",  # Southwestern Power Administration, US DOE
    "TVA": "federal",  # Tennessee Valley Authority, federal corporation
    "SCL": "municipal",  # Seattle City Light, City of Seattle department
    "TPWR": "municipal",  # Tacoma Power, City of Tacoma department
    "FPL": "investor-owned",  # Florida Power & Light, NextEra subsidiary
    "DUK": "investor-owned",  # Duke Energy Carolinas
    "AECI": "cooperative",  # Associated Electric Cooperative Inc
    "PJM": "rto-iso",
    "CAISO": "rto-iso",
    "ERCOT": "rto-iso",
}


@dataclass
class ControlReport:
    """Outcome of the seeded known-answer check."""

    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str, str]] = field(default_factory=list)  # code, got, want
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.missing

    def summary(self) -> str:
        if self.ok:
            return f"controls: {len(self.passed)}/{len(self.passed)} correct"
        parts = [f"controls: {len(self.passed)} correct"]
        if self.failed:
            wrong = ", ".join(f"{c} got {g!r} want {w!r}" for c, g, w in self.failed)
            parts.append(f"{len(self.failed)} WRONG ({wrong})")
        if self.missing:
            parts.append(f"{len(self.missing)} missing ({', '.join(self.missing)})")
        return "; ".join(parts)


def check_controls(
    entries: Iterable[dict], controls: dict[str, str] | None = None
) -> ControlReport:
    """Score an agent's classifications against documented ground truth.

    Per §8.3 this runs BEFORE any result is inspected. An agent that misses a
    control has its remaining output treated as unverified.
    """
    controls = CONTROLS if controls is None else controls
    by_code = {e["code"]: e for e in entries}
    report = ControlReport()
    for code, want in controls.items():
        entry = by_code.get(code)
        if entry is None:
            report.missing.append(code)
            continue
        got = entry.get("classification")
        if got == want:
            report.passed.append(code)
        else:
            report.failed.append((code, str(got), want))
    return report


def check_schema(entries: Iterable[dict]) -> list[str]:
    """Reject outputs that cannot be interpreted, rather than interpreting them."""
    problems: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        code = entry.get("code")
        if not code:
            problems.append(f"row {i}: no `code`")
            continue
        if code in seen:
            problems.append(f"{code}: duplicate row")
        seen.add(code)
        cls = entry.get("classification")
        if cls not in VALID_CLASSES:
            problems.append(f"{code}: classification {cls!r} not in {sorted(VALID_CLASSES)}")
        if not entry.get("source_url"):
            problems.append(f"{code}: no source_url")
    return problems


@dataclass
class AgreementReport:
    """Inter-rater comparison between two independent classification passes."""

    agreed: dict[str, str] = field(default_factory=dict)
    disagreed: dict[str, tuple[str, str]] = field(default_factory=dict)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    self_flagged: list[str] = field(default_factory=list)

    @property
    def needs_adjudication(self) -> list[str]:
        """Everything a human or an adjudicating pass must resolve."""
        return sorted(
            set(self.disagreed) | set(self.self_flagged) | set(self.only_a) | set(self.only_b)
        )

    def rate(self) -> float:
        total = len(self.agreed) + len(self.disagreed)
        return len(self.agreed) / total if total else 0.0


def compare_classifications(a: Iterable[dict], b: Iterable[dict]) -> AgreementReport:
    """Compare two independent passes. Disagreement is the signal.

    Note the asymmetry this exists to exploit: a model asked to self-report
    uncertainty is poorly calibrated, but two models asked the same question
    independently disagree *where the evidence is thin*. Self-flags are
    unioned in rather than trusted alone.
    """
    a_by, b_by = {e["code"]: e for e in a}, {e["code"]: e for e in b}
    report = AgreementReport()
    for code in sorted(set(a_by) | set(b_by)):
        ea, eb = a_by.get(code), b_by.get(code)
        if ea is None:
            report.only_b.append(code)
            continue
        if eb is None:
            report.only_a.append(code)
            continue
        if ea.get("ambiguous") or eb.get("ambiguous"):
            report.self_flagged.append(code)
        ca, cb = ea.get("classification"), eb.get("classification")
        if ca == cb:
            report.agreed[code] = str(ca)
        else:
            report.disagreed[code] = (str(ca), str(cb))
    return report


def check_urls(entries: Iterable[dict], timeout: float = 10.0) -> dict[str, str]:
    """Confirm each cited URL resolves. Catches fabricated citations only.

    A 200 says the page exists. It says nothing about whether the page
    supports the claim, which no deterministic check can establish.
    """
    results: dict[str, str] = {}
    for entry in entries:
        url = entry.get("source_url")
        code = entry.get("code", "?")
        if not url:
            results[code] = "MISSING"
            continue
        try:
            req = Request(url, method="GET", headers={"User-Agent": "gridpulse-study-verify"})
            with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — cited sources only
                results[code] = f"{resp.status}"
        except HTTPError as exc:
            results[code] = f"HTTP {exc.code}"
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            results[code] = f"ERROR {type(exc).__name__}"
    return results


def _normalize(text: str) -> str:
    """Fold the differences a PDF text layer introduces but a reader ignores."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"[‐-―]", "-", text)  # dash variants
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(page_text: str, quote: str) -> bool:
    """Confirm a verbatim span really appears in the page it was cited from.

    Whitespace and typographic variants are folded, because a PDF text layer
    mangles those without a human noticing. Nothing else is relaxed: a value
    the source does not contain must fail.
    """
    if not quote or not quote.strip():
        return False
    return _normalize(quote) in _normalize(page_text)


def rule_of_three_bound(n_checked: int, n_errors: int = 0) -> float | None:
    """95% upper bound on an error rate given a clean check of ``n``.

    Returns the bound as a percentage. Defined only for the zero-error case,
    which is the one that tempts an overclaim; with errors found, report the
    observed rate and a binomial interval instead of this shortcut.
    """
    if n_errors:
        return None
    if n_checked <= 0:
        return 100.0
    return min(100.0, 300.0 / n_checked)


def main() -> int:
    """CLI: verify one or two classification files.

    ``python scripts/study_verify.py a.json [b.json]``
    """
    import sys

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    entries_a = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    print(f"== schema: {args[0]}")
    for problem in check_schema(entries_a) or ["clean"]:
        print(f"   {problem}")

    print("\n== controls (checked before results are inspected)")
    print(f"   {check_controls(entries_a).summary()}")

    if len(args) > 1:
        entries_b = json.loads(Path(args[1]).read_text(encoding="utf-8"))
        print(f"\n== schema: {args[1]}")
        for problem in check_schema(entries_b) or ["clean"]:
            print(f"   {problem}")
        print(f"\n== controls: {args[1]}")
        print(f"   {check_controls(entries_b).summary()}")

        agreement = compare_classifications(entries_a, entries_b)
        print(f"\n== inter-rater agreement: {agreement.rate():.1%}")
        for code, (ca, cb) in sorted(agreement.disagreed.items()):
            print(f"   DISAGREE {code}: {ca} vs {cb}")
        print(f"   needs adjudication: {', '.join(agreement.needs_adjudication) or 'none'}")

    print("\n== cited URLs")
    for code, status in sorted(check_urls(entries_a).items()):
        flag = "" if status == "200" else "  <-- CHECK"
        print(f"   {code:6s} {status}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
