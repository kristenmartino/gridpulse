"""The study verification harness must fail on the things it exists to catch.

A verification instrument tested only on clean input proves nothing — it is
the `guard-blind-by-construction` pattern this repo has already graduated a
rule for. So every test here feeds the check the defect it is supposed to
notice and asserts it notices, and the passing cases exist only to show it is
not failing indiscriminately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from study_verify import (  # noqa: E402
    check_controls,
    check_schema,
    compare_classifications,
    rule_of_three_bound,
    verify_quote,
)


def _entry(code: str, cls: str, **over) -> dict:
    base = {"code": code, "classification": cls, "source_url": "https://example.org/x"}
    base.update(over)
    return base


class TestControls:
    """The check that bites on systematic error, so it must actually bite."""

    def test_a_wrong_control_fails(self) -> None:
        # An agent confidently calling Bonneville investor-owned is exactly
        # the silent-confident-error case controls exist for.
        report = check_controls([_entry("BPAT", "investor-owned")], {"BPAT": "federal"})
        assert not report.ok
        assert report.failed == [("BPAT", "investor-owned", "federal")]

    def test_a_missing_control_fails(self) -> None:
        """Omitting a control must not read as passing it."""
        report = check_controls([_entry("SCL", "municipal")], {"BPAT": "federal"})
        assert not report.ok
        assert report.missing == ["BPAT"]

    def test_correct_controls_pass(self) -> None:
        report = check_controls(
            [_entry("BPAT", "federal"), _entry("FPL", "investor-owned")],
            {"BPAT": "federal", "FPL": "investor-owned"},
        )
        assert report.ok
        assert set(report.passed) == {"BPAT", "FPL"}

    def test_controls_exclude_the_contested_entities(self) -> None:
        """Grading an agent on SRP or BANC would score it against a guess.

        Those are what the study is trying to resolve; seeding them as
        ground truth would launder an assumption into a control.
        """
        from study_verify import CONTROLS

        for contested in ("SRP", "BANC", "IID", "SC", "SEC"):
            assert contested not in CONTROLS


class TestSchema:
    def test_invalid_class_is_rejected(self) -> None:
        problems = check_schema([_entry("XYZ", "public")])  # not a valid class
        assert any("not in" in p for p in problems)

    def test_missing_source_url_is_rejected(self) -> None:
        problems = check_schema([_entry("XYZ", "municipal", source_url="")])
        assert any("source_url" in p for p in problems)

    def test_duplicate_code_is_rejected(self) -> None:
        problems = check_schema([_entry("DUK", "investor-owned"), _entry("DUK", "municipal")])
        assert any("duplicate" in p for p in problems)

    def test_clean_input_yields_nothing(self) -> None:
        assert check_schema([_entry("DUK", "investor-owned")]) == []


class TestInterRaterAgreement:
    def test_disagreement_is_surfaced(self) -> None:
        report = compare_classifications(
            [_entry("SRP", "municipal")], [_entry("SRP", "state-authority")]
        )
        assert report.disagreed == {"SRP": ("municipal", "state-authority")}
        assert "SRP" in report.needs_adjudication
        assert report.rate() == 0.0

    def test_self_flag_from_either_side_forces_adjudication(self) -> None:
        """A flag from one rater is enough; agreement does not override it."""
        report = compare_classifications(
            [_entry("BANC", "municipal", ambiguous=True)], [_entry("BANC", "municipal")]
        )
        assert report.agreed == {"BANC": "municipal"}
        assert "BANC" in report.needs_adjudication

    def test_an_entity_only_one_rater_returned_is_adjudicated(self) -> None:
        report = compare_classifications([_entry("TAL", "municipal")], [])
        assert report.only_a == ["TAL"]
        assert "TAL" in report.needs_adjudication

    def test_full_agreement_needs_no_adjudication(self) -> None:
        pair = [_entry("DUK", "investor-owned")]
        report = compare_classifications(pair, list(pair))
        assert report.rate() == 1.0
        assert report.needs_adjudication == []


class TestQuoteVerification:
    """The check standing between a fabricated number and the study."""

    PAGE = (
        "Table 5-3. Summer Peak Demand Forecast\n"
        "The Company projects a 2030 summer peak of 14,207 MW, reflecting "
        "an annual growth rate of 2.1 percent."
    )

    def test_a_fabricated_quote_is_rejected(self) -> None:
        # Plausible, well-formed, and not in the document.
        assert not verify_quote(self.PAGE, "a 2030 summer peak of 19,400 MW")

    def test_an_empty_quote_is_rejected(self) -> None:
        assert not verify_quote(self.PAGE, "")
        assert not verify_quote(self.PAGE, "   ")

    def test_a_real_quote_is_accepted(self) -> None:
        assert verify_quote(self.PAGE, "a 2030 summer peak of 14,207 MW")

    def test_pdf_text_layer_mangling_does_not_cause_false_rejection(self) -> None:
        """Line breaks and curly quotes are artefacts, not evidence of fraud."""
        mangled = "The  Company\nprojects a 2030 summer\n peak of 14,207 MW"
        assert verify_quote(self.PAGE, mangled)

    def test_verification_does_not_catch_misattribution(self) -> None:
        """Documented limit, asserted so it cannot be forgotten.

        The quote is genuinely on the page, so the mechanical check passes —
        but if the study wanted WINTER peak, this record is wrong and only
        the human pass in §8.4 can tell.
        """
        assert verify_quote(self.PAGE, "a 2030 summer peak of 14,207 MW")


class TestRuleOfThree:
    def test_bounds_match_the_registered_table(self) -> None:
        assert round(rule_of_three_bound(30), 1) == 10.0
        assert round(rule_of_three_bound(60), 1) == 5.0
        assert round(rule_of_three_bound(100), 1) == 3.0
        assert round(rule_of_three_bound(300), 1) == 1.0

    def test_checking_nothing_bounds_nothing(self) -> None:
        assert rule_of_three_bound(0) == 100.0

    def test_the_shortcut_refuses_when_errors_were_found(self) -> None:
        """With errors observed, 3/n is not the right statistic."""
        assert rule_of_three_bound(100, n_errors=1) is None
