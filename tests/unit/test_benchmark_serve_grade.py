"""#348 — the one unflattering fact on /benchmark that wasn't published.

The page's whole argument is that it discloses what an adversarial reader
would otherwise find: drop counts, exclusions, the not-lead-matched limit,
the dual official arm. A row we internally grade ``rollback`` was the
exception — published as an ordinary comparison, with nothing said.

Two facts are disclosed here, both derived from data already in Redis:

* ``serve_grade`` — our own rolling grade for the **exact** series the row
  scores (same model, same lead), not a healthier neighbouring measurement.
* ``served_series`` — this arm always scores the model, so where a BA was
  substituted onto the seasonal-naive baseline, the row measures the
  forecaster rather than what that BA's users are served.

SEC is the live case for both at once.
"""

from __future__ import annotations

from models.benchmark import compute_benchmark_payload, serve_grade

#: SEC's real 24h drift block, /api/v1/drift/SEC on 2026-07-28. Its ensemble
#: is the series the benchmark scores, and we grade it rollback.
SEC_ENSEMBLE_24H = {"grade": "rollback", "rolling_mape_7d": 12.215, "n_7d": 160}


def _horizon(model="ensemble", lead="24h", block=None):
    return {"models": {model: {lead: (SEC_ENSEMBLE_24H if block is None else block)}}}


class TestServeGrade:
    def test_reads_the_grade_of_the_scored_series(self):
        g = serve_grade(_horizon(), "ensemble", "24h")
        assert g["grade"] == "rollback"
        assert g["model"] == "ensemble"
        assert g["horizon"] == "24h"
        assert g["rolling_mape_7d"] == 12.215

    def test_an_acceptable_row_reports_its_grade_not_nothing(self):
        """`acceptable` must round-trip, so the page can distinguish
        "graded fine" from "not graded yet"."""
        g = serve_grade(
            _horizon(block={"grade": "acceptable", "rolling_mape_7d": 4.1}), "ensemble", "24h"
        )
        assert g["grade"] == "acceptable"

    def test_grade_is_horizon_and_model_matched_not_borrowed(self):
        """The marker must describe the line in THAT row.

        A payload graded rollback at 24h but healthy at 48h must not leak its
        24h verdict onto the 48h row — the conservative arm is a different
        measurement and gets its own grade.
        """
        payload = {
            "models": {
                "ensemble": {
                    "24h": {"grade": "rollback", "rolling_mape_7d": 12.215},
                    "48h": {"grade": "acceptable", "rolling_mape_7d": 5.4},
                }
            }
        }
        assert serve_grade(payload, "ensemble", "24h")["grade"] == "rollback"
        assert serve_grade(payload, "ensemble", "48h")["grade"] == "acceptable"

    def test_the_boundary_published_is_the_one_that_actually_applies(self):
        """`rollback` is earned by exceeding **acceptable** (7.0 at 24h).

        `MAPE_BY_HORIZON["24h"]["rollback"]` is 12.0, but `mape_grade` never
        uses it as a threshold — anything above `acceptable` is already
        rollback. Quoting 12.0 on the page would tell a reader a flagged row
        is worse than it has to be, on a page whose argument is that its
        numbers mean exactly what they say.
        """
        from config import MAPE_BY_HORIZON, mape_grade

        g = serve_grade(_horizon(), "ensemble", "24h")
        assert g["acceptable_max"] == MAPE_BY_HORIZON["24h"]["acceptable"] == 7.0
        assert mape_grade(7.1, "24h") == "rollback"
        assert mape_grade(6.9, "24h") == "acceptable"

    def test_a_different_model_is_not_substituted_when_absent(self):
        """No falling back to whichever model happens to carry a grade."""
        assert serve_grade(_horizon(model="arima"), "ensemble", "24h") is None

    def test_no_grade_is_none_not_a_pass(self):
        """A warming region is ungraded, never implicitly healthy."""
        assert serve_grade(None, "ensemble", "24h") is None
        assert serve_grade({}, "ensemble", "24h") is None
        assert serve_grade(_horizon(block={"rolling_mape_7d": 9.9}), "ensemble", "24h") is None


class TestPayloadCarriesTheDisclosure:
    def _records(self):
        """Enough vintage records for a scoreable payload is expensive to
        build here; the contract under test is that the fields exist and are
        computed, which an unscoreable payload still exercises for the
        region-level pair."""
        return []

    def test_served_series_is_recorded_at_region_level(self):
        p = compute_benchmark_payload(
            "SEC", self._records(), _horizon(), "bulk", served_series="seasonal-naive"
        )
        assert p["scored_model"] == "ensemble"
        assert p["served_series"] == "seasonal-naive"
        assert p["serves_scored_model"] is False

    def test_a_normal_ba_serves_what_it_scores(self):
        p = compute_benchmark_payload(
            "PJM", self._records(), _horizon(), "churn", served_series="model"
        )
        assert p["serves_scored_model"] is True

    def test_unknown_served_series_is_none_not_false(self):
        """Before the first substitution-aware tick, `served_series` is absent.

        `None` must not render as "we don't serve this" — that would flag
        every BA on the page during the deploy window.
        """
        p = compute_benchmark_payload("PJM", self._records(), _horizon(), "churn")
        assert p["served_series"] is None
        assert p["serves_scored_model"] is None
