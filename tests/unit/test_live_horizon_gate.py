"""#349 — the gate answers a generous question; something must answer the sharp one.

``is_forecast_quality_acceptable`` grades the TRAINING HOLDOUT against the
7-day band (22% rollback). That is intentionally the most permissive question
available, because hiding a balancing authority from the product is a heavy
act — and as measured on 2026-07-28 it hides **none** of the 51.

The defect was never that too little was hidden. It was that the horizon the
benchmark actually publishes, measured on the path that actually serves, had
no voice at all: SEC could sit at ``rollback`` on every model at 24h while
passing the gate comfortably, and nothing anywhere said so.

These tests pin the second opinion and the disagreement signal.
"""

from __future__ import annotations

import pytest

from models.model_service import (
    OPERATING_HORIZON,
    gate_disagrees_with_live,
    live_horizon_verdict,
)

# SEC's real serve-path drift, read from /api/v1/drift/SEC on 2026-07-28.
# Every served model is rollback-graded at 24h; the ensemble is merely the
# least bad of them. n_7d was 160 records for each.
SEC_DRIFT_24H = {
    "arima": 16.942,
    "ensemble": 12.215,
    "prophet": 33.977,
    "xgboost": 25.155,
}

#: SEC's holdout champion at the time #349 was filed — comfortably inside the
#: 7-day band's 22% rollback bar, which is exactly why the gate stayed silent.
SEC_HOLDOUT_BEST_MAPE = 6.96


def _drift_payload(by_model: dict[str, float], horizon: str = "24h") -> dict:
    """Build a drift_horizon payload in the shape the scoring job writes."""
    return {
        "models": {
            name: {horizon: {"rolling_mape_7d": mape, "n_7d": 160}}
            for name, mape in by_model.items()
        }
    }


class TestLiveHorizonVerdict:
    def test_sec_every_model_rollback_at_24h(self):
        """SEC's real numbers: the champion is rollback, not merely mediocre."""
        v = live_horizon_verdict(_drift_payload(SEC_DRIFT_24H))
        assert v is not None
        assert v["grade"] == "rollback"
        assert v["champion"] == "ensemble"
        assert v["champion_mape"] == pytest.approx(12.215, abs=0.001)
        assert v["n_models"] == 4
        assert v["horizon"] == "24h"

    def test_champion_is_the_best_model_not_the_ensemble_by_default(self):
        """A region where a base model beats the ensemble reports that model.

        The gate's sibling bug (#255) was assuming a fixed champion. This
        picks the minimum across whatever was actually scored.
        """
        v = live_horizon_verdict(_drift_payload({"ensemble": 9.0, "xgboost": 4.2, "arima": 11.0}))
        assert v["champion"] == "xgboost"
        assert v["champion_mape"] == pytest.approx(4.2)
        assert v["grade"] == "acceptable"

    def test_operating_horizon_is_24h_not_the_gates_7d(self):
        """The whole point: one number, two bands, opposite verdicts.

        8.5% is `rollback` at 24h but only `target` at 7d — and sits far
        under the 22% bar the gate actually tests, so the gate passes it
        without comment. If this function ever drifted onto the gate's band
        the escalation would go silent again, and this is what notices.

        (Drift records 24h/48h/72h — never 7d — so the 7d band is reached
        through `mape_grade` directly rather than a payload that cannot
        exist.)
        """
        from config import MAPE_BY_HORIZON, mape_grade

        assert OPERATING_HORIZON == "24h"
        v = live_horizon_verdict(_drift_payload({"ensemble": 8.5}))
        assert v["grade"] == "rollback"
        assert mape_grade(8.5, "7d") == "target"
        assert MAPE_BY_HORIZON["7d"]["rollback"] > 8.5

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"models": {}}, {"models": {"ensemble": {}}}],
        ids=["none", "empty", "no-models", "no-horizon-block"],
    )
    def test_no_signal_is_not_a_failing_grade(self, payload):
        """A warming or untrained region must return None, never a verdict.

        Same rule the gate follows for an absent region: absence of evidence
        is not evidence of a bad forecast, and grading it would escalate every
        cold start.
        """
        assert live_horizon_verdict(payload) is None

    def test_non_finite_mape_is_skipped_not_ranked(self):
        """NaN must not win the min() and become the champion."""
        v = live_horizon_verdict(_drift_payload({"ensemble": float("nan"), "arima": 6.0}))
        assert v["champion"] == "arima"
        assert v["n_models"] == 1

    def test_measurement_is_labelled_on_the_verdict(self):
        """The verdict says how it was measured, so a consumer cannot mistake
        it for the holdout number sitting next to it."""
        v = live_horizon_verdict(_drift_payload(SEC_DRIFT_24H))
        assert v["measurement"] == "serve-path drift, 7d rolling"


class TestDisagreement:
    def test_sec_is_the_disagreement_case(self):
        """The exact silent state #349 describes, with SEC's real numbers.

        Holdout 6.96% passes the 22% bar; live 12.215% is rollback at 24h.
        Both instruments are right about their own question — the point is
        that the pair is now flagged instead of reconciled by silence.
        """
        gate = {"acceptable": True, "best_mape": SEC_HOLDOUT_BEST_MAPE}
        live = live_horizon_verdict(_drift_payload(SEC_DRIFT_24H))
        assert gate_disagrees_with_live(gate, live) is True

    def test_sec_still_passes_the_gate(self, monkeypatch):
        """SEC does NOT get hidden, escalation notwithstanding. Deliberate.

        A hard cutover to serve-path grading would have hidden 7 of 51 BAs
        (SPA 25.3, SEC 12.2, IID 11.4, AZPS 9.6, WALC 7.7, LDWP 7.5, CPLE
        7.1 — measured 2026-07-28), three of them within 0.7 points of the
        threshold, i.e. inside the noise. The escalation is a signal, not a
        new hiding rule.

        This drives the REAL gate function against a published entry carrying
        the disagreement, because the thing worth pinning is that visibility
        is unmoved by it.
        """
        import models.model_service as ms

        entry = {
            "acceptable": True,
            "best_mape": SEC_HOLDOUT_BEST_MAPE,
            "disagrees": True,
            "live_horizon": live_horizon_verdict(_drift_payload(SEC_DRIFT_24H)),
        }
        monkeypatch.setattr(ms, "_get_gate_status", lambda: {"SEC": entry})
        assert ms.is_forecast_quality_acceptable("SEC") is True
        # ...and the second opinion is retrievable from the same map, so the
        # API can disclose it without a second Redis round-trip.
        assert ms.published_live_horizon("SEC")["grade"] == "rollback"

    def test_published_live_horizon_absent_for_pre_349_entries(self, monkeypatch):
        """A gate entry written by an older scoring job has no live_horizon.

        Must read as "no second opinion yet", never as a passing grade — the
        deploy window where jobs and web tier disagree on schema is real.
        """
        import models.model_service as ms

        monkeypatch.setattr(
            ms, "_get_gate_status", lambda: {"SEC": {"acceptable": True, "best_mape": 6.96}}
        )
        assert ms.published_live_horizon("SEC") is None
        assert ms.published_live_horizon("NOT_A_REGION") is None

    def test_a_healthy_region_does_not_escalate(self):
        gate = {"acceptable": True, "best_mape": 3.1}
        live = live_horizon_verdict(_drift_payload({"ensemble": 2.9}))
        assert gate_disagrees_with_live(gate, live) is False

    def test_an_already_gated_region_is_not_a_disagreement(self):
        """If the gate already hides it, the instruments agree — no alert."""
        gate = {"acceptable": False, "best_mape": 30.0}
        live = live_horizon_verdict(_drift_payload({"ensemble": 28.0}))
        assert live["grade"] == "rollback"
        assert gate_disagrees_with_live(gate, live) is False

    @pytest.mark.parametrize(
        "gate,live",
        [
            (None, {"grade": "rollback"}),
            ({"acceptable": True}, None),
            ("not-a-dict", {"grade": "rollback"}),
            ({"acceptable": True}, "not-a-dict"),
        ],
        ids=["no-gate", "no-live", "bad-gate", "bad-live"],
    )
    def test_missing_or_malformed_inputs_never_escalate(self, gate, live):
        """A missing measurement must not page anyone."""
        assert gate_disagrees_with_live(gate, live) is False


class TestGateHysteresis:
    """P2-17 (#273): the DECISION is sticky; the metric is untouched."""

    def _m(self, mape):
        return {"xgboost": {"mape": mape}}

    def test_no_prior_state_reproduces_the_bare_threshold(self):
        """A BA the system has never judged must be unaffected."""
        from models.model_service import gate_verdict_from_metrics

        assert gate_verdict_from_metrics(self._m(21.9))["acceptable"] is True
        assert gate_verdict_from_metrics(self._m(22.1))["acceptable"] is False
        # explicit None is the same as omitting it
        assert (
            gate_verdict_from_metrics(self._m(22.1), currently_visible=None)["acceptable"] is False
        )

    def test_visible_ba_hides_at_the_bar_not_below_it(self):
        """Hysteresis is one-sided: it must not make hiding HARDER.

        The discriminating value is inside the band but under the bar. At 22.1
        both the correct one-sided rule and a two-sided one say "hide", so that
        value proves nothing — the first version of this test used it, and a
        mutation applying the band on BOTH transitions passed. This asserts the
        band applies on the way back IN only.
        """
        from config import GATE_HYSTERESIS_PTS, MAPE_BY_HORIZON
        from models.model_service import gate_verdict_from_metrics

        bar = MAPE_BY_HORIZON["7d"]["rollback"]
        inside_band = bar - GATE_HYSTERESIS_PTS / 2  # under the bar, inside the band

        # A visible BA here stays visible — the band must NOT apply.
        assert (
            gate_verdict_from_metrics(self._m(inside_band), currently_visible=True)["acceptable"]
            is True
        )
        # ...and a genuinely bad one still hides AT the bar, not below it.
        assert (
            gate_verdict_from_metrics(self._m(bar + 0.1), currently_visible=True)["acceptable"]
            is False
        )

    def test_hidden_ba_needs_to_clear_the_band_to_return(self):
        from config import GATE_HYSTERESIS_PTS, MAPE_BY_HORIZON
        from models.model_service import gate_verdict_from_metrics

        bar = MAPE_BY_HORIZON["7d"]["rollback"]
        just_under = bar - GATE_HYSTERESIS_PTS / 2  # under the bar, inside the band
        clearly_under = bar - GATE_HYSTERESIS_PTS - 0.1

        # This is the flap the study measured: without hysteresis it reappears.
        assert gate_verdict_from_metrics(self._m(just_under))["acceptable"] is True
        assert (
            gate_verdict_from_metrics(self._m(just_under), currently_visible=False)["acceptable"]
            is False
        )
        assert (
            gate_verdict_from_metrics(self._m(clearly_under), currently_visible=False)["acceptable"]
            is True
        )

    def test_band_is_configured_not_hardcoded(self):
        from config import GATE_HYSTERESIS_PTS

        assert GATE_HYSTERESIS_PTS > 0

    def test_no_signal_still_stays_visible_even_when_hidden(self):
        """Absent metrics = warming, never a reason to keep a BA hidden."""
        from models.model_service import gate_verdict_from_metrics

        out = gate_verdict_from_metrics({}, currently_visible=False)
        assert out["acceptable"] is True and out["best_mape"] is None

    def test_best_mape_is_reported_raw_regardless_of_state(self):
        """The published metric must not change — only the verdict does."""
        from models.model_service import gate_verdict_from_metrics

        for state in (None, True, False):
            assert gate_verdict_from_metrics(self._m(20.5), currently_visible=state)[
                "best_mape"
            ] == pytest.approx(20.5)
