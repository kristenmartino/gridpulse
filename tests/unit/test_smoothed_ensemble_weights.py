"""#451: EWMA-smoothed holdout MAPE as the ensemble-weight input.

The pre-registered A/B (docs/WEIGHTS_AB_STUDY.md) gave alpha=0.3 a decisive WAPE
win. These tests pin the machinery that carries the decision, not the decision —
the arithmetic of one EWMA step, the flag gate, and the fallbacks that must keep a
model weighted on a real measurement when its smoothed value is missing.
"""

from __future__ import annotations

import math

import pytest

from models.ensemble import resolve_ensemble_weights, update_smoothed_mape, weighting_mape


class TestUpdateSmoothedMape:
    def test_first_observation_seeds_the_series(self):
        """With no history there is nothing to smooth toward — take the value.

        Seeding at the first observation rather than at 0 matters: a 0 seed would
        make a new model look impossibly accurate and hand it the blend, because
        ADR-004 weights by (1/MAPE)**3.
        """
        assert update_smoothed_mape(None, 4.0, alpha=0.3) == 4.0

    def test_one_step_is_the_ewma_recurrence(self):
        assert update_smoothed_mape(4.0, 9.0, alpha=0.3) == pytest.approx(0.3 * 9.0 + 0.7 * 4.0)

    def test_alpha_one_is_the_raw_value(self):
        """alpha=1 must degenerate to today's behaviour exactly.

        This is the property that makes the flag safe to reason about: the
        treatment contains the control as a limiting case.
        """
        assert update_smoothed_mape(4.0, 9.0, alpha=1.0) == 9.0

    def test_a_missing_latest_keeps_the_previous_value(self):
        """A run that failed to score must not reset the history."""
        assert update_smoothed_mape(4.0, None, alpha=0.3) == 4.0

    def test_nothing_usable_yields_none_not_zero(self):
        """None flows to resolve_ensemble_weights as "not measured" → equal weights.

        A 0.0 would be a *number*, and would survive into the cubed weighting as
        a division by zero or an infinite weight.
        """
        assert update_smoothed_mape(None, None, alpha=0.3) is None

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_unusable_observations_are_ignored_not_folded_in(self, bad):
        assert update_smoothed_mape(4.0, bad, alpha=0.3) == 4.0

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_an_alpha_outside_the_unit_interval_raises(self, alpha):
        with pytest.raises(ValueError):
            update_smoothed_mape(4.0, 9.0, alpha=alpha)

    def test_smoothing_damps_a_flap_without_ignoring_it(self):
        """The behaviour the study measured: a single bad draw moves the input less.

        docs/HOLDOUT_STABILITY_STUDY.md measured the estimator flapping a median
        12% run-to-run. A weight input that tracks every flap chases noise; one
        that ignores it entirely would never notice a model degrading.
        """
        steady = 4.0
        spike = update_smoothed_mape(steady, 12.0, alpha=0.3)
        assert steady < spike < 12.0
        assert spike == pytest.approx(6.4)


class TestWeightingMapeFlagGate:
    def test_flag_off_returns_the_raw_holdout_mape(self, monkeypatch):
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        assert weighting_mape(4.0, {"mape_ewma": 9.0}) == 4.0

    def test_flag_on_prefers_the_smoothed_value(self, monkeypatch):
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        assert weighting_mape(4.0, {"mape_ewma": 9.0}) == 9.0

    def test_flag_on_falls_back_to_raw_when_no_series_exists_yet(self, monkeypatch):
        """The first run after the flag flips has no persisted EWMA.

        Falling back to the raw MAPE keeps that run weighting on a real
        measurement. Returning None would drop the whole ensemble to equal
        weights for a night — a bigger change than the one under test.
        """
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        assert weighting_mape(4.0, {}) == 4.0
        assert weighting_mape(4.0, None) == 4.0

    @pytest.mark.parametrize("bad", [0.0, -2.0, float("nan")])
    def test_an_unusable_smoothed_value_falls_back_rather_than_propagating(self, monkeypatch, bad):
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        assert weighting_mape(4.0, {"mape_ewma": bad}) == 4.0

    def test_an_unmeasured_model_stays_unmeasured_under_either_setting(self, monkeypatch):
        for flag in (True, False):
            monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": flag})
            assert weighting_mape(None, {}) is None


class TestItStillFeedsTheSharedRule:
    def test_the_smoothed_input_flows_through_resolve_ensemble_weights(self, monkeypatch):
        """P2-16's shared rule must be what consumes the new input.

        The point of #451 is to change *which number* is weighted by, not how.
        If a caller ever computed weights itself from the smoothed value, the two
        jobs could disagree again — the exact defect P2-16 closed.
        """
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        members = ["xgboost", "prophet", "arima"]
        metas = {
            "xgboost": (2.0, {"mape_ewma": 2.0}),
            "prophet": (9.0, {"mape_ewma": 4.0}),
            "arima": (9.0, {"mape_ewma": 4.0}),
        }
        scores = {m: weighting_mape(raw, extra) for m, (raw, extra) in metas.items()}
        weights, rule = resolve_ensemble_weights(members, scores)

        assert rule == "inverse_mape_cubed"
        assert sum(weights.values()) == pytest.approx(1.0)
        # (1/2)^3 = 0.125 against two of (1/4)^3 = 0.015625 → 0.8 / 0.1 / 0.1
        assert weights["xgboost"] == pytest.approx(0.8)
        assert weights["prophet"] == pytest.approx(0.1)

    def test_smoothing_here_concentrates_rather_than_flattens(self, monkeypatch):
        """The study's counter-intuitive finding, pinned as a property.

        Measured HHI went 0.603 (raw) → 0.617 (ewma_0.3): smoothing made the
        blend MORE concentrated. The intuition that it "spreads weight around"
        is wrong, and a future reader who assumes otherwise would mis-explain the
        A/B — so an example where a stale-but-better model gains mass is worth
        keeping.
        """
        members = ["xgboost", "prophet"]

        def hhi(w):
            return sum(v * v for v in w.values())

        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        raw_w, _ = resolve_ensemble_weights(
            members,
            {
                m: weighting_mape(*v)
                for m, v in {
                    "xgboost": (5.0, {"mape_ewma": 3.0}),
                    "prophet": (6.0, {"mape_ewma": 9.0}),
                }.items()
            },
        )
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        sm_w, _ = resolve_ensemble_weights(
            members,
            {
                m: weighting_mape(*v)
                for m, v in {
                    "xgboost": (5.0, {"mape_ewma": 3.0}),
                    "prophet": (6.0, {"mape_ewma": 9.0}),
                }.items()
            },
        )
        assert hhi(sm_w) > hhi(raw_w)
        assert sm_w["xgboost"] > raw_w["xgboost"]


class TestTrainingJobCarriesTheSeries:
    def test_the_previous_vintages_value_is_folded_forward(self, monkeypatch):
        from jobs import training_job

        class _Meta:
            extra = {"mape_ewma": 4.0}

        monkeypatch.setattr("models.persistence.get_model_metadata", lambda r, m: _Meta())
        got = training_job._carry_smoothed_mape("ERCOT", "xgboost", 9.0)
        assert got == pytest.approx(0.3 * 9.0 + 0.7 * 4.0)

    def test_an_unreadable_history_does_not_block_the_save(self, monkeypatch):
        """A model that trained fine must not fail to persist over its own history.

        The value degrades to this run's raw MAPE, which is exactly the control
        arm — the worst case is no smoothing, never no model.
        """
        from jobs import training_job

        def _boom(region, model_name):
            raise RuntimeError("GCS down")

        monkeypatch.setattr("models.persistence.get_model_metadata", _boom)
        assert training_job._carry_smoothed_mape("ERCOT", "xgboost", 9.0) == 9.0

    def test_no_previous_vintage_seeds_from_this_run(self, monkeypatch):
        from jobs import training_job

        monkeypatch.setattr("models.persistence.get_model_metadata", lambda r, m: None)
        assert training_job._carry_smoothed_mape("ERCOT", "prophet", 3.5) == 3.5

    def test_the_series_is_persisted_even_while_the_flag_is_off(self, monkeypatch):
        """Otherwise flipping the flag later starts from one observation.

        The whole value of the treatment is history; building it only after the
        decision would mean the first weeks of "smoothed" weights were not
        smoothed at all.
        """
        from jobs import training_job

        class _Meta:
            extra = {"mape_ewma": 4.0}

        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        monkeypatch.setattr("models.persistence.get_model_metadata", lambda r, m: _Meta())
        assert training_job._carry_smoothed_mape("ERCOT", "arima", 9.0) is not None


class TestDefaults:
    def test_alpha_matches_the_winning_arm(self):
        import config

        assert config.ENSEMBLE_MAPE_EWMA_ALPHA == 0.3

    def test_the_flag_ships_off(self):
        """Pinned deliberately: the A/B was decisive on 12 of 51 BAs and its
        t-statistic sits near its own threshold. Turning this on is a separate,
        deliberate act — and this test is where that act must be recorded."""
        import config

        assert config.FEATURE_FLAGS["smoothed_ensemble_weights"] is False

    def test_alpha_is_a_usable_smoothing_constant(self):
        import config

        assert 0.0 < config.ENSEMBLE_MAPE_EWMA_ALPHA <= 1.0
        assert math.isfinite(config.ENSEMBLE_MAPE_EWMA_ALPHA)


class TestASubOnePercentMapeIsAMeasurementNotNoise:
    """The usability filter is `v > 0`, not `v > 1` (#484).

    Mutation testing found the same survivor in all three of #451/#478's new
    functions: relaxing the filter to `> 1` rejects any MAPE in (0, 1] as
    unusable. That band is not hypothetical — it is the BEST models on the
    easiest BAs, and this repo's own docs quote a 1.6% figure as a *baseline*.

    Under the mutation, the models with the strongest holdout accuracy are the
    ones silently dropped, which is the worst possible direction for a filter
    whose job is to reject junk.
    """

    def test_the_ewma_accepts_an_excellent_latest_observation(self):
        """`update_smoothed_mape` — a 0.8% MAPE must update the series.

        Dropped, the EWMA silently ignores the best measurement a model has
        ever produced and carries the previous, worse value forward.
        """
        from models.ensemble import update_smoothed_mape

        assert update_smoothed_mape(None, 0.8) == pytest.approx(0.8)
        assert update_smoothed_mape(2.0, 0.5, alpha=0.5) == pytest.approx(1.25)

    def test_the_ewma_accepts_an_excellent_previous_value_as_its_seed(self):
        """The same filter runs over `previous_smoothed`. Rejected, a model
        whose history is excellent restarts from its latest draw and loses the
        smoothing #451 exists to provide."""
        from models.ensemble import update_smoothed_mape

        assert update_smoothed_mape(0.6, None) == pytest.approx(0.6)
        assert update_smoothed_mape(0.6, 4.0, alpha=0.5) == pytest.approx(2.3)

    def test_the_served_arm_weights_on_an_excellent_smoothed_mape(self, monkeypatch):
        """`weighting_mape` — with the flag on, a 0.9% EWMA must be used.

        Rejected, it falls back to the raw MAPE, so the best models are exactly
        the ones that never receive the smoothing being evaluated.
        """
        import config
        from models.ensemble import weighting_mape

        monkeypatch.setitem(config.FEATURE_FLAGS, "smoothed_ensemble_weights", True)

        assert weighting_mape(5.0, {"mape_ewma": 0.9}) == pytest.approx(0.9)

    def test_the_shadow_arm_still_has_an_observation_for_the_best_models(self, monkeypatch):
        """`shadow_weighting_mape` — the consequence that matters most.

        This function returns ``None`` when the alternative is unavailable, and
        that is deliberate: inventing a fallback would make both arms identical
        and the comparison vacuously "no difference". So a filter that rejects
        sub-1% values does not merely mis-weight — it removes the best models
        from the shadow arm entirely, biasing a live A/B toward whichever
        conclusion the remaining, worse models support.
        """
        import config
        from models.ensemble import shadow_weighting_mape

        monkeypatch.setitem(config.FEATURE_FLAGS, "smoothed_ensemble_weights", False)

        assert shadow_weighting_mape(5.0, {"mape_ewma": 0.9}) == pytest.approx(0.9)

    def test_zero_and_negative_are_still_rejected(self):
        """The filter must stay a filter — this is about where the line sits,
        not about removing it. A zero MAPE would divide by zero in
        `compute_ensemble_weights` (the #383 finding)."""
        import config
        from models.ensemble import shadow_weighting_mape, update_smoothed_mape, weighting_mape

        assert update_smoothed_mape(None, 0.0) is None
        assert update_smoothed_mape(None, -3.0) is None
        assert update_smoothed_mape(0.0, None) is None

        config.FEATURE_FLAGS["smoothed_ensemble_weights"] = True
        try:
            assert weighting_mape(5.0, {"mape_ewma": 0.0}) == 5.0
        finally:
            config.FEATURE_FLAGS["smoothed_ensemble_weights"] = False
        assert shadow_weighting_mape(5.0, {"mape_ewma": 0.0}) is None


class TestWeightingInputLabel:
    """``weighting_input`` names the basis ``weighting_mape`` used (#514).

    The two must agree by construction, not by both being edited together —
    a label that can disagree with the value it describes is the same defect
    this pair exists to detect, one level down. Both derive from
    ``_weighting_choice``; these tests pin that they stay in step.
    """

    META = {"mape_ewma": 4.0}

    def _both(self, monkeypatch, flag, extra):
        from models import ensemble

        monkeypatch.setattr(ensemble, "feature_enabled", lambda _n: flag, raising=False)
        monkeypatch.setitem(__import__("config").FEATURE_FLAGS, "smoothed_ensemble_weights", flag)
        return (
            ensemble.weighting_mape(6.0, extra),
            ensemble.weighting_input(6.0, extra),
        )

    def test_flag_off_reports_the_holdout_basis(self, monkeypatch):
        from models.ensemble import WEIGHT_INPUT_HOLDOUT

        value, basis = self._both(monkeypatch, False, self.META)
        assert value == 6.0
        assert basis == WEIGHT_INPUT_HOLDOUT

    def test_flag_on_with_a_series_reports_the_ewma_basis(self, monkeypatch):
        from models.ensemble import WEIGHT_INPUT_EWMA

        value, basis = self._both(monkeypatch, True, self.META)
        assert value == 4.0
        assert basis == WEIGHT_INPUT_EWMA

    def test_the_fallback_reports_holdout_not_ewma(self, monkeypatch):
        """The subtle one. With the flag ON but no series yet, the VALUE falls
        back to the raw MAPE — so the label must say ``holdout_mape`` too.

        Labelling it ``mape_ewma`` because the flag is on would make the
        divergence check compare a basis nothing actually used, and the first
        run after a flip is exactly when that happens fleet-wide.
        """
        from models.ensemble import WEIGHT_INPUT_HOLDOUT

        value, basis = self._both(monkeypatch, True, {})
        assert value == 6.0
        assert basis == WEIGHT_INPUT_HOLDOUT

    @pytest.mark.parametrize("flag", [True, False])
    @pytest.mark.parametrize("extra", [{}, {"mape_ewma": 4.0}, {"mape_ewma": float("nan")}])
    def test_label_never_disagrees_with_the_value(self, monkeypatch, flag, extra):
        """Across the whole gate: the basis is ``mape_ewma`` if and only if the
        returned value is the smoothed one."""
        from models.ensemble import WEIGHT_INPUT_EWMA

        value, basis = self._both(monkeypatch, flag, extra)
        assert (basis == WEIGHT_INPUT_EWMA) == (value == 4.0)
