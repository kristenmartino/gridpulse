"""The harness that exists because a single window lied to us.

The ARIMA order study ran a 168h holdout on two consecutive days and CAISO's
verdict reversed sign (−7.24 → +3.87 pts). The decision it fed was right for
other reasons, but four published per-BA numbers were noise presented as
evidence.

These tests pin the two guards that would have caught it, using the real
observed numbers wherever possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.rolling_eval import (
    DECISION_METRIC,
    MAX_ABS_BIAS_PCT,
    MAX_MAPE_REGRESSION_PTS,
    MIN_SIGN_CONSISTENCY,
    MIN_WINDOWS_FOR_A_VERDICT,
    REPORTED_METRIC,
    bias_pct,
    paired_deltas,
    rolling_origin_splits,
    satisficing_check,
    verdict,
    wape,
)


class TestMetricPolicy:
    def test_we_optimise_wape_and_only_report_mape(self):
        """The deliberate split. MAPE is published, WAPE is optimised."""
        assert DECISION_METRIC == "wape"
        assert REPORTED_METRIC == "mape"

    def test_mape_is_asymmetric_which_is_why_it_is_not_the_target(self):
        """The concrete defect, asserted rather than asserted-in-prose.

        Over- and under-forecasting by the SAME number of MW score very
        differently under MAPE: the under-forecast looks better. Minimise that
        and you drift toward under-forecasting demand, which is the expensive
        direction for a grid. WAPE scores the two identically.
        """
        from models.evaluation import compute_mape

        actual = np.array([100.0, 100.0, 100.0, 100.0])
        over = actual + 20.0
        under = actual - 20.0

        assert compute_mape(actual, over) == pytest.approx(20.0)
        assert compute_mape(actual, under) == pytest.approx(20.0)
        # Symmetric at equal magnitude — the asymmetry shows on the tails:
        assert compute_mape(actual, actual * 3) == pytest.approx(200.0)  # 200% over
        assert compute_mape(actual, actual * 0) == pytest.approx(100.0)  # capped at 100%
        # Same absolute MW error, wildly different penalty. WAPE does not care
        # about direction, only magnitude:
        assert wape(actual, over) == pytest.approx(wape(actual, under))

    def test_wape_does_not_explode_on_a_low_demand_hour(self):
        """The SEC problem: a ~300 MW co-op has overnight hours where a small
        MW error is a huge percentage. MAPE lets that one hour dominate."""
        from models.evaluation import compute_mape

        actual = np.array([1000.0, 1000.0, 1000.0, 5.0])
        pred = np.array([1010.0, 1010.0, 1010.0, 15.0])
        assert compute_mape(actual, pred) > 50  # dragged by the 5 MW hour
        assert wape(actual, pred) < 2  # total error / total demand


class TestSatisficingConstraints:
    def test_a_systematic_under_forecast_is_vetoed(self):
        """The guard MAPE specifically needs and lacks."""
        r = satisficing_check(treatment_bias_pct=-3.5, control_mape=5.0, treatment_mape=5.0)
        assert r["passed"] is False
        assert "under-forecasting" in r["failures"][0]

    def test_a_systematic_over_forecast_is_also_vetoed(self):
        r = satisficing_check(treatment_bias_pct=4.0, control_mape=5.0, treatment_mape=5.0)
        assert r["passed"] is False
        assert "over-forecasting" in r["failures"][0]

    def test_the_published_metric_may_not_quietly_regress(self):
        """A WAPE win that costs a point of MAPE is not shippable — MAPE is
        what the benchmark page publishes."""
        r = satisficing_check(treatment_bias_pct=0.1, control_mape=5.0, treatment_mape=6.0)
        assert r["passed"] is False
        assert "regress" in r["failures"][0]

    def test_a_small_mape_movement_is_tolerated(self):
        r = satisficing_check(treatment_bias_pct=0.1, control_mape=5.0, treatment_mape=5.3)
        assert r["passed"] is True

    def test_a_bias_exactly_on_the_band_edge_passes(self):
        """The band is inclusive: |bias| == MAX_ABS_BIAS_PCT is inside it.

        Pinned because both comparisons in this function could be flipped
        from ``>`` to ``>=`` with all 2,687 unit tests green (mutation
        testing, docs/TEST_QUALITY.md). Every existing test sits comfortably
        inside or outside the band, so the edge itself — the only input where
        the two spellings disagree — was decided by an untested branch.

        Which side the edge falls on is a real policy choice, not a detail:
        EVALUATION_POLICY.md states the constraint as |bias| <= 2%, so exactly
        2% must SHIP.
        """
        r = satisficing_check(
            treatment_bias_pct=MAX_ABS_BIAS_PCT, control_mape=5.0, treatment_mape=5.0
        )
        assert r["passed"] is True, "a bias of exactly the limit is inside the band, not outside"

        r_neg = satisficing_check(
            treatment_bias_pct=-MAX_ABS_BIAS_PCT, control_mape=5.0, treatment_mape=5.0
        )
        assert r_neg["passed"] is True, "the band is symmetric; the low edge is inside it too"

        over = satisficing_check(
            treatment_bias_pct=MAX_ABS_BIAS_PCT + 0.01, control_mape=5.0, treatment_mape=5.0
        )
        assert over["passed"] is False, "one hundredth of a point past the edge is outside"

    def test_a_mape_regression_exactly_on_the_limit_passes(self):
        """Same edge, the other constraint: a regression of exactly
        MAX_MAPE_REGRESSION_PTS is tolerated, one hundredth more is not.

        5.5 - 5.0 is exactly 0.5 in binary floating point, so this really does
        land on the boundary rather than near it.
        """
        exactly = satisficing_check(
            treatment_bias_pct=0.1,
            control_mape=5.0,
            treatment_mape=5.0 + MAX_MAPE_REGRESSION_PTS,
        )
        assert exactly["passed"] is True, "a regression of exactly the limit is still shippable"

        just_over = satisficing_check(
            treatment_bias_pct=0.1,
            control_mape=5.0,
            treatment_mape=5.0 + MAX_MAPE_REGRESSION_PTS + 0.01,
        )
        assert just_over["passed"] is False
        assert "regress" in just_over["failures"][0]

    def test_an_unmeasured_constraint_fails_rather_than_passes(self):
        """An unchecked constraint is not a satisfied one."""
        assert (
            satisficing_check(treatment_bias_pct=None, control_mape=5.0, treatment_mape=5.0)[
                "passed"
            ]
            is False
        )
        assert (
            satisficing_check(treatment_bias_pct=0.1, control_mape=None, treatment_mape=5.0)[
                "passed"
            ]
            is False
        )

    def test_bias_sign_convention_is_forecast_minus_actual(self):
        """Negative = forecasting low. The docstring says so; pin it, because
        a flipped sign would veto exactly the wrong arm."""
        actual = np.array([100.0, 100.0])
        assert bias_pct(actual, np.array([90.0, 90.0])) == pytest.approx(-10.0)
        assert bias_pct(actual, np.array([110.0, 110.0])) == pytest.approx(10.0)


class TestRollingOriginSplits:
    def test_no_window_can_see_its_own_future(self):
        """Train always ends exactly where test begins. The whole point."""
        for train, test in rolling_origin_splits(2000, n_windows=5, holdout_h=168, min_train_h=500):
            assert train.start == 0
            assert train.stop == test.start
            assert test.stop - test.start == 168

    def test_holdouts_are_disjoint_and_walk_backwards(self):
        splits = rolling_origin_splits(2000, n_windows=5, holdout_h=168, min_train_h=500)
        assert len(splits) == 5
        spans = [(t.start, t.stop) for _, t in splits]
        assert spans == sorted(spans, reverse=True), "newest window first"
        for (s1, _), (_, e2) in zip(spans, spans[1:], strict=False):
            assert e2 <= s1, "holdout windows must not overlap"

    def test_history_running_out_truncates_rather_than_inventing_windows(self):
        """Asking for 20 windows out of 1200 rows yields the honest count.

        A harness that quietly returned fewer windows *and* said nothing is how
        a thin study passes for a thorough one — the callers print `len()`.
        """
        splits = rolling_origin_splits(1200, n_windows=20, holdout_h=168, min_train_h=500)
        assert 0 < len(splits) < 20
        for train, _ in splits:
            assert train.stop - train.start >= 500

    def test_min_train_is_respected_not_approximated(self):
        splits = rolling_origin_splits(1000, n_windows=10, holdout_h=168, min_train_h=800)
        assert all(t.stop >= 800 for t, _ in splits)

    def test_stride_can_overlap_holdouts_when_asked(self):
        """Overlapping holdouts buy more windows from short history, at the
        cost of correlated windows — allowed, but never the default."""
        tight = rolling_origin_splits(
            2000, n_windows=6, holdout_h=168, stride_h=24, min_train_h=500
        )
        assert len(tight) == 6
        assert tight[0][1].start - tight[1][1].start == 24

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_windows": 0, "holdout_h": 168},
            {"n_windows": 5, "holdout_h": 0},
            {"n_windows": 5, "holdout_h": 168, "stride_h": 0},
        ],
        ids=["no-windows", "no-holdout", "zero-stride"],
    )
    def test_degenerate_configuration_yields_nothing_rather_than_looping(self, kwargs):
        assert rolling_origin_splits(2000, min_train_h=100, **kwargs) == []


class TestPairedDeltas:
    def test_positive_means_treatment_is_better(self):
        """Matches the benchmark payload's `delta_*` convention."""
        d = paired_deltas([10.0, 10.0], [8.0, 12.0])
        assert d.tolist() == [2.0, -2.0]

    def test_a_failed_window_drops_from_both_arms(self):
        """One arm blowing up must not leave the arms scored on different
        window sets — that would compare a full control against a
        survivorship-filtered treatment."""
        d = paired_deltas([10.0, 10.0, 10.0], [8.0, float("nan"), 12.0])
        assert d.tolist() == [2.0, -2.0]

    def test_mismatched_lengths_raise_rather_than_zip_short(self):
        with pytest.raises(ValueError):
            paired_deltas([1.0, 2.0, 3.0], [1.0, 2.0])


class TestVerdict:
    def test_the_caiso_case_is_inconclusive(self):
        """The two real observed deltas, one day apart: −7.24 and +3.87.

        This is the whole reason the module exists. Any rule that calls a
        winner here is not fit for purpose.
        """
        v = verdict([-7.24, 3.87])
        assert v["decisive"] is False
        assert v["winner"] is None
        assert "window" in v["reason"]

    def test_one_window_never_decides_however_large_the_gap(self):
        v = verdict([-19.18])
        assert v["decisive"] is False
        assert v["n"] == 1
        assert str(MIN_WINDOWS_FOR_A_VERDICT) in v["reason"]

    def test_a_consistent_effect_is_called(self):
        v = verdict([1.9, 2.1, 2.0, 2.2, 1.8])
        assert v["decisive"] is True
        assert v["winner"] == "treatment"
        assert v["sign_consistency"] == 1.0

    def test_a_consistent_loss_names_the_control(self):
        v = verdict([-1.9, -2.1, -2.0, -2.2])
        assert v["decisive"] is True
        assert v["winner"] == "control"

    def test_one_catastrophic_window_cannot_manufacture_a_verdict(self):
        """The ISONE shape: 3 small wins and one −19.18.

        The mean is strongly negative, so a mean-only rule would declare the
        control the winner. But the MEDIAN is positive — most windows favour
        the treatment. Mean and median disagreeing in sign is the signature of
        outlier domination, and the honest reading is "usually a wash,
        occasionally catastrophic": a tail-risk statement, not a win.
        """
        v = verdict([0.1, 0.15, 0.05, -19.18])
        assert v["mean"] < 0
        assert v["median"] > 0
        assert v["decisive"] is False
        assert "outlier" in v["reason"] and "tail risk" in v["reason"]

    def test_outlier_domination_is_diagnosed_before_plain_noise(self):
        """Both guards reject the ISONE shape — an outlier inflates the very
        variance it is tested against — but the reason a human reads should be
        the specific one, not "within noise"."""
        assert "noise" not in verdict([0.1, 0.15, 0.05, -19.18])["reason"]
        # ...and genuine noise still reports as noise, not as an outlier.
        assert "noise" in verdict([0.4, -0.3, 0.5, -0.6, 0.2])["reason"]

    def test_a_small_but_rock_steady_effect_still_counts(self):
        """Consistency is not a proxy for size — a tiny reliable gain is a
        real gain, and the harness must not require drama to see it."""
        v = verdict([0.21, 0.19, 0.20, 0.22, 0.18, 0.20])
        assert v["decisive"] is True
        assert v["winner"] == "treatment"
        assert abs(v["mean"]) < 0.25

    def test_a_real_but_unreliable_effect_is_refused_on_consistency_alone(self):
        """Wins big in 4 of 7 windows, loses small in 3.

        Mean and median both positive, so this is not outlier domination; the
        magnitude clears 2x stderr, so it is not noise. It fails on
        CONSISTENCY alone — an effect that is real on average but absent in
        43% of windows is not something to ship.

        Added because mutation testing removed the consistency guard entirely
        and every other test still passed: the guard was reachable but
        uncovered, which is the same as not having it.
        """
        v = verdict([3.0, 3.0, 3.0, 3.0, -0.5, -0.5, -0.5])
        assert v["mean"] > 0 and v["median"] > 0, "not the outlier-domination shape"
        assert v["decisive"] is False
        assert "reliable" in v["reason"]
        assert v["sign_consistency"] == pytest.approx(4 / 7, abs=0.01)

    def test_noise_around_zero_is_refused(self):
        v = verdict([0.4, -0.3, 0.5, -0.6, 0.2])
        assert v["decisive"] is False
        assert "noise" in v["reason"]

    def test_consistency_exactly_at_the_threshold_still_ships(self):
        """MIN_SIGN_CONSISTENCY is a floor, not a bar to clear: winning
        exactly 75% of windows is enough.

        Three wins in four windows is exactly 0.75 — the only input where
        ``consistency < min`` and ``consistency <= min`` disagree, and the
        mutation that flips them survived the whole suite. The shape is chosen
        so it fails on nothing else: mean and median are both positive (not
        outlier domination) and |mean| clears 2x stderr (not noise), leaving
        consistency as the single deciding constraint.
        """
        v = verdict([2.0, 2.0, 2.0, -0.2])

        assert v["sign_consistency"] == MIN_SIGN_CONSISTENCY, "must land ON the edge, not near it"
        assert v["mean"] > 0 and v["median"] > 0, "not the outlier-domination shape"
        assert v["decisive"] is True, "exactly the required consistency is sufficient"
        assert v["winner"] == "treatment"

    def test_an_effect_exactly_at_the_noise_threshold_still_ships(self):
        """|mean| == min_t x stderr is out of the noise band, not in it.

        The threshold is injected rather than reverse-engineered from the
        default so the equality is exact in floating point: min_t is set to
        the ratio the data actually produces, which makes
        ``abs(mean) < min_t * stderr`` a comparison of a value against itself.
        With ``<`` the effect ships; with ``<=`` it is refused as noise. The
        suite could not tell those apart.
        """
        deltas = [2.0, 2.0, 2.0, -0.2]
        d = np.asarray(deltas)
        exact_t = abs(float(d.mean())) / float(d.std(ddof=1) / np.sqrt(d.size))

        v = verdict(deltas, min_t=exact_t)
        assert v["decisive"] is True, "sitting exactly on the noise threshold is not noise"
        assert "noise" not in v["reason"]

        just_over = verdict(deltas, min_t=exact_t * 1.001)
        assert just_over["decisive"] is False
        assert "noise" in just_over["reason"]

    def test_an_exact_no_op_reports_no_difference_not_a_winner(self):
        """13 of 51 BAs select the same ARIMA order either way — delta 0 in
        every window. That is a finding ('this knob does nothing here'), not a
        tie to be broken."""
        v = verdict([0.0, 0.0, 0.0, 0.0, 0.0])
        assert v["decisive"] is False
        assert v["reason"] == "no difference"

    def test_tail_risk_is_always_reported_even_when_decisive(self):
        """A passing mean is not the whole story, so worst/best travel with
        every verdict."""
        v = verdict([2.0, 2.1, 1.9, 2.2, -0.4])
        assert v["worst_window"] == -0.4
        assert v["best_window"] == 2.2

    # ------------------------------------------------------------------
    # Below: found by mutation testing (docs/TEST_QUALITY.md). The tests
    # above pin the DECISION — decisive, winner, a reason substring — and
    # almost nothing pins the DIAGNOSTICS that justify it. Those numbers are
    # what an A/B study publishes, so a wrong one is a wrong conclusion
    # presented with a straight face.
    # ------------------------------------------------------------------

    def test_identical_deltas_in_every_window_do_not_crash(self):
        """A treatment that wins every window by the SAME amount is decisive.

        This was a live ZeroDivisionError, not just an untested path. The
        function deliberately handles ``stderr == 0`` at the top — the comment
        there names identical deltas explicitly — but the closing reason string
        divided by ``stderr`` unconditionally. Any four identical non-zero
        deltas raised, so a perfectly consistent improvement crashed the
        verdict instead of winning it.

        Only the all-zero case escaped, because ``mean == 0`` returns earlier.
        That is why ``test_an_exact_no_op_reports_no_difference_not_a_winner``
        never caught it.

        Reachable: deltas computed from rounded per-window metrics collide
        exactly, and ``rolling_eval`` is the mandatory route for every model
        change (CLAUDE.md).
        """
        v = verdict([2.0, 2.0, 2.0, 2.0])

        assert v["decisive"] is True
        assert v["winner"] == "treatment"
        assert v["stderr"] == 0.0
        assert v["t"] is None, "a t-statistic needs spread; there is none"
        assert v["sign_consistency"] == 1.0
        assert v["reason"] == (
            "treatment wins 100% of 4 windows, mean +2.000 pts (identical in every window)"
        )

        # The same shape against the control arm.
        loss = verdict([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5])
        assert loss["decisive"] is True
        assert loss["winner"] == "control"

    def test_the_diagnostic_payload_is_exact(self):
        """Every reported field, pinned to the value it should carry.

        `mean`/`median`/`stderr`/`t`/`sign_consistency`/`worst_window`/
        `best_window` are all published by a study, and every one of them
        could be re-rounded, nulled, or — for `t` — computed as
        ``mean * stderr`` instead of ``mean / stderr``, with the suite green.
        A t-statistic that is a product rather than a quotient is not a
        rounding difference; it is a different claim.

        The deltas carry six decimals deliberately. With the tidy fixtures
        used elsewhere in this class every statistic lands on a short decimal,
        so the documented 4-dp (3-dp for `t` and consistency) rounding is
        invisible and could drift unnoticed. Seven windows also makes
        `sign_consistency` a repeating 6/7 rather than a clean fraction.
        """
        v = verdict([2.123456, 2.234567, 2.345678, -0.456789, 1.987654, 2.111111, 2.222222])

        assert v == {
            "decisive": True,
            "winner": "treatment",
            "n": 7,
            "mean": 1.7954,
            "median": 2.1235,
            "stderr": 0.3778,
            "t": 4.752,
            "sign_consistency": 0.857,
            "worst_window": -0.4568,
            "best_window": 2.3457,
            "reason": "treatment wins 86% of 7 windows, mean +1.795 pts (4.8x stderr)",
        }

    def test_sign_consistency_is_zero_when_there_is_no_difference(self):
        """The no-op path reports 0% consistency, not 100%.

        ``consistency = 1.0 if mean != 0 else 0.0`` — the else arm is what
        stops an exact no-op from being reported as a perfectly consistent
        result. ``test_an_exact_no_op_...`` asserts the reason but not this
        field, so flipping the arm to 1.0 went unnoticed.
        """
        v = verdict([0.0, 0.0, 0.0, 0.0])

        assert v["reason"] == "no difference"
        assert v["sign_consistency"] == 0.0, "no difference is not perfect agreement"

    def test_a_mean_of_exactly_one_is_still_a_difference(self):
        """Guards the ``mean != 0`` test against becoming ``mean != 1``.

        A one-point-per-window improvement is an ordinary result, and it is
        the single input where those two spellings disagree.
        """
        v = verdict([1.0, 1.0, 1.0, 1.0])

        assert v["sign_consistency"] == 1.0
        assert v["decisive"] is True
        assert v["winner"] == "treatment"

    def test_outlier_domination_is_detected_at_a_median_of_one(self):
        """The outlier guard tests ``median != 0``, not ``median != 1``.

        Three +1.0 windows and one −20.0: the mean is negative, the median is
        exactly +1.0, and they disagree in sign — the ISONE tail-risk shape.
        A median of 1.0 is unremarkable, and it is where the mutated
        comparison stops recognising the shape and lets the outlier through.
        """
        v = verdict([1.0, 1.0, 1.0, -20.0])

        assert v["median"] == 1.0
        assert v["mean"] < 0, "the outlier drags the mean across zero"
        assert v["decisive"] is False
        assert "disagree in sign" in v["reason"]

    def test_no_windows_at_all(self):
        v = verdict([])
        assert v["decisive"] is False
        assert v["n"] == 0
        assert v["reason"] == "no scored windows"

    def test_non_finite_windows_are_dropped_not_propagated(self):
        v = verdict([2.0, float("nan"), 2.1, 1.9, 2.2])
        assert v["n"] == 4
        assert v["decisive"] is True
        assert np.isfinite(v["mean"])
