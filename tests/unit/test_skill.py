"""Skill against a naive baseline (``models/skill.py``).

The gap this closes: nothing measured whether a served model beats
"yesterday, same hour". SEC ran at 18.0% against a naive 11.5% — worse than
free — and every instrument called it "bad" rather than "negative skill".

The tests pin the two things that would let that recur: a baseline computed
across a data gap (which flatters the baseline and hides the model), and an
unmeasurable region reading as a passing one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.skill import (
    BASELINE_SUBSTITUTION_MIN_POINTS,
    SEASONAL_NAIVE_LAG_H,
    mape,
    seasonal_naive_forecast,
    seasonal_naive_mape,
    should_serve_baseline,
    skill_payload,
    skill_score,
)


def _daily(n_days=10, base=300.0, swing=100.0):
    """An hourly series with a clean daily cycle — naive lag-24 is perfect."""
    t = np.arange(n_days * 24)
    return base + swing * np.sin(2 * np.pi * t / 24)


class TestBaseline:
    def test_perfect_daily_cycle_makes_the_baseline_exact(self):
        """The baseline's whole claim is that yesterday repeats. On a series
        where it does, its error must be zero — otherwise every skill score
        computed against it is offset by the baseline's own bug."""
        assert seasonal_naive_mape(_daily()) == pytest.approx(0.0, abs=1e-9)

    def test_lag_is_24_hours_not_one(self):
        """A 1-hour-lag baseline is better informed than the 24h-lead forecast
        it judges, which would make every skill score flattering."""
        assert SEASONAL_NAIVE_LAG_H == 24

    def test_a_series_of_exactly_one_lag_is_unmeasurable(self):
        """``size <= lag_h`` — 24 hours yields no pairs, 25 yields one.

        With exactly ``lag_h`` samples the lagged comparison has nothing to
        compare: ``y[24:]`` is empty. Relaxing the guard to ``<`` lets that
        through and ``mape`` of an empty selection returns NaN anyway — the
        same answer by accident, which is worth pinning precisely because the
        accident could stop happening.
        """
        assert np.isnan(seasonal_naive_mape(np.arange(24, dtype=float)))
        assert np.isfinite(seasonal_naive_mape(np.arange(25, dtype=float)))

    def test_series_shorter_than_the_lag_is_unmeasurable(self):
        assert np.isnan(seasonal_naive_mape(np.arange(10, dtype=float)))

    def test_gaps_must_be_present_as_nan_not_compacted(self):
        """Load-bearing contract, documented in the module: a compacted series
        silently lags across the gap. Pinning the behaviour so the docstring
        cannot quietly become wrong — NaN hours drop out of the comparison
        rather than pairing unrelated hours."""
        y = _daily()
        with_gap = y.copy()
        with_gap[48:72] = np.nan
        # the baseline stays finite and close to zero: the NaN window is
        # excluded, not silently paired against a different day
        assert seasonal_naive_mape(with_gap) == pytest.approx(0.0, abs=1e-9)


class TestSkillScore:
    def test_positive_when_the_model_wins(self):
        assert skill_score(5.0, 10.0) == pytest.approx(0.5)

    def test_negative_when_the_model_loses(self):
        """SEC's real shape: 18.0 model against an 11.5 baseline."""
        assert skill_score(18.0, 11.5) < 0

    def test_zero_when_it_ties(self):
        assert skill_score(4.0, 4.0) == pytest.approx(0.0)

    def test_unmeasurable_is_none_never_zero(self):
        """0.0 means "matched the baseline"; None means "no comparison was
        possible". Collapsing them lets an unmeasured region read as neutral —
        exactly how a worse-than-nothing forecast stayed invisible."""
        assert skill_score(float("nan"), 10.0) is None
        assert skill_score(5.0, float("nan")) is None
        assert skill_score(5.0, 0.0) is None


class TestPayload:
    def test_a_losing_region_is_flagged(self):
        """The field a consumer acts on. SEC-shaped: a flat series makes the
        naive baseline strong, so a mediocre model loses to it."""
        payload = skill_payload(18.0, _daily())
        assert payload["beats_baseline"] is False
        assert payload["skill"] < 0
        assert payload["points_vs_baseline"] < 0

    def test_a_winning_region_is_flagged(self):
        # A day-over-day level shift: yesterday no longer repeats, so the
        # naive baseline errs. (An alternating pattern would NOT work — any
        # period dividing 24 repeats exactly at the lag and leaves the
        # baseline perfect.)
        y = _daily()
        noisy = y + np.repeat(np.arange(10) * 30.0, 24)
        payload = skill_payload(2.0, noisy)
        assert payload["beats_baseline"] is True
        assert payload["skill"] > 0

    def test_unmeasurable_region_is_neither(self):
        payload = skill_payload(float("nan"), _daily())
        assert payload["beats_baseline"] is None
        assert payload["skill"] is None

    def test_payload_names_its_baseline(self):
        """A skill number without its baseline is unquotable — the same rule
        the benchmark methodology applies to every published figure."""
        assert "seasonal-naive" in skill_payload(5.0, _daily())["baseline"]
        assert "24h" in skill_payload(5.0, _daily())["baseline"]


class TestPayloadIsExact:
    """The whole Redis block, pinned field by field.

    ``gridpulse:skill:{region}`` is consumed by the substitution policy and
    published on the Models tab. The tests above check individual flags; every
    numeric field could be re-rounded or nulled with the suite green —
    including ``beats_baseline``, which this module's own docstring calls "the
    field worth acting on".
    """

    #: 24 h at 100 then 24 h at 125: the lag-24 baseline is |125-100|/125,
    #: exactly 20%, so every derived number below is exact rather than a
    #: floating-point tail.
    SERIES = np.array([100.0] * 24 + [125.0] * 24)

    def test_a_winning_region(self):
        assert skill_payload(15.0, self.SERIES) == {
            "model_mape": 15.0,
            "baseline_mape": 20.0,
            "baseline": "seasonal-naive lag 24h",
            "skill": 0.25,
            "points_vs_baseline": 5.0,
            "beats_baseline": True,
            "n_hours": 48,
        }

    def test_a_losing_region(self):
        assert skill_payload(25.0, self.SERIES) == {
            "model_mape": 25.0,
            "baseline_mape": 20.0,
            "baseline": "seasonal-naive lag 24h",
            "skill": -0.25,
            "points_vs_baseline": -5.0,
            "beats_baseline": False,
            "n_hours": 48,
        }

    def test_matching_the_baseline_is_not_beating_it(self):
        """``skill > 0``, strictly. A tie does not clear the bar.

        The module exists because a model that merely matches "yesterday, same
        hour" is adding nothing. Relaxing the comparison to ``>=`` would
        publish ``beats_baseline: True`` for a model with exactly zero skill —
        the precise claim this file was written to make impossible.
        """
        baseline = seasonal_naive_mape(self.SERIES)
        tie = skill_payload(baseline, self.SERIES)

        assert tie["skill"] == 0.0
        assert tie["points_vs_baseline"] == 0.0
        assert tie["beats_baseline"] is False, "matching the baseline is not beating it"

    def test_the_published_precision_is_pinned(self):
        """3 decimals on the error figures, 4 on the skill score.

        The fixtures above use round numbers so the win/lose semantics read
        clearly — which leaves the rounding itself invisible, because 15.0 and
        20.0 survive any precision. This series produces long decimals on
        every field so a drifted `round()` shows up.
        """
        rng = np.arange(48, dtype=float)
        series = 100.0 + 10.0 * np.sin(2 * np.pi * rng / 24.0) + rng * 0.37

        payload = skill_payload(5.4321, series)

        assert payload["model_mape"] == 5.432
        assert payload["baseline_mape"] == 7.866
        assert payload["points_vs_baseline"] == 2.434
        assert payload["skill"] == 0.3095

    def test_an_unmeasurable_region_nulls_every_derived_field(self):
        """Not zero, not False — None, on all four derived fields at once.

        A `beats_baseline` of False would read as "measured, and it lost",
        which is the confusion that let SEC serve a worse-than-nothing
        forecast unnoticed. `n_hours` still reports what was seen.
        """
        assert skill_payload(15.0, np.arange(10, dtype=float)) == {
            "model_mape": 15.0,
            "baseline_mape": None,
            "baseline": "seasonal-naive lag 24h",
            "skill": None,
            "points_vs_baseline": None,
            "beats_baseline": None,
            "n_hours": 10,
        }


class TestMape:
    def test_hours_below_one_mw_still_count(self):
        """The filter is ``actual > 0``, not ``> 1``.

        Tightened to ``> 1`` it silently drops sub-1-MW hours. GridPulse
        serves BAs at that scale — SPA's median demand is ~24 MW and the
        module docstring names SEC as a ~300 MW co-op — so a filter that
        quietly excludes small hours would flatter exactly the regions whose
        skill matters most.
        """
        actual = np.array([0.5, 100.0])
        predicted = np.array([1.0, 100.0])

        # The 0.5 MW hour is 100% wrong, the 100 MW hour exact -> mean 50%.
        assert mape(actual, predicted) == pytest.approx(50.0)

    def test_ignores_nonpositive_actuals(self):
        assert mape(np.array([100.0, 0.0, -5.0]), np.array([110.0, 5.0, 5.0])) == pytest.approx(
            10.0
        )


class TestSubstitutionPolicy:
    """When may a region's forecast be replaced by the baseline?

    This is the only rule in the codebase that changes what a user is served
    based on a measurement, so its failure modes are asymmetric: substituting
    when we shouldn't is worse than not substituting when we should.
    """

    def _block(self, points, hours=24 * 30):
        return {"points_vs_baseline": points, "n_hours": hours}

    def test_substitutes_on_a_clear_deficit(self):
        """SEC's shape: ~4 points down at every measured lead."""
        ok, reason = should_serve_baseline(self._block(-3.98))
        assert ok is True
        assert "3.98" in reason and "seasonal-naive" in reason

    def test_leaves_marginal_regions_alone(self):
        """Eight regions sat within ~1 point of the line. Churning served
        forecasts for a fractional, sample-noise deficit buys nothing."""
        for points in (-1.02, -0.94, -0.4, -0.03):
            ok, _ = should_serve_baseline(self._block(points))
            assert ok is False, points

    def test_never_substitutes_without_a_measurement(self):
        """A broken skill pipeline must degrade to today's behaviour, not
        swap every region onto a naive forecast."""
        assert should_serve_baseline(None)[0] is False
        assert should_serve_baseline({})[0] is False
        assert should_serve_baseline(self._block(None))[0] is False

    def test_requires_enough_hours(self):
        """A deficit measured over two days is not evidence."""
        assert should_serve_baseline(self._block(-9.0, hours=48))[0] is False
        assert should_serve_baseline(self._block(-9.0, hours=24 * 7))[0] is True

    def test_a_winning_model_is_never_substituted(self):
        assert should_serve_baseline(self._block(+3.5))[0] is False

    def test_a_deficit_of_exactly_the_threshold_substitutes(self):
        """ "At least this many points" includes the boundary.

        `BASELINE_SUBSTITUTION_MIN_POINTS` documents "must lose ... by at
        least this many error points", so a deficit of exactly 2.00 qualifies.
        The comparison is `points > -MIN` -> keep, and relaxing it to `>=`
        flips the boundary case to keeping the model — with the suite green,
        because every existing case sits comfortably on one side or the other.

        This is the only rule in the codebase that changes what a user is
        served, so which side of the line the boundary falls on is a policy
        decision and belongs in a test rather than in the shape of an
        inequality.
        """
        exactly = should_serve_baseline(self._block(-BASELINE_SUBSTITUTION_MIN_POINTS))
        assert exactly[0] is True, "a deficit of exactly the threshold is 'at least' the threshold"

        just_inside = should_serve_baseline(self._block(-BASELINE_SUBSTITUTION_MIN_POINTS + 0.01))
        assert just_inside[0] is False

    def test_a_block_with_no_hours_reports_zero_not_one(self):
        """`n_hours` missing falls back to 0, and the reason says so.

        `... or 0` mutated to `... or 1` still refuses to substitute, so the
        decision is unchanged and only the published reason differs — from
        "only 0h measured" to "only 1h measured". That reason is surfaced to
        explain why a region was left alone, and inventing an hour of
        measurement that never happened is a small lie in a field whose whole
        job is honesty.
        """
        _, reason = should_serve_baseline({"points_vs_baseline": -9.0})

        assert reason == "only 0h measured, need 168h"


class TestForwardBaseline:
    def test_every_lead_reads_the_last_observed_day(self):
        """Stated as the PROPERTY, not as indices. The first version of this
        test asserted index arithmetic and therefore agreed with a buggy
        implementation that walked an extra day back per block — lead 25 read
        72h before its target instead of 48h, and it shipped.

        The invariant: the source is the same clock hour, a whole number of
        days before the target, always inside the final 24h of history."""
        hist = np.arange(96, dtype=float)  # values == index, so value == position
        horizon = 72
        out = seasonal_naive_forecast(hist, horizon)
        last = 95

        for h in range(1, horizon + 1):
            src = int(out[h - 1])  # the position it read
            gap = h - (src - last)  # hours between target and source
            assert gap % 24 == 0, f"lead {h}: source is not a whole day back ({gap}h)"
            assert gap == 24 * -(-h // 24), f"lead {h}: read {gap}h back, not the nearest day"
            assert last - 23 <= src <= last, f"lead {h}: read outside the last observed day"

    def test_the_daily_profile_repeats_across_horizon_days(self):
        """What a reader sees, and a consequence of the invariant: day 2 of
        the forecast has the same shape as day 1, because both replay the
        last observed day."""
        hist = np.arange(96, dtype=float)
        out = seasonal_naive_forecast(hist, 72)
        assert list(out[0:24]) == list(out[24:48]) == list(out[48:72])

    def test_repeats_the_daily_shape(self):
        y = _daily(n_days=5)
        out = seasonal_naive_forecast(y, 24)
        assert np.allclose(out, y[-24:])

    def test_history_shorter_than_a_day_yields_nothing(self):
        """The caller must keep the model rather than serve a stub."""
        assert seasonal_naive_forecast(np.arange(10, dtype=float), 24).size == 0

    def test_a_gap_in_the_last_day_steps_back_a_further_day(self):
        """One missing hour must not disable substitution for a whole region —
        the same clock hour a further day back is still a real observation."""
        hist = np.arange(96, dtype=float)
        hist[72] = np.nan  # first hour of the last observed day
        out = seasonal_naive_forecast(hist, 24)
        assert np.isfinite(out).all()
        assert out[0] == hist[72 - 24]  # stepped back exactly one further day

    # -- edges the property test above cannot reach ------------------------

    def test_exactly_one_day_of_history_is_enough(self):
        """The guard is ``size < lag_h``, so 24 hours qualifies and 23 do not.

        ``test_history_shorter_than_a_day_yields_nothing`` uses 10 hours,
        which is nowhere near the line. Relaxing the guard to ``<=`` would
        refuse a region that has exactly one clean day — the most likely
        moment for a newly-onboarded BA to want a baseline.
        """
        assert seasonal_naive_forecast(np.arange(24, dtype=float), 3).tolist() == [0.0, 1.0, 2.0]
        assert seasonal_naive_forecast(np.arange(23, dtype=float), 3).size == 0

    def test_the_stepback_gives_up_rather_than_reading_off_the_start(self):
        """When no day has that clock hour, return nothing — do not wrap.

        The stepback walks back whole days looking for a real observation. If
        it runs past the start of the series the function must return an empty
        array so the caller keeps the model. The out-of-range guard is
        ``idx < 0 or idx >= y.size``; weakened to ``and`` it can never fire,
        and a negative index silently wraps to the END of the history — the
        baseline would then serve the newest hours as if they were the oldest,
        which is both wrong and undetectable downstream.
        """
        # One clean day, but the 06:00 slot is missing: there is no earlier
        # day to fall back to.
        hist = np.arange(24, dtype=float)
        hist[5] = np.nan

        assert seasonal_naive_forecast(hist, 24).size == 0, "give up, never wrap"

    def test_a_missing_hour_in_every_day_gives_up(self):
        """Same guard, reached via the ``idx >= 0`` loop bound.

        Index 0 is a legitimate source, so the stepback must test it before
        stopping. With the bound at ``idx > 0`` the loop exits one step early
        and serves the NaN it was trying to skip.
        """
        hist = np.arange(48, dtype=float)
        hist[24] = np.nan  # that clock hour in the last day
        hist[0] = np.nan  # ...and in the only earlier day

        assert seasonal_naive_forecast(hist, 24).size == 0

    def test_the_horizon_length_is_exactly_what_was_asked_for(self):
        """A short array would leave a caller reading uninitialised memory —
        ``np.empty`` is not zero-filled, so the values would be arbitrary
        rather than obviously wrong.
        """
        hist = np.arange(96, dtype=float)
        for horizon in (1, 5, 24, 48, 72):
            assert seasonal_naive_forecast(hist, horizon).size == horizon


class TestServingIntegration:
    """The substitution as the scoring job actually applies it.

    This is the only code path in the system that changes what a user is
    shown based on a measurement, so the tests pin the disclosure as hard as
    the arithmetic: a substituted series that reads as a model forecast would
    reproduce the original failure with extra steps.
    """

    @staticmethod
    def _frame(hours=24 * 10, base=300.0):
        ts = pd.date_range("2026-07-01", periods=hours, freq="h", tz="UTC")
        t = np.arange(hours)
        return pd.DataFrame({"timestamp": ts, "demand_mw": base + 100 * np.sin(2 * np.pi * t / 24)})

    def _run(self, monkeypatch, *, flag, model_mape):
        from jobs import phases

        monkeypatch.setattr("config.feature_enabled", lambda name: flag)
        monkeypatch.setattr(
            "data.redis_client.redis_get",
            lambda key: {"models": {"ensemble": {"24h": {"rolling_mape_7d": model_mape}}}},
        )
        monkeypatch.setattr("data.redis_client.redis_key", lambda k: k)
        return phases._baseline_substitution("SEC", self._frame(), 48)

    def test_flag_off_never_substitutes(self, monkeypatch):
        """Ships dark. Flag-off must be byte-identical to today."""
        assert self._run(monkeypatch, flag=False, model_mape=99.0) is None

    def test_substitutes_when_the_model_loses_badly(self, monkeypatch):
        out = self._run(monkeypatch, flag=True, model_mape=20.0)
        assert out is not None
        values, block = out
        assert len(values) == 48
        assert np.isfinite(values).all()
        assert block["points_vs_baseline"] < -2.0
        assert "seasonal-naive" in block["decision"]

    def test_keeps_the_model_when_it_is_competitive(self, monkeypatch):
        """A clean daily series makes the baseline near-perfect, so only a
        genuinely terrible model loses to it. 1.0% must not substitute."""
        assert self._run(monkeypatch, flag=True, model_mape=1.0) is None

    def test_a_missing_drift_signal_keeps_the_model(self, monkeypatch):
        from jobs import phases

        monkeypatch.setattr("config.feature_enabled", lambda name: True)
        monkeypatch.setattr("data.redis_client.redis_get", lambda key: None)
        monkeypatch.setattr("data.redis_client.redis_key", lambda k: k)
        assert phases._baseline_substitution("SEC", self._frame(), 48) is None

    def test_never_raises_into_the_forecast_path(self, monkeypatch):
        """A measurement bug must not cost a region its forecast."""
        from jobs import phases

        monkeypatch.setattr("config.feature_enabled", lambda name: True)
        monkeypatch.setattr(
            "data.redis_client.redis_get", lambda key: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert phases._baseline_substitution("SEC", self._frame(), 48) is None
