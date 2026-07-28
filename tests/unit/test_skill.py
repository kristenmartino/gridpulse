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


class TestMape:
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


class TestForwardBaseline:
    def test_each_lead_reads_the_most_recent_observed_same_hour(self):
        """Lead 1-24 reads yesterday, 25-48 two days ago — the same predictor
        the skill score measures, so what is served is what was measured."""
        hist = np.arange(96, dtype=float)  # 4 days, values == index
        out = seasonal_naive_forecast(hist, 48)
        assert out[0] == hist[96 - 24]  # lead 1  -> yesterday, same hour
        assert out[23] == hist[96 - 1]  # lead 24 -> yesterday, same hour
        assert out[24] == hist[96 - 48]  # lead 25 -> two days ago
        assert out[47] == hist[96 - 25]  # lead 48 -> two days ago

    def test_repeats_the_daily_shape(self):
        y = _daily(n_days=5)
        out = seasonal_naive_forecast(y, 24)
        assert np.allclose(out, y[-24:])

    def test_history_shorter_than_a_day_yields_nothing(self):
        """The caller must keep the model rather than serve a stub."""
        assert seasonal_naive_forecast(np.arange(10, dtype=float), 24).size == 0

    def test_a_nan_source_hour_falls_back_to_the_nearest_observed_day(self):
        hist = np.arange(96, dtype=float)
        hist[96 - 48] = np.nan  # the lead-25 source hour
        out = seasonal_naive_forecast(hist, 48)
        assert np.isfinite(out).all()
        assert out[24] == hist[96 - 24]  # fell back one day forward


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
