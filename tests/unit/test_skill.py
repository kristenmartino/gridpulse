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
import pytest

from models.skill import (
    SEASONAL_NAIVE_LAG_H,
    mape,
    seasonal_naive_mape,
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
