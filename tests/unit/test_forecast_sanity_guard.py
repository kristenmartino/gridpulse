"""Unit tests for the #296 long-horizon forecast sanity guard.

Covers the shared checker (``models.evaluation.check_long_horizon_sanity``),
the serve-time per-horizon wrapper (``jobs.phases._horizon_guard_for_series``),
and the ARIMA fit-time guard (``models.arima_model._apply_long_horizon_guard``).

Root cause being guarded: a doubly-integrated SARIMAX (d=1 AND D=1)
extrapolates the training window's local weather-driven trend as a permanent
linear trend — SC/PSCO decayed through 0 MW and BPAT grew ~2x across the
30-day view while every AR/MA characteristic root sat on the stationary side.
The synthetic series below reproduce those trajectory shapes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config

H = 720


def _daily_cycle(n: int, base: float = 10_000.0, amp: float = 2_000.0) -> np.ndarray:
    """Flat demand series with a daily cycle — the healthy shape."""
    return base + amp * np.sin(2 * np.pi * np.arange(n) / 24)


@pytest.fixture
def recent():
    """Four weeks of healthy recent demand: band ≈ [8k, 12k]."""
    return _daily_cycle(28 * 24)


class TestCheckLongHorizonSanity:
    def test_healthy_cycle_passes(self, recent):
        from models.evaluation import check_long_horizon_sanity

        assert check_long_horizon_sanity(_daily_cycle(H), recent) is None

    def test_sc_shaped_decay_flagged_below_band(self, recent):
        """The SC/PSCO signature: linear decay from in-band through zero."""
        from models.evaluation import check_long_horizon_sanity

        decay = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)  # ends ~ −3k
        assert check_long_horizon_sanity(decay, recent) == "below_recent_band"

    def test_bpat_shaped_growth_flagged_above_band(self, recent):
        """The BPAT signature: linear growth to ~2x the recent envelope."""
        from models.evaluation import check_long_horizon_sanity

        growth = _daily_cycle(H) + np.linspace(0.0, 10_000.0, H)  # peaks ~22k vs 12k recent max
        assert check_long_horizon_sanity(growth, recent) == "above_recent_band"

    def test_in_band_sustained_drift_flagged_on_long_series(self, recent):
        """A one-directional shift that stays inside the band is still drift
        on a 30-day series: |last-day − first-day| > 40% of recent mean."""
        from models.evaluation import check_long_horizon_sanity

        # 10k → 14.5k: max 16.5k < 1.6*12k=19.2k ceiling, min 8k > 4k floor,
        # daily-mean shift 4.5k > 0.40 * 10k mean.
        drift = _daily_cycle(H) + np.linspace(0.0, 4_500.0, H)
        assert check_long_horizon_sanity(drift, recent) == "sustained_drift"

    def test_drift_check_skipped_on_short_series(self, recent):
        """The same shift across a 7-day slice is a legitimate weather swing —
        the drift check only engages at ≥ LONG_HORIZON_GUARD_DRIFT_MIN_LEN."""
        from models.evaluation import check_long_horizon_sanity

        assert config.LONG_HORIZON_GUARD_DRIFT_MIN_LEN > 168
        drift = _daily_cycle(168) + np.linspace(0.0, 4_500.0, 168)
        assert check_long_horizon_sanity(drift, recent) is None

    def test_non_finite_forecast_flagged(self, recent):
        from models.evaluation import check_long_horizon_sanity

        bad = _daily_cycle(H)
        bad[100] = np.nan
        assert check_long_horizon_sanity(bad, recent) == "non_finite"

    def test_empty_forecast_flagged(self, recent):
        from models.evaluation import check_long_horizon_sanity

        assert check_long_horizon_sanity(np.array([]), recent) == "non_finite"

    def test_too_little_history_returns_none(self):
        """< 1 week of recent demand → no band to judge against; don't guess."""
        from models.evaluation import check_long_horizon_sanity

        short_recent = _daily_cycle(100)
        decay = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        assert check_long_horizon_sanity(decay, short_recent) is None

    def test_junk_recent_rows_ignored(self, recent):
        """NaN / zero / negative rows in recent demand must not poison the
        band (a zero row would zero the floor and disarm the check)."""
        from models.evaluation import check_long_horizon_sanity

        junk = recent.copy()
        junk[:24] = 0.0
        junk[24:30] = np.nan
        junk[30] = -500.0
        decay = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        assert check_long_horizon_sanity(decay, junk) == "below_recent_band"

    def test_zero_recent_rows_do_not_disarm_floor(self, recent):
        """#296 mutation pin (verification finding): dropping the ``r > 0``
        half of the recent-demand filter must fail this test. Uses
        serve-path-realistic inputs — zero rows in recent demand (real
        pre-fix cache states) and a forecast already clipped at 0 by the
        serve floor, so its min is exactly 0.0. Without the positivity
        filter, the poisoned floor is 0.5 x 0 = 0 and the collapse serves
        unguarded."""
        from models.evaluation import check_long_horizon_sanity

        junk = recent.copy()
        junk[:24] = 0.0  # zero rows only — no negative sentinel to hide behind
        clipped_decay = np.maximum(_daily_cycle(H) - np.linspace(0.0, 13_000.0, H), 0.0)
        assert check_long_horizon_sanity(clipped_decay, junk) == "below_recent_band"

    def test_flat_zero_collapse_flagged_despite_zero_poisoned_recent(self, recent):
        """A total collapse to 0 MW must flag even when recent demand
        carries zero rows — this shape has no drift signature (flat), so
        the positive-min band floor is the only detector."""
        from models.evaluation import check_long_horizon_sanity

        junk = recent.copy()
        junk[:24] = 0.0
        assert check_long_horizon_sanity(np.zeros(H), junk) == "below_recent_band"

    def test_realistic_seasonal_ramp_not_flagged_as_drift(self, recent):
        """#296 verification MEDIUM regression: a decelerating seasonal ramp
        with weekday/weekend texture and synoptic noise — the exact shape
        the #283 weather-normal tail produces across spring→summer — must
        NOT be flagged, even when the first→last daily-mean shift exceeds
        the 40% threshold. (Perfect forecasts built from real EIA demand
        across the 2026 spring ramp false-flagged 21/51 BAs before the
        linearity gate.)"""
        from models.evaluation import check_long_horizon_sanity

        rng = np.random.default_rng(7)
        days = np.arange(30)
        ramp = 10_000.0 + 4_800.0 * np.sqrt(days / 29.0)  # saturating, not linear
        weekly = np.where((days % 7) >= 5, -600.0, 200.0)  # weekend dips
        noise = rng.normal(0.0, 300.0, size=30)  # synoptic variation
        daily_means = ramp + weekly + noise
        series = np.repeat(daily_means, 24) + np.tile(
            2_000.0 * np.sin(2 * np.pi * np.arange(24) / 24), 30
        )
        # Precondition: the shift alone would have tripped the old check.
        shift = abs(daily_means[-1] - daily_means[0])
        assert shift > config.LONG_HORIZON_GUARD_DRIFT_FRAC * recent.mean()
        assert check_long_horizon_sanity(series, recent) is None

    # ------------------------------------------------------------------
    # The tests above pin the SHAPES this guard exists to catch, using
    # realistic trajectories — which is the right way to write them, and is
    # why the #296 regression cannot come back.
    #
    # What they do not pin is where each band ENDS. Every fixture above sits
    # comfortably inside or outside its threshold, so all four comparisons
    # could be relaxed by one `=` with the suite green (18 survivors,
    # docs/TEST_QUALITY.md). This guard decides whether a forecast reaches
    # the serve path, so its edges are a decision, not an implementation
    # detail.
    #
    # A flat 1,000 MW history makes min == max == mean, so each threshold can
    # be hit exactly without the others moving.
    # ------------------------------------------------------------------

    FLAT = 1_000.0

    @pytest.fixture
    def flat_recent(self):
        """Exactly the minimum history, perfectly flat: min = max = mean."""
        return np.full(config.LONG_HORIZON_GUARD_MIN_RECENT_ROWS, self.FLAT)

    def test_the_floor_band_is_inclusive_at_its_edge(self, flat_recent):
        """A forecast bottoming at exactly the floor fraction is in-band.

        `f.min() < FLOOR_FRAC * r.min()` — relaxed to `<=`, a forecast sitting
        precisely on the boundary is rejected and the region loses its
        forecast for that horizon.
        """
        from models.evaluation import check_long_horizon_sanity

        floor = config.LONG_HORIZON_GUARD_FLOOR_FRAC * self.FLAT
        on_the_line = np.full(48, self.FLAT)
        on_the_line[0] = floor
        assert check_long_horizon_sanity(on_the_line, flat_recent) is None

        below = on_the_line.copy()
        below[0] = floor - 0.001
        assert check_long_horizon_sanity(below, flat_recent) == "below_recent_band"

    def test_the_ceiling_band_is_inclusive_at_its_edge(self, flat_recent):
        """Same edge on the other side: `f.max() > CEIL_FRAC * r.max()`."""
        from models.evaluation import check_long_horizon_sanity

        ceiling = config.LONG_HORIZON_GUARD_CEIL_FRAC * self.FLAT
        on_the_line = np.full(48, self.FLAT)
        on_the_line[0] = ceiling
        assert check_long_horizon_sanity(on_the_line, flat_recent) is None

        above = on_the_line.copy()
        above[0] = ceiling + 0.001
        assert check_long_horizon_sanity(above, flat_recent) == "above_recent_band"

    def test_exactly_one_week_of_history_is_enough_to_judge(self, flat_recent):
        """`r.size < MIN_RECENT_ROWS` — 168 rows qualifies, 167 does not.

        Both outcomes are `None` for a healthy forecast, which is why this
        needs a *degenerate* one to be visible at all: with enough history the
        guard fires, with one row fewer it declines to judge. Relaxing the
        comparison to `<=` silently disarms the guard for any region with
        exactly a week of data — a newly-onboarded BA, or one just back from
        an outage.
        """
        from models.evaluation import check_long_horizon_sanity

        collapsed = np.full(48, self.FLAT * 0.1)

        assert check_long_horizon_sanity(collapsed, flat_recent) == "below_recent_band"

        one_row_short = np.full(config.LONG_HORIZON_GUARD_MIN_RECENT_ROWS - 1, self.FLAT)
        assert check_long_horizon_sanity(collapsed, one_row_short) is None

    def test_the_drift_check_engages_at_exactly_its_minimum_length(self, flat_recent):
        """`f.size >= DRIFT_MIN_LEN` — 360 hours engages it, 359 does not.

        `test_drift_check_skipped_on_short_series` above uses a 72-hour
        series, far from the line. Tightening this to `>` moves the guard's
        reach by a day at exactly the length where long-horizon forecasts
        start being judged.
        """
        from models.evaluation import check_long_horizon_sanity

        n = config.LONG_HORIZON_GUARD_DRIFT_MIN_LEN
        ramp = np.linspace(self.FLAT, self.FLAT * 1.5, n)  # in-band, perfectly linear

        assert check_long_horizon_sanity(ramp, flat_recent) == "sustained_drift"

        just_short = np.linspace(self.FLAT, self.FLAT * 1.5, n - 1)
        assert check_long_horizon_sanity(just_short, flat_recent) is None

    def test_the_drift_window_spans_every_day_including_the_last(self, flat_recent):
        """`n_days = f.size // 24` and `shift = daily[-1] - daily[0]`.

        A ramp sized so the **final day** is what carries it over the
        threshold: across all 15 days the first→last shift is 420 MW against a
        400 MW limit, but across only the first 14 it is 390 and passes.

        Analysing a day fewer (`// 25`) or comparing to the second-to-last day
        (`daily[-2]`) both drop that day and let the drift through. Every other
        drift fixture in this class ramps hard enough that losing one day
        changes nothing, so neither mutation was visible.
        """
        from models.evaluation import check_long_horizon_sanity

        n = config.LONG_HORIZON_GUARD_DRIFT_MIN_LEN
        n_days = n // 24
        rise = 420.0  # > 400 across 15 days, < 400 across 14

        series = np.repeat(np.linspace(self.FLAT, self.FLAT + rise, n_days), 24)
        daily = series.reshape(n_days, 24).mean(axis=1)

        # Preconditions: the whole window trips the threshold, one day less does not.
        limit = config.LONG_HORIZON_GUARD_DRIFT_FRAC * flat_recent.mean()
        assert abs(daily[-1] - daily[0]) > limit
        assert abs(daily[-2] - daily[0]) < limit

        assert check_long_horizon_sanity(series, flat_recent) == "sustained_drift"

    def test_a_partial_trailing_day_is_dropped_not_reshaped(self, flat_recent):
        """`f[: n_days * 24]` — the slice exists to discard a partial last day.

        Every other drift fixture is an exact multiple of 24, so the truncation
        is a no-op in all of them and could be widened to any larger multiplier
        with the suite green. A horizon that ends mid-day — which is what a
        16-day Open-Meteo window trimmed to the last settled hour looks like —
        would then reshape a short array and raise inside the guard.
        """
        from models.evaluation import check_long_horizon_sanity

        n_days = config.LONG_HORIZON_GUARD_DRIFT_MIN_LEN // 24
        series = np.repeat(np.linspace(self.FLAT, self.FLAT + 420.0, n_days), 24)
        ragged = np.append(series, np.full(7, series[-1]))  # 15 days + 7 hours

        assert ragged.size % 24 != 0, "the point of the fixture is the partial day"
        assert check_long_horizon_sanity(ragged, flat_recent) == "sustained_drift"

    def test_a_one_megawatt_history_is_still_history(self, flat_recent):
        """The history filter is `r > 0`, not `r > 1`.

        Tightened by one, a BA whose demand sits at 1 MW has its entire
        history discarded and the guard declines to judge it. SPA's median
        demand is ~24 MW, so single-digit readings are not hypothetical here.
        """
        from models.evaluation import check_long_horizon_sanity

        tiny = np.full(config.LONG_HORIZON_GUARD_MIN_RECENT_ROWS, 1.0)

        assert check_long_horizon_sanity(np.full(48, 0.4), tiny) == "below_recent_band"


class TestHorizonGuardForSeries:
    def test_all_horizons_pass_returns_none(self, recent):
        from jobs.phases import _horizon_guard_for_series

        assert _horizon_guard_for_series(_daily_cycle(H), recent) is None

    def test_late_onset_decay_keeps_short_horizons(self, recent):
        """Degeneracy that only bites past day 7: 24h/168h stay served,
        720h is flagged — the #227 by-horizon philosophy."""
        from jobs.phases import _horizon_guard_for_series

        series = _daily_cycle(H)
        series[168:] -= np.linspace(0.0, 13_000.0, H - 168)
        guard = _horizon_guard_for_series(series, recent)
        assert guard is not None
        assert guard["max_ok_horizon"] == 168
        assert guard["flagged_horizon"] == 720
        assert guard["reason"] == "below_recent_band"

    def test_immediate_collapse_flags_all_horizons(self, recent):
        from jobs.phases import _horizon_guard_for_series

        series = np.full(H, 100.0)  # far below 0.5 * 8k floor from hour 0
        guard = _horizon_guard_for_series(series, recent)
        assert guard is not None
        assert guard["max_ok_horizon"] == 0
        assert guard["flagged_horizon"] == 24

    def test_short_series_only_checks_covered_horizons(self, recent):
        """A 168-length series never reaches the 720h check; if its covered
        slices pass, there is no guard entry."""
        from jobs.phases import _horizon_guard_for_series

        assert _horizon_guard_for_series(_daily_cycle(168), recent) is None


class TestApplyLongHorizonGuardFitTime:
    """``_apply_long_horizon_guard`` runs inside ``train_arima`` after the
    fit. It must never raise, and on a degenerate 720h trajectory it refits
    with the safe DEFAULT orders."""

    def _y(self):
        return _daily_cycle(28 * 24)

    def test_healthy_fit_passes_without_refit(self):
        from models.arima_model import _apply_long_horizon_guard

        fitted = MagicMock()
        fitted.forecast.return_value = _daily_cycle(H)
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_sarimax:
            out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
                fitted, self._y(), None, (2, 0, 1), (1, 1, 0, 24)
            )
            mock_sarimax.assert_not_called()
        assert out_fitted is fitted
        assert (order, seasonal) == ((2, 0, 1), (1, 1, 0, 24))
        assert ok is True

    def test_degenerate_fit_refits_with_default_and_heals(self):
        from models.arima_model import (
            DEFAULT_ORDER,
            DEFAULT_SEASONAL_ORDER,
            _apply_long_horizon_guard,
        )

        fitted = MagicMock()
        fitted.forecast.return_value = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        healed = MagicMock()
        healed.forecast.return_value = _daily_cycle(H)
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_sarimax:
            mock_sarimax.return_value.fit.return_value = healed
            out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
                fitted, self._y(), None, (2, 1, 0), (1, 1, 0, 24)
            )
        assert out_fitted is healed
        assert order == DEFAULT_ORDER
        assert seasonal == DEFAULT_SEASONAL_ORDER
        assert ok is True

    def test_refit_still_degenerate_keeps_original_and_flags(self):
        from models.arima_model import _apply_long_horizon_guard

        decay = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        fitted = MagicMock()
        fitted.forecast.return_value = decay
        still_bad = MagicMock()
        still_bad.forecast.return_value = decay
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_sarimax:
            mock_sarimax.return_value.fit.return_value = still_bad
            out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
                fitted, self._y(), None, (2, 1, 0), (1, 1, 0, 24)
            )
        assert out_fitted is fitted
        assert (order, seasonal) == ((2, 1, 0), (1, 1, 0, 24))
        assert ok is False

    def test_already_default_degenerate_skips_refit(self):
        from models.arima_model import (
            DEFAULT_ORDER,
            DEFAULT_SEASONAL_ORDER,
            _apply_long_horizon_guard,
        )

        fitted = MagicMock()
        fitted.forecast.return_value = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_sarimax:
            out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
                fitted, self._y(), None, DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER
            )
            mock_sarimax.assert_not_called()
        assert out_fitted is fitted
        assert ok is False

    def test_short_history_returns_indeterminate_not_verified(self):
        """#296 verification finding: with fewer than a week of valid
        training rows the check cannot run — the payload must record
        long_horizon_ok=None ('check could not run'), NOT True ('720h
        check passed'), and no refit may be attempted."""
        from models.arima_model import _apply_long_horizon_guard

        fitted = MagicMock()
        fitted.forecast.return_value = _daily_cycle(H) - np.linspace(0.0, 13_000.0, H)
        with patch("statsmodels.tsa.statespace.sarimax.SARIMAX") as mock_sarimax:
            out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
                fitted, _daily_cycle(96), None, (2, 1, 0), (1, 1, 0, 24)
            )
            mock_sarimax.assert_not_called()
        assert out_fitted is fitted
        assert (order, seasonal) == ((2, 1, 0), (1, 1, 0, 24))
        assert ok is None
        fitted.forecast.assert_not_called()

    def test_check_failure_returns_unknown_not_raise(self):
        """A guard must not be able to take down training."""
        from models.arima_model import _apply_long_horizon_guard

        fitted = MagicMock()
        fitted.forecast.side_effect = ValueError("synthetic forecast failure")
        out_fitted, order, seasonal, ok = _apply_long_horizon_guard(
            fitted, self._y(), None, (2, 0, 1), (1, 1, 0, 24)
        )
        assert out_fitted is fitted
        assert ok is None

    def test_exog_repeats_last_training_day(self):
        """With exog present, the check forecast must receive a full
        720-row exog built from the last training day."""
        from models.arima_model import _apply_long_horizon_guard

        fitted = MagicMock()
        fitted.forecast.return_value = _daily_cycle(H)
        exog = np.arange(240 * 5, dtype=float).reshape(240, 5)
        _apply_long_horizon_guard(fitted, self._y(), exog, (2, 0, 1), (1, 1, 0, 24))
        kwargs = fitted.forecast.call_args.kwargs
        assert kwargs["steps"] == H
        horizon_exog = kwargs["exog"]
        assert horizon_exog.shape == (H, 5)
        # First 24 rows repeat the last training day verbatim.
        np.testing.assert_array_equal(horizon_exog[:24], exog[-24:])
        np.testing.assert_array_equal(horizon_exog[24:48], exog[-24:])
