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
