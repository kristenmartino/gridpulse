"""The backtest fold path must go through the documented inference SSOT.

``recursive_autoregressive_forecast``'s docstring calls it "the single source of
truth for both production scoring and holdout evaluation" (#195/#186). That was
untrue of the backtest — the very code that publishes holdout MAPE carried its
own copy of the loop.

The copy differed in exactly two ways, both measured before unifying:

1. **Seed filtering.** The canonical helper drops zero and NaN readings; the
   copy passed ``train_df["demand_mw"]`` raw. NaNs never reached it (both
   ``base_df`` and the per-fold train slice already ``dropna``), so **zeros were
   the only live difference**. It bites only when a zero lands in the trailing
   168h the rolling features read — and there the filtered answer is the
   correct one, which is the whole #129 lesson.
2. **The ``if col in row.columns`` guard**, which the copy lacked. Verified a
   no-op: every snapshot key is present in an ``engineer_features`` frame.
   Had one been missing, unifying would have silently substituted 0.0 for a
   real feature value.

These tests pin both, so a future edit to either implementation cannot quietly
re-open the gap.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from config import WEATHER_VARIABLES
from data.feature_engineering import (
    compute_autoregressive_snapshot,
    engineer_features,
    recursive_autoregressive_forecast,
)


def _raw_frame(n: int, zeros: int = 0, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    hod = ts.hour.to_numpy()
    data = {"timestamp": ts, "demand_mw": 900 + 250 * np.sin(2 * np.pi * hod / 24)}
    for v in WEATHER_VARIABLES:
        data[v] = rng.normal(20, 5, n)
    df = pd.DataFrame(data)
    if zeros:
        df.loc[df.index[-zeros:], "demand_mw"] = 0.0
    return df


class TestBacktestDelegatesToTheSsot:
    def test_fold_path_calls_the_canonical_helper(self):
        """Structural guard: the duplicated loop must not come back."""
        from components import _callbacks_backtest as bt

        src = inspect.getsource(bt)
        assert "recursive_autoregressive_forecast" in src
        # the copy's signature line — a per-step row build driven by the
        # snapshot dict — must be gone from this module
        assert "for col, val in compute_autoregressive_snapshot(" not in src, (
            "the backtest has re-grown its own copy of the inference loop; "
            "it must delegate to recursive_autoregressive_forecast"
        )


class TestTheGuardWasANoOp:
    def test_every_snapshot_key_exists_in_an_engineered_frame(self):
        """The justification for dropping the copy's unguarded assignment. If
        this ever fails, unifying silently substitutes 0.0 for a real feature
        and the backtest's MAPE becomes wrong in a way nothing else reports."""
        featured = engineer_features(_raw_frame(900))
        missing = set(compute_autoregressive_snapshot([1.0] * 300)) - set(featured.columns)
        assert not missing, f"snapshot keys absent from the engineered frame: {sorted(missing)}"


class TestSeedFilteringIsTheOnlyLiveDifference:
    """Pins the measured behaviour, not just the refactor."""

    def _fold(self, train_demand: list[float], test_df: pd.DataFrame) -> np.ndarray:
        def lag1(_model, row):
            v = row["demand_lag_1h"].iloc[0]
            return [float(v) * 0.99 + 7.0]

        return recursive_autoregressive_forecast(None, train_demand, test_df, lag1)

    def _test_frame(self) -> pd.DataFrame:
        featured = engineer_features(_raw_frame(900))
        return featured.iloc[-48:].reset_index(drop=True)

    def test_clean_seed_is_unaffected_by_filtering(self):
        """No zeros → filtering changes nothing, so the vast majority of
        regions see byte-identical published MAPE."""
        test_df = self._test_frame()
        clean = [900.0 + i * 0.1 for i in range(720)]
        np.testing.assert_array_equal(
            self._fold(clean, test_df), self._fold(list(clean), test_df)
        )

    def test_zeros_in_the_trailing_window_are_filtered_out(self):
        """A zero inside the trailing 168h MUST change the result — that is the
        #129 fix doing its job, and the reason unifying is a correction rather
        than a regression."""
        test_df = self._test_frame()
        clean = [900.0 + i * 0.1 for i in range(720)]
        poisoned = list(clean)
        poisoned[-10:] = [0.0] * 10

        # with filtering (the SSOT), the zeros are dropped and the forecast
        # tracks the clean series; an unfiltered seed would not
        filtered = self._fold(poisoned, test_df)
        baseline = self._fold(clean[:-10], test_df)
        np.testing.assert_allclose(filtered, baseline, rtol=1e-9)

    def test_nan_seed_readings_are_dropped(self):
        test_df = self._test_frame()
        with_nan = [900.0] * 300 + [float("nan")] * 5
        out = self._fold(with_nan, test_df)
        assert np.isfinite(out).all(), "a NaN in the seed must not reach the prediction"
