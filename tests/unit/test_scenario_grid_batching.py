"""Differential parity for the batched recursive forecaster (#127 / #462).

The scenario grid was reverted because it issued 1,920 single-row predicts per
region against production's 384. Batching the 80 variants into one call per
step fixes the cost — but only if it is *byte-identical* to the single-frame
helper, which is the documented single source of truth for multi-step
inference. "The forecasts still look right" is not evidence at this seam, so
this file runs both implementations and compares.
"""

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import (
    batched_recursive_autoregressive_forecast,
    recursive_autoregressive_forecast,
)

HORIZON = 12


def _frame(n: int = HORIZON, temp: float = 70.0, int_col: bool = False) -> pd.DataFrame:
    hours = np.arange(n)
    df = pd.DataFrame(
        {
            "temperature_2m": temp + 5.0 * np.sin(hours / 24 * 2 * np.pi),
            "hour_sin": np.sin(hours / 24 * 2 * np.pi),
            "demand_lag_1h": np.zeros(n),
            "demand_lag_24h": np.zeros(n),
            "demand_roll_24h_mean": np.zeros(n),
            "ramp_rate": np.zeros(n),
        }
    )
    if int_col:
        # `_build_future_feature_frame` does `future_df[col] = 0` for features it
        # could not fill, which creates an int64 column — the dtype trap the
        # single-frame helper documents.
        df["demand_lag_1h"] = 0
    return df


class _RecordingModel:
    """Predicts from the row's own features, and records every frame it sees."""

    def __init__(self):
        self.seen: list[pd.DataFrame] = []

    def __call__(self, _model, frame: pd.DataFrame) -> np.ndarray:
        self.seen.append(frame.copy())
        return (
            1000.0
            + 10.0 * frame["temperature_2m"].to_numpy()
            + 0.5 * frame["demand_lag_1h"].to_numpy()
        )


SEED = [1400.0, 1450.0, 1500.0, 1480.0, 1520.0] * 40


class TestParityWithTheSingleFrameHelper:
    @pytest.mark.parametrize("n_scenarios", [1, 2, 9], ids=["one", "two", "nine"])
    def test_batched_output_matches_frame_by_frame_output(self, n_scenarios):
        """The contract: same numbers, whichever way they were computed."""
        frames = [_frame(temp=60.0 + 5 * k) for k in range(n_scenarios)]

        one_at_a_time = [
            recursive_autoregressive_forecast(None, SEED, f.copy(), _RecordingModel())
            for f in frames
        ]
        batched = batched_recursive_autoregressive_forecast(
            None, SEED, [f.copy() for f in frames], _RecordingModel()
        )

        assert len(batched) == n_scenarios
        for single, batch in zip(one_at_a_time, batched, strict=True):
            np.testing.assert_array_equal(single, batch)

    def test_the_rows_handed_to_the_model_are_identical(self):
        """Parity of *inputs*, not just outputs.

        Equal predictions could hide a divergent frame if the stub happened not
        to read the differing column. This compares the actual rows.
        """
        frames = [_frame(temp=60.0), _frame(temp=90.0)]

        singles = []
        for f in frames:
            rec = _RecordingModel()
            recursive_autoregressive_forecast(None, SEED, f.copy(), rec)
            singles.append(pd.concat(rec.seen, ignore_index=True))

        batch_rec = _RecordingModel()
        batched_recursive_autoregressive_forecast(None, SEED, [f.copy() for f in frames], batch_rec)
        # Batched frames are step-major (all scenarios for step 0, then step 1);
        # regroup to scenario-major to compare like with like.
        steps = pd.concat(batch_rec.seen, ignore_index=True)
        for j, single in enumerate(singles):
            rebuilt = steps.iloc[[i * len(frames) + j for i in range(HORIZON)]].reset_index(
                drop=True
            )
            pd.testing.assert_frame_equal(single, rebuilt, check_dtype=True)

    def test_the_int64_dtype_trap_is_handled_the_same_way(self):
        """`future_df[col] = 0` leaves an int64 column and positional assignment
        writes into the block. Both helpers must cast up front."""
        frames = [_frame(int_col=True), _frame(temp=85.0, int_col=True)]

        singles = [
            recursive_autoregressive_forecast(None, SEED, f.copy(), _RecordingModel())
            for f in frames
        ]
        batched = batched_recursive_autoregressive_forecast(
            None, SEED, [f.copy() for f in frames], _RecordingModel()
        )

        for s, b in zip(singles, batched, strict=True):
            np.testing.assert_array_equal(s, b)

    def test_each_scenario_chains_only_its_own_predictions(self):
        """The chaining stays per-frame — no scenario may see another's history.

        Two frames with very different temperatures: if the histories were
        shared, the cooler scenario's lag features would be contaminated by the
        hotter one's predictions and the two would converge.
        """
        cool, hot = _frame(temp=50.0), _frame(temp=95.0)

        batched = batched_recursive_autoregressive_forecast(
            None, SEED, [cool.copy(), hot.copy()], _RecordingModel()
        )
        alone_cool = recursive_autoregressive_forecast(None, SEED, cool.copy(), _RecordingModel())

        np.testing.assert_array_equal(batched[0], alone_cool)
        assert batched[1][-1] > batched[0][-1] * 1.05, "the two must not converge"


class TestBatchingMechanics:
    def test_the_model_is_called_once_per_step_not_once_per_scenario(self):
        """The entire point. 9 scenarios x 12 steps is 12 calls, not 108."""
        rec = _RecordingModel()

        batched_recursive_autoregressive_forecast(
            None, SEED, [_frame(temp=60.0 + k) for k in range(9)], rec
        )

        assert len(rec.seen) == HORIZON
        assert all(len(f) == 9 for f in rec.seen)

    def test_no_frames_is_not_an_error(self):
        assert batched_recursive_autoregressive_forecast(None, SEED, [], _RecordingModel()) == []

    def test_ragged_frames_are_refused(self):
        """Silently truncating would produce curves of different lengths for
        different grid cells, and the payload promises a fixed horizon."""
        with pytest.raises(ValueError):
            batched_recursive_autoregressive_forecast(
                None, SEED, [_frame(12), _frame(11)], _RecordingModel()
            )

    def test_the_seed_filter_matches_the_single_frame_helper(self):
        """Zeros and NaNs are dropped — a single 0 poisons 168 rolling
        features (#129) — and the batched path must drop them identically."""
        dirty = [0.0, np.nan, 1400.0, 1450.0, -5.0] * 40

        single = recursive_autoregressive_forecast(None, dirty, _frame().copy(), _RecordingModel())
        batched = batched_recursive_autoregressive_forecast(
            None, dirty, [_frame().copy()], _RecordingModel()
        )

        np.testing.assert_array_equal(single, batched[0])
