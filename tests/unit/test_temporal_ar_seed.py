"""#559 — the autoregressive seed must resolve lags by HOUR, not by position.

``engineer_features`` drops every row whose lag source was null, so ``featured``
carries real discontinuities, and the recursion is seeded with that frame. The
positional snapshot then reads ``history[-168]`` — 168 *surviving rows* back,
which after a hole is not 168 hours back. Measured 34h off on LGEE at a live
origin (``docs/POSITIONAL_LAG_SEED_STUDY.md``).

The parity test these sit beside
(``test_training_features_match_inference_snapshot_row_by_row``) could not catch
that: it compares the two implementations on a **gapless** fixture, where they
agree by construction. Every test here uses a gapped one, which is the whole
point.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data.feature_engineering import (
    AUTOREGRESSIVE_DEMAND_FEATURES,
    add_autoregressive_demand_features,
    compute_autoregressive_snapshot,
    compute_temporal_autoregressive_snapshot,
    recursive_autoregressive_forecast,
)

GAP_LEN = 16  # matches LGEE's real 2026-08-12 hole
# The hole must fall INSIDE the 168h lookback or positional and temporal agree
# and the fixture proves nothing — which is the same trap the gapless parity
# test fell into, one level up.
GAP_START = 500


@pytest.fixture
def temporal_on(monkeypatch):
    monkeypatch.setitem(config.FEATURE_FLAGS, "temporal_ar_seed", True)


@pytest.fixture
def temporal_off(monkeypatch):
    monkeypatch.setitem(config.FEATURE_FLAGS, "temporal_ar_seed", False)


def _gapped_demand(n: int = 600, gap_start: int = GAP_START) -> pd.DataFrame:
    """A demand series with a contiguous hole cut OUT of the row grid.

    This is the post-``dropna`` shape: the hours are simply absent, which is
    exactly what the serve path is seeded with.
    """
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n)
    demand = (
        20_000
        + 5_000 * np.sin(2 * np.pi * hours / 24)
        + 1_000 * np.sin(2 * np.pi * hours / (24 * 7))
        + rng.normal(0, 300, size=n)
    )
    df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
    return df.drop(index=range(gap_start, gap_start + GAP_LEN)).reset_index(drop=True)


def _history(df: pd.DataFrame) -> dict[pd.Timestamp, float]:
    return {t: float(v) for t, v in zip(df["timestamp"], df["demand_mw"], strict=True)}


class TestTheDefect:
    """Characterisation. These pin the bug, so a silent change to it fails."""

    def test_positional_lag_reads_the_wrong_hour_after_a_gap(self):
        df = _gapped_demand()
        now = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        positional = compute_autoregressive_snapshot(df["demand_mw"].tolist())

        # The hole sits inside the 168h lookback, so `history[-168]` overshoots
        # by exactly the number of hours removed.
        want = float(df.loc[df["timestamp"] == now - pd.Timedelta(hours=168), "demand_mw"].iloc[0])
        assert positional["demand_lag_168h"] != pytest.approx(want)

        reached = df["timestamp"].iloc[-168]
        assert (now - pd.Timedelta(hours=168)) - reached == pd.Timedelta(hours=GAP_LEN)

    def test_gapless_series_is_blind_to_the_defect(self):
        """Why the existing parity test could not see this.

        With no hole, positional and temporal agree exactly — so a gapless
        fixture cannot distinguish a correct implementation from a broken one.
        """
        rng = np.random.default_rng(7)
        n = 400
        ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        demand = 20_000 + 5_000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, n)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        now = ts[-1] + pd.Timedelta(hours=1)

        pos = compute_autoregressive_snapshot(df["demand_mw"].tolist())
        tmp = compute_temporal_autoregressive_snapshot(_history(df), now)
        for key in AUTOREGRESSIVE_DEMAND_FEATURES:
            assert pos[key] == pytest.approx(tmp[key], rel=1e-9), key


class TestTemporalSnapshot:
    def test_lags_resolve_to_the_requested_hour_across_a_gap(self):
        df = _gapped_demand()
        now = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        snap = compute_temporal_autoregressive_snapshot(_history(df), now)

        for k in (1, 3, 24, 168):
            want_ts = now - pd.Timedelta(hours=k)
            want = float(df.loc[df["timestamp"] == want_ts, "demand_mw"].iloc[0])
            assert snap[f"demand_lag_{k}h"] == pytest.approx(want), f"lag_{k}h"

    def test_an_absent_hour_is_nan_not_a_further_reach(self):
        """The honest answer for a missing hour is "unknown".

        The positional version cannot express this — it always returns *some*
        value, which is the defect.
        """
        df = _gapped_demand()
        # An origin whose 24h-ago hour lands inside the hole.
        # 24h before this origin is the FIRST removed hour.
        now = df["timestamp"].iloc[GAP_START - 1] + pd.Timedelta(hours=25)
        snap = compute_temporal_autoregressive_snapshot(_history(df), now)
        assert np.isnan(snap["demand_lag_24h"])

    def test_rolling_windows_cover_hours_present_in_the_window(self):
        df = _gapped_demand()
        now = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        snap = compute_temporal_autoregressive_snapshot(_history(df), now)

        window = df[(df["timestamp"] >= now - pd.Timedelta(hours=168)) & (df["timestamp"] < now)][
            "demand_mw"
        ]
        assert snap["demand_roll_168h_mean"] == pytest.approx(float(window.mean()))
        assert snap["demand_roll_168h_max"] == pytest.approx(float(window.max()))
        assert snap["demand_roll_168h_min"] == pytest.approx(float(window.min()))

    def test_key_set_matches_the_positional_snapshot_exactly(self):
        df = _gapped_demand()
        now = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        assert set(compute_temporal_autoregressive_snapshot(_history(df), now)) == set(
            compute_autoregressive_snapshot(df["demand_mw"].tolist())
        )


class TestTrainInferenceParityOnAGappedFrame:
    """#186's real content: the two implementations must agree where it counts.

    ``add_autoregressive_demand_features`` shifts a frame that still has its
    rows, so its lags are temporally exact. The inference snapshot must produce
    the same values — and on a gapped frame only the temporal one does.
    """

    def test_temporal_snapshot_matches_training_features(self):
        n = 600
        rng = np.random.default_rng(11)
        ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        demand = 20_000 + 5_000 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 300, size=n)
        full = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        feats = add_autoregressive_demand_features(full.copy())

        # Row i's training lags were computed on the intact grid. Hand the
        # inference side only the PRIOR hours, minus a hole, and it must still
        # land on the same values.
        i = 500
        gap_at = i - 100  # inside the 168h lookback from row i
        prior = full.iloc[:i]
        holed = prior.drop(index=range(gap_at, gap_at + GAP_LEN))
        now = ts[i]
        snap = compute_temporal_autoregressive_snapshot(_history(holed), now)

        for key in ("demand_lag_1h", "demand_lag_3h", "demand_lag_24h", "demand_lag_168h"):
            assert snap[key] == pytest.approx(float(feats[key].iloc[i]), rel=1e-9), key


class TestRecursionWiring:
    @staticmethod
    def _frame(stamps):
        return pd.DataFrame(
            {"timestamp": stamps, **{k: 0.0 for k in AUTOREGRESSIVE_DEMAND_FEATURES}}
        )

    @staticmethod
    def _echo(_model, row):
        return [float(row["demand_lag_168h"].iloc[0])]

    def test_flag_off_ignores_seed_timestamps_entirely(self, temporal_off):
        """Fail-open contract: off must be byte-identical to before this change."""
        df = _gapped_demand()
        stamps = pd.date_range(
            df["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=6, freq="h", tz="UTC"
        )
        frame = self._frame(stamps)

        without = recursive_autoregressive_forecast(
            None, df["demand_mw"].tolist(), frame.copy(), self._echo
        )
        with_ts = recursive_autoregressive_forecast(
            None,
            df["demand_mw"].tolist(),
            frame.copy(),
            self._echo,
            seed_timestamps=df["timestamp"],
        )
        np.testing.assert_array_equal(without, with_ts)

    def test_flag_on_feeds_the_model_the_right_hour(self, temporal_on):
        df = _gapped_demand()
        origin = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        frame = self._frame(pd.date_range(origin, periods=3, freq="h", tz="UTC"))

        got = recursive_autoregressive_forecast(
            None,
            df["demand_mw"].tolist(),
            frame.copy(),
            self._echo,
            seed_timestamps=df["timestamp"],
        )
        want = float(
            df.loc[df["timestamp"] == origin - pd.Timedelta(hours=168), "demand_mw"].iloc[0]
        )
        assert got[0] == pytest.approx(want)

    def test_flag_on_without_timestamps_falls_back_rather_than_failing(self, temporal_on):
        """Every caller that cannot supply timestamps keeps working."""
        df = _gapped_demand()
        stamps = pd.date_range(
            df["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=4, freq="h", tz="UTC"
        )
        frame = self._frame(stamps)

        fallback = recursive_autoregressive_forecast(
            None, df["demand_mw"].tolist(), frame.copy(), self._echo
        )
        positional = compute_autoregressive_snapshot(df["demand_mw"].tolist())
        assert fallback[0] == pytest.approx(positional["demand_lag_168h"])

    def test_mismatched_seed_lengths_fall_back(self, temporal_on):
        df = _gapped_demand()
        stamps = pd.date_range(
            df["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="h", tz="UTC"
        )
        got = recursive_autoregressive_forecast(
            None,
            df["demand_mw"].tolist(),
            self._frame(stamps),
            self._echo,
            seed_timestamps=df["timestamp"].iloc[:-5],
        )
        assert np.isfinite(got).all()
