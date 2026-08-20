"""#559 candidate 1 — the forecast origin must not stall behind ``dropna``.

``engineer_features`` drops every row whose autoregressive lag source was null,
so one null hour deletes the rows 1, 2, 3, 24 and 168 hours after it — including
the *tail*, when the null hour sits that far before the frame end. The old cap
``min(last_real_demand, last_featured_ts)`` therefore asked "did this row survive
``dropna``" rather than "do we hold demand for it", and the origin froze while
fresh hours kept arriving (LGEE: 16 hours, live on 2026-08-20).

Two things are pinned here and both matter:

* the origin advances across the real demand we hold (``temporal_ar_seed`` on);
* it does **not** advance under the positional seed, where an advanced origin
  would index ``demand_lag_1h`` to the tail of ``featured`` instead of to
  ``origin - 1h`` — a silent wrong value, worse than the stall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jobs import phases

HOUR = pd.Timedelta(hours=1)


def _hours(end: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n, freq="h")


def _demand(end: str, n: int = 400) -> pd.DataFrame:
    ts = _hours(end, n)
    # A diurnal-ish shape so a wrongly-indexed lag is a different NUMBER, not
    # just a different label — a flat series would hide every misindexing.
    mw = 20_000.0 + 3_000.0 * np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
    return pd.DataFrame({"timestamp": ts, "demand_mw": mw})


def _featured_truncated_at(demand_df: pd.DataFrame, last_kept: str) -> pd.DataFrame:
    """``featured`` as ``dropna`` leaves it: the tail rows are simply gone."""
    cut = pd.Timestamp(last_kept, tz="UTC")
    return demand_df.loc[demand_df["timestamp"] <= cut].reset_index(drop=True)


@pytest.fixture
def temporal_on(monkeypatch):
    """Turn ``temporal_ar_seed`` on the way production reads it."""
    import config

    monkeypatch.setitem(config.FEATURE_FLAGS, "temporal_ar_seed", True)
    assert config.feature_enabled("temporal_ar_seed") is True
    return True


# ── the stall itself ──────────────────────────────────────────────────


class TestOriginAdvancesAcrossDroppedRows:
    def test_dropped_tail_still_resolves_at_last_real_demand(self, temporal_on):
        """The defect, stated as an assertion.

        Demand runs through 2026-08-14T05:00. ``featured`` ends at
        2026-08-13T13:00 because a 16-hour null hole 24 hours earlier deleted
        exactly those rows. The origin must be 2026-08-14T06:00, not
        2026-08-13T14:00.
        """
        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df, region="LGEE")
        assert start == pd.Timestamp("2026-08-14 06:00", tz="UTC")

    def test_the_stall_is_16_hours_wide(self, temporal_on):
        """Guards the magnitude, not only the direction — a fix that advanced
        by one hour per tick would still leave the origin 15 hours stale."""
        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        old = featured["timestamp"].max() + HOUR
        new = phases._resolve_forecast_start(featured, demand_df)
        assert (new - old) == pd.Timedelta(hours=16)

    def test_flag_off_keeps_the_pre_fix_cap(self):
        """The positional seed cannot serve a bridged origin, so with
        ``temporal_ar_seed`` off the cap stays exactly where #129 put it."""
        import config

        assert config.feature_enabled("temporal_ar_seed") is False
        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-13 14:00", tz="UTC")

    def test_positional_seed_never_matches_a_bridged_origin(self, temporal_on):
        """Why the gate exists, as a fact rather than a comment.

        ``positional_seed_matches_hours`` is the exact condition for the
        positional arm reading the hours it names. It is false at the bridged
        origin, which is why the advance is gated on the hour-indexed path.
        """
        from data.feature_engineering import positional_seed_matches_hours

        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")
        start = phases._resolve_forecast_start(featured, demand_df)

        assert positional_seed_matches_hours(featured["timestamp"], start) is False
        # ...and it IS true at the un-advanced origin, so the predicate is
        # discriminating rather than uniformly False on this fixture.
        assert (
            positional_seed_matches_hours(featured["timestamp"], featured["timestamp"].max() + HOUR)
            is True
        )

    def test_no_gap_is_untouched(self, temporal_on):
        """The common case — 44 of 51 BAs — must not move."""
        demand_df = _demand("2026-08-14 05:00")
        featured = demand_df.copy()

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-14 06:00", tz="UTC")

    def test_publishing_lag_gap_still_wins(self, temporal_on):
        """#129's case is the other direction (``featured`` ahead of demand)
        and the bridge must not reach into it."""
        demand_df = _demand("2026-08-20 10:00")
        featured = _demand("2026-08-20 14:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-20 11:00", tz="UTC")


# ── what bounds the bridge ────────────────────────────────────────────


class TestBridgeIsBoundedByRealHourlyDemand:
    def test_a_hole_inside_the_bridge_stops_it(self, temporal_on):
        """Advance only across CONTIGUOUS real hours. A null hour at 20:00
        means 21:00 onward cannot seed ``demand_lag_1h`` honestly, so the
        origin stops at 20:00 — advanced, but not past what we hold."""
        demand_df = _demand("2026-08-14 05:00")
        demand_df.loc[
            demand_df["timestamp"] == pd.Timestamp("2026-08-13 20:00", tz="UTC"), "demand_mw"
        ] = np.nan
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-13 20:00", tz="UTC")

    def test_a_zero_hour_stops_the_bridge(self, temporal_on):
        """#129's zero-poison filter is preserved: a literal 0 MW is missing
        data, not demand, and may not be bridged across."""
        demand_df = _demand("2026-08-14 05:00")
        demand_df.loc[
            demand_df["timestamp"] == pd.Timestamp("2026-08-13 18:00", tz="UTC"), "demand_mw"
        ] = 0.0
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-13 18:00", tz="UTC")

    def test_a_missing_first_hour_blocks_the_bridge_entirely(self, temporal_on):
        """No contiguous run starting at ``last_featured_ts + 1h`` → no
        advance. Degrades to the pre-fix cap rather than skipping the hole."""
        demand_df = _demand("2026-08-14 05:00")
        demand_df.loc[
            demand_df["timestamp"] == pd.Timestamp("2026-08-13 14:00", tz="UTC"), "demand_mw"
        ] = np.nan
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-13 14:00", tz="UTC")

    def test_absent_rows_are_not_bridged_across(self, temporal_on):
        """An hour EIA never reported is an absent ROW, not a null one. The
        bridge is hourly-contiguous, so it stops there too."""
        demand_df = _demand("2026-08-14 05:00")
        demand_df = demand_df[
            demand_df["timestamp"] != pd.Timestamp("2026-08-13 19:00", tz="UTC")
        ].reset_index(drop=True)
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")

        start = phases._resolve_forecast_start(featured, demand_df)
        assert start == pd.Timestamp("2026-08-13 19:00", tz="UTC")


# ── the origin and its seed cannot drift apart ────────────────────────


class TestOriginAndSeedAreResolvedTogether:
    def test_the_seed_ends_at_the_hour_before_the_origin(self, temporal_on):
        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")
        start = phases._resolve_forecast_start(featured, demand_df)

        resolved, seed = phases._ar_seed_for_origin(featured, demand_df, start)
        assert resolved == start
        assert seed is not None
        assert seed["timestamp"].max() == start - HOUR
        # Contiguous across the bridge, and carrying the REAL values — not the
        # 24h-step-back imputation an unbridged hour-indexed seed would use.
        bridged = seed[seed["timestamp"] > featured["timestamp"].max()]
        assert len(bridged) == 16
        expected = demand_df.set_index("timestamp")["demand_mw"]
        for _, row in bridged.iterrows():
            assert row["demand_mw"] == pytest.approx(expected[row["timestamp"]])

    def test_an_unbridgeable_origin_is_clamped_back(self, temporal_on):
        """The forbidden state — origin advanced, seed not — is unreachable,
        not merely unlikely: an origin that no bridge reaches is clamped."""
        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")
        impossible = pd.Timestamp("2026-08-14 06:00", tz="UTC")

        # A demand frame that cannot produce the bridge the origin implies.
        resolved, seed = phases._ar_seed_for_origin(featured, featured, impossible)
        assert resolved == featured["timestamp"].max() + HOUR
        assert seed is None

    def test_an_unadvanced_origin_seeds_from_featured(self, temporal_on):
        demand_df = _demand("2026-08-14 05:00")
        featured = demand_df.copy()
        start = phases._resolve_forecast_start(featured, demand_df)

        resolved, seed = phases._ar_seed_for_origin(featured, demand_df, start)
        assert resolved == start
        assert seed is None


class TestTheRecursionReceivesTheBridgedSeed:
    def test_seed_frame_reaches_recursive_autoregressive_forecast(self, temporal_on, monkeypatch):
        """The seed is not merely computed — it is what the recursion sees."""
        import data.feature_engineering as fe

        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")
        start = phases._resolve_forecast_start(featured, demand_df)
        _, seed = phases._ar_seed_for_origin(featured, demand_df, start)

        seen: dict[str, object] = {}

        def _spy(model, seed_demand, future_df, predict_fn, seed_timestamps=None, **kw):
            seen["n"] = seen.get("n", 0) + 1
            seen["last_ts"] = pd.to_datetime(pd.Series(list(seed_timestamps)), utc=True).max()
            seen["last_mw"] = list(seed_demand)[-1]
            return np.zeros(len(future_df), dtype=float)

        monkeypatch.setattr(fe, "recursive_autoregressive_forecast", _spy)

        future_df = pd.DataFrame({"timestamp": pd.date_range(start, periods=24, freq="h")})
        phases._predict_xgboost_with_recursive_autoregressive(
            object(), featured, future_df, horizon=24, seed_frame=seed
        )

        # The mock actually intercepted — a silently-defeated one would leave
        # this empty and every assertion below vacuous.
        assert seen.get("n") == 1
        assert seen["last_ts"] == start - HOUR
        expected = demand_df.set_index("timestamp")["demand_mw"][start - HOUR]
        assert seen["last_mw"] == pytest.approx(expected)

    def test_without_a_seed_frame_the_recursion_still_sees_featured(self, temporal_on, monkeypatch):
        """Flag-on but unbridged ticks are unchanged — the seed source only
        moves on the ticks the origin moved."""
        import data.feature_engineering as fe

        demand_df = _demand("2026-08-14 05:00")
        featured = _featured_truncated_at(demand_df, "2026-08-13 13:00")
        seen: dict[str, object] = {}

        def _spy(model, seed_demand, future_df, predict_fn, seed_timestamps=None, **kw):
            seen["n"] = seen.get("n", 0) + 1
            seen["last_ts"] = pd.to_datetime(pd.Series(list(seed_timestamps)), utc=True).max()
            return np.zeros(len(future_df), dtype=float)

        monkeypatch.setattr(fe, "recursive_autoregressive_forecast", _spy)

        future_df = pd.DataFrame(
            {"timestamp": pd.date_range(featured["timestamp"].max() + HOUR, periods=24, freq="h")}
        )
        phases._predict_xgboost_with_recursive_autoregressive(
            object(), featured, future_df, horizon=24
        )
        assert seen.get("n") == 1
        assert seen["last_ts"] == featured["timestamp"].max()


# ── against the real dropna, not a hand-cut fixture ───────────────────


class TestAgainstRealEngineerFeatures:
    def test_a_null_hour_24h_back_truncates_featured_and_the_origin_survives(self, temporal_on):
        """The fixtures above cut ``featured`` by hand. This one lets
        ``engineer_features`` do it, so the test cannot quietly stop describing
        the mechanism if the drop subset changes.
        """
        from data.feature_engineering import engineer_features

        ts = _hours("2026-08-14 05:00", 600)
        mw = 20_000.0 + 3_000.0 * np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
        merged = pd.DataFrame(
            {"timestamp": ts, "demand_mw": mw, "temperature_2m": 70.0, "wind_speed_10m": 5.0}
        )
        hole = (merged["timestamp"] >= pd.Timestamp("2026-08-12 14:00", tz="UTC")) & (
            merged["timestamp"] <= pd.Timestamp("2026-08-13 05:00", tz="UTC")
        )
        merged.loc[hole, "demand_mw"] = np.nan

        featured = engineer_features(merged)
        demand_df = merged.loc[:, ["timestamp", "demand_mw"]]

        # The mechanism is present in the fixture: dropna truncated the tail.
        assert featured["timestamp"].max() == pd.Timestamp("2026-08-13 13:00", tz="UTC")
        assert demand_df["demand_mw"].notna().to_numpy()[-1]

        start = phases._resolve_forecast_start(featured, demand_df, region="LGEE")
        assert start == pd.Timestamp("2026-08-14 06:00", tz="UTC")

        _, seed = phases._ar_seed_for_origin(featured, demand_df, start)
        assert seed is not None and seed["timestamp"].max() == start - HOUR

    def test_the_bridged_seed_resolves_the_origins_near_lags_to_real_hours(self, temporal_on):
        """The reason the seed and the origin are coupled, measured.

        An hour-indexed seed built from ``featured`` alone would *impute*
        ``demand_lag_1h`` at the bridged origin — the hole is 16 hours wide, so
        ``HourIndexedHistory.lag`` falls past interpolation into the 24h
        step-back regime and reads a value from the previous day. The bridged
        seed reads the hour itself.
        """
        from data.feature_engineering import (
            HourIndexedHistory,
            compute_temporal_autoregressive_snapshot,
            engineer_features,
        )

        ts = _hours("2026-08-14 05:00", 600)
        mw = (
            20_000.0
            + 3_000.0 * np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
            + 500.0 * np.sin(2 * np.pi * np.arange(600) / 168)
        )
        merged = pd.DataFrame(
            {"timestamp": ts, "demand_mw": mw, "temperature_2m": 70.0, "wind_speed_10m": 5.0}
        )
        hole = (merged["timestamp"] >= pd.Timestamp("2026-08-12 14:00", tz="UTC")) & (
            merged["timestamp"] <= pd.Timestamp("2026-08-13 05:00", tz="UTC")
        )
        merged.loc[hole, "demand_mw"] = np.nan
        featured = engineer_features(merged)
        demand_df = merged.loc[:, ["timestamp", "demand_mw"]]
        truth = merged.set_index("timestamp")["demand_mw"]

        origin = phases._resolve_forecast_start(featured, demand_df)
        _, seed = phases._ar_seed_for_origin(featured, demand_df, origin)

        def _lag(frame, k):
            hist = HourIndexedHistory.build(frame["timestamp"], frame["demand_mw"], extra_hours=400)
            return compute_temporal_autoregressive_snapshot(hist, origin)[f"demand_lag_{k}h"]

        for k in (1, 3):
            want = float(truth[origin - pd.Timedelta(hours=k)])
            assert _lag(seed, k) == pytest.approx(want)
            # The control: without the bridge the same lag is materially wrong,
            # so this assertion is discriminating rather than trivially true.
            assert abs(_lag(featured, k) - want) > 100.0
