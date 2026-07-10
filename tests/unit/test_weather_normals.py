"""Unit tests for data/weather_normals.py — the #283 Phase 1 weather-normal
artifact (per-(day_of_year, hour) ERA5 normal for the days-17-30 forecast tail).

Weather I/O is mocked with synthetic multi-year data; no live API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from data.feature_engineering import compute_cdd


def _synth_weather(years: int = 3) -> pd.DataFrame:
    """Synthetic hourly weather with a real seasonal + diurnal cycle and noise.

    Warm summer (doy ~200) / cold winter; warmer afternoons. The ±6°F noise
    makes shoulder-season hours straddle the 65°F CDD base, which is what
    exercises the Jensen (E[CDD] > CDD(E[T])) property.
    """
    end = pd.Timestamp("2024-01-01", tz="UTC")
    ts = pd.date_range(end - pd.Timedelta(days=365 * years), end, freq="h")
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    rng = np.random.default_rng(0)
    seasonal = 60 + 25 * np.sin(2 * np.pi * (doy - 100) / 365)  # ~35°F winter, ~85°F summer
    diurnal = 8 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = seasonal + diurnal + rng.normal(0, 6, len(ts))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": temp,
            "wind_speed_80m": np.abs(12 + rng.normal(0, 4, len(ts))),
            "shortwave_radiation": np.maximum(0, 600 * np.sin(2 * np.pi * (hour - 6) / 24)),
        }
    )


class TestBuildWeatherNormal:
    def _build(self, weather=None, years=3):
        import data.weather_normals as wn

        weather = weather if weather is not None else _synth_weather(years)
        with patch("data.weather_client.fetch_historical_weather", return_value=weather):
            return wn.build_weather_normal("DUK", years=years)

    def test_produces_doy_hour_normal(self):
        normal = self._build()
        assert normal is not None
        assert {"doy", "hour"}.issubset(normal.columns)
        assert "cooling_degree_days" in normal.columns
        assert "temperature_2m" in normal.columns
        # ≤ 366 doy × 24 hour cells
        assert 0 < len(normal) <= 366 * 24
        assert normal["hour"].between(0, 23).all()
        assert normal["doy"].between(1, 366).all()

    def test_calendar_correct_ordering(self):
        """July is warmer / higher-CDD than January — the whole point."""
        normal = self._build()
        jul = normal[(normal["doy"] >= 190) & (normal["doy"] <= 210)]
        jan = normal[(normal["doy"] >= 5) & (normal["doy"] <= 25)]
        assert jul["temperature_2m"].mean() > jan["temperature_2m"].mean() + 20
        assert jul["cooling_degree_days"].mean() > jan["cooling_degree_days"].mean() + 5

    def test_jensen_stored_cdd_ge_cdd_of_mean_temp(self):
        """Storing the DERIVED feature directly (mean of CDD) must never fall
        below CDD(mean temp) — the convexity that a mean-temp-then-derive normal
        would lose — and must be strictly greater at a shoulder cell."""
        normal = self._build()
        stored_cdd = normal["cooling_degree_days"].to_numpy()
        cdd_of_mean = compute_cdd(normal["temperature_2m"]).to_numpy()
        # Jensen holds everywhere (small float tolerance).
        assert np.all(stored_cdd >= cdd_of_mean - 1e-6)
        # Strict at the shoulder cell whose normal temp is closest to 65°F.
        i = int(np.argmin(np.abs(normal["temperature_2m"].to_numpy() - 65.0)))
        assert stored_cdd[i] > cdd_of_mean[i] + 0.5

    def test_thin_history_returns_none(self):
        # Ask for 10 years but supply only ~1 → below the 60% floor → None.
        normal = self._build(weather=_synth_weather(years=1), years=10)
        assert normal is None

    def test_no_data_returns_none(self):
        assert self._build(weather=pd.DataFrame()) is None

    def test_leap_year_doy_366_present(self):
        """A window spanning a leap year (2020) yields a doy-366 (Dec 31) cell,
        finite and populated — the cold-start / leap path."""
        normal = self._build(weather=_synth_weather(years=5), years=5)
        assert 366 in set(normal["doy"])
        d366 = normal[normal["doy"] == 366]
        assert len(d366) > 0
        assert d366["temperature_2m"].notna().all()


class TestSmoothDoy:
    """The circular ±7-day day-of-year smoothing is the core of a stable normal —
    without a direct test, removing it or the Dec↔Jan wrap passes everything."""

    def _flat_with_spike(self, spike_idx):
        idx = pd.Index(range(1, 367), name="doy")
        pivot = pd.DataFrame({0: np.full(366, 10.0)}, index=idx)
        pivot.iloc[spike_idx, 0] = 100.0
        return pivot

    def test_dampens_a_spike(self):
        import data.weather_normals as wn

        out = wn._smooth_doy(self._flat_with_spike(179))  # spike at doy 180
        val = out.iloc[179, 0]
        assert 10.0 < val < 30.0  # a 100 spike averaged over a 15-day window → ~16

    def test_circular_wrap_dec_into_jan(self):
        """A spike at the last day-of-year must pull the first day up (and vice
        versa) — proves the window wraps rather than truncating at the edges."""
        import data.weather_normals as wn

        out_end = wn._smooth_doy(self._flat_with_spike(-1))  # spike at doy 366
        assert out_end.iloc[0, 0] > 10.0  # doy 1 lifted by the late-Dec spike
        out_start = wn._smooth_doy(self._flat_with_spike(0))  # spike at doy 1
        assert out_start.iloc[-1, 0] > 10.0  # doy 366 lifted by the early-Jan spike


class TestPersistAndLoad:
    def _normal(self):
        return pd.DataFrame(
            {
                "doy": [1, 1, 200],
                "hour": [0, 1, 12],
                "temperature_2m": [35.0, 34.0, 90.0],
                "cooling_degree_days": [0.0, 0.0, 25.0],
            }
        )

    def test_persist_writes_gcs_full_and_small_redis_marker(self):
        """The full 366×24 artifact goes to GCS (durable); Redis holds only a
        small metadata MARKER (no row data) at the long TTL, so a dormant feature
        doesn't cost ~210MB of Redis. GCS is written FIRST."""
        import data.weather_normals as wn
        from config import WEATHER_NORMAL_TTL_SECONDS

        calls = []
        wp = MagicMock(side_effect=lambda df, dt, region: calls.append("gcs"))
        rs = MagicMock(side_effect=lambda k, v, ttl=None: calls.append(("redis", v, ttl)) or True)
        with (
            patch("data.gcs_store.write_parquet", wp),
            patch("data.redis_client.redis_set", rs),
        ):
            wn.persist_weather_normal("DUK", self._normal(), years=10)

        # GCS gets the full DataFrame, first.
        assert calls[0] == "gcs"
        wp.assert_called_once()
        assert wp.call_args.args[0].equals(self._normal())
        # Redis marker: metadata only (no "rows" row-list), at the long TTL.
        _, marker, ttl = calls[1]
        assert ttl == WEATHER_NORMAL_TTL_SECONDS
        assert ttl > 90 * 24 * 3600  # survives the 90-day refresh cadence
        assert marker["rows"] == 3 and isinstance(marker["rows"], int)
        assert "updated_at" in marker and "features" in marker

    def test_load_reads_full_artifact_from_gcs(self):
        import data.weather_normals as wn

        with patch("data.gcs_store.read_parquet", return_value=self._normal()):
            loaded = wn.load_weather_normal("DUK")
        assert loaded is not None
        assert len(loaded) == 3
        assert set(loaded["doy"]) == {1, 200}

    def test_load_returns_none_when_gcs_absent(self):
        import data.weather_normals as wn

        with patch("data.gcs_store.read_parquet", return_value=None):
            assert wn.load_weather_normal("DUK") is None

    def test_normal_age_days(self):
        from datetime import UTC, datetime, timedelta

        import data.weather_normals as wn

        ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        with patch("data.redis_client.redis_get", return_value={"updated_at": ts, "rows": 8784}):
            age = wn.normal_age_days("DUK")
        assert age is not None
        assert 29.5 < age < 30.5

    def test_normal_age_none_when_absent(self):
        import data.weather_normals as wn

        with patch("data.redis_client.redis_get", return_value=None):
            assert wn.normal_age_days("DUK") is None


class TestLoadWeatherNormalCached:
    """#283 Phase 2: the scoring tail reads the normal every tick, so it's cached
    in-process (quarterly-changing data). Pins the three docstring claims."""

    def setup_method(self):
        import data.weather_normals as wn

        wn._normal_cache.clear()

    def teardown_method(self):
        import data.weather_normals as wn

        wn._normal_cache.clear()

    def test_one_gcs_read_within_ttl(self):
        import data.weather_normals as wn

        load = MagicMock(return_value=pd.DataFrame({"doy": [1]}))
        with patch("data.weather_normals.load_weather_normal", load):
            a = wn.load_weather_normal_cached("DUK")
            b = wn.load_weather_normal_cached("DUK")
        assert load.call_count == 1  # memoized — one underlying GCS read
        assert a is b

    def test_none_is_cached(self):
        import data.weather_normals as wn

        load = MagicMock(return_value=None)
        with patch("data.weather_normals.load_weather_normal", load):
            assert wn.load_weather_normal_cached("PJM") is None
            assert wn.load_weather_normal_cached("PJM") is None
        assert load.call_count == 1  # a not-yet-backfilled BA doesn't re-hit GCS every tick

    def test_rereads_after_ttl(self):
        import data.weather_normals as wn

        load = MagicMock(side_effect=[pd.DataFrame({"doy": [1]}), pd.DataFrame({"doy": [2]})])
        clock = [1000.0]
        with (
            patch("data.weather_normals.load_weather_normal", load),
            patch("data.weather_normals.time.time", side_effect=lambda: clock[0]),
        ):
            first = wn.load_weather_normal_cached("MISO")
            clock[0] += wn._NORMAL_CACHE_TTL_S + 1  # past the TTL
            second = wn.load_weather_normal_cached("MISO")
        assert load.call_count == 2  # re-read after expiry
        assert first["doy"].iloc[0] == 1 and second["doy"].iloc[0] == 2


class TestRefreshWeatherNormals:
    def test_skips_fresh_regions(self):
        import data.weather_normals as wn

        built = MagicMock()
        with (
            patch("data.weather_normals.normal_age_days", return_value=10),  # fresh (< 90)
            patch("data.weather_normals.build_weather_normal", built),
        ):
            summary = wn.refresh_weather_normals(["DUK", "PJM"], min_age_days=90)
        built.assert_not_called()
        assert set(summary["skipped"]) == {"DUK", "PJM"}
        assert summary["built"] == []

    def test_rebuilds_stale_and_missing(self):
        import data.weather_normals as wn

        # DUK missing (age None), PJM stale (age 200).
        ages = {"DUK": None, "PJM": 200}
        with (
            patch("data.weather_normals.normal_age_days", side_effect=lambda r: ages[r]),
            patch(
                "data.weather_normals.build_weather_normal", return_value=pd.DataFrame({"doy": [1]})
            ),
            patch("data.weather_normals.persist_weather_normal") as persist,
        ):
            summary = wn.refresh_weather_normals(["DUK", "PJM"], min_age_days=90, throttle_s=0)
        assert set(summary["built"]) == {"DUK", "PJM"}
        assert persist.call_count == 2

    def test_caps_at_max_rebuild(self):
        """The per-run cap spreads a cold-start backfill across runs."""
        import data.weather_normals as wn

        regions = ["A", "B", "C", "D"]
        with (
            patch("data.weather_normals.normal_age_days", return_value=None),  # all missing
            patch(
                "data.weather_normals.build_weather_normal", return_value=pd.DataFrame({"doy": [1]})
            ),
            patch("data.weather_normals.persist_weather_normal"),
        ):
            summary = wn.refresh_weather_normals(
                regions, min_age_days=90, max_rebuild=2, throttle_s=0
            )
        assert len(summary["built"]) == 2
        assert len(summary["skipped"]) == 2  # deferred to a later run

    def test_best_effort_on_build_failure(self):
        import data.weather_normals as wn

        with (
            patch("data.weather_normals.normal_age_days", return_value=None),
            patch(
                "data.weather_normals.build_weather_normal", side_effect=RuntimeError("archive 429")
            ),
        ):
            summary = wn.refresh_weather_normals(["DUK"], min_age_days=90, throttle_s=0)
        assert summary["failed"] == ["DUK"]
        assert summary["built"] == []

    def test_cap_bounds_attempts_not_just_successes(self):
        """The reliability fix: during an ERA5 outage every build fails, so the
        cap must bound ATTEMPTS (fetch timeouts), not successes — otherwise a
        degraded run attempts a fetch for every region and can overrun the task."""
        import data.weather_normals as wn

        build = MagicMock(return_value=None)  # every build "fails" (returns None)
        with (
            patch("data.weather_normals.normal_age_days", return_value=None),  # all missing
            patch("data.weather_normals.build_weather_normal", build),
        ):
            summary = wn.refresh_weather_normals(
                ["A", "B", "C", "D", "E"], min_age_days=90, max_rebuild=2, throttle_s=0
            )
        assert build.call_count == 2  # attempts capped despite zero successes
        assert len(summary["failed"]) == 2
        assert len(summary["skipped"]) == 3  # deferred to a later run
