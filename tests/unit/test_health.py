"""Unit tests for health.py — the deep /health report (PR-G3 / #147).

Covers the three checks (redis / last_scored / forecast_sample), their
REQUIRE_REDIS gating, and the overall status aggregation. Redis helpers
are patched at ``data.redis_client.*`` and the env flag at
``config.REQUIRE_REDIS`` — health.py imports both lazily inside each
function, so source-module patches take effect per call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import health

# ── _check_redis ─────────────────────────────────────────────────────


class TestCheckRedis:
    def test_required_and_up_is_ok(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
        ):
            r = health._check_redis()
        assert r["status"] == "ok"
        assert r["required"] is True
        assert r["up"] is True

    def test_required_and_down_is_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            r = health._check_redis()
        assert r["status"] == "degraded"

    def test_not_required_and_down_is_skipped(self):
        """Dev / CI: Redis absence is expected, not a health failure."""
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            r = health._check_redis()
        assert r["status"] == "skipped"


# ── _check_last_scored ───────────────────────────────────────────────


class TestCheckLastScored:
    def _meta(self, *, minutes_ago: float) -> dict:
        ts = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
        return {"updated_at": ts, "regions_scored": 51, "mode": "scoring-job"}

    def test_fresh_marker_is_ok(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=self._meta(minutes_ago=10)),
        ):
            r = health._check_last_scored()
        assert r["status"] == "ok"
        assert r["stale"] is False
        assert r["age_seconds"] < health.LAST_SCORED_STALE_SECONDS

    def test_stale_marker_required_is_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=self._meta(minutes_ago=180)),
        ):
            r = health._check_last_scored()
        assert r["status"] == "degraded"
        assert r["stale"] is True

    def test_stale_marker_not_required_is_ok(self):
        """When Redis isn't required, staleness isn't held against health."""
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=self._meta(minutes_ago=180)),
        ):
            r = health._check_last_scored()
        assert r["status"] == "ok"  # stale, but not required → not degraded

    def test_missing_marker_required_is_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=None),
        ):
            r = health._check_last_scored()
        assert r["status"] == "degraded"
        assert r["reason"] == "no_last_scored_marker"

    def test_redis_down_is_skipped(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            r = health._check_last_scored()
        assert r["status"] == "skipped"

    def test_naive_timestamp_is_treated_as_utc(self):
        """A marker written without tzinfo must not crash the age math."""
        naive = (datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=5)).isoformat()
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value={"updated_at": naive}),
        ):
            r = health._check_last_scored()
        assert r["status"] == "ok"


# ── _check_forecast_sample ───────────────────────────────────────────


class TestCheckForecastSample:
    def _good_payload(self) -> dict:
        return {
            "forecasts": [
                {"timestamp": "2026-05-29T00:00:00+00:00", "predicted_demand_mw": 18342.0},
                {"timestamp": "2026-05-29T01:00:00+00:00", "predicted_demand_mw": 17900.0},
            ]
        }

    def test_valid_payload_is_ok(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("config.PRECOMPUTE_DEFAULT_REGION", "FPL"),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=self._good_payload()),
        ):
            r = health._check_forecast_sample()
        assert r["status"] == "ok"
        assert r["rows"] == 2
        assert r["region"] == "FPL"

    def test_empty_forecasts_required_is_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("config.PRECOMPUTE_DEFAULT_REGION", "FPL"),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value={"forecasts": []}),
        ):
            r = health._check_forecast_sample()
        assert r["status"] == "degraded"
        assert r["reason"] == "empty_forecasts"

    def test_malformed_row_required_is_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("config.PRECOMPUTE_DEFAULT_REGION", "FPL"),
            patch("data.redis_client.redis_available", return_value=True),
            patch(
                "data.redis_client.redis_get",
                return_value={"forecasts": [{"timestamp": "x"}]},  # no predicted_demand_mw
            ),
        ):
            r = health._check_forecast_sample()
        assert r["status"] == "degraded"
        assert r["reason"] == "malformed_forecast_row"

    def test_no_payload_not_required_is_skipped(self):
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("config.PRECOMPUTE_DEFAULT_REGION", "FPL"),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=None),
        ):
            r = health._check_forecast_sample()
        assert r["status"] == "skipped"


# ── build_health_report ──────────────────────────────────────────────


class TestBuildHealthReport:
    def test_shallow_omits_forecast_sample(self):
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            body, code = health.build_health_report(deep=False)
        assert code == 200
        assert body["deep"] is False
        assert "redis" in body["checks"]
        assert "last_scored" in body["checks"]
        assert "forecast_sample" not in body["checks"]

    def test_deep_includes_forecast_sample(self):
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            body, code = health.build_health_report(deep=True)
        assert body["deep"] is True
        assert "forecast_sample" in body["checks"]

    def test_ci_like_no_redis_reports_healthy(self):
        """The CI Docker container runs with no Redis + REQUIRE_REDIS
        false. All Redis checks skip → overall healthy → keeps the CI
        docker health assertion (status == 'healthy') green."""
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            body, code = health.build_health_report(deep=True)
        assert code == 200
        assert body["status"] == "healthy"

    def test_production_redis_down_reports_degraded(self):
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            body, code = health.build_health_report(deep=True)
        assert code == 200  # still 200 — degraded ≠ down
        assert body["status"] == "degraded"
        assert body["checks"]["redis"]["status"] == "degraded"

    def test_always_returns_200_even_when_degraded(self):
        """A degraded instance must stay 200 so the load balancer doesn't
        kill a warming-but-serving container."""
        with (
            patch("config.REQUIRE_REDIS", True),
            patch("data.redis_client.redis_available", return_value=True),
            patch("data.redis_client.redis_get", return_value=None),  # no marker → degraded
        ):
            body, code = health.build_health_report(deep=False)
        assert code == 200
        assert body["status"] == "degraded"

    def test_precompute_counts_present(self):
        with (
            patch("config.REQUIRE_REDIS", False),
            patch("data.redis_client.redis_available", return_value=False),
        ):
            body, _ = health.build_health_report()
        assert "precompute" in body
        assert "models_cached" in body["precompute"]
