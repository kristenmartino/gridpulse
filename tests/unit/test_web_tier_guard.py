"""Web-tier operational guard (#253): API rate limit, /health + /metrics gating.

- ``/api/v1/*`` returns 429 (+ Retry-After, no-store) when the limiter blocks,
  passes through when allowed, and is unthrottled when inactive (dev).
- Public ``/health`` is liveness-only; the detailed body (Redis state,
  last-scored, cache counts) + ``?deep=1`` are gated behind the /metrics IP
  allowlist. ``/metrics`` itself is allowlist-gated.
- ``MAX_CONTENT_LENGTH`` is wired to ``config.MAX_REQUEST_BYTES``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask

from api import api_v1
from ratelimit import RateLimitResult


@pytest.fixture()
def api_client():
    app = Flask(__name__)
    app.register_blueprint(api_v1)
    return app.test_client()


class TestApiRateLimit:
    def test_429_when_active_and_over_limit(self, api_client):
        with (
            patch("config.rate_limit_active", return_value=True),
            patch("ratelimit.check_rate_limit", return_value=RateLimitResult(False, 0, 42)),
        ):
            r = api_client.get("/api/v1")
        assert r.status_code == 429
        assert r.headers["Retry-After"] == "42"
        assert r.headers["Cache-Control"] == "no-store"  # never cache a rejection
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        body = r.get_json()
        assert body["error"] == "rate_limited"
        assert body["retry_after_seconds"] == 42

    def test_allowed_passes_through(self, api_client):
        with (
            patch("config.rate_limit_active", return_value=True),
            patch("ratelimit.check_rate_limit", return_value=RateLimitResult(True, 5, 0)),
        ):
            r = api_client.get("/api/v1")
        assert r.status_code == 200
        assert r.get_json()["version"] == "v1"

    def test_exempt_ip_bypasses_limit(self, api_client):
        """A trusted (shared-NAT) IP skips the limiter even when it would block —
        the exemption is checked before check_rate_limit runs."""
        with (
            patch("config.rate_limit_active", return_value=True),
            patch("ratelimit.is_exempt", return_value=True),
            patch("ratelimit.check_rate_limit", return_value=RateLimitResult(False, 0, 99)),
        ):
            r = api_client.get("/api/v1")
        assert r.status_code == 200

    def test_inactive_is_unthrottled(self, api_client):
        """When the guard is inactive (dev), the limiter never runs even if it
        would block — the request passes through."""
        with (
            patch("config.rate_limit_active", return_value=False),
            patch("ratelimit.check_rate_limit", return_value=RateLimitResult(False, 0, 99)),
        ):
            r = api_client.get("/api/v1")
        assert r.status_code == 200


@pytest.fixture(scope="module")
def server_client():
    from app import server

    return server.test_client()


class TestHealthGating:
    def test_public_health_is_liveness_only(self, server_client):
        with patch("app._is_internal_caller", return_value=False):
            r = server_client.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        # Only the top-level status — no checks/precompute/deep internals.
        assert set(body.keys()) == {"status"}

    def test_public_deep_is_ignored(self, server_client):
        """A public caller can neither trigger the deep forecast read nor see
        the detailed body."""
        with patch("app._is_internal_caller", return_value=False):
            r = server_client.get("/health?deep=1")
        assert set(r.get_json().keys()) == {"status"}

    def test_internal_health_is_detailed(self, server_client):
        with patch("app._is_internal_caller", return_value=True):
            r = server_client.get("/health")
        body = r.get_json()
        assert "checks" in body
        assert "precompute" in body


class TestMetricsGating:
    def test_forbidden_for_public(self, server_client):
        with patch("app._is_internal_caller", return_value=False):
            r = server_client.get("/metrics")
        assert r.status_code == 403

    def test_allowed_for_internal(self, server_client):
        with patch("app._is_internal_caller", return_value=True):
            r = server_client.get("/metrics")
        assert r.status_code == 200


class TestMaxContentLength:
    def test_config_wired(self):
        import config
        from app import server

        assert server.config["MAX_CONTENT_LENGTH"] == config.MAX_REQUEST_BYTES
        assert config.MAX_REQUEST_BYTES > 0
