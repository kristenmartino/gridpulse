"""Tests for the FastAPI serving layer."""
import ast
import inspect
import json
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def empty_cache(mock_redis):
    """Server with empty Redis cache."""
    from src.api.cache import ForecastCache
    from src.api import server

    cache = ForecastCache()
    cache.client = mock_redis
    server.cache = cache
    server.db_conn = None
    return TestClient(server.app)


@pytest.fixture
def loaded_cache(populated_redis):
    """Server with populated Redis cache."""
    from src.api.cache import ForecastCache
    from src.api import server

    cache = ForecastCache()
    cache.client = populated_redis
    server.cache = cache
    server.db_conn = None
    return TestClient(server.app)


class TestArchitecturalConstraint:
    """Verify the serving layer never imports model training code."""

    def test_server_does_not_import_models(self):
        """CRITICAL: server.py must never import from models/."""
        from src.api import server

        source = inspect.getsource(server)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("models"), (
                    f"server.py imports from models/: {node.module}"
                )

    def test_cache_does_not_import_models(self):
        """CRITICAL: cache.py must never import from models/."""
        from src.api import cache

        source = inspect.getsource(cache)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("models"), (
                    f"cache.py imports from models/: {node.module}"
                )

    def test_cache_does_not_import_data(self):
        """cache.py must not import from data/ (no v1 dependencies)."""
        from src.api import cache

        source = inspect.getsource(cache)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("data"), (
                    f"cache.py imports from data/: {node.module}"
                )


class TestHealthEndpoint:

    def test_health_returns_stale_when_empty(self, empty_cache):
        """Health check returns stale status when no scorer has run."""
        response = empty_cache.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stale"
        assert data["cache_healthy"] is False
        assert data["regions_available"] == 0

    def test_health_returns_healthy_when_populated(self, loaded_cache):
        """Health check returns healthy when scorer has run recently."""
        response = loaded_cache.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["cache_healthy"] is True
        assert data["regions_available"] == 8


class TestForecastEndpoints:

    def test_forecast_returns_503_when_empty(self, empty_cache):
        """If Redis has no data, API returns 503 — NOT a fallback computation."""
        response = empty_cache.get("/forecasts/ERCOT")
        assert response.status_code == 503
        assert "pipeline" in response.json()["detail"].lower()

    def test_forecast_returns_200_when_populated(self, loaded_cache):
        """Pre-computed forecast is returned from Redis."""
        response = loaded_cache.get("/forecasts/ERCOT?granularity=1h")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert data["granularity"] == "1h"
        assert len(data["forecasts"]) == 2

    def test_forecast_unknown_region_returns_404(self, loaded_cache):
        """Unknown region returns 404."""
        response = loaded_cache.get("/forecasts/ZZZZ")
        assert response.status_code == 404

    def test_all_forecasts_returns_503_when_empty(self, empty_cache):
        """All-regions endpoint returns 503 when Redis is empty."""
        response = empty_cache.get("/forecasts")
        assert response.status_code == 503

    def test_all_forecasts_returns_200_when_populated(self, loaded_cache):
        """All-regions endpoint returns list of forecasts."""
        response = loaded_cache.get("/forecasts?granularity=1h")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1


class TestMetadataEndpoints:

    def test_regions_endpoint(self, empty_cache):
        """/regions returns the 8 grid regions."""
        response = empty_cache.get("/regions")
        assert response.status_code == 200
        data = response.json()
        assert "regions" in data
        assert len(data["regions"]) == 8

    def test_granularities_endpoint(self, empty_cache):
        """/granularities returns the 3 time buckets."""
        response = empty_cache.get("/granularities")
        assert response.status_code == 200
        data = response.json()
        assert "granularities" in data
        assert "15min" in data["granularities"]
        assert "1h" in data["granularities"]
        assert "1d" in data["granularities"]
