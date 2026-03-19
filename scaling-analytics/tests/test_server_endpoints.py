"""Tests for all new API endpoints in server.py."""
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
    server.db_conn = None  # No DB for unit tests
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


# ── Architectural constraints ──────────────────

class TestArchitecturalConstraint:
    """Verify the serving layer's import isolation."""

    def test_server_module_level_no_models_import(self):
        """server.py must never import from models/ at module level."""
        from src.api import server

        source = inspect.getsource(server)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Allow lazy imports inside functions
                if isinstance(node, ast.ImportFrom):
                    # Check if this import is at module level (col_offset == 0)
                    if hasattr(node, "col_offset") and node.col_offset == 0:
                        assert not node.module.startswith("models"), (
                            f"server.py has module-level import from models/: {node.module}"
                        )

    def test_cache_does_not_import_models(self):
        """cache.py must never import from models/."""
        from src.api import cache

        source = inspect.getsource(cache)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("models"), (
                    f"cache.py imports from models/: {node.module}"
                )

    def test_cache_does_not_import_data(self):
        """cache.py must not import from data/."""
        from src.api import cache

        source = inspect.getsource(cache)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("data"), (
                    f"cache.py imports from data/: {node.module}"
                )


# ── Actuals (Tab 1) ───────────────────────────

class TestActualsEndpoints:

    def test_actuals_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/actuals/ERCOT")
        assert response.status_code == 503

    def test_actuals_returns_200_when_populated(self, loaded_cache):
        response = loaded_cache.get("/actuals/ERCOT")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert "demand_mw" in data
        assert "timestamps" in data

    def test_actuals_hours_parameter(self, loaded_cache):
        response = loaded_cache.get("/actuals/ERCOT?hours=24")
        assert response.status_code == 200

    def test_actuals_unknown_region_returns_404(self, loaded_cache):
        response = loaded_cache.get("/actuals/ZZZZ")
        assert response.status_code == 404

    def test_weather_overlay_returns_200(self, loaded_cache):
        response = loaded_cache.get("/actuals/ERCOT/weather-overlay")
        assert response.status_code == 200
        data = response.json()
        assert "temperature_2m" in data

    def test_weather_overlay_503_when_empty(self, empty_cache):
        response = empty_cache.get("/actuals/ERCOT/weather-overlay")
        assert response.status_code == 503


# ── Backtests (Tab 3) ─────────────────────────

class TestBacktestEndpoints:

    def test_backtest_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/backtests/ERCOT?horizon=24")
        assert response.status_code == 503

    def test_backtest_returns_200_when_populated(self, loaded_cache):
        response = loaded_cache.get("/backtests/ERCOT?horizon=24")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == 24
        assert "metrics" in data
        assert "actual" in data

    def test_backtest_with_model_filter(self, loaded_cache):
        response = loaded_cache.get("/backtests/ERCOT?horizon=24&model=xgboost")
        assert response.status_code == 200
        data = response.json()
        assert "xgboost" in data["metrics"]

    def test_residuals_returns_200(self, loaded_cache):
        response = loaded_cache.get("/backtests/ERCOT/residuals?horizon=24")
        assert response.status_code == 200
        data = response.json()
        assert "residuals" in data
        assert data["region"] == "ERCOT"

    def test_error_by_hour_returns_200(self, loaded_cache):
        response = loaded_cache.get("/backtests/ERCOT/error-by-hour?horizon=24")
        assert response.status_code == 200
        data = response.json()
        assert "error_by_hour" in data


# ── Weather (Tab 4) ───────────────────────────

class TestWeatherEndpoints:

    def test_weather_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/weather/ERCOT")
        assert response.status_code == 503

    def test_weather_returns_200_when_populated(self, loaded_cache):
        response = loaded_cache.get("/weather/ERCOT")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert "temperature_2m" in data

    def test_weather_correlation_returns_200(self, loaded_cache):
        response = loaded_cache.get("/weather/ERCOT/correlation")
        assert response.status_code == 200
        data = response.json()
        assert "correlations" in data
        assert "temperature_2m" in data["correlations"]


# ── Models (Tab 5) ────────────────────────────

class TestModelEndpoints:

    def test_model_metrics_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/models/ERCOT/metrics")
        assert response.status_code == 503

    def test_model_metrics_returns_200(self, loaded_cache):
        response = loaded_cache.get("/models/ERCOT/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert "metrics" in data

    def test_model_weights_returns_200(self, loaded_cache):
        response = loaded_cache.get("/models/ERCOT/weights")
        assert response.status_code == 200
        data = response.json()
        assert "weights" in data
        assert data["weights"]["xgboost"] == 0.50

    def test_feature_importance_returns_200(self, loaded_cache):
        response = loaded_cache.get("/models/ERCOT/feature-importance")
        assert response.status_code == 200


# ── Generation (Tab 6) ────────────────────────

class TestGenerationEndpoints:

    def test_generation_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/generation/ERCOT")
        assert response.status_code == 503

    def test_generation_returns_200(self, loaded_cache):
        response = loaded_cache.get("/generation/ERCOT")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert "wind" in data
        assert "renewable_pct" in data

    def test_capacity_factors_returns_200(self, loaded_cache):
        response = loaded_cache.get("/generation/ERCOT/capacity-factors")
        assert response.status_code == 200
        data = response.json()
        assert "wind_cf_pct" in data
        assert "solar_cf_pct" in data


# ── Alerts (Tab 7) ────────────────────────────

class TestAlertEndpoints:

    def test_alerts_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/alerts/ERCOT")
        assert response.status_code == 503

    def test_alerts_returns_200(self, loaded_cache):
        response = loaded_cache.get("/alerts/ERCOT")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert data["stress_score"] == 45
        assert data["stress_label"] == "Elevated"
        assert len(data["alerts"]) == 2

    def test_anomalies_returns_200(self, loaded_cache):
        response = loaded_cache.get("/alerts/ERCOT/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert "anomalies" in data
        # One alert has type="anomaly"
        assert len(data["anomalies"]) == 1

    def test_extreme_events_returns_200(self, loaded_cache):
        response = loaded_cache.get("/alerts/ERCOT/extreme-events")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        # One alert has severity="critical"
        assert len(data["events"]) == 1


# ── Scenarios (Tab 8) ─────────────────────────

class TestScenarioEndpoints:

    def test_presets_list_returns_200(self, loaded_cache):
        response = loaded_cache.get("/scenarios/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data

    def test_preset_returns_200_when_populated(self, loaded_cache):
        response = loaded_cache.get("/scenarios/presets/winter_storm_uri?region=ERCOT")
        assert response.status_code == 200
        data = response.json()
        assert data["region"] == "ERCOT"
        assert "baseline" in data
        assert "scenario" in data

    def test_preset_returns_503_for_missing(self, loaded_cache):
        response = loaded_cache.get("/scenarios/presets/nonexistent?region=ERCOT")
        assert response.status_code == 503

    def test_simulate_requires_overrides(self, loaded_cache):
        """POST with no weather overrides returns 400."""
        response = loaded_cache.post(
            "/scenarios/simulate",
            json={"region": "ERCOT", "duration_hours": 24},
        )
        assert response.status_code == 400
        assert "override" in response.json()["detail"].lower()

    def test_simulate_rejects_unknown_region(self, loaded_cache):
        response = loaded_cache.post(
            "/scenarios/simulate",
            json={"region": "ZZZZ", "temperature_2m": 105},
        )
        assert response.status_code == 404


# ── Personas ──────────────────────────────────

class TestPersonaEndpoints:

    def test_list_personas_returns_200(self, loaded_cache):
        response = loaded_cache.get("/personas")
        assert response.status_code == 200
        data = response.json()
        assert "personas" in data

    def test_unknown_persona_returns_404(self, loaded_cache):
        response = loaded_cache.get("/personas/nonexistent")
        assert response.status_code == 404


# ── Cross-cutting ─────────────────────────────

class TestCrossCuttingEndpoints:

    def test_news_returns_200(self, loaded_cache):
        response = loaded_cache.get("/news")
        assert response.status_code == 200
        data = response.json()
        assert "articles" in data
        assert len(data["articles"]) == 1

    def test_news_returns_503_when_empty(self, empty_cache):
        response = empty_cache.get("/news")
        assert response.status_code == 503

    def test_data_freshness_returns_200(self, loaded_cache):
        """data-freshness returns 200 even without DB (graceful degradation)."""
        response = loaded_cache.get("/data-freshness")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data

    def test_audit_returns_503_without_db(self, loaded_cache):
        """audit returns 503 when no DB connection."""
        response = loaded_cache.get("/audit/ERCOT")
        assert response.status_code == 503

    def test_regions_endpoint(self, empty_cache):
        response = empty_cache.get("/regions")
        assert response.status_code == 200
        assert len(response.json()["regions"]) == 8

    def test_granularities_endpoint(self, empty_cache):
        response = empty_cache.get("/granularities")
        assert response.status_code == 200
        data = response.json()
        assert "15min" in data["granularities"]
        assert "1h" in data["granularities"]
        assert "1d" in data["granularities"]


# ── CORS ──────────────────────────────────────

class TestCORS:

    def test_cors_allows_post(self, loaded_cache):
        """CORS middleware allows POST methods (needed for /scenarios/simulate)."""
        response = loaded_cache.options(
            "/scenarios/simulate",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
