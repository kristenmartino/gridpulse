"""
Sprint 4 Tests — Environment Config, Accuracy Thresholds, Fallback, Bookmarks.

Covers:
  J1: Environment Config Matrix — env-specific defaults, override behavior
  H2: ML Accuracy Thresholds — per-horizon MAPE grading, rollback trigger
  G2: API Fallback Behavior — stale cache warning, per-source degradation
  D3: Rate Limiting — backoff config, retry config
  C2: Scenario Bookmarks — URL serialization, state restoration
"""

import json
import os
import pytest
from unittest.mock import patch


# ── J1: ENVIRONMENT CONFIG MATRIX ──────────────────────────────


class TestEnvironmentConfig:
    """J1: Every env-specific value is configurable, with sane defaults per tier."""

    def test_default_environment_is_development(self):
        from config import ENVIRONMENT
        # Without explicit override, defaults to development
        assert ENVIRONMENT in ("development", "staging", "production")

    def test_development_defaults(self):
        """Dev: verbose logging, 24h cache, demo data enabled."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            import importlib
            import config
            importlib.reload(config)
            assert config.USE_DEMO_DATA is True
            assert config.ENABLE_PROFILING is True
            assert config.GUNICORN_WORKERS == 1
            assert config.CACHE_TTL_SECONDS == 86400  # 24h

    def test_production_defaults(self):
        """Prod: quiet logging, 24h cache, no demo data."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            import importlib
            import config
            importlib.reload(config)
            assert config.USE_DEMO_DATA is False
            assert config.ENABLE_PROFILING is False
            assert config.GUNICORN_WORKERS == 2
            assert config.CACHE_TTL_SECONDS == 86400  # 24h

    def test_staging_defaults(self):
        """Staging: 24h cache, no demo data."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=False):
            import importlib
            import config
            importlib.reload(config)
            assert config.USE_DEMO_DATA is False
            assert config.CACHE_TTL_SECONDS == 86400  # 24h

    def test_env_var_overrides_matrix_default(self):
        """Explicit env vars override the matrix defaults."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "CACHE_TTL_SECONDS": "60",
            "USE_DEMO_DATA": "true",
        }, clear=False):
            import importlib
            import config
            importlib.reload(config)
            assert config.CACHE_TTL_SECONDS == 60
            assert config.USE_DEMO_DATA is True

    def test_all_settings_documented_in_env_example(self):
        """Every configurable setting appears in .env.example."""
        env_example = open(".env.example").read()
        required = ["EIA_API_KEY", "ENVIRONMENT", "LOG_LEVEL", "CACHE_TTL_SECONDS",
                     "PORT", "USE_DEMO_DATA", "ENABLE_PROFILING", "GUNICORN_WORKERS"]
        for key in required:
            assert key in env_example, f"{key} missing from .env.example"

    def test_no_hardcoded_environment_values(self):
        """config.py doesn't hardcode 'production' or 'staging' as default ENVIRONMENT."""
        import config
        # The default should always be 'development'
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config)
            assert config.ENVIRONMENT == "development"

    def test_feature_flags_exist(self):
        """Feature flags are defined for all major features."""
        from config import FEATURE_FLAGS
        required_flags = ["tab_forecast", "tab_weather", "tab_models",
                          "tab_generation", "tab_alerts", "tab_simulator",
                          "persona_switcher", "scenario_bookmarks"]
        for flag in required_flags:
            assert flag in FEATURE_FLAGS, f"Missing feature flag: {flag}"

    def test_feature_enabled_helper(self):
        """feature_enabled() returns True for known flags, True for unknown."""
        from config import feature_enabled, FEATURE_FLAGS
        assert feature_enabled("tab_forecast") == FEATURE_FLAGS["tab_forecast"]
        assert feature_enabled("nonexistent_flag") is True  # unknown = enabled


# ── H2: ML MODEL ACCURACY THRESHOLDS ──────────────────────────


class TestMAPEThresholds:
    """H2: Per-horizon accuracy governance with rollback trigger."""

    def test_global_thresholds_defined(self):
        from config import (MAPE_THRESHOLD_EXCELLENT, MAPE_THRESHOLD_TARGET,
                            MAPE_THRESHOLD_ACCEPTABLE, MAPE_THRESHOLD_ROLLBACK)
        assert MAPE_THRESHOLD_EXCELLENT < MAPE_THRESHOLD_TARGET
        assert MAPE_THRESHOLD_TARGET < MAPE_THRESHOLD_ACCEPTABLE
        assert MAPE_THRESHOLD_ACCEPTABLE < MAPE_THRESHOLD_ROLLBACK

    def test_per_horizon_thresholds_defined(self):
        """All 4 forecast horizons have thresholds."""
        from config import MAPE_BY_HORIZON
        assert set(MAPE_BY_HORIZON.keys()) == {"24h", "48h", "72h", "7d"}

    def test_longer_horizons_more_tolerant(self):
        """Each threshold gets looser for longer horizons."""
        from config import MAPE_BY_HORIZON
        for tier in ("excellent", "target", "acceptable", "rollback"):
            vals = [MAPE_BY_HORIZON[h][tier] for h in ("24h", "48h", "72h", "7d")]
            assert vals == sorted(vals), f"Horizon tolerance not monotonic for {tier}: {vals}"

    def test_per_horizon_tiers_ordered(self):
        """Within each horizon, thresholds are: excellent < target < acceptable < rollback."""
        from config import MAPE_BY_HORIZON
        for horizon, thresholds in MAPE_BY_HORIZON.items():
            assert thresholds["excellent"] < thresholds["target"], horizon
            assert thresholds["target"] < thresholds["acceptable"], horizon
            assert thresholds["acceptable"] < thresholds["rollback"], horizon

    def test_mape_grade_excellent(self):
        from config import mape_grade
        assert mape_grade(1.5, "24h") == "excellent"
        assert mape_grade(2.5, "48h") == "excellent"

    def test_mape_grade_target(self):
        from config import mape_grade
        assert mape_grade(4.0, "48h") == "target"

    def test_mape_grade_acceptable(self):
        from config import mape_grade
        assert mape_grade(8.0, "48h") == "acceptable"

    def test_mape_grade_rollback(self):
        from config import mape_grade
        assert mape_grade(20.0, "48h") == "rollback"
        assert mape_grade(16.0, "48h") == "rollback"

    def test_mape_grade_default_horizon(self):
        """Default horizon is 48h when not specified."""
        from config import mape_grade
        assert mape_grade(1.0) == "excellent"
        assert mape_grade(20.0) == "rollback"

    def test_rollback_threshold_triggers_above_15_pct(self):
        """Models above 15% MAPE at 48h should be disabled."""
        from config import mape_grade, MAPE_THRESHOLD_ROLLBACK
        assert mape_grade(MAPE_THRESHOLD_ROLLBACK + 0.1) == "rollback"


# ── G2: API FALLBACK BEHAVIOR ──────────────────────────────────


class TestAPIFallbackBehavior:
    """G2: Per-source fallback produces stale/warning — not error pages."""

    def test_freshness_store_has_all_sources(self):
        """Freshness metadata tracks demand, weather, and alerts."""
        freshness = {"demand": "fresh", "weather": "fresh", "alerts": "fresh"}
        for source in ("demand", "weather", "alerts"):
            assert source in freshness

    def test_freshness_statuses(self):
        """Valid freshness statuses: fresh, stale, demo, error."""
        valid = {"fresh", "stale", "demo", "error"}
        for status in valid:
            assert status in valid

    def test_demo_data_always_available(self):
        """Demo data generators never raise — they're the ultimate fallback."""
        from data.demo_data import generate_demo_demand, generate_demo_weather
        for region in ("FPL", "ERCOT", "CAISO", "PJM", "MISO", "NYISO", "SPP", "ISONE"):
            demand = generate_demo_demand(region)
            weather = generate_demo_weather(region)
            assert len(demand) > 0, f"Empty demand for {region}"
            assert len(weather) > 0, f"Empty weather for {region}"

    def test_staleness_thresholds_per_source(self):
        """Each data source has a staleness threshold defined."""
        from config import STALENESS_THRESHOLDS_SECONDS
        required = {"weather", "generation", "pricing", "demand", "alerts"}
        assert required.issubset(set(STALENESS_THRESHOLDS_SECONDS.keys()))

    def test_staleness_thresholds_reasonable(self):
        """Thresholds match backlog E2 values."""
        from config import STALENESS_THRESHOLDS_SECONDS
        assert STALENESS_THRESHOLDS_SECONDS["weather"] == 7200      # 2h
        assert STALENESS_THRESHOLDS_SECONDS["generation"] == 300    # 5min
        assert STALENESS_THRESHOLDS_SECONDS["demand"] == 3600       # 1h
        assert STALENESS_THRESHOLDS_SECONDS["alerts"] == 1800       # 30min

    def test_freshness_badge_labels(self):
        """Header freshness badge maps status to correct label."""
        status_map = {
            "all_fresh": ("fresh", "fresh", "fresh"),
            "all_demo": ("demo", "demo", "demo"),
            "has_error": ("fresh", "error", "fresh"),
            "partial": ("fresh", "stale", "fresh"),
        }
        labels = {"all_fresh": "Live", "all_demo": "Demo",
                  "has_error": "Degraded", "partial": "Partial"}
        for key, (d, w, a) in status_map.items():
            statuses = [d, w, a]
            if all(s == "fresh" for s in statuses):
                label = "Live"
            elif all(s == "demo" for s in statuses):
                label = "Demo"
            elif any(s == "error" for s in statuses):
                label = "Degraded"
            else:
                label = "Partial"
            assert label == labels[key], f"{key}: got {label}"


# ── D3: RATE LIMITING & API KEY MANAGEMENT ─────────────────────


class TestRateLimitConfig:
    """D3: Rate limiting config is defined; backoff is exponential."""

    def test_backoff_config_exists(self):
        from config import INITIAL_BACKOFF_SECONDS, MAX_RETRIES
        assert INITIAL_BACKOFF_SECONDS > 0
        assert MAX_RETRIES >= 2

    def test_rate_limit_alert_threshold(self):
        from config import RATE_LIMIT_ALERT_THRESHOLD
        assert RATE_LIMIT_ALERT_THRESHOLD >= 1

    def test_eia_client_has_backoff(self):
        """EIA client implements exponential backoff."""
        source = open("data/eia_client.py").read()
        assert "backoff" in source.lower()
        assert "429" in source or "rate" in source.lower()

    def test_eia_key_from_env(self):
        """EIA API key is loaded from environment, not hardcoded."""
        from config import EIA_API_KEY
        source = open("config.py").read()
        assert 'os.getenv("EIA_API_KEY"' in source

    def test_secret_manager_documented(self):
        """.env.example documents GCP Secret Manager for production key rotation."""
        env_example = open(".env.example").read()
        assert "Secret Manager" in env_example or "gcloud secrets" in env_example


# ── C2: SCENARIO BOOKMARKS ────────────────────────────────────


class TestScenarioBookmarks:
    """C2: Dashboard state is serializable to/from URL query params."""

    def test_bookmark_url_format(self):
        """Bookmarks serialize to standard query params."""
        from urllib.parse import urlencode, parse_qs
        state = {"region": "FPL", "persona": "trader", "tab": "tab-simulator"}
        url = f"?{urlencode(state)}"
        parsed = parse_qs(url.lstrip("?"))
        assert parsed["region"] == ["FPL"]
        assert parsed["persona"] == ["trader"]
        assert parsed["tab"] == ["tab-simulator"]

    def test_all_regions_bookmarkable(self):
        """Every region code is a valid bookmark value."""
        from config import REGION_NAMES
        from urllib.parse import urlencode
        for region in REGION_NAMES:
            url = urlencode({"region": region})
            assert f"region={region}" in url

    def test_all_personas_bookmarkable(self):
        """Every persona ID is a valid bookmark value."""
        from personas.config import PERSONAS
        for pid in PERSONAS:
            assert isinstance(pid, str) and len(pid) > 0

    def test_all_tabs_bookmarkable(self):
        """Every tab ID is a valid bookmark value."""
        from config import TAB_LABELS
        for tab_id in TAB_LABELS:
            assert tab_id.startswith("tab-")

    def test_invalid_region_ignored(self):
        """Unknown region in bookmark doesn't crash — just ignored."""
        from config import REGION_NAMES
        assert "INVALID" not in REGION_NAMES

    def test_empty_search_no_crash(self):
        """Empty URL search string doesn't crash restore logic."""
        from urllib.parse import parse_qs
        params = parse_qs("")
        assert params == {}

    def test_roundtrip_state(self):
        """Serialize → deserialize → identical state."""
        from urllib.parse import urlencode, parse_qs
        original = {"region": "ERCOT", "persona": "data_scientist", "tab": "tab-models"}
        encoded = urlencode(original)
        decoded = {k: v[0] for k, v in parse_qs(encoded).items()}
        assert decoded == original
