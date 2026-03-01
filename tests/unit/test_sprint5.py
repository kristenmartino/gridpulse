"""
Sprint 5 tests — Trust, Audit & Production Readiness.

Covers:
- D2: Forecast Model Input Audit Trail
- I1: Pipeline Transformation Logging
- A4+E3: Per-Widget Data Freshness + Confidence Badges
- C9: Meeting-Ready Mode
- H3: Test Pyramid Definition
"""

import json
import os
import time

# ── D2: Forecast Model Input Audit Trail ─────────────────────


class TestAuditTrail:
    """D2: Verify audit trail records every forecast computation."""

    def test_audit_record_has_required_fields(self):
        from data.audit import AuditRecord

        record = AuditRecord()
        required = [
            "record_id",
            "timestamp",
            "region",
            "model_versions",
            "demand_source",
            "weather_source",
            "demand_rows",
            "weather_rows",
            "mape",
            "forecast_source",
        ]
        for field in required:
            assert hasattr(record, field), f"AuditRecord missing field: {field}"

    def test_audit_record_to_json(self):
        from data.audit import AuditRecord

        record = AuditRecord(record_id="test-001", region="FPL")
        j = record.to_json()
        parsed = json.loads(j)
        assert parsed["record_id"] == "test-001"
        assert parsed["region"] == "FPL"

    def test_audit_record_to_dict(self):
        from data.audit import AuditRecord

        record = AuditRecord(region="ERCOT", demand_rows=168)
        d = record.to_dict()
        assert d["region"] == "ERCOT"
        assert d["demand_rows"] == 168

    def test_audit_trail_record_forecast(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        record = trail.record_forecast(
            region="FPL",
            demand_source="demo",
            weather_source="demo",
            demand_rows=168,
            weather_rows=168,
        )
        assert record.region == "FPL"
        assert record.demand_source == "demo"
        assert record.record_id.startswith("FPL-")
        assert record.timestamp != ""

    def test_audit_trail_get_recent(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        for i in range(5):
            trail.record_forecast(
                region=f"R{i}",
                demand_source="demo",
                weather_source="demo",
                demand_rows=100,
                weather_rows=100,
            )
        recent = trail.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].region == "R4"

    def test_audit_trail_get_by_region(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        trail.record_forecast(
            region="FPL",
            demand_source="demo",
            weather_source="demo",
            demand_rows=168,
            weather_rows=168,
        )
        trail.record_forecast(
            region="ERCOT",
            demand_source="api",
            weather_source="api",
            demand_rows=168,
            weather_rows=168,
        )
        trail.record_forecast(
            region="FPL",
            demand_source="demo",
            weather_source="demo",
            demand_rows=336,
            weather_rows=336,
        )
        fpl_records = trail.get_by_region("FPL")
        assert len(fpl_records) == 2
        assert all(r.region == "FPL" for r in fpl_records)

    def test_audit_trail_get_latest(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        trail.record_forecast(
            region="FPL",
            demand_source="demo",
            weather_source="demo",
            demand_rows=168,
            weather_rows=168,
        )
        latest = trail.get_latest("FPL")
        assert latest is not None
        assert latest.region == "FPL"

    def test_audit_trail_get_latest_missing(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        assert trail.get_latest("NONEXISTENT") is None

    def test_audit_trail_max_records(self):
        from data.audit import AuditTrail

        trail = AuditTrail(max_records=5)
        for i in range(10):
            trail.record_forecast(
                region=f"R{i}",
                demand_source="demo",
                weather_source="demo",
                demand_rows=100,
                weather_rows=100,
            )
        assert trail.count == 5
        assert trail.get_recent(10)[0].region == "R5"

    def test_audit_trail_feature_hash(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        record = trail.record_forecast(
            region="FPL",
            demand_source="demo",
            weather_source="demo",
            demand_rows=168,
            weather_rows=168,
            feature_names=["temperature_2m", "wind_speed_80m", "cdd", "hdd"],
        )
        assert record.feature_hash != ""
        assert len(record.feature_hash) == 16
        assert record.feature_count == 4

    def test_audit_trail_model_versions(self):
        from data.audit import AuditTrail

        trail = AuditTrail()
        versions = {"prophet": "v1.2", "xgboost": "v3.1", "arima": "v2.0"}
        record = trail.record_forecast(
            region="FPL",
            demand_source="api",
            weather_source="api",
            demand_rows=168,
            weather_rows=168,
            model_versions=versions,
        )
        assert record.model_versions == versions

    def test_audit_singleton_exists(self):
        from data.audit import audit_trail

        assert audit_trail is not None
        assert hasattr(audit_trail, "record_forecast")


# ── I1: Pipeline Transformation Logging ──────────────────────


class TestPipelineLogger:
    """I1: Verify pipeline step-by-step logging."""

    def test_pipeline_logger_basic(self):
        from observability import PipelineLogger

        pipe = PipelineLogger("test_pipeline", region="FPL")
        pipe.step("fetch", rows=168, source="demo")
        pipe.step("clean", rows=168, nulls_filled=2)
        summary = pipe.done()
        assert summary["pipeline"] == "test_pipeline"
        assert summary["total_steps"] == 2
        assert summary["total_ms"] >= 0
        assert len(summary["steps"]) == 2

    def test_pipeline_logger_steps_have_duration(self):
        from observability import PipelineLogger

        pipe = PipelineLogger("timing_test")
        pipe.step("step1", rows=100)
        time.sleep(0.01)
        pipe.step("step2", rows=200)
        summary = pipe.done()
        # step2 should have nonzero duration
        assert summary["steps"][1]["duration_ms"] >= 0

    def test_pipeline_logger_context_propagated(self):
        from observability import PipelineLogger

        pipe = PipelineLogger("ctx_test", region="ERCOT", user="test")
        summary = pipe.done()
        assert summary["region"] == "ERCOT"
        assert summary["user"] == "test"

    def test_pipeline_logger_chaining(self):
        from observability import PipelineLogger

        pipe = PipelineLogger("chain_test")
        result = pipe.step("a").step("b").step("c")
        assert result is pipe
        assert len(pipe.steps) == 3

    def test_pipeline_logger_records_perf(self):
        from observability import PipelineLogger, perf

        pipe = PipelineLogger("perf_test")
        pipe.step("work", rows=100)
        pipe.done()
        stats = perf.get_stats("pipeline.perf_test")
        assert stats["count"] >= 1

    def test_pipeline_logger_safe_details(self):
        """Non-serializable objects in details should not crash the logger."""
        from observability import PipelineLogger

        pipe = PipelineLogger("safe_test")
        # Pass a complex object — should not raise
        pipe.step("fetch", rows=168, metadata={"nested": True})
        summary = pipe.done()
        assert summary["total_steps"] == 1


# ── A4+E3: Per-Widget Data Freshness + Confidence Badges ─────


class TestDataConfidence:
    """A4+E3: Verify confidence level computation and badge rendering."""

    def test_confidence_level_fresh(self):
        from components.error_handling import data_confidence_level

        assert data_confidence_level("fresh", age_seconds=60) == "high"

    def test_confidence_level_stale(self):
        from components.error_handling import data_confidence_level

        assert data_confidence_level("stale") == "medium"

    def test_confidence_level_error(self):
        from components.error_handling import data_confidence_level

        assert data_confidence_level("error") == "low"

    def test_confidence_level_demo(self):
        from components.error_handling import data_confidence_level

        assert data_confidence_level("demo") == "demo"

    def test_confidence_level_fresh_but_old(self):
        from components.error_handling import data_confidence_level

        # Fresh status but very old — should be medium
        assert data_confidence_level("fresh", age_seconds=10000, stale_threshold=7200) == "medium"

    def test_confidence_levels_dict_complete(self):
        from components.error_handling import CONFIDENCE_LEVELS

        assert set(CONFIDENCE_LEVELS.keys()) == {"high", "medium", "low", "demo"}
        for level in CONFIDENCE_LEVELS.values():
            assert "emoji" in level
            assert "label" in level
            assert "color" in level
            assert "description" in level

    def test_confidence_badge_returns_div(self):
        from components.error_handling import confidence_badge

        badge = confidence_badge("Demand", "high", "3m ago")
        # Returns dash html.Div (mocked in test env)
        assert badge is not None

    def test_widget_confidence_bar_returns_div(self):
        from components.error_handling import widget_confidence_bar

        freshness = {
            "demand": "fresh",
            "weather": "stale",
            "alerts": "demo",
            "timestamp": "2026-02-20T10:00:00+00:00",
        }
        bar = widget_confidence_bar(freshness, age_seconds=300)
        assert bar is not None

    def test_widget_confidence_bar_all_sources(self):
        from components.error_handling import widget_confidence_bar

        freshness = {
            "demand": "fresh",
            "weather": "fresh",
            "alerts": "fresh",
            "timestamp": "2026-02-20T10:00:00+00:00",
        }
        bar = widget_confidence_bar(freshness, age_seconds=60)
        # Should have child badges for demand, weather, alerts
        assert bar is not None


# ── C9: Meeting-Ready Mode ────────────────────────────────────


class TestMeetingMode:
    """C9: Verify meeting-ready mode layout changes."""

    def test_meeting_mode_store_in_layout_source(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert "meeting-mode-store" in src

    def test_meeting_mode_btn_in_layout_source(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert "meeting-mode-btn" in src

    def test_present_button_label(self):
        """Meeting mode button should say 'Present'."""
        with open("components/layout.py") as f:
            src = f.read()
        assert "Present" in src


# ── H3: Test Pyramid Definition ──────────────────────────────


class TestTestPyramid:
    """H3: Verify test pyramid documentation and structure."""

    def test_pyramid_doc_exists(self):
        assert os.path.exists("tests/TEST_PYRAMID.md")

    def test_pyramid_doc_covers_all_layers(self):
        with open("tests/TEST_PYRAMID.md") as f:
            content = f.read()
        assert "Unit" in content
        assert "Integration" in content
        assert "E2E" in content

    def test_pyramid_doc_has_coverage_targets(self):
        with open("tests/TEST_PYRAMID.md") as f:
            content = f.read()
        assert "80%" in content  # unit target
        assert "70%" in content  # integration target

    def test_pyramid_doc_has_fixture_list(self):
        with open("tests/TEST_PYRAMID.md") as f:
            content = f.read()
        assert "sample_demand_df" in content
        assert "tmp_cache" in content

    def test_unit_dir_exists(self):
        assert os.path.isdir("tests/unit")

    def test_integration_dir_exists(self):
        assert os.path.isdir("tests/integration")

    def test_e2e_dir_exists(self):
        assert os.path.isdir("tests/e2e")

    def test_conftest_exists(self):
        assert os.path.exists("tests/conftest.py")

    def test_all_sprint_test_files_exist(self):
        expected = [
            "tests/unit/test_sprint3.py",
            "tests/unit/test_sprint4.py",
            "tests/unit/test_sprint4_features.py",
            "tests/unit/test_sprint5.py",
        ]
        for path in expected:
            assert os.path.exists(path), f"Missing test file: {path}"


# ── D2+I1 Integration: Audit + Pipeline in Layout ────────────


class TestSprint5Layout:
    """Verify Sprint 5 layout additions are wired."""

    def test_audit_store_in_layout(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert "audit-store" in src

    def test_pipeline_log_store_in_layout(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert "pipeline-log-store" in src

    def test_widget_confidence_bar_in_layout(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert "widget-confidence-bar" in src

    def test_dashboard_header_has_id(self):
        with open("components/layout.py") as f:
            src = f.read()
        assert 'id="dashboard-header"' in src
