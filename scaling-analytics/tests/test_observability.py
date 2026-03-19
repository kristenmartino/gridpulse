"""Tests for the PipelineLogger observability module."""
import time

import pytest


class TestPipelineLogger:

    def test_step_records_entry(self):
        """step() records a step with elapsed time."""
        from src.observability import PipelineLogger

        pl = PipelineLogger("test_pipeline", region="ERCOT")
        pl.step("fetch_data", rows=100)
        pl.step("transform", features=43)

        assert len(pl.steps) == 2
        assert pl.steps[0]["step"] == "fetch_data"
        assert pl.steps[0]["rows"] == 100
        assert pl.steps[1]["step"] == "transform"
        assert pl.steps[1]["features"] == 43
        assert "elapsed_ms" in pl.steps[0]

    def test_done_returns_summary(self):
        """done() returns complete summary with total_ms."""
        from src.observability import PipelineLogger

        pl = PipelineLogger("test_pipeline")
        pl.step("step1")
        summary = pl.done()

        assert summary["pipeline_name"] == "test_pipeline"
        assert "total_ms" in summary
        assert summary["total_ms"] >= 0
        assert len(summary["steps"]) == 1
        assert "completed_at" in summary

    def test_elapsed_ms_is_positive(self):
        """Elapsed time should be non-negative."""
        from src.observability import PipelineLogger

        pl = PipelineLogger("test_pipeline")
        time.sleep(0.01)
        pl.step("delayed_step")

        assert pl.steps[0]["elapsed_ms"] >= 0

    def test_chaining(self):
        """step() returns self for chaining."""
        from src.observability import PipelineLogger

        pl = PipelineLogger("test_pipeline")
        result = pl.step("step1").step("step2")

        assert result is pl
        assert len(pl.steps) == 2

    def test_context_preserved(self):
        """Context kwargs are preserved in summary."""
        from src.observability import PipelineLogger

        pl = PipelineLogger("test_pipeline", region="ERCOT", scored_at="2025-01-15T12:00:00+00:00")
        summary = pl.done()

        assert summary["context"]["region"] == "ERCOT"
        assert summary["context"]["scored_at"] == "2025-01-15T12:00:00+00:00"
