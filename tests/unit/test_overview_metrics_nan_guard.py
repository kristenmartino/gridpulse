"""Regression tests for _build_overview_metrics_items NaN handling.

EIA-930 has a publishing lag for the most recent hour, especially for
newer / smaller BAs (PSCO, NEVP, AZPS). Until the row catches up the
hourly demand series ends with a NaN. The hero "Now" metric was reading
``df["demand_mw"].iloc[-1]`` directly and rendering the resulting NaN
as the literal string "nan" in the UI (PSCO, 2026-05-02). Now it pulls
from the same NaN-aware ``nonzero`` filter the other metrics use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hour_range(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")


class TestNowMetricNaNGuard:
    def test_clean_data_renders_latest_value(self):
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(48)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": [5000.0] * 47 + [5500.0]})
        items = _build_overview_metrics_items(df)
        now = next(i for i in items if i["label"] == "Now")
        assert now["value"] == "5,500"

    def test_trailing_nan_falls_back_to_latest_real_reading(self):
        """The PSCO bug: latest hour is NaN → metric should show the most
        recent real value, not "nan"."""
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(48)
        demand = [5000.0] * 47 + [float("nan")]
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        items = _build_overview_metrics_items(df)
        now = next(i for i in items if i["label"] == "Now")
        assert now["value"] == "5,000"
        assert "nan" not in now["value"].lower()

    def test_trailing_zero_also_skipped(self):
        """EIA's other missing-observation marker (zero) is filtered too."""
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(48)
        demand = [5000.0] * 47 + [0.0]
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        items = _build_overview_metrics_items(df)
        now = next(i for i in items if i["label"] == "Now")
        assert now["value"] == "5,000"

    def test_all_nan_renders_em_dash_not_nan(self):
        """Defensive: a fully sparse series → "—" placeholder, never "nan"."""
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(24)
        df = pd.DataFrame({"timestamp": ts, "demand_mw": [float("nan")] * 24})
        items = _build_overview_metrics_items(df)
        now = next(i for i in items if i["label"] == "Now")
        trend = next(i for i in items if i["label"] == "24h Trend")
        assert now["value"] == "—"
        assert trend["value"] == "—"
        assert "nan" not in now["value"].lower()

    def test_trend_skips_nan_in_ago_24h_position(self):
        """Trend's 24-hours-ago point is also NaN-aware: a NaN at index
        -25 of the raw frame must not produce a NaN trend percentage."""
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(48)
        demand = [5000.0] * 48
        # Inject NaN exactly at the -25 position the old code used
        demand[-25] = float("nan")
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        items = _build_overview_metrics_items(df)
        trend = next(i for i in items if i["label"] == "24h Trend")
        # nonzero series still has 47 entries; the NaN-aware iloc[-25] of
        # nonzero falls within the real data so trend is finite.
        assert "nan" not in trend["value"].lower()
        assert "%" in trend["value"]

    def test_empty_dataframe_returns_placeholder(self):
        """Already-handled fast path: empty df → placeholder stays the same."""
        from components.callbacks import _build_overview_metrics_items

        items = _build_overview_metrics_items(pd.DataFrame(columns=["timestamp", "demand_mw"]))
        assert all(i["value"] == "—" for i in items)

    def test_none_input_returns_placeholder(self):
        from components.callbacks import _build_overview_metrics_items

        items = _build_overview_metrics_items(None)
        assert all(i["value"] == "—" for i in items)

    def test_nan_with_numpy_dtype(self):
        """np.nan in a float64 column → same behavior as a Python float NaN."""
        from components.callbacks import _build_overview_metrics_items

        ts = _hour_range(48)
        demand = np.full(48, 5000.0)
        demand[-1] = np.nan
        df = pd.DataFrame({"timestamp": ts, "demand_mw": demand})
        items = _build_overview_metrics_items(df)
        now = next(i for i in items if i["label"] == "Now")
        assert now["value"] == "5,000"


class TestArtifactExclusionDisclosure:
    """#309 PR 2 — when the scoring job excluded readings, the tiles must SAY so.

    The series arrives pre-cleaned (NaN at excluded hours), so the values
    render correctly with or without this; the disclosure is the part only
    these pins protect — silently-right numbers are how the last three
    display bugs shipped.
    """

    def _df(self):
        ts = pd.date_range("2026-07-16 00:00", periods=30, freq="h", tz="UTC")
        mw = [3300.0] * 29 + [float("nan")]  # cleaned tail
        return pd.DataFrame({"timestamp": ts, "demand_mw": mw})

    def _exclusions(self):
        return [
            {
                "ts": "2026-07-17T05:00:00+00:00",
                "mw": 730.0,
                "reason": "79% single-hour drop to 22% of the daily median",
            }
        ]

    def test_now_tile_discloses_exclusion(self):
        from components.callbacks import _build_overview_metrics_items

        items = _build_overview_metrics_items(self._df(), self._exclusions())
        now = next(i for i in items if i["label"] == "Now")

        assert now["value"] == "3,300"  # last real reading, not the artifact
        assert "1 newer reading excluded" in (now.get("subtext") or "")
        assert "730" in (now.get("help") or "")
        assert "PLAUSIBLE" in (now.get("help") or "")

    def test_no_exclusions_renders_exactly_as_before(self):
        from components.callbacks import _build_overview_metrics_items

        baseline = _build_overview_metrics_items(self._df())
        with_empty = _build_overview_metrics_items(self._df(), [])
        assert baseline == with_empty

    def test_summary_sentence_names_the_exclusion(self):
        from components._callbacks_overview import _build_overview_insight

        card = _build_overview_insight("LDWP", self._df(), "grid_ops", self._exclusions())
        text = str(card)
        assert "730" in text
        assert "excluded" in text
        assert "artifact" in text

    def test_summary_without_exclusions_has_no_artifact_prose(self):
        from components._callbacks_overview import _build_overview_insight

        card = _build_overview_insight("LDWP", self._df(), "grid_ops", [])
        assert "excluded" not in str(card)


class TestExclusionWireFromFreshnessStore:
    """#309 follow-up — the wire the render pins missed.

    The data-freshness-store holds a JSON STRING; the first cut checked
    isinstance(dict) and silently dropped the disclosure in prod while every
    builder-level pin stayed green. These pins feed the wire the REAL store
    format so that gap cannot reopen.
    """

    def test_parses_the_real_store_format_a_json_string(self):
        import json

        from components._callbacks_overview import _exclusions_from_freshness

        payload = json.dumps(
            {
                "demand": "fresh",
                "artifact_excluded": [{"ts": "t", "mw": 730.0, "reason": "r"}],
            }
        )
        assert _exclusions_from_freshness(payload) == [{"ts": "t", "mw": 730.0, "reason": "r"}]

    def test_tolerates_dict_and_junk(self):
        from components._callbacks_overview import _exclusions_from_freshness

        assert _exclusions_from_freshness({"artifact_excluded": [{"mw": 1}]}) == [{"mw": 1}]
        assert _exclusions_from_freshness(None) == []
        assert _exclusions_from_freshness("not json{") == []
        assert _exclusions_from_freshness('{"artifact_excluded": "corrupt"}') == []
        assert _exclusions_from_freshness('"just a string"') == []


class TestProvenanceDataNote:
    """PR 3 — one measured, class-conditional sentence; silence elsewhere."""

    def _insight_text(self, summary, monkeypatch):
        import components._callbacks_shared as shared
        from components._callbacks_overview import _build_overview_insight

        monkeypatch.setattr(shared, "_read_vintage_summary", lambda region: summary)
        ts = pd.date_range("2026-07-17 00:00", periods=30, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": ts, "demand_mw": [3300.0] * 30})
        return str(_build_overview_insight("LDWP", df, "grid_ops", []))

    def test_broken_class_note(self, monkeypatch):
        text = self._insight_text(
            {"revision_class": "broken", "mean_fresh_revision_pct": 70.3}, monkeypatch
        )
        assert "Data note" in text
        assert "70%" in text
        assert "firm up over the following day" in text

    def test_bulk_class_note(self, monkeypatch):
        text = self._insight_text(
            {"revision_class": "bulk", "mean_fresh_revision_pct": 18.0}, monkeypatch
        )
        assert "next-morning resubmission" in text

    def test_churn_class_note(self, monkeypatch):
        text = self._insight_text(
            {"revision_class": "churn", "mean_fresh_revision_pct": 15.0}, monkeypatch
        )
        assert "as metering completes" in text

    def test_clean_unknown_and_missing_are_silent(self, monkeypatch):
        for summary in (
            {"revision_class": "clean", "mean_fresh_revision_pct": 1.0},
            {"revision_class": "unknown", "mean_fresh_revision_pct": 50.0},
            None,
        ):
            assert "Data note" not in self._insight_text(summary, monkeypatch)
