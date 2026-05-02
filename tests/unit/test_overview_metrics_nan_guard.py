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
