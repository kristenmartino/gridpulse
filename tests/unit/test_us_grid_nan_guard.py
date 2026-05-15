"""Regression tests for the US Grid tab's NaN/zero-division guards.

User-reported bug 2026-05-02: Colorado's region card and the Total
Demand metric on the US Grid tab both rendered "nan" because PSCO's
EIA-930 demand series ends with NaN whenever the latest hour hasn't
been published yet. The fix adds a shared `_latest_real_demand` helper
plus an `_is_real_positive` strict guard, applied at every site that
sums or divides by `current_mw`.

Coverage:
- `_latest_real_demand` walks backward, skipping NaN/zero/inf, with
  optional offset for the trend-baseline use case.
- `_is_real_positive` rejects None / NaN / inf / negative / string.
- `_collect_us_grid_region_data` returns `current_mw=None` when the
  demand tail is NaN — the placeholder path renders, not "nan".
- `_build_us_grid_metrics_items` sums across regions cleanly even if
  one region's `current_mw` slips through as NaN (defensive filter).
- `_build_us_grid_region_card` renders the placeholder for NaN/zero
  `current_mw` rather than formatting it as "nan GW".
- `_build_us_grid_title` reports the right reporting count when one
  region has a NaN tail.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestLatestRealDemand:
    def test_clean_list_returns_last_value(self):
        from components.callbacks import _latest_real_demand

        assert _latest_real_demand([100.0, 200.0, 300.0]) == 300.0

    def test_trailing_nan_skipped(self):
        from components.callbacks import _latest_real_demand

        assert _latest_real_demand([100.0, 200.0, float("nan")]) == 200.0

    def test_multiple_trailing_nans_skipped(self):
        from components.callbacks import _latest_real_demand

        nan = float("nan")
        assert _latest_real_demand([300.0, nan, nan, nan]) == 300.0

    def test_trailing_zero_skipped(self):
        from components.callbacks import _latest_real_demand

        # Zero is EIA's *other* missing-observation marker.
        assert _latest_real_demand([5000.0, 0.0]) == 5000.0

    def test_negative_skipped(self):
        from components.callbacks import _latest_real_demand

        # Defensive — interchange handles negatives elsewhere, but this
        # helper is for demand which is always positive.
        assert _latest_real_demand([5000.0, -100.0]) == 5000.0

    def test_inf_skipped(self):
        from components.callbacks import _latest_real_demand

        assert _latest_real_demand([5000.0, float("inf")]) == 5000.0

    def test_all_nan_returns_none(self):
        from components.callbacks import _latest_real_demand

        nan = float("nan")
        assert _latest_real_demand([nan, nan, nan]) is None

    def test_empty_returns_none(self):
        from components.callbacks import _latest_real_demand

        assert _latest_real_demand([]) is None

    def test_none_input_returns_none(self):
        from components.callbacks import _latest_real_demand

        assert _latest_real_demand(None) is None

    def test_offset_finds_previous_real_value(self):
        from components.callbacks import _latest_real_demand

        # offset=1 → second-most-recent real value
        assert _latest_real_demand([100.0, 200.0, 300.0], offset=1) == 200.0

    def test_offset_skips_nan_at_target_position(self):
        """A NaN spike at the offset position must NOT silently shift the
        baseline — the helper keeps walking backward to find a real value."""
        from components.callbacks import _latest_real_demand

        nan = float("nan")
        # offset=1 would land on the NaN; helper continues to 100.0
        assert _latest_real_demand([100.0, nan, 300.0], offset=1) == 100.0

    def test_pandas_series_works(self):
        from components.callbacks import _latest_real_demand

        s = pd.Series([100.0, 200.0, float("nan")])
        assert _latest_real_demand(s) == 200.0

    def test_numpy_array_works(self):
        from components.callbacks import _latest_real_demand

        arr = np.array([100.0, 200.0, np.nan])
        assert _latest_real_demand(arr) == 200.0


class TestIsRealPositive:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.0, True),
            (1, True),
            (1000.5, True),
            (None, False),
            (0, False),
            (0.0, False),
            (-1.0, False),
            (float("nan"), False),
            (float("inf"), False),
            (-float("inf"), False),
            ("100", False),  # strings rejected — never silently coerce
            ("abc", False),
        ],
    )
    def test_classification(self, value, expected):
        from components.callbacks import _is_real_positive

        assert _is_real_positive(value) is expected


class TestCollectUsGridRegionData:
    """The data-collector layer is where the PSCO bug originated. All
    downstream helpers assume `current_mw` is either `None` or a real
    positive number."""

    def _patch_redis(self, monkeypatch, redis_state):
        """Helper: patch redis_get to return matching keys from the dict."""

        def _get(key):
            return redis_state.get(key)

        monkeypatch.setattr("components._callbacks_us_grid.redis_get", _get)

    def test_clean_demand_tail(self, monkeypatch):
        from components.callbacks import _collect_us_grid_region_data

        self._patch_redis(
            monkeypatch,
            {"wattcast:actuals:PJM": {"demand_mw": [70000.0, 71000.0, 72000.0]}},
        )
        # Limit to one region for clarity
        with patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "Mid-Atlantic (PJM)"}):
            data = _collect_us_grid_region_data()
        assert data["PJM"]["current_mw"] == 72000.0
        assert data["PJM"]["prev_mw"] == 71000.0

    def test_nan_tail_falls_back_to_previous_real(self, monkeypatch):
        """The PSCO scenario: latest hour is NaN → current_mw is the most
        recent real reading, NOT NaN."""
        from components.callbacks import _collect_us_grid_region_data

        self._patch_redis(
            monkeypatch,
            {"wattcast:actuals:PSCO": {"demand_mw": [9000.0, 9100.0, 9200.0, float("nan")]}},
        )
        with patch("components._callbacks_us_grid.REGION_NAMES", {"PSCO": "Colorado (Xcel)"}):
            data = _collect_us_grid_region_data()
        assert data["PSCO"]["current_mw"] == 9200.0
        assert data["PSCO"]["prev_mw"] == 9100.0
        # Sparkline keeps raw values — the chart-side `where(>0)` masks
        # them. The headline number paths read current_mw / prev_mw.
        assert data["PSCO"]["today_mw"][-1] != data["PSCO"]["today_mw"][-1]  # NaN check

    def test_all_nan_returns_none_current_mw(self, monkeypatch):
        from components.callbacks import _collect_us_grid_region_data

        nan = float("nan")
        self._patch_redis(
            monkeypatch,
            {"wattcast:actuals:HECO": {"demand_mw": [nan, nan, nan, nan]}},
        )
        with patch("components._callbacks_us_grid.REGION_NAMES", {"HECO": "Hawaii (HECO)"}):
            data = _collect_us_grid_region_data()
        assert data["HECO"]["current_mw"] is None
        assert data["HECO"]["prev_mw"] is None


class TestUsGridMetricsBarNaNGuard:
    def test_total_demand_excludes_nan_region(self):
        """Even if one region slips through with NaN current_mw — for
        instance from a stale test fixture — the Total Demand metric
        must show the sum across REAL regions only, not 'nan GW'."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
            "PSCO": {"current_mw": float("nan"), "today_mw": [9000.0] * 24},
            "ERCOT": {"current_mw": 50000.0, "today_mw": [50000.0] * 24},
        }
        items = _build_us_grid_metrics_items(region_data)
        total = next(i for i in items if i["label"] == "Total Demand")
        assert total["value"] == "120.0"  # 70 + 50 GW, PSCO NaN excluded
        assert "nan" not in total["value"].lower()

    def test_total_demand_excludes_none_current_mw(self):
        """`_collect_us_grid_region_data` returns None for cold/NaN regions —
        that's the canonical input shape after the source fix."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "PJM": {"current_mw": 70000.0, "today_mw": [70000.0] * 24},
            "PSCO": {"current_mw": None, "today_mw": []},
        }
        items = _build_us_grid_metrics_items(region_data)
        total = next(i for i in items if i["label"] == "Total Demand")
        assert total["value"] == "70.0"

    def test_all_regions_nan_renders_placeholder(self):
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "PJM": {"current_mw": float("nan")},
            "PSCO": {"current_mw": float("nan")},
        }
        items = _build_us_grid_metrics_items(region_data)
        assert all(i["value"] == "—" for i in items)

    def test_peak_24h_excludes_nan_in_today_window(self):
        """A NaN spike inside today_mw[-24:] must not poison the peak metric."""
        from components.callbacks import _build_us_grid_metrics_items

        region_data = {
            "PJM": {
                "current_mw": 70000.0,
                "today_mw": [70000.0, float("nan"), 75000.0],
            },
        }
        items = _build_us_grid_metrics_items(region_data)
        peak = next(i for i in items if i["label"] == "National Peak (24h)")
        # Peak is 75 GW from PJM, NaN ignored
        assert peak["value"] == "75.0"


class TestUsGridRegionCardNaNGuard:
    def test_nan_current_renders_placeholder_card(self):
        """Even if a NaN slips past the source fix, the card renders as
        the empty/warming placeholder rather than 'nan GW'."""
        from components.callbacks import _build_us_grid_region_card

        # Card path mocks REGION_NAMES indirectly via REGION_NAMES.get
        with patch("components._callbacks_us_grid.REGION_NAMES", {"PSCO": "Colorado (Xcel)"}):
            card = _build_us_grid_region_card("PSCO", {"current_mw": float("nan")})
        # Walk the card looking for "nan" in any string content

        def _all_text(component):
            out = []
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, str):
                    out.append(children)
                elif isinstance(children, (list, tuple)):
                    for c in children:
                        out.extend(_all_text(c))
                elif children is not None:
                    out.extend(_all_text(children))
            return out

        rendered_text = " ".join(_all_text(card)).lower()
        assert "nan" not in rendered_text
        assert "—" in rendered_text  # placeholder em-dash

    def test_none_current_renders_placeholder(self):
        from components.callbacks import _build_us_grid_region_card

        with patch("components._callbacks_us_grid.REGION_NAMES", {"PSCO": "Colorado (Xcel)"}):
            card = _build_us_grid_region_card("PSCO", {"current_mw": None})
        # Should pick the empty-card class, not the populated one
        assert "gp-region-card--empty" in (card.className or "")


class TestUsGridTitleNaNGuard:
    def test_title_excludes_nan_from_total(self):
        """Title bar's 'X.X GW total demand' must not say 'nan GW'."""
        from components.callbacks import _build_us_grid_title

        with patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM", "PSCO": "PSCO"}):
            title = _build_us_grid_title(
                {
                    "PJM": {"current_mw": 70000.0},
                    "PSCO": {"current_mw": float("nan")},
                }
            )

        # The title is a Div; walk it for any literal "nan"

        def _all_text(component):
            out = []
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, str):
                    out.append(children)
                elif isinstance(children, (list, tuple)):
                    for c in children:
                        out.extend(_all_text(c))
                elif children is not None:
                    out.extend(_all_text(children))
            return out

        rendered = " ".join(_all_text(title)).lower()
        assert "nan" not in rendered
        # Should report 1 of 2 reporting (PJM only)
        assert "1 of 2" in rendered

    def test_all_nan_renders_warming(self):
        from components.callbacks import _build_us_grid_title

        with patch("components._callbacks_us_grid.REGION_NAMES", {"PJM": "PJM", "PSCO": "PSCO"}):
            title = _build_us_grid_title(
                {
                    "PJM": {"current_mw": float("nan")},
                    "PSCO": {"current_mw": None},
                }
            )

        def _all_text(component):
            out = []
            if hasattr(component, "children"):
                children = component.children
                if isinstance(children, str):
                    out.append(children)
                elif isinstance(children, (list, tuple)):
                    for c in children:
                        out.extend(_all_text(c))
                elif children is not None:
                    out.extend(_all_text(children))
            return out

        rendered = " ".join(_all_text(title)).lower()
        assert "warming up" in rendered
        assert "nan" not in rendered
