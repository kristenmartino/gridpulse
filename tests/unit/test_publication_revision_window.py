"""Shared-target elapsed windows and actionable revision summary semantics."""

from copy import deepcopy

import pandas as pd
import pytest

from components.weather_explorer_data import compare_issues, comparison_window, revision_summary
from data.user_prefs import (
    FILTER_DEFAULTS,
    _validate_filter_value,
    deserialize_bookmark_params,
    serialize_bookmark_params,
    validate_prefs,
)


def issue_pair():
    """A repeating-day baseline pair with a changed 03:00 clock-hour."""
    start = pd.Timestamp("2026-10-27T03:00Z")
    earlier = {
        "publication_id": "earlier",
        "loaded": True,
        "series_source": "baseline",
        "series_label": "Seasonal baseline",
        "timing": {"generated_at": "2026-10-27T03:02Z"},
        "forecast": [
            {"timestamp": t.isoformat(), "demand_mw": 19882 if t.hour == 20 else 15146}
            for t in pd.date_range(start, periods=720, freq="h")
        ],
    }
    later = deepcopy(earlier)
    later.update(publication_id="later", timing={"generated_at": "2026-10-27T04:02Z"})
    later["forecast"] = [
        {
            "timestamp": t.isoformat(),
            "demand_mw": 19882 if t.hour == 20 else 16149 if t.hour == 3 else 15146,
        }
        for t in pd.date_range(start + pd.Timedelta(hours=1), periods=720, freq="h")
    ]
    return earlier, later


@pytest.mark.parametrize("hours,count,changed", [("24", 24, 1), ("168", 168, 7), ("720", 718, 29)])
def test_repeating_baseline_windows(hours, count, changed):
    a, b = issue_pair()
    selected, duration = comparison_window(compare_issues(a, b), hours)
    stats = revision_summary(selected)
    assert len(selected) == count and stats["changed"] == changed
    assert duration == int(hours)
    assert selected[0]["timestamp"] == "2026-10-27T05:00:00+00:00"
    assert stats["peak_change_mw"] == stats["peak_shift_hours"] == 0
    assert stats["earlier_peak_mw"] == stats["later_peak_mw"] == 19882
    assert stats["earlier_peak_target"] == "2026-10-27T20:00:00+00:00"
    assert stats["largest"]["change_mw"] == 1003
    assert stats["largest"]["timestamp"] == "2026-10-28T03:00:00+00:00"
    assert (
        selected[-1]["timestamp"]
        == (pd.Timestamp(selected[0]["timestamp"]) + pd.Timedelta(hours=count - 1)).isoformat()
    )


def test_window_uses_clock_bounds_across_gaps_and_exclusive_end():
    a, b = issue_pair()
    rows = compare_issues(a, b)
    sparse = [rows[0], rows[2], rows[23], rows[24], rows[48]]
    selected, _ = comparison_window(list(reversed(sparse)))
    assert selected == [rows[0], rows[2], rows[23]]
    assert comparison_window(rows[:3], "720")[0] == rows[:3]


def test_distinct_peak_amount_timing_negative_revision_and_ties():
    stamps = pd.date_range("2026-10-27T05:00Z", periods=4, freq="h")
    rows = [
        {"timestamp": t.isoformat(), "earlier_mw": a, "later_mw": b, "change_mw": b - a}
        for t, a, b in zip(stamps, [10, 30, 30, 10], [35, 35, 10, 10], strict=True)
    ]
    stats = revision_summary(list(reversed(rows)))
    assert stats["peak_change_mw"] == 5
    assert stats["peak_shift_hours"] == -1
    assert stats["earlier_peak_target"] == rows[1]["timestamp"]
    assert stats["later_peak_target"] == rows[0]["timestamp"]
    assert stats["earlier_peak_ties"] == stats["later_peak_ties"] == 2
    assert stats["largest"]["change_mw"] == 25
    assert stats["changed"] == 3
    rows[0].update(later_mw=-10, change_mw=-20)
    stats = revision_summary(rows)
    assert stats["largest"]["timestamp"] == rows[0]["timestamp"]
    assert stats["largest"]["change_mw"] == -20 and stats["largest_ties"] == 2


def test_empty_unchanged_and_small_nonzero_changes():
    assert comparison_window([]) == ([], 24)
    assert revision_summary([]) is None
    a, b = issue_pair()
    b["forecast"] = deepcopy(a["forecast"])
    rows, _ = comparison_window(compare_issues(a, b))
    assert revision_summary(rows)["changed"] == 0
    rows[0]["change_mw"] = 0.001
    rows[0]["later_mw"] += 0.001
    assert revision_summary(rows)["changed"] == 1
    assert revision_summary(rows)["largest"]["change_mw"] == 0.001


@pytest.mark.parametrize("bad", [None, "garbage", "0", "25", 168, [], {}])
def test_malformed_duration_falls_back_to_24(bad):
    a, b = issue_pair()
    selected, duration = comparison_window(compare_issues(a, b), bad)
    assert len(selected) == duration == 24


@pytest.mark.parametrize("hours", ["24", "168", "720"])
def test_window_preference_and_view_link_roundtrip(hours):
    filters = {
        "published-review-window": hours,
        "outlook-horizon": "168",
        "published-review-mode": "runs",
    }
    assert validate_prefs({"filters": filters}).filters == filters
    link = serialize_bookmark_params("FPL", "grid_ops", "tab-outlook", filters)
    assert deserialize_bookmark_params(link)["filters"] == filters
    assert FILTER_DEFAULTS["published-review-window"] == "24"
    assert _validate_filter_value("published-review-window", "25") is None
    assert deserialize_bookmark_params("?f.published-review-window=25") == {}
