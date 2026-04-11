"""
Unit tests for data/user_prefs.py (NEXD-9: "Smart Defaults That Learn").

Covers:
- UserPrefs creation, serialization, round-trip
- validate_prefs() sanitization against known valid values
- _validate_filter_value() per-filter option checking
- TRACKED_FILTERS and FILTER_DEFAULTS consistency
"""

from data.user_prefs import (
    FILTER_DEFAULTS,
    TRACKED_FILTERS,
    UserPrefs,
    _validate_filter_value,
    validate_prefs,
)

# ── UserPrefs dataclass ──────────────────────────────────────────────


class TestUserPrefs:
    def test_defaults(self):
        prefs = UserPrefs()
        assert prefs.region == "FPL"
        assert prefs.persona == "grid_ops"
        assert prefs.tab == "tab-overview"
        assert prefs.filters == {}

    def test_to_dict(self):
        prefs = UserPrefs(region="ERCOT", persona="trader")
        d = prefs.to_dict()
        assert d["region"] == "ERCOT"
        assert d["persona"] == "trader"
        assert isinstance(d, dict)

    def test_from_dict_round_trip(self):
        prefs = UserPrefs(
            region="CAISO",
            persona="renewables",
            tab="tab-generation",
            filters={"tab1-timerange": "720"},
        )
        d = prefs.to_dict()
        restored = UserPrefs.from_dict(d)
        assert restored.region == "CAISO"
        assert restored.tab == "tab-generation"
        assert restored.filters == {"tab1-timerange": "720"}

    def test_from_dict_none(self):
        prefs = UserPrefs.from_dict(None)
        assert prefs.region == "FPL"

    def test_from_dict_empty(self):
        prefs = UserPrefs.from_dict({})
        assert prefs.region == "FPL"

    def test_from_dict_ignores_unknown_keys(self):
        prefs = UserPrefs.from_dict({"region": "PJM", "unknown_field": 42})
        assert prefs.region == "PJM"

    def test_from_dict_not_dict(self):
        prefs = UserPrefs.from_dict("invalid")
        assert prefs.region == "FPL"


# ── _validate_filter_value ───────────────────────────────────────────


class TestValidateFilterValue:
    def test_valid_radio_item(self):
        assert _validate_filter_value("tab1-timerange", "168") == "168"

    def test_invalid_radio_item(self):
        assert _validate_filter_value("tab1-timerange", "999") is None

    def test_valid_checklist(self):
        result = _validate_filter_value("tab3-model-selector", ["prophet", "xgboost"])
        assert result == ["prophet", "xgboost"]

    def test_checklist_filters_invalid(self):
        result = _validate_filter_value("tab3-model-selector", ["prophet", "bad_model"])
        assert result == ["prophet"]

    def test_checklist_all_invalid(self):
        result = _validate_filter_value("tab3-model-selector", ["bad1", "bad2"])
        assert result is None

    def test_valid_sim_duration(self):
        assert _validate_filter_value("sim-duration", 48) == 48

    def test_invalid_sim_duration(self):
        assert _validate_filter_value("sim-duration", 99) is None

    def test_sim_duration_string_coerced(self):
        assert _validate_filter_value("sim-duration", "72") == 72

    def test_sim_duration_none(self):
        assert _validate_filter_value("sim-duration", None) is None

    def test_unknown_filter_id(self):
        assert _validate_filter_value("nonexistent-filter", "value") is None

    def test_outlook_model_valid(self):
        assert _validate_filter_value("outlook-model", "ensemble") == "ensemble"

    def test_outlook_model_invalid(self):
        assert _validate_filter_value("outlook-model", "random_forest") is None


# ── validate_prefs ───────────────────────────────────────────────────


class TestValidatePrefs:
    def test_valid_prefs(self):
        data = {
            "region": "ERCOT",
            "persona": "trader",
            "tab": "tab-outlook",
            "filters": {"tab1-timerange": "720", "outlook-horizon": "24"},
        }
        prefs = validate_prefs(data)
        assert prefs.region == "ERCOT"
        assert prefs.persona == "trader"
        assert prefs.tab == "tab-outlook"
        assert prefs.filters["tab1-timerange"] == "720"
        assert prefs.filters["outlook-horizon"] == "24"

    def test_invalid_region_falls_back(self):
        prefs = validate_prefs({"region": "INVALID_BA"})
        assert prefs.region == "FPL"

    def test_invalid_persona_falls_back(self):
        prefs = validate_prefs({"persona": "hacker"})
        assert prefs.persona == "grid_ops"

    def test_invalid_tab_falls_back(self):
        prefs = validate_prefs({"tab": "tab-nonexistent"})
        assert prefs.tab == "tab-overview"

    def test_unknown_filter_stripped(self):
        prefs = validate_prefs({"filters": {"unknown-filter": "value", "tab1-timerange": "24"}})
        assert "unknown-filter" not in prefs.filters
        assert prefs.filters["tab1-timerange"] == "24"

    def test_invalid_filter_value_stripped(self):
        prefs = validate_prefs({"filters": {"tab1-timerange": "9999"}})
        assert "tab1-timerange" not in prefs.filters

    def test_none_input(self):
        prefs = validate_prefs(None)
        assert prefs.region == "FPL"
        assert prefs.filters == {}

    def test_empty_input(self):
        prefs = validate_prefs({})
        assert prefs.region == "FPL"

    def test_preserves_valid_checklist(self):
        prefs = validate_prefs({"filters": {"tab3-model-selector": ["prophet", "xgboost"]}})
        assert prefs.filters["tab3-model-selector"] == ["prophet", "xgboost"]

    def test_sim_duration_preserved(self):
        prefs = validate_prefs({"filters": {"sim-duration": 72}})
        assert prefs.filters["sim-duration"] == 72


# ── TRACKED_FILTERS / FILTER_DEFAULTS consistency ────────────────────


class TestTrackedFiltersConsistency:
    def test_all_tracked_have_defaults(self):
        for fid in TRACKED_FILTERS:
            assert fid in FILTER_DEFAULTS, f"Missing default for tracked filter: {fid}"

    def test_no_extra_defaults(self):
        for fid in FILTER_DEFAULTS:
            assert fid in TRACKED_FILTERS, f"Default for untracked filter: {fid}"

    def test_tracked_filters_not_empty(self):
        assert len(TRACKED_FILTERS) > 0

    def test_defaults_are_valid_values(self):
        """Each default value should pass its own validation."""
        for fid, default in FILTER_DEFAULTS.items():
            result = _validate_filter_value(fid, default)
            assert result is not None, f"Default for {fid} fails validation: {default}"
