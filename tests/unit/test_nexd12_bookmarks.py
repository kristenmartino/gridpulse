"""
Unit tests for NEXD-12: Scenario Bookmarks (full-state URL serialization).

Covers:
- serialize_bookmark_params(): core params, filters, checklist, sliders
- deserialize_bookmark_params(): parsing, validation, backward compat, roundtrip
- SIM_SLIDERS / SIM_SLIDER_DEFAULTS / SIM_SLIDER_RANGES consistency
- SIM_DURATION_DEFAULT
"""

from data.user_prefs import (
    SIM_DURATION_DEFAULT,
    SIM_SLIDER_DEFAULTS,
    SIM_SLIDER_RANGES,
    SIM_SLIDERS,
    deserialize_bookmark_params,
    serialize_bookmark_params,
)

# ── serialize_bookmark_params ───────────────────────────────────────


class TestSerializeBookmarkParams:
    def test_core_params(self):
        result = serialize_bookmark_params("FPL", "trader", "tab-simulator", {})
        assert "region=FPL" in result
        assert "persona=trader" in result
        assert "tab=tab-simulator" in result
        assert result.startswith("?")

    def test_filter_with_f_prefix(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-forecast",
            {"tab1-timerange": "168", "outlook-model": "ensemble"},
        )
        assert "f.tab1-timerange=168" in result
        assert "f.outlook-model=ensemble" in result

    def test_checklist_comma_separated(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-models",
            {"tab3-model-selector": ["prophet", "xgboost"]},
        )
        # urlencode will encode comma as %2C
        assert "f.tab3-model-selector=prophet%2Cxgboost" in result

    def test_sim_sliders_with_s_prefix(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-simulator",
            {},
            {"sim-temp": 5, "sim-wind": 80},
        )
        assert "s.temp=5" in result
        assert "s.wind=80" in result

    def test_none_slider_omitted(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-simulator",
            {},
            {"sim-temp": None},
        )
        assert "s.temp" not in result

    def test_none_filter_omitted(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-simulator",
            {"tab1-timerange": None},
        )
        assert "f.tab1-timerange" not in result

    def test_empty_region_defaults(self):
        result = serialize_bookmark_params("", "trader", "tab-simulator", {})
        assert "region=FPL" in result

    def test_empty_persona_defaults(self):
        result = serialize_bookmark_params("FPL", "", "tab-simulator", {})
        assert "persona=grid_ops" in result

    def test_empty_tab_defaults(self):
        result = serialize_bookmark_params("FPL", "trader", "", {})
        assert "tab=tab-overview" in result

    def test_no_sliders_means_no_s_prefix(self):
        result = serialize_bookmark_params("FPL", "trader", "tab-forecast", {})
        assert "s." not in result

    def test_sim_duration_as_filter(self):
        result = serialize_bookmark_params(
            "FPL",
            "trader",
            "tab-simulator",
            {"sim-duration": 72},
        )
        assert "f.sim-duration=72" in result


# ── deserialize_bookmark_params ─────────────────────────────────────


class TestDeserializeBookmarkParams:
    def test_core_params(self):
        result = deserialize_bookmark_params("?region=ERCOT&persona=trader&tab=tab-models")
        assert result["region"] == "ERCOT"
        assert result["persona"] == "trader"
        assert result["tab"] == "tab-models"

    def test_filter_deserialization(self):
        result = deserialize_bookmark_params("?region=FPL&f.tab1-timerange=168")
        assert result["filters"]["tab1-timerange"] == "168"

    def test_checklist_deserialization(self):
        result = deserialize_bookmark_params("?f.tab3-model-selector=prophet,xgboost")
        assert result["filters"]["tab3-model-selector"] == ["prophet", "xgboost"]

    def test_sim_slider_deserialization(self):
        result = deserialize_bookmark_params("?s.temp=5&s.wind=80")
        assert result["sim_sliders"]["sim-temp"] == 5.0
        assert result["sim_sliders"]["sim-wind"] == 80.0

    def test_invalid_region_omitted(self):
        result = deserialize_bookmark_params("?region=INVALID")
        assert "region" not in result

    def test_invalid_persona_omitted(self):
        result = deserialize_bookmark_params("?persona=hacker")
        assert "persona" not in result

    def test_invalid_tab_omitted(self):
        result = deserialize_bookmark_params("?tab=tab-nonexistent")
        assert "tab" not in result

    def test_invalid_filter_value_omitted(self):
        result = deserialize_bookmark_params("?f.tab1-timerange=9999")
        assert "filters" not in result or "tab1-timerange" not in result.get("filters", {})

    def test_slider_out_of_range_omitted(self):
        result = deserialize_bookmark_params("?s.temp=999")
        assert "sim_sliders" not in result or "sim-temp" not in result.get("sim_sliders", {})

    def test_slider_below_range_omitted(self):
        result = deserialize_bookmark_params("?s.temp=-50")
        assert "sim_sliders" not in result or "sim-temp" not in result.get("sim_sliders", {})

    def test_slider_non_numeric_omitted(self):
        result = deserialize_bookmark_params("?s.temp=abc")
        assert "sim_sliders" not in result

    def test_empty_search(self):
        assert deserialize_bookmark_params("") == {}

    def test_none_like_empty(self):
        assert deserialize_bookmark_params("") == {}

    def test_backward_compatible_old_bookmark(self):
        """Old URLs with only region/persona/tab still work."""
        result = deserialize_bookmark_params("?region=FPL&persona=trader&tab=tab-simulator")
        assert result["region"] == "FPL"
        assert result["persona"] == "trader"
        assert result["tab"] == "tab-simulator"
        assert "filters" not in result
        assert "sim_sliders" not in result

    def test_unknown_params_ignored(self):
        result = deserialize_bookmark_params("?region=FPL&foo=bar&f.unknown=val")
        assert result.get("region") == "FPL"
        assert "filters" not in result or "unknown" not in result.get("filters", {})

    def test_sim_duration_filter_roundtrip(self):
        """sim-duration is a filter (f. prefix), validated as int."""
        result = deserialize_bookmark_params("?f.sim-duration=72")
        assert result["filters"]["sim-duration"] == 72

    def test_invalid_sim_duration_dropped(self):
        result = deserialize_bookmark_params("?f.sim-duration=99")
        assert "filters" not in result or "sim-duration" not in result.get("filters", {})


# ── Roundtrip ───────────────────────────────────────────────────────


class TestBookmarkRoundtrip:
    def test_full_roundtrip(self):
        filters = {
            "tab1-timerange": "720",
            "outlook-model": "ensemble",
            "tab3-model-selector": ["prophet", "arima"],
            "sim-duration": 48,
        }
        sim_sliders = {"sim-temp": 5.0, "sim-wind": 80.0, "sim-solar": 200.0}
        encoded = serialize_bookmark_params(
            "ERCOT", "trader", "tab-simulator", filters, sim_sliders
        )
        decoded = deserialize_bookmark_params(encoded)

        assert decoded["region"] == "ERCOT"
        assert decoded["persona"] == "trader"
        assert decoded["tab"] == "tab-simulator"
        assert decoded["filters"]["tab1-timerange"] == "720"
        assert decoded["filters"]["outlook-model"] == "ensemble"
        assert decoded["filters"]["tab3-model-selector"] == ["prophet", "arima"]
        assert decoded["filters"]["sim-duration"] == 48
        assert decoded["sim_sliders"]["sim-temp"] == 5.0
        assert decoded["sim_sliders"]["sim-wind"] == 80.0
        assert decoded["sim_sliders"]["sim-solar"] == 200.0

    def test_core_only_roundtrip(self):
        encoded = serialize_bookmark_params("FPL", "grid_ops", "tab-overview", {})
        decoded = deserialize_bookmark_params(encoded)
        assert decoded["region"] == "FPL"
        assert decoded["persona"] == "grid_ops"
        assert decoded["tab"] == "tab-overview"
        assert "filters" not in decoded
        assert "sim_sliders" not in decoded


# ── SIM_SLIDERS constants consistency ───────────────────────────────


class TestSimSliderConstants:
    def test_all_sliders_have_defaults(self):
        for sid in SIM_SLIDERS:
            assert sid in SIM_SLIDER_DEFAULTS, f"Missing default for: {sid}"

    def test_all_sliders_have_ranges(self):
        for sid in SIM_SLIDERS:
            assert sid in SIM_SLIDER_RANGES, f"Missing range for: {sid}"

    def test_defaults_within_ranges(self):
        for sid, default in SIM_SLIDER_DEFAULTS.items():
            min_v, max_v = SIM_SLIDER_RANGES[sid]
            assert min_v <= default <= max_v, f"{sid}: {default} not in [{min_v}, {max_v}]"

    def test_no_extra_defaults(self):
        for sid in SIM_SLIDER_DEFAULTS:
            assert sid in SIM_SLIDERS, f"Default for unknown slider: {sid}"

    def test_slider_ranges_match_tab_simulator(self):
        """Ranges must match the _slider() calls in tab_simulator.py."""
        expected = {
            "sim-temp": (-10, 120),
            "sim-wind": (0, 80),
            "sim-cloud": (0, 100),
            "sim-humidity": (0, 100),
            "sim-solar": (0, 1000),
        }
        assert expected == SIM_SLIDER_RANGES

    def test_sim_duration_default_is_valid(self):
        assert SIM_DURATION_DEFAULT in (24, 48, 72, 168)
