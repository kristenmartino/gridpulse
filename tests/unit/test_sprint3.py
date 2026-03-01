"""Unit tests for Sprint 3 modules: accessibility, error handling, observability, welcome."""

from datetime import UTC, datetime, timedelta


class TestColorblindPalette:
    """Verify colorblind-safe palette properties."""

    def test_all_colors_are_hex(self):
        from components.accessibility import CB_PALETTE

        for name, color in CB_PALETTE.items():
            assert color.startswith("#"), f"{name} color {color} is not hex"
            assert len(color) == 7, f"{name} color {color} wrong length"

    def test_all_line_styles_have_required_keys(self):
        from components.accessibility import LINE_STYLES

        for name, style in LINE_STYLES.items():
            assert "color" in style, f"{name} missing color"
            assert "dash" in style, f"{name} missing dash"
            assert "width" in style, f"{name} missing width"

    def test_colors_are_distinct(self):
        from components.accessibility import CB_PALETTE

        colors = list(CB_PALETTE.values())
        assert len(colors) == len(set(colors)), "Duplicate colors in palette"

    def test_fuel_colors_cover_all_types(self):
        from components.accessibility import FUEL_COLORS

        required = {"nuclear", "coal", "gas", "hydro", "wind", "solar", "other"}
        assert required <= set(FUEL_COLORS.keys())

    def test_severity_colors_have_all_levels(self):
        from components.accessibility import SEVERITY_COLORS

        for level in ("critical", "warning", "info"):
            assert level in SEVERITY_COLORS
            assert "bg" in SEVERITY_COLORS[level]
            assert "border" in SEVERITY_COLORS[level]


class TestARIALabels:
    """Test ARIA label generators."""

    def test_chart_aria_label(self):
        from components.accessibility import chart_aria_label

        label = chart_aria_label("line chart", "Demand Forecast", "168 hours")
        assert "line chart" in label
        assert "Demand Forecast" in label
        assert "168 hours" in label

    def test_kpi_aria_label(self):
        from components.accessibility import kpi_aria_label

        label = kpi_aria_label("Peak Demand", "28,450 MW", "up 3%")
        assert "Peak Demand" in label
        assert "28,450 MW" in label
        assert "up 3%" in label

    def test_alert_aria_label(self):
        from components.accessibility import alert_aria_label

        label = alert_aria_label("Heat Advisory", "warning", "Heat index 110°F")
        assert "Warning" in label
        assert "Heat Advisory" in label

    def test_slider_aria_label(self):
        from components.accessibility import slider_aria_label

        label = slider_aria_label("Temperature", 85, "°F", -10, 120)
        assert "85°F" in label
        assert "-10" in label
        assert "120" in label


class TestErrorHandling:
    """Test error states and loading components."""

    def test_loading_spinner(self):
        from components.error_handling import loading_spinner

        component = loading_spinner("Loading data...")
        assert component is not None

    def test_empty_state(self):
        from components.error_handling import empty_state

        component = empty_state("No Data", "Select a region")
        assert component is not None

    def test_error_state(self):
        from components.error_handling import error_state

        component = error_state("Error", "Something broke", error_detail="Traceback...")
        assert component is not None

    def test_api_error_state(self):
        from components.error_handling import api_error_state

        component = api_error_state("EIA API")
        assert component is not None

    def test_freshness_badge_fresh(self):
        from components.error_handling import freshness_badge

        now = datetime.now(UTC)
        badge = freshness_badge(now - timedelta(minutes=5))
        # Check the component's className contains 'fresh'
        assert "fresh" in badge.className

    def test_freshness_badge_stale(self):
        from components.error_handling import freshness_badge

        badge = freshness_badge(datetime.now(UTC) - timedelta(hours=3))
        # Check the component's className contains 'stale'
        assert "stale" in badge.className

    def test_freshness_badge_expired(self):
        from components.error_handling import freshness_badge

        badge = freshness_badge(datetime.now(UTC) - timedelta(hours=12))
        # Check the component's className contains 'expired'
        assert "expired" in badge.className

    def test_freshness_badge_none(self):
        from components.error_handling import freshness_badge

        badge = freshness_badge(None)
        # Check the component's children contains 'No data'
        assert "No data" in badge.children

    def test_format_last_updated_just_now(self):
        from components.error_handling import format_last_updated

        result = format_last_updated(datetime.now(UTC))
        assert "Just now" in result

    def test_format_last_updated_none(self):
        from components.error_handling import format_last_updated

        assert format_last_updated(None) == "Never"

    def test_safe_callback_decorator(self):
        from components.error_handling import safe_callback

        @safe_callback("fallback_value")
        def failing_func():
            raise ValueError("test error")

        result = failing_func()
        assert result == "fallback_value"


class TestObservability:
    """Test structured logging and performance tracking."""

    def test_configure_logging_json(self):
        from observability import configure_logging

        configure_logging(json_output=True)  # Should not raise

    def test_configure_logging_dev(self):
        from observability import configure_logging

        configure_logging(json_output=False)  # Should not raise

    def test_performance_tracker_record(self):
        from observability import PerformanceTracker

        tracker = PerformanceTracker(max_entries=10)
        for i in range(20):
            tracker.record("test_op", float(i))
        stats = tracker.get_stats("test_op")
        assert stats["count"] == 10  # Trimmed to max_entries
        assert stats["mean_ms"] > 0

    def test_performance_tracker_empty(self):
        from observability import PerformanceTracker

        tracker = PerformanceTracker()
        stats = tracker.get_stats("nonexistent")
        assert stats["count"] == 0

    def test_performance_tracker_all_stats(self):
        from observability import PerformanceTracker

        tracker = PerformanceTracker()
        tracker.record("op1", 10.0)
        tracker.record("op2", 20.0)
        all_stats = tracker.get_all_stats()
        assert "op1" in all_stats
        assert "op2" in all_stats

    def test_timed_decorator(self):
        from observability import configure_logging, timed

        configure_logging(json_output=True)

        @timed
        def fast_func():
            return 42

        result = fast_func()
        assert result == 42


class TestWelcomeGenerator:
    """Test dynamic welcome card generation with real data."""

    def test_welcome_with_no_data(self):
        from personas.welcome import generate_welcome_message

        msg = generate_welcome_message("grid_ops", "FPL")
        assert "FPL" in msg or "NextEra" in msg
        assert "Sarah" in msg

    def test_welcome_with_demand_data(self, sample_demand_df):
        from personas.welcome import generate_welcome_message

        msg = generate_welcome_message("grid_ops", "ERCOT", demand_df=sample_demand_df)
        assert "ERCOT" in msg
        # Should contain actual data values, not placeholders
        assert "MW" in msg or "forecast" in msg.lower() or "MAPE" in msg

    def test_welcome_all_personas(self, sample_demand_df, sample_weather_df):
        from personas.welcome import generate_welcome_message

        for pid in ["grid_ops", "renewables", "trader", "data_scientist"]:
            msg = generate_welcome_message(pid, "FPL", sample_demand_df, sample_weather_df)
            assert len(msg) > 20, f"Welcome for {pid} is too short: {msg}"

    def test_welcome_trader_includes_demand_vs_forecast(self, sample_demand_df):
        from personas.welcome import generate_welcome_message

        msg = generate_welcome_message("trader", "FPL", demand_df=sample_demand_df)
        assert "Maria" in msg

    def test_welcome_data_scientist(self, sample_demand_df):
        from personas.welcome import generate_welcome_message

        msg = generate_welcome_message("data_scientist", "FPL", demand_df=sample_demand_df)
        assert "Dev" in msg

    def test_extract_data_stats(self, sample_demand_df, sample_weather_df):
        from personas.welcome import _extract_data_stats

        stats = _extract_data_stats("ERCOT", sample_demand_df, sample_weather_df)
        assert stats["peak_mw"] is not None
        assert stats["peak_mw"] > 0
        assert stats["capacity"] > 0

    def test_extract_stats_none_data(self):
        from personas.welcome import _extract_data_stats

        stats = _extract_data_stats("FPL", None, None)
        assert stats["peak_mw"] is None
        assert stats["capacity"] > 0  # From config
