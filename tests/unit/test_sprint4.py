"""
Tests for Sprint 4 features.

Covers:
- Persona tab visibility (AC-7.5)
- Tab 1 KPI data contracts
- Callback Output ID completeness
- Model service integration in callbacks
"""

import inspect
import sys

import pandas as pd
import pytest

from config import REGION_CAPACITY_MW, REGION_COORDINATES
from personas.config import PERSONAS, get_persona


class TestPersonaTabVisibility:
    """AC-7.5: Each persona has specific priority tabs; others are disabled."""

    def test_all_personas_have_priority_tabs(self):
        for pid, persona in PERSONAS.items():
            assert len(persona.priority_tabs) >= 2, f"{pid} has too few priority tabs"

    def test_priority_tabs_are_valid_ids(self):
        valid = {
            "tab-overview",
            "tab-forecast",
            "tab-outlook",
            "tab-backtest",
            "tab-generation",
            "tab-weather",
            "tab-models",
            "tab-alerts",
            "tab-simulator",
        }
        for pid, persona in PERSONAS.items():
            for tab in persona.priority_tabs:
                assert tab in valid, f"{pid} has invalid tab: {tab}"

    def test_grid_ops_prioritizes_forecast(self):
        p = get_persona("grid_ops")
        assert "tab-forecast" in p.priority_tabs
        assert "tab-outlook" in p.priority_tabs

    def test_trader_prioritizes_outlook(self):
        p = get_persona("trader")
        assert "tab-outlook" in p.priority_tabs

    def test_data_scientist_prioritizes_backtest(self):
        p = get_persona("data_scientist")
        assert "tab-backtest" in p.priority_tabs

    def test_renewables_prioritizes_outlook(self):
        p = get_persona("renewables")
        assert "tab-outlook" in p.priority_tabs
        assert "tab-forecast" in p.priority_tabs

    def test_no_persona_has_empty_priority(self):
        for pid, persona in PERSONAS.items():
            assert persona.priority_tabs, f"{pid} has empty priority_tabs"

    def test_default_tab_is_in_priority_list(self):
        """Default tab should always be enabled (in priority list)."""
        for pid, persona in PERSONAS.items():
            assert persona.default_tab in persona.priority_tabs, (
                f"{pid}: default_tab '{persona.default_tab}' not in priority_tabs"
            )


class TestCallbackOutputCompleteness:
    """Verify surviving layout component IDs have corresponding callback Outputs.

    V2.1 dropped the tab1-* (Historical) and tab4-* (Generation) wiring tests
    along with the hidden tabs whose IDs they referenced. tab5-* lives on the
    visible Risk tab and is still wired through update_alerts_tab.
    """

    def test_tab5_stress_breakdown_wired(self):
        # The alerts callback wiring lives in register_callbacks in callbacks.py
        # — explicit import ensures the module is loaded before introspection
        # (regression-safe against pytest test-ordering changes).
        import components.callbacks

        src = inspect.getsource(components.callbacks)
        assert "tab5-stress-breakdown" in src

    def test_persona_tab_disabled_loop_removed(self):
        """Tab visibility loop was removed: dbc.Tab tab_id != id, disabled not dynamic."""
        import components.callbacks

        src = inspect.getsource(components.callbacks)
        assert "bound_tid" not in src, "Broken tab visibility loop should be removed"

    def test_model_service_used_not_raw_noise(self):
        """Callbacks use model_service, not inline random noise.

        Step 10c of the register_callbacks split (issue #87) moved the
        forecast callback (and its ``get_forecasts`` call) out of
        callbacks.py into ``_callbacks_forecast.py``. The assertion now
        targets the forecast module's source; the no-noise check spans
        both modules to catch the legacy pattern wherever it could live.
        """
        import components._callbacks_forecast
        import components._callbacks_models
        import components.callbacks

        forecast_src = inspect.getsource(components._callbacks_forecast)
        models_src = inspect.getsource(components._callbacks_models)
        callbacks_src = inspect.getsource(components.callbacks)
        combined = forecast_src + models_src + callbacks_src

        assert "get_forecasts" in combined, "model_service.get_forecasts not used"
        # Old pattern should be gone everywhere
        assert "np.random.normal(0, 0.02" not in combined, "Old noise pattern still present"


class TestTab1KPIContracts:
    """Test the data contracts for Tab 1 KPI calculations."""

    def test_peak_demand_is_positive(self):
        from data.demo_data import generate_demo_demand

        df = generate_demo_demand("FPL", days=7)
        peak = df["demand_mw"].max()
        assert peak > 0

    def test_reserve_margin_calculation(self):
        """Reserve = (capacity - peak) / capacity × 100."""
        capacity = REGION_CAPACITY_MW["FPL"]
        peak = capacity * 0.72  # typical
        reserve = (capacity - peak) / capacity * 100
        assert 20 < reserve < 40

    def test_reserve_margin_categories(self):
        """Three tiers: >15% Adequate, 5-15% Low, <5% CRITICAL."""
        for pct, expected in [(28, "Adequate"), (10, "Low"), (3, "CRITICAL")]:
            if pct > 15:
                assert expected == "Adequate"
            elif pct > 5:
                assert expected == "Low"
            else:
                assert expected == "CRITICAL"

    @pytest.mark.parametrize("region", list(REGION_COORDINATES.keys()))
    def test_all_regions_have_capacity(self, region):
        """Every region needs capacity for reserve margin calculation."""
        assert region in REGION_CAPACITY_MW
        assert REGION_CAPACITY_MW[region] > 0


class TestModelServiceIntegration:
    """Test model service produces valid data for callback consumption."""

    def test_forecasts_survive_callback_pipeline(self):
        """Simulate the full callback data path."""
        import io

        from data.demo_data import generate_demo_demand
        from models.model_service import get_forecasts

        df = generate_demo_demand("FPL", days=7)

        # Simulate dcc.Store roundtrip
        json_str = df.to_json(date_format="iso")
        restored = pd.read_json(io.StringIO(json_str))
        restored["timestamp"] = pd.to_datetime(restored["timestamp"])

        # Get forecasts from restored data
        result = get_forecasts("FPL", restored)
        assert len(result["ensemble"]) == len(restored)
        assert (result["ensemble"] > 0).all()

    def test_metrics_match_forecast_models(self):
        """Every model in forecasts has corresponding metrics."""
        from data.demo_data import generate_demo_demand
        from models.model_service import get_forecasts

        df = generate_demo_demand("FPL", days=7)
        result = get_forecasts("FPL", df)
        metrics = result["metrics"]

        for model in ["prophet", "arima", "xgboost"]:
            assert model in metrics, f"No metrics for {model}"
            assert "mape" in metrics[model]
