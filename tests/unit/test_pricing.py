"""Unit tests for models/pricing.py."""

import numpy as np
import pytest

from config import PRICING_BASE_USD_MWH
from models.pricing import (
    capacity_headroom_pct,
    estimate_price_for_region,
    estimate_price_impact,
    grid_stress,
    utilization_pct,
)


class TestEstimatePriceImpact:
    """Merit-order pricing model tests."""

    def test_low_utilization_base_price(self):
        """Below 70% → base price."""
        price = estimate_price_impact(50000, 100000)
        assert price == pytest.approx(PRICING_BASE_USD_MWH)

    def test_moderate_utilization_increases(self):
        """70-90% → price increases."""
        price = estimate_price_impact(80000, 100000)
        assert price > PRICING_BASE_USD_MWH

    def test_high_utilization_exponential(self):
        """90-100% → exponential spike."""
        price = estimate_price_impact(95000, 100000)
        assert price > PRICING_BASE_USD_MWH * 2

    def test_emergency_utilization(self):
        """Above 100% → emergency pricing."""
        price = estimate_price_impact(110000, 100000)
        assert price == pytest.approx(PRICING_BASE_USD_MWH * 20)

    def test_vectorized(self):
        demand = np.array([50000, 80000, 95000, 110000])
        prices = estimate_price_impact(demand, 100000)
        assert len(prices) == 4
        # Prices should be monotonically increasing
        assert all(prices[i] <= prices[i + 1] for i in range(3))

    def test_scalar_returns_float(self):
        price = estimate_price_impact(50000.0, 100000.0)
        assert isinstance(price, float)


class TestEstimatePriceForRegion:
    """Region convenience wrapper."""

    def test_known_region(self):
        price = estimate_price_for_region(50000, "ERCOT")
        assert isinstance(price, float)
        assert price > 0

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            estimate_price_for_region(50000, "INVALID")

    def test_fpl_region(self):
        price = estimate_price_for_region(20000, "FPL")
        assert price == pytest.approx(PRICING_BASE_USD_MWH)


class TestCapacityHeadroom:
    """Capacity-headroom calculation (nameplate-based; not NERC reserve margin)."""

    def test_surplus(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["ERCOT"]
        # Half capacity is unambiguously slack regardless of annual refreshes.
        assert capacity_headroom_pct(cap // 2, cap) > 0

    def test_deficit(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["ERCOT"]
        # Load above nameplate is unambiguously negative headroom.
        assert capacity_headroom_pct(int(cap * 1.1), cap) < 0

    def test_at_capacity(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["ERCOT"]
        assert capacity_headroom_pct(cap, cap) == pytest.approx(0.0)

    def test_utilization_complements_headroom(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["ERCOT"]
        load = int(cap * 0.6)
        assert utilization_pct(load, cap) == pytest.approx(60.0, abs=0.5)
        # Headroom + utilization = 100% of nameplate.
        assert capacity_headroom_pct(load, cap) + utilization_pct(load, cap) == pytest.approx(100.0)


class TestGridStress:
    """#265 — grid stress is demand/capacity supply-tightness, not an alert count.

    Replaces the old ``min(100, n_crit*30 + n_warn*15 + 20)`` heuristic that
    pinned at 100 for nearly every BA.
    """

    def test_utilization_bands(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["PJM"]
        # < 70% → Normal
        assert grid_stress("PJM", int(cap * 0.40)) == (40, "Normal")
        # 70-85% → Elevated
        assert grid_stress("PJM", int(cap * 0.78)) == (78, "Elevated")
        # >= 85% → High
        assert grid_stress("PJM", int(cap * 0.90)) == (90, "High")

    def test_capped_at_100(self):
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["PJM"]
        score, label = grid_stress("PJM", int(cap * 1.3))  # demand above plate
        assert score == 100
        assert label == "High"

    def test_unreliable_capacity_returns_none(self):
        # Peak-derived (SOCO) and import-dominated (SPA) BAs have no reliable
        # measured plate (#254) → no fabricated stress number.
        assert grid_stress("SOCO", 40_000) == (None, "Capacity n/a")
        assert grid_stress("SPA", 1_000) == (None, "Capacity n/a")

    def test_missing_demand_returns_none(self):
        assert grid_stress("PJM", None) == (None, "Unavailable")
        assert grid_stress("PJM", 0) == (None, "Unavailable")

    def test_does_not_depend_on_alerts(self):
        """Sanity: the same demand yields the same stress regardless of any
        alert context — the whole point of #265."""
        from config import REGION_CAPACITY_MW

        cap = REGION_CAPACITY_MW["ERCOT"]
        assert grid_stress("ERCOT", int(cap * 0.5))[0] == 50
