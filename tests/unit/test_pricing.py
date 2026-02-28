"""Unit tests for models/pricing.py."""

import numpy as np
import pytest

from models.pricing import (
    estimate_price_impact,
    estimate_price_for_region,
    compute_reserve_margin,
)
from config import PRICING_BASE_USD_MWH


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


class TestReserveMargin:
    """Reserve margin calculation."""

    def test_surplus(self):
        margin = compute_reserve_margin(80000, "ERCOT")
        assert margin > 0

    def test_deficit(self):
        margin = compute_reserve_margin(150000, "ERCOT")
        assert margin < 0

    def test_at_capacity(self):
        from config import REGION_CAPACITY_MW
        capacity = REGION_CAPACITY_MW["ERCOT"]
        margin = compute_reserve_margin(capacity, "ERCOT")
        assert margin == pytest.approx(0.0)
