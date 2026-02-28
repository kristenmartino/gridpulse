"""
Simplified merit-order pricing model.

Per spec §Pricing Model:
- demand < 70% capacity: base price ($30-50/MWh)
- demand 70-90% capacity: moderate increase (gas peakers)
- demand > 90% capacity: exponential spike (scarcity pricing)
- demand > 100% capacity: emergency pricing ($1000+/MWh)

Accepts scalars or arrays. Returns same type as input.
"""

import numpy as np

from config import (
    PRICING_BASE_USD_MWH,
    PRICING_TIER_MODERATE,
    PRICING_TIER_HIGH,
    PRICING_TIER_EMERGENCY,
    PRICING_EMERGENCY_MULTIPLIER,
    REGION_CAPACITY_MW,
)


def estimate_price_impact(
    demand_forecast: float | np.ndarray,
    generation_capacity: float | np.ndarray,
    base_price: float = PRICING_BASE_USD_MWH,
) -> float | np.ndarray:
    """
    Estimate electricity price based on demand/capacity utilization.

    Args:
        demand_forecast: Forecasted demand in MW (scalar or array).
        generation_capacity: Total generation capacity in MW.
        base_price: Base price in $/MWh (default from config).

    Returns:
        Estimated price in $/MWh (same shape as input).
    """
    demand_forecast = np.asarray(demand_forecast, dtype=float)
    generation_capacity = np.asarray(generation_capacity, dtype=float)
    scalar_input = demand_forecast.ndim == 0

    utilization = demand_forecast / generation_capacity

    price = np.where(
        utilization < PRICING_TIER_MODERATE,
        base_price,
        np.where(
            utilization < PRICING_TIER_HIGH,
            base_price * (1 + 2 * (utilization - PRICING_TIER_MODERATE)),
            np.where(
                utilization < PRICING_TIER_EMERGENCY,
                base_price * np.exp(15 * (utilization - PRICING_TIER_HIGH)),
                base_price * PRICING_EMERGENCY_MULTIPLIER,
            ),
        ),
    )

    return float(price) if scalar_input else price


def estimate_price_for_region(
    demand_forecast: float | np.ndarray,
    region: str,
    base_price: float = PRICING_BASE_USD_MWH,
) -> float | np.ndarray:
    """
    Convenience wrapper: estimate price using the region's capacity from config.

    Args:
        demand_forecast: Forecasted demand in MW.
        region: Balancing authority code (e.g., "ERCOT").
        base_price: Base price in $/MWh.

    Returns:
        Estimated price in $/MWh.
    """
    if region not in REGION_CAPACITY_MW:
        raise ValueError(f"Unknown region: {region}")

    capacity = REGION_CAPACITY_MW[region]
    return estimate_price_impact(demand_forecast, capacity, base_price)


def compute_reserve_margin(
    demand_forecast: float | np.ndarray,
    region: str,
) -> float | np.ndarray:
    """
    Compute reserve margin: (capacity - demand) / capacity × 100.

    Positive = surplus capacity. Negative = demand exceeds capacity (emergency).

    Returns:
        Reserve margin as percentage.
    """
    if region not in REGION_CAPACITY_MW:
        raise ValueError(f"Unknown region: {region}")

    capacity = REGION_CAPACITY_MW[region]
    demand = np.asarray(demand_forecast, dtype=float)
    margin = (capacity - demand) / capacity * 100
    return float(margin) if demand.ndim == 0 else margin
