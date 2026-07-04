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
    PRICING_EMERGENCY_MULTIPLIER,
    PRICING_TIER_EMERGENCY,
    PRICING_TIER_HIGH,
    PRICING_TIER_MODERATE,
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


def capacity_headroom_pct(
    load_mw: float | np.ndarray,
    capacity_mw: float,
) -> float | np.ndarray:
    """Capacity headroom: ``(capacity - load) / capacity × 100``.

    The share of *nameplate* capacity not currently serving ``load``. Positive =
    slack; negative = load exceeds nameplate.

    This is deliberately **not** called "reserve margin". The NERC planning
    reserve margin is ``(accredited_capacity - peak) / peak``, and
    ``REGION_CAPACITY_MW`` is EIA-860M *nameplate*, which overstates firm
    capacity for intermittent resources (PRM on nameplate reads ~60-70%, vs a
    real ~15-25%). Surface this as "capacity headroom", never "reserve margin".
    A true reserve margin needs the accredited-capacity (ELCC) model in #243.

    Args:
        load_mw: Current or peak load, MW (scalar or array).
        capacity_mw: Nameplate capacity, MW.

    Returns:
        Headroom as a percentage — ``float`` for scalar ``load_mw``, else array.
    """
    load = np.asarray(load_mw, dtype=float)
    headroom = (capacity_mw - load) / capacity_mw * 100
    return float(headroom) if load.ndim == 0 else headroom


def utilization_pct(
    load_mw: float | np.ndarray,
    capacity_mw: float,
) -> float | np.ndarray:
    """Utilization: ``load / capacity × 100`` — share of *nameplate* capacity in use.

    Complement of :func:`capacity_headroom_pct`. Nameplate-based; see #243 for
    the accredited-capacity work a true reserve/adequacy metric would need.
    """
    load = np.asarray(load_mw, dtype=float)
    util = load / capacity_mw * 100
    return float(util) if load.ndim == 0 else util
