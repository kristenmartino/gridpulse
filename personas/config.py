"""
Persona configuration for role-based dashboard UX.

Per spec §Persona Switcher:
- 4 personas with distinct default tabs, KPI priorities, and alert thresholds
- Welcome cards with role-specific language
- Dynamic KPI highlighting per persona
"""

from dataclasses import dataclass


@dataclass
class Persona:
    """Dashboard persona configuration."""

    id: str
    name: str
    title: str
    avatar: str
    default_tab: str
    priority_tabs: list[str]
    kpi_metrics: list[str]
    alert_threshold: str  # "critical", "warning", "info"
    welcome_title: str
    welcome_message: str
    color: str


PERSONAS: dict[str, Persona] = {
    "grid_ops": Persona(
        id="grid_ops",
        name="Sarah",
        title="Grid Operations Manager",
        avatar="👩‍💼",
        default_tab="tab-forecast",
        priority_tabs=[
            "tab-forecast",
            "tab-outlook",
            "tab-backtest",
            "tab-generation",
            "tab-alerts",
        ],
        kpi_metrics=[
            "peak_demand_mw",
            "reserve_margin_pct",
            "active_alerts_count",
            "ramp_rate_max",
        ],
        alert_threshold="warning",
        welcome_title="Grid Operations Dashboard",
        welcome_message=(
            "Good morning, Sarah. Here's your grid status overview. "
            "Focus areas: demand forecast accuracy, reserve margins, "
            "and any active severe weather alerts that could impact operations."
        ),
        color="#1f77b4",
    ),
    "renewables": Persona(
        id="renewables",
        name="James",
        title="Renewables Portfolio Analyst",
        avatar="🌱",
        default_tab="tab-outlook",
        priority_tabs=[
            "tab-generation",
            "tab-outlook",
            "tab-forecast",
            "tab-backtest",
            "tab-weather",
        ],
        kpi_metrics=[
            "wind_capacity_factor",
            "solar_capacity_factor",
            "renewable_generation_pct",
            "curtailment_mw",
        ],
        alert_threshold="info",
        welcome_title="Renewables Analytics",
        welcome_message=(
            "Welcome, James. Your renewable generation outlook is ready. "
            "Key focus: wind and solar capacity factors, weather-driven "
            "generation variability, and curtailment risk."
        ),
        color="#2ca02c",
    ),
    "trader": Persona(
        id="trader",
        name="Maria",
        title="Energy Trader",
        avatar="📊",
        default_tab="tab-outlook",
        priority_tabs=[
            "tab-outlook",
            "tab-forecast",
            "tab-generation",
            "tab-backtest",
            "tab-simulator",
        ],
        kpi_metrics=[
            "price_estimate_usd_mwh",
            "demand_vs_forecast_pct",
            "volatility_index",
            "price_spike_probability",
        ],
        alert_threshold="warning",
        welcome_title="Trading Intelligence",
        welcome_message=(
            "Maria, your market intelligence is updated. "
            "Focus areas: price impact scenarios, demand/supply imbalances, "
            "and weather-driven volatility signals."
        ),
        color="#ff7f0e",
    ),
    "data_scientist": Persona(
        id="data_scientist",
        name="Dev",
        title="Data Scientist",
        avatar="🔬",
        default_tab="tab-backtest",
        priority_tabs=[
            "tab-backtest",
            "tab-outlook",
            "tab-forecast",
            "tab-generation",
            "tab-models",
            "tab-weather",
        ],
        kpi_metrics=[
            "ensemble_mape_pct",
            "model_drift_score",
            "feature_importance_top3",
            "prediction_interval_coverage",
        ],
        alert_threshold="info",
        welcome_title="Model Performance Lab",
        welcome_message=(
            "Hey Dev. Model diagnostics are ready for review. "
            "Focus areas: ensemble accuracy trends, individual model MAPE, "
            "SHAP feature contributions, and residual analysis."
        ),
        color="#9467bd",
    ),
}


def get_persona(persona_id: str) -> Persona:
    """Get a persona by ID. Raises KeyError if not found."""
    if persona_id not in PERSONAS:
        raise KeyError(f"Unknown persona: '{persona_id}'. Valid: {list(PERSONAS.keys())}")
    return PERSONAS[persona_id]


def list_personas() -> list[dict]:
    """List all personas with summary info for the switcher dropdown."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "title": p.title,
            "avatar": p.avatar,
            "color": p.color,
        }
        for p in PERSONAS.values()
    ]


def get_welcome_card(persona_id: str) -> dict:
    """
    Generate welcome card content for a persona.

    Returns:
        Dict with 'title', 'message', 'avatar', 'default_tab', 'kpis'.
    """
    p = get_persona(persona_id)
    return {
        "title": p.welcome_title,
        "message": p.welcome_message,
        "avatar": p.avatar,
        "default_tab": p.default_tab,
        "kpis": p.kpi_metrics,
        "color": p.color,
    }
