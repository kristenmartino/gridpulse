"""
Accessibility utilities for WCAG 2.1 AA compliance.

Provides:
- Colorblind-safe palette (verified with Sim Daltonism)
- ARIA label generators for charts and interactive elements
- Keyboard navigation helpers
- Screen reader text generators for KPI cards and alerts
"""

# ── Colorblind-Safe Palette ───────────────────────────────────
# Verified distinguishable under protanopia, deuteranopia, tritanopia.
# Based on Wong (2011) "Points of View: Color blindness" Nature Methods.

CB_PALETTE = {
    "blue": "#0072B2",  # Actual demand
    "orange": "#E69F00",  # Prophet
    "green": "#009E73",  # ARIMA / positive
    "vermillion": "#D55E00",  # Ensemble / alert
    "sky_blue": "#56B4E9",  # XGBoost / info
    "yellow": "#F0E442",  # Solar / warning
    "purple": "#CC79A7",  # Nuclear
    "black": "#000000",
}

# Chart line styles paired with colors for double-encoding
# (color + dash pattern = accessible even in grayscale)
LINE_STYLES = {
    "actual": {"color": CB_PALETTE["blue"], "dash": "solid", "width": 2},
    "prophet": {"color": CB_PALETTE["orange"], "dash": "dash", "width": 1.5},
    "arima": {"color": CB_PALETTE["green"], "dash": "dot", "width": 1.5},
    "xgboost": {"color": CB_PALETTE["sky_blue"], "dash": "dashdot", "width": 1.5},
    "ensemble": {"color": CB_PALETTE["vermillion"], "dash": "solid", "width": 3},
    "eia_forecast": {"color": "#7f7f7f", "dash": "dot", "width": 1},
    "temperature": {"color": CB_PALETTE["yellow"], "dash": "solid", "width": 1.5},
}

# Fuel type colors (accessible)
FUEL_COLORS = {
    "nuclear": CB_PALETTE["purple"],
    "coal": "#7f7f7f",
    "gas": CB_PALETTE["orange"],
    "hydro": CB_PALETTE["blue"],
    "wind": CB_PALETTE["green"],
    "solar": CB_PALETTE["yellow"],
    "other": "#b0b0b0",
}

# Severity colors with sufficient contrast on dark backgrounds
SEVERITY_COLORS = {
    "critical": {"bg": "rgba(213, 94, 0, 0.15)", "border": "#D55E00", "text": "#D55E00"},
    "warning": {"bg": "rgba(240, 228, 66, 0.15)", "border": "#F0E442", "text": "#F0E442"},
    "info": {"bg": "rgba(86, 180, 233, 0.15)", "border": "#56B4E9", "text": "#56B4E9"},
}


# ── ARIA Label Generators ─────────────────────────────────────


def chart_aria_label(chart_type: str, title: str, data_summary: str = "") -> str:
    """
    Generate an ARIA label for a Plotly chart.

    Args:
        chart_type: "line chart", "bar chart", "heatmap", etc.
        title: Chart title.
        data_summary: Optional summary of data (e.g., "showing 168 hours of demand data").

    Returns:
        ARIA label string.
    """
    label = f"{chart_type}: {title}"
    if data_summary:
        label += f". {data_summary}"
    return label


def kpi_aria_label(label: str, value: str, delta: str = "") -> str:
    """
    Generate an ARIA label for a KPI card.

    Example: "Peak Demand: 28,450 MW, up 3% versus yesterday"
    """
    text = f"{label}: {value}"
    if delta:
        text += f", {delta}"
    return text


def alert_aria_label(event: str, severity: str, headline: str) -> str:
    """
    Generate an ARIA label for an alert card.

    Example: "Critical alert: Excessive Heat Warning. Heat index up to 115°F."
    """
    return f"{severity.capitalize()} alert: {event}. {headline}"


def slider_aria_label(name: str, value: float, unit: str, min_val: float, max_val: float) -> str:
    """
    Generate an ARIA label for a scenario slider.

    Example: "Temperature slider: 85°F, range -10 to 120°F"
    """
    return f"{name} slider: {value}{unit}, range {min_val} to {max_val}{unit}"


# ── Screen Reader Summary Generators ──────────────────────────


def forecast_summary(
    region: str,
    peak_mw: float,
    peak_time: str,
    mape: float,
    reserve_pct: float,
) -> str:
    """
    Generate a screen-reader-friendly summary of the demand forecast tab.

    Returns:
        Plain text summary suitable for aria-live regions.
    """
    return (
        f"Demand forecast for {region}. "
        f"Today's peak demand is forecast at {peak_mw:,.0f} megawatts at {peak_time}. "
        f"Forecast accuracy over the past 7 days is {mape:.1f}% MAPE. "
        f"Reserve margin is {reserve_pct:.0f}%."
    )


def scenario_summary(
    scenario_name: str,
    demand_delta: float,
    price_impact: float,
    reserve_margin: float,
) -> str:
    """
    Generate a screen-reader-friendly summary of a scenario simulation result.
    """
    direction = "increase" if demand_delta > 0 else "decrease"
    return (
        f"Scenario: {scenario_name}. "
        f"Demand would {direction} by {abs(demand_delta):,.0f} megawatts. "
        f"Estimated price impact: ${price_impact:.0f} per megawatt-hour. "
        f"Reserve margin: {reserve_margin:.0f}%."
    )


# ── Keyboard Navigation Helpers ───────────────────────────────

TAB_KEY_MAP = {
    "tab-forecast": "1",
    "tab-weather": "2",
    "tab-models": "3",
    "tab-generation": "4",
    "tab-alerts": "5",
    "tab-simulator": "6",
}

KEYBOARD_SHORTCUTS = """
Keyboard shortcuts:
  Alt+1 through Alt+6: Switch to tabs 1-6
  Alt+R: Focus region selector
  Alt+P: Focus persona selector
  Escape: Close any open tooltip or dropdown
"""
