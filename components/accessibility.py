"""
Accessibility utilities for WCAG 2.1 AA compliance.

Provides:
- ARIA label generators for charts and interactive elements
- Keyboard navigation helpers
- Screen reader text generators for KPI cards and alerts
- The color+dash double-encoding for model-identity series

Color VALUES live in ``components.tokens`` (the single source of truth); this
module owns how they are *applied* to accessible chart encodings.
"""

from components.tokens import ACCENT, NEUTRAL_SERIES

# Explicit re-exports (`X as X`) — callers still import the accessible palettes
# from this module, but the VALUES live in components.tokens. The redundant-
# alias form is the convention that marks a re-export, and it stops `ruff
# --fix` from deleting these as "unused imports" (it already did once, which
# broke every `from components.accessibility import FUEL_COLORS` callsite).
from components.tokens import CB_PALETTE as CB_PALETTE
from components.tokens import FUEL_COLORS as FUEL_COLORS

# ── Model-identity double-encoding ────────────────────────────
#
# Color + dash pattern, so model identity survives grayscale and all three
# CVD types (WCAG 1.4.1). Colors come from the Wong palette; the dash is the
# independent second channel. Both are load-bearing — do not drop either.
#
# "actual" is the demand series and carries the brand ACCENT rather than a
# Wong slot: demand is the product's subject, not one model among peers. It is
# verified to clear every series it actually shares a figure with (Wong's
# sky_blue is only ever drawn in single-trace residual figures, so the two
# never meet). ``scripts/verify_palette.py`` re-proves this; if you add a
# figure that draws "actual" alongside "xgboost", that check will fail —
# which is the point.
LINE_STYLES = {
    "actual": {"color": ACCENT, "dash": "solid", "width": 2},
    "prophet": {"color": CB_PALETTE["orange"], "dash": "dash", "width": 1.5},
    "arima": {"color": CB_PALETTE["green"], "dash": "dot", "width": 1.5},
    "xgboost": {"color": CB_PALETTE["sky_blue"], "dash": "dashdot", "width": 1.5},
    "ensemble": {"color": CB_PALETTE["vermillion"], "dash": "solid", "width": 3},
    "eia_forecast": {"color": NEUTRAL_SERIES, "dash": "dot", "width": 1},
    "temperature": {"color": CB_PALETTE["yellow"], "dash": "solid", "width": 1.5},
}

# CB_PALETTE / FUEL_COLORS are imported above rather than defined here, and are
# re-exported so callers can keep importing the accessible palettes from the
# accessibility module. Their values live in components.tokens.


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
    peak_mw: float | None = None,
    peak_time: str | None = None,
    mape: float | None = None,
    headroom_pct: float | None = None,
    mape_label: str = "MAPE",
) -> str:
    """
    Generate a screen-reader-friendly summary of the demand forecast tab.

    Honesty rule: only clauses backed by a real value are emitted. Any of
    ``peak_mw``/``peak_time``, ``mape``, or ``headroom_pct`` left as ``None``
    is dropped rather than announced as a fabricated or zero figure, so the
    aria-live region respects the warming/unavailable states.

    ``mape_label`` is rendered verbatim next to the accuracy figure — pass
    the metric name actually used (e.g. ``"sMAPE"`` or ``"live 7d MAPE"``)
    so an sMAPE value is never announced as MAPE.

    Returns:
        Plain text summary suitable for aria-live regions.
    """
    parts = [f"Demand forecast for {region}."]
    if peak_mw is not None and peak_time:
        parts.append(f"Today's peak demand is forecast at {peak_mw:,.0f} megawatts at {peak_time}.")
    if mape is not None:
        parts.append(f"Recent forecast accuracy is {mape:.1f}% {mape_label}.")
    if headroom_pct is not None:
        parts.append(f"Capacity headroom is {headroom_pct:.0f}%.")
    return " ".join(parts)


def scenario_summary(
    scenario_name: str,
    demand_delta: float,
    price_impact: float,
    headroom_pct: float,
) -> str:
    """
    Generate a screen-reader-friendly summary of a scenario simulation result.
    """
    direction = "increase" if demand_delta > 0 else "decrease"
    return (
        f"Scenario: {scenario_name}. "
        f"Demand would {direction} by {abs(demand_delta):,.0f} megawatts. "
        f"Estimated price impact: ${price_impact:.0f} per megawatt-hour. "
        f"Capacity headroom: {headroom_pct:.0f}%."
    )


# P2-42 (#273): the old TAB_KEY_MAP / KEYBOARD_SHORTCUTS constants described
# the pre-R4 8-tab shell, were referenced by nothing, and contradicted the
# actual bindings in assets/accessibility.js (the single source of truth for
# keyboard shortcuts). Deleted rather than fixed — a second map invites the
# same drift again.
