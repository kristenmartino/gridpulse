"""
AI-powered executive briefing generator.

Uses Anthropic Claude API to produce persona-specific, data-driven
briefings for the Overview tab. Falls back to a deterministic
rule-based summary when no API key is configured or the API fails.

Cache: 15-minute TTL in SQLite (same pattern as news_client).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import pandas as pd
import structlog

from config import ANTHROPIC_API_KEY, REGION_CAPACITY_MW, REGION_NAMES
from data.cache import get_cache
from personas.config import get_persona

log = structlog.get_logger()

_CACHE_TTL = 900  # 15 minutes


@dataclass
class BriefingResult:
    """Container for an executive briefing."""

    summary: str
    observations: list[str] = field(default_factory=list)
    source: str = "rule_based"  # "claude" | "rule_based"
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()


def generate_briefing(
    persona_id: str,
    region: str,
    demand_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
) -> BriefingResult:
    """
    Generate an executive briefing for the given persona and region.

    Tries Claude API first (if key is set), then falls back to rule-based.
    Results are cached for 15 minutes keyed on (persona, region, data_hash).

    Args:
        persona_id: Active persona identifier.
        region: Active region code.
        demand_df: Current demand DataFrame.
        weather_df: Current weather DataFrame.

    Returns:
        BriefingResult with summary text and observations.
    """
    cache = get_cache()
    data_hash = _compute_data_hash(demand_df, weather_df)
    cache_key = f"briefing:{persona_id}:{region}:{data_hash}"

    cached = cache.get(cache_key)
    if cached is not None:
        log.info("briefing_cache_hit", persona=persona_id, region=region)
        return BriefingResult(**cached)

    context = _extract_data_context(region, demand_df, weather_df)

    result: BriefingResult
    if ANTHROPIC_API_KEY:
        try:
            result = _call_claude_api(persona_id, region, context)
        except Exception as exc:
            log.error("briefing_api_failed", error=str(exc))
            result = _build_rule_based_briefing(persona_id, region, context)
    else:
        result = _build_rule_based_briefing(persona_id, region, context)

    cache.set(cache_key, asdict(result), ttl=_CACHE_TTL)
    return result


# ---------------------------------------------------------------------------
# Claude API integration
# ---------------------------------------------------------------------------


def _call_claude_api(
    persona_id: str,
    region: str,
    context: dict,
) -> BriefingResult:
    """Call Anthropic Claude to generate a briefing."""
    import anthropic

    persona = get_persona(persona_id)
    region_name = REGION_NAMES.get(region, region)

    system_prompt = (
        f"You are a senior energy analyst providing a concise executive briefing "
        f"to a {persona.title} ({persona.name}) responsible for the {region_name} "
        f"balancing authority. Be direct, data-driven, and actionable. "
        f"Write in second person ('your region'). No markdown formatting. "
        f"No greetings or sign-offs."
    )

    user_content = _format_context_for_prompt(context)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=10.0)
    response = client.messages.create(
        model="claude-haiku-4-20250414",
        max_tokens=400,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on this live data for {region_name}, write:\n"
                    f"1. A 2-3 sentence executive summary\n"
                    f"2. Exactly 3 key observations as short bullet points\n\n"
                    f"Separate the summary from observations with '---'\n\n"
                    f"{user_content}"
                ),
            }
        ],
    )

    raw = response.content[0].text
    return _parse_claude_response(raw)


def _parse_claude_response(raw: str) -> BriefingResult:
    """Parse Claude's response into structured BriefingResult."""
    parts = raw.split("---", 1)
    summary = parts[0].strip()

    observations: list[str] = []
    if len(parts) > 1:
        for line in parts[1].strip().splitlines():
            line = line.strip().lstrip("-*").strip()
            if line:
                observations.append(line)

    return BriefingResult(
        summary=summary,
        observations=observations[:3],
        source="claude",
    )


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _build_rule_based_briefing(
    persona_id: str,
    region: str,
    context: dict,
) -> BriefingResult:
    """Build a deterministic briefing from extracted data context."""
    persona = get_persona(persona_id)
    region_name = REGION_NAMES.get(region, region)

    generators = {
        "grid_ops": _briefing_grid_ops,
        "renewables": _briefing_renewables,
        "trader": _briefing_trader,
        "data_scientist": _briefing_data_scientist,
    }

    gen = generators.get(persona_id, _briefing_grid_ops)
    summary, observations = gen(persona.name, region_name, context)

    return BriefingResult(
        summary=summary,
        observations=observations,
        source="rule_based",
    )


def _briefing_grid_ops(name: str, region: str, ctx: dict) -> tuple[str, list[str]]:
    """Grid Ops briefing: load, reserves, temperature."""
    parts = [f"System status for {region} is nominal."]
    observations = []

    if ctx["peak_mw"]:
        capacity = ctx["capacity"]
        util = ctx["peak_mw"] / capacity * 100 if capacity else 0
        parts.append(f"Peak demand reached {ctx['peak_mw']:,.0f} MW ({util:.0f}% of capacity).")
        reserve_pct = (capacity - ctx["peak_mw"]) / capacity * 100 if capacity else 0
        if reserve_pct < 15:
            observations.append(f"Reserve margin at {reserve_pct:.0f}% — below 15% threshold.")
        else:
            observations.append(f"Reserve margin comfortable at {reserve_pct:.0f}%.")

    if ctx["current_temp"] is not None:
        observations.append(f"Current temperature: {ctx['current_temp']:.0f}°F.")

    if ctx["mape"] is not None:
        grade = "accurate" if ctx["mape"] < 5 else "needs review"
        observations.append(f"Forecast accuracy: {ctx['mape']:.1f}% MAPE ({grade}).")

    return " ".join(parts), observations[:3]


def _briefing_renewables(name: str, region: str, ctx: dict) -> tuple[str, list[str]]:
    """Renewables briefing: wind, solar potential, generation share."""
    parts = [f"Renewable generation outlook for {region}."]
    observations = []

    if ctx["wind_speed"] is not None:
        wind_qual = (
            "strong" if ctx["wind_speed"] > 15 else "moderate" if ctx["wind_speed"] > 8 else "light"
        )
        parts.append(f"Wind conditions are {wind_qual} at {ctx['wind_speed']:.0f} mph average.")
        observations.append(
            f"Wind speed averaging {ctx['wind_speed']:.0f} mph over the last 24 hours."
        )

    if ctx["current_temp"] is not None:
        if ctx["current_temp"] > 85:
            observations.append(
                f"High temperatures ({ctx['current_temp']:.0f}°F) boosting solar irradiance but increasing cooling demand."
            )
        else:
            observations.append(f"Temperature at {ctx['current_temp']:.0f}°F.")

    if ctx["peak_mw"] and ctx["capacity"]:
        observations.append(
            f"System load at {ctx['peak_mw']:,.0f} MW peak — renewable share data available in Generation tab."
        )

    return " ".join(parts), observations[:3]


def _briefing_trader(name: str, region: str, ctx: dict) -> tuple[str, list[str]]:
    """Trader briefing: demand vs capacity, pricing signals."""
    parts = [f"Market conditions for {region}."]
    observations = []

    if ctx["peak_mw"] and ctx["capacity"]:
        util = ctx["peak_mw"] / ctx["capacity"] * 100
        tier = (
            "emergency"
            if util > 95
            else "high"
            if util > 85
            else "moderate"
            if util > 70
            else "base"
        )
        parts.append(f"System utilization at {util:.0f}%, pricing in {tier} tier.")
        observations.append(
            f"Peak load {ctx['peak_mw']:,.0f} MW vs {ctx['capacity']:,.0f} MW capacity."
        )

    if ctx["demand_vs_forecast_pct"] is not None:
        direction = "above" if ctx["demand_vs_forecast_pct"] > 0 else "below"
        observations.append(
            f"Actual demand running {abs(ctx['demand_vs_forecast_pct']):.1f}% {direction} forecast."
        )

    if ctx["current_temp"] is not None:
        observations.append(
            f"Temperature at {ctx['current_temp']:.0f}°F — monitor for demand-driven price spikes."
        )

    return " ".join(parts), observations[:3]


def _briefing_data_scientist(name: str, region: str, ctx: dict) -> tuple[str, list[str]]:
    """Data Scientist briefing: model performance, data quality."""
    parts = [f"Model diagnostics summary for {region}."]
    observations = []

    if ctx["mape"] is not None:
        grade = "excellent" if ctx["mape"] < 3 else "good" if ctx["mape"] < 5 else "degraded"
        parts.append(f"Ensemble MAPE at {ctx['mape']:.1f}% ({grade}).")
        observations.append(
            f"7-day rolling MAPE: {ctx['mape']:.1f}% — {'within target' if ctx['mape'] < 5 else 'investigate residuals'}."
        )

    if ctx["data_points"]:
        observations.append(f"Working with {ctx['data_points']} demand observations.")

    if ctx["weather_points"]:
        observations.append(
            f"{ctx['weather_points']} weather data points available for feature engineering."
        )

    return " ".join(parts), observations[:3]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_data_context(
    region: str,
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> dict:
    """Extract key statistics for briefing generation."""
    ctx: dict = {
        "peak_mw": None,
        "peak_time": None,
        "avg_mw": None,
        "current_temp": None,
        "max_temp": None,
        "wind_speed": None,
        "mape": None,
        "demand_vs_forecast_pct": None,
        "capacity": REGION_CAPACITY_MW.get(region, 50000),
        "data_points": 0,
        "weather_points": 0,
    }

    if demand_df is not None and len(demand_df) > 0:
        ctx["data_points"] = len(demand_df)
        df = demand_df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        recent = df.tail(24)
        if len(recent) > 0 and "demand_mw" in recent.columns:
            peak_idx = recent["demand_mw"].idxmax()
            ctx["peak_mw"] = float(recent.loc[peak_idx, "demand_mw"])
            ctx["avg_mw"] = float(recent["demand_mw"].mean())
            if "timestamp" in recent.columns:
                ctx["peak_time"] = str(recent.loc[peak_idx, "timestamp"])

        if "forecast_mw" in df.columns:
            tail = df.tail(168)
            mask = tail["demand_mw"].abs() > 1
            if mask.sum() > 0:
                mape = (
                    (tail.loc[mask, "demand_mw"] - tail.loc[mask, "forecast_mw"]).abs()
                    / tail.loc[mask, "demand_mw"].abs()
                ).mean() * 100
                ctx["mape"] = round(float(mape), 1)

            recent_mask = recent["demand_mw"].abs() > 1
            if "forecast_mw" in recent.columns and recent_mask.sum() > 0:
                actual_mean = recent.loc[recent_mask, "demand_mw"].mean()
                forecast_mean = recent.loc[recent_mask, "forecast_mw"].mean()
                if forecast_mean > 0:
                    ctx["demand_vs_forecast_pct"] = round(
                        float((actual_mean - forecast_mean) / forecast_mean * 100), 1
                    )

    if weather_df is not None and len(weather_df) > 0:
        ctx["weather_points"] = len(weather_df)
        wdf = weather_df.copy()
        if "timestamp" in wdf.columns:
            wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])

        recent_w = wdf.tail(24)
        if "temperature_2m" in recent_w.columns and len(recent_w) > 0:
            ctx["current_temp"] = float(recent_w["temperature_2m"].iloc[-1])
            ctx["max_temp"] = float(recent_w["temperature_2m"].max())
        if "wind_speed_80m" in recent_w.columns and len(recent_w) > 0:
            ctx["wind_speed"] = float(recent_w["wind_speed_80m"].mean())

    return ctx


def _format_context_for_prompt(context: dict) -> str:
    """Format extracted context as readable text for Claude prompt."""
    lines = []
    if context["peak_mw"]:
        lines.append(f"Peak demand (24h): {context['peak_mw']:,.0f} MW")
    if context["avg_mw"]:
        lines.append(f"Average demand (24h): {context['avg_mw']:,.0f} MW")
    if context["capacity"]:
        lines.append(f"System capacity: {context['capacity']:,.0f} MW")
    if context["peak_mw"] and context["capacity"]:
        util = context["peak_mw"] / context["capacity"] * 100
        lines.append(f"Utilization: {util:.1f}%")
    if context["current_temp"] is not None:
        lines.append(f"Current temperature: {context['current_temp']:.0f}°F")
    if context["max_temp"] is not None:
        lines.append(f"Max temperature (24h): {context['max_temp']:.0f}°F")
    if context["wind_speed"] is not None:
        lines.append(f"Average wind speed: {context['wind_speed']:.0f} mph")
    if context["mape"] is not None:
        lines.append(f"Forecast MAPE (7d): {context['mape']:.1f}%")
    if context["demand_vs_forecast_pct"] is not None:
        lines.append(f"Demand vs forecast: {context['demand_vs_forecast_pct']:+.1f}%")
    if context["data_points"]:
        lines.append(
            f"Data points: {context['data_points']} demand, {context['weather_points']} weather"
        )
    return "\n".join(lines) if lines else "No live data available."


def _compute_data_hash(
    demand_df: pd.DataFrame | None,
    weather_df: pd.DataFrame | None,
) -> str:
    """Compute a short hash of the data for cache key stability."""
    h = hashlib.md5(usedforsecurity=False)
    if demand_df is not None and len(demand_df) > 0:
        h.update(str(len(demand_df)).encode())
        h.update(str(demand_df["demand_mw"].iloc[-1]).encode())
    if weather_df is not None and len(weather_df) > 0:
        h.update(str(len(weather_df)).encode())
    return h.hexdigest()[:8]
