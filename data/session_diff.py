"""
Session-aware change detection (Backlog NEXD-8: "What Changed Since Last Time?").

Computes a lightweight snapshot of dashboard metrics on each visit, persisted
to localStorage via dcc.Store.  On return visits, compares the current snapshot
with the previous one and produces a list of human-readable change items.

Pure logic — no Dash or I/O dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import pandas as pd
import structlog

log = structlog.get_logger()

# ── Severity ordering (higher = more urgent) ─────────────────────────
_SEVERITY_ORDER: dict[str, int] = {"warning": 3, "notable": 2, "info": 1}

# ── Persona relevance mapping ────────────────────────────────────────
_PERSONA_RELEVANCE: dict[str, list[str]] = {
    "demand": ["grid_ops", "trader"],
    "forecast": ["grid_ops", "trader", "renewables"],
    "alerts": ["grid_ops"],
    "generation": ["renewables", "grid_ops"],
    "models": ["data_scientist", "trader"],
    "data": ["grid_ops", "renewables", "trader", "data_scientist"],
}

MAX_CHANGES = 5


@dataclass
class SessionSnapshot:
    """Point-in-time capture of dashboard metrics for a region.

    Only scalar values — no DataFrames — to keep localStorage small.
    """

    region: str = ""
    persona: str = ""
    timestamp: str = ""  # ISO format

    # Demand metrics (from demand-store)
    peak_demand_mw: float | None = None
    avg_demand_24h: float | None = None

    # Forecast (from audit-store)
    forecast_peak_mw: float | None = None

    # Model accuracy (from audit-store)
    mape: dict[str, float] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)

    # Data freshness (from data-freshness-store)
    data_sources: dict[str, str] = field(default_factory=dict)

    # Optional — populated when available
    alert_count: int | None = None
    renewable_pct: float | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SessionSnapshot:
        """Reconstruct from a JSON-round-tripped dict."""
        if not d:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChangeItem:
    """A single detected change between two snapshots."""

    category: str  # demand | forecast | alerts | models | generation | data
    text: str  # Human-readable description
    severity: str  # info | notable | warning
    icon: str  # Unicode icon character
    persona_relevance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ChangeItem:
        """Reconstruct from dict."""
        if not d:
            return cls(category="", text="", severity="info", icon="")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Helpers ──────────────────────────────────────────────────────────


def _pct_change(old: float | None, new: float | None) -> float | None:
    """Percentage change from *old* to *new*.  Returns None when unusable."""
    if old is None or new is None:
        return None
    if old == 0:
        return None
    return ((new - old) / abs(old)) * 100.0


def format_relative_time(iso_timestamp: str) -> str:
    """Format an ISO timestamp as a human-friendly relative string."""
    try:
        then = datetime.fromisoformat(iso_timestamp)
        if then.tzinfo is None:
            then = then.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = now - then
        seconds = delta.total_seconds()
        if seconds < 0:
            return "just now"
        if seconds < 60:
            return "just now"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes}m ago"
        hours = int(minutes // 60)
        if hours < 24:
            return f"{hours}h ago"
        days = int(hours // 24)
        if days == 1:
            return "yesterday"
        return f"{days}d ago"
    except Exception:
        return ""


# ── Snapshot computation ─────────────────────────────────────────────


def compute_snapshot(
    region: str,
    persona: str,
    demand_df: pd.DataFrame | None = None,
    audit_data: dict | None = None,
    freshness_data: dict | None = None,
) -> SessionSnapshot:
    """Build a snapshot of current dashboard state from store data.

    Args:
        region: Active balancing authority.
        persona: Active persona ID.
        demand_df: Parsed demand DataFrame (may be None).
        audit_data: Parsed audit-store dict (may be None/empty).
        freshness_data: Parsed data-freshness-store dict (may be None).

    Returns:
        SessionSnapshot with available metrics populated.
    """
    snap = SessionSnapshot(
        region=region,
        persona=persona,
        timestamp=datetime.now(UTC).isoformat(),
    )

    # ── Demand metrics ───────────────────────────────────────────
    if demand_df is not None and not demand_df.empty:
        try:
            if "demand_mw" in demand_df.columns:
                snap.peak_demand_mw = float(demand_df["demand_mw"].max())
                # Average over last 24 rows (hourly data = 24h)
                tail = demand_df["demand_mw"].dropna().tail(24)
                if len(tail) > 0:
                    snap.avg_demand_24h = float(tail.mean())
        except Exception as exc:
            log.debug("snapshot_demand_error", error=str(exc))

    # ── Audit / model metrics ────────────────────────────────────
    if audit_data and isinstance(audit_data, dict):
        peak = audit_data.get("peak_forecast_mw", 0.0)
        if peak and peak > 0:
            snap.forecast_peak_mw = float(peak)

        mape = audit_data.get("mape")
        if mape and isinstance(mape, dict):
            snap.mape = {k: float(v) for k, v in mape.items() if v is not None}

        weights = audit_data.get("ensemble_weights")
        if weights and isinstance(weights, dict):
            snap.ensemble_weights = {k: float(v) for k, v in weights.items()}

        versions = audit_data.get("model_versions")
        if versions and isinstance(versions, dict):
            snap.model_versions = dict(versions)

    # ── Freshness metrics ────────────────────────────────────────
    if freshness_data and isinstance(freshness_data, dict):
        snap.data_sources = {
            k: v
            for k, v in freshness_data.items()
            if k not in ("timestamp", "latest_data") and isinstance(v, str)
        }

    return snap


# ── Diff computation ─────────────────────────────────────────────────


def compute_diff(
    previous: SessionSnapshot,
    current: SessionSnapshot,
    persona: str,
) -> list[ChangeItem]:
    """Compare two snapshots and produce a persona-filtered change list.

    Args:
        previous: Snapshot from the user's last visit.
        current: Snapshot from the current visit.
        persona: Active persona ID (for relevance filtering).

    Returns:
        Up to MAX_CHANGES items sorted by severity (warning first).
    """
    changes: list[ChangeItem] = []

    # ── Peak demand ──────────────────────────────────────────────
    pct = _pct_change(previous.peak_demand_mw, current.peak_demand_mw)
    if pct is not None and abs(pct) > 5:
        direction = "up" if pct > 0 else "down"
        icon = "\u2197" if pct > 0 else "\u2198"  # ↗ / ↘
        sev = "notable" if abs(pct) > 10 else "info"
        val = current.peak_demand_mw
        changes.append(
            ChangeItem(
                category="demand",
                text=f"Peak demand {direction} {abs(pct):.0f}% to {val:,.0f} MW",
                severity=sev,
                icon=icon,
                persona_relevance=_PERSONA_RELEVANCE["demand"],
            )
        )

    # ── Average demand 24h ───────────────────────────────────────
    pct = _pct_change(previous.avg_demand_24h, current.avg_demand_24h)
    if pct is not None and abs(pct) > 5:
        direction = "up" if pct > 0 else "down"
        icon = "\u2195"  # ↕
        changes.append(
            ChangeItem(
                category="demand",
                text=f"Average demand shifted {direction} {abs(pct):.0f}%",
                severity="info",
                icon=icon,
                persona_relevance=_PERSONA_RELEVANCE["demand"],
            )
        )

    # ── Forecast peak ────────────────────────────────────────────
    pct = _pct_change(previous.forecast_peak_mw, current.forecast_peak_mw)
    if pct is not None and abs(pct) > 5:
        direction = "upward" if pct > 0 else "downward"
        icon = "\U0001f4c8" if pct > 0 else "\U0001f4c9"  # 📈 / 📉
        changes.append(
            ChangeItem(
                category="forecast",
                text=f"Peak forecast revised {direction} {abs(pct):.0f}%",
                severity="notable",
                icon=icon,
                persona_relevance=_PERSONA_RELEVANCE["forecast"],
            )
        )

    # ── Alert count ──────────────────────────────────────────────
    if previous.alert_count is not None and current.alert_count is not None:
        diff = current.alert_count - previous.alert_count
        if diff > 0:
            changes.append(
                ChangeItem(
                    category="alerts",
                    text=f"{diff} new weather alert{'s' if diff > 1 else ''}",
                    severity="warning",
                    icon="\u26a0",  # ⚠
                    persona_relevance=_PERSONA_RELEVANCE["alerts"],
                )
            )
        elif diff < 0 and current.alert_count == 0:
            changes.append(
                ChangeItem(
                    category="alerts",
                    text="All weather alerts cleared",
                    severity="info",
                    icon="\u2714",  # ✔
                    persona_relevance=_PERSONA_RELEVANCE["alerts"],
                )
            )

    # ── Renewable share ──────────────────────────────────────────
    if previous.renewable_pct is not None and current.renewable_pct is not None:
        pp_diff = current.renewable_pct - previous.renewable_pct
        if abs(pp_diff) > 3:
            direction = "rose" if pp_diff > 0 else "fell"
            icon = "\U0001f33f" if pp_diff > 0 else "\U0001f4a8"  # 🌿 / 💨
            changes.append(
                ChangeItem(
                    category="generation",
                    text=f"Renewable share {direction} to {current.renewable_pct:.1f}%",
                    severity="info",
                    icon=icon,
                    persona_relevance=_PERSONA_RELEVANCE["generation"],
                )
            )

    # ── Model accuracy (MAPE) ────────────────────────────────────
    for model in set(previous.mape) | set(current.mape):
        old_mape = previous.mape.get(model)
        new_mape = current.mape.get(model)
        if old_mape is not None and new_mape is not None:
            pp_diff = new_mape - old_mape
            if abs(pp_diff) > 2:
                if pp_diff > 0:
                    changes.append(
                        ChangeItem(
                            category="models",
                            text=f"{model} accuracy degraded ({new_mape:.1f}% MAPE)",
                            severity="warning",
                            icon="\u2b07",  # ⬇
                            persona_relevance=_PERSONA_RELEVANCE["models"],
                        )
                    )
                else:
                    changes.append(
                        ChangeItem(
                            category="models",
                            text=f"{model} accuracy improved ({new_mape:.1f}% MAPE)",
                            severity="info",
                            icon="\u2b06",  # ⬆
                            persona_relevance=_PERSONA_RELEVANCE["models"],
                        )
                    )

    # ── Data source status ───────────────────────────────────────
    for source in set(previous.data_sources) | set(current.data_sources):
        old_status = previous.data_sources.get(source)
        new_status = current.data_sources.get(source)
        if old_status and new_status and old_status != new_status:
            degraded = new_status in ("stale", "demo", "error")
            changes.append(
                ChangeItem(
                    category="data",
                    text=f"{source.title()} data now {new_status}",
                    severity="warning" if degraded else "info",
                    icon="\U0001f534" if degraded else "\U0001f7e2",  # 🔴 / 🟢
                    persona_relevance=_PERSONA_RELEVANCE["data"],
                )
            )

    # ── Filter by persona relevance ──────────────────────────────
    filtered = [c for c in changes if persona in c.persona_relevance]

    # ── Sort by severity (warning first) and cap at MAX_CHANGES ──
    filtered.sort(key=lambda c: _SEVERITY_ORDER.get(c.severity, 0), reverse=True)
    return filtered[:MAX_CHANGES]
