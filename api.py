"""Public read-only JSON API (v1) — #250.

Thin Flask blueprint over the same ``gridpulse:*`` Redis keys the web tier
reads. The scoring job is the only writer; these routes never fetch from
EIA/Open-Meteo or touch models on disk (web-tier I/O guardrail).

Honesty contract (mirrors the UI):

- Every payload carries provenance (``scored_at``, model identity, weights).
- Cold/warming Redis → **503** with ``{"status": "warming"}`` — never
  fabricated data.
- Unknown region → **404** with the valid-region list. The raw input is never
  reflected back.
- Prediction intervals are deliberately **omitted** until per-model interval
  calibration lands (#196) — the UI's current intervals are XGBoost-residual
  derived for every model selection, which is not honest enough to export.
- Capacity figures are EIA-860M **nameplate** (never "reserve margin"; #243).

Registered on the Flask ``server`` in ``app.py``.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from flask import Blueprint, jsonify

from config import (
    IS_IMPORT_DOMINATED,
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
    REGION_NAMES,
)
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")

#: Hourly data — a short public cache keeps repeat clients off Redis without
#: masking a fresh scoring tick for long.
_CACHE_SECONDS = 60

#: The scoring job writes 720 hourly rows (30 days), but the API deliberately
#: exports only the first 168h: the week most strongly driven by numerical
#: weather forecasts. Beyond Open-Meteo's ~16-day window the dashboard's
#: 30-day view leans on climatology inputs (ADR-008) — programmatic clients
#: shouldn't consume that tail as if it were a weather-driven forecast.
_MAX_HORIZON_HOURS = 168
_DEFAULT_HORIZON_HOURS = 24

#: Allow-list of exported model names. The internal Redis payloads are a
#: cache schema, not a contract — any future field the scoring job adds
#: (debug annotations, uncalibrated intervals) must NOT auto-publish to a
#: public trust boundary. Export only what is explicitly listed.
_EXPORTED_MODELS = ("prophet", "arima", "xgboost", "ensemble")
_EXPORTED_LIVE_DRIFT_FIELDS = ("rolling_mape_7d", "rolling_mape_30d", "n_records")
_EXPORTED_HORIZON_DRIFT_FIELDS = ("rolling_mape_7d", "grade", "n_records")

#: In-process memo for the fan-out endpoints (/regions, /grid/summary): they
#: aggregate ~50-100 Redis reads per request and their bodies are identical
#: for every client, so an unauthenticated cache-busting client must not be
#: able to translate requests 1:1 into Redis fan-outs (shared-fate with the
#: Dash UI on the same Cloud Run instance). Success bodies only — warming
#: states are never memoized, so first data is never delayed.
_MEMO_TTL_SECONDS = 30.0
_memo: dict[str, tuple[float, Any]] = {}


def _memo_get(key: str) -> Any | None:
    hit = _memo.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


def _memo_set(key: str, body: Any) -> None:
    _memo[key] = (time.monotonic() + _MEMO_TTL_SECONDS, body)


@api_v1.after_request
def _api_headers(resp):
    """Permissive CORS for read-only GETs; cache successes only.

    Non-200s get ``no-store`` — an explicit ``public, max-age`` on a 503
    would let a shared cache keep serving "warming" for up to 60s after
    the scoring tick lands (RFC 9111 allows caching any response with an
    explicit freshness directive). 503s also carry ``Retry-After``.
    """
    resp.headers["Access-Control-Allow-Origin"] = "*"
    if resp.status_code == 200:
        resp.headers["Cache-Control"] = f"public, max-age={_CACHE_SECONDS}"
    else:
        resp.headers["Cache-Control"] = "no-store"
        if resp.status_code == 503:
            resp.headers["Retry-After"] = "60"
    return resp


def _resolve_region(raw: str) -> str | None:
    """Uppercased region code when known, else ``None`` (never reflect raw)."""
    code = (raw or "").strip().upper()
    return code if code in REGION_NAMES else None


def _unknown_region_response():
    return (
        jsonify(
            {
                "error": "unknown_region",
                "valid_regions": sorted(REGION_NAMES),
            }
        ),
        404,
    )


def _warming_response(detail: str):
    """503 + explicit warming status — the API never fabricates data."""
    return (
        jsonify({"status": "warming", "detail": detail}),
        503,
    )


@api_v1.get("")
@api_v1.get("/")
def index():
    """Endpoint index — the hand-written v1 'docs'."""
    return jsonify(
        {
            "service": "gridpulse-api",
            "version": "v1",
            "description": (
                "Read-only access to GridPulse demand forecasts, grid state, "
                "and model-drift grades for 51 US balancing authorities."
            ),
            "endpoints": {
                "GET /api/v1/regions": "Balancing authorities + metadata",
                "GET /api/v1/forecast/{region}?horizon=24": (
                    "Hourly demand forecast (ensemble + per-model), max horizon 168"
                ),
                "GET /api/v1/grid/summary": (
                    "National totals: demand, simultaneous 24h peak, utilization"
                ),
                "GET /api/v1/drift/{region}": (
                    "Live 1h drift + horizon-matched (24/48/72h) accuracy grades"
                ),
            },
            "notes": [
                "Data updates hourly from the scoring pipeline (EIA-930 + Open-Meteo).",
                "Prediction intervals are omitted until per-model calibration (#196).",
                "Capacity figures are EIA-860M nameplate, not accredited capacity.",
                "Forecast horizon is capped at 168h: the week most strongly driven "
                "by numerical weather forecasts. The dashboard's longer view leans "
                "on climatology beyond the weather window (ADR-008), which the API "
                "does not export as if it were a weather-driven forecast.",
            ],
        }
    )


@api_v1.get("/regions")
def regions():
    """The 51 balancing authorities with coordinates + capacity metadata."""
    from models.model_service import is_forecast_quality_acceptable

    memoized = _memo_get("regions")
    if memoized is not None:
        return jsonify(memoized)

    out = []
    for code in sorted(REGION_NAMES):
        coords = REGION_COORDINATES.get(code, {})
        out.append(
            {
                "code": code,
                "name": REGION_NAMES[code],
                "lat": coords.get("lat"),
                "lon": coords.get("lon"),
                "nameplate_capacity_mw": REGION_CAPACITY_MW.get(code),
                "import_dominated": code in IS_IMPORT_DOMINATED,
                # Quality-gated = XGBoost holdout MAPE in the rollback grade;
                # the UI hides these regions, the API discloses them instead.
                "quality_gated": not is_forecast_quality_acceptable(code),
            }
        )
    body = {"count": len(out), "regions": out}
    _memo_set("regions", body)
    return jsonify(body)


@api_v1.get("/forecast/<raw_region>")
def forecast(raw_region: str):
    """Hourly demand forecast for one region, ensemble + per-model series."""
    from flask import request

    region = _resolve_region(raw_region)
    if region is None:
        return _unknown_region_response()

    horizon_arg = request.args.get("horizon", str(_DEFAULT_HORIZON_HOURS))
    try:
        horizon = int(horizon_arg)
    except (TypeError, ValueError):
        return (
            jsonify(
                {
                    "error": "invalid_horizon",
                    "detail": f"horizon must be an integer between 1 and {_MAX_HORIZON_HOURS}",
                }
            ),
            400,
        )
    if horizon < 1 or horizon > _MAX_HORIZON_HOURS:
        return (
            jsonify(
                {
                    "error": "invalid_horizon",
                    "detail": f"horizon must be an integer between 1 and {_MAX_HORIZON_HOURS}",
                }
            ),
            400,
        )

    payload = redis_get(redis_key(f"forecast:{region}:1h"))
    if not isinstance(payload, dict) or not payload.get("forecasts"):
        return _warming_response(
            "No forecast in cache for this region yet — the hourly scoring "
            "job populates it; retry shortly."
        )

    rows_in = payload["forecasts"][:horizon]
    # The production series is the ensemble when present (ADR-004); fall back
    # to the payload's primary model and say so — never silently.
    has_ensemble = any("ensemble" in r for r in rows_in)
    series_source = "ensemble" if has_ensemble else payload.get("primary_model", "unknown")

    rows_out: list[dict[str, Any]] = []
    for r in rows_in:
        rows_out.append(
            {
                "timestamp": r.get("timestamp"),
                "demand_mw": r.get("ensemble", r.get("predicted_demand_mw")),
                # Allow-list, never pass-through: unknown fields a future
                # writer adds to the cache schema must not auto-publish.
                "by_model": {name: r[name] for name in _EXPORTED_MODELS if name in r},
            }
        )

    body: dict[str, Any] = {
        "region": region,
        "name": REGION_NAMES[region],
        "scored_at": payload.get("scored_at"),
        "granularity": payload.get("granularity", "1h"),
        "series_source": series_source,
        "horizon_hours": len(rows_out),
        "forecast": rows_out,
        "notes": [
            "Prediction intervals omitted until per-model calibration (#196).",
        ],
    }
    if payload.get("ensemble_weights"):
        body["ensemble_weights"] = payload["ensemble_weights"]
    if payload.get("model_metrics"):
        body["holdout_metrics"] = payload["model_metrics"]
    return jsonify(body)


@api_v1.get("/grid/summary")
def grid_summary():
    """National roll-up — the same semantics as the US Grid tab's KPI bar.

    Reuses the tab's own helpers — including the implausible-artifact filter
    (#225 class) — so the API and the UI cannot disagree about totals,
    "national utilization", or "top stress" (single source of truth for the
    artifact / import-dominated / reliability-ceiling exclusions). Artifact
    exclusions are disclosed in the body rather than silent.
    """
    from components._callbacks_us_grid import (
        _STRESS_RELIABLE_CEILING,
        _collect_us_grid_region_data,
        _is_implausible_demand_artifact,
        _is_real_positive,
        _simultaneous_national_peak_mw,
    )
    from models.model_service import hidden_regions

    memoized = _memo_get("grid_summary")
    if memoized is not None:
        return jsonify(memoized)

    region_data = _collect_us_grid_region_data()
    populated = {r: d for r, d in region_data.items() if _is_real_positive(d.get("current_mw"))}
    if not populated:
        return _warming_response(
            "No regional demand in cache yet — the hourly scoring job populates it; retry shortly."
        )

    # Mirror the US Grid KPI bar exactly: artifact readings (a latest value
    # far below the BA's own 24h median — the #225 glitch class) are excluded
    # from every aggregate, and the exclusion is DISCLOSED rather than silent.
    plausible = {
        r: d
        for r, d in populated.items()
        if not _is_implausible_demand_artifact(
            d["current_mw"], d.get("today_mw") or [], d.get("prev_mw")
        )
    }
    artifact_excluded = sorted(set(populated) - set(plausible))
    if not plausible:
        return _warming_response(
            "Regional demand in cache looks like publishing artifacts — "
            "waiting for a clean scoring tick; retry shortly."
        )

    total_mw = sum(d["current_mw"] for d in plausible.values())
    peak_24h_mw = _simultaneous_national_peak_mw(plausible)

    stress_by_region = {
        r: d["current_mw"] / cap
        for r, d in plausible.items()
        if (cap := REGION_CAPACITY_MW.get(r, 0)) > 0 and r not in IS_IMPORT_DOMINATED
    }
    reliable = {r: s for r, s in stress_by_region.items() if s <= _STRESS_RELIABLE_CEILING}

    top_stress: dict[str, Any] | None = None
    national_utilization_pct: float | None = None
    if reliable:
        top_region = max(reliable, key=reliable.get)
        top_stress = {
            "region": top_region,
            "name": REGION_NAMES.get(top_region),
            "utilization_pct": round(min(reliable[top_region], 1.0) * 100, 1),
        }
        util_demand = sum(plausible[r]["current_mw"] for r in reliable)
        util_capacity = sum(REGION_CAPACITY_MW[r] for r in reliable)
        national_utilization_pct = round(util_demand / util_capacity * 100, 1)

    body = {
        "reporting_regions": len(plausible),
        "total_demand_mw": round(total_mw, 1),
        "simultaneous_peak_24h_mw": round(peak_24h_mw, 1),
        # Nameplate-based; not a NERC reserve margin (#243). Computed over
        # the reliable-capacity BA set only (import-dominated excluded).
        "national_utilization_pct": national_utilization_pct,
        "top_stress": top_stress,
        "artifact_excluded_regions": artifact_excluded,
        "quality_gated_regions": sorted(hidden_regions(REGION_NAMES.keys())),
        "notes": [
            "Utilization is against EIA-860M nameplate capacity — not a "
            "NERC reserve margin (see issue #243).",
            "artifact_excluded_regions carry a latest reading far below "
            "their own 24h median (EIA publishing glitch) and are excluded "
            "from all aggregates, matching the dashboard.",
        ],
    }
    _memo_set("grid_summary", body)
    return jsonify(body)


def _export_drift_models(models_block: dict, allowed_fields: tuple[str, ...]) -> dict:
    """Allow-listed export of a drift models block — known models, known fields.

    The Redis payload is an internal cache schema; raw ``records`` arrays and
    any future writer-added fields must not auto-publish (#250 review).
    """
    out: dict[str, Any] = {}
    for name in _EXPORTED_MODELS:
        block = models_block.get(name)
        if not isinstance(block, dict):
            continue
        out[name] = {k: block.get(k) for k in allowed_fields if k in block}
    return out


@api_v1.get("/drift/<raw_region>")
def drift(raw_region: str):
    """Live 1h nowcast drift + horizon-matched (24/48/72h) grades."""
    region = _resolve_region(raw_region)
    if region is None:
        return _unknown_region_response()

    live = redis_get(redis_key(f"drift:{region}"))
    horizon = redis_get(redis_key(f"drift_horizon:{region}"))

    live_ok = isinstance(live, dict) and live.get("models")
    horizon_ok = isinstance(horizon, dict) and horizon.get("models")
    if not live_ok and not horizon_ok:
        return _warming_response(
            "No drift records for this region yet — they accumulate over "
            "the first hours (1h) to days (24-72h) after deploy."
        )

    body: dict[str, Any] = {
        "region": region,
        "name": REGION_NAMES[region],
        "live_1h": None,
        "by_horizon": None,
        "notes": [
            "1h drift is a nowcast diagnostic; models without a 1-hour anchor "
            "are expected to sit above the 1h band. Horizon-matched grades "
            "(24/48/72h) are the operating-horizon verdict (#227).",
        ],
    }
    if live_ok:
        body["live_1h"] = {
            "last_updated_at": live.get("last_updated_at"),
            "models": _export_drift_models(live["models"], _EXPORTED_LIVE_DRIFT_FIELDS),
        }
    if horizon_ok:
        by_model: dict[str, Any] = {}
        for name in _EXPORTED_MODELS:
            horizons_block = horizon["models"].get(name)
            if not isinstance(horizons_block, dict):
                continue
            by_model[name] = {
                h: {k: block.get(k) for k in _EXPORTED_HORIZON_DRIFT_FIELDS if k in block}
                for h, block in horizons_block.items()
                if isinstance(block, dict)
            }
        body["by_horizon"] = {
            "horizons": horizon.get("horizons", ["24h", "48h", "72h"]),
            "models": by_model,
        }
    return jsonify(body)
