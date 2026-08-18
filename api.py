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
- Capacity figures are EIA-860M **nameplate** for most BAs, or a peak×1.15
  **estimate** for the 7 peak-derived BAs (``capacity_source`` on ``/regions``
  disambiguates; #254) — never accredited capacity or a "reserve margin" (#243).

Registered on the Flask ``server`` in ``app.py``.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from flask import Blueprint, jsonify

from config import (
    IS_IMPORT_DOMINATED,
    PEAK_DERIVED_CAPACITY,
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
    REGION_NAMES,
    UNRELIABLE_CAPACITY,
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
# P2-21 (#273): n_7d/n_30d are the per-window post-filter sample counts —
# the honest denominators behind the rolling means (n_records is total
# retained history, which can be dominated by records older than either
# window). Exported so API consumers can gate exactly like the UI does.
_EXPORTED_LIVE_DRIFT_FIELDS = ("rolling_mape_7d", "rolling_mape_30d", "n_records", "n_7d", "n_30d")
_EXPORTED_HORIZON_DRIFT_FIELDS = ("rolling_mape_7d", "grade", "n_records", "n_7d")

#: Data-source attribution that must travel with redistributed values so
#: downstream API consumers can meet the upstream license terms. Open-Meteo
#: weather is CC-BY-4.0 (attribution required); EIA-930 and NWS are US-Gov
#: public-domain works credited as good practice. Emitted on the index and on
#: every data payload. Fuller UI-footer + commercial-posture work is #256.
_ATTRIBUTION = {
    "demand": {
        "source": "U.S. Energy Information Administration, Form EIA-930",
        "url": "https://www.eia.gov/opendata/",
        "license": "U.S. Government work (public domain)",
    },
    "weather": {
        "source": "Open-Meteo",
        "url": "https://open-meteo.com/",
        "license": "CC-BY-4.0",
    },
    "alerts": {
        "source": "NOAA / National Weather Service",
        "url": "https://www.weather.gov/",
        "license": "U.S. Government work (public domain)",
    },
}

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


@api_v1.before_request
def _api_rate_limit():
    """Per-IP fixed-window rate limit on ``/api/v1/*`` (#253).

    Enforced only in Redis-only mode (staging/prod, ``rate_limit_active``); dev
    is unthrottled. Fails open on any Redis error. A blocked caller gets a 429
    with ``Retry-After``; ``_api_headers`` then marks it ``no-store`` so a shared
    cache can't keep replaying the rejection.
    """
    from config import API_RATE_LIMIT_PER_MIN, rate_limit_active

    if not rate_limit_active():
        return None

    from flask import request

    from ratelimit import caller_ip, check_rate_limit, is_exempt

    ip = caller_ip(request)
    if is_exempt(ip):
        return None
    result = check_rate_limit("api", ip, API_RATE_LIMIT_PER_MIN)
    if result.allowed:
        return None
    log.warning("api_rate_limited", ip=ip, path=request.path, limit=API_RATE_LIMIT_PER_MIN)
    resp = jsonify(
        {
            "error": "rate_limited",
            "detail": f"Exceeded {API_RATE_LIMIT_PER_MIN} requests/min. Retry after the window resets.",
            "retry_after_seconds": result.retry_after,
        }
    )
    resp.status_code = 429
    resp.headers["Retry-After"] = str(result.retry_after)
    return resp


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
                "GET /api/v1/benchmark": (
                    "GridPulse vs each BA's own EIA-930 day-ahead forecast, "
                    "fleet rollup + every per-BA row"
                ),
                "GET /api/v1/benchmark/{region}": "One BA's benchmark row",
            },
            "notes": [
                "Data updates hourly from the scoring pipeline (EIA-930 + Open-Meteo).",
                "Prediction intervals are omitted until per-model calibration (#196).",
                "Capacity is EIA-860M nameplate for most BAs, or a peak×1.15 "
                "estimate for 7 peak-derived BAs (see capacity_source on /regions; "
                "#254) — never accredited capacity.",
                "Forecast horizon is capped at 168h: the week most strongly driven "
                "by numerical weather forecasts. The dashboard's longer view leans "
                "on climatology beyond the weather window (ADR-008), which the API "
                "does not export as if it were a weather-driven forecast.",
                "Benchmark figures carry the statistic that produced them, and "
                "the rules, exclusions and known limits are published in "
                "docs/BENCHMARK_METHODOLOGY.md — read the limits before quoting "
                "a number.",
            ],
            "attribution": list(_ATTRIBUTION.values()),
        }
    )


@api_v1.get("/regions")
def regions():
    """The 51 balancing authorities with coordinates + capacity metadata."""
    from models.model_service import (
        is_forecast_quality_acceptable,
        published_live_horizon,
    )

    memoized = _memo_get("regions")
    if memoized is not None:
        return jsonify(memoized)

    out = []
    for code in sorted(REGION_NAMES):
        coords = REGION_COORDINATES.get(code, {})
        live = published_live_horizon(code)
        live_grade = (
            {
                "horizon": live.get("horizon"),
                "grade": live.get("grade"),
                "champion": live.get("champion"),
                "champion_mape": live.get("champion_mape"),
                "measurement": live.get("measurement"),
            }
            if live
            else None
        )
        out.append(
            {
                "code": code,
                "name": REGION_NAMES[code],
                "lat": coords.get("lat"),
                "lon": coords.get("lon"),
                # capacity_mw is EIA-860M nameplate for most BAs, but a
                # peak-demand × 1.15 estimate for the 7 peak-derived BAs (#254) —
                # capacity_source disambiguates so it's never mislabeled
                # "nameplate" (the field was `nameplate_capacity_mw` pre-#254).
                "capacity_mw": REGION_CAPACITY_MW.get(code),
                "capacity_source": (
                    "peak_estimate" if code in PEAK_DERIVED_CAPACITY else "nameplate"
                ),
                "import_dominated": code in IS_IMPORT_DOMINATED,
                # Quality-gated = the BA's best served model (ensemble or
                # champion base, not XGBoost-alone; #255) is still in the 7d
                # rollback grade. The UI hides these regions; the API discloses.
                #
                # The measurement behind it is the TRAINING HOLDOUT against the
                # 7-day band — the most generous question we ask. It is stated
                # in `quality_gate_measurement` rather than left to be inferred,
                # because the horizon-matched serve-path grade beside it can and
                # does disagree (#349): a BA can sit at `rollback` for 24h while
                # passing this gate comfortably. Both are published; only the
                # first one hides anything.
                "quality_gated": not is_forecast_quality_acceptable(code),
                "quality_gate_measurement": "training-holdout, 7d band",
                "operating_horizon_grade": live_grade,
            }
        )
    body = {"count": len(out), "regions": out}
    _memo_set("regions", body)
    return jsonify(body)


@api_v1.get("/scenario/<raw_region>")
def scenario(raw_region: str):
    """The precomputed what-if grid for one region (#127).

    Reads ``gridpulse:scenario_grid:{region}`` through the same helper the
    web tier uses, so this exercises the serving path rather than just
    confirming the payload exists — the scoring job writing a grid and the
    UI being able to read one are different claims.

    Optional ``temp``/``wind``/``solar`` query params return the interpolated
    hourly factor curve at that slider position instead of the whole grid;
    omitting them returns the raw grid. The two together are what make the
    physics auditable: whether the response is BA-dependent (the #119
    heuristic is BA-independent by construction) and whether the CDD/HDD kink
    at 65 F is visible in the temperature axis.
    """
    from flask import request

    from config import feature_enabled

    # Gated on the same flag as the data it serves. With the flag off no
    # scoring job writes a grid, so the endpoint could only ever return a
    # warming response — and a public surface that exists solely to say
    # "nothing here" is worse than one that is not published yet. 404 rather
    # than 503: the resource does not exist, it is not temporarily cold.
    if not feature_enabled("scenario_grid"):
        return (
            jsonify(
                {
                    "error": "not_found",
                    "detail": "The scenario grid is not enabled on this deployment.",
                }
            ),
            404,
        )

    region = _resolve_region(raw_region)
    if region is None:
        return _unknown_region_response()

    payload = redis_get(redis_key(f"scenario_grid:{region}"))
    if not isinstance(payload, dict) or not payload.get("factors"):
        return _warming_response(
            "No scenario grid in cache for this region — it is written by the "
            "hourly scoring job behind the `scenario_grid` flag."
        )

    args = request.args
    if not any(k in args for k in ("temp", "wind", "solar")):
        return jsonify({"region": region, **payload})

    try:
        temp = float(args.get("temp", 0.0))
        wind = float(args.get("wind", 0.0))
        solar = float(args.get("solar", 0.0))
    except (TypeError, ValueError):
        return (
            jsonify({"error": "invalid_delta", "detail": "temp/wind/solar must be numbers"}),
            400,
        )

    from simulation.scenario_grid import interpolate_scenario_factors

    curve = interpolate_scenario_factors(payload, temp, wind, solar)
    if curve is None:
        return _warming_response("Scenario grid present but unusable for this position.")

    # Say whether this position is inside the model's observed range. A tree
    # ensemble does not extrapolate: past the training envelope the response
    # goes flat, then unconstrained. Returning the numbers without this flag
    # would present an extrapolation as a forecast.
    env = payload.get("envelope") or {}
    axes = payload.get("axes") or {}

    def _in(axis: str, value: float) -> bool:
        flags, positions = env.get(axis), axes.get(axis)
        if not flags or not positions or len(flags) != len(positions):
            return True
        nearest = min(range(len(positions)), key=lambda i: abs(positions[i] - value))
        return bool(flags[nearest])

    extrapolated = not all((_in("temp_f", temp), _in("wind_mph", wind), _in("solar_wm2", solar)))

    return jsonify(
        {
            "region": region,
            "deltas": {"temp_f": temp, "wind_mph": wind, "solar_wm2": solar},
            "generated_at": payload.get("generated_at"),
            "extrapolated": extrapolated,
            "origin_drift": payload.get("origin_drift"),
            "factors": [round(float(v), 5) for v in curve],
        }
    )


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
    # A region served the seasonal-naive baseline (models/skill.py) must never
    # read as a model forecast here — the substitution exists BECAUSE the model
    # was worse than the baseline, and publishing it as "ensemble" would hide
    # exactly the fact that justified it.
    if payload.get("served_series") == "seasonal-naive":
        series_source = "seasonal-naive-baseline"
        # The headline series MUST follow the label. Reading `ensemble` here
        # published the model's numbers under a "baseline" source — a false
        # disclosure, and worse than not substituting at all, because a
        # consumer would trust the label. The substituted series lives in
        # `baseline` / `predicted_demand_mw`; the models stay in `by_model`.
        headline_keys = ("baseline", "predicted_demand_mw")
    else:
        series_source = "ensemble" if has_ensemble else payload.get("primary_model", "unknown")
        headline_keys = ("ensemble", "predicted_demand_mw")

    rows_out: list[dict[str, Any]] = []
    for r in rows_in:
        rows_out.append(
            {
                "timestamp": r.get("timestamp"),
                "demand_mw": next(
                    (r[k] for k in headline_keys if k in r),
                    r.get("predicted_demand_mw"),
                ),
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
        "served_reason": payload.get("served_reason"),
        "skill": payload.get("skill"),
        "horizon_hours": len(rows_out),
        "forecast": rows_out,
        "notes": [
            "Prediction intervals omitted until per-model calibration (#196).",
        ],
        # Weather-driven demand forecast → both the EIA target and the
        # Open-Meteo (CC-BY) feature source travel with the payload.
        "attribution": [_ATTRIBUTION["demand"], _ATTRIBUTION["weather"]],
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
    # #309: since the scoring job pre-cleans the payload series, read-time
    # detection alone would go blind to exclusions that already happened —
    # merge in the verdicts the guard stamped on each payload.
    stamped = {r for r, d in populated.items() if d.get("artifact_excluded")}
    artifact_excluded = sorted((set(populated) - set(plausible)) | stamped)
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
        # Exclude BAs without a reliable measured plate — import-dominated OR
        # peak-derived (cap = peak×1.15, so util is self-referential; #254).
        # Mirrors the US Grid KPI bar exactly (single source of truth in config).
        if (cap := REGION_CAPACITY_MW.get(r, 0)) > 0 and r not in UNRELIABLE_CAPACITY
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
        # Nameplate-based; not a NERC reserve margin (#243). Computed over the
        # reliable-capacity BA set only (import-dominated + peak-derived excluded,
        # #254).
        "national_utilization_pct": national_utilization_pct,
        "top_stress": top_stress,
        "artifact_excluded_regions": artifact_excluded,
        "quality_gated_regions": sorted(hidden_regions(REGION_NAMES.keys())),
        "notes": [
            "Utilization is against EIA-860M nameplate capacity — not a "
            "NERC reserve margin (see issue #243). BAs whose capacity is not a "
            "reliable stress denominator are excluded from utilization and "
            "top_stress (#254): import-dominated BAs (served load far exceeds "
            "in-territory plate) and peak-derived BAs (capacity_source="
            "peak_estimate on /regions). The two sets overlap but differ — an "
            "import-dominated BA can still carry a true nameplate (e.g. SPA).",
            "artifact_excluded_regions carry a latest reading far below "
            "their own 24h median (EIA publishing glitch) and are excluded "
            "from all aggregates, matching the dashboard.",
        ],
        "attribution": [_ATTRIBUTION["demand"]],
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
        # Drift = forecast-vs-actual demand; the EIA-930 actual is the anchor.
        "attribution": [_ATTRIBUTION["demand"]],
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


# ── Public forecast benchmark (E0-3) ─────────────────────────

#: Export allow-list for one lead block. The Redis payload is a cache schema,
#: not a contract — a future debug field must not auto-publish here. Both
#: official arms ship, and both verdicts, because publishing only the
#: favourable scoring is the objection the dual arm exists to pre-empt
#: (docs/BENCHMARK_METHODOLOGY.md §6).
_EXPORTED_BENCHMARK_LEAD_FIELDS = (
    "scoreable",
    "n",
    "official",
    "official_revised",
    "gridpulse",
    "delta_mape",
    "delta_wape",
    "delta_median_ape",
    "delta_mape_vs_revised",
    "winner",
    "winner_vs_revised",
    "excluded_hours",
    "observed_lead_h",
    "lead_basis",
    "conservative",
    "conservative_basis",
    "reason",
    # #348: our own live grade for the series this row scores. Same model,
    # same lead — not a healthier neighbouring measurement.
    "serve_grade",
    # #358: how excluding backfilled hours moved this BA's numbers. Published
    # because the direction is not uniform across the fleet, so a single
    # methodology sentence would be wrong for roughly half of it.
    "stale_capture_impact",
)

#: Every metric block travels with the statistic that produced it. §8 of the
#: methodology forbids publishing a per-BA figure without metric, window, `n`
#: and arm — this is that rule expressed as a field.
_BENCHMARK_STATISTICS = {
    "mape": "mean absolute percentage error over the paired hours",
    "median_ape": "median absolute percentage error over the paired hours",
    "mae": "mean absolute error, MW",
    "wape": "sum|error| / sum|actual|, percent",
}

_BENCHMARK_NOTES = [
    "Both arms are scored on the SAME hours against the SAME settled actual "
    "(EIA's revised value), so neither is graded on its own yardstick.",
    "Hours where EIA published the day-ahead forecast AS the actual are "
    "excluded — scoring them would credit the official forecast with a "
    "perfect prediction on hours it never made. Per-reason drop counts ship "
    "as excluded_hours.",
    "That exclusion protects the truth side only. Our forecast anchors on the "
    "last hour carrying a positive D, so where that hour is one of the same "
    "placeholders, the anchor seeding our forecast is the BA's own day-ahead "
    "value — the hour we score is dropped, the hour that seeded it is not. "
    "placeholder_pct reports how often the BA's readings arrive that way. The "
    "effect correlates our error with the operator's rather than shrinking "
    "it; see the methodology's limits.",
    "The official arm is scored twice: as-issued (the earliest day-ahead "
    "forecast we observed, the fair comparison) and as-revised (EIA's current "
    "value, which for a revising BA carries hindsight). Both verdicts are "
    "published.",
    "Verdicts are decided on mean MAPE. median_ape, mae and wape are "
    "published for interpretation and decide nothing.",
    "Leads are nominal unless lead_basis is 'observed'. Our realized lead is "
    "shorter than its label, and the 24h arm is NOT lead-matched against the "
    "operators' documented 17-41h submission window — see the methodology's "
    "limits before quoting a lead.",
    "Excluded BAs are published with their reason rather than omitted.",
    "Hours first seen more than a few hours after they passed are excluded "
    "(stale_capture): their earliest-observed forecast is a post-revision "
    "value, so scoring it as 'as-issued' would collapse the distinction the "
    "dual official arm exists to draw. Where any were excluded, "
    "stale_capture_impact reports how their removal moved both arms.",
    "Rows carry serve_grade — our own rolling grade for the exact series "
    "scored, at the same horizon. A row graded 'rollback' is one we already "
    "know is failing, and it is marked rather than presented as an ordinary "
    "comparison.",
    "This arm always scores the model, so where serves_scored_model is false "
    "the row measures the forecaster rather than the series a user of that "
    "BA is served — see served_series.",
]

_BENCHMARK_DOCS = {
    "methodology": (
        "https://github.com/kristenmartino/gridpulse/blob/main/docs/BENCHMARK_METHODOLOGY.md"
    ),
    "scoreability": (
        "https://github.com/kristenmartino/gridpulse/blob/main/docs/BENCHMARK_SCOREABILITY.md"
    ),
    "provenance": (
        "https://github.com/kristenmartino/gridpulse/blob/main/docs/BENCHMARK_PROVENANCE.md"
    ),
}


def _export_isolated(fleet: Any) -> dict[str, Any] | None:
    """Isolated regions (ERCOT), through the same lead allow-list."""
    if not isinstance(fleet, dict) or not isinstance(fleet.get("isolated"), dict):
        return None
    return {
        region: {k: block[k] for k in _EXPORTED_BENCHMARK_LEAD_FIELDS if k in block}
        for region, block in fleet["isolated"].items()
        if isinstance(block, dict)
    }


def _export_excluded_list(fleet: Any) -> list[dict[str, Any]] | None:
    """The rollup's excluded regions, region + reason only."""
    if not isinstance(fleet, dict) or not isinstance(fleet.get("excluded"), list):
        return None
    return [
        {"region": e.get("region"), "reason": e.get("reason")}
        for e in fleet["excluded"]
        if isinstance(e, dict)
    ]


def _export_benchmark_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """One BA's payload, allow-listed for a public trust boundary."""
    leads = {}
    for lead, block in (payload.get("leads") or {}).items():
        if isinstance(block, dict):
            leads[lead] = {k: block[k] for k in _EXPORTED_BENCHMARK_LEAD_FIELDS if k in block}
    return {
        "region": payload.get("region"),
        "name": REGION_NAMES.get(payload.get("region", ""), payload.get("region")),
        # Per-BA freshness: the fleet key's updated_at stamps the rollup, not
        # this row, and the per-region route has no fleet key to fall back on.
        "scored_at": payload.get("scored_at"),
        "scoreable": bool(payload.get("scoreable")),
        "reason": payload.get("reason"),
        "reason_detail": payload.get("reason_detail"),
        "revision_class": payload.get("revision_class"),
        # BOTH coverages, deliberately (#535). `df_coverage` is the BA's
        # publication rate and gates the exclusion; `df_asissued_coverage` is
        # OUR capture rate and gates nothing. Publishing only the first is
        # exactly how #535 hid for three weeks — the excluded BAs' reason said
        # "this BA barely publishes" and no public field could contradict it.
        # Exporting one without the other rebuilds that blind spot.
        "df_coverage": payload.get("df_coverage"),
        "df_asissued_coverage": payload.get("df_asissued_coverage"),
        "placeholder_pct": payload.get("placeholder_pct"),
        # #348: which series this row scores, and which one the product
        # actually serves for this BA. They differ wherever a BA was
        # substituted onto the seasonal-naive baseline, and a reader has no
        # way to know that from the numbers alone.
        "scored_model": payload.get("scored_model"),
        "served_series": payload.get("served_series"),
        "serves_scored_model": payload.get("serves_scored_model"),
        "leads": leads,
    }


#: Shown when no BA has accumulated enough paired hours yet.
BENCHMARK_WARMING_DETAIL = (
    "No benchmark results yet — a BA needs roughly nine days of "
    "scoring ticks before enough matured forecasts pair with settled "
    "actuals to publish a verdict."
)


def build_benchmark_payload() -> dict[str, Any] | None:
    """The fleet benchmark body, or None when nothing is scoreable yet.

    Extracted from the route so the ``/benchmark`` *page* can server-render
    from the identical structure the endpoint returns, rather than growing a
    second, friendlier path to the same Redis keys. The allow-list
    (``_export_benchmark_payload``) is applied here, so an SSR consumer
    cannot reach a field an API consumer could not.

    Shares the endpoint's 30-second memo: a crawl burst on the page must not
    fan out 51 Redis reads per request.
    """
    memoized = _memo_get("benchmark")
    if memoized is not None:
        return memoized

    fleet = redis_get(redis_key("meta:benchmark_fleet"))
    regions_out = []
    for code in sorted(REGION_NAMES):
        payload = redis_get(redis_key(f"benchmark:{code}"))
        if isinstance(payload, dict) and payload.get("region"):
            regions_out.append(_export_benchmark_payload(payload))

    if not regions_out:
        return None

    body = {
        "comparison": "GridPulse forecast vs the balancing authority's own day-ahead forecast",
        "truth": "EIA's settled demand value for the hour, used for both arms",
        "window_days": 30,
        "updated_at": fleet.get("updated_at") if isinstance(fleet, dict) else None,
        "fleet": fleet.get("fleet") if isinstance(fleet, dict) else None,
        "n_scoreable": fleet.get("n_scoreable") if isinstance(fleet, dict) else None,
        "n_excluded": fleet.get("n_excluded") if isinstance(fleet, dict) else None,
        # The rollup's own list, so a consumer can reconcile the count above
        # against named regions instead of trusting it.
        "excluded": _export_excluded_list(fleet),
        # Same allow-list as every other lead block — an isolated region is
        # not a back door for internal fields.
        "isolated": _export_isolated(fleet),
        "statistics": _BENCHMARK_STATISTICS,
        "regions": regions_out,
        "notes": _BENCHMARK_NOTES,
        "docs": _BENCHMARK_DOCS,
        "attribution": [_ATTRIBUTION["demand"]],
    }
    _memo_set("benchmark", body)
    return body


@api_v1.get("/benchmark")
def benchmark():
    """GridPulse vs each BA's own EIA-930 day-ahead forecast.

    The fleet rollup plus every per-BA row — including the excluded ones,
    which carry their reason. Rules, limits and reproduction scripts:
    ``docs/BENCHMARK_METHODOLOGY.md``.
    """
    body = build_benchmark_payload()
    if body is None:
        return _warming_response(BENCHMARK_WARMING_DETAIL)
    return jsonify(body)


@api_v1.get("/benchmark/<raw_region>")
def benchmark_region(raw_region: str):
    """One BA's benchmark row, excluded ones included with their reason."""
    region = _resolve_region(raw_region)
    if region is None:
        return _unknown_region_response()

    payload = redis_get(redis_key(f"benchmark:{region}"))
    if not isinstance(payload, dict) or not payload.get("region"):
        return _warming_response(
            "No benchmark payload cached for this region. The hourly scoring "
            "job writes it once the region has settled vintage history; a "
            "region that is scored but still accumulating paired hours "
            "appears here with its reason instead."
        )

    body = _export_benchmark_payload(payload)
    body.update(
        {
            "window_days": 30,
            "statistics": _BENCHMARK_STATISTICS,
            "notes": _BENCHMARK_NOTES,
            "docs": _BENCHMARK_DOCS,
            "attribution": [_ATTRIBUTION["demand"]],
        }
    )
    return jsonify(body)
