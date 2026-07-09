"""
Model service layer: single interface for all forecast consumers.

This module is the ONLY place callbacks should get model predictions from.
It handles the full chain: load trained models → predict → fall back to
simulated outputs if models aren't available.

Usage:
    from models.model_service import get_forecasts, get_model_metrics
    forecasts = get_forecasts(region, demand_df)
    # forecasts = {"prophet": np.array, "arima": np.array, "xgboost": np.array,
    #              "ensemble": np.array, "upper_80": np.array, "lower_80": np.array,
    #              "weights": dict, "metrics": dict, "source": "trained"|"simulated"}
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import MODEL_DIR
from hash_utils import stable_int_seed

log = structlog.get_logger()

# In-memory model cache: region → loaded model data
_model_cache: dict[str, dict[str, Any]] = {}


def get_forecasts(
    region: str,
    demand_df: pd.DataFrame,
    models_shown: list[str] | None = None,
) -> dict[str, Any]:
    """
    Get forecasts from all models for a region.

    Tries to load trained models from disk. If unavailable, generates
    deterministic simulated forecasts (seeded by region hash for consistency).

    Args:
        region: Balancing authority code (e.g., "FPL").
        demand_df: DataFrame with 'timestamp' and 'demand_mw' columns.
        models_shown: Optional filter — only compute these models.

    Returns:
        Dict with keys: prophet, arima, xgboost, ensemble, upper_80, lower_80,
        weights, metrics, source ("trained" or "simulated").
    """
    actual = demand_df["demand_mw"].values

    # Try trained models first
    model_data = _load_cached_models(region)
    if model_data is not None:
        return _predict_from_trained(model_data, demand_df, models_shown)

    # No trained models on local disk. In production/staging
    # (``REQUIRE_REDIS``) we must NOT fabricate plausible-looking
    # simulated forecasts — the web tier reads real forecasts from Redis,
    # and any inline-compute path reaching here means the real data is
    # genuinely unavailable. Return an explicit ``unavailable`` marker
    # (empty predictions) so the caller renders a warming/degraded state
    # instead of fake numbers. Simulated forecasts stay available only in
    # development / offline demo (``REQUIRE_REDIS=False``). (#149)
    from config import REQUIRE_REDIS

    if REQUIRE_REDIS:
        log.info("get_forecasts_unavailable_prod", region=region)
        return {"source": "unavailable", "metrics": {}, "weights": {}}

    # Dev / demo: deterministic simulated forecasts.
    return _simulate_forecasts(region, actual, models_shown)


_HOLDOUT_FIELDS = ("mape", "rmse", "mae", "r2")


def get_model_metrics(region: str) -> dict[str, dict[str, float]]:
    """
    Get validation metrics for a region's trained models.

    **All four metrics (MAPE / RMSE / MAE / R²) are training-time
    holdout values** — each model's last 168-hour holdout, computed
    by the daily training job and persisted in
    ``meta.extra["holdout_metrics"]``. The ensemble row is computed
    from the SAME holdout predictions and stashed in xgboost's meta
    extra under ``ensemble_holdout_metrics``. Provenance is uniform:
    no metric is silently sourced from a different distribution
    than its siblings.

    Resolution order:
    0. ``gridpulse:forecast:{region}:1h`` → ``model_metrics`` field —
       written by the scoring job (#131, 2026-05-20) from each model's
       ``meta.extra["holdout_metrics"]``. **This is the production
       path:** the web-tier container has no meta.json files on local
       disk (they live only on the Job container), so layers 1–3 and 5
       below all fail in production. Without this layer, every
       production page render falls all the way through to layer 6 —
       the hardcoded simulated baseline — which is what triggered the
       2026-05-20 "MAPE 1.6%" bug surfaced on the Overview model card.
    1. ``meta.extra["holdout_metrics"]`` per model — full {mape, rmse,
       mae, r2} dict, real training-time holdout. Canonical path for
       any BA whose models trained on the new code path. Used in dev
       (where meta.json is on local disk) and during training-job
       in-process callbacks.
    2. Top-level ``meta.mape`` per model — backward-compat for pickles
       trained before this path landed (only MAPE is real; RMSE/MAE/R²
       are still supplemented from Redis below).
    3. Ensemble metrics from xgboost's meta extra
       (``ensemble_holdout_metrics``).
    4. Redis diagnostics ``gridpulse:diagnostics:{region}`` —
       fallback ONLY for fields not provided by the meta path.
       In production this content is currently
       ``_simulate_forecasts``-derived; the meta-first ordering means
       a viewer never sees a simulated value when a real one exists.
    5. Local pickle (dev mode with in-process precompute).
    6. Hardcoded simulated baseline — last-resort dev fallback.

    Returns:
        Dict of model_name → {mape?, rmse?, mae?, r2?}. Empty fields
        omitted; the UI tolerates partial dicts.
    """
    # Layer 0: gridpulse:forecast:{region}:1h.model_metrics — written
    # by the scoring job. This is the production read path. Falls
    # through to layers 1-6 only when the field is empty/absent (offline
    # dev, fresh deploy before first scoring tick, or pre-#131 forecasts
    # still in Redis with their TTL not yet expired).
    try:
        from data.redis_client import redis_get, redis_key

        forecast_payload = redis_get(redis_key(f"forecast:{region}:1h"))
        if isinstance(forecast_payload, dict):
            cached = forecast_payload.get("model_metrics")
            if isinstance(cached, dict) and cached:
                cleaned: dict[str, dict[str, float]] = {}
                for model_name, metrics in cached.items():
                    if not isinstance(metrics, dict):
                        continue
                    model_clean: dict[str, float] = {}
                    for field in _HOLDOUT_FIELDS:
                        val = metrics.get(field)
                        if val is None:
                            continue
                        try:
                            f = float(val)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(f):
                            model_clean[field] = f
                    if model_clean:
                        cleaned[model_name] = model_clean
                if cleaned:
                    return cleaned
    except Exception as e:  # pragma: no cover — defensive (Redis outage)
        log.debug("get_model_metrics_redis_forecast_miss", region=region, error=str(e))

    out: dict[str, dict[str, float]] = {}

    # 1 + 2. Per-model holdout metrics from meta extras.
    ensemble_from_xgb_meta: dict[str, float] | None = None
    try:
        from models.persistence import get_model_metadata

        for model_name in ("prophet", "arima", "xgboost"):
            try:
                meta = get_model_metadata(region, model_name)
            except Exception:
                meta = None
            if meta is None:
                continue
            base: dict[str, float] = {}

            # Layer 1: full holdout metrics from extra (preferred).
            extra = getattr(meta, "extra", None) or {}
            holdout = extra.get("holdout_metrics") if isinstance(extra, dict) else None
            if isinstance(holdout, dict):
                for field in _HOLDOUT_FIELDS:
                    val = holdout.get(field)
                    if val is not None and np.isfinite(val):
                        base[field] = float(val)

            # Layer 2: legacy fallback — top-level meta.mape only.
            if "mape" not in base:
                mape = getattr(meta, "mape", None)
                if mape is not None and np.isfinite(mape) and mape > 0:
                    base["mape"] = float(mape)

            if base:
                out[model_name] = base

            # Layer 3: ensemble metrics ride along on xgboost's extra.
            if model_name == "xgboost" and isinstance(extra, dict):
                ens = extra.get("ensemble_holdout_metrics")
                if isinstance(ens, dict):
                    candidate = {
                        f: float(ens[f])
                        for f in _HOLDOUT_FIELDS
                        if f in ens and np.isfinite(ens[f])
                    }
                    if candidate:
                        ensemble_from_xgb_meta = candidate
    except Exception as e:  # pragma: no cover — defensive (GCS outage)
        log.debug("get_model_metrics_meta_miss", region=region, error=str(e))

    if ensemble_from_xgb_meta:
        out["ensemble"] = ensemble_from_xgb_meta

    # ── Production strict-fallback gate (#149) ───────────────────────
    # Layers 0-3 above are the only REAL metric sources: Redis
    # ``model_metrics`` (layer 0) and the per-model meta holdout
    # (layers 1-3). Layers 4-6 below are either ``_simulate_forecasts``-
    # derived (layer 4 diagnostics in production), dev-only local pickle
    # (layer 5), or a hardcoded baseline (layer 6). In production /
    # staging (``REQUIRE_REDIS``) a viewer must see real holdout metrics
    # or NOTHING — never a plausible-looking simulated/hardcoded value
    # (the failure mode behind the 2026-05-20 "MAPE 1.6%" model-card
    # bug). Returning ``{}`` here lets the model card render a
    # warming/unavailable state; every caller already does
    # ``get_model_metrics(region) or {}``.
    from config import REQUIRE_REDIS

    if REQUIRE_REDIS:
        if not out:
            log.info("get_model_metrics_unavailable_prod", region=region)
        return out

    # 4. Redis diagnostics — supplements ONLY fields the meta path
    # didn't provide. In production this is currently
    # ``_simulate_forecasts``-derived, so we only fall back to it
    # for legacy pickles or the ensemble row when xgboost's meta
    # didn't carry ensemble_holdout_metrics.
    try:
        from data.redis_client import redis_get, redis_key

        cached = redis_get(redis_key(f"diagnostics:{region}"))
        if cached and isinstance(cached.get("metrics"), dict):
            for model_name, redis_m in cached["metrics"].items():
                if not isinstance(redis_m, dict):
                    continue
                base = out.get(model_name, {})
                for field in _HOLDOUT_FIELDS:
                    if field in redis_m and field not in base:
                        base[field] = redis_m[field]
                if base:
                    out[model_name] = base
    except Exception as e:  # pragma: no cover — defensive
        log.debug("get_model_metrics_redis_miss", region=region, error=str(e))

    if out:
        return out

    # 5. Local pickle — dev mode.
    model_data = _load_cached_models(region)
    if model_data and "metrics" in model_data:
        return model_data["metrics"]

    # 6. Simulated baseline — last resort.
    return {
        "prophet": {"mape": 2.8, "rmse": 450, "mae": 320, "r2": 0.967},
        "arima": {"mape": 3.5, "rmse": 580, "mae": 410, "r2": 0.945},
        "xgboost": {"mape": 2.1, "rmse": 380, "mae": 280, "r2": 0.974},
        "ensemble": {"mape": 1.9, "rmse": 340, "mae": 250, "r2": 0.979},
    }


def get_ensemble_weights(region: str) -> dict[str, float]:
    """Get ensemble weights for a region."""
    model_data = _load_cached_models(region)
    if model_data and "ensemble_weights" in model_data:
        return model_data["ensemble_weights"]

    # Default weights (1/MAPE normalized)
    return {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}


def is_trained(region: str) -> bool:
    """Check if a real trained-model forecast is available for the region.

    The web tier's source of truth is Redis: ``gridpulse:forecast:{region}:1h``
    is written hourly by the scoring job, and the ``ensemble_weights`` field
    is only present when ≥2 trained base models produced finite predictions
    during that tick. Presence of ``ensemble_weights`` is therefore the
    tightest "real trained models are live for this region" signal we can
    read without scanning GCS.

    The pre-2026-05-20 implementation checked for a local ``trained_models/
    {region}_models.pkl`` file, which in production *always* returns False
    on the web tier (only the Cloud Run Job container has trained pickles
    on disk). That caused every region to render the "simulated" badge on
    the Overview model card even when real trained-model output was being
    rendered in the forecast chart and Forecast tab.

    Falls back to the legacy local-pickle check when Redis isn't reachable
    (dev mode with no Memorystore). Returns False for unknown regions.
    """
    from models.training import _safe_model_path, _validate_region

    try:
        _validate_region(region)
    except ValueError:
        return False

    # Primary path: Redis forecast with ensemble_weights → real trained models
    try:
        from data.redis_client import redis_get, redis_key

        payload = redis_get(redis_key(f"forecast:{region}:1h"))
        if isinstance(payload, dict):
            weights = payload.get("ensemble_weights")
            if isinstance(weights, dict) and weights:
                return True
            # Forecast exists but no ensemble weights — could mean only one
            # model loaded, OR weights were dropped. Treat as "live but
            # not full ensemble" — still trained.
            forecasts = payload.get("forecasts") or []
            if forecasts:
                first_row = forecasts[0]
                # Any per-model column populated (xgboost / prophet / arima)
                # indicates a real trained-model prediction in flight.
                model_cols = {"xgboost", "prophet", "arima"}
                if any(isinstance(first_row.get(k), (int, float)) for k in model_cols):
                    return True
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("is_trained_redis_check_failed", region=region, error=str(exc))

    # Fallback: legacy local-pickle check for dev environments without Redis
    filepath = _safe_model_path(MODEL_DIR, region)
    return os.path.exists(filepath)


# ── Forecast quality gate (V3.ζ follow-up) ───────────────────────


# Module-level cache so we don't read 51 meta.json blobs on every page
# render. Keyed by region; refreshed every ``_QUALITY_GATE_TTL`` seconds.
# Daily training cadence makes 10 min plenty fresh for production. Only the
# dev/offline inline path (:func:`get_best_holdout_mape`) uses this now — the
# production path reads the scoring-job-published verdict instead (#271).
_QUALITY_GATE_TTL = 600
_quality_gate_cache: dict[str, tuple[float | None, float]] = {}

# Web-tier cache for the scoring-job-published gate verdict map
# (``gridpulse:meta:gate_status``). One Redis read serves a whole 51-BA
# dropdown / US-Grid sweep instead of 51. (#271 / P2-10)
_GATE_STATUS_TTL = 60.0
_gate_status_cache: tuple[dict | None, float] | None = None
_gate_unavailable_logged_at = 0.0


def gate_verdict_from_metrics(model_metrics: dict) -> dict:
    """Compute a region's forecast-quality gate verdict from its model metrics.

    Returns ``{"acceptable": bool, "best_mape": float | None}``. ``best_mape`` is
    the champion — ``min`` MAPE across every *served* model (the ensemble + the
    three base models) — or ``None`` when no finite MAPE exists yet.

    ``acceptable`` is True when there is no signal (``best_mape is None`` →
    "warming up", don't hide a not-yet-trained BA) or when the champion is at or
    under the 7-day ``rollback`` grade (≤22% per ``MAPE_BY_HORIZON``). It is
    False only when *no* served model reaches the acceptable grade — the gate
    must not hide a BA we can forecast acceptably with *some* served model.
    Keying off XGBoost-alone (pre-#255) hid SEC (XGBoost 38.6%, ensemble 13.6%);
    keying off the ensemble alone would newly hide SPA (ensemble 22.8%, XGBoost
    21.1%). The champion (``min``, ensemble included) avoids both.

    Pure — no I/O — so the scoring job calls it to PUBLISH the per-region verdict
    and the dev/offline path calls it inline. The stateless web tier never
    recomputes; it reads the published verdict (#271 / P2-10).
    """
    mapes = [
        float(m["mape"])
        for m in model_metrics.values()
        if isinstance(m, dict) and m.get("mape") is not None and np.isfinite(m["mape"])
    ]
    best = min(mapes) if mapes else None
    if best is None:
        return {"acceptable": True, "best_mape": None}
    from config import MAPE_BY_HORIZON

    return {"acceptable": best <= MAPE_BY_HORIZON["7d"]["rollback"], "best_mape": best}


def get_best_holdout_mape(region: str) -> float | None:
    """Best achievable holdout MAPE for ``region`` — the champion across every
    served model (ensemble + 3 bases), or ``None`` when no real metric exists.

    Reads via :func:`get_model_metrics` (Redis-first, GCS-meta fallback), so this
    is the **dev/offline** gate path only — reachable from
    :func:`is_forecast_quality_acceptable` when ``REQUIRE_REDIS`` is False. In
    production the gate reads the scoring-job-published verdict and never calls
    this (which is what kept a cold-Redis web tier sweeping GCS metas per render,
    P2-10 / #271). Also used directly for metric display + by the gate tests.
    """
    import time

    cached = _quality_gate_cache.get(region)
    if cached is not None and time.time() - cached[1] < _QUALITY_GATE_TTL:
        return cached[0]

    try:
        metrics = get_model_metrics(region) or {}
    except Exception:
        # Operational outage (Redis/GCS) → no signal, don't black out the
        # dropdown. Matches the pre-#255 fail-open contract.
        metrics = {}
    best = gate_verdict_from_metrics(metrics)["best_mape"]
    _quality_gate_cache[region] = (best, time.time())
    return best


def _get_gate_status() -> dict | None:
    """Read the scoring-job-published gate verdict map from
    ``gridpulse:meta:gate_status`` (``{region: {"acceptable", "best_mape"}}``),
    briefly cached in-process so a 51-BA sweep does one Redis read, not 51.

    Returns ``None`` when no verdict is published (cold/flushed Redis) or on a
    read error. Redis-only — never touches GCS, so it can't drag the request
    path into per-render meta reads. (#271 / P2-10)
    """
    global _gate_status_cache
    import time

    now = time.time()
    if _gate_status_cache is not None and now - _gate_status_cache[1] < _GATE_STATUS_TTL:
        return _gate_status_cache[0]

    regions: dict | None = None
    try:
        from data.redis_client import redis_get, redis_key

        payload = redis_get(redis_key("meta:gate_status"))
        # Require a NON-EMPTY regions map. An empty {} — a degraded scoring run
        # that produced no verdicts, or a corrupted write — must be treated as
        # "no verdict" (return None -> the caller's pass-open+log path), NOT read
        # as "every region warming -> visible", which would silently un-hide
        # rollback-grade BAs during an outage. (#271, defense-in-depth alongside
        # the scoring-side merge-guard.)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("regions"), dict)
            and payload["regions"]
        ):
            regions = payload["regions"]
    except Exception as e:  # pragma: no cover — defensive (Redis outage)
        log.debug("gate_status_read_failed", error=str(e))
        regions = None
    _gate_status_cache = (regions, now)
    return regions


def _log_gate_unavailable_once() -> None:
    """Warn (rate-limited to once per ``_GATE_STATUS_TTL``) that the published
    gate verdict is missing in production, so the pass-open fallback is never
    *silent* — the fail-open-on-outage half of the P2-10 bug. Health/freshness
    reflect the same outage via the missing-scoring (``last_scored``) signal."""
    global _gate_unavailable_logged_at
    import time

    now = time.time()
    if now - _gate_unavailable_logged_at >= _GATE_STATUS_TTL:
        _gate_unavailable_logged_at = now
        log.warning("forecast_gate_status_unavailable_pass_open")


def is_forecast_quality_acceptable(region: str) -> bool:
    """Return True when ``region`` may be shown in the dropdown / US-Grid — i.e.
    at least one served model forecasts it in the ``acceptable`` grade or better
    (7-day horizon), or no MAPE signal exists yet (warming — don't hide a
    not-yet-trained BA). False only when *no* served model clears the rollback
    grade.

    Resolution (#271 / P2-10):

    * **Production** reads the scoring-job-published verdict map
      (``gridpulse:meta:gate_status``, Redis-only). A region present in the map
      uses its published ``acceptable``; a region absent (not yet scored/trained)
      passes (warming). When the whole map is missing — cold/flushed Redis, where
      the app is already rendering its warming state — the gate passes but LOGS
      (``_log_gate_unavailable_once``) so the fallback isn't silent, and does
      **not** read GCS. This replaces the old path that fataled open on any
      Redis/GCS exception AND swept per-render GCS metas on cold Redis.
    * **Dev/offline** (``REQUIRE_REDIS`` False, no scoring job publishing) falls
      back to computing the verdict inline from local metrics.

    The ``forecast_quality_gate`` feature flag short-circuits to True when False.
    """
    from config import REQUIRE_REDIS, feature_enabled

    if not feature_enabled("forecast_quality_gate"):
        return True

    status = _get_gate_status()
    if status is not None:
        verdict = status.get(region)
        if not isinstance(verdict, dict):
            # Region absent (not yet scored/trained) or a malformed entry.
            # Warming: don't hide a BA whose first training run hasn't landed,
            # and never let a corrupt map entry crash the dropdown render.
            return True
        return bool(verdict.get("acceptable", True))

    # No published verdict map.
    if not REQUIRE_REDIS:
        # Dev/offline: nothing publishes gate_status, so compute inline from
        # local metrics (the pre-#271 path). Safe — dev reads local pickles, not
        # the request-path GCS the stateless web tier must avoid.
        mape = get_best_holdout_mape(region)
        if mape is None:
            return True
        from config import MAPE_BY_HORIZON

        return mape <= MAPE_BY_HORIZON["7d"]["rollback"]

    # Production, no verdict at all (cold/flushed Redis). Pass but not silently.
    _log_gate_unavailable_once()
    return True


def stable_visible_regions(all_regions) -> list[str]:
    """Return the subset of ``all_regions`` that pass the quality gate,
    preserving input order. Convenience for the dropdown + US Grid
    consumers that iterate over the full ``REGION_NAMES`` keyspace."""
    return [r for r in all_regions if is_forecast_quality_acceptable(r)]


def hidden_regions(all_regions) -> list[str]:
    """Return the regions in ``all_regions`` that fail the quality gate.

    Used by the UI to render a small "*N BAs hidden — forecast quality
    below threshold*" note with a hover-list of affected codes.
    """
    return [r for r in all_regions if not is_forecast_quality_acceptable(r)]


def _reset_quality_gate_cache() -> None:
    """Test hook — clear the gate caches so unit tests can vary the underlying
    metrics / published verdict without waiting out the TTLs."""
    global _gate_status_cache, _gate_unavailable_logged_at
    _quality_gate_cache.clear()
    _gate_status_cache = None
    _gate_unavailable_logged_at = 0.0


# ── Private helpers ──────────────────────────────────────────────


def _load_cached_models(region: str) -> dict[str, Any] | None:
    """Load models from disk, caching in memory."""
    if region in _model_cache:
        return _model_cache[region]

    try:
        from models.training import load_models

        model_data = load_models(region)
        _model_cache[region] = model_data
        log.info("model_cache_loaded", region=region)
        return model_data
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning("model_load_failed", region=region, error=str(e))
        return None


def _predict_from_trained(
    model_data: dict[str, Any],
    demand_df: pd.DataFrame,
    models_shown: list[str] | None,
) -> dict[str, Any]:
    """Generate predictions from trained model objects.

    Runs feature engineering on ``demand_df`` before passing it to model
    predictors so that XGBoost receives its expected feature columns and
    Prophet receives time-based regressors.  Without this gate the raw
    DataFrame would cause XGBoost to silently zero-fill missing features
    and Prophet to receive no weather regressors.
    """
    actual = demand_df["demand_mw"].values
    n = len(actual)
    result: dict[str, Any] = {"source": "trained"}
    result["metrics"] = model_data.get("metrics", {})
    result["weights"] = model_data.get("ensemble_weights", {})

    # Feature-engineer the input so models receive the columns they were
    # trained on.  Only demand data is available here (no separate weather
    # DataFrame), so weather-derived features will be zero/NaN — but
    # time-based and demand-derived features (lags, rolling stats, hour,
    # dow, CDD/HDD from any inline temperature column) will be correct.
    try:
        from data.feature_engineering import engineer_features

        featured_df = engineer_features(demand_df)
        featured_df = featured_df.dropna(subset=["demand_mw"])
        if len(featured_df) >= n:
            predict_df = featured_df
        else:
            log.warning("model_service_feature_eng_short", original=n, featured=len(featured_df))
            predict_df = demand_df
    except Exception as e:
        log.warning("model_service_feature_eng_failed", error=str(e))
        predict_df = demand_df

    all_preds = {}

    for name in ["prophet", "arima", "xgboost"]:
        if models_shown and name not in models_shown:
            continue
        try:
            model_key = f"{name}_model"
            if model_key in model_data:
                if name == "prophet":
                    from models.prophet_model import predict_prophet

                    pred_result = predict_prophet(model_data[model_key], predict_df, periods=n)
                    all_preds[name] = pred_result["forecast"][:n]
                elif name == "arima":
                    from models.arima_model import predict_arima

                    all_preds[name] = predict_arima(model_data[model_key], predict_df, periods=n)[
                        :n
                    ]
                elif name == "xgboost":
                    from models.xgboost_model import predict_xgboost

                    xgb_model = {
                        "model": model_data[model_key],
                        "feature_names": model_data.get("xgboost_feature_names", []),
                    }
                    all_preds[name] = predict_xgboost(xgb_model, predict_df)[:n]
        except Exception as e:
            log.warning("model_predict_failed", model=name, error=str(e))
            # Fall back to simulated for this model
            seed = stable_int_seed(("model_fallback", name))
            rng = np.random.RandomState(seed)
            noise_scale = {"prophet": 0.025, "arima": 0.035, "xgboost": 0.018}.get(name, 0.025)
            all_preds[name] = actual * (1 + rng.normal(0, noise_scale, n))

    # No base model produced a prediction — e.g. ``models_shown=["ensemble"]``
    # alone (the loop skips every base model) or all predicts failed. An
    # ensemble has nothing to combine, so there is no honest forecast to
    # return. Do NOT echo actuals as a "trained" forecast (a max|residual|=0
    # fabricated-perfect series); signal unavailable (2026-07 review P2-31).
    if not all_preds:
        return {"source": "unavailable", "metrics": {}, "weights": {}}

    # Ensemble — the single combine path (models.ensemble.ensemble_combine, #184).
    # Preserve the prior per-model 1/len default for a base model missing from the
    # weights dict; ensemble_combine then renormalizes over the models present
    # (and equal-weights when the total is zero, matching the old np.mean branch).
    from models.ensemble import ensemble_combine

    weights = result["weights"]
    combine_weights = {name: weights.get(name, 1.0 / len(all_preds)) for name in all_preds}
    all_preds["ensemble"] = ensemble_combine(all_preds, combine_weights)

    # Indicative range (±3% heuristic — not a calibrated confidence interval)
    ensemble = all_preds.get("ensemble", actual)
    result.update(all_preds)
    result["upper_80"] = ensemble * 1.03
    result["lower_80"] = ensemble * 0.97

    return result


def _simulate_forecasts(
    region: str,
    actual: np.ndarray,
    models_shown: list[str] | None,
) -> dict[str, Any]:
    """
    Generate deterministic simulated forecasts.

    Uses a seeded RNG so the same region + data always produces the same
    "model" outputs. This is consistent across page loads (no random flicker).
    """
    n = len(actual)
    seed = stable_int_seed(("simulate_forecasts", region))
    rng = np.random.RandomState(seed)

    # Model-specific noise levels (realistic relative accuracy)
    noise = {
        "prophet": rng.normal(0, 0.025, n),
        "arima": rng.normal(0, 0.035, n),
        "xgboost": rng.normal(0, 0.018, n),
    }

    preds = {}
    for name, n_arr in noise.items():
        preds[name] = actual * (1 + n_arr)

    # Ensemble = weighted average (XGBoost-heavy, ARIMA-light) via the single
    # combine path (#184). The weights already sum to 1, so ensemble_combine's
    # renormalization is a no-op — byte-identical to the old explicit sum().
    from models.ensemble import ensemble_combine

    weights = {"prophet": 0.30, "arima": 0.20, "xgboost": 0.50}
    ensemble = ensemble_combine(preds, weights)
    preds["ensemble"] = ensemble

    # Indicative range (heuristic — not a calibrated confidence interval)
    preds["upper_80"] = ensemble * 1.03
    preds["lower_80"] = ensemble * 0.97

    # Simulated metrics
    from models.evaluation import compute_all_metrics

    metrics = {}
    for name in ["prophet", "arima", "xgboost", "ensemble"]:
        if name in preds:
            m = compute_all_metrics(
                actual[-720:] if len(actual) > 720 else actual,
                preds[name][-720:] if len(preds[name]) > 720 else preds[name],
            )
            metrics[name] = m

    preds["weights"] = weights
    preds["metrics"] = metrics
    preds["source"] = "simulated"

    return preds
