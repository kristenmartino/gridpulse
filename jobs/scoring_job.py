"""
GridPulse scoring job — hourly.

Refreshes Redis with the data the Dash web service reads:

- actuals (``gridpulse:actuals:{region}``)
- weather (``gridpulse:weather:{region}``)
- generation by fuel (``gridpulse:generation:{region}``)
- 720h forward forecast (``gridpulse:forecast:{region}:1h``)
- weather-correlation payload (``gridpulse:weather-correlation:{region}``)
- model diagnostics (``gridpulse:diagnostics:{region}``)
- alerts / stress / anomalies (``gridpulse:alerts:{region}``)
- ``gridpulse:meta:last_scored`` marker

Per-region failures are isolated — one region going sideways must not
abort the whole run. The job returns a non-zero exit code only when
nothing at all succeeded, so Cloud Run Jobs surface hard failures while
tolerating transient partial outages.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from config import PRECOMPUTE_DEFAULT_REGION, PRECOMPUTE_MAX_WORKERS
from jobs import phases
from models.persistence import ModelMetadata, load_model

log = structlog.get_logger()


_HOLDOUT_METRIC_FIELDS = ("mape", "rmse", "mae", "r2")


def _extract_holdout_metrics(meta: ModelMetadata | None) -> dict[str, float]:
    """Pull the per-model holdout metrics dict from a ``ModelMetadata``.

    Resolution order (matches what ``models.model_service.get_model_metrics``
    expects to find in Redis):

    1. ``meta.extra["holdout_metrics"]`` — full {mape, rmse, mae, r2}
       dict from the training job's evaluation pass.
    2. Top-level ``meta.mape`` — legacy fallback for pickles trained
       before the holdout_metrics block was added.

    Returns an empty dict when no useful metrics are present.
    #131 (2026-05-20) added this so the scoring job can write real
    holdout metrics into the Redis forecast payload — without it the
    web tier falls all the way through to the simulated baseline.
    """
    if meta is None:
        return {}
    extra = getattr(meta, "extra", None) or {}

    out: dict[str, float] = {}
    holdout = extra.get("holdout_metrics") if isinstance(extra, dict) else None
    if isinstance(holdout, dict):
        for field in _HOLDOUT_METRIC_FIELDS:
            val = holdout.get(field)
            if val is None:
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if f == f and f not in (float("inf"), float("-inf")):  # finite check
                out[field] = f

    if "mape" not in out:
        top_level_mape = getattr(meta, "mape", None)
        if top_level_mape is not None:
            try:
                f = float(top_level_mape)
                if f > 0 and f == f:
                    out["mape"] = f
            except (TypeError, ValueError):
                pass

    return out


def _extract_ensemble_metrics(xgb_meta: ModelMetadata | None) -> dict[str, float]:
    """Pull ensemble metrics from xgboost's meta extras.

    The training job writes the ensemble holdout under
    ``xgb_meta.extra["ensemble_holdout_metrics"]`` — same convention
    ``get_model_metrics`` already reads from. Returns an empty dict
    when the field is absent (e.g. legacy pickle without the
    ensemble row).
    """
    if xgb_meta is None:
        return {}
    extra = getattr(xgb_meta, "extra", None) or {}
    ens = extra.get("ensemble_holdout_metrics") if isinstance(extra, dict) else None
    if not isinstance(ens, dict):
        return {}

    out: dict[str, float] = {}
    for field in _HOLDOUT_METRIC_FIELDS:
        val = ens.get(field)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f == f and f not in (float("inf"), float("-inf")):
            out[field] = f
    return out


def _score_region(region: str) -> dict:
    """Run all scoring phases for a single region. Returns a summary dict."""
    t0 = time.time()
    summary: dict = {"region": region, "ok": False, "phases": {}}

    region_data = phases.fetch_region_data(region)
    if region_data is None:
        summary["phases"]["fetch"] = {"ok": False, "error": "no_data"}
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary
    summary["phases"]["fetch"] = {
        "ok": True,
        "demand_rows": len(region_data.demand_df),
        "weather_rows": len(region_data.weather_df),
    }

    # #121 part 1: snapshot the about-to-be-overwritten forecast key
    # BEFORE write_actuals_and_weather + predict_and_write_forecast run.
    # The drift phase later in this function compares this previous
    # forecast's 1-hour-ahead prediction against the now-known actuals.
    # Read failure → drift phase becomes a no-op for this tick.
    previous_forecast = phases.read_existing_forecast(region)

    actuals_res = phases.write_actuals_and_weather(region_data)
    summary["phases"]["actuals_weather"] = {
        "ok": actuals_res.ok,
        **(actuals_res.details if actuals_res.ok else {"error": actuals_res.error}),
    }

    # #121 part 1: continuous drift signal. Runs after actuals are
    # written + before predict_and_write_forecast overwrites the
    # forecast key. Failures are isolated — a drift-side error never
    # blocks the broader scoring run because drift is a secondary
    # signal, not a critical path.
    drift_res = phases.write_drift_metrics(region, previous_forecast, region_data.demand_df)
    summary["phases"]["drift"] = {
        "ok": drift_res.ok,
        **(drift_res.details if drift_res.ok else {"error": drift_res.error}),
    }

    # #227: horizon-matched drift (24h/48h/72h) — snapshots the same
    # about-to-be-overwritten forecast + resolves matured snapshots, graded
    # against each horizon's own band. Isolated like the 1h phase above.
    horizon_drift_res = phases.write_horizon_drift_metrics(
        region, previous_forecast, region_data.demand_df
    )
    summary["phases"]["drift_horizon"] = {
        "ok": horizon_drift_res.ok,
        **(
            horizon_drift_res.details
            if horizon_drift_res.ok
            else {"error": horizon_drift_res.error}
        ),
    }

    gen_res = phases.write_generation(region)
    summary["phases"]["generation"] = {
        "ok": gen_res.ok,
        **(gen_res.details if gen_res.ok else {"error": gen_res.error}),
    }

    # V3.α: BA-to-BA interchange snapshot. Independent of generation —
    # a sparse interchange fetch shouldn't fail the broader scoring run,
    # so we ignore the PhaseResult's ok flag for the summary aggregate.
    ix_res = phases.write_interchange(region)
    summary["phases"]["interchange"] = {
        "ok": ix_res.ok,
        **(ix_res.details if ix_res.ok else {"error": ix_res.error}),
    }

    # Feature engineering + model load are only needed for the forecast
    # and diagnostics phases. If the model is missing (first deploy before
    # training job has run) we still emit actuals + weather + generation.
    #
    # Stage 1 of scoring-job-multi-model: load Prophet alongside XGBoost
    # and pass both to the forecast phase as a dict so every available
    # model gets a Redis row. Diagnostics still uses XGBoost only —
    # training-quality evaluation lives in the daily training job.
    xgb_loaded = load_model(region, "xgboost")
    prophet_loaded = load_model(region, "prophet")
    arima_loaded = load_model(region, "arima")

    loaded_models: dict[str, object] = {}
    # Stage 3: per-model MAPE harvested from each pickle's metadata, used
    # to weight the ensemble inversely (lower MAPE → higher weight).
    # None values fall back to equal weights inside compute_ensemble_weights.
    model_mapes: dict[str, float | None] = {}
    # #131: full per-model holdout metrics (MAPE / RMSE / MAE / R²)
    # harvested from each meta's ``extra["holdout_metrics"]`` block so
    # they can be persisted to Redis for the web tier. Ensemble row
    # rides on xgb_meta's ``extra["ensemble_holdout_metrics"]`` (existing
    # convention used by models.model_service.get_model_metrics).
    model_metrics: dict[str, dict[str, float]] = {}
    if xgb_loaded is not None:
        xgb_model, xgb_meta = xgb_loaded
        loaded_models["xgboost"] = xgb_model
        model_mapes["xgboost"] = xgb_meta.mape
        summary["model_version"] = xgb_meta.version
        xgb_metrics = _extract_holdout_metrics(xgb_meta)
        if xgb_metrics:
            model_metrics["xgboost"] = xgb_metrics
        ensemble_metrics = _extract_ensemble_metrics(xgb_meta)
        if ensemble_metrics:
            model_metrics["ensemble"] = ensemble_metrics
    if prophet_loaded is not None:
        prophet_model, prophet_meta = prophet_loaded
        loaded_models["prophet"] = prophet_model
        model_mapes["prophet"] = prophet_meta.mape
        summary["prophet_version"] = prophet_meta.version
        prophet_metrics = _extract_holdout_metrics(prophet_meta)
        if prophet_metrics:
            model_metrics["prophet"] = prophet_metrics
    if arima_loaded is not None:
        arima_model, arima_meta = arima_loaded
        loaded_models["arima"] = arima_model
        model_mapes["arima"] = arima_meta.mape
        summary["arima_version"] = arima_meta.version
        arima_metrics = _extract_holdout_metrics(arima_meta)
        if arima_metrics:
            model_metrics["arima"] = arima_metrics

    has_features = phases.engineer_region_features(region_data) is not None

    if has_features and loaded_models:
        fc_res = phases.predict_and_write_forecast(
            region_data, loaded_models, model_mapes, model_metrics=model_metrics
        )
        summary["phases"]["forecast"] = {
            "ok": fc_res.ok,
            **(fc_res.details if fc_res.ok else {"error": fc_res.error}),
        }
        # Diagnostics needs XGBoost specifically (SHAP + per-residual).
        if "xgboost" in loaded_models:
            diag_res = phases.write_diagnostics(region_data, loaded_models["xgboost"])
            summary["phases"]["diagnostics"] = {
                "ok": diag_res.ok,
                **(diag_res.details if diag_res.ok else {"error": diag_res.error}),
            }
        else:
            summary["phases"]["diagnostics"] = {
                "ok": False,
                "error": "no_xgboost_for_diagnostics",
            }
    else:
        log.info(
            "scoring_job_no_model_yet",
            region=region,
            reason="model_missing_or_insufficient_features",
        )
        summary["phases"]["forecast"] = {"ok": False, "error": "no_model"}
        summary["phases"]["diagnostics"] = {"ok": False, "error": "no_model"}

    wc_res = phases.write_weather_correlation(region_data)
    summary["phases"]["weather_correlation"] = {
        "ok": wc_res.ok,
        **(wc_res.details if wc_res.ok else {"error": wc_res.error}),
    }

    alerts_res = phases.write_alerts(region_data)
    summary["phases"]["alerts"] = {
        "ok": alerts_res.ok,
        **(alerts_res.details if alerts_res.ok else {"error": alerts_res.error}),
    }

    summary["ok"] = any(p.get("ok") for p in summary["phases"].values())
    summary["elapsed_s"] = round(time.time() - t0, 2)
    return summary


def run() -> int:
    """Run the scoring job end-to-end. Returns an exit code."""
    t0 = time.time()
    regions = phases.ordered_regions(PRECOMPUTE_DEFAULT_REGION)
    log.info("scoring_job_start", regions=regions)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {pool.submit(_score_region, r): r for r in regions}
        for fut in as_completed(futures):
            region = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                log.warning(
                    "scoring_job_region_crashed",
                    region=region,
                    error=str(e),
                )
                results.append({"region": region, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    fail_regions = [r["region"] for r in results if not r.get("ok")]

    phases.write_meta(
        "last_scored",
        extra={
            "regions_scored": ok_count,
            "regions_failed": fail_regions,
            "mode": "scoring-job",
        },
    )

    elapsed = round(time.time() - t0, 2)
    log.info(
        "scoring_job_complete",
        ok_count=ok_count,
        fail_count=len(fail_regions),
        elapsed_s=elapsed,
        failed_regions=fail_regions,
    )
    sys.stdout.flush()

    # Non-zero exit only when every region failed — Cloud Scheduler retries
    # are valuable for total outages but noisy for partial failures.
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
