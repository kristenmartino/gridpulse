"""
GridPulse training job — daily.

For each region:

1. Fetch fresh demand + weather data.
2. Engineer features.
3. Train XGBoost (primary), Prophet, and SARIMAX (best-effort).
4. Persist each model to GCS via :mod:`models.persistence`.
5. Recompute backtests and write them to Redis.
6. Mark ``wattcast:meta:last_trained``.

Does not trigger scoring — the next hourly scoring run picks up the new
models via :func:`models.persistence.load_model`.
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np
import structlog

from config import PRECOMPUTE_DEFAULT_REGION
from jobs import phases
from models.persistence import save_model

log = structlog.get_logger()


def _compute_data_hash(region_data: phases.RegionData) -> str:
    """Stable hash used to detect identical training inputs across runs."""
    from components.callbacks import _compute_data_hash

    return _compute_data_hash(region_data.demand_df, region_data.weather_df, region_data.region)


_HOLDOUT_HOURS = 168
_MIN_TRAIN_HOURS = 720  # need at least 30 days of training before a 7-day holdout


def _holdout_metrics_xgboost(featured_df, region: str) -> dict | None:
    """Train a holdout XGBoost on ``featured_df[:-_HOLDOUT_HOURS]`` and score
    it on the final 168-hour window. Returns ``{metrics: {mape, rmse, mae, r2},
    forecast: ndarray, actual: ndarray}`` or ``None`` if the window is too
    short / the holdout fit fails.

    The forecast / actual arrays are kept on the return so the ensemble metric
    can be computed from the SAME holdout predictions later, no recomputation.
    """
    import numpy as np

    from models.evaluation import compute_all_metrics
    from models.xgboost_model import predict_xgboost, train_xgboost

    if len(featured_df) <= _HOLDOUT_HOURS + _MIN_TRAIN_HOURS:
        return None
    train_df = featured_df.iloc[:-_HOLDOUT_HOURS]
    val_df = featured_df.iloc[-_HOLDOUT_HOURS:]
    try:
        holdout_model = train_xgboost(train_df, n_splits=3)
        forecast = np.asarray(predict_xgboost(holdout_model, val_df), dtype=float)[: len(val_df)]
        y_val = np.asarray(val_df["demand_mw"].values, dtype=float)
        if not np.isfinite(forecast).all():
            return None
        metrics = compute_all_metrics(y_val, forecast)
        if not np.isfinite(metrics["mape"]) or metrics["mape"] <= 0:
            return None
        log.info(
            "training_xgboost_holdout_metrics",
            region=region,
            mape=round(metrics["mape"], 3),
            rmse=round(metrics["rmse"], 1),
            mae=round(metrics["mae"], 1),
            r2=round(metrics["r2"], 4),
        )
        return {"metrics": metrics, "forecast": forecast, "actual": y_val}
    except Exception as e:
        log.warning("training_xgboost_holdout_failed", region=region, error=str(e))
        return None


def _holdout_metrics_prophet(featured_df, region: str) -> dict | None:
    """Compute Prophet's full holdout metric set on the last 168 hours.

    Replaces the earlier ``_holdout_mape_prophet`` which only returned MAPE.
    Returns ``{metrics: {mape, rmse, mae, r2}, forecast: ndarray, actual: ndarray}``
    so the ensemble can be computed from the same predictions.
    """
    import numpy as np

    from models.evaluation import compute_all_metrics
    from models.prophet_model import predict_prophet, train_prophet

    if len(featured_df) <= _HOLDOUT_HOURS + _MIN_TRAIN_HOURS:
        return None
    train_df = featured_df.iloc[:-_HOLDOUT_HOURS]
    val_df = featured_df.iloc[-_HOLDOUT_HOURS:]
    try:
        holdout_model = train_prophet(train_df)
        pred = predict_prophet(holdout_model, val_df, periods=len(val_df))
        forecast = np.asarray(pred["forecast"], dtype=float)[: len(val_df)]
        y_val = np.asarray(val_df["demand_mw"].values, dtype=float)
        if not np.isfinite(forecast).all():
            return None
        metrics = compute_all_metrics(y_val, forecast)
        if not np.isfinite(metrics["mape"]) or metrics["mape"] <= 0:
            return None
        log.info(
            "training_prophet_holdout_metrics",
            region=region,
            mape=round(metrics["mape"], 3),
            rmse=round(metrics["rmse"], 1),
            mae=round(metrics["mae"], 1),
            r2=round(metrics["r2"], 4),
        )
        return {"metrics": metrics, "forecast": forecast, "actual": y_val}
    except Exception as e:
        log.warning("training_prophet_holdout_failed", region=region, error=str(e))
        return None


def _holdout_metrics_arima(featured_df, region: str) -> dict | None:
    """Compute SARIMAX's full holdout metric set on the last 168 hours.

    Replaces the earlier ``_holdout_mape_arima`` (MAPE only). See
    :func:`_holdout_metrics_prophet` for return shape.
    """
    import numpy as np

    from models.arima_model import predict_arima, train_arima
    from models.evaluation import compute_all_metrics

    if len(featured_df) <= _HOLDOUT_HOURS + _MIN_TRAIN_HOURS:
        return None
    train_df = featured_df.iloc[:-_HOLDOUT_HOURS]
    val_df = featured_df.iloc[-_HOLDOUT_HOURS:]
    try:
        holdout_model = train_arima(train_df)
        forecast = np.asarray(
            predict_arima(holdout_model, val_df, periods=len(val_df)),
            dtype=float,
        )[: len(val_df)]
        y_val = np.asarray(val_df["demand_mw"].values, dtype=float)
        if not np.isfinite(forecast).all():
            return None
        metrics = compute_all_metrics(y_val, forecast)
        if not np.isfinite(metrics["mape"]) or metrics["mape"] <= 0:
            return None
        log.info(
            "training_arima_holdout_metrics",
            region=region,
            mape=round(metrics["mape"], 3),
            rmse=round(metrics["rmse"], 1),
            mae=round(metrics["mae"], 1),
            r2=round(metrics["r2"], 4),
        )
        return {"metrics": metrics, "forecast": forecast, "actual": y_val}
    except Exception as e:
        log.warning("training_arima_holdout_failed", region=region, error=str(e))
        return None


def _ensemble_holdout_metrics(
    per_model_holdouts: dict[str, dict],
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Combine per-model holdout predictions into an ensemble forecast,
    score it against the shared holdout actuals, and return
    ``(metrics, weights)``. Returns ``None`` when fewer than two models
    produced finite holdout predictions (the ensemble degenerates to a
    single model and adds nothing).

    Weights are inverse-MAPE — same formula the scoring job uses to
    combine the FORWARD-LOOKING forecasts, applied here to the
    holdout window so the persisted ensemble metric reflects the
    same combination rule that runs in production.
    """
    import numpy as np

    from models.ensemble import compute_ensemble_weights, ensemble_combine
    from models.evaluation import compute_all_metrics

    valid = {name: payload for name, payload in per_model_holdouts.items() if payload is not None}
    if len(valid) < 2:
        return None

    # Every per-model holdout uses the same train/val split so actuals
    # are identical across models — assert this defensively.
    actuals = next(iter(valid.values()))["actual"]
    for name, payload in valid.items():
        if len(payload["actual"]) != len(actuals):
            log.warning(
                "ensemble_holdout_actuals_mismatch",
                model=name,
                expected=len(actuals),
                got=len(payload["actual"]),
            )
            return None

    mape_scores = {name: payload["metrics"]["mape"] for name, payload in valid.items()}
    weights = compute_ensemble_weights(mape_scores)
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    forecasts = {name: payload["forecast"] for name, payload in valid.items()}
    ensemble_pred = np.asarray(ensemble_combine(forecasts, weights), dtype=float)

    metrics = compute_all_metrics(actuals, ensemble_pred)
    if not np.isfinite(metrics["mape"]) or metrics["mape"] <= 0:
        return None
    return metrics, weights


def _train_xgboost(
    region_data: phases.RegionData,
    holdout: dict | None = None,
) -> str | None:
    """Train + persist an XGBoost model. Returns the saved version or ``None``.

    When ``holdout`` is supplied (the dict from
    :func:`_holdout_metrics_xgboost`), its full metric set is persisted in
    the meta's ``extra["holdout_metrics"]`` so ``get_model_metrics`` can
    surface real RMSE / MAE / R² (not just MAPE) on the Models tab.
    """
    from models.xgboost_model import train_xgboost

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        model_dict = train_xgboost(region_data.featured_df, n_splits=3)
    except Exception as e:
        log.warning("training_xgboost_failed", region=region, error=str(e))
        return None

    # Holdout MAPE wins over CV-mean MAPE when available — same window
    # as the prophet/arima holdouts so the inverse-MAPE ensemble weights
    # are computed against a consistent metric basis.
    cv_scores = model_dict.get("cv_scores") if isinstance(model_dict, dict) else None
    cv_mape = None
    if cv_scores is not None and len(cv_scores) > 0:
        try:
            cv_mape = float(np.mean(cv_scores))
        except Exception:
            cv_mape = None

    holdout_metrics = holdout["metrics"] if holdout else None
    saved_mape = (holdout_metrics or {}).get("mape") or cv_mape

    extra: dict = {"cv_scores": cv_scores if cv_scores is not None else []}
    if holdout_metrics is not None:
        extra["holdout_metrics"] = holdout_metrics

    return save_model(
        region=region,
        model_name="xgboost",
        model_obj=model_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=saved_mape,
        extra=extra,
    )


def _train_prophet(
    region_data: phases.RegionData,
    holdout: dict | None = None,
) -> str | None:
    """Best-effort Prophet training. Returns the saved version or ``None``.

    Trains twice: once on the train portion (to score the full holdout
    metric set used for ensemble weighting + Models-tab display), and
    once on the full window for the production model that gets persisted
    to GCS. ``holdout`` is the payload from :func:`_holdout_metrics_prophet`
    — stashed here as ``extra["holdout_metrics"]`` so RMSE / MAE / R² are
    real at display time.
    """
    from models.prophet_model import train_prophet

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        prophet_obj = train_prophet(region_data.featured_df)
    except Exception as e:
        log.warning("training_prophet_failed", region=region, error=str(e))
        return None

    holdout_metrics = holdout["metrics"] if holdout else None
    extra: dict = {}
    if holdout_metrics is not None:
        extra["holdout_metrics"] = holdout_metrics

    return save_model(
        region=region,
        model_name="prophet",
        model_obj=prophet_obj,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=(holdout_metrics or {}).get("mape"),
        extra=extra,
    )


def _train_arima(
    region_data: phases.RegionData,
    holdout: dict | None = None,
) -> str | None:
    """Best-effort SARIMAX training. Returns the saved version or ``None``.

    Holdout-metric extraction mirrors :func:`_train_prophet` — full
    {mape, rmse, mae, r2} dict persisted in ``extra["holdout_metrics"]``.
    """
    from models.arima_model import train_arima

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        arima_dict = train_arima(region_data.featured_df)
    except Exception as e:
        log.warning("training_arima_failed", region=region, error=str(e))
        return None

    holdout_metrics = holdout["metrics"] if holdout else None
    extra: dict = {}
    if holdout_metrics is not None:
        extra["holdout_metrics"] = holdout_metrics

    return save_model(
        region=region,
        model_name="arima",
        model_obj=arima_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=(holdout_metrics or {}).get("mape"),
        extra=extra,
    )


def _train_region(region: str) -> dict:
    """Run all training phases for a single region. Returns a summary dict."""
    t0 = time.time()
    summary: dict = {"region": region, "ok": False, "models": {}, "backtests": {}}

    region_data = phases.fetch_region_data(region)
    if region_data is None:
        summary["error"] = "no_data"
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary

    if phases.engineer_region_features(region_data) is None:
        summary["error"] = "feature_engineering_failed"
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary

    # Score every model on the SAME 168-hour holdout window before
    # persisting production models. This is the only time predictions
    # against real ground truth are produced for the Models tab — the
    # scoring job's diagnostics path used to do this against
    # ``_simulate_forecasts`` and write fake RMSE / MAE / R² to Redis;
    # now those numbers come from here, persisted into each meta's
    # ``extra["holdout_metrics"]``. The ensemble metric is computed
    # off the same predictions and stashed in xgboost's meta extra.
    featured = region_data.featured_df
    xgb_holdout = _holdout_metrics_xgboost(featured, region)
    prophet_holdout = _holdout_metrics_prophet(featured, region)
    arima_holdout = _holdout_metrics_arima(featured, region)

    # Compute ensemble metric BEFORE persisting xgboost so we can
    # stash it in xgboost's extra. xgboost is the canonical "primary"
    # model; if other models drop out, the ensemble row degrades to
    # whatever survived.
    ensemble_summary = _ensemble_holdout_metrics(
        {
            "xgboost": xgb_holdout,
            "prophet": prophet_holdout,
            "arima": arima_holdout,
        }
    )

    xgb_version = _train_xgboost(region_data, holdout=xgb_holdout)
    summary["models"]["xgboost"] = xgb_version
    prophet_version = _train_prophet(region_data, holdout=prophet_holdout)
    summary["models"]["prophet"] = prophet_version
    arima_version = _train_arima(region_data, holdout=arima_holdout)
    summary["models"]["arima"] = arima_version

    # Persist the ensemble metric into xgboost's meta if both succeeded
    # — keeps the meta-as-display-source contract intact (no separate
    # ensemble blob to introduce here). ``get_model_metrics`` checks
    # this key when building the ensemble row for the Models tab.
    if xgb_version and ensemble_summary is not None:
        ensemble_metrics, ensemble_weights = ensemble_summary
        try:
            from models.persistence import write_extra_to_meta

            write_extra_to_meta(
                region=region,
                model_name="xgboost",
                version=xgb_version,
                key_updates={
                    "ensemble_holdout_metrics": ensemble_metrics,
                    "ensemble_weights": ensemble_weights,
                },
            )
            summary["ensemble"] = {
                "mape": round(ensemble_metrics["mape"], 3),
                "rmse": round(ensemble_metrics["rmse"], 1),
                "weights": ensemble_weights,
            }
        except Exception as e:
            log.warning(
                "ensemble_extra_persist_failed",
                region=region,
                error=str(e),
            )

    # Recompute backtests against the freshly fetched data — these become
    # the values the Validation tab reads out of Redis.
    bt_res = phases.write_backtests(region_data)
    summary["backtests"] = {
        "ok": bt_res.ok,
        **(bt_res.details if bt_res.ok else {"error": bt_res.error}),
    }

    gc.collect()
    summary["ok"] = xgb_version is not None
    summary["elapsed_s"] = round(time.time() - t0, 2)
    return summary


def run() -> int:
    """Run the training job end-to-end. Returns an exit code."""
    t0 = time.time()
    regions = phases.ordered_regions(PRECOMPUTE_DEFAULT_REGION)
    log.info("training_job_start", regions=regions)

    # Training is memory-intensive (SARIMAX auto-order especially). Run
    # regions sequentially to keep peak RSS bounded; the daily cadence
    # tolerates the longer wall-clock.
    results: list[dict] = []
    for region in regions:
        try:
            results.append(_train_region(region))
        except Exception as e:
            log.warning(
                "training_job_region_crashed",
                region=region,
                error=str(e),
            )
            results.append({"region": region, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    fail_regions = [r["region"] for r in results if not r.get("ok")]

    phases.write_meta(
        "last_trained",
        extra={
            "regions_trained": ok_count,
            "regions_failed": fail_regions,
            "mode": "training-job",
        },
    )

    elapsed = round(time.time() - t0, 2)
    log.info(
        "training_job_complete",
        ok_count=ok_count,
        fail_count=len(fail_regions),
        elapsed_s=elapsed,
        failed_regions=fail_regions,
    )
    sys.stdout.flush()

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
