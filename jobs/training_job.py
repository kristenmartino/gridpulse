"""
GridPulse training job — daily.

For each region:

1. Fetch fresh demand + weather data.
2. Engineer features.
3. Train XGBoost (primary), Prophet, and SARIMAX (best-effort).
4. Persist each model to GCS via :mod:`models.persistence`.
5. Recompute backtests and write them to Redis.
6. Mark ``gridpulse:meta:last_trained``.

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
# Per-model holdout residuals live longer than an hourly key — training is daily
# and the interval bands tolerate a slightly stale calibration sample.
_HOLDOUT_RESIDUALS_TTL = 172800  # 48h


def _holdout_metrics_xgboost(featured_df, region: str) -> dict | None:
    """Train a holdout XGBoost on ``featured_df[:-_HOLDOUT_HOURS]`` and score
    it on the final 168-hour window. Returns ``{metrics: {mape, rmse, mae, r2},
    forecast: ndarray, actual: ndarray}`` or ``None`` if the window is too
    short / the holdout fit fails.

    Scored RECURSIVELY (#195): each step's autoregressive lags come from the
    model's own prior predictions, not observed in-window actuals — the same
    protocol production serves. Before #195 this used
    ``predict_xgboost(model, val_df)`` directly, which is teacher-forced
    one-step-ahead (val_df already carries real-demand lag features) and is
    NOT commensurable with the Prophet/SARIMAX multi-step holdouts. For one
    release both the recursive (headline) and the old teacher-forced MAPE are
    logged so the shift is observable before it drives weights/gates.

    The forecast / actual arrays are kept on the return so the ensemble metric
    can be computed from the SAME holdout predictions later, no recomputation.
    """
    import numpy as np

    from data.feature_engineering import recursive_autoregressive_forecast
    from models.evaluation import compute_all_metrics
    from models.xgboost_model import predict_xgboost, train_xgboost

    if len(featured_df) <= _HOLDOUT_HOURS + _MIN_TRAIN_HOURS:
        return None
    train_df = featured_df.iloc[:-_HOLDOUT_HOURS]
    val_df = featured_df.iloc[-_HOLDOUT_HOURS:]
    try:
        holdout_model = train_xgboost(train_df, n_splits=3)
        y_val = np.asarray(val_df["demand_mw"].values, dtype=float)
        # Recursive multi-step (the honest, production-matching score).
        forecast = recursive_autoregressive_forecast(
            holdout_model, train_df["demand_mw"].tolist(), val_df, predict_xgboost
        )[: len(val_df)]
        if not np.isfinite(forecast).all():
            return None
        metrics = compute_all_metrics(y_val, forecast)
        if not np.isfinite(metrics["mape"]) or metrics["mape"] <= 0:
            return None

        # One-release observability: the old teacher-forced number alongside
        # the new recursive one, so the accuracy shift is visible in logs
        # before it moves ensemble weights / the visibility gate.
        mape_teacher_forced: float | None = None
        try:
            tf = np.asarray(predict_xgboost(holdout_model, val_df), dtype=float)[: len(val_df)]
            if np.isfinite(tf).all():
                mape_teacher_forced = round(float(compute_all_metrics(y_val, tf)["mape"]), 3)
        except Exception:  # pragma: no cover — comparison-only, never fatal
            mape_teacher_forced = None

        log.info(
            "training_xgboost_holdout_metrics",
            region=region,
            mape=round(metrics["mape"], 3),
            mape_teacher_forced=mape_teacher_forced,
            protocol="recursive",
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


def _read_cached_arima_order(region: str) -> tuple | None:
    """Pull the previously-selected (order, seasonal_order) from the
    existing arima meta's extra. Returns None if no meta exists or the
    keys aren't present (i.e. the BA hasn't been trained on the
    cache-aware code path yet)."""
    try:
        from models.persistence import get_model_metadata

        meta = get_model_metadata(region, "arima")
        if meta is None or not isinstance(getattr(meta, "extra", None), dict):
            return None
        order = meta.extra.get("order")
        seasonal_order = meta.extra.get("seasonal_order")
        if order is None or seasonal_order is None:
            return None
        return tuple(order), tuple(seasonal_order)
    except Exception as e:  # pragma: no cover — defensive
        log.debug("arima_cached_order_lookup_failed", region=region, error=str(e))
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
    cached = _read_cached_arima_order(region)
    cached_order = cached[0] if cached else None
    cached_seasonal_order = cached[1] if cached else None
    try:
        holdout_model = train_arima(
            train_df,
            cached_order=cached_order,
            cached_seasonal_order=cached_seasonal_order,
        )
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


def _persist_holdout_residuals(
    region: str,
    per_model_holdouts: dict[str, dict | None],
    ensemble_summary: tuple[dict[str, float], dict[str, float]] | None,
) -> None:
    """Persist every model's holdout forecast + the shared actuals to
    ``gridpulse:holdout:{region}`` — zero extra compute (reuses the holdouts
    already computed for the meta metrics).

    Enables per-model self-calibrated prediction intervals: the web tier's
    ``_collect_backtest_residuals`` reads this so ensemble/prophet/arima bands
    use their OWN residuals instead of substituting XGBoost (#196). It is also
    the per-model, commensurable (post-#195) holdout data #181 needs to compare
    the ensemble against the best base model. Best-effort — a write failure
    never fails the training run.
    """
    from datetime import UTC, datetime

    import numpy as np

    from data.redis_client import redis_key, redis_set
    from models.ensemble import ensemble_combine

    valid = {n: h for n, h in per_model_holdouts.items() if h is not None}
    if not valid:
        return
    try:
        actuals = np.asarray(next(iter(valid.values()))["actual"], dtype=float)
        predictions: dict[str, list[float]] = {}
        for name, h in valid.items():
            fc = np.asarray(h["forecast"], dtype=float)
            if len(fc) == len(actuals) and np.isfinite(fc).all():
                predictions[name] = fc.tolist()

        # Reconstruct the ensemble holdout forecast from the same per-model
        # forecasts + inverse-MAPE weights the ensemble metric used — no recompute.
        if ensemble_summary is not None:
            _metrics, weights = ensemble_summary
            fc_map = {n: np.asarray(h["forecast"], dtype=float) for n, h in valid.items()}
            fc_map = {n: f for n, f in fc_map.items() if len(f) == len(actuals)}
            if len(fc_map) >= 2:
                ens = np.asarray(ensemble_combine(fc_map, weights), dtype=float)
                if np.isfinite(ens).all():
                    predictions["ensemble"] = ens.tolist()

        if not predictions:
            return
        redis_set(
            redis_key(f"holdout:{region}"),
            {
                "region": region,
                "horizon": _HOLDOUT_HOURS,
                "scored_at": datetime.now(UTC).isoformat(),
                "source": "training_holdout",
                "actual": actuals.tolist(),
                "predictions": predictions,
            },
            ttl=_HOLDOUT_RESIDUALS_TTL,
        )
        log.info("holdout_residuals_persisted", region=region, models=sorted(predictions))
    except Exception as e:  # pragma: no cover — never fail the run over diagnostics
        log.warning("holdout_residuals_persist_failed", region=region, error=str(e))


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

    # #326 serve-path acceptance gate: the holdout above scores a DIFFERENT
    # model on a different frame and is provably blind to the fit lottery.
    # Replay the actual candidate through the real serve path; a rejected
    # candidate is persisted (forensics) but never repoints latest.json.
    gate = phases.serve_path_gate(
        model_dict, region_data.featured_df, region_data.weather_df, region
    )
    extra["serve_gate"] = gate

    return save_model(
        region=region,
        model_name="xgboost",
        model_obj=model_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=saved_mape,
        extra=extra,
        update_latest=bool(gate.get("passed", True)),
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

    The selected ``(order, seasonal_order)`` is stashed in
    ``extra`` so the next training run can skip the pmdarima
    auto_arima stepwise search (the dominant per-BA cost). The
    cache is invalidated automatically if the data changes
    enough to make the order obsolete — at the limit, a force
    retrain bypasses it entirely.
    """
    from models.arima_model import train_arima

    region = region_data.region
    assert region_data.featured_df is not None
    cached = _read_cached_arima_order(region)
    cached_order = cached[0] if cached else None
    cached_seasonal_order = cached[1] if cached else None
    try:
        arima_dict = train_arima(
            region_data.featured_df,
            cached_order=cached_order,
            cached_seasonal_order=cached_seasonal_order,
        )
    except Exception as e:
        log.warning("training_arima_failed", region=region, error=str(e))
        return None

    holdout_metrics = holdout["metrics"] if holdout else None
    extra: dict = {}
    if holdout_metrics is not None:
        extra["holdout_metrics"] = holdout_metrics
    # Cache the order for next run's fast path. We persist whatever
    # train_arima ended up using — whether that came from the cache
    # itself (round-trip preserves it) or from a fresh auto_arima.
    if isinstance(arima_dict, dict):
        if "order" in arima_dict:
            extra["order"] = list(arima_dict["order"])
        if "seasonal_order" in arima_dict:
            extra["seasonal_order"] = list(arima_dict["seasonal_order"])

    return save_model(
        region=region,
        model_name="arima",
        model_obj=arima_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=(holdout_metrics or {}).get("mape"),
        extra=extra,
    )


def _skip_if_data_hash_matches(region: str, current_hash: str) -> dict | None:
    """Return a summary's ``models`` dict if every model is already up
    to date on ``current_hash``, else None.

    "Up to date" means: a saved version exists for each of xgboost /
    prophet / arima AND each meta's ``data_hash`` equals ``current_hash``.
    If even one model is stale or missing, return None and let the
    caller train all three (the inverse-MAPE ensemble weighting needs
    the per-model holdouts to come from the same training pass).
    """
    try:
        from models.persistence import get_model_metadata

        versions: dict[str, str | None] = {}
        metas: dict[str, object] = {}
        for model_name in ("xgboost", "prophet", "arima"):
            try:
                meta = get_model_metadata(region, model_name)
            except Exception:
                meta = None
            if meta is None:
                return None
            if getattr(meta, "data_hash", None) != current_hash:
                return None
            metas[model_name] = meta
            versions[model_name] = meta.version

        # #176 self-heal: the ensemble holdout metric is written post-hoc
        # (write_extra_to_meta) onto the xgboost meta AFTER all three models
        # save. If that write didn't land on a prior run, the xgboost meta is
        # data-hash-current but missing ``ensemble_holdout_metrics`` — and
        # resuming would preserve that gap forever. Decline the resume when an
        # ensemble is achievable (>=2 per-model holdouts present) but its metric
        # is absent, so the full path recomputes and persists it. Single-model
        # regions (no achievable ensemble) still resume.
        xgb_extra = getattr(metas["xgboost"], "extra", None) or {}
        if not xgb_extra.get("ensemble_holdout_metrics"):
            n_holdouts = 0
            for m in metas.values():
                ex = getattr(m, "extra", None)
                if isinstance(ex, dict) and ex.get("holdout_metrics"):
                    n_holdouts += 1
            if n_holdouts >= 2:
                log.info(
                    "training_resume_declined_missing_ensemble",
                    region=region,
                    n_model_holdouts=n_holdouts,
                )
                return None
        return versions
    except Exception as e:  # pragma: no cover — defensive (GCS outage)
        log.debug("training_resume_check_failed", region=region, error=str(e))
        return None


def _train_region(region: str, force: bool = False) -> dict:
    """Run all training phases for a single region. Returns a summary dict.

    When ``force`` is False (the default), the function short-circuits if
    every model already has a saved version whose ``data_hash`` matches
    the data we'd produce now. This is the resume-friendly behavior — a
    Cloud Run retry that restarts from BA 0 doesn't redundantly retrain
    BAs the previous attempt already finished, and the daily cron only
    re-fits BAs whose underlying data actually changed.

    Hash check is cheap (one ``get_model_metadata`` per model = three GCS
    metadata reads, no pickle download). Compared to training cost
    (~12 min/BA on the slow path), the overhead is negligible.
    """
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

    # Resume short-circuit: bail out if every model is already up to
    # date on the current data. Saves the entire training cost when a
    # previous run already finished this region. Counts as success in
    # the per-region summary so ``ok_count`` reflects "regions with a
    # fresh model on disk" — the operationally meaningful number.
    if not force:
        current_hash = _compute_data_hash(region_data)
        skipped = _skip_if_data_hash_matches(region, current_hash)
        if skipped is not None:
            summary["models"] = skipped
            summary["ok"] = True
            summary["resumed"] = True
            summary["elapsed_s"] = round(time.time() - t0, 2)
            log.info(
                "training_region_resumed",
                region=region,
                reason="data_hash_unchanged",
                data_hash=current_hash[:8],
            )
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
    if ensemble_summary is None:
        # Surface WHY there's no ensemble metric to persist (#176) — silence
        # here is how the gap went undiagnosed. Either <2 per-model holdouts
        # succeeded, or the combined metric was non-finite.
        _valid_holdouts = [
            name
            for name, h in (
                ("xgboost", xgb_holdout),
                ("prophet", prophet_holdout),
                ("arima", arima_holdout),
            )
            if h is not None
        ]
        log.warning(
            "ensemble_holdout_unavailable",
            region=region,
            valid_holdouts=_valid_holdouts,
            reason=(
                "fewer_than_two_holdouts"
                if len(_valid_holdouts) < 2
                else "ensemble_metric_nonfinite"
            ),
        )

    # Persist every model's holdout forecast vs the shared actuals for
    # per-model self-calibrated intervals (#196) + #181's comparison data.
    # Zero extra compute — reuses the holdouts just computed.
    _persist_holdout_residuals(
        region,
        {"xgboost": xgb_holdout, "prophet": prophet_holdout, "arima": arima_holdout},
        ensemble_summary,
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

            persisted = write_extra_to_meta(
                region=region,
                model_name="xgboost",
                version=xgb_version,
                key_updates={
                    "ensemble_holdout_metrics": ensemble_metrics,
                    "ensemble_weights": ensemble_weights,
                },
            )
            if persisted:
                log.info(
                    "ensemble_holdout_persisted",
                    region=region,
                    xgb_version=xgb_version,
                    mape=round(ensemble_metrics["mape"], 3),
                )
            else:
                # write_extra_to_meta returns False (no exception) when GCS is
                # disabled or the just-saved meta blob is unexpectedly missing.
                # Surface it — a silent False here is exactly how #176 stayed
                # undiagnosed across every BA.
                log.warning(
                    "ensemble_holdout_persist_returned_false",
                    region=region,
                    xgb_version=xgb_version,
                )
            summary["ensemble"] = {
                "mape": round(ensemble_metrics["mape"], 3),
                "rmse": round(ensemble_metrics["rmse"], 1),
                "weights": ensemble_weights,
                "persisted": bool(persisted),
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


def _partition_regions_for_task(all_regions: list[str]) -> tuple[list[str], int, int]:
    """Partition the full region list across parallel Cloud Run Job tasks.

    Cloud Run Jobs exposes ``CLOUD_RUN_TASK_INDEX`` (0-based) and
    ``CLOUD_RUN_TASK_COUNT`` env vars; the latter equals the job's
    configured ``taskCount`` (set via the deploy workflow). Each task
    runs the same image and entrypoint — partitioning is the job code's
    responsibility.

    Uses **interleaved stride** rather than contiguous slicing because
    ``ordered_regions`` lists the largest BAs first (FPL, ERCOT, CAISO,
    PJM, MISO, ...). A contiguous split would put the four heaviest BAs
    all in task 0; interleaving spreads them across tasks so each task
    sees a similar total cost.

    Returns ``(partition, task_index, task_count)``. When the env vars
    are unset (local dev, ``taskCount=1``) the partition equals the full
    list — no behavior change.
    """
    import os

    task_index = int(os.getenv("CLOUD_RUN_TASK_INDEX", "0"))
    task_count = int(os.getenv("CLOUD_RUN_TASK_COUNT", "1"))
    partition = all_regions[task_index::task_count]
    return partition, task_index, task_count


def run() -> int:
    """Run the training job end-to-end. Returns an exit code.

    Designed to be run by N parallel Cloud Run Job tasks. Each task
    handles its own interleaved slice of the region list; per-region
    work doesn't share state across tasks except for GCS pointer
    files, which are written with optimistic-concurrency
    (see :func:`models.persistence._write_latest`).

    Only task 0 writes the ``gridpulse:meta:last_trained`` Redis pointer
    at the end — the other tasks log their per-task completion but
    don't race the meta write. That key is consumed by the UI's
    data-freshness chip; partial-meta from one task would mislead.
    """
    t0 = time.time()
    all_regions = phases.ordered_regions(PRECOMPUTE_DEFAULT_REGION)
    regions, task_index, task_count = _partition_regions_for_task(all_regions)
    log.info(
        "training_job_start",
        task_index=task_index,
        task_count=task_count,
        regions=regions,
        total_regions=len(all_regions),
    )

    # Training is memory-intensive (SARIMAX auto-order especially). Run
    # regions sequentially WITHIN a task to keep peak RSS bounded.
    # Parallelism comes from running N tasks across the partition, not
    # from within-task threading.
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

    # Meta-write coordinator pattern: only task 0 records the
    # last-trained summary. With taskCount>1 the other tasks have an
    # incomplete view (each sees only its own slice), so a meta write
    # from task N≠0 would falsely report N's failures as the whole
    # run's failures. Task 0's view is also partial, but it's the
    # canonical writer — the UI consumes this only for "approximately
    # when did training last complete," not for per-region status.
    if task_index == 0:
        phases.write_meta(
            "last_trained",
            extra={
                "regions_trained": ok_count,
                "regions_failed": fail_regions,
                "task_count": task_count,
                "mode": "training-job",
            },
        )

    # #283 Phase 1: refresh stale/missing weather-normal artifacts for this task's
    # region slice (drives the days-17-30 forecast tail off a normal weather
    # year). Runs AFTER the last_trained pointer write so a slow/failing ERA5
    # fetch can never block it (the refresh caps build ATTEMPTS per run, but a
    # degraded archive could still burn its whole budget in fetch timeouts).
    # Quarterly per region — skips fresh ones — so a cold-start backfill spreads
    # across runs/tasks. Best-effort: never fails training.
    try:
        from data.weather_normals import refresh_weather_normals

        wn = refresh_weather_normals(regions)
        log.info(
            "training_weather_normals",
            built=len(wn["built"]),
            skipped=len(wn["skipped"]),
            failed=len(wn["failed"]),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("training_weather_normals_failed", error=str(e))

    elapsed = round(time.time() - t0, 2)
    log.info(
        "training_job_complete",
        task_index=task_index,
        task_count=task_count,
        ok_count=ok_count,
        fail_count=len(fail_regions),
        elapsed_s=elapsed,
        failed_regions=fail_regions,
    )
    sys.stdout.flush()

    return 0 if ok_count > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
