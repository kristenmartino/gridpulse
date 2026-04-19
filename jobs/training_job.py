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


def _train_xgboost(region_data: phases.RegionData) -> str | None:
    """Train + persist an XGBoost model. Returns the saved version or ``None``."""
    from models.xgboost_model import train_xgboost

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        model_dict = train_xgboost(region_data.featured_df, n_splits=3)
    except Exception as e:
        log.warning("training_xgboost_failed", region=region, error=str(e))
        return None

    mape = None
    cv_scores = model_dict.get("cv_scores") if isinstance(model_dict, dict) else None
    if cv_scores is not None and len(cv_scores) > 0:
        try:
            mape = float(np.mean(cv_scores))
        except Exception:
            mape = None

    return save_model(
        region=region,
        model_name="xgboost",
        model_obj=model_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=mape,
        extra={"cv_scores": cv_scores if cv_scores is not None else []},
    )


def _train_prophet(region_data: phases.RegionData) -> str | None:
    """Best-effort Prophet training. Returns the saved version or ``None``."""
    from models.prophet_model import train_prophet

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        prophet_obj = train_prophet(region_data.featured_df)
    except Exception as e:
        log.warning("training_prophet_failed", region=region, error=str(e))
        return None

    return save_model(
        region=region,
        model_name="prophet",
        model_obj=prophet_obj,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=None,
    )


def _train_arima(region_data: phases.RegionData) -> str | None:
    """Best-effort SARIMAX training. Returns the saved version or ``None``."""
    from models.arima_model import train_arima

    region = region_data.region
    assert region_data.featured_df is not None
    try:
        arima_dict = train_arima(region_data.featured_df)
    except Exception as e:
        log.warning("training_arima_failed", region=region, error=str(e))
        return None

    return save_model(
        region=region,
        model_name="arima",
        model_obj=arima_dict,
        data_hash=_compute_data_hash(region_data),
        train_rows=len(region_data.featured_df),
        mape=None,
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

    xgb_version = _train_xgboost(region_data)
    summary["models"]["xgboost"] = xgb_version
    prophet_version = _train_prophet(region_data)
    summary["models"]["prophet"] = prophet_version
    arima_version = _train_arima(region_data)
    summary["models"]["arima"] = arima_version

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
