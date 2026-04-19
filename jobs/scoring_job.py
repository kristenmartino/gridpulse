"""
GridPulse scoring job — hourly.

Refreshes Redis with the data the Dash web service reads:

- actuals (``wattcast:actuals:{region}``)
- weather (``wattcast:weather:{region}``)
- generation by fuel (``wattcast:generation:{region}``)
- 720h forward forecast (``wattcast:forecast:{region}:1h``)
- weather-correlation payload (``wattcast:weather-correlation:{region}``)
- model diagnostics (``wattcast:diagnostics:{region}``)
- alerts / stress / anomalies (``wattcast:alerts:{region}``)
- ``wattcast:meta:last_scored`` marker

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
from models.persistence import load_model

log = structlog.get_logger()


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

    actuals_res = phases.write_actuals_and_weather(region_data)
    summary["phases"]["actuals_weather"] = {
        "ok": actuals_res.ok,
        **(actuals_res.details if actuals_res.ok else {"error": actuals_res.error}),
    }

    gen_res = phases.write_generation(region)
    summary["phases"]["generation"] = {
        "ok": gen_res.ok,
        **(gen_res.details if gen_res.ok else {"error": gen_res.error}),
    }

    # Feature engineering + model load are only needed for the forecast
    # and diagnostics phases. If the model is missing (first deploy before
    # training job has run) we still emit actuals + weather + generation.
    xgb_loaded = load_model(region, "xgboost")
    if phases.engineer_region_features(region_data) is not None and xgb_loaded is not None:
        xgb_model, meta = xgb_loaded
        summary["model_version"] = meta.version
        fc_res = phases.predict_and_write_forecast(region_data, xgb_model)
        summary["phases"]["forecast"] = {
            "ok": fc_res.ok,
            **(fc_res.details if fc_res.ok else {"error": fc_res.error}),
        }
        diag_res = phases.write_diagnostics(region_data, xgb_model)
        summary["phases"]["diagnostics"] = {
            "ok": diag_res.ok,
            **(diag_res.details if diag_res.ok else {"error": diag_res.error}),
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
