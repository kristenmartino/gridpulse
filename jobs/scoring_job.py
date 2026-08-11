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


def _log_region_complete(summary: dict) -> None:
    """Emit one per-BA runtime line, phases slowest first.

    #389: per-region wall time was computed in :func:`_score_region` and never
    logged, so the only runtime number the job emitted was the fleet total.
    Called on BOTH exits — the no-data early return and the normal one — so an
    upstream slowdown, whose signature is a slow fetch that ends in no data, is
    not the one case that goes unreported.
    """
    log.info(
        "scoring_region_complete",
        region=summary.get("region"),
        ok=summary.get("ok"),
        elapsed_s=summary.get("elapsed_s"),
        timings=dict(sorted((summary.get("timings") or {}).items(), key=lambda kv: -kv[1])),
        subtimings=dict(sorted((summary.get("subtimings") or {}).items(), key=lambda kv: -kv[1])),
        fetch_subtimings=dict(
            sorted((summary.get("fetch_subtimings") or {}).items(), key=lambda kv: -kv[1])
        ),
        # #427: one key each, but per-BA is where a pathological EIA call for a
        # single region shows up — which a fleet total averages away.
        generation_subtimings=summary.get("generation_subtimings") or {},
        interchange_subtimings=summary.get("interchange_subtimings") or {},
    )


def _published_gate_visibility(region: str) -> bool | None:
    """The region's currently-published gate verdict, or ``None`` if unknown.

    Read best-effort from ``gridpulse:meta:gate_status`` — the same merged map
    the epilogue writes. A miss, an unreadable map or an absent region all
    return ``None``, which makes the gate fall back to the bare threshold:
    hysteresis must never be the reason a BA is judged, only the reason a
    borderline judgement is sticky.
    """
    try:
        from data.redis_client import redis_get, redis_key

        payload = redis_get(redis_key("meta:gate_status")) or {}
        entry = (payload.get("regions") or {}).get(region)
        if isinstance(entry, dict) and isinstance(entry.get("acceptable"), bool):
            return entry["acceptable"]
    except Exception:  # pragma: no cover — advisory only, never fails a run
        return None
    return None


def _score_region(region: str, deadline: float | None = None) -> dict:
    """Run all scoring phases for a single region. Returns a summary dict.

    ``deadline`` is a ``time.monotonic()`` instant past which this BA is
    skipped without touching the network. All 51 futures are submitted to the
    pool up front, so shedding cannot be done by not-submitting — a worker has
    to check on pickup. Skipping costs ~0s, which is what lets the run drain
    its queue and reach its epilogue instead of being killed mid-fan-out.
    """
    t0 = time.time()
    summary: dict = {
        "region": region,
        "ok": False,
        "phases": {},
        "timings": {},
        "subtimings": {},
        "fetch_subtimings": {},
        "generation_subtimings": {},
        "interchange_subtimings": {},
    }

    if deadline is not None and time.monotonic() >= deadline:
        # Not a failure — this BA never ran, and its previous forecast is still
        # live in Redis under the 24h TTL. `_forecast_errored` excludes it for
        # exactly that reason; `partial_failure` counts it separately.
        summary["skipped"] = "deadline"
        summary["phases"]["forecast"] = {"ok": False, "error": "deadline_skipped"}
        summary["elapsed_s"] = 0.0
        return summary

    def timed(name: str, fn, *args, **kwargs):
        """Run a phase and record its wall time under ``summary["timings"]``.

        #389: the runtime-creep alert fired with no per-phase breakdown to
        act on — total ``elapsed_s`` was the only number the job emitted, so
        the runbook's "reduce runtime" step had nothing to aim at. Timing is
        recorded even when the phase raises, so a slow FAILING phase is as
        visible as a slow succeeding one.
        """
        _t = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            summary["timings"][name] = round(time.time() - _t, 2)

    def timed_with_substeps(name: str, key: str, fn, *args, **kwargs):
        """``timed``, plus collect this phase's sub-steps into ``summary[key]``.

        One collector PER PHASE, never one shared across phases. That keeps
        the invariant that makes these readable: a phase's sub-steps sum to
        (at most) its own phase total, so `generation_substeps.eia_generation`
        can be checked against `phases.generation`. A shared collector would
        destroy it. Same double-count reasoning that keeps sub-steps out of
        ``summary["timings"]`` — `_phase_rollup` sums every key it finds.
        """
        with phases.collect_substeps() as sub:
            res = timed(name, fn, *args, **kwargs)
        summary[key] = dict(sub)
        return res

    # #389 follow-up: the `fetch` breakdown — EIA leg vs the three Open-Meteo
    # legs.
    region_data = timed_with_substeps("fetch", "fetch_subtimings", phases.fetch_region_data, region)
    if region_data is None:
        summary["phases"]["fetch"] = {"ok": False, "error": "no_data"}
        summary["elapsed_s"] = round(time.time() - t0, 2)
        # A BA that fails its fetch is the one whose timing you most want —
        # under an upstream slowdown its `fetch` is the SLOWEST in the fleet
        # (it paid the full retry budget and still got nothing). Emitting only
        # on the success path would hide exactly that case.
        _log_region_complete(summary)
        return summary
    summary["phases"]["fetch"] = {
        "ok": True,
        "demand_rows": len(region_data.demand_df),
        "weather_rows": len(region_data.weather_df),
    }

    # #309: vintage capture MUST run on the RAW frame — it records what EIA
    # actually said, and the quality guard below would erase exactly the
    # artifact values the study exists to measure. Order is load-bearing.
    vintage_res = timed("vintage", phases.write_vintage_records, region, region_data.demand_df)
    summary["phases"]["vintage"] = {
        "ok": vintage_res.ok,
        **(vintage_res.details if vintage_res.ok else {"error": vintage_res.error}),
    }

    # #309: NaN-coerce implausible trailing readings ONCE, so every downstream
    # consumer — actuals payload (tiles), drift scoring, the forecast anchor —
    # sees the same cleaned frame. Never fatal.
    guard_res = timed("quality_guard", phases.apply_demand_quality_guard, region_data)
    summary["phases"]["quality_guard"] = {
        "ok": guard_res.ok,
        **(guard_res.details if guard_res.ok else {"error": guard_res.error}),
    }

    # ADR-009: for broken-feed regions, fork a conditioned frame whose
    # trailing hours anchor on the BA's own day-ahead forecast. Runs AFTER
    # the guard (it conditions the guard-cleaned real frame) and never
    # touches data.demand_df — actuals/drift/alerts stay real. Flag-dark.
    conditioning_res = timed("anchor_conditioning", phases.condition_anchor_frame, region_data)
    summary["phases"]["anchor_conditioning"] = {
        "ok": conditioning_res.ok,
        **(conditioning_res.details if conditioning_res.ok else {"error": conditioning_res.error}),
    }

    # #121 part 1: snapshot the about-to-be-overwritten forecast key
    # BEFORE write_actuals_and_weather + predict_and_write_forecast run.
    # The drift phase later in this function compares this previous
    # forecast's 1-hour-ahead prediction against the now-known actuals.
    # Read failure → drift phase becomes a no-op for this tick.
    previous_forecast = timed("read_existing_forecast", phases.read_existing_forecast, region)

    actuals_res = timed("actuals_weather", phases.write_actuals_and_weather, region_data)
    summary["phases"]["actuals_weather"] = {
        "ok": actuals_res.ok,
        **(actuals_res.details if actuals_res.ok else {"error": actuals_res.error}),
    }

    # #121 part 1: continuous drift signal. Runs after actuals are
    # written + before predict_and_write_forecast overwrites the
    # forecast key. Failures are isolated — a drift-side error never
    # blocks the broader scoring run because drift is a secondary
    # signal, not a critical path.
    drift_res = timed(
        "drift", phases.write_drift_metrics, region, previous_forecast, region_data.demand_df
    )
    summary["phases"]["drift"] = {
        "ok": drift_res.ok,
        **(drift_res.details if drift_res.ok else {"error": drift_res.error}),
    }

    # (vintage capture moved above the quality guard — it must see the raw frame)

    # #227: horizon-matched drift (24h/48h/72h) — snapshots the same
    # about-to-be-overwritten forecast + resolves matured snapshots, graded
    # against each horizon's own band. Isolated like the 1h phase above.
    horizon_drift_res = timed(
        "drift_horizon",
        phases.write_horizon_drift_metrics,
        region,
        previous_forecast,
        region_data.demand_df,
    )
    summary["phases"]["drift_horizon"] = {
        "ok": horizon_drift_res.ok,
        **(
            horizon_drift_res.details
            if horizon_drift_res.ok
            else {"error": horizon_drift_res.error}
        ),
    }

    # E0 public benchmark: GridPulse vs. this BA's OWN day-ahead forecast.
    # Reads the vintage window + vintage_summary (written earlier this tick)
    # and the horizon-drift records (written immediately above), so it must
    # stay after both. Isolated like the drift phases — the benchmark is
    # published evidence, never a reason to fail a scoring run.
    benchmark_res = timed(
        "benchmark",
        phases.write_benchmark_metrics,
        region,
        previous_forecast,
        region_data.demand_df,
    )
    summary["phases"]["benchmark"] = {
        "ok": benchmark_res.ok,
        **(benchmark_res.details if benchmark_res.ok else {"error": benchmark_res.error}),
    }

    gen_res = timed_with_substeps(
        "generation", "generation_subtimings", phases.write_generation, region
    )
    summary["phases"]["generation"] = {
        "ok": gen_res.ok,
        **(gen_res.details if gen_res.ok else {"error": gen_res.error}),
    }

    # V3.α: BA-to-BA interchange snapshot. Independent of generation —
    # a sparse interchange fetch shouldn't fail the broader scoring run,
    # so we ignore the PhaseResult's ok flag for the summary aggregate.
    ix_res = timed_with_substeps(
        "interchange", "interchange_subtimings", phases.write_interchange, region
    )
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
    def _load_all() -> tuple:
        return (
            load_model(region, "xgboost"),
            load_model(region, "prophet"),
            load_model(region, "arima"),
        )

    xgb_loaded, prophet_loaded, arima_loaded = timed("model_load", _load_all)

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
    # #451: which MAPE drives the weights — latest holdout, or its EWMA when
    # ``smoothed_ensemble_weights`` is on. Routed through one helper so scoring
    # and training cannot disagree about the INPUT, the same way
    # ``resolve_ensemble_weights`` stops them disagreeing about the RULE (P2-16).
    from models.ensemble import shadow_weighting_mape, weighting_mape

    model_mapes_shadow: dict[str, float | None] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    if xgb_loaded is not None:
        xgb_model, xgb_meta = xgb_loaded
        loaded_models["xgboost"] = xgb_model
        model_mapes["xgboost"] = weighting_mape(xgb_meta.mape, xgb_meta.extra)
        model_mapes_shadow["xgboost"] = shadow_weighting_mape(xgb_meta.mape, xgb_meta.extra)
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
        model_mapes["prophet"] = weighting_mape(prophet_meta.mape, prophet_meta.extra)
        model_mapes_shadow["prophet"] = shadow_weighting_mape(prophet_meta.mape, prophet_meta.extra)
        summary["prophet_version"] = prophet_meta.version
        prophet_metrics = _extract_holdout_metrics(prophet_meta)
        if prophet_metrics:
            model_metrics["prophet"] = prophet_metrics
    if arima_loaded is not None:
        arima_model, arima_meta = arima_loaded
        loaded_models["arima"] = arima_model
        model_mapes["arima"] = weighting_mape(arima_meta.mape, arima_meta.extra)
        model_mapes_shadow["arima"] = shadow_weighting_mape(arima_meta.mape, arima_meta.extra)
        summary["arima_version"] = arima_meta.version
        arima_metrics = _extract_holdout_metrics(arima_meta)
        if arima_metrics:
            model_metrics["arima"] = arima_metrics

    # Publish this BA's forecast-quality gate verdict (#271 / P2-10). Computed
    # here — where the real holdout metrics live — so the stateless web tier reads
    # an authoritative verdict from Redis instead of fataling open on an outage or
    # sweeping GCS metas per render. Only regions with a real metric signal carry
    # a verdict; a no-metric region is simply absent from the map (the web gate
    # treats absent as warming → visible), so untrained BAs are never hidden.
    if model_metrics:
        from models.model_service import (
            OPERATING_HORIZON,
            gate_disagrees_with_live,
            gate_verdict_from_metrics,
            live_horizon_verdict,
        )

        # P2-17 (#273): pass the region's CURRENT published verdict so the gate
        # can apply hysteresis. Without it the bare threshold flaps a
        # near-the-bar BA in and out of the UI. ``None`` on a first-ever
        # judgement or an unreadable map, which reproduces the old behaviour.
        summary["gate"] = gate_verdict_from_metrics(
            model_metrics, currently_visible=_published_gate_visibility(region)
        )

        # #349. The verdict above answers "can we forecast this BA at all?"
        # against the TRAINING HOLDOUT and the generous 7-day band. It is a
        # deliberately low bar and it stays low — hiding a region is heavy-
        # handed, and today it hides none of the 51. What was missing is the
        # second opinion: how the models actually did in the serve path, at
        # the horizon we publish. SEC passed the gate at 6.96% holdout while
        # every model graded `rollback` at 24h, and nothing anywhere noticed.
        # Both numbers are now published side by side, and the disagreement
        # is logged as an alert rather than being silently reconciled.
        try:
            from data.redis_client import redis_get, redis_key

            live = live_horizon_verdict(redis_get(redis_key(f"drift_horizon:{region}")))
        except Exception as e:  # pragma: no cover - defensive; must not fail scoring
            log.warning("gate_live_horizon_read_failed", region=region, error=str(e))
            live = None
        if live:
            summary["gate"]["live_horizon"] = live
            if gate_disagrees_with_live(summary["gate"], live):
                summary["gate"]["disagrees"] = True
                log.warning(
                    "gate_live_horizon_disagreement",
                    region=region,
                    holdout_mape=summary["gate"].get("best_mape"),
                    live_mape=live["champion_mape"],
                    live_grade=live["grade"],
                    live_champion=live["champion"],
                    horizon=OPERATING_HORIZON,
                )

    has_features = timed("features", phases.engineer_region_features, region_data) is not None

    if has_features and loaded_models:
        # #389 follow-up: `forecast` is 60.1% of worker time and the phase-level
        # number cannot say which part. Collect its sub-steps here. Kept out of
        # `summary["timings"]` on purpose — `_phase_rollup` sums every key there,
        # so mixing sub-steps in would double-count against the phase total.
        with phases.collect_substeps() as _sub:
            fc_res = timed(
                "forecast",
                phases.predict_and_write_forecast,
                region_data,
                loaded_models,
                model_mapes,
                model_metrics=model_metrics,
                model_mapes_shadow=model_mapes_shadow,
            )
        summary["subtimings"] = dict(_sub)
        summary["phases"]["forecast"] = {
            "ok": fc_res.ok,
            **(fc_res.details if fc_res.ok else {"error": fc_res.error}),
        }
        # Diagnostics needs XGBoost specifically (SHAP + per-residual).
        if "xgboost" in loaded_models:
            diag_res = timed(
                "diagnostics", phases.write_diagnostics, region_data, loaded_models["xgboost"]
            )
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

    wc_res = timed("weather_correlation", phases.write_weather_correlation, region_data)
    summary["phases"]["weather_correlation"] = {
        "ok": wc_res.ok,
        **(wc_res.details if wc_res.ok else {"error": wc_res.error}),
    }

    alerts_res = timed("alerts", phases.write_alerts, region_data)
    summary["phases"]["alerts"] = {
        "ok": alerts_res.ok,
        **(alerts_res.details if alerts_res.ok else {"error": alerts_res.error}),
    }

    # A region counts as SCORED only when its forecast phase produced output —
    # the core deliverable. Before #267 this was ``any(phase ok)``, so a region
    # whose forecast failed but whose alerts/weather-correlation write succeeded
    # was still counted "scored"; a run where all 51 forecasts failed then exited
    # 0 and refreshed last_scored, masking the outage the alerting exists to catch
    # (P2-01). ``no_model`` regions (untrained BAs) legitimately have no forecast
    # and count as not-scored, which is correct — they carry no fresh forecast.
    summary["ok"] = bool(summary["phases"].get("forecast", {}).get("ok"))
    summary["elapsed_s"] = round(time.time() - t0, 2)
    _log_region_complete(summary)
    return summary


def _phase_rollup(results: list[dict], key: str = "timings") -> dict[str, dict]:
    """Fold per-region phase timings into a fleet-wide breakdown.

    Returns ``{phase: {total_s, max_s, slowest_region, n}}`` sorted by total
    time descending — the answer to "where did the run actually go", which
    is what the runtime-creep runbook needs and previously had to be
    reconstructed by hand from raw logs.

    Note the totals are summed CPU/wall across regions scored concurrently,
    so they exceed the run's wall clock; they rank the phases, they don't
    partition the elapsed time.

    ``key`` selects which per-region timing dict to fold. It stays
    ``"timings"`` by default; ``"subtimings"`` folds the forecast sub-steps
    (#389 follow-up) through the identical arithmetic. The two are rolled up
    and logged SEPARATELY on purpose — summing sub-steps into the phase table
    would double-count them against their own parent phase.
    """
    agg: dict[str, dict] = {}
    for r in results:
        for phase, secs in (r.get(key) or {}).items():
            slot = agg.setdefault(
                phase, {"total_s": 0.0, "max_s": 0.0, "slowest_region": None, "n": 0}
            )
            slot["total_s"] += secs
            slot["n"] += 1
            if secs > slot["max_s"]:
                slot["max_s"] = secs
                slot["slowest_region"] = r.get("region")
    for slot in agg.values():
        slot["total_s"] = round(slot["total_s"], 1)
        slot["max_s"] = round(slot["max_s"], 1)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]["total_s"]))


def _check_runtime_headroom(elapsed_s: float, rollup: dict[str, dict] | None = None) -> None:
    """#171 recurrence guardrail — warn on runtime CREEP before it times out.

    The PR-G10 job-failure alert only fires on an outright timeout (~1700s under
    the 1800s cap); by then a scoring tick has already been killed. This tracks
    consecutive completed runs whose ``elapsed_s`` exceeds
    ``SCORING_RUNTIME_HEADROOM_FRACTION`` of ``SCORING_TASK_TIMEOUT_S`` and, once
    the streak reaches ``SCORING_RUNTIME_CREEP_RUNS``, emits a
    ``scoring_runtime_creep`` ERROR log — matched by the Cloud Monitoring policy
    in ``docs/monitoring/scoring_runtime_creep_alert.json`` — so runtime can be
    reduced on-schedule instead of paged at the next outage. The consecutive
    count lives in Redis (``gridpulse:scoring_runtime_state``) because each job
    run is a fresh process. Best-effort: an error here never fails the run.

    ``rollup`` (#389) is the fleet-wide per-phase breakdown from
    ``_phase_rollup``. Its top entries ride along on the alert log so the
    Cloud Monitoring notification names the phases to go after, instead of
    only reporting that the run was slow.
    """
    from config import (
        SCORING_RUNTIME_CREEP_RUNS,
        SCORING_RUNTIME_HEADROOM_FRACTION,
        SCORING_TASK_TIMEOUT_S,
    )

    if SCORING_TASK_TIMEOUT_S <= 0:
        return
    threshold_s = SCORING_TASK_TIMEOUT_S * SCORING_RUNTIME_HEADROOM_FRACTION
    pct = round(elapsed_s / SCORING_TASK_TIMEOUT_S * 100, 1)
    breached = elapsed_s >= threshold_s
    try:
        from data.redis_client import redis_get, redis_key, redis_set

        key = redis_key("scoring_runtime_state")
        state = redis_get(key)
        prior = int(state.get("consecutive_breaches", 0)) if isinstance(state, dict) else 0
        consecutive = prior + 1 if breached else 0
        redis_set(
            key,
            {
                "consecutive_breaches": consecutive,
                "last_elapsed_s": elapsed_s,
                "pct_of_timeout": pct,
                "threshold_s": round(threshold_s, 1),
                "task_timeout_s": SCORING_TASK_TIMEOUT_S,
            },
            ttl=7 * 24 * 3600,
        )
        if consecutive >= SCORING_RUNTIME_CREEP_RUNS:
            log.error(
                "scoring_runtime_creep",
                elapsed_s=elapsed_s,
                pct_of_timeout=pct,
                threshold_s=round(threshold_s, 1),
                task_timeout_s=SCORING_TASK_TIMEOUT_S,
                consecutive_breaches=consecutive,
                top_phases=list((rollup or {}).items())[:5],
                message=(
                    f"Scoring runtime {elapsed_s}s is {pct}% of the "
                    f"{SCORING_TASK_TIMEOUT_S}s task timeout for {consecutive} "
                    "consecutive runs — reduce runtime before it times out (#171)."
                ),
            )
        elif breached:
            log.warning(
                "scoring_runtime_headroom_low",
                elapsed_s=elapsed_s,
                pct_of_timeout=pct,
                consecutive_breaches=consecutive,
            )
    except Exception as e:  # never let the guardrail fail the run
        log.warning("scoring_runtime_headroom_check_failed", error=str(e))


def _soft_deadline() -> float | None:
    """Monotonic instant past which the run stops starting new BAs.

    None disables shedding — flag off, or the fraction set to 0. Monotonic
    rather than wall clock on purpose: a container clock step must not extend
    or collapse the budget.
    """
    from config import SCORING_SOFT_DEADLINE_FRACTION, SCORING_TASK_TIMEOUT_S, feature_enabled

    if not feature_enabled("soft_deadline"):
        return None
    if SCORING_SOFT_DEADLINE_FRACTION <= 0 or SCORING_TASK_TIMEOUT_S <= 0:
        return None
    return time.monotonic() + SCORING_TASK_TIMEOUT_S * SCORING_SOFT_DEADLINE_FRACTION


def run() -> int:
    """Run the scoring job end-to-end. Returns an exit code."""
    t0 = time.time()
    regions = phases.ordered_regions(PRECOMPUTE_DEFAULT_REGION)
    log.info("scoring_job_start", regions=regions)

    # 2026-08-04: without this the task is simply SIGKILLed at the timeout, and
    # a run that had already scored 49 of 51 BAs records NONE of it, because
    # the freshness meta and fleet rollup below sit after the fan-out.
    deadline = _soft_deadline()
    results: list[dict] = []
    shed: list[str] = []
    with ThreadPoolExecutor(max_workers=PRECOMPUTE_MAX_WORKERS) as pool:
        futures = {pool.submit(_score_region, r, deadline): r for r in regions}
        for fut in as_completed(futures):
            region = futures[fut]
            try:
                summary = fut.result()
                if summary.get("skipped") == "deadline":
                    shed.append(region)
                results.append(summary)
            except Exception as e:
                log.warning(
                    "scoring_job_region_crashed",
                    region=region,
                    error=str(e),
                )
                results.append({"region": region, "ok": False, "error": str(e)})

    from config import SCORING_MIN_OK_REGIONS

    def _forecast_errored(r: dict) -> bool:
        # A REAL forecast failure — attempted and failed — as distinct from an
        # expected ``no_model`` skip (an untrained/new BA, which is neither
        # scored nor a failure) or a ``deadline_skipped`` shed (which never ran
        # at all and still holds a live forecast in Redis). A region that
        # crashed entirely has no ``phases``.
        if r.get("ok"):
            return False
        err = r.get("phases", {}).get("forecast", {}).get("error")
        return err not in ("no_model", "deadline_skipped")

    # "Scored" now means the forecast phase produced output (#267): a region that
    # only wrote alerts/weather no longer inflates this count.
    ok_count = sum(1 for r in results if r.get("ok"))
    errored = [r["region"] for r in results if _forecast_errored(r)]
    fail_regions = [r["region"] for r in results if not r.get("ok")]  # incl. no_model
    # Partial failure = real forecast ERRORS *or* deadline shedding dragged the
    # scored count below the floor while some regions still succeeded.
    # All-``no_model`` (a fresh deploy with nothing trained yet) is expected,
    # not a failure. It's not a total outage either, so Cloud Scheduler retry
    # wouldn't help — but it must be VISIBLE: emit an alertable ERROR + degrade
    # the freshness meta.
    #
    # ``shed`` belongs in this condition and it is easy to leave out. Excluding
    # shed BAs from ``errored`` is correct — they did not fail — but if that
    # were the whole change, a run that scored 30/51 purely by deadline would
    # have ``errored == []``, so ``partial_failure`` would be False and
    # /health would read OK over 21 stale BAs. Silent, and worse than the
    # timeout this replaces.
    partial_failure = bool(errored or shed) and ok_count < SCORING_MIN_OK_REGIONS

    phases.write_meta(
        "last_scored",
        extra={
            "regions_scored": ok_count,
            "regions_total": len(results),
            "regions_failed": fail_regions,
            "regions_errored": errored,
            "partial_failure": partial_failure,
            # Disclose shedding rather than let a short run look like a full
            # one: /health and the grid summary can then say "44/51 scored,
            # deadline hit" instead of asserting freshness they don't have.
            "deadline_hit": bool(shed),
            "regions_deadline_skipped": shed,
            "mode": "scoring-job",
        },
    )

    # Publish the consolidated forecast-quality gate verdict map (#271 / P2-10).
    # The web tier reads this one key (Redis-only) to filter the dropdown /
    # US-Grid, instead of recomputing per-render from GCS metas.
    #
    # MERGE over the last-known map rather than replacing it. A degraded run — a
    # model-store outage makes load_model() return None for every region, so
    # model_metrics is empty and no region carries a verdict — would otherwise
    # write an empty/partial map over the good 51-region one on the same 24h key.
    # The web tier reads a present-but-empty map as "every region warming ->
    # visible" (a non-None dict short-circuits before the pass-open log), silently
    # un-hiding rollback-grade BAs whose stale forecasts are still served — the
    # exact fail-open this fix exists to remove. Merging preserves un-scored
    # regions' last-known verdicts; a run that produced NO verdicts at all skips
    # the write, so the prior map lives out its 24h TTL untouched and self-heals.
    # (Caught by adversarial verification of this PR.)
    this_run = {r["region"]: r["gate"] for r in results if isinstance(r.get("gate"), dict)}
    if this_run:
        try:
            from data.redis_client import redis_get, redis_key

            prev = redis_get(redis_key("meta:gate_status"))
            prev_regions = prev.get("regions") if isinstance(prev, dict) else None
            merged = dict(prev_regions) if isinstance(prev_regions, dict) else {}
        except Exception as e:
            log.warning("scoring_gate_status_prev_read_failed", error=str(e))
            merged = {}
        merged.update(this_run)
        phases.write_meta("gate_status", extra={"regions": merged})
        log.info("scoring_gate_status_written", total=len(merged), updated=len(this_run))
    else:
        # No region produced a verdict this run (total model-store outage). Leave
        # the last-known map in place rather than clobbering it with {}.
        log.warning("scoring_gate_status_skipped_no_verdicts")

    # E0 fleet rollup: fold the per-BA benchmark payloads into the public
    # headline (win/loss, medians, the accuracy-spread comparison) plus the
    # excluded list with reasons. Read back from Redis rather than threaded
    # through the parallel per-region results — one cheap pass, and it stays
    # correct if a region was scored by an earlier run.
    #
    # Mirrored to GCS because the benchmark claims a *trajectory*: the Redis
    # keys carry a 24h TTL, so a longer outage would silently erase the
    # public track record (the vintage instrument mirrors for the same
    # reason). Fire-and-forget — never fail a scoring run for it.
    try:
        from data.redis_client import redis_get, redis_key
        from models.benchmark import fleet_rollup

        payloads = []
        for r in regions:
            p = redis_get(redis_key(f"benchmark:{r}"))
            if isinstance(p, dict) and p.get("region"):
                payloads.append(p)
        if payloads:
            rollup = fleet_rollup(payloads)
            phases.write_meta("benchmark_fleet", extra=rollup)
            log.info(
                "benchmark_fleet_written",
                scoreable=rollup["n_scoreable"],
                excluded=rollup["n_excluded"],
                wins=rollup["fleet"]["wins"],
                losses=rollup["fleet"]["losses"],
            )
        else:
            log.warning("benchmark_fleet_skipped_no_payloads")
    except Exception as e:
        log.warning("benchmark_fleet_failed", error=str(e))

    elapsed = round(time.time() - t0, 2)
    # Named apart from the benchmark ``rollup`` above — different shape,
    # different purpose, and they would otherwise share a name eight lines apart.
    phase_rollup = _phase_rollup(results)
    log.info(
        "scoring_job_complete",
        ok_count=ok_count,
        fail_count=len(fail_regions),
        errored_count=len(errored),
        elapsed_s=elapsed,
        failed_regions=fail_regions,
    )
    # #389: the phase breakdown the runtime-creep runbook needs — emitted on
    # EVERY run, not just breaching ones, so the trend is already on record
    # by the time an alert fires.
    log.info(
        "scoring_phase_rollup",
        elapsed_s=elapsed,
        workers=PRECOMPUTE_MAX_WORKERS,
        regions=len(results),
        phases=phase_rollup,
        # #389 follow-up: the breakdown INSIDE `forecast`, which is 60.1% of
        # phase time and until now was one opaque number. Separate field, not
        # merged into `phases` — see `_phase_rollup`'s `key` docstring.
        forecast_substeps=_phase_rollup(results, key="subtimings"),
        # #389: the breakdown INSIDE `fetch` — `eia_demand` vs the three
        # Open-Meteo legs (`weather_forecast` / `weather_nbm` /
        # `weather_archive`). This is what sizes a weather-side change against
        # the upstream latency it shares a phase with.
        fetch_substeps=_phase_rollup(results, key="fetch_subtimings"),
        # #427: the EIA legs outside `fetch`. Separate fields, one per
        # phase, so each stays checkable against its own phase total.
        generation_substeps=_phase_rollup(results, key="generation_subtimings"),
        interchange_substeps=_phase_rollup(results, key="interchange_subtimings"),
    )
    # The `fetch` phase above is upstream-latency-dominated, so publish the
    # EIA success-latency distribution beside it. This is the measurement the
    # 30s read timeout in `_request_with_backoff` has never had — a healthy
    # p99 here is what a lowered timeout must be derived from.
    try:
        from data.eia_client import drain_latency_stats

        eia_stats = drain_latency_stats()
        if eia_stats:
            log.info("eia_client_stats", **eia_stats)
    except Exception as e:  # never let instrumentation fail a run
        log.warning("eia_client_stats_failed", error=str(e))

    # Fail-SOFT Redis writes (redis_set) that were dropped this run. The
    # critical payloads — actuals, weather, forecast, vintage — go through
    # fail-loud persist() and already fail their phase (#268), so this covers
    # the secondary ones: generation, interchange, drift, benchmark, backtest,
    # weather-correlation, alerts and meta. Individually non-fatal by design,
    # but a run that silently dropped 40 of them is a Redis problem nobody was
    # being told about — every one of those 15 call sites ignores the returned
    # False, and until now the only trace was a stdlib-logging warning that
    # reached Cloud Logging as textPayload, which no log-based alert can match.
    try:
        from data.redis_client import drain_write_failures

        wf = drain_write_failures()
        if wf:
            log.error(
                "redis_write_failures",
                **wf,
                message=(
                    f"{wf['count']} fail-soft Redis writes were dropped this run "
                    f"({wf['by_kind']}). Secondary payloads only — actuals/weather/"
                    "forecast/vintage are fail-loud — but the affected surfaces are "
                    "serving stale or absent data without any phase reporting failure."
                ),
            )
    except Exception as e:  # never let instrumentation fail a run
        log.warning("redis_write_failures_check_failed", error=str(e))

    if shed:
        # Its own event, deliberately not folded into scoring_partial_failure:
        # that alert means "forecasts are erroring", and this means "we ran out
        # of time". Same severity, different runbook — one points at the model
        # path, this one points at runtime.
        from config import SCORING_SOFT_DEADLINE_FRACTION, SCORING_TASK_TIMEOUT_S

        log.error(
            "scoring_deadline_shed",
            regions_scored=ok_count,
            regions_skipped=len(shed),
            skipped_regions=shed,
            elapsed_s=elapsed,
            soft_deadline_fraction=SCORING_SOFT_DEADLINE_FRACTION,
            task_timeout_s=SCORING_TASK_TIMEOUT_S,
            message=(
                f"Scoring shed {len(shed)} of {len(results)} BAs at the soft "
                f"deadline ({elapsed}s elapsed) — the run completed and wrote "
                "what it had instead of being killed. Reduce runtime."
            ),
        )
    if partial_failure:
        log.error(
            "scoring_partial_failure",
            regions_scored=ok_count,
            regions_total=len(results),
            min_ok_regions=SCORING_MIN_OK_REGIONS,
            errored_regions=errored,
        )
    # #171: warn on runtime creep before it becomes a timeout (see the guardrail).
    _check_runtime_headroom(elapsed, phase_rollup)
    sys.stdout.flush()

    # Non-zero exit only on a true total forecast outage — no region scored AND
    # at least one forecast actually errored (models exist but every attempt
    # failed), where a Cloud Scheduler retry is worth firing. An all-``no_model``
    # run (fresh deploy, nothing trained yet) exits 0 — retrying can't help, and
    # the untrained state is a training concern, not a scoring failure. Partial
    # failures surface via the scoring_partial_failure alert, not the exit code.
    return 1 if (ok_count == 0 and errored) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
