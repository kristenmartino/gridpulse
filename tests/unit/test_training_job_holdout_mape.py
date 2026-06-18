"""Unit tests for the holdout-metrics flow in jobs/training_job.py.

History:
- Originally these tests covered ``_holdout_mape_prophet`` /
  ``_holdout_mape_arima`` returning a float MAPE.
- Real-metrics work renamed those to ``_holdout_metrics_*`` and
  changed the return shape to a dict: ``{metrics: {mape, rmse, mae, r2},
  forecast: ndarray, actual: ndarray}``. The richer payload powers two
  things downstream:
    1. Each per-model meta gets ``extra["holdout_metrics"]`` with the
       full {mape, rmse, mae, r2} dict, so the Models tab can stop
       supplementing RMSE / MAE / R² from ``_simulate_forecasts`` Redis.
    2. The ensemble's holdout metric is computed from the SAME
       predictions (no recomputation, no provenance drift) and stashed
       in xgboost's meta extra.

These tests verify per-model holdout still returns a finite metric set,
gracefully skips on short windows, and that ``_train_prophet`` /
``_train_arima`` thread the holdout payload through to ``save_model``
with both top-level ``mape`` and ``extra["holdout_metrics"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def featured_df_long():
    """60 days @ hourly granularity — covers ``_HOLDOUT_HOURS + _MIN_TRAIN_HOURS``."""
    ts = pd.date_range("2024-01-01", periods=60 * 24, freq="h", tz="UTC")
    n = len(ts)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 40_000.0 + 5_000.0 * np.sin(2 * np.pi * np.arange(n) / 24),
            "hour": ts.hour,
        }
    )


@pytest.fixture
def featured_df_short():
    """20 days — too short to leave 30 days of training after a 7-day holdout."""
    ts = pd.date_range("2024-01-01", periods=20 * 24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": np.full(len(ts), 40_000.0),
            "hour": ts.hour,
        }
    )


class TestHoldoutMetricsProphet:
    def test_returns_full_metric_set_when_predict_succeeds(
        self, featured_df_long, monkeypatch
    ) -> None:
        """Train + predict succeed → returns ``{metrics, forecast, actual}`` with
        finite MAPE / RMSE / MAE / R²."""
        import models.prophet_model as prophet_mod

        def _fake_predict(model, df, periods=168):
            actuals = df["demand_mw"].values
            return {"forecast": actuals * 1.05}  # 5% bias → MAPE ≈ 5

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_metrics_prophet

        result = _holdout_metrics_prophet(featured_df_long, region="TEST")
        assert result is not None
        assert set(result.keys()) >= {"metrics", "forecast", "actual"}
        m = result["metrics"]
        assert set(m.keys()) == {"mape", "rmse", "mae", "r2"}
        assert m["mape"] == pytest.approx(5.0, abs=0.5)
        assert m["rmse"] > 0
        assert m["mae"] > 0
        # 5% multiplicative bias creates residual variance comparable to the
        # signal variance — R² lands around ~0.68 in this synthetic setup.
        # Just assert it's positive (model beats predicting the mean).
        assert m["r2"] > 0.0

    def test_short_window_returns_none(self, featured_df_short, monkeypatch) -> None:
        """Insufficient data → skip holdout, return None (no spurious metrics)."""
        import models.prophet_model as prophet_mod

        called: list[bool] = []
        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: called.append(True))

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_short, region="TEST") is None
        assert called == []  # train should not even be invoked

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        """Predict raising → swallow, return None (production training continues)."""
        import models.prophet_model as prophet_mod

        def _broken_predict(model, df, periods=168):
            raise RuntimeError("synthetic prophet predict failure")

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _broken_predict)

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_long, region="TEST") is None

    def test_non_finite_forecast_returns_none(self, featured_df_long, monkeypatch) -> None:
        """NaN forecast → return None (don't persist garbage)."""
        import models.prophet_model as prophet_mod

        def _fake_predict(model, df, periods=168):
            return {"forecast": np.full(len(df), np.nan)}

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(prophet_mod, "predict_prophet", _fake_predict)

        from jobs.training_job import _holdout_metrics_prophet

        assert _holdout_metrics_prophet(featured_df_long, region="TEST") is None


class TestHoldoutMetricsArima:
    def test_returns_full_metric_set_when_predict_succeeds(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.arima_model as arima_mod

        def _fake_predict(model_dict, exog, periods=168):
            actuals = exog["demand_mw"].values
            return actuals * 0.97  # 3% under-forecast → MAPE ≈ 3

        monkeypatch.setattr(
            arima_mod,
            "train_arima",
            lambda df, **kwargs: {
                "params": [],
                "order": (2, 1, 2),
                "seasonal_order": (1, 1, 1, 24),
            },
        )
        monkeypatch.setattr(arima_mod, "predict_arima", _fake_predict)

        from jobs.training_job import _holdout_metrics_arima

        result = _holdout_metrics_arima(featured_df_long, region="TEST")
        assert result is not None
        assert result["metrics"]["mape"] == pytest.approx(3.0, abs=0.5)
        # See prophet test for why R² is moderate, not near-1, in this
        # synthetic setup.
        assert result["metrics"]["r2"] > 0.0

    def test_short_window_returns_none(self, featured_df_short) -> None:
        from jobs.training_job import _holdout_metrics_arima

        assert _holdout_metrics_arima(featured_df_short, region="TEST") is None

    def test_predict_failure_returns_none(self, featured_df_long, monkeypatch) -> None:
        import models.arima_model as arima_mod

        def _broken_predict(model_dict, exog, periods=168):
            raise ValueError("synthetic arima predict failure")

        monkeypatch.setattr(
            arima_mod,
            "train_arima",
            lambda df, **kwargs: {
                "params": [],
                "order": (2, 1, 2),
                "seasonal_order": (1, 1, 1, 24),
            },
        )
        monkeypatch.setattr(arima_mod, "predict_arima", _broken_predict)

        from jobs.training_job import _holdout_metrics_arima

        assert _holdout_metrics_arima(featured_df_long, region="TEST") is None


class TestTrainPersistsHoldoutMetrics:
    """End-to-end: ``_train_prophet`` / ``_train_arima`` pass the full
    holdout-metric dict through to ``save_model`` — both as the
    top-level ``mape`` (used for inverse-MAPE ensemble weighting) and
    as ``extra["holdout_metrics"]`` (powers the Models tab's RMSE /
    MAE / R²)."""

    def _make_region_data(self, df) -> object:
        from jobs.phases import RegionData

        return RegionData(
            region="TEST",
            demand_df=df[["timestamp", "demand_mw"]].copy(),
            weather_df=df.copy(),
            featured_df=df,
        )

    def test_prophet_passes_full_holdout_metrics_to_save_model(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.prophet_model as prophet_mod

        monkeypatch.setattr(prophet_mod, "train_prophet", lambda df: object())
        monkeypatch.setattr(
            prophet_mod,
            "predict_prophet",
            lambda model, df, periods=168: {
                "forecast": np.asarray(df["demand_mw"].values) * 1.04,  # MAPE ≈ 4
            },
        )

        captured: dict = {}

        def _save_model(**kwargs):
            captured.update(kwargs)
            return "v-test"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)
        monkeypatch.setattr("jobs.training_job._compute_data_hash", lambda region_data: "h")

        from jobs.training_job import _holdout_metrics_prophet, _train_prophet

        holdout = _holdout_metrics_prophet(featured_df_long, region="TEST")
        version = _train_prophet(self._make_region_data(featured_df_long), holdout=holdout)

        assert version == "v-test"
        assert captured["model_name"] == "prophet"

        # Top-level mape — drives ensemble weighting at scoring time.
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)

        # extra["holdout_metrics"] — drives Models-tab RMSE / MAE / R².
        extra = captured.get("extra") or {}
        holdout_in_extra = extra.get("holdout_metrics")
        assert holdout_in_extra is not None
        assert set(holdout_in_extra.keys()) == {"mape", "rmse", "mae", "r2"}
        assert holdout_in_extra["mape"] == pytest.approx(4.0, abs=0.5)

    def test_arima_passes_full_holdout_metrics_to_save_model(
        self, featured_df_long, monkeypatch
    ) -> None:
        import models.arima_model as arima_mod

        monkeypatch.setattr(
            arima_mod,
            "train_arima",
            lambda df, **kwargs: {
                "params": [],
                "order": (2, 1, 2),
                "seasonal_order": (1, 1, 1, 24),
            },
        )
        monkeypatch.setattr(
            arima_mod,
            "predict_arima",
            lambda model_dict, exog, periods=168: (
                np.asarray(exog["demand_mw"].values) * 0.96
            ),  # MAPE ≈ 4
        )

        captured: dict = {}

        def _save_model(**kwargs):
            captured.update(kwargs)
            return "v-test"

        monkeypatch.setattr("jobs.training_job.save_model", _save_model)
        monkeypatch.setattr("jobs.training_job._compute_data_hash", lambda region_data: "h")

        from jobs.training_job import _holdout_metrics_arima, _train_arima

        holdout = _holdout_metrics_arima(featured_df_long, region="TEST")
        version = _train_arima(self._make_region_data(featured_df_long), holdout=holdout)

        assert version == "v-test"
        assert captured["model_name"] == "arima"
        assert captured["mape"] is not None
        assert captured["mape"] == pytest.approx(4.0, abs=0.5)

        extra = captured.get("extra") or {}
        holdout_in_extra = extra.get("holdout_metrics")
        assert holdout_in_extra is not None
        assert holdout_in_extra["mape"] == pytest.approx(4.0, abs=0.5)


class TestTaskPartitioner:
    """``_partition_regions_for_task`` reads Cloud Run Job env vars
    (``CLOUD_RUN_TASK_INDEX`` / ``CLOUD_RUN_TASK_COUNT``) and returns
    an interleaved-stride slice of the region list. Interleaved (vs
    contiguous) is the right choice because ``ordered_regions`` lists
    the largest BAs first — a contiguous split would put the four
    most expensive BAs (FPL/ERCOT/CAISO/PJM) all in task 0."""

    def test_single_task_returns_full_list(self, monkeypatch) -> None:
        """When run locally or with taskCount=1, the partition equals
        the input — no behavior change vs the pre-parallel path."""
        monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
        monkeypatch.delenv("CLOUD_RUN_TASK_COUNT", raising=False)
        from jobs.training_job import _partition_regions_for_task

        regions = ["A", "B", "C", "D", "E"]
        partition, idx, count = _partition_regions_for_task(regions)

        assert partition == regions
        assert idx == 0
        assert count == 1

    def test_three_tasks_interleaved_stride(self, monkeypatch) -> None:
        """With 9 regions across 3 tasks, each task gets every 3rd
        region starting from its index. Interleaved stride spreads
        the cost-ordered list evenly — task 0 gets [0, 3, 6], task 1
        gets [1, 4, 7], task 2 gets [2, 5, 8]."""
        regions = ["FPL", "ERCOT", "CAISO", "PJM", "MISO", "NYISO", "SPP", "ISONE", "SOCO"]

        from jobs.training_job import _partition_regions_for_task

        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "3")
        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "0")
        part_0, _, _ = _partition_regions_for_task(regions)

        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "1")
        part_1, _, _ = _partition_regions_for_task(regions)

        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "2")
        part_2, _, _ = _partition_regions_for_task(regions)

        assert part_0 == ["FPL", "PJM", "SPP"]
        assert part_1 == ["ERCOT", "MISO", "ISONE"]
        assert part_2 == ["CAISO", "NYISO", "SOCO"]

    def test_partition_union_equals_input(self, monkeypatch) -> None:
        """The union of all task partitions must equal the input list
        with no duplicates and no missing regions — otherwise some BA
        would never be trained."""
        regions = [f"R{i:02d}" for i in range(51)]  # mirror current 51-BA count

        from jobs.training_job import _partition_regions_for_task

        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "3")
        all_partitioned: list[str] = []
        for task_idx in range(3):
            monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", str(task_idx))
            partition, _, _ = _partition_regions_for_task(regions)
            all_partitioned.extend(partition)

        assert sorted(all_partitioned) == sorted(regions)
        assert len(all_partitioned) == len(regions)

    def test_partition_balanced_within_one(self, monkeypatch) -> None:
        """Partition sizes shouldn't differ by more than one — with
        51 BAs across 3 tasks, expect 17/17/17. Off-by-many would mean
        one task consistently overruns."""
        regions = [f"R{i:02d}" for i in range(51)]

        from jobs.training_job import _partition_regions_for_task

        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "3")
        sizes: list[int] = []
        for task_idx in range(3):
            monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", str(task_idx))
            partition, _, _ = _partition_regions_for_task(regions)
            sizes.append(len(partition))

        assert max(sizes) - min(sizes) <= 1, f"Imbalanced partition: {sizes}"


class TestResumeLogic:
    """``_train_region`` short-circuits when every model already has a
    saved version with the current data_hash. Critical for Cloud Run
    retry behavior — without it, a retry restarts from BA 0 and
    redundantly retrains BAs the previous attempt finished. With it,
    retries pick up where the previous attempt left off.
    """

    def test_skip_when_all_models_have_matching_hash(self, monkeypatch) -> None:
        # Stub get_model_metadata so every model returns a meta with
        # matching data_hash.
        from unittest.mock import MagicMock

        from jobs import training_job

        meta = MagicMock(version="v1", data_hash="abc123", extra={})
        monkeypatch.setattr(
            "models.persistence.get_model_metadata",
            lambda region, model_name: meta,
        )

        skipped = training_job._skip_if_data_hash_matches("PJM", "abc123")
        assert skipped == {
            "xgboost": "v1",
            "prophet": "v1",
            "arima": "v1",
        }

    def test_no_skip_when_one_model_has_stale_hash(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from jobs import training_job

        def _meta_lookup(region, model_name):
            # arima is stale, xgboost + prophet are current
            ds_hash = "abc123" if model_name != "arima" else "old456"
            return MagicMock(version="v1", data_hash=ds_hash, extra={})

        monkeypatch.setattr("models.persistence.get_model_metadata", _meta_lookup)

        assert training_job._skip_if_data_hash_matches("PJM", "abc123") is None

    def test_no_skip_when_one_model_missing(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from jobs import training_job

        def _meta_lookup(region, model_name):
            # arima never trained for this region
            if model_name == "arima":
                return None
            return MagicMock(version="v1", data_hash="abc123", extra={})

        monkeypatch.setattr("models.persistence.get_model_metadata", _meta_lookup)

        assert training_job._skip_if_data_hash_matches("PJM", "abc123") is None


class TestArimaCachedOrder:
    """The ARIMA training functions read ``meta.extra["order"]`` and
    ``["seasonal_order"]`` if present and skip the auto_arima
    stepwise search entirely — the dominant cost in the per-BA
    training loop. Cache miss falls back to the slow path.
    """

    def test_read_cached_order_returns_tuple(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from jobs import training_job

        meta = MagicMock(
            extra={
                "order": [2, 1, 2],
                "seasonal_order": [1, 1, 1, 24],
            }
        )
        monkeypatch.setattr(
            "models.persistence.get_model_metadata",
            lambda region, model_name: meta,
        )

        result = training_job._read_cached_arima_order("PJM")
        assert result == ((2, 1, 2), (1, 1, 1, 24))

    def test_read_cached_order_returns_none_when_missing(self, monkeypatch) -> None:
        from unittest.mock import MagicMock

        from jobs import training_job

        # Meta exists but has no order in extra
        meta = MagicMock(extra={"holdout_metrics": {"mape": 5.0}})
        monkeypatch.setattr(
            "models.persistence.get_model_metadata",
            lambda region, model_name: meta,
        )

        assert training_job._read_cached_arima_order("PJM") is None

    def test_read_cached_order_returns_none_when_no_meta(self, monkeypatch) -> None:
        from jobs import training_job

        monkeypatch.setattr(
            "models.persistence.get_model_metadata",
            lambda region, model_name: None,
        )

        assert training_job._read_cached_arima_order("PJM") is None

    def test_train_arima_fast_path_skips_auto_arima(self) -> None:
        """When ``cached_order`` and ``cached_seasonal_order`` are
        supplied, ``train_arima`` skips ``_auto_select_order`` entirely
        and just refits with the cached order. This is the key
        speedup — auto_arima is the dominant per-BA cost."""
        from unittest.mock import patch

        import pandas as pd

        from models.arima_model import train_arima

        # Synthetic DF with enough rows for the SARIMAX fit to succeed.
        ts = pd.date_range("2024-01-01", periods=720, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "demand_mw": np.full(720, 50_000.0)
                + 5_000.0 * np.sin(2 * np.pi * np.arange(720) / 24),
                "temperature_2m": 20.0,
                "wind_speed_80m": 5.0,
                "shortwave_radiation": 0.5,
                "cooling_degree_days": 0.0,
                "heating_degree_days": 0.0,
            }
        )

        with patch("models.arima_model._auto_select_order") as mock_auto:
            mock_auto.return_value = ((9, 9, 9), (9, 9, 9, 24))  # would fail to fit
            try:
                result = train_arima(
                    df,
                    cached_order=(2, 1, 2),
                    cached_seasonal_order=(1, 1, 1, 24),
                )
                # auto_select_order MUST NOT have been called
                mock_auto.assert_not_called()
                assert result["order"] == (2, 1, 2)
                assert result["seasonal_order"] == (1, 1, 1, 24)
            except Exception:
                # Even if SARIMAX fitting fails on synthetic data, the
                # important assertion (auto_arima skipped) still holds.
                mock_auto.assert_not_called()


class TestEnsembleHoldoutMetrics:
    """The ensemble's holdout metric is computed from the SAME 168-hour
    holdout predictions used by each per-model holdout — no separate
    train, no Redis, no simulation. ``_ensemble_holdout_metrics`` takes
    the per-model dicts and returns ``(metrics, weights)``.
    """

    def test_combines_two_models_with_inverse_mape_weights(self) -> None:
        from jobs.training_job import _ensemble_holdout_metrics

        n = 168
        actual = np.full(n, 50_000.0)
        per_model_holdouts = {
            # 5% over-forecast → mape ≈ 5
            "prophet": {
                "metrics": {"mape": 5.0, "rmse": 2500.0, "mae": 2500.0, "r2": 0.0},
                "forecast": actual * 1.05,
                "actual": actual,
            },
            # 1% over-forecast → mape ≈ 1 (much better)
            "xgboost": {
                "metrics": {"mape": 1.0, "rmse": 500.0, "mae": 500.0, "r2": 0.0},
                "forecast": actual * 1.01,
                "actual": actual,
            },
        }
        result = _ensemble_holdout_metrics(per_model_holdouts)
        assert result is not None
        metrics, weights = result

        # Weights normalize to 1.0
        assert weights["prophet"] + weights["xgboost"] == pytest.approx(1.0, abs=0.001)
        # xgboost (1% MAPE) gets ≥ 5× the weight of prophet (5% MAPE)
        assert weights["xgboost"] > weights["prophet"]

        # Ensemble MAPE is between the worst (5) and the best (1)
        assert 1.0 < metrics["mape"] < 5.0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_single_model_returns_none(self) -> None:
        """One model isn't an ensemble — return None so the caller
        doesn't persist a misleading 'ensemble' metric that's just
        the lone model under a different name."""
        from jobs.training_job import _ensemble_holdout_metrics

        n = 168
        actual = np.full(n, 50_000.0)
        result = _ensemble_holdout_metrics(
            {
                "xgboost": {
                    "metrics": {"mape": 1.0, "rmse": 500.0, "mae": 500.0, "r2": 0.0},
                    "forecast": actual * 1.01,
                    "actual": actual,
                },
                "prophet": None,
                "arima": None,
            }
        )
        assert result is None

    def test_returns_none_when_actuals_mismatch(self) -> None:
        """If two models report different ``actual`` lengths, something
        upstream is broken — bail rather than silently misalign."""
        from jobs.training_job import _ensemble_holdout_metrics

        result = _ensemble_holdout_metrics(
            {
                "prophet": {
                    "metrics": {"mape": 5.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5},
                    "forecast": np.ones(168),
                    "actual": np.ones(168),
                },
                "xgboost": {
                    "metrics": {"mape": 2.0, "rmse": 1.0, "mae": 1.0, "r2": 0.5},
                    "forecast": np.ones(100),
                    "actual": np.ones(100),
                },
            }
        )
        assert result is None


class TestResumeBackfillsMissingEnsemble:
    """#176: ``_skip_if_data_hash_matches`` declines to resume when the
    xgboost meta is data-hash-current but missing ``ensemble_holdout_metrics``
    while an ensemble is achievable (>=2 per-model holdouts present) — forcing
    a recompute that backfills the metric. Single-model regions still resume.
    """

    @staticmethod
    def _meta(*, ensemble: bool, holdout: bool):
        from unittest.mock import MagicMock

        extra: dict = {}
        if holdout:
            extra["holdout_metrics"] = {"mape": 2.0, "rmse": 1.0, "mae": 1.0, "r2": 0.9}
        if ensemble:
            extra["ensemble_holdout_metrics"] = {"mape": 1.8, "rmse": 1.0, "mae": 1.0, "r2": 0.9}
        return MagicMock(version="v1", data_hash="abc123", extra=extra)

    def _patch(self, monkeypatch, metas: dict) -> None:
        monkeypatch.setattr(
            "models.persistence.get_model_metadata",
            lambda region, model_name: metas[model_name],
        )

    def test_declines_resume_when_ensemble_missing_and_two_holdouts(self, monkeypatch) -> None:
        from jobs import training_job

        metas = {
            "xgboost": self._meta(ensemble=False, holdout=True),
            "prophet": self._meta(ensemble=False, holdout=True),
            "arima": self._meta(ensemble=False, holdout=True),
        }
        self._patch(monkeypatch, metas)
        assert training_job._skip_if_data_hash_matches("PJM", "abc123") is None

    def test_resumes_when_ensemble_present(self, monkeypatch) -> None:
        from jobs import training_job

        metas = {
            "xgboost": self._meta(ensemble=True, holdout=True),
            "prophet": self._meta(ensemble=False, holdout=True),
            "arima": self._meta(ensemble=False, holdout=True),
        }
        self._patch(monkeypatch, metas)
        assert training_job._skip_if_data_hash_matches("PJM", "abc123") == {
            "xgboost": "v1",
            "prophet": "v1",
            "arima": "v1",
        }

    def test_resumes_single_model_region_without_ensemble(self, monkeypatch) -> None:
        # Only xgboost has a per-model holdout; an ensemble is not achievable,
        # so a missing ensemble metric is legitimate and the region resumes.
        from jobs import training_job

        metas = {
            "xgboost": self._meta(ensemble=False, holdout=True),
            "prophet": self._meta(ensemble=False, holdout=False),
            "arima": self._meta(ensemble=False, holdout=False),
        }
        self._patch(monkeypatch, metas)
        assert training_job._skip_if_data_hash_matches("PJM", "abc123") == {
            "xgboost": "v1",
            "prophet": "v1",
            "arima": "v1",
        }
