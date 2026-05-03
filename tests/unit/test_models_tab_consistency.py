"""Regression tests: Models tab leaderboard + diagnostics table read
the same metrics source.

User-reported bugs 2026-05-02:

1. **Leaderboard / table mismatch**: top row of MAPEs didn't match
   the table below. Root cause: leaderboard called
   ``get_model_metrics`` which fell through to a hardcoded simulated
   dict; table read the diagnostics Redis payload (different
   simulated path). Both ended up "fake" but via different chains.

2. **Both surfaces showed simulated MAPE values, not real**: even
   after the Cloud Run training job writes real holdout MAPE per
   model to GCS at ``cache/models/{region}/{model}/{version}.meta.json``
   (V0.2 / V3.ε), neither surface read those values.

Fix: ``get_model_metrics`` is now meta.json-first. It reads the real
holdout MAPE from each model's pickle metadata, supplements
RMSE/MAE/R² (and the ensemble MAPE) from the Redis diagnostics
payload, and falls back to local pickles + simulated only when
nothing else is available. ``_models_tab_from_redis`` and the v1
fallback both call ``get_model_metrics`` so leaderboard + table can't
diverge.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _meta(model_name, mape, region="PJM"):
    """Build a minimal ``ModelMetadata`` for the patch."""
    from models.persistence import ModelMetadata

    return ModelMetadata(
        region=region,
        model_name=model_name,
        version="v-test",
        data_hash="h",
        trained_at="2026-05-01T04:00:00+00:00",
        train_rows=8000,
        mape=mape,
        lib_versions={},
        extra={},
    )


def _meta_lookup(mape_by_model):
    """Returns a stub ``get_model_metadata(region, model_name)``
    function backed by ``{model_name: mape}``."""

    def _stub(region, model_name):
        if model_name in mape_by_model:
            return _meta(model_name, mape_by_model[model_name], region)
        return None

    return _stub


_REDIS_PAYLOAD_FAKE_METRICS = {
    "metrics": {
        # Same shape, different values — what `_simulate_forecasts`
        # would yield for some particular seed
        "prophet": {"mape": 4.20, "rmse": 555, "mae": 410, "r2": 0.91},
        "arima": {"mape": 5.55, "rmse": 700, "mae": 525, "r2": 0.88},
        "xgboost": {"mape": 3.30, "rmse": 480, "mae": 360, "r2": 0.94},
        "ensemble": {"mape": 2.94, "rmse": 460, "mae": 340, "r2": 0.95},
    },
    "timestamps": ["2026-05-02T00:00:00"],
    "ensemble": [50000.0],
    "residuals": [100.0],
    "hourly_error": {h: 1.0 for h in range(24)},
    "feature_importance": {"temperature_2m": 0.5},
}


class TestRealMapeSource:
    """The user's question: 'shoot those MAPE values are fake?' —
    after this PR they're not. Real holdout MAPE per model comes from
    each pickle's GCS meta.json (V0.2 / V3.ε wrote them)."""

    def test_real_mape_overrides_simulated(self):
        """Real holdout MAPE in meta.json wins over Redis simulated values."""
        from models.model_service import get_model_metrics

        real_mapes = {"prophet": 11.04, "arima": 5.19, "xgboost": 1.10}
        with (
            patch(
                "models.persistence.get_model_metadata",
                side_effect=_meta_lookup(real_mapes),
            ),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        assert got["prophet"]["mape"] == 11.04  # real, not 4.20 simulated
        assert got["arima"]["mape"] == 5.19  # real, not 5.55 simulated
        assert got["xgboost"]["mape"] == 1.10  # real, not 3.30 simulated

    def test_redis_supplies_rmse_mae_r2_when_meta_lacks_them(self):
        """Meta.json has only MAPE; RMSE/MAE/R² come from Redis."""
        from models.model_service import get_model_metrics

        real_mapes = {"prophet": 11.04, "arima": 5.19, "xgboost": 1.10}
        with (
            patch(
                "models.persistence.get_model_metadata",
                side_effect=_meta_lookup(real_mapes),
            ),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        assert got["xgboost"]["rmse"] == 480  # from Redis
        assert got["xgboost"]["mae"] == 360
        assert got["xgboost"]["r2"] == 0.94

    def test_ensemble_mape_from_redis(self):
        """Ensemble has no per-model meta.json — its MAPE must come
        from the Redis diagnostics payload."""
        from models.model_service import get_model_metrics

        real_mapes = {"prophet": 11.04, "arima": 5.19, "xgboost": 1.10}
        with (
            patch(
                "models.persistence.get_model_metadata",
                side_effect=_meta_lookup(real_mapes),
            ),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        assert got["ensemble"]["mape"] == 2.94  # Redis ensemble MAPE
        assert got["ensemble"]["rmse"] == 460


class TestResolutionOrderFallthrough:
    def test_redis_only_when_meta_missing(self):
        from models.model_service import get_model_metrics

        with (
            patch("models.persistence.get_model_metadata", return_value=None),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        # No meta → Redis values for everything
        assert got["xgboost"]["mape"] == 3.30  # Redis
        assert got["ensemble"]["mape"] == 2.94

    def test_simulated_when_everything_missing(self):
        from models.model_service import get_model_metrics

        with (
            patch("models.persistence.get_model_metadata", return_value=None),
            patch("data.redis_client.redis_get", return_value=None),
            patch("models.model_service._load_cached_models", return_value=None),
        ):
            got = get_model_metrics("PJM")

        # Hardcoded last-resort defaults
        assert got["xgboost"]["mape"] == 2.1
        assert got["ensemble"]["mape"] == 1.9

    def test_meta_with_null_mape_falls_through(self):
        """A model whose meta.json has ``mape=None`` (training failed
        to record) shouldn't show up in the output unless Redis or
        simulated layer fills it in."""
        from models.model_service import get_model_metrics

        # Only xgboost has real MAPE; prophet/arima have null in meta
        def _stub(region, model_name):
            return (
                _meta(model_name, mape=None)
                if model_name != "xgboost"
                else _meta(model_name, mape=1.10)
            )

        with (
            patch("models.persistence.get_model_metadata", side_effect=_stub),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        assert got["xgboost"]["mape"] == 1.10  # real
        # prophet / arima fall through to Redis
        assert got["prophet"]["mape"] == 4.20
        assert got["arima"]["mape"] == 5.55

    def test_meta_with_inf_mape_falls_through(self):
        """``inf`` MAPE means the holdout failed; treat as missing."""
        from models.model_service import get_model_metrics

        with (
            patch(
                "models.persistence.get_model_metadata",
                side_effect=_meta_lookup({"xgboost": float("inf")}),
            ),
            patch("data.redis_client.redis_get", return_value=_REDIS_PAYLOAD_FAKE_METRICS),
        ):
            got = get_model_metrics("PJM")

        # inf → ignore; fall through to Redis
        assert got["xgboost"]["mape"] == 3.30


class TestLeaderboardTableConsistency:
    """The user-reported divergence — leaderboard and table can't show
    different MAPEs anymore because they both call ``get_model_metrics``."""

    def test_leaderboard_calls_get_model_metrics(self):
        from components import callbacks as cb

        spy = MagicMock(return_value={"xgboost": {"mape": 1.10}})
        with patch("models.model_service.get_model_metrics", spy):
            cb._build_models_leaderboard("PJM")
        spy.assert_called_once_with("PJM")

    def test_redis_path_table_uses_shared_metrics(self):
        """``_models_tab_from_redis`` reads metrics through
        ``get_model_metrics`` so it can't diverge from the leaderboard."""
        import inspect

        from components import callbacks as cb

        src = inspect.getsource(cb._models_tab_from_redis)
        assert "get_model_metrics(region)" in src

    def test_v1_fallback_uses_shared_metrics(self):
        """The v1 compute fallback in ``update_models_tab`` also reads
        through ``get_model_metrics`` so cold-Redis dev mode stays
        consistent with the (also-fallback) leaderboard."""
        import inspect

        from components import callbacks as cb

        src = inspect.getsource(cb.register_callbacks)
        assert "get_model_metrics(region) or forecasts.get" in src
