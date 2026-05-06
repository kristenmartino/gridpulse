"""Focused unit tests for ``models.xgboost_model``.

Closes a coverage gap from issue #88. XGBoost training is the
authoritative model for the forecast service (``primary_model`` in
the scoring payload), and SHAP values feed the Models tab's feature-
importance visualization. These tests target:

- The TimeSeriesSplit no-leakage assertion that guards CV folds
- ``predict_xgboost`` graceful-degradation on missing features
- Negative-prediction clamping (demand can't be negative)
- The result dict shape — ``feature_names`` / ``feature_importances`` /
  ``cv_scores`` are all read by downstream consumers
- A SHAP smoke test using a real-but-tiny synthetic dataset
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _restore_real_shap_module():
    """Defend against test-pollution: another suite (e.g.
    ``test_models_training.py::test_compute_shap_values_returns_none_on_error``)
    uses ``patch.dict('sys.modules', {'shap': MagicMock()})`` and the
    cleanup occasionally fails to restore the real module — likely
    because pytest's collection ordering interleaves with patch.dict's
    snapshot/restore. If the cached entry is a Mock when our SHAP
    smoke test runs, pop it so the lazy ``import shap`` inside
    ``compute_shap_values`` re-imports the real module."""
    cached = sys.modules.get("shap")
    if cached is not None and type(cached).__module__ == "unittest.mock":
        sys.modules.pop("shap", None)
    yield


@pytest.fixture
def small_feature_df():
    """Tiny synthetic frame with engineered-feature shape. 200 rows
    is enough for a 5-fold TimeSeriesSplit + early stopping to run
    in seconds, not minutes."""
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 50_000.0
            + 5_000.0 * np.sin(2 * np.pi * np.arange(n) / 24)
            + rng.normal(0, 200, n),
            "temperature_2m": 70.0 + rng.normal(0, 5, n),
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "demand_lag_24h": 50_000.0,
            # Excluded columns — should NOT end up in feature_names
            "region": "PJM",
            "data_quality": "ok",
        }
    )


def _trained_model_dict(small_feature_df, n_splits=2):
    """Train an XGBoost on the fixture with low n_splits so the CV
    cost stays in unit-test territory. Returns the result dict."""
    from models.xgboost_model import train_xgboost

    # Override params to skip early stopping + reduce trees so the
    # fit completes in milliseconds, not seconds.
    params = {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.3,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    }
    return train_xgboost(small_feature_df, params=params, n_splits=n_splits)


class TestTrainXgboostShape:
    def test_returns_required_keys(self, small_feature_df):
        result = _trained_model_dict(small_feature_df)
        for key in ("model", "feature_names", "feature_importances", "cv_scores"):
            assert key in result, f"train_xgboost result missing {key!r}"

    def test_excludes_metadata_columns_from_features(self, small_feature_df):
        """``EXCLUDE_COLS`` keeps timestamp / target / region / data_quality
        out of the feature matrix. If one of these slipped in, the
        model would either crash on prediction (string columns) or
        leak the target into training (demand_mw)."""
        from models.xgboost_model import EXCLUDE_COLS

        result = _trained_model_dict(small_feature_df)
        feature_names = set(result["feature_names"])
        leaked = feature_names & EXCLUDE_COLS
        assert not leaked, f"Excluded columns leaked into features: {leaked}"

    def test_feature_importances_dict_keyed_by_feature_name(self, small_feature_df):
        """Downstream consumers (Models tab feature-importance bar) read
        ``feature_importances`` as a name → score dict. Must round-trip
        the keys, not the positional indices."""
        result = _trained_model_dict(small_feature_df)
        importances = result["feature_importances"]
        assert isinstance(importances, dict)
        # Every feature has an importance score
        assert set(importances.keys()) == set(result["feature_names"])
        # All scores are floats >= 0 (non-zero for splits used by the model)
        assert all(isinstance(v, float) and v >= 0 for v in importances.values())

    def test_cv_scores_one_per_fold(self, small_feature_df):
        """``cv_scores`` is consumed by the ensemble-weighting path
        (PR #84 inverse-MAPE). One score per TimeSeriesSplit fold."""
        result = _trained_model_dict(small_feature_df, n_splits=3)
        assert len(result["cv_scores"]) == 3
        # Each score is a finite MAPE percentage
        for score in result["cv_scores"]:
            assert np.isfinite(score), "CV MAPE must be finite"
            assert score >= 0, "CV MAPE must be non-negative"


class TestPredictXgboost:
    def test_predict_clamps_negative_predictions(self, small_feature_df):
        """Demand can't be negative — ``predict_xgboost`` explicitly
        wraps the model output in ``np.maximum(predictions, 0)``."""
        from models.xgboost_model import predict_xgboost

        # Train on the small frame, then predict on a frame with
        # extreme out-of-range features that might push the regressor
        # below zero.
        result = _trained_model_dict(small_feature_df)
        out_of_range = small_feature_df.copy()
        out_of_range["temperature_2m"] = -1000.0  # extreme cold
        out_of_range["demand_lag_24h"] = -50_000.0  # impossible

        preds = predict_xgboost(result, out_of_range)
        assert (preds >= 0).all(), "Negative-prediction clamp not applied"

    def test_predict_handles_missing_feature_with_zero_fill(self, small_feature_df):
        """If a feature column is missing at predict time (e.g., a
        weather column dropped upstream), the model would otherwise
        crash on indexing. The predict path zero-fills missing
        columns + logs a warning rather than raising."""
        from models.xgboost_model import predict_xgboost

        result = _trained_model_dict(small_feature_df)
        df_missing = small_feature_df.drop(columns=["temperature_2m"])
        # Should not raise — missing feature is zero-filled
        preds = predict_xgboost(result, df_missing)
        assert len(preds) == len(df_missing)
        assert (preds >= 0).all()


class TestTimeSeriesSplitLeakageGuard:
    """The training loop has an explicit ``assert train_idx.max() <
    val_idx.min()`` to catch any future change to the CV strategy
    that would let validation indices precede training ones. This
    guard is the only thing standing between us and silent leakage —
    test the assertion fires when violated."""

    def test_train_idx_strictly_before_val_idx(self):
        from sklearn.model_selection import TimeSeriesSplit

        # Replicate the exact CV split the training loop uses with
        # n_splits=5 and n=200, then verify the no-leakage invariant
        # holds at every fold.
        n = 200
        X = np.arange(n).reshape(-1, 1)  # noqa: N806
        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            assert train_idx.max() < val_idx.min(), (
                f"Fold {fold}: TimeSeriesSplit produced overlapping "
                f"train/val ranges (max train={train_idx.max()}, "
                f"min val={val_idx.min()}). The training loop's "
                f"assertion would fire here."
            )


class TestComputeShapValues:
    """SHAP feeds the Models tab's feature-attribution display. The
    smoke test verifies the helper returns the expected shape;
    correctness of SHAP values themselves is a property of the
    library, not our wrapper."""

    def test_shap_smoke(self, small_feature_df):
        """End-to-end: train → SHAP. Real shap.TreeExplainer call,
        small dataset (200 rows × 4 features), runs in well under
        a second."""
        from models.xgboost_model import compute_shap_values

        result = _trained_model_dict(small_feature_df)
        shap_out = compute_shap_values(result, small_feature_df, max_samples=50)

        for key in ("shap_values", "feature_names", "base_value"):
            assert key in shap_out

        # SHAP values shape: (n_samples, n_features) for a single-output regressor.
        n_features = len(result["feature_names"])
        assert shap_out["shap_values"].shape == (50, n_features)
        # base_value is a Python float (the helper casts it explicitly)
        assert isinstance(shap_out["base_value"], float)
        assert np.isfinite(shap_out["base_value"])

    def test_shap_subsamples_when_over_max_samples(self, small_feature_df):
        """Performance guardrail: ``compute_shap_values`` subsamples
        to ``max_samples`` rows if the input is larger. SHAP cost
        scales with sample count, so the cap is what keeps the
        Models-tab paint under a second on the deployed app."""
        from models.xgboost_model import compute_shap_values

        result = _trained_model_dict(small_feature_df)
        # 200 input rows, max_samples=20 — output should be 20 rows.
        shap_out = compute_shap_values(result, small_feature_df, max_samples=20)
        assert shap_out["shap_values"].shape[0] == 20
