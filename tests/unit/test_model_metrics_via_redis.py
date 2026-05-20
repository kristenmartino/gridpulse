"""Tests for #131 — scoring job writes real holdout metrics to Redis;
web tier reads them from there.

Two surfaces covered:

1. ``jobs.scoring_job._extract_holdout_metrics`` /
   ``_extract_ensemble_metrics`` — pure helpers that pull metrics out
   of each model's ``ModelMetadata.extra`` blocks. These cover the
   "model has full holdout dict" path AND the "legacy pickle with only
   top-level meta.mape" path.

2. ``models.model_service.get_model_metrics`` Layer 0 — reads
   ``gridpulse:forecast:{region}:1h``'s ``model_metrics`` field and
   returns it when present. Falls through to existing layers 1-6
   when absent.

Plus the writer side (``jobs.phases.predict_and_write_forecast``):
the payload it writes to Redis includes the ``model_metrics`` field
when the caller passes one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _meta(
    *,
    mape: float | None = None,
    extra: dict | None = None,
) -> SimpleNamespace:
    """Minimal ModelMetadata stand-in for the extract helpers (they only
    use ``.mape`` and ``.extra``).
    """
    return SimpleNamespace(mape=mape, extra=extra or {})


class TestExtractHoldoutMetrics:
    def test_returns_empty_for_none_meta(self):
        from jobs.scoring_job import _extract_holdout_metrics

        assert _extract_holdout_metrics(None) == {}

    def test_returns_full_dict_from_extra(self):
        from jobs.scoring_job import _extract_holdout_metrics

        meta = _meta(
            mape=5.0,
            extra={
                "holdout_metrics": {
                    "mape": 4.21,
                    "rmse": 1234.5,
                    "mae": 890.1,
                    "r2": 0.967,
                }
            },
        )
        result = _extract_holdout_metrics(meta)
        assert result == {
            "mape": 4.21,
            "rmse": 1234.5,
            "mae": 890.1,
            "r2": 0.967,
        }

    def test_falls_back_to_top_level_mape_when_extra_missing(self):
        from jobs.scoring_job import _extract_holdout_metrics

        meta = _meta(mape=5.55, extra={})  # No holdout_metrics
        result = _extract_holdout_metrics(meta)
        assert result == {"mape": 5.55}

    def test_skips_nonfinite_values(self):
        from jobs.scoring_job import _extract_holdout_metrics

        meta = _meta(
            extra={
                "holdout_metrics": {
                    "mape": float("inf"),
                    "rmse": float("nan"),
                    "mae": 890.1,
                    "r2": None,
                }
            }
        )
        result = _extract_holdout_metrics(meta)
        # Only mae survives the finite check; mape is +inf, rmse is NaN
        assert result == {"mae": 890.1}

    def test_returns_empty_when_no_useful_data(self):
        from jobs.scoring_job import _extract_holdout_metrics

        meta = _meta(mape=None, extra={})
        assert _extract_holdout_metrics(meta) == {}

    def test_extra_holdout_takes_precedence_over_top_level(self):
        from jobs.scoring_job import _extract_holdout_metrics

        # Extra has the full dict + top-level has a stale mape.
        meta = _meta(
            mape=99.0,  # would be stale
            extra={"holdout_metrics": {"mape": 4.21, "rmse": 1234.5}},
        )
        result = _extract_holdout_metrics(meta)
        assert result["mape"] == 4.21  # From extra, not top-level
        assert result["rmse"] == 1234.5


class TestExtractEnsembleMetrics:
    def test_returns_empty_for_none(self):
        from jobs.scoring_job import _extract_ensemble_metrics

        assert _extract_ensemble_metrics(None) == {}

    def test_returns_empty_when_extra_missing(self):
        from jobs.scoring_job import _extract_ensemble_metrics

        meta = _meta(extra={})  # No ensemble_holdout_metrics
        assert _extract_ensemble_metrics(meta) == {}

    def test_reads_ensemble_metrics_from_xgb_meta_extra(self):
        from jobs.scoring_job import _extract_ensemble_metrics

        xgb_meta = _meta(
            extra={
                "ensemble_holdout_metrics": {
                    "mape": 3.85,
                    "rmse": 1100.0,
                    "mae": 780.0,
                    "r2": 0.978,
                }
            }
        )
        result = _extract_ensemble_metrics(xgb_meta)
        assert result == {
            "mape": 3.85,
            "rmse": 1100.0,
            "mae": 780.0,
            "r2": 0.978,
        }

    def test_skips_nonfinite(self):
        from jobs.scoring_job import _extract_ensemble_metrics

        xgb_meta = _meta(
            extra={
                "ensemble_holdout_metrics": {
                    "mape": 3.85,
                    "rmse": float("nan"),
                    "mae": float("inf"),
                    "r2": 0.978,
                }
            }
        )
        result = _extract_ensemble_metrics(xgb_meta)
        assert result == {"mape": 3.85, "r2": 0.978}


class TestGetModelMetricsLayerZero:
    """``get_model_metrics`` reads from Redis forecast payload first."""

    @patch("data.redis_client.redis_get")
    def test_returns_redis_metrics_when_present(self, mock_redis_get):
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = {
            "region": "FPL",
            "model_metrics": {
                "xgboost": {"mape": 4.21, "rmse": 1234.5, "mae": 890.1, "r2": 0.967},
                "prophet": {"mape": 7.88, "rmse": 1450.0},
                "ensemble": {"mape": 3.85, "rmse": 1100.0, "mae": 780.0, "r2": 0.978},
            },
        }

        result = get_model_metrics("FPL")
        assert result["xgboost"]["mape"] == 4.21
        assert result["xgboost"]["r2"] == 0.967
        assert result["prophet"]["mape"] == 7.88
        # Partial dict (no mae / r2) is fine — UI tolerates missing fields
        assert "mae" not in result["prophet"]
        assert result["ensemble"]["mape"] == 3.85

    @patch("data.redis_client.redis_get")
    def test_falls_through_when_redis_payload_empty(self, mock_redis_get):
        """No Redis hit at the forecast key → falls through to existing
        layers 1-6. With no meta.json files in test env, ends at layer 6."""
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = None  # cache miss

        result = get_model_metrics("FPL")
        # Falls to layer 6 hardcoded baseline — still returns *something*.
        # The point is it didn't crash and Layer 0 didn't short-circuit
        # with an empty dict.
        assert isinstance(result, dict)
        assert "ensemble" in result  # Layer 6 has ensemble

    @patch("data.redis_client.redis_get")
    def test_falls_through_when_model_metrics_field_absent(self, mock_redis_get):
        """Forecast payload exists but no model_metrics yet (e.g. pre-#131
        scoring tick still in Redis with its TTL). Layer 0 doesn't return,
        falls through to later layers."""
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = {
            "region": "FPL",
            "forecasts": [{"timestamp": "2026-05-20T06:00:00+00:00", "xgboost": 18000}],
            # No model_metrics field here
        }

        result = get_model_metrics("FPL")
        # Falls to a later layer → still returns SOMETHING (layer 6 at minimum).
        assert isinstance(result, dict)

    @patch("data.redis_client.redis_get")
    def test_skips_nonfinite_metrics(self, mock_redis_get):
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = {
            "model_metrics": {
                "xgboost": {
                    "mape": float("nan"),  # skipped
                    "rmse": float("inf"),  # skipped
                    "mae": 890.1,  # kept
                    "r2": 0.967,  # kept
                }
            },
        }

        result = get_model_metrics("FPL")
        # Layer 0 returns the xgboost row with only the finite fields.
        # Other layers may or may not contribute further, but our
        # xgboost row should reflect Layer 0's cleaning.
        assert "xgboost" in result
        assert "mae" in result["xgboost"]
        assert "r2" in result["xgboost"]
        # Verify the NaN/Inf were dropped (specifically that NaN doesn't
        # bleed through as a "real" 0).
        assert "mape" not in result["xgboost"]
        assert "rmse" not in result["xgboost"]

    @patch("data.redis_client.redis_get")
    def test_skips_malformed_model_entries(self, mock_redis_get):
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = {
            "model_metrics": {
                "xgboost": {"mape": 4.21},  # valid
                "prophet": "not a dict",  # malformed — should be skipped
                "arima": {},  # empty — should be skipped (no finite fields)
                "ensemble": {"mape": 3.85},  # valid
            }
        }

        result = get_model_metrics("FPL")
        assert "xgboost" in result
        assert "ensemble" in result
        # prophet/arima rows dropped at Layer 0 (malformed/empty); they
        # may still be filled by later layers but the Layer 0 path is
        # clean.

    @patch("data.redis_client.redis_get")
    def test_reads_correct_redis_key(self, mock_redis_get):
        from models.model_service import get_model_metrics

        mock_redis_get.return_value = None  # cache miss, but check what was queried
        get_model_metrics("ERCOT")

        # The FIRST Redis call should be for the forecast key (Layer 0).
        # Later layers may also call Redis for diagnostics.
        first_call = mock_redis_get.call_args_list[0]
        assert first_call.args[0] == "gridpulse:forecast:ERCOT:1h"


class TestPredictAndWriteForecastIncludesMetrics:
    """The scoring-job writer persists ``model_metrics`` to Redis."""

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get")
    def test_model_metrics_included_in_payload(self, _mock_get, mock_redis_set):
        """When model_metrics is passed to predict_and_write_forecast,
        the Redis payload includes a sanitized ``model_metrics`` field."""
        import numpy as np
        import pandas as pd

        from jobs.phases import RegionData, predict_and_write_forecast

        # Build a minimal RegionData with featured_df that the predict
        # path can use. We're testing the *payload shape*, not the
        # model_predict mechanics, so we mock the underlying _predict_one
        # to return canned predictions.
        ts = pd.date_range("2026-05-19", periods=200, freq="h", tz="UTC")
        featured_df = pd.DataFrame({"timestamp": ts, "demand_mw": np.full(200, 18000.0)})
        # Add some columns that _build_future_feature_frame expects
        featured_df["hour"] = featured_df["timestamp"].dt.hour
        featured_df["day_of_week"] = featured_df["timestamp"].dt.dayofweek
        featured_df["month"] = featured_df["timestamp"].dt.month
        featured_df["day_of_year"] = featured_df["timestamp"].dt.dayofyear

        data = RegionData(
            region="FPL",
            demand_df=featured_df[["timestamp", "demand_mw"]],
            weather_df=pd.DataFrame(),
            featured_df=featured_df,
        )

        # Patch the actual prediction so we don't need a real model
        with patch("jobs.phases._predict_one") as mock_predict:
            mock_predict.return_value = np.full(720, 18000.0)
            predict_and_write_forecast(
                data,
                models={"xgboost": object(), "prophet": object()},
                model_mapes={"xgboost": 4.21, "prophet": 7.88},
                model_metrics={
                    "xgboost": {
                        "mape": 4.21,
                        "rmse": 1234.5,
                        "mae": 890.1,
                        "r2": 0.967,
                    },
                    "prophet": {"mape": 7.88, "rmse": 1450.0},
                    "ensemble": {"mape": 3.85},
                },
            )

        # Inspect the payload that was written
        assert mock_redis_set.call_count == 1
        write_args = mock_redis_set.call_args
        key = write_args.args[0]
        payload = write_args.args[1]
        assert key == "gridpulse:forecast:FPL:1h"
        assert "model_metrics" in payload
        # Sanitized — values preserved
        assert payload["model_metrics"]["xgboost"]["mape"] == 4.21
        assert payload["model_metrics"]["ensemble"]["mape"] == 3.85

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get")
    def test_payload_omits_model_metrics_when_none_provided(self, _mock_get, mock_redis_set):
        """No model_metrics → field is absent (not None, not empty dict).
        Lets get_model_metrics's Layer 0 fall through cleanly."""
        import numpy as np
        import pandas as pd

        from jobs.phases import RegionData, predict_and_write_forecast

        ts = pd.date_range("2026-05-19", periods=200, freq="h", tz="UTC")
        featured_df = pd.DataFrame({"timestamp": ts, "demand_mw": np.full(200, 18000.0)})
        featured_df["hour"] = featured_df["timestamp"].dt.hour
        featured_df["day_of_week"] = featured_df["timestamp"].dt.dayofweek
        featured_df["month"] = featured_df["timestamp"].dt.month
        featured_df["day_of_year"] = featured_df["timestamp"].dt.dayofyear

        data = RegionData(
            region="FPL",
            demand_df=featured_df[["timestamp", "demand_mw"]],
            weather_df=pd.DataFrame(),
            featured_df=featured_df,
        )

        with patch("jobs.phases._predict_one") as mock_predict:
            mock_predict.return_value = np.full(720, 18000.0)
            predict_and_write_forecast(
                data,
                models={"xgboost": object()},
                model_mapes=None,
                model_metrics=None,
            )

        payload = mock_redis_set.call_args.args[1]
        assert "model_metrics" not in payload
