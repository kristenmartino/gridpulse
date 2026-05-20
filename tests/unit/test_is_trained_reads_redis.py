"""Tests for the updated ``models.model_service.is_trained`` — 2026-05-20.

Bug: ``is_trained(region)`` used to check for ``trained_models/{region}_models.pkl``
on local disk. In production the Cloud Run **Service** container has no
such directory (trained pickles live only on the Cloud Run **Job**
container), so ``is_trained`` returned False for every region every time.

Surface impact: the Overview model card's ``[trained] / [simulated]``
badge was always "[simulated]" in production, even when the chart and
the Forecast tab were correctly rendering real trained-model output
from Redis.

The fix shifts the primary signal to Redis: ``gridpulse:forecast:{region}:1h``
with a populated ``ensemble_weights`` (or any per-model column populated)
indicates real trained-model output is live for that region.

These tests pin the new behavior and lock the architectural pattern.
"""

from __future__ import annotations

from unittest.mock import patch


class TestIsTrainedReadsRedis:
    """Primary path: Redis forecast with ensemble_weights → trained."""

    @patch("data.redis_client.redis_get")
    def test_ensemble_weights_present_returns_true(self, mock_redis_get):
        from models.model_service import is_trained

        mock_redis_get.return_value = {
            "region": "FPL",
            "forecasts": [
                {
                    "timestamp": "2026-05-20T06:00:00+00:00",
                    "predicted_demand_mw": 18000.0,
                    "xgboost": 18000.0,
                    "ensemble": 18000.0,
                }
            ],
            "ensemble_weights": {"xgboost": 0.58, "prophet": 0.29, "arima": 0.13},
        }
        assert is_trained("FPL") is True

    @patch("data.redis_client.redis_get")
    def test_per_model_column_without_ensemble_weights_still_trained(self, mock_redis_get):
        """When only one model was loadable during scoring, ensemble_weights
        is absent but the per-model prediction is real — still trained."""
        from models.model_service import is_trained

        mock_redis_get.return_value = {
            "region": "FPL",
            "forecasts": [
                {
                    "timestamp": "2026-05-20T06:00:00+00:00",
                    "predicted_demand_mw": 18000.0,
                    "xgboost": 18000.0,
                }
            ],
            # No ensemble_weights here
        }
        assert is_trained("FPL") is True

    @patch("os.path.exists")
    @patch("data.redis_client.redis_get")
    def test_empty_redis_falls_back_to_local_pickle(self, mock_redis_get, mock_exists):
        """No Redis data → check the legacy local-pickle path for dev mode."""
        from models.model_service import is_trained

        mock_redis_get.return_value = None
        mock_exists.return_value = True  # Local pickle exists (dev mode)

        assert is_trained("FPL") is True
        mock_exists.assert_called()

    @patch("os.path.exists")
    @patch("data.redis_client.redis_get")
    def test_no_redis_no_local_returns_false(self, mock_redis_get, mock_exists):
        from models.model_service import is_trained

        mock_redis_get.return_value = None
        mock_exists.return_value = False

        assert is_trained("FPL") is False

    @patch("data.redis_client.redis_get")
    def test_invalid_region_returns_false(self, mock_redis_get):
        """Region name fails validation → False without consulting Redis."""
        from models.model_service import is_trained

        assert is_trained("not-a-real-region!") is False
        # The Redis check should be skipped for invalid regions.
        mock_redis_get.assert_not_called()

    @patch("data.redis_client.redis_get")
    def test_empty_forecasts_list_with_no_weights_falls_back_to_local(self, mock_redis_get):
        """Redis exists but forecasts is empty — falls back to local check."""
        from models.model_service import is_trained

        mock_redis_get.return_value = {
            "region": "FPL",
            "forecasts": [],
            # No ensemble_weights
        }
        # With mock_redis_get returning this empty-but-present payload,
        # is_trained should fall through to the legacy local check.
        # In test env (no local pickle), result is False.
        with patch("os.path.exists", return_value=False):
            assert is_trained("FPL") is False

    @patch("data.redis_client.redis_get")
    def test_malformed_redis_payload_falls_back_to_local(self, mock_redis_get):
        """Redis returns non-dict — fall back to local check."""
        from models.model_service import is_trained

        mock_redis_get.return_value = "not a dict"
        with patch("os.path.exists", return_value=False):
            assert is_trained("FPL") is False

    @patch("data.redis_client.redis_get")
    def test_uses_correct_redis_key(self, mock_redis_get):
        from models.model_service import is_trained

        mock_redis_get.return_value = None
        with patch("os.path.exists", return_value=False):
            is_trained("ERCOT")

        called_key = mock_redis_get.call_args[0][0]
        assert called_key == "gridpulse:forecast:ERCOT:1h"
