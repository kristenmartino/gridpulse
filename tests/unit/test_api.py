"""Unit tests for the public read-only JSON API v1 (``api.py``, #250).

The blueprint is mounted on a bare Flask app (no Dash boot) and Redis is
mocked at the ``api`` module boundary. The honesty contract under test:

- warming/cold Redis → 503 ``{"status": "warming"}``, never fabricated data
- unknown region → 404 with the valid-region list, raw input never reflected
- horizon outside [1, 168] → 400 (168 is a deliberate export cap; below the
  cap the response truncates to the rows actually in cache)
- exports are allow-listed (known models / known fields) — internal cache-
  schema fields never auto-publish; raw ``records`` arrays never leave
- /grid/summary mirrors the UI's artifact filter (#225 class) + discloses it,
  and memoizes its fan-out body in-process
- CORS on everything; ``public, max-age`` on 200s only, errors ``no-store``
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask

import api as api_module
from api import api_v1


@pytest.fixture(autouse=True)
def _clear_memo():
    """The fan-out endpoints memoize success bodies in-process — isolate tests."""
    api_module._memo.clear()
    yield
    api_module._memo.clear()


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.register_blueprint(api_v1)
    return app.test_client()


def _forecast_payload(rows: int = 168, with_ensemble: bool = True) -> dict:
    forecasts = []
    for i in range(rows):
        row = {
            "timestamp": f"2026-07-07T{i % 24:02d}:00:00+00:00",
            "predicted_demand_mw": 20000.0 + i,
            "xgboost": 20000.0 + i,
            "prophet": 20100.0 + i,
        }
        if with_ensemble:
            row["ensemble"] = 20050.0 + i
        forecasts.append(row)
    return {
        "region": "FPL",
        "scored_at": "2026-07-07T15:02:00+00:00",
        "granularity": "1h",
        "primary_model": "xgboost",
        "forecasts": forecasts,
        "ensemble_weights": {"xgboost": 0.8, "prophet": 0.2},
        "model_metrics": {"xgboost": {"mape": 2.5}},
    }


class TestIndexAndHeaders:
    def test_index_lists_endpoints(self, client):
        resp = client.get("/api/v1")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["version"] == "v1"
        assert any("/forecast/" in k for k in body["endpoints"])

    def test_cors_and_cache_headers_on_success(self, client):
        resp = client.get("/api/v1")
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert "max-age=60" in resp.headers["Cache-Control"]

    def test_index_carries_data_source_attribution(self, client):
        """CORS is '*', so payloads are redistributed anywhere. Open-Meteo is
        CC-BY-4.0 — its credit + link must travel with the data."""
        body = client.get("/api/v1").get_json()
        sources = {a["source"]: a for a in body["attribution"]}
        om = next(a for a in body["attribution"] if "Open-Meteo" in a["source"])
        assert om["license"] == "CC-BY-4.0"
        assert om["url"] == "https://open-meteo.com/"
        assert any("EIA-930" in s for s in sources)

    @patch("api.redis_get")
    def test_forecast_payload_carries_attribution(self, mock_redis, client):
        """A redistributed forecast row is weather-driven demand → both the EIA
        and the CC-BY Open-Meteo credits accompany it."""
        mock_redis.return_value = _forecast_payload()
        body = client.get("/api/v1/forecast/FPL").get_json()
        licenses = {a["source"]: a["license"] for a in body["attribution"]}
        assert any("Open-Meteo" in s for s in licenses)
        assert any("EIA-930" in s for s in licenses)

    @patch("api.redis_get", return_value=None)
    def test_errors_are_never_shared_cacheable(self, _redis, client):
        """A cached 503 'warming' would delay first data by up to 60s — errors
        must be no-store, and 503s must carry Retry-After."""
        warming = client.get("/api/v1/forecast/FPL")
        assert warming.status_code == 503
        assert warming.headers["Cache-Control"] == "no-store"
        assert warming.headers["Retry-After"] == "60"
        missing = client.get("/api/v1/forecast/NOPE")
        assert missing.status_code == 404
        assert missing.headers["Cache-Control"] == "no-store"


class TestRegions:
    @patch("models.model_service.is_forecast_quality_acceptable", return_value=True)
    def test_lists_all_regions_with_metadata(self, _gate, client):
        resp = client.get("/api/v1/regions")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 51
        fpl = next(r for r in body["regions"] if r["code"] == "FPL")
        assert fpl["name"] == "Florida (FPL/NextEra)"
        # Field renamed nameplate_capacity_mw -> capacity_mw (#254): FPL is a
        # measured plate, so capacity_source is "nameplate".
        assert fpl["capacity_mw"] == 35_963
        assert fpl["capacity_source"] == "nameplate"
        assert fpl["import_dominated"] is False
        assert fpl["quality_gated"] is False
        hst = next(r for r in body["regions"] if r["code"] == "HST")
        assert hst["import_dominated"] is True
        # Peak-derived capacity (peak×1.15) is disclosed as an estimate, not
        # mislabeled "nameplate" (#254). SOCO is peak-derived but NOT import-
        # dominated — the gap this fix closes.
        soco = next(r for r in body["regions"] if r["code"] == "SOCO")
        assert soco["capacity_source"] == "peak_estimate"
        assert soco["import_dominated"] is False
        assert hst["capacity_source"] == "peak_estimate"
        # SPA is import-dominated but a TRUE nameplate (federal dam fleet).
        spa = next(r for r in body["regions"] if r["code"] == "SPA")
        assert spa["capacity_source"] == "nameplate"
        assert spa["import_dominated"] is True


class TestForecast:
    @patch("api.redis_get")
    def test_warming_when_cache_cold(self, mock_redis, client):
        mock_redis.return_value = None
        resp = client.get("/api/v1/forecast/FPL")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "warming"

    @patch("api.redis_get")
    def test_happy_path_defaults_to_24h_ensemble(self, mock_redis, client):
        mock_redis.return_value = _forecast_payload()
        resp = client.get("/api/v1/forecast/FPL")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["region"] == "FPL"
        assert body["scored_at"] == "2026-07-07T15:02:00+00:00"
        assert body["series_source"] == "ensemble"
        assert body["horizon_hours"] == 24
        assert len(body["forecast"]) == 24
        first = body["forecast"][0]
        assert first["demand_mw"] == 20050.0  # ensemble, not primary
        assert first["by_model"]["xgboost"] == 20000.0
        assert body["ensemble_weights"]["xgboost"] == 0.8
        assert body["holdout_metrics"]["xgboost"]["mape"] == 2.5

    @patch("api.redis_get")
    def test_no_ensemble_falls_back_to_primary_and_says_so(self, mock_redis, client):
        mock_redis.return_value = _forecast_payload(with_ensemble=False)
        resp = client.get("/api/v1/forecast/FPL")
        body = resp.get_json()
        assert body["series_source"] == "xgboost"
        assert body["forecast"][0]["demand_mw"] == 20000.0

    @patch("api.redis_get")
    def test_horizon_clamps_to_available_rows(self, mock_redis, client):
        mock_redis.return_value = _forecast_payload(rows=48)
        resp = client.get("/api/v1/forecast/FPL?horizon=168")
        body = resp.get_json()
        assert body["horizon_hours"] == 48  # only 48 rows exist

    @patch("api.redis_get")
    def test_invalid_horizons_are_400(self, mock_redis, client):
        mock_redis.return_value = _forecast_payload()
        for bad in ("abc", "0", "-5", "169"):
            resp = client.get(f"/api/v1/forecast/FPL?horizon={bad}")
            assert resp.status_code == 400, bad
            assert resp.get_json()["error"] == "invalid_horizon"

    def test_unknown_region_404_never_reflects_input(self, client):
        resp = client.get("/api/v1/forecast/evil<script>")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["error"] == "unknown_region"
        assert "FPL" in body["valid_regions"]
        assert "script" not in resp.get_data(as_text=True)

    @patch("api.redis_get")
    def test_region_is_case_insensitive(self, mock_redis, client):
        mock_redis.return_value = _forecast_payload()
        resp = client.get("/api/v1/forecast/fpl")
        assert resp.status_code == 200
        assert resp.get_json()["region"] == "FPL"

    @patch("api.redis_get")
    def test_unknown_row_fields_never_auto_publish(self, mock_redis, client):
        """Allow-list, not deny-list: a future writer-added field (debug
        annotation, uncalibrated interval) must not leak to the public API."""
        payload = _forecast_payload(rows=2)
        payload["forecasts"][0]["debug_residual_p95"] = 123.4
        payload["forecasts"][0]["interval_lower"] = 19000.0
        mock_redis.return_value = payload
        body = client.get("/api/v1/forecast/FPL?horizon=2").get_json()
        row = body["forecast"][0]
        assert "debug_residual_p95" not in row["by_model"]
        assert "interval_lower" not in row["by_model"]
        assert set(row["by_model"]) <= {"prophet", "arima", "xgboost", "ensemble"}


class TestGridSummary:
    @patch("models.model_service.hidden_regions", return_value=[])
    @patch("components._callbacks_us_grid._collect_us_grid_region_data")
    def test_summary_math_matches_us_grid_semantics(self, mock_collect, _hidden, client):
        mock_collect.return_value = {
            "PJM": {"current_mw": 90000.0, "today_mw": [90000.0] * 24},
            "ERCOT": {"current_mw": 40000.0, "today_mw": [40000.0] * 24},
            # import-dominated: must count toward totals but NOT stress/util
            "HST": {"current_mw": 100.0, "today_mw": [100.0] * 24},
        }
        resp = client.get("/api/v1/grid/summary")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["reporting_regions"] == 3
        assert body["total_demand_mw"] == pytest.approx(130100.0)
        assert body["top_stress"]["region"] in ("PJM", "ERCOT")
        assert body["national_utilization_pct"] is not None
        # nameplate honesty note present
        assert any("nameplate" in n for n in body["notes"])

    @patch("components._callbacks_us_grid._collect_us_grid_region_data", return_value={})
    def test_summary_warming_when_no_regions(self, _collect, client):
        resp = client.get("/api/v1/grid/summary")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "warming"

    @patch("models.model_service.hidden_regions", return_value=[])
    @patch("components._callbacks_us_grid._collect_us_grid_region_data")
    def test_artifact_readings_excluded_and_disclosed(self, mock_collect, _hidden, client):
        """UI parity (#225 glitch class): a latest reading far below the BA's
        own 24h median is excluded from every aggregate — and disclosed."""
        mock_collect.return_value = {
            "PJM": {"current_mw": 90000.0, "today_mw": [90000.0] * 24},
            # ERCOT glitched: latest 200 MW vs a 40,000 MW 24h median.
            "ERCOT": {"current_mw": 200.0, "today_mw": [40000.0] * 23 + [200.0]},
        }
        body = client.get("/api/v1/grid/summary").get_json()
        assert body["reporting_regions"] == 1
        assert body["total_demand_mw"] == pytest.approx(90000.0)
        assert body["artifact_excluded_regions"] == ["ERCOT"]
        assert body["top_stress"]["region"] == "PJM"

    @patch("models.model_service.hidden_regions", return_value=[])
    @patch("components._callbacks_us_grid._collect_us_grid_region_data")
    def test_summary_memoized_against_fanout_hammering(self, mock_collect, _hidden, client):
        """The summary body is client-independent — repeat requests within the
        memo TTL must not re-run the ~100-Redis-read fan-out."""
        mock_collect.return_value = {
            "PJM": {"current_mw": 90000.0, "today_mw": [90000.0] * 24},
        }
        first = client.get("/api/v1/grid/summary")
        second = client.get("/api/v1/grid/summary")
        assert first.status_code == second.status_code == 200
        assert first.get_json() == second.get_json()
        assert mock_collect.call_count == 1


class TestDrift:
    @patch("api.redis_get")
    def test_warming_when_both_payloads_cold(self, mock_redis, client):
        mock_redis.return_value = None
        resp = client.get("/api/v1/drift/FPL")
        assert resp.status_code == 503

    @patch("api.redis_get")
    def test_partial_data_renders_with_null_other_half(self, mock_redis, client):
        def route(key):
            if "drift_horizon" in key:
                return None
            return {
                "region": "FPL",
                "last_updated_at": "2026-07-07T15:00:00+00:00",
                "models": {
                    "xgboost": {
                        "rolling_mape_7d": 1.1,
                        "rolling_mape_30d": 1.3,
                        "n_records": 652,
                        "records": [{"big": "array"}] * 100,
                    }
                },
            }

        mock_redis.side_effect = route
        resp = client.get("/api/v1/drift/FPL")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["by_horizon"] is None
        xgb = body["live_1h"]["models"]["xgboost"]
        assert xgb["rolling_mape_7d"] == 1.1
        assert "records" not in xgb  # raw arrays stripped

    @patch("api.redis_get")
    def test_horizon_blocks_stripped_of_records(self, mock_redis, client):
        def route(key):
            if "drift_horizon" in key:
                return {
                    "region": "FPL",
                    "horizons": ["24h", "48h", "72h"],
                    "models": {
                        "prophet": {
                            "24h": {
                                "rolling_mape_7d": 5.41,
                                "grade": "acceptable",
                                "n_records": 24,
                                "records": [1, 2, 3],
                            }
                        }
                    },
                }
            return None

        mock_redis.side_effect = route
        resp = client.get("/api/v1/drift/FPL")
        body = resp.get_json()
        p24 = body["by_horizon"]["models"]["prophet"]["24h"]
        assert p24["grade"] == "acceptable"
        assert "records" not in p24

    @patch("api.redis_get")
    def test_unknown_models_and_fields_never_auto_publish(self, mock_redis, client):
        """Allow-list on both axes: unknown model names and unknown block
        fields in the internal cache schema must not leak."""

        def route(key):
            if "drift_horizon" in key:
                return None
            return {
                "region": "FPL",
                "models": {
                    "xgboost": {
                        "rolling_mape_7d": 1.1,
                        "n_records": 100,
                        "internal_debug_path": "/app/trained_models/x.pkl",
                    },
                    "experimental_v2_model": {"rolling_mape_7d": 9.9},
                },
            }

        mock_redis.side_effect = route
        body = client.get("/api/v1/drift/FPL").get_json()
        models = body["live_1h"]["models"]
        assert "experimental_v2_model" not in models
        assert "internal_debug_path" not in models["xgboost"]

    def test_unknown_region_404(self, client):
        resp = client.get("/api/v1/drift/NOPE")
        assert resp.status_code == 404
