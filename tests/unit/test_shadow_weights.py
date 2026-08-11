"""#478: the shadow weighting — measuring the arm that is NOT served.

#451 proved the WAPE half of the smoothed-weights question with a replay and
could not prove the bias half: a replayed vintage over-forecasts by ~6% in the
*control* arm, and a harness whose control fails a constraint cannot certify the
treatment against it. This records both arms on production forecasts instead.

The load-bearing property is that recording the shadow changes nothing that
ships. Most of what follows tests that, not the arithmetic.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jobs.phases import RegionData, _write_shadow_weights, predict_and_write_forecast
from models.ensemble import shadow_weighting_mape, weighting_mape


def _region_data(region: str = "FPL", hours: int = 200) -> RegionData:
    ts = pd.date_range("2026-05-19", periods=hours, freq="h", tz="UTC")
    featured = pd.DataFrame({"timestamp": ts, "demand_mw": np.full(hours, 18000.0)})
    featured["hour"] = featured["timestamp"].dt.hour
    featured["day_of_week"] = featured["timestamp"].dt.dayofweek
    featured["month"] = featured["timestamp"].dt.month
    featured["day_of_year"] = featured["timestamp"].dt.dayofyear
    return RegionData(
        region=region,
        demand_df=featured[["timestamp", "demand_mw"]],
        weather_df=pd.DataFrame(),
        featured_df=featured,
    )


class TestTheServedForecastIsUnaffected:
    """#478's first acceptance box, and the only one that can cause an incident."""

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get")
    def test_the_served_payload_is_identical_with_and_without_a_shadow(
        self, _get, mock_set
    ) -> None:
        """Byte-identical, asserted rather than inspected.

        Run the phase twice — once with a shadow arm available, once without —
        and compare the payload written to ``forecast:{region}:1h``. Anything
        that leaked from the shadow path into the served one shows up as a diff.
        """
        payloads = {}
        for label, shadow in (("with", {"xgboost": 3.0, "prophet": 6.0}), ("without", None)):
            mock_set.reset_mock()
            with patch("jobs.phases._predict_one") as predict:
                predict.side_effect = lambda m, *a, **k: np.full(720, 18000.0)
                predict_and_write_forecast(
                    _region_data(),
                    models={"xgboost": object(), "prophet": object()},
                    model_mapes={"xgboost": 4.0, "prophet": 8.0},
                    model_mapes_shadow=shadow,
                )
            served = [
                c.args[1]
                for c in mock_set.call_args_list
                if c.args[0] == "gridpulse:forecast:FPL:1h"
            ]
            assert len(served) == 1
            payloads[label] = served[0]

        a, b = payloads["with"], payloads["without"]
        # scored_at is wall-clock and differs between the two runs by design.
        a.pop("scored_at", None)
        b.pop("scored_at", None)
        assert a == b

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get")
    def test_no_shadow_series_appears_in_the_forecast_rows(self, _get, mock_set) -> None:
        """The reason the shadow lives in its own key.

        ``drift.extract_one_hour_ahead_predictions`` treats EVERY numeric key in
        a forecast row as a model. A shadow series added to those rows would
        silently acquire drift records, a Models-tab entry, and a place in the
        rolling MAPE the visibility gate reads — a shadow that is not a shadow.
        """
        with patch("jobs.phases._predict_one") as predict:
            predict.side_effect = lambda m, *a, **k: np.full(720, 18000.0)
            predict_and_write_forecast(
                _region_data(),
                models={"xgboost": object(), "prophet": object()},
                model_mapes={"xgboost": 4.0, "prophet": 8.0},
                model_mapes_shadow={"xgboost": 3.0, "prophet": 6.0},
            )
        served = next(
            c.args[1] for c in mock_set.call_args_list if c.args[0] == "gridpulse:forecast:FPL:1h"
        )
        row = served["forecasts"][0]
        assert set(row) == {"timestamp", "predicted_demand_mw", "xgboost", "prophet", "ensemble"}

        from models.drift import extract_one_hour_ahead_predictions

        assert set(extract_one_hour_ahead_predictions(served, row["timestamp"])) == {
            "xgboost",
            "prophet",
            "ensemble",
        }

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get")
    def test_a_failing_shadow_write_does_not_fail_the_phase(self, _get, mock_set) -> None:
        """Enrichment only. The forecast is already in Redis when this runs."""
        with (
            patch("jobs.phases._predict_one") as predict,
            patch("jobs.phases._write_shadow_weights", side_effect=RuntimeError("boom")),
        ):
            predict.side_effect = lambda m, *a, **k: np.full(720, 18000.0)
            result = predict_and_write_forecast(
                _region_data(),
                models={"xgboost": object(), "prophet": object()},
                model_mapes={"xgboost": 4.0, "prophet": 8.0},
                model_mapes_shadow={"xgboost": 3.0, "prophet": 6.0},
            )

        # The forecast landed, so the phase must report ok. #267 makes `ok` the
        # signal for "this region was scored", and the phase's outer except
        # would otherwise turn a shadow failure into a failed region whose
        # forecast is sitting in Redis — the #268 mistake inverted.
        assert result.ok, f"a failed shadow write must not fail the phase: {result.error}"
        assert any(c.args[0] == "gridpulse:forecast:FPL:1h" for c in mock_set.call_args_list)


class TestTheShadowMeasuresTheRightThing:
    def _rows(self, n: int = 3) -> list[dict]:
        ts = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
        return [{"timestamp": t.isoformat(), "ensemble": 100.0} for t in ts]

    @patch("data.redis_client.persist")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_both_arms_blend_the_same_per_model_forecasts(self, _get, persist) -> None:
        """The experiment is only valid if weights are the ONLY difference.

        Same arrays in, two weightings out. If the shadow ever recomputed a
        forecast, it would be measuring the inference path, not the weights —
        the #458 mistake in a different costume.
        """
        preds = {"xgboost": np.full(3, 100.0), "prophet": np.full(3, 200.0)}
        assert _write_shadow_weights(
            region="FPL",
            predictions_by_model=preds,
            model_mapes_shadow={"xgboost": 2.0, "prophet": 4.0},
            served_weights={"xgboost": 0.5, "prophet": 0.5},
            rows=self._rows(),
            demand_df=None,
        )
        payload = persist.call_args.args[1]
        assert payload["served_weights"] == {"xgboost": 0.5, "prophet": 0.5}
        # (1/2)^3 = 0.125 vs (1/4)^3 = 0.015625 → 8:1
        assert payload["shadow_weights"]["xgboost"] == pytest.approx(0.8889, abs=1e-3)
        # 0.8889*100 + 0.1111*200 = 111.1, against the served 150.
        assert payload["forecasts"][0]["shadow"] == pytest.approx(111.11, abs=0.1)
        assert payload["forecasts"][0]["served"] == 100.0

    @patch("data.redis_client.persist")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_a_partial_shadow_arm_is_skipped_not_equal_weighted(self, _get, persist) -> None:
        """resolve_ensemble_weights falls back to EQUAL weights on partial cover.

        Letting that through would file an "equal vs cubed" comparison under the
        name "smoothed vs raw" — the mislabelling class this repo keeps finding.
        A missing comparison is honest; a misnamed one is not.
        """
        preds = {"xgboost": np.full(3, 100.0), "prophet": np.full(3, 200.0)}
        assert not _write_shadow_weights(
            region="FPL",
            predictions_by_model=preds,
            model_mapes_shadow={"xgboost": 2.0},  # prophet has no EWMA yet
            served_weights={"xgboost": 0.5, "prophet": 0.5},
            rows=self._rows(),
            demand_df=None,
        )
        persist.assert_not_called()

    @patch("data.redis_client.persist")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_the_skip_says_which_models_lack_a_shadow(self, _get, persist) -> None:
        """The early guard is behaviourally redundant — assert its log instead.

        Found by mutation testing: deleting the explicit membership check
        changes nothing observable, because `resolve_ensemble_weights` then
        returns rule "equal" and the rule check rejects it anyway. The guard
        earns its place only through the log line, which is what tells an
        operator WHICH models have no EWMA yet and therefore when the shadow
        will start recording at all.

        That makes this the "unobservable, not equivalent" category named in
        docs/TEST_QUALITY.md — a survivor that no assertion on the return value
        can reach. Same fix as there: capture the logs.
        """
        from unittest.mock import MagicMock

        # Patch the module logger rather than using structlog.testing.capture_logs.
        # capture_logs is defeated by a bound logger that was cached before it
        # installed its processor, which makes it pass alone and fail inside the
        # full suite depending on which test configured structlog first. This
        # assertion does not depend on structlog's global state at all.
        recorder = MagicMock()
        with patch("jobs.phases.log", recorder):
            assert not _write_shadow_weights(
                region="FPL",
                predictions_by_model={
                    "xgboost": np.full(3, 100.0),
                    "prophet": np.full(3, 200.0),
                },
                model_mapes_shadow={"xgboost": 2.0},
                served_weights={"xgboost": 0.5, "prophet": 0.5},
                rows=self._rows(),
                demand_df=None,
            )
        calls = [
            c for c in recorder.info.call_args_list if c.args[0] == "shadow_weights_incomplete"
        ]
        assert calls, f"expected shadow_weights_incomplete; got {recorder.info.call_args_list}"
        assert calls[0].kwargs["missing"] == ["prophet"]
        assert calls[0].kwargs["have"] == ["xgboost"]
        persist.assert_not_called()

    @patch("data.redis_client.persist")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_a_single_model_has_no_ensemble_to_shadow(self, _get, persist) -> None:
        assert not _write_shadow_weights(
            region="FPL",
            predictions_by_model={"xgboost": np.full(3, 100.0)},
            model_mapes_shadow={"xgboost": 2.0},
            served_weights={"xgboost": 1.0},
            rows=self._rows(),
            demand_df=None,
        )
        persist.assert_not_called()


class TestGrading:
    def _previous(self) -> dict:
        ts = pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC")
        return {
            "region": "FPL",
            "forecasts": [
                {"timestamp": t.isoformat(), "served": 100.0, "shadow": 110.0} for t in ts
            ],
            "records": [],
        }

    @patch("data.redis_client.persist")
    def test_a_settled_hour_grades_both_arms_into_one_record(self, persist) -> None:
        """One record per tick carrying both arms and the shared actual.

        Both arms must be graded on the SAME hour by the SAME code — separate
        record streams could drift apart on which hours they cover and turn a
        weighting comparison into a coverage comparison.
        """
        ts = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC")
        demand = pd.DataFrame({"timestamp": ts, "demand_mw": [105.0, 105.0]})
        with patch("data.redis_client.redis_get", return_value=self._previous()):
            _write_shadow_weights(
                region="FPL",
                predictions_by_model={"a": np.full(3, 100.0), "b": np.full(3, 200.0)},
                model_mapes_shadow={"a": 2.0, "b": 4.0},
                served_weights={"a": 0.5, "b": 0.5},
                rows=[
                    {"timestamp": t.isoformat(), "ensemble": 100.0}
                    for t in pd.date_range("2026-06-02", periods=3, freq="h", tz="UTC")
                ],
                demand_df=demand,
            )
        records = persist.call_args.args[1]["records"]
        assert len(records) == 1
        assert records[0]["actual"] == 105.0
        assert records[0]["served_predicted"] == 100.0
        assert records[0]["shadow_predicted"] == 110.0

    @patch("data.redis_client.persist")
    def test_the_record_window_is_bounded(self, persist) -> None:
        """Unbounded growth would eventually make the payload unwritable."""
        from jobs.phases import _SHADOW_MAX_RECORDS

        previous = self._previous()
        previous["records"] = [{"n": i} for i in range(_SHADOW_MAX_RECORDS + 50)]
        with patch("data.redis_client.redis_get", return_value=previous):
            _write_shadow_weights(
                region="FPL",
                predictions_by_model={"a": np.full(3, 100.0), "b": np.full(3, 200.0)},
                model_mapes_shadow={"a": 2.0, "b": 4.0},
                served_weights={"a": 0.5, "b": 0.5},
                rows=[{"timestamp": "2026-06-02T00:00:00+00:00", "ensemble": 100.0}],
                demand_df=None,
            )
        assert len(persist.call_args.args[1]["records"]) == _SHADOW_MAX_RECORDS

    @patch("data.redis_client.persist")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_the_first_tick_has_nothing_to_grade_and_says_so(self, _get, persist) -> None:
        _write_shadow_weights(
            region="FPL",
            predictions_by_model={"a": np.full(3, 100.0), "b": np.full(3, 200.0)},
            model_mapes_shadow={"a": 2.0, "b": 4.0},
            served_weights={"a": 0.5, "b": 0.5},
            rows=[{"timestamp": "2026-06-02T00:00:00+00:00", "ensemble": 100.0}],
            demand_df=None,
        )
        assert persist.call_args.args[1]["records"] == []


class TestShadowArmSelection:
    def test_the_shadow_is_always_the_arm_that_is_not_served(self, monkeypatch) -> None:
        """Mirror property: flip the flag and the two swap.

        Without this the shadow could silently become a copy of the served arm
        and report "no difference" forever.
        """
        extra = {"mape_ewma": 3.0}
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        assert weighting_mape(5.0, extra) == 5.0
        assert shadow_weighting_mape(5.0, extra) == 3.0

        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": True})
        assert weighting_mape(5.0, extra) == 3.0
        assert shadow_weighting_mape(5.0, extra) == 5.0

    def test_a_missing_ewma_yields_no_shadow_rather_than_a_copy(self, monkeypatch) -> None:
        """The asymmetry with weighting_mape, and why they are separate functions.

        ``weighting_mape`` falls back to the raw MAPE so a served forecast always
        weights on a real measurement. Doing that here would make both arms
        identical and the comparison vacuous.
        """
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        assert shadow_weighting_mape(5.0, {}) is None
        assert shadow_weighting_mape(5.0, None) is None
        assert weighting_mape(5.0, {}) == 5.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_an_unusable_ewma_yields_no_shadow(self, monkeypatch, bad) -> None:
        monkeypatch.setattr("config.FEATURE_FLAGS", {"smoothed_ensemble_weights": False})
        assert shadow_weighting_mape(5.0, {"mape_ewma": bad}) is None
