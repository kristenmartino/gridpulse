"""Tests for the V3.ζ-follow-up forecast quality gate.

The gate hides a BA from the header dropdown and the US Grid card grid only
when its **best achievable** holdout MAPE — the champion across the ensemble +
the three base models — is in the ``rollback`` grade (>22% on the 7-day
horizon per ``MAPE_BY_HORIZON``). Threshold sourced from the project's existing
``mape_grade()`` governance framework rather than invented (config.py:582-587).

Pre-#255 the gate keyed off **XGBoost-alone**, which hid SEC (XGBoost 38.6%)
even though production serves the ensemble (13.6%) and its best base (Prophet)
is 11.2%. Gating on the champion also avoids newly hiding SPA (ensemble 22.8%
but XGBoost 21.1%).

Coverage:
- ``is_forecast_quality_acceptable`` returns True for healthy MAPEs, False only
  when *every* served model is above the rollback threshold, and True when no
  MAPE signal exists yet (preserves "warming up" UX).
- #255: a BA is judged on its served champion, not XGBoost-alone (SEC visible,
  SPA not newly hidden).
- Feature flag short-circuits the gate when False.
- The TTL cache returns cached values on repeat calls and
  ``_reset_quality_gate_cache`` forces a refresh.
- ``hidden_regions`` and ``stable_visible_regions`` partition the input.
- US Grid title: "N hidden" annotation appears when applicable.
"""

from __future__ import annotations

from unittest.mock import patch


def _metrics(**model_mapes):
    """Build a ``get_model_metrics``-shaped dict: ``{model: {"mape": v}}``."""
    return {name: {"mape": v} for name, v in model_mapes.items()}


class TestIsForecastQualityAcceptable:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_healthy_champion_passes(self):
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(ensemble=2.5, xgboost=2.6),
        ):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_borderline_acceptable_passes(self):
        """22% is the rollback line — anything ≤ should pass."""
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(ensemble=21.99),
        ):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_rollback_grade_fails_only_when_every_model_rolls_back(self):
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(xgboost=25.0, prophet=30.0, arima=28.0, ensemble=26.0),
        ):
            assert is_forecast_quality_acceptable("CPLW") is False

    def test_gates_on_champion_not_xgboost_alone(self):
        """#255: SEC — XGBoost 38.63% (rollback) but served ensemble 13.61% and
        best base Prophet 11.22% (both acceptable). Must stay VISIBLE."""
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(xgboost=38.63, prophet=11.22, arima=12.94, ensemble=13.61),
        ):
            assert is_forecast_quality_acceptable("SEC") is True

    def test_ensemble_over_threshold_but_base_ok_stays_visible(self):
        """#255 anti-regression: SPA — ensemble 22.81% (just over rollback) but
        XGBoost 21.13% (just under). Gating on the ensemble alone would newly
        hide it; the champion gate keeps it visible."""
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(xgboost=21.13, prophet=25.83, arima=36.96, ensemble=22.81),
        ):
            assert is_forecast_quality_acceptable("SPA") is True

    def test_hidden_only_when_all_models_rollback(self):
        """Hide a BA only when *no* served model reaches the acceptable grade."""
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(xgboost=30.0, prophet=28.0, arima=35.0, ensemble=31.0),
        ):
            assert is_forecast_quality_acceptable("XXX") is False

    def test_no_metrics_passes(self):
        """No signal → don't hide. Preserves 'warming up' UX."""
        from models.model_service import is_forecast_quality_acceptable

        with patch("models.model_service.get_model_metrics", return_value={}):
            assert is_forecast_quality_acceptable("HST") is True

    def test_metrics_without_mape_passes(self):
        """Defensive: metrics dict present but no ``mape`` field — prefer
        showing the BA over hiding it."""
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.model_service.get_model_metrics",
            return_value={"xgboost": {"rmse": 100.0}},
        ):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_feature_flag_disabled_short_circuits(self):
        """When the flag is False (dev default), gate is bypassed."""
        from models.model_service import is_forecast_quality_acceptable

        with (
            patch("config.feature_enabled", return_value=False),
            patch(
                "models.model_service.get_model_metrics",
                return_value=_metrics(xgboost=99.0, ensemble=99.0),
            ),
        ):
            assert is_forecast_quality_acceptable("CPLW") is True

    def test_get_model_metrics_exception_treated_as_no_signal(self):
        """If the metric read raises, treat as 'no signal' and pass — an
        operational outage shouldn't black out the dropdown."""
        from models.model_service import is_forecast_quality_acceptable

        def _raise(*a, **kw):
            raise RuntimeError("synthetic Redis/GCS outage")

        with patch("models.model_service.get_model_metrics", side_effect=_raise):
            assert is_forecast_quality_acceptable("PJM") is True


class TestQualityGateCache:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_cache_returns_same_value_on_repeated_calls(self):
        """Second call to ``get_best_holdout_mape`` for the same region should
        hit the cache without re-reading the metrics."""
        from models.model_service import get_best_holdout_mape

        n_calls = 0

        def _counting(region):
            nonlocal n_calls
            n_calls += 1
            return _metrics(ensemble=3.5)

        with patch("models.model_service.get_model_metrics", side_effect=_counting):
            assert get_best_holdout_mape("PJM") == 3.5
            assert get_best_holdout_mape("PJM") == 3.5
            assert get_best_holdout_mape("PJM") == 3.5
        assert n_calls == 1

    def test_reset_cache_forces_refresh(self):
        from models.model_service import (
            _reset_quality_gate_cache,
            get_best_holdout_mape,
        )

        n_calls = 0

        def _counting(region):
            nonlocal n_calls
            n_calls += 1
            return _metrics(ensemble=3.5)

        with patch("models.model_service.get_model_metrics", side_effect=_counting):
            assert get_best_holdout_mape("PJM") == 3.5
            _reset_quality_gate_cache()
            assert get_best_holdout_mape("PJM") == 3.5
        assert n_calls == 2


class TestHiddenAndVisibleHelpers:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_partition_by_gate(self):
        """``hidden_regions`` and ``stable_visible_regions`` together partition
        the input."""
        from models.model_service import hidden_regions, stable_visible_regions

        # PJM passes (champion 3.5), CPLW fails (all models > 22),
        # HST is "no signal" → passes.
        def _metrics_by_region(region):
            table = {
                "PJM": _metrics(ensemble=3.5),
                "CPLW": _metrics(xgboost=25.0, prophet=30.0, arima=28.0, ensemble=26.0),
                "HST": {},
            }
            return table.get(region, {})

        regions = ["PJM", "CPLW", "HST"]
        with patch("models.model_service.get_model_metrics", side_effect=_metrics_by_region):
            visible = stable_visible_regions(regions)
            hidden = hidden_regions(regions)
        assert set(visible + hidden) == set(regions)
        assert set(visible) & set(hidden) == set()
        assert "PJM" in visible
        assert "HST" in visible  # no signal preserves visibility
        assert "CPLW" in hidden

    def test_visible_preserves_input_order(self):
        from models.model_service import stable_visible_regions

        with patch(
            "models.model_service.get_model_metrics",
            return_value=_metrics(ensemble=3.5),
        ):
            assert stable_visible_regions(["PJM", "MISO", "ERCOT"]) == [
                "PJM",
                "MISO",
                "ERCOT",
            ]


class TestUsGridTitleHiddenAnnotation:
    """The page title annotates "N hidden" when the gate has hidden any BAs."""

    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_no_hidden_no_annotation(self):
        from components.callbacks import _build_us_grid_title

        with (
            patch("models.model_service.hidden_regions", return_value=[]),
            patch("components.callbacks.REGION_NAMES", {"PJM": "PJM", "MISO": "MISO"}),
        ):
            title = _build_us_grid_title({"PJM": {"current_mw": 70000.0}})

        rendered = _all_text(title)
        assert "hidden" not in " ".join(rendered).lower()

    def test_hidden_count_appears_in_subtitle(self):
        from components.callbacks import _build_us_grid_title

        with (
            patch(
                "models.model_service.hidden_regions",
                return_value=["CPLW", "HST", "SPA"],
            ),
            patch("components.callbacks.REGION_NAMES", {f"R{i}": f"R{i}" for i in range(10)}),
        ):
            title = _build_us_grid_title({"R0": {"current_mw": 50000.0}})

        rendered_text = " ".join(_all_text(title))
        assert "3 hidden" in rendered_text


def _all_text(component):
    """Walk a Dash component tree and return all string children."""
    out = []
    children = getattr(component, "children", None)
    if isinstance(children, str):
        out.append(children)
    elif isinstance(children, (list, tuple)):
        for c in children:
            out.extend(_all_text(c))
    elif children is not None:
        out.extend(_all_text(children))
    return out


class TestUsGridCollectorFilteringByGate:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_hidden_region_absent_from_collector(self, monkeypatch):
        """A region that fails the gate must not appear in the collector output
        at all — downstream sums + cards never see it."""
        from components import callbacks as cb

        monkeypatch.setattr(cb, "redis_get", lambda key: {"demand_mw": [50000.0]})

        def _gate(region):
            return region != "CPLW"  # only CPLW fails

        with (
            patch(
                "components.callbacks.REGION_NAMES",
                {"PJM": "PJM", "CPLW": "CPLW", "HST": "HST"},
            ),
            patch("models.model_service.is_forecast_quality_acceptable", side_effect=_gate),
        ):
            data = cb._collect_us_grid_region_data()
        assert "PJM" in data
        assert "HST" in data  # passes (gate said True)
        assert "CPLW" not in data
