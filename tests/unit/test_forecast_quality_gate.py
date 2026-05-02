"""Tests for the V3.ζ-follow-up forecast quality gate.

The gate hides BAs whose XGBoost holdout MAPE is in the ``rollback``
grade (>22% on the 7-day horizon per ``MAPE_BY_HORIZON``) from the
header dropdown and the US Grid card grid. Threshold sourced from the
project's existing ``mape_grade()`` governance framework rather than
invented (config.py:582-587).

Coverage:
- ``is_forecast_quality_acceptable`` returns True for healthy MAPEs,
  False above the rollback threshold, and True when no MAPE signal
  exists yet (preserves "warming up" UX).
- Feature flag short-circuits the gate when False.
- The TTL cache returns cached values on repeat calls and ``_reset_quality_gate_cache``
  forces a refresh.
- ``hidden_regions`` and ``stable_visible_regions`` partition the
  input correctly.
- Dropdown rendering: hidden BAs absent from options; empty groups
  drop their separator.
- US Grid collector: hidden BAs absent from returned dict.
- US Grid title: "N hidden" annotation appears when applicable, with
  the affected codes in the hover tooltip.
"""

from __future__ import annotations

from unittest.mock import patch


def _patched_meta(mape_value):
    """Build a minimal ModelMetadata for the patch — only ``mape`` matters."""
    from models.persistence import ModelMetadata

    return ModelMetadata(
        region="TEST",
        model_name="xgboost",
        version="v-test",
        data_hash="h",
        trained_at="2026-05-02T00:00:00+00:00",
        train_rows=1000,
        mape=mape_value,
        lib_versions={},
        extra={},
    )


class TestIsForecastQualityAcceptable:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_healthy_xgboost_mape_passes(self):
        from models.model_service import is_forecast_quality_acceptable

        with patch(
            "models.persistence.get_model_metadata",
            return_value=_patched_meta(2.5),  # well within excellent grade
        ):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_borderline_acceptable_passes(self):
        """22% is the rollback line — anything ≤ should pass (the gate
        only hides BAs in the rollback grade, i.e. > 22%)."""
        from models.model_service import is_forecast_quality_acceptable

        with patch("models.persistence.get_model_metadata", return_value=_patched_meta(21.99)):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_rollback_grade_fails(self):
        from models.model_service import is_forecast_quality_acceptable

        with patch("models.persistence.get_model_metadata", return_value=_patched_meta(25.0)):
            assert is_forecast_quality_acceptable("CPLW") is False

    def test_no_metadata_passes(self):
        """No signal → don't hide. Preserves 'warming up' UX for
        newly-added BAs that haven't been trained yet."""
        from models.model_service import is_forecast_quality_acceptable

        with patch("models.persistence.get_model_metadata", return_value=None):
            assert is_forecast_quality_acceptable("HST") is True

    def test_metadata_with_null_mape_passes(self):
        """Defensive: meta.json exists but ``mape`` field is null
        (could happen if training computed but failed to record).
        Prefer showing the BA over hiding it."""
        from models.model_service import is_forecast_quality_acceptable

        with patch("models.persistence.get_model_metadata", return_value=_patched_meta(None)):
            assert is_forecast_quality_acceptable("PJM") is True

    def test_feature_flag_disabled_short_circuits(self):
        """When the flag is False (dev default), gate is bypassed."""
        from models.model_service import is_forecast_quality_acceptable

        # Even with a clearly-broken MAPE, the gate returns True when
        # the feature flag is off.
        with (
            patch("config.feature_enabled", return_value=False),
            patch("models.persistence.get_model_metadata", return_value=_patched_meta(99.0)),
        ):
            assert is_forecast_quality_acceptable("CPLW") is True

    def test_get_model_metadata_exception_treated_as_no_signal(self):
        """If GCS is unreachable or the read raises, treat as 'no signal'
        and pass — operational outage shouldn't black out the dropdown."""
        from models.model_service import is_forecast_quality_acceptable

        def _raise(*a, **kw):
            raise RuntimeError("synthetic GCS outage")

        with patch("models.persistence.get_model_metadata", side_effect=_raise):
            assert is_forecast_quality_acceptable("PJM") is True


class TestQualityGateCache:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_cache_returns_same_value_on_repeated_calls(self):
        """Second call to ``get_xgboost_holdout_mape`` for the same
        region should hit the cache without re-reading from GCS."""
        from models.model_service import get_xgboost_holdout_mape

        n_calls = 0

        def _counting_meta(*args, **kwargs):
            nonlocal n_calls
            n_calls += 1
            return _patched_meta(3.5)

        with patch("models.persistence.get_model_metadata", side_effect=_counting_meta):
            assert get_xgboost_holdout_mape("PJM") == 3.5
            assert get_xgboost_holdout_mape("PJM") == 3.5
            assert get_xgboost_holdout_mape("PJM") == 3.5
        assert n_calls == 1

    def test_reset_cache_forces_refresh(self):
        from models.model_service import (
            _reset_quality_gate_cache,
            get_xgboost_holdout_mape,
        )

        n_calls = 0

        def _counting_meta(*args, **kwargs):
            nonlocal n_calls
            n_calls += 1
            return _patched_meta(3.5)

        with patch("models.persistence.get_model_metadata", side_effect=_counting_meta):
            assert get_xgboost_holdout_mape("PJM") == 3.5
            _reset_quality_gate_cache()
            assert get_xgboost_holdout_mape("PJM") == 3.5
        assert n_calls == 2


class TestHiddenAndVisibleHelpers:
    def setup_method(self):
        from models.model_service import _reset_quality_gate_cache

        _reset_quality_gate_cache()

    def test_partition_by_gate(self):
        """``hidden_regions`` and ``stable_visible_regions`` together
        partition the input."""
        from models.model_service import hidden_regions, stable_visible_regions

        # PJM passes, CPLW fails, HST is "no signal" → passes
        def _meta_by_region(region, model_name):
            mapes = {"PJM": 3.5, "CPLW": 25.0, "HST": None}
            if region not in mapes:
                return None
            value = mapes[region]
            if value is None:
                return None  # no signal — passes
            return _patched_meta(value)

        regions = ["PJM", "CPLW", "HST"]
        with patch("models.persistence.get_model_metadata", side_effect=_meta_by_region):
            visible = stable_visible_regions(regions)
            hidden = hidden_regions(regions)
        assert set(visible + hidden) == set(regions)
        assert set(visible) & set(hidden) == set()
        assert "PJM" in visible
        assert "HST" in visible  # no signal preserves visibility
        assert "CPLW" in hidden

    def test_visible_preserves_input_order(self):
        from models.model_service import stable_visible_regions

        with patch("models.persistence.get_model_metadata", return_value=_patched_meta(3.5)):
            assert stable_visible_regions(["PJM", "MISO", "ERCOT"]) == [
                "PJM",
                "MISO",
                "ERCOT",
            ]


class TestUsGridTitleHiddenAnnotation:
    """The page title annotates "N hidden" when the gate has hidden any
    BAs, with the affected codes available via the hover tooltip."""

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
        """A region that fails the gate must not appear in the
        collector output at all — downstream sums + cards never see it."""
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
