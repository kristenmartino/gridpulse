"""Unit tests for the Models-tab drift panel — #121 part 2.

Covers ``_build_drift_panel`` in ``components/_callbacks_models.py``.

The function reads:
- ``gridpulse:drift:{region}`` (per-model rolling 7d / 30d MAPE +
  records, written hourly by jobs.phases.write_drift_metrics in #121 part 1)
- ``models.model_service.get_model_metrics`` (per-model holdout MAPE
  from each trained pickle's meta.json)

…and renders a per-model table with status chips classifying each model
by its LIVE 7d MAPE's own governance grade (``mape_grade`` on the 7-day
band), with the live÷holdout ratio shown for reference only (#217).

Tests verify:
- Warming state when Redis has no drift entry
- Per-model warming when n_records < 24 (insufficient sample)
- Status grades (excellent ≤6 / target ≤9 / acceptable ≤15 / rollback >15)
  keyed to the live MAPE, NOT a cross-horizon ratio to the holdout
- Mixed state (some models excellent, others in rollback)
- Missing-holdout fallback ("Live only" when meta.json lost holdout MAPE)
- Headline summary reflects whether any model is in rollback
"""

from __future__ import annotations

from unittest.mock import patch

from dash import html


def _drift_payload(
    *,
    region: str = "PJM",
    xgboost: dict | None = None,
    prophet: dict | None = None,
    arima: dict | None = None,
    ensemble: dict | None = None,
) -> dict:
    """Build a realistic ``gridpulse:drift:{region}`` payload for tests."""
    models = {}
    for name, data in (
        ("xgboost", xgboost),
        ("prophet", prophet),
        ("arima", arima),
        ("ensemble", ensemble),
    ):
        if data is not None:
            models[name] = data
    return {
        "region": region,
        "last_updated_at": "2026-05-20T15:00:00+00:00",
        "models": models,
    }


def _drift_model(
    rolling_mape_7d: float | None,
    *,
    rolling_mape_30d: float | None = None,
    n_records: int = 168,
) -> dict:
    """Build a single-model entry inside the drift payload."""
    return {
        "rolling_mape_7d": rolling_mape_7d,
        "rolling_mape_30d": rolling_mape_30d if rolling_mape_30d is not None else rolling_mape_7d,
        "n_records": n_records,
        "records": [],  # records list not used by the panel directly
    }


def _holdout_metrics(
    *,
    xgboost: float | None = None,
    prophet: float | None = None,
    arima: float | None = None,
    ensemble: float | None = None,
) -> dict:
    """Build a get_model_metrics() return shape for tests."""
    out: dict = {}
    for name, val in (
        ("xgboost", xgboost),
        ("prophet", prophet),
        ("arima", arima),
        ("ensemble", ensemble),
    ):
        if val is not None:
            out[name] = {"mape": val, "rmse": 1000.0, "mae": 800.0, "r2": 0.95}
    return out


def _find_status_chips(div: html.Div) -> list[str]:
    """Walk the rendered Div and return the inner-text of every status chip."""
    found: list[str] = []

    def walk(node):
        if (
            hasattr(node, "className")
            and node.className
            and "gp-status-chip--" in (node.className or "")
        ):
            # node.children is the chip's label text
            label = node.children
            if isinstance(label, str):
                found.append(label)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for c in children:
                walk(c)
        elif children is not None and not isinstance(children, str):
            walk(children)

    walk(div)
    return found


def _find_all_text(div: html.Div) -> str:
    """Concatenate all string children of the Div tree — for substring assertions."""
    pieces: list[str] = []

    def walk(node):
        if isinstance(node, str):
            pieces.append(node)
            return
        children = getattr(node, "children", None)
        if isinstance(children, str):
            pieces.append(children)
        elif isinstance(children, (list, tuple)):
            for c in children:
                walk(c)
        elif children is not None:
            walk(children)

    walk(div)
    return " ".join(pieces)


class TestWarmingStates:
    """No drift data yet → render a warming placeholder rather than empty/broken."""

    @patch("components._callbacks_models.redis_get")
    def test_no_drift_payload_in_redis_renders_warming(self, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = None  # cache miss

        result = _build_drift_panel("PJM")
        text = _find_all_text(result)
        assert "Warming up" in text
        # No status chips because no model rows render in warming state.
        assert _find_status_chips(result) == []

    @patch("components._callbacks_models.redis_get")
    def test_empty_models_dict_renders_warming(self, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = {"region": "PJM", "models": {}}

        result = _build_drift_panel("PJM")
        text = _find_all_text(result)
        assert "Warming up" in text

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_per_model_warming_when_n_records_below_24(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # xgboost has only 12 hours of records — too short for a meaningful 7d MAPE.
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(5.0, n_records=12),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        chips = _find_status_chips(result)
        assert chips == ["Warming"]


class TestStatusThresholds:
    """Status derives from the LIVE 7d MAPE's own governance grade
    (``mape_grade`` on the 7-day band), NOT a cross-horizon live÷holdout
    ratio (#217). 7d bands: excellent ≤6, target ≤9, acceptable ≤15,
    rollback >15."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_excellent_when_live_mape_low(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 5.0% → excellent (≤6), even though ×6.25 the 0.8% holdout.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(5.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=0.8)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Excellent"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_target_grade(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 8.0% → target (>6, ≤9)
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(8.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Target"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_acceptable_grade_not_flagged_degraded(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 12.0% → acceptable (>9, ≤15). The OLD ratio logic would have
        # cried "Degraded" here (×3 vs a 4.0 holdout); the #217 fix must not.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(12.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Acceptable"]
        assert "in rollback" not in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_rollback_when_live_mape_exceeds_acceptable(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 20.0% → rollback (>15) — the one genuinely bad state.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(20.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Rollback"]
        assert "rollback" in _find_all_text(result).lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_low_live_mape_never_degraded_by_cross_horizon_ratio(
        self, mock_metrics, mock_redis_get
    ):
        """#217 regression: a healthy live MAPE must NOT be flagged just
        because it is a large multiple of a (teacher-forced) holdout — the
        exact live scenario (XGBoost live 1.53% vs holdout 0.75%, ×2.04)."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(1.53))
        mock_metrics.return_value = _holdout_metrics(xgboost=0.75)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Excellent"]
        assert "in rollback" not in _find_all_text(result)


class TestMixedAndPartialStates:
    """Real production: rarely uniform. Status per model from its own live grade."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_mixed_states_render_independently(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live grades: xgb 4.2→excellent, prophet 20.0→rollback,
        # arima 8.0→target, ensemble 4.0→excellent.
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(4.2),
            prophet=_drift_model(20.0),
            arima=_drift_model(8.0),
            ensemble=_drift_model(4.0),
        )
        mock_metrics.return_value = _holdout_metrics(
            xgboost=4.0,
            prophet=11.0,
            arima=5.0,
            ensemble=3.8,
        )

        result = _build_drift_panel("PJM")
        chips = _find_status_chips(result)
        # Render order matches display_order: prophet, arima, xgboost, ensemble
        assert chips == ["Rollback", "Target", "Excellent", "Excellent"]
        # Headline reflects the one rollback model.
        assert "in rollback" in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_all_models_acceptable_or_better_headline_says_so(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # All live MAPEs ≤ 15 (acceptable or better) → no rollback headline.
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(4.0),
            prophet=_drift_model(13.0),
            arima=_drift_model(5.0),
            ensemble=_drift_model(3.8),
        )
        mock_metrics.return_value = _holdout_metrics(
            xgboost=4.0,
            prophet=11.0,
            arima=5.0,
            ensemble=3.8,
        )

        result = _build_drift_panel("PJM")
        text = _find_all_text(result)
        assert "acceptable performance or better" in text
        assert "in rollback" not in text

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_subset_of_models_in_drift_payload_still_renders(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # Only xgboost has drift records (e.g., Prophet failed to predict
        # in the previous tick). Panel should still render with just xgboost.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(4.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0, prophet=11.0)

        result = _build_drift_panel("PJM")
        chips = _find_status_chips(result)
        assert len(chips) == 1
        assert chips[0] == "Excellent"


class TestMissingHoldoutFallback:
    """Holdout MAPE missing (lost meta.json) → render live data without a comparison."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_no_holdout_renders_live_only(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # Drift data exists but holdout MAPE doesn't.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(5.0))
        mock_metrics.return_value = {}  # No holdout data

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Live only"]


class TestRedisIntegrationContract:
    """Verifies the panel reads from the exact Redis key the writer produces."""

    @patch("components._callbacks_models.redis_get")
    def test_reads_gridpulse_drift_key_for_region(self, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = None  # cache miss is fine for this assertion
        _build_drift_panel("ERCOT")

        # First positional arg is the Redis key.
        called_key = mock_redis_get.call_args[0][0]
        assert called_key == "gridpulse:drift:ERCOT"

    @patch("components._callbacks_models.redis_get")
    def test_defaults_to_fpl_when_region_is_none(self, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = None
        _build_drift_panel(None)

        called_key = mock_redis_get.call_args[0][0]
        assert called_key == "gridpulse:drift:FPL"


class TestPanelStructure:
    """Surface-level structural assertions — the panel's container/eyebrow exists."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_panel_has_gp_models_drift_panel_class(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(4.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert result.className == "gp-models-drift-panel"

    @patch("components._callbacks_models.redis_get")
    def test_warming_panel_also_has_panel_class(self, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = None

        result = _build_drift_panel("PJM")
        assert result.className == "gp-models-drift-panel"
        # Warming variant has a distinguishing id for CSS / test targeting.
        assert result.id == "drift-panel-warming"
