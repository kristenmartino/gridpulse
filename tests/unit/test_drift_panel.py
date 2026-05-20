"""Unit tests for the Models-tab drift panel — #121 part 2.

Covers ``_build_drift_panel`` in ``components/_callbacks_models.py``.

The function reads:
- ``gridpulse:drift:{region}`` (per-model rolling 7d / 30d MAPE +
  records, written hourly by jobs.phases.write_drift_metrics in #121 part 1)
- ``models.model_service.get_model_metrics`` (per-model holdout MAPE
  from each trained pickle's meta.json)

…and renders a per-model table with status chips classifying each model
as on-track / drifting / degraded relative to its holdout baseline.

Tests verify:
- Warming state when Redis has no drift entry
- Per-model warming when n_records < 24 (insufficient sample)
- Status thresholds (≤1.10× → positive, ≤1.50× → warning, >1.50× → negative)
- Mixed state (some models healthy, others drifting)
- Missing-holdout fallback ("Live only" when meta.json lost holdout MAPE)
- Headline summary text matches whether any model is drifting
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
    """Live ÷ holdout boundaries: ≤1.10 on track, ≤1.50 drifting, >1.50 degraded."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_on_track_when_live_within_10pct_of_holdout(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # holdout=4.0, live=4.4 → ratio=1.10, on track (boundary case)
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(4.4),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["On track"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_drifting_at_1_25x(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # holdout=4.0, live=5.0 → ratio=1.25, drifting
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(5.0),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Drifting"]
        assert "drifting vs holdout" in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_drifting_at_exactly_1_50x_boundary(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # holdout=4.0, live=6.0 → ratio=1.50, still drifting (boundary)
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(6.0),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Drifting"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_degraded_above_1_50x(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # holdout=4.0, live=7.0 → ratio=1.75, degraded
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(7.0),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Degraded"]
        assert "drifting vs holdout" in _find_all_text(result)


class TestMixedAndPartialStates:
    """Real production: rarely uniform. Some models track, others drift."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_mixed_states_render_independently(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # XGBoost on track (4.0 → 4.2), Prophet drifting (11.0 → 14.0),
        # ARIMA degraded (5.0 → 9.0), Ensemble on track (3.8 → 4.0).
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(4.2),
            prophet=_drift_model(14.0),
            arima=_drift_model(9.0),
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
        assert chips == ["Drifting", "Degraded", "On track", "On track"]
        # Headline reflects any-drifting state
        assert "drifting vs holdout" in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_all_models_on_track_headline_says_so(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(4.0),
            prophet=_drift_model(11.0),
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
        assert "tracking within ±10" in text
        assert "drifting vs holdout" not in text

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
        assert chips[0] == "On track"


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
