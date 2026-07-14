"""Unit tests for the Models-tab drift panel — #121 part 2.

Covers ``_build_drift_panel`` in ``components/_callbacks_models.py``.

The function reads:
- ``gridpulse:drift:{region}`` (per-model rolling 7d / 30d MAPE +
  records, written hourly by jobs.phases.write_drift_metrics in #121 part 1)
- ``models.model_service.get_model_metrics`` (per-model holdout MAPE
  from each trained pickle's meta.json)

…and renders a per-model table with 1h-grade chips classifying each model
by its LIVE 1h-ahead MAPE's own governance grade (``mape_grade`` on the
**1-hour-ahead** band — the drift metric is 1h-ahead). The former
live÷holdout ratio column was removed in #273: the holdout is the
training job's 168h recursive score — context at a different lead, not
comparable (band fixed to 1h per #217).

**Verdict-confirmation rule (#217 de-alarm):** a model above the 5% 1h
band only gets the "Rollback" verdict when its own 24h-ahead grade (from
``gridpulse:drift_horizon:{region}``, #227) is also ``rollback``;
otherwise it gets a descriptive "Off band" chip and the headline stays
non-alarming — a day-ahead model without a 1-hour anchor is expected to
sit above the 1h band.

Tests verify:
- Warming state when Redis has no drift entry
- Per-model warming when n_records < 24 (insufficient sample)
- 1h grades (excellent ≤1 / target ≤2.5 / acceptable ≤5) keyed to the
  live MAPE, NOT a cross-horizon ratio to the holdout
- The verdict-confirmation rule: >5% at 1h → "Off band" unless the 24h
  grade confirms rollback; headline alarms only on confirmed rollback
- Mixed state (some models excellent, others off band)
- Missing-holdout fallback ("Live only" when meta.json lost holdout MAPE)
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


def _horizon_payload(**grades_24h: str | None) -> dict:
    """Build a ``gridpulse:drift_horizon:{region}`` payload with 24h grades.

    ``_horizon_payload(prophet="acceptable")`` → prophet has a resolved,
    meaningful 24h block graded "acceptable".
    """
    models = {}
    for name, grade in grades_24h.items():
        if grade is not None:
            models[name] = {"24h": {"rolling_mape_7d": 4.0, "grade": grade, "n_records": 24}}
    return {"region": "PJM", "horizons": ["24h", "48h", "72h"], "models": models}


def _redis_router(drift: dict | None, horizon: dict | None = None):
    """side_effect routing the two Redis keys the panel reads."""

    def route(key: str):
        if "drift_horizon" in key:
            return horizon
        return drift

    return route


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
    """Status derives from the LIVE 1h-ahead MAPE's own governance grade
    (``mape_grade`` on the **1-hour-ahead** band — the drift metric is
    1h-ahead), NOT a cross-horizon live÷holdout ratio (#217; the ratio
    column itself was removed entirely in #273). 1h bands:
    excellent ≤1.0, target ≤2.5, acceptable ≤5.0, rollback >5.0."""


class TestLeadHonestLabels:
    """#273: the panel's columns carry their forecast leads and the ratio
    column is gone — a full or partial revert of the label fix must fail."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_headers_carry_leads_and_ratio_column_is_gone(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(0.8))
        mock_metrics.return_value = _holdout_metrics(xgboost=4.3)

        result = _build_drift_panel("PJM")
        text = _find_all_text(result)
        assert "Holdout (168h recursive)" in text
        assert "Live 1h-ahead (7d avg)" in text
        assert "Live 1h-ahead (30d avg)" in text
        assert "not directly comparable" in text
        assert "Live ÷ Holdout" not in text
        assert "×" not in text  # no rendered ratio values

        # No cell carries the retired ratio class anywhere in the tree.
        def _has_ratio_cell(node) -> bool:
            cls = getattr(node, "className", "") or ""
            if "gp-drift-cell--ratio" in cls:
                return True
            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                return any(_has_ratio_cell(c) for c in children)
            if children is not None and not isinstance(children, str):
                return _has_ratio_cell(children)
            return False

        assert not _has_ratio_cell(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_excellent_when_live_mape_low(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 0.8% → excellent (≤1.0), even though ×1.6 the 0.5% holdout.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(0.8))
        mock_metrics.return_value = _holdout_metrics(xgboost=0.5)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Excellent"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_target_grade(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 2.0% → target (>1.0, ≤2.5)
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(2.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=1.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Target"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_acceptable_grade_not_flagged_degraded(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 4.0% → acceptable (>2.5, ≤5.0). The OLD ratio logic would have
        # cried "Degraded" here (×4 vs a 1.0 holdout); the #217 fix must not.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(4.0))
        mock_metrics.return_value = _holdout_metrics(xgboost=1.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Acceptable"]
        assert "in rollback" not in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_rollback_only_when_24h_grade_confirms(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live 13.0% at 1h AND rollback at 24h → confirmed degradation:
        # the verdict fires and the headline alarms.
        mock_redis_get.side_effect = _redis_router(
            _drift_payload(xgboost=_drift_model(13.0)),
            _horizon_payload(xgboost="rollback"),
        )
        mock_metrics.return_value = _holdout_metrics(xgboost=2.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Rollback"]
        assert "investigate" in _find_all_text(result).lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_low_live_mape_never_degraded_by_cross_horizon_ratio(
        self, mock_metrics, mock_redis_get
    ):
        """#217 regression: XGBoost's real live 1.53% must NOT be flagged
        just because it is ×2.04 the (teacher-forced) 0.75% holdout — under
        the 1h band it is a healthy 'Target', never rollback."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(1.53))
        mock_metrics.return_value = _holdout_metrics(xgboost=0.75)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Target"]
        assert "in rollback" not in _find_all_text(result)


class TestVerdictConfirmationRule:
    """#217 de-alarm: >5% at 1h is a *measurement*; "Rollback" is a *verdict*
    that requires the model's own 24h grade to confirm. Prophet/SARIMAX
    lacking a 1h anchor must not be condemned by the 1h band alone."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_off_band_when_24h_grade_healthy(self, mock_metrics, mock_redis_get):
        """The live prod case (2026-07-06): Prophet 10.29% at 1h but
        Acceptable at 24h → descriptive "Off band", NOT Rollback, and the
        headline must not alarm."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.side_effect = _redis_router(
            _drift_payload(prophet=_drift_model(10.29)),
            _horizon_payload(prophet="acceptable"),
        )
        mock_metrics.return_value = _holdout_metrics(prophet=4.68)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Off band"]
        text = _find_all_text(result)
        assert "No model in confirmed rollback" in text
        # Copy must be honest about the ≥24h maturation lag — never
        # present-tense "healthy at 24h ahead" from lagged evidence.
        assert "matured 24h-ahead scores" in text
        assert "investigate" not in text.lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_fail_closed_when_horizon_feed_absent(self, mock_metrics, mock_redis_get):
        """The horizon feed writes hourly with a 24h TTL — a missing payload
        means it's been broken for a day+. An off-band model must then ALARM
        as unverified, not hide behind 'still resolving' forever (a dead
        pipeline must not silently disable the panel's only alarm path)."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.side_effect = _redis_router(
            _drift_payload(arima=_drift_model(10.86)),
            None,  # drift_horizon key gone → feed broken
        )
        mock_metrics.return_value = _holdout_metrics(arima=4.07)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Off band"]
        text = _find_all_text(result)
        assert "feed not reporting" in text
        assert "investigate" in text.lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_pending_when_feed_alive_but_model_grade_unresolved(self, mock_metrics, mock_redis_get):
        """Feed alive (payload present for other models) but THIS model's 24h
        grade hasn't matured → benign 'resolving' copy, no alarm."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.side_effect = _redis_router(
            _drift_payload(arima=_drift_model(10.86)),
            _horizon_payload(xgboost="target"),  # feed alive; no arima block yet
        )
        mock_metrics.return_value = _holdout_metrics(arima=4.07)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Off band"]
        text = _find_all_text(result)
        assert "still resolving" in text
        assert "investigate" not in text.lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_unknown_24h_grade_maps_to_pending_not_healthy(self, mock_metrics, mock_redis_get):
        """A grade string outside the governance vocabulary (e.g. a future
        writer's 'degraded') must NOT veto the alarm as health — it maps to
        the pending path."""
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.side_effect = _redis_router(
            _drift_payload(prophet=_drift_model(12.0)),
            _horizon_payload(prophet="degraded"),  # unknown vocabulary
        )
        mock_metrics.return_value = _holdout_metrics(prophet=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Off band"]
        text = _find_all_text(result)
        assert "still resolving" in text
        assert "healthy" not in text.lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_malformed_horizon_payload_does_not_crash(self, mock_metrics, mock_redis_get):
        """Non-dict intermediates in the horizon payload (models as a list,
        model block as a string) must degrade gracefully, not AttributeError
        the whole Models-tab render."""
        from components._callbacks_models import _build_drift_panel

        for bad_horizon in (
            {"models": ["not", "a", "dict"]},
            {"models": {"prophet": "not-a-dict"}},
            {"models": {"prophet": {"24h": ["not-a-dict"]}}},
        ):
            mock_redis_get.side_effect = _redis_router(
                _drift_payload(prophet=_drift_model(12.0)), bad_horizon
            )
            mock_metrics.return_value = _holdout_metrics(prophet=4.0)
            result = _build_drift_panel("PJM")  # must not raise
            assert _find_status_chips(result) == ["Off band"]

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_immature_24h_block_counts_as_unresolved(self, mock_metrics, mock_redis_get):
        """A 24h block with n_records below the meaningfulness threshold (6)
        must not confirm a verdict either way."""
        from components._callbacks_models import _build_drift_panel

        horizon = _horizon_payload(prophet="rollback")
        horizon["models"]["prophet"]["24h"]["n_records"] = 3  # immature
        mock_redis_get.side_effect = _redis_router(
            _drift_payload(prophet=_drift_model(12.0)), horizon
        )
        mock_metrics.return_value = _holdout_metrics(prophet=4.0)

        result = _build_drift_panel("PJM")
        assert _find_status_chips(result) == ["Off band"]
        assert "still resolving" in _find_all_text(result)

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_chip_tooltip_points_to_horizon_panel(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        mock_redis_get.side_effect = _redis_router(
            _drift_payload(prophet=_drift_model(10.29)),
            _horizon_payload(prophet="target"),
        )
        mock_metrics.return_value = _holdout_metrics(prophet=4.68)

        result = _build_drift_panel("PJM")

        # Find the chip span and check its title mentions the horizon panel.
        titles: list[str] = []

        def walk(node):
            if (
                hasattr(node, "className")
                and node.className
                and "gp-status-chip--" in (node.className or "")
            ):
                t = getattr(node, "title", None)
                if t:
                    titles.append(t)
            children = getattr(node, "children", None)
            if isinstance(children, (list, tuple)):
                for c in children:
                    walk(c)
            elif children is not None and not isinstance(children, str):
                walk(children)

        walk(result)
        assert titles and "Drift by Horizon" in titles[0]
        assert "target" in titles[0]


class TestMixedAndPartialStates:
    """Real production: rarely uniform. Status per model from its own live grade."""

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_mixed_states_render_independently(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # live grades (1h band): xgb 0.8→excellent, prophet 13.0→off band
        # (24h grade healthy → no verdict), arima 4.0→acceptable,
        # ensemble 2.0→target.
        mock_redis_get.side_effect = _redis_router(
            _drift_payload(
                xgboost=_drift_model(0.8),
                prophet=_drift_model(13.0),
                arima=_drift_model(4.0),
                ensemble=_drift_model(2.0),
            ),
            _horizon_payload(prophet="acceptable"),
        )
        mock_metrics.return_value = _holdout_metrics(
            xgboost=0.75,
            prophet=3.0,
            arima=1.8,
            ensemble=0.9,
        )

        result = _build_drift_panel("PJM")
        chips = _find_status_chips(result)
        # Render order matches display_order: prophet, arima, xgboost, ensemble
        assert chips == ["Off band", "Acceptable", "Excellent", "Target"]
        # Headline explains the 1h excursion without alarming.
        text = _find_all_text(result)
        assert "No model in confirmed rollback" in text
        assert "investigate" not in text.lower()

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_all_models_acceptable_or_better_headline_says_so(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # All live MAPEs ≤ 5.0 (acceptable or better at 1h) → no rollback headline.
        mock_redis_get.return_value = _drift_payload(
            xgboost=_drift_model(1.0),
            prophet=_drift_model(4.5),
            arima=_drift_model(2.0),
            ensemble=_drift_model(3.0),
        )
        mock_metrics.return_value = _holdout_metrics(
            xgboost=0.75,
            prophet=3.0,
            arima=1.8,
            ensemble=0.9,
        )

        result = _build_drift_panel("PJM")
        text = _find_all_text(result)
        assert "acceptable or better" in text
        assert "in rollback" not in text

    @patch("components._callbacks_models.redis_get")
    @patch("models.model_service.get_model_metrics")
    def test_subset_of_models_in_drift_payload_still_renders(self, mock_metrics, mock_redis_get):
        from components._callbacks_models import _build_drift_panel

        # Only xgboost has drift records (e.g., Prophet failed to predict
        # in the previous tick). Panel should still render with just xgboost.
        mock_redis_get.return_value = _drift_payload(xgboost=_drift_model(0.8))
        mock_metrics.return_value = _holdout_metrics(xgboost=0.75, prophet=3.0)

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
