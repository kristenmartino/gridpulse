"""Unit tests for the Models-tab per-horizon drift panel — #227.

Covers ``_build_horizon_drift_panel`` in ``components/_callbacks_models.py``,
which reads ``gridpulse:drift_horizon:{region}`` and renders a Model x Horizon
table where each cell is graded against its OWN band (grade precomputed in the
payload). The point of #227 is visible here: a model that grades poorly at 1h
can grade well at day-ahead.
"""

from __future__ import annotations

from unittest.mock import patch

PATCH = "components._callbacks_models.redis_get"


def _block(mape, grade, n=168):
    return {
        "rolling_mape_7d": mape,
        "rolling_smape_7d": mape,
        "rolling_mape_30d": mape,
        "n_records": n,
        "grade": grade,
        "records": [],
    }


def _payload(models, *, region="PJM", horizons=("24h", "48h", "72h")):
    return {
        "region": region,
        "last_updated_at": "2026-07-04T00:00:00+00:00",
        "horizons": list(horizons),
        "pending": [],
        "models": models,
    }


def _chips(div):
    """Inner text of every rendered status chip."""
    found: list[str] = []

    def walk(node):
        if (
            hasattr(node, "className")
            and node.className
            and "gp-status-chip--" in (node.className or "")
            and isinstance(node.children, str)
        ):
            found.append(node.children)
        ch = getattr(node, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None and not isinstance(ch, str):
            walk(ch)

    walk(div)
    return found


def _count_tone(div, tone: str) -> int:
    n = 0

    def walk(node):
        nonlocal n
        if hasattr(node, "className") and f"gp-status-chip--{tone}" in (node.className or ""):
            n += 1
        ch = getattr(node, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None and not isinstance(ch, str):
            walk(ch)

    walk(div)
    return n


def _text(div):
    out: list[str] = []

    def walk(node):
        if isinstance(node, str):
            out.append(node)
            return
        ch = getattr(node, "children", None)
        if isinstance(ch, str):
            out.append(ch)
        elif isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None:
            walk(ch)

    walk(div)
    return " ".join(out)


class TestHorizonDriftPanel:
    def test_warming_when_no_payload(self):
        from components._callbacks_models import _build_horizon_drift_panel

        with patch(PATCH, return_value=None):
            div = _build_horizon_drift_panel("PJM")
        assert "Warming up" in _text(div)
        assert div.id == "horizon-drift-panel-warming"

    def test_renders_grades_and_horizon_columns(self):
        from components._callbacks_models import _build_horizon_drift_panel

        models = {
            "xgboost": {
                "24h": _block(2.0, "excellent"),
                "48h": _block(3.5, "target"),
                "72h": _block(6.0, "acceptable"),
            }
        }
        with patch(PATCH, return_value=_payload(models)):
            div = _build_horizon_drift_panel("PJM")
        chips = _chips(div)
        assert "Excellent" in chips and "Target" in chips and "Acceptable" in chips
        txt = _text(div)
        assert "24h ahead" in txt and "48h ahead" in txt and "72h ahead" in txt
        assert "2.00%" in txt  # the 24h MAPE renders

    def test_vindication_model_good_at_day_ahead(self):
        # #227's whole point: a model graded poorly at 1h grades well day-ahead.
        from components._callbacks_models import _build_horizon_drift_panel

        models = {
            "prophet": {
                "24h": _block(1.8, "excellent"),
                "48h": _block(2.9, "target"),
                "72h": _block(3.8, "target"),
            }
        }
        with patch(PATCH, return_value=_payload(models)):
            div = _build_horizon_drift_panel("PJM")
        chips = _chips(div)
        assert "Excellent" in chips
        assert "Rollback" not in chips  # NOT condemned at these horizons

    def test_per_horizon_warming_below_min_records(self):
        from components._callbacks_models import _build_horizon_drift_panel

        models = {
            "xgboost": {
                "24h": _block(2.0, "excellent", n=200),
                "48h": _block(3.0, "target", n=2),  # < _MIN_N -> Warming
                "72h": _block(4.0, "target", n=0),  # -> "—"
            }
        }
        with patch(PATCH, return_value=_payload(models)):
            div = _build_horizon_drift_panel("PJM")
        chips = _chips(div)
        assert "Excellent" in chips  # 24h graded
        assert "Warming" in chips  # 48h warming
        assert "—" in chips  # 72h no records

    def test_rollback_tone_negative(self):
        from components._callbacks_models import _build_horizon_drift_panel

        models = {
            "arima": {
                "24h": _block(20.0, "rollback"),
                "48h": _block(22.0, "rollback"),
                "72h": _block(25.0, "rollback"),
            }
        }
        with patch(PATCH, return_value=_payload(models)):
            div = _build_horizon_drift_panel("PJM")
        assert _count_tone(div, "negative") == 3  # all three horizons flagged
