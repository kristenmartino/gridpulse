"""Scenarios-panel baseline regression tests (#P2-31).

The Overview Scenarios panel's baseline forecast must come from the real scored
ensemble in Redis (``_read_ensemble_forecast_from_redis``), not
``model_service.get_forecasts`` — which on the stateless web tier is strict-
gated to "unavailable" in prod (a permanent "Awaiting baseline forecast"
dead-end) and echoed actuals as a fabricated-perfect forecast in dev.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _demand_json(n: int = 48) -> str:
    ts = pd.date_range("2026-06-20", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "demand_mw": 20000.0 + np.arange(n)}).to_json(
        date_format="iso"
    )


def _fig_annotations(fig) -> str:
    return " ".join(a.text or "" for a in fig.layout.annotations)


class TestScenariosBaseline:
    def test_uses_redis_reader_not_get_forecasts(self):
        import components._callbacks_overview as ov

        ts = pd.date_range("2026-06-22", periods=24, freq="h", tz="UTC")
        ensemble = np.full(24, 30000.0)
        with (
            patch.object(
                ov,
                "_read_ensemble_forecast_from_redis",
                return_value=(ts, ensemble, "2026-06-22T00:00:00Z"),
            ) as reader,
            patch("models.model_service.get_forecasts") as gf,
        ):
            kpi, fig = ov._build_scenarios_panel(5, 0, 0, "ERCOT", _demand_json())

        reader.assert_called_once()
        gf.assert_not_called()  # no disk-touching / strict-gated call in the request path
        # A real baseline → a real chart with traces, not the empty state.
        assert len(fig.data) > 0
        assert "Awaiting baseline forecast" not in _fig_annotations(fig)

    def test_redis_miss_shows_awaiting_not_fabricated(self):
        import components._callbacks_overview as ov

        with (
            patch.object(ov, "_read_ensemble_forecast_from_redis", return_value=None),
            patch("models.model_service.get_forecasts") as gf,
        ):
            kpi, fig = ov._build_scenarios_panel(5, 0, 0, "ERCOT", _demand_json())

        gf.assert_not_called()
        # Honest empty state, never a fabricated baseline.
        assert "Awaiting baseline forecast" in _fig_annotations(fig)
