"""Regression tests for the loose-P1 remediation (2026-07 review).

- #201 (P1-7): absent metric fields on the Forecast model card must render as
  "—", not a fabricated positive-toned "MAPE 0.0%".
- #203 (P1-10): the US Grid "National Peak (24h)" must be the true simultaneous
  cross-BA peak (max over time of the demand sum), not the largest single BA.
- #199 (P1-5): the Forecast Generation panel must read Redis, never fetch EIA in
  the web request path under REQUIRE_REDIS.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# #203 — simultaneous national peak
# ---------------------------------------------------------------------------


class TestSimultaneousNationalPeak:
    def _peak(self, populated):
        from components._callbacks_us_grid import _simultaneous_national_peak_mw

        return _simultaneous_national_peak_mw(populated)

    def test_sums_across_bas_at_the_same_hour(self):
        # 51 BAs each peaking 9.5 GW simultaneously → 484.5 GW, not 9.5 GW.
        populated = {f"BA{i}": {"today_mw": [5000.0] * 23 + [9500.0]} for i in range(51)}
        peak = self._peak(populated)
        assert peak == 9500.0 * 51  # true simultaneous national peak

    def test_not_the_largest_single_ba(self):
        populated = {
            "A": {"today_mw": [50000.0] * 24},
            "B": {"today_mw": [70000.0] * 24},
        }
        # Old (wrong) metric = 70000 (largest single BA). New = 120000 (sum).
        assert self._peak(populated) == 120000.0

    def test_ignores_nonpositive_readings(self):
        populated = {
            "A": {"today_mw": [np.nan, 0.0, 40000.0]},
            "B": {"today_mw": [10000.0, 10000.0, 10000.0]},
        }
        # Aligned columns (right-aligned len 3): [10000, 10000, 50000] → 50000
        assert self._peak(populated) == 50000.0

    def test_empty_returns_zero(self):
        assert self._peak({}) == 0.0
        assert self._peak({"A": {"today_mw": []}}) == 0.0


# ---------------------------------------------------------------------------
# #199 — Generation panel is Redis-first, no EIA in the request path
# ---------------------------------------------------------------------------


class TestGenerationRedisFirst:
    def _redis_payload(self):
        ts = [t.isoformat() for t in pd.date_range("2026-06-20", periods=3, freq="h", tz="UTC")]
        return {
            "region": "ERCOT",
            "timestamps": ts,
            "gas": [20000.0, 21000.0, 22000.0],
            "wind": [5000.0, 4000.0, 6000.0],
            "renewable_pct": [20.0, 16.0, 21.0],
        }

    def test_reads_and_unpivots_redis_payload(self):
        import components._callbacks_overview as ov

        with patch.object(ov, "redis_get", return_value=self._redis_payload()):
            df = ov._fetch_generation_cached("ERCOT")

        assert df is not None
        assert set(df.columns) >= {"timestamp", "fuel_type", "generation_mw", "region"}
        assert set(df["fuel_type"]) == {"gas", "wind"}  # renewable_pct excluded
        assert len(df) == 6  # 2 fuels x 3 hours

    def test_require_redis_miss_returns_none_without_eia_fetch(self):
        import components._callbacks_overview as ov

        with (
            patch.object(ov, "REQUIRE_REDIS", True),
            patch.object(ov, "redis_get", return_value=None),
            patch("data.eia_client.fetch_generation_by_fuel") as eia,
        ):
            out = ov._fetch_generation_cached("ERCOT")

        assert out is None
        eia.assert_not_called()  # the guardrail: no EIA in the web request path


# ---------------------------------------------------------------------------
# #201 — no fabricated perfection on the Forecast model card
# ---------------------------------------------------------------------------


def _collect_text(node) -> str:
    parts: list[str] = []

    def walk(n):
        if n is None:
            return
        if isinstance(n, str):
            parts.append(n)
            return
        if isinstance(n, (list, tuple)):
            for c in n:
                walk(c)
            return
        walk(getattr(n, "children", None))

    walk(node)
    return " ".join(parts)


class TestForecastCardNoFabricatedPerfection:
    def test_partial_metrics_render_em_dash_not_zero(self):
        """The real update_outlook_model_card callback: a partial metric dict
        (mape/mae/r2 absent, rmse present) must not render 'MAPE 0.0%'."""
        import dash
        import dash_bootstrap_components as dbc

        app = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True
        )
        from components.layout import build_layout

        app.layout = build_layout()
        from components.callbacks import register_callbacks

        register_callbacks(app)
        callbacks = {}
        for _k, val in app.callback_map.items():
            fn = val.get("callback")
            if fn and hasattr(fn, "__name__"):
                callbacks[fn.__name__] = getattr(fn, "__wrapped__", fn)

        partial = {"xgboost": {"rmse": 900.0}}  # mape/mae/r2 dropped (#176/#179-era state)
        with (
            patch("models.model_service.get_model_metrics", return_value=partial),
            patch("models.model_service.is_trained", return_value=True),
        ):
            card = callbacks["update_outlook_model_card"]("ERCOT", "xgboost", "tab-outlook")

        text = _collect_text(card)
        assert "0.0%" not in text
        assert "—" in text
        assert "900 MW" in text  # the one real metric still shows
