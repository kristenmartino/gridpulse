"""P2-23 (#273): the Forecast tab's Generation panel "Net Load (avg)" hero
KPI must never silently substitute average TOTAL generation (a differently-
defined quantity) when demand data is missing/unparsable/misaligned — it
renders an honest degraded cell instead.

First coverage for ``_build_generation_panel``; the fallback branch was
previously untested, which is how the substitution shipped.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _gen_df(n_hours: int = 24, base_ts: str = "2026-07-01 00:00:00") -> pd.DataFrame:
    """Minimal generation frame: gas + wind + solar, hourly."""
    ts = pd.date_range(base_ts, periods=n_hours, freq="h")
    rows = []
    for t in ts:
        rows.append({"timestamp": t, "fuel_type": "natural_gas", "generation_mw": 30_000.0})
        rows.append({"timestamp": t, "fuel_type": "wind", "generation_mw": 5_000.0})
        rows.append({"timestamp": t, "fuel_type": "solar", "generation_mw": 2_000.0})
    return pd.DataFrame(rows)


def _demand_json(n_hours: int = 24, base_ts: str = "2026-07-01 00:00:00") -> str:
    ts = pd.date_range(base_ts, periods=n_hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "demand_mw": np.full(n_hours, 34_000.0)}).to_json(
        date_format="iso"
    )


class TestNetLoadKpiHonesty:
    @patch("components._callbacks_overview._fetch_generation_cached")
    def test_missing_demand_renders_degraded_cell_not_total_generation(self, mock_fetch):
        """No demand store (every cold/warming page load): the hero cell must
        show an em-dash + disclosure — NOT avg total generation (37,000 MW
        here) under the net-load label."""
        from components._callbacks_overview import _build_generation_panel

        mock_fetch.return_value = _gen_df()
        panel = str(_build_generation_panel("FPL", demand_json=None))

        assert "Net Load (avg)" in panel
        assert "—" in panel
        assert "demand data unavailable" in panel
        # The old lie: avg total generation (30k+5k+2k = 37,000) formatted
        # under the net-load label.
        assert "37,000" not in panel

    @patch("components._callbacks_overview._fetch_generation_cached")
    def test_misaligned_demand_renders_degraded_cell(self, mock_fetch):
        """Demand present but sharing <2 timestamps with generation — the
        alignment failure must also degrade honestly."""
        from components._callbacks_overview import _build_generation_panel

        mock_fetch.return_value = _gen_df(base_ts="2026-07-01 00:00:00")
        misaligned = _demand_json(base_ts="2026-06-01 00:00:00")
        panel = str(_build_generation_panel("FPL", demand_json=misaligned))

        assert "demand data unavailable" in panel
        assert "37,000" not in panel

    @patch("components._callbacks_overview._fetch_generation_cached")
    def test_aligned_demand_renders_real_net_load(self, mock_fetch):
        """Happy path: net load = demand − wind − solar = 34k − 5k − 2k = 27k."""
        from components._callbacks_overview import _build_generation_panel

        mock_fetch.return_value = _gen_df()
        panel = str(_build_generation_panel("FPL", demand_json=_demand_json()))

        assert "27,000" in panel
        assert "demand data unavailable" not in panel
