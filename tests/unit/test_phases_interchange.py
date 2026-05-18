"""Unit tests for the V3.α interchange phase in jobs/phases.py.

Covers ``write_interchange``'s Redis payload shape:

- Empty fetch → placeholder payload with ``net_mw=None`` so the UI
  renders "—" rather than guessing.
- Populated fetch → top-3 counterparties by absolute interchange,
  signed net (+ export / - import), latest-hour timestamp.
- API error → PhaseResult.ok=False, no Redis write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_redis(monkeypatch):
    """Capture redis_set writes in an in-memory dict."""
    store: dict[str, dict] = {}

    def _set(key: str, value, ttl: int = 86400) -> bool:
        store[key] = value
        return True

    import data.redis_client as rc

    monkeypatch.setattr(rc, "redis_set", _set)
    return store


@pytest.fixture
def stub_eia_key(monkeypatch):
    """Pretend EIA_API_KEY is configured so the phase doesn't bail early."""
    import jobs.phases as phases

    monkeypatch.setattr(phases, "_has_eia_key", lambda: True)


def _interchange_df(rows):
    """Build a DataFrame matching ``fetch_interchange``'s return shape."""
    return pd.DataFrame(
        rows,
        columns=["timestamp", "from_ba", "to_ba", "interchange_mw"],
    )


class TestWriteInterchange:
    def test_empty_dataframe_writes_placeholder(self, fake_redis, stub_eia_key, monkeypatch):
        """Empty fetch → Redis row with net_mw=None and empty counterparties."""
        import data.eia_client as eia
        from jobs import phases

        monkeypatch.setattr(eia, "fetch_interchange", lambda region: _interchange_df([]))

        result = phases.write_interchange("PJM")

        assert result.ok is True
        payload = fake_redis["gridpulse:interchange:PJM:1h"]
        assert payload["region"] == "PJM"
        assert payload["net_mw"] is None
        assert payload["counterparties"] == []
        assert payload["latest_hour"] is None
        assert "scored_at" in payload

    def test_populated_picks_top3_by_absolute_mw(self, fake_redis, stub_eia_key, monkeypatch):
        """Top-3 selection respects sign — biggest |MW| first, sign preserved."""
        import data.eia_client as eia
        from jobs import phases

        latest = pd.Timestamp("2026-05-01T09:00:00Z")
        rows = [
            (latest, "PJM", "MISO", -1200.5),  # importing 1.2 GW
            (latest, "PJM", "NYISO", -800.0),  # importing 0.8 GW
            (latest, "PJM", "DUK", 350.0),  # exporting 0.35 GW
            (latest, "PJM", "TVA", 25.0),  # tiny export — should drop
        ]
        monkeypatch.setattr(
            eia,
            "fetch_interchange",
            lambda region: _interchange_df(rows),
        )

        result = phases.write_interchange("PJM")

        assert result.ok is True
        payload = fake_redis["gridpulse:interchange:PJM:1h"]
        assert payload["latest_hour"] == latest.isoformat()
        # Net = sum of all four = -1625.5
        assert payload["net_mw"] == pytest.approx(-1625.5, abs=0.01)
        cps = payload["counterparties"]
        assert len(cps) == 3
        # Top by |MW|: MISO -1200.5, NYISO -800, DUK 350
        assert [cp["to_ba"] for cp in cps] == ["MISO", "NYISO", "DUK"]
        assert cps[0]["mw"] == pytest.approx(-1200.5)
        assert cps[2]["mw"] == pytest.approx(350.0)

    def test_only_latest_hour_aggregated(self, fake_redis, stub_eia_key, monkeypatch):
        """Older rows in the DataFrame must not contaminate the snapshot."""
        import data.eia_client as eia
        from jobs import phases

        old = pd.Timestamp("2026-05-01T07:00:00Z")
        latest = pd.Timestamp("2026-05-01T09:00:00Z")
        rows = [
            # Old hour — should be ignored
            (old, "PJM", "MISO", 9999.0),
            # Latest hour — only this should drive the snapshot
            (latest, "PJM", "MISO", -500.0),
            (latest, "PJM", "NYISO", -250.0),
        ]
        monkeypatch.setattr(
            eia,
            "fetch_interchange",
            lambda region: _interchange_df(rows),
        )

        phases.write_interchange("PJM")
        payload = fake_redis["gridpulse:interchange:PJM:1h"]
        assert payload["latest_hour"] == latest.isoformat()
        assert payload["net_mw"] == pytest.approx(-750.0, abs=0.01)
        assert payload["counterparties"][0]["mw"] == pytest.approx(-500.0)

    def test_dataframe_with_nan_drops_nan_rows(self, fake_redis, stub_eia_key, monkeypatch):
        """NaN interchange values are dropped before aggregation."""
        import data.eia_client as eia
        from jobs import phases

        latest = pd.Timestamp("2026-05-01T09:00:00Z")
        rows = [
            (latest, "PJM", "MISO", -1000.0),
            (latest, "PJM", "NYISO", np.nan),  # dropped
        ]
        monkeypatch.setattr(
            eia,
            "fetch_interchange",
            lambda region: _interchange_df(rows),
        )

        phases.write_interchange("PJM")
        payload = fake_redis["gridpulse:interchange:PJM:1h"]
        assert payload["net_mw"] == pytest.approx(-1000.0)
        assert len(payload["counterparties"]) == 1
        assert payload["counterparties"][0]["to_ba"] == "MISO"

    def test_fetch_failure_returns_phase_failure_no_write(
        self, fake_redis, stub_eia_key, monkeypatch
    ):
        """Fetch raises → PhaseResult.ok=False, nothing written to Redis."""
        import data.eia_client as eia
        from jobs import phases

        def _broken_fetch(region):
            raise RuntimeError("synthetic EIA outage")

        monkeypatch.setattr(eia, "fetch_interchange", _broken_fetch)

        result = phases.write_interchange("PJM")

        assert result.ok is False
        assert "synthetic EIA outage" in (result.error or "")
        assert "gridpulse:interchange:PJM:1h" not in fake_redis

    def test_no_eia_api_key_returns_failure(self, fake_redis, monkeypatch):
        """No API key → bail before fetch attempt."""
        from jobs import phases

        monkeypatch.setattr(phases, "_has_eia_key", lambda: False)

        result = phases.write_interchange("PJM")

        assert result.ok is False
        assert result.error == "no_eia_api_key"
        assert "gridpulse:interchange:PJM:1h" not in fake_redis


class TestInterchangeChipRender:
    """Cover the UI-side ``_build_interchange_chip`` rendering logic.

    The chip lives in components/callbacks.py because Dash callbacks
    reference it directly; tests here are pure-function shape asserts.
    """

    def test_none_payload_returns_none(self):
        from components.callbacks import _build_interchange_chip

        assert _build_interchange_chip(None) is None

    def test_payload_with_null_net_returns_none(self):
        from components.callbacks import _build_interchange_chip

        assert _build_interchange_chip({"net_mw": None, "counterparties": []}) is None

    def test_export_renders_with_plus_sign(self):
        from components.callbacks import _build_interchange_chip

        chip = _build_interchange_chip(
            {
                "net_mw": 1234.5,
                "counterparties": [{"to_ba": "MISO", "mw": 1234.5}],
            }
        )
        assert chip is not None
        assert "export" in chip.className
        # Visible label uses ASCII-friendly + sign for export
        assert chip.children == "+1.2 GW"

    def test_import_renders_with_minus_sign(self):
        from components.callbacks import _build_interchange_chip

        chip = _build_interchange_chip(
            {
                "net_mw": -1234.5,
                "counterparties": [{"to_ba": "MISO", "mw": -1234.5}],
            }
        )
        assert chip is not None
        assert "import" in chip.className
        # Minus uses unicode "−" (U+2212), the typographically correct minus
        assert chip.children == "−1.2 GW"

    def test_near_zero_renders_neutral(self):
        from components.callbacks import _build_interchange_chip

        chip = _build_interchange_chip(
            {"net_mw": 12.0, "counterparties": [{"to_ba": "MISO", "mw": 12.0}]}
        )
        assert chip is not None
        assert "neutral" in chip.className
        assert chip.children == "≈0"

    def test_tooltip_lists_counterparties(self):
        from components.callbacks import _build_interchange_chip

        chip = _build_interchange_chip(
            {
                "net_mw": -2000.0,
                "counterparties": [
                    {"to_ba": "MISO", "mw": -1500.0},
                    {"to_ba": "NYISO", "mw": -500.0},
                ],
            }
        )
        # Plotly's `title` kwarg is the hover tooltip
        tooltip = chip.title
        assert "MISO" in tooltip
        assert "NYISO" in tooltip
        # Top counterparty appears first
        assert tooltip.find("MISO") < tooltip.find("NYISO")
