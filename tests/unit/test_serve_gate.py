"""The #326 serve-path acceptance gate (jobs/phases.py::serve_path_gate).

Daily retrains are a fit lottery — ~27% of persisted LDWP vintages produce
recursive forecasts that collapse overnight demand — and the published
holdout is provably blind to it. The gate replays the CANDIDATE pickle
through the real serve path at persist time and refuses the latest.json
repoint on a degenerate curve. Calibrated on real vintages
(docs/FORECAST_DIVE_DIAGNOSIS.md): rejects 0708/0710/0715/0717, accepts
0716/0718 — under which the live 1,302 MW dive never happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from jobs.phases import _gate_decision, serve_path_gate


class _EchoModel:
    """Predicts ``factor`` x the demand_lag_1h feature — factor 1.0 is a
    persistence model (sane on a near-flat series), factor < 1 compounds
    into the measured collapse through the recursion."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([row[0] * self.factor for row in x], dtype=float)


def _model(factor: float) -> dict:
    return {"model": _EchoModel(factor), "feature_names": ["demand_lag_1h"]}


def _featured(hours: int = 14 * 24, level: float = 3500.0) -> pd.DataFrame:
    """A near-flat featured frame — mild oscillation so a persistence echo
    tracks truth tightly and any collapse is unambiguous."""
    ts = pd.date_range("2026-07-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    demand = level + 100 * np.sin(2 * np.pi * np.arange(hours) / 24)
    demand = demand + rng.normal(0, 10, hours)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": demand,
            "demand_lag_1h": np.roll(demand, 1),
            "temperature_2m": 75.0,
        }
    )


class TestServeGate:
    def test_sane_candidate_passes(self):
        verdict = serve_path_gate(_model(1.0), _featured(), None, "TEST")
        assert verdict["passed"] is True
        judged = [a for a in verdict["anchors"] if "ok" in a]
        assert len(judged) == config.MODEL_GATE_PROBE_ANCHORS
        assert all(a["ok"] for a in judged)

    def test_collapsing_candidate_rejected(self):
        """factor 0.85 compounds to ~0.02x by hour 24 — the measured dive
        shape. The live anchor's trough floor must catch it."""
        verdict = serve_path_gate(_model(0.85), _featured(), None, "TEST")
        assert verdict["passed"] is False
        live = [a for a in verdict["anchors"] if "ok" in a and "truth_median_ape" not in a]
        assert live and not live[0]["ok"]
        assert live[0]["trough_ratio"] < config.MODEL_GATE_TROUGH_FRACTION

    def test_offset_anchors_judged_on_truth_not_the_band(self):
        """Offset anchors replay into known history: the verdict must carry
        truth-based fields (the calibration lesson — the trailing-week band
        false-rejects honest models during genuine demand dips)."""
        verdict = serve_path_gate(_model(1.0), _featured(), None, "TEST")
        offsets = [a for a in verdict["anchors"] if "truth_median_ape" in a]
        assert len(offsets) == config.MODEL_GATE_PROBE_ANCHORS - 1
        for a in offsets:
            assert a["truth_median_ape"] < config.MODEL_GATE_TRUTH_MEDIAN_APE_MAX
            assert "truth_trough_ratio" in a

    def test_exploding_candidate_rejected_by_level_ceiling(self):
        """The #296 lesson runs both directions — a runaway-growth fit must
        not pass just because its trough is high."""
        verdict = serve_path_gate(_model(1.08), _featured(), None, "TEST")
        assert verdict["passed"] is False

    def test_flag_off_skips(self, monkeypatch):
        monkeypatch.setitem(config.FEATURE_FLAGS, "model_serve_gate", False)
        verdict = serve_path_gate(_model(0.5), _featured(), None, "TEST")
        assert verdict == {"passed": True, "skipped": "flag_off", "anchors": []}

    def test_insufficient_history_fails_open(self):
        """A bootstrap region (thin frame) must not be frozen out of its
        first model."""
        verdict = serve_path_gate(_model(0.5), _featured(hours=100), None, "TEST")
        assert verdict["passed"] is True
        assert verdict["skipped"] == "insufficient_history"

    def test_erroring_replay_fails_open(self):
        """A gate harness bug must not freeze model rollout (fail-open by
        design; refusals only ever come from a judged replay)."""

        class _Boom:
            def predict(self, x):
                raise RuntimeError("boom")

        verdict = serve_path_gate(
            {"model": _Boom(), "feature_names": ["demand_lag_1h"]},
            _featured(),
            None,
            "TEST",
        )
        assert verdict["passed"] is True
        assert verdict["skipped"] == "insufficient_history"
        assert all("error" in a for a in verdict["anchors"])

    def test_verdict_is_json_serializable(self):
        import json

        verdict = serve_path_gate(_model(1.0), _featured(), None, "TEST")
        json.dumps(verdict)  # meta.json embeds it — must never raise


class TestGateDecision:
    """The decision rule, isolated: live failure rejects; one offset pocket
    is tolerated; a pattern of failures rejects."""

    LIVE_OK = {"ok": True}
    LIVE_BAD = {"ok": False}
    OFF_OK = {"ok": True, "truth_median_ape": 3.0}
    OFF_BAD = {"ok": False, "truth_median_ape": 3.0}

    def test_all_ok_passes(self):
        assert _gate_decision([self.LIVE_OK, self.OFF_OK, self.OFF_OK]) is True

    def test_live_failure_rejects_outright(self):
        assert _gate_decision([self.LIVE_BAD, self.OFF_OK, self.OFF_OK]) is False

    def test_single_offset_pocket_tolerated(self):
        """The 0716 case — one transient pocket, sane on the frame that
        matters; rejecting it would have been wrong (it was proven sane on
        the Jul-18 frame)."""
        assert _gate_decision([self.LIVE_OK, self.OFF_BAD, self.OFF_OK]) is True

    def test_two_offset_failures_reject(self):
        """The 0715 case — a pattern of pockets is systemic degeneracy."""
        assert _gate_decision([self.LIVE_OK, self.OFF_BAD, self.OFF_BAD]) is False


@pytest.fixture(autouse=True)
def _gate_flag_on(monkeypatch):
    monkeypatch.setitem(config.FEATURE_FLAGS, "model_serve_gate", True)
