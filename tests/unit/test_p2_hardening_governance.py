"""P2 hardening + governance-honesty regression tests (2026-07 review).

- P2-24: the Overview models leaderboard tones each model by the H2 governance
  grade (``mape_grade`` on the 7d band), not ad-hoc 2.5/5.0 literals — so the
  live color matches governance for the very metric shown.
- P2-52: the ``/metrics`` IP allowlist reads the RIGHTMOST X-Forwarded-For hop
  (edge-appended), so a client-prepended ``127.0.0.1`` can't spoof past it.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# P2-24 — leaderboard tone follows mape_grade governance
# ---------------------------------------------------------------------------


class TestLeaderboardGovernanceTone:
    def test_tone_matches_mape_grade_not_inline_thresholds(self):
        from components._callbacks_overview import _leaderboard_mape_tone

        # 5.5% is "excellent" on the 7d band (excellent <= 6.0) → positive.
        # The old inline rule painted anything > 5.0 negative — the bug.
        assert _leaderboard_mape_tone(5.5) == "positive"
        # 1.8% clearly excellent → positive.
        assert _leaderboard_mape_tone(1.8) == "positive"
        # 12% is "acceptable" on 7d (<= 15) → secondary, not negative.
        assert _leaderboard_mape_tone(12.0) == "secondary"
        # 25% exceeds the 7d rollback (22) → negative.
        assert _leaderboard_mape_tone(25.0) == "negative"

    def test_agrees_with_config_mape_grade(self):
        from components._callbacks_overview import _MAPE_GRADE_TONE, _leaderboard_mape_tone
        from config import mape_grade

        for mape in (0.5, 6.0, 9.0, 15.0, 22.0, 30.0):
            assert _leaderboard_mape_tone(mape) == _MAPE_GRADE_TONE[mape_grade(mape, "7d")]


# ---------------------------------------------------------------------------
# P2-52 — /metrics XFF spoof
# ---------------------------------------------------------------------------


class TestUntrustedClientIp:
    """Pure spoof-resistant XFF resolution (runs everywhere — no app import)."""

    def test_prepended_localhost_does_not_win(self):
        from observability import untrusted_client_ip

        # Attacker prepends 127.0.0.1; trusted edge appends the real source.
        assert untrusted_client_ip("127.0.0.1, 203.0.113.7", "10.0.0.1") == "203.0.113.7"

    def test_rightmost_internal_hop_is_used(self):
        from observability import untrusted_client_ip

        assert untrusted_client_ip("203.0.113.7, 127.0.0.1", "10.0.0.1") == "127.0.0.1"

    def test_no_xff_falls_back_to_peer(self):
        from observability import untrusted_client_ip

        assert untrusted_client_ip("", "198.51.100.2") == "198.51.100.2"
        assert untrusted_client_ip(None, "198.51.100.2") == "198.51.100.2"

    def test_allowlist_gate_rejects_spoof_admits_real(self):
        """The exact check the /metrics route performs."""
        from observability import untrusted_client_ip

        allowed = {"127.0.0.1", "::1"}
        spoof = untrusted_client_ip("127.0.0.1, 203.0.113.7", "10.0.0.1")
        real_internal = untrusted_client_ip("203.0.113.7, 127.0.0.1", "10.0.0.1")
        assert spoof not in allowed  # 403
        assert real_internal in allowed  # 200


class TestMetricsEndpoint:
    """End-to-end via the flask client — runs in CI; skips if the Dash app
    import conflicts locally (same limitation as the existing infra tests)."""

    @pytest.fixture(autouse=True)
    def _server(self):
        try:
            from app import server
        except Exception as e:  # pragma: no cover — heavy import can conflict
            pytest.skip(f"cannot import app server: {e}")
        self.server = server

    def test_spoofed_leftmost_xff_is_rejected_in_prod(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("METRICS_ALLOWED_IPS", "127.0.0.1,::1")
        resp = self.server.test_client().get(
            "/metrics", headers={"X-Forwarded-For": "127.0.0.1, 203.0.113.7"}
        )
        assert resp.status_code == 403
