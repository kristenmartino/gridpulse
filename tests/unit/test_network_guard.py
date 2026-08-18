"""The network guard must be verifiably active.

`_no_network` in ``tests/conftest.py`` is what keeps the suite hermetic, and a
guard nobody confirms is running is indistinguishable from one that isn't —
the same reasoning CLAUDE.md applies to the mistake hooks. If someone removes
the fixture, weakens it to patch only ``requests``, or the autouse wiring
breaks, these two tests are what notices.

The history that makes this worth pinning: the suite previously made 79 live
calls per run to api.eia.gov and archive-api.open-meteo.com, and passed the
whole time.
"""

from __future__ import annotations

import socket

import pytest


class TestNetworkGuardIsActive:
    def test_outbound_connection_is_blocked(self) -> None:
        """A real connect must raise, not hang and not succeed."""
        with pytest.raises(Exception) as excinfo:
            socket.create_connection(("api.eia.gov", 443), timeout=5)

        message = str(excinfo.value).lower()
        assert "network" in message or "resolve" in message, (
            f"connection failed, but not because of the guard: {excinfo.value!r}. "
            "A plain connection error means the guard is not installed and the "
            "suite is only hermetic by luck."
        )

    def test_name_resolution_is_blocked(self) -> None:
        """Blocked before DNS, so no round trip is paid discovering it."""
        with pytest.raises(Exception) as excinfo:
            socket.getaddrinfo("archive-api.open-meteo.com", 443)

        assert "resolve" in str(excinfo.value).lower(), excinfo.value


class TestEscapeHatch:
    @pytest.mark.allow_network
    def test_marker_lifts_the_guard(self) -> None:
        """``allow_network`` must really opt out, or it is a trap for whoever needs it.

        Nothing in the suite uses this marker today. It is tested so that if
        someone ever does, it works as documented.
        """
        sock = socket.socket()
        sock.close()
