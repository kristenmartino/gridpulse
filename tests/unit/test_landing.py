"""The /about marketing landing route (landing.py + web/landing.html).

Beyond route mechanics, this file pins the page's POSTURE: the
portfolio-neutral, BSC-safe rules from the market-entry plan and the
archived spec's postmortem are asserted as tests, so commercial language
cannot drift in silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask

import landing
from landing import landing_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(landing_bp)
    return app.test_client()


class TestLandingRoute:
    def test_serves_html_200(self, client) -> None:
        resp = client.get("/about")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")

    def test_iterable_cache_header(self, client) -> None:
        """Deliberately NOT the assets-route 1-year immutable header — a
        marketing page must be iterable within an hour of a deploy."""
        resp = client.get("/about")
        assert resp.headers["Cache-Control"] == "public, max-age=3600"
        assert "immutable" not in resp.headers["Cache-Control"]

    def test_missing_file_degrades_to_404(self, client, monkeypatch) -> None:
        """A dockerignore accident must 404 this route loudly, never break
        app import (the file is read per request for exactly this)."""
        monkeypatch.setattr(landing, "_LANDING_HTML", Path("/nonexistent/landing.html"))
        assert client.get("/about").status_code == 404


class TestLandingContent:
    @pytest.fixture
    def body(self, client) -> str:
        return client.get("/about").get_data(as_text=True)

    def test_canonical_framing_present(self, body) -> None:
        """CLAUDE.md's canonical category + tagline, verbatim."""
        assert "Energy Intelligence Platform" in body
        assert "See demand sooner. Decide with confidence." in body

    def test_ctas_point_at_the_live_platform_and_public_docs(self, body) -> None:
        assert 'href="/"' in body
        assert "github.com/kristenmartino/gridpulse" in body

    def test_numbers_are_the_canonical_ones(self, body) -> None:
        """Every number traces to docs/CANONICAL_FACTS.md (51 BAs; 4.8%
        median per-BA served-ensemble holdout — the sanctioned quoting
        form, never a pooled across-51 figure)."""
        assert "balancing authorities" in body
        assert "51" in body
        assert "4.8%" in body
        assert "median per-BA" in body

    def test_in_product_module_names_only(self, body) -> None:
        """GP-P2-03: marketing copy uses the five real tab names."""
        for tab in ("Overview", "US Grid", "Forecast", "Risk", "Models"):
            assert tab in body

    def test_posture_pins_no_commercial_language(self, body) -> None:
        """The BSC-era guardrail as a test: portfolio-neutral, nothing
        commercial, no combat claims (market-entry plan rule; archived
        spec postmortem). Flipping these later is a deliberate edit HERE."""
        lowered = body.lower()
        for banned in (
            "request a demo",
            "schedule a call",
            "contact sales",
            "pricing",
            "beats the",
            "solutions",
        ):
            assert banned not in lowered, f"posture pin violated: {banned!r}"
