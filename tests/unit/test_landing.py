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

    def test_accuracy_uses_the_sanctioned_per_ba_quoting_form(self, body) -> None:
        """Accuracy must be quoted per-BA, never as a pooled across-51 figure
        (CANONICAL_FACTS "Forecast accuracy" opens with that rule).

        The *values* are checked in
        ``test_public_copy_traces_to_canonical_facts.py``, which asserts each
        one against its source doc. This test only pins the framing — the
        earlier version asserted the literal ``"4.8%"`` here, which meant a
        stale number could not be corrected without a test change, and the
        source doc was never consulted at all.
        """
        assert "balancing authorities" in body
        assert "median per-BA" in body

    def test_in_product_module_names_only(self, body) -> None:
        """GP-P2-03: marketing copy uses the five real tab names."""
        for tab in ("Overview", "US Grid", "Forecast", "Risk", "Models"):
            assert tab in body

    def test_external_links_new_tab_internal_links_same_tab(self, body) -> None:
        """External GitHub links must not navigate visitors away from the
        page (target=_blank + rel=noopener); the product CTAs into ``/``
        stay same-tab."""
        import re

        external = re.findall(r'<a [^>]*href="https://github\.com[^"]*"[^>]*>', body)
        assert external, "expected external GitHub links on the page"
        for tag in external:
            assert 'target="_blank"' in tag and 'rel="noopener"' in tag, tag
        assert re.search(r'<a class="btn-primary" href="/">', body)
        assert 'href="/" target' not in body

    def test_focus_indicator_is_opaque(self, body) -> None:
        """--accent-ring is 30% alpha; as an outline it measures 1.46:1 on
        --bg-base and 1.50:1 on --bg-raised, against the 3:1 WCAG 1.4.11
        requires of a focus indicator. It shipped that way and a keyboard
        user could not see where they were on the page. The opaque accent
        measures 5.38:1 / 5.13:1 — and is what the dashboard already uses."""
        assert "outline: 2px solid var(--accent-base);" in body
        assert "outline: 2px solid var(--accent-ring);" not in body

    def test_every_focusable_element_is_covered_by_the_rule(self, body) -> None:
        """The page is almost entirely links, so a rule that missed anchors
        would leave the whole page unfocusable-looking. Pinning the selector
        catches a future <button> or <summary> being added without one."""
        import re as _re

        rule = _re.search(r"([^\n]*):focus-visible[^{]*\{[^}]*outline[^}]*\}", body)
        assert rule, "no focus-visible rule on the page"
        selectors = rule.group(0).split("{")[0]
        for tag in ("a", "button", "summary"):
            assert f"{tag}:focus-visible" in selectors, tag

    def test_posture_pins_no_commercial_language(self, body) -> None:
        """The BSC-era guardrail as a test: portfolio-neutral, nothing
        commercial, no combat claims (market-entry plan rule; archived
        spec postmortem). Flipping these later is a deliberate edit HERE.

        Scans the WHOLE document including structured data. The obvious
        SoftwareApplication completion is an ``offers`` node with
        ``priceCurrency`` — true, and encouraged by Google's docs — and it
        would have slipped past a prose-only check, because "pricecurrency"
        does not contain "pricing". A posture pin silently circumvented by
        machine-readable markup is worse than one that fails cleanly.
        """
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


class TestAboutBenchmarkClaim:
    """/about carries one live benchmark sentence, server-rendered from the
    same allow-listed payload /benchmark renders from — never hardcoded in
    the static HTML (the #535 lesson)."""

    def test_claim_is_injected_from_the_payload(self, client, monkeypatch) -> None:
        import api

        payload = {
            "fleet": {"median_gridpulse_mape": 4.08, "median_official_mape": 3.85, "n": 44},
            "regions": [],
        }
        monkeypatch.setattr(api, "build_benchmark_payload", lambda: payload)
        body = client.get("/about").get_data(as_text=True)
        assert '<p class="bench-claim">' in body
        assert "within 0.23 points" in body
        assert landing._CLAIM_MARKER not in body

    def test_claim_handles_the_winning_direction_too(self, client, monkeypatch) -> None:
        """Both directions derived — a flipped median must flip the sentence,
        not break it."""
        import api

        payload = {
            "fleet": {"median_gridpulse_mape": 3.5, "median_official_mape": 4.1, "n": 44},
            "regions": [],
        }
        monkeypatch.setattr(api, "build_benchmark_payload", lambda: payload)
        body = client.get("/about").get_data(as_text=True)
        assert "the closer forecast on the typical operator" in body

    def test_fails_open_when_the_payload_raises(self, client, monkeypatch) -> None:
        """Enrichment must never 500 the page: the card's qualitative copy
        stands on its own and the marker never leaks."""
        import api

        calls: list[int] = []

        def boom():
            calls.append(1)
            raise RuntimeError("redis down")

        monkeypatch.setattr(api, "build_benchmark_payload", boom)
        resp = client.get("/about")
        # The patch must actually have intercepted — a silently-defeated mock
        # would make this test pass without exercising the fail-open path.
        assert calls, "patched build_benchmark_payload was never called"
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert '<p class="bench-claim">' not in body
        assert landing._CLAIM_MARKER not in body


class TestMetricRobustness:
    """The verdict's metric-robustness sentence is derived, never asserted:
    "the same" only when both alternate statistics are computable and agree,
    a flip named when one disagrees, silence when the data cannot say."""

    @staticmethod
    def _row(g_med, o_med, g_wape, o_wape):
        return {
            "gridpulse": {"median_ape": g_med, "wape": g_wape},
            "official": {"median_ape": o_med, "wape": o_wape},
        }

    def test_agreement_on_both_statistics(self) -> None:
        rows = [self._row(4.0, 3.0, 4.1, 3.1), self._row(5.0, 4.0, 5.1, 4.1)]
        out = landing._metric_robustness(rows, we_lead=False)
        assert "the verdict is the same" in out

    def test_a_flip_is_named_not_hidden(self) -> None:
        # median APE favours us while the headline (we_lead=False) does not.
        rows = [self._row(2.0, 3.0, 4.1, 3.1), self._row(3.0, 4.0, 5.1, 4.1)]
        out = landing._metric_robustness(rows, we_lead=False)
        assert "median APE" in out and "flips" in out

    def test_silent_when_not_computable(self) -> None:
        # One row per arm is below the two-value floor — no claim either way.
        rows = [self._row(4.0, 3.0, 4.1, 3.1)]
        assert landing._metric_robustness(rows, we_lead=False) == ""
