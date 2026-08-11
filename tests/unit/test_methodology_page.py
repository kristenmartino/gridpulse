"""The /methodology page (landing.py + web/methodology.html).

The page is hand-converted from ``docs/HOW_IT_WORKS.md`` and committed rather
than rendered from it, because ``docs/`` is in ``.dockerignore`` and does not
exist in the production image. That decision buys deploy safety and costs
drift risk, so the drift is what these tests are mostly about: every figure
the page publishes is checked against the doc that sources it, and the
posture pins from ``/about`` and ``/benchmark`` apply here too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flask import Flask

import landing
from landing import landing_bp

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(landing_bp)
    return app.test_client()


@pytest.fixture
def body(client) -> str:
    return client.get("/methodology").get_data(as_text=True)


class TestMethodologyRoute:
    def test_serves_html_200(self, client) -> None:
        resp = client.get("/methodology")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")

    def test_iterable_cache_header(self, client) -> None:
        resp = client.get("/methodology")
        assert resp.headers["Cache-Control"] == "public, max-age=3600"
        assert "immutable" not in resp.headers["Cache-Control"]

    def test_missing_file_degrades_to_404(self, client, monkeypatch) -> None:
        """A dockerignore accident must 404 this route loudly rather than
        break app import — the reason _serve reads per request."""
        monkeypatch.setattr(landing, "_METHODOLOGY_HTML", Path("/nonexistent.html"))
        assert client.get("/methodology").status_code == 404

    def test_sibling_routes_still_serve(self, client) -> None:
        """Three routes now share _serve; a refactor there must not silently
        drop one (the regression guard test_benchmark_page.py:352 added)."""
        assert client.get("/about").status_code == 200
        assert client.get("/benchmark").status_code == 200


class TestNumbersTraceToSource:
    """Each figure is checked against the doc it came from.

    Same inversion as test_public_copy_traces_to_canonical_facts.py: when a
    retrain moves a value, the assertion fails on the SOURCE side and names
    this page, rather than the page quietly going stale.
    """

    @pytest.mark.parametrize(
        ("literal", "source"),
        [
            ("4.35%", "CANONICAL_FACTS.md"),  # served ensemble median
            ("3.69%", "CANONICAL_FACTS.md"),  # best-base median
            ("14.27%", "CANONICAL_FACTS.md"),  # ensemble p90
            ("9.87%", "CANONICAL_FACTS.md"),  # XGBoost-alone p90
            ("21 of 51", "CANONICAL_FACTS.md"),  # ensemble wins
            ("42 of 51", "CANONICAL_FACTS.md"),  # XGBoost best base
            ("+1.14 sMAPE", "HOW_IT_WORKS.md"),  # multi-point weather
            ("6.96%", "HOW_IT_WORKS.md"),  # the visibility-gate example
            ("38.63%", "CANONICAL_FACTS.md"),  # the withdrawn tail claim
        ],
    )
    def test_figure_is_on_the_page_and_in_its_source(self, body, literal, source) -> None:
        assert literal in body, f"{literal!r} missing from the page"
        doc = (_DOCS / source).read_text(encoding="utf-8")
        assert literal in doc, (
            f"{source} no longer contains {literal!r}, but web/methodology.html "
            f"still publishes it. Update the page, then this test."
        )

    def test_no_pooled_accuracy_figure(self, body) -> None:
        """CANONICAL_FACTS opens the accuracy section with a hard rule:
        accuracy is per-BA, never a single pooled across-51 number."""
        assert "per balancing authority" in body
        assert "across-51" in body  # the page states the rule explicitly

    def test_publishes_no_count_of_adrs_or_limits(self, body) -> None:
        """The failure mode that put a wrong number on /about twice."""
        import re

        assert not re.search(
            r"\b(ten|eight|nine|\d+)\s+(architecture decision|known limit)", body, re.I
        )


class TestPosture:
    def test_no_commercial_language(self, body) -> None:
        """The /about posture pins apply to every public page, not just the
        one they were written for."""
        lowered = body.lower()
        for banned in (
            "request a demo",
            "schedule a call",
            "contact sales",
            "pricing",
            "solutions",
        ):
            assert banned not in lowered, f"posture pin violated: {banned!r}"

    def test_no_combat_claims(self, body) -> None:
        """The benchmark page's additional pins. This page describes a system
        that loses to the operator on most major ISOs; superiority framing
        would be false as well as off-posture."""
        lowered = body.lower()
        for banned in ("beats the", "outperforms", "best-in-class", "state of the art"):
            assert banned not in lowered, f"posture pin violated: {banned!r}"

    def test_states_what_it_does_not_do(self, body) -> None:
        """A methodology page without its limits is marketing — the same
        standard the benchmark page is held to."""
        assert "What this does not do" in body
        for limit in ("drift between trainings", "Alaska or Hawaii", "not stable"):
            assert limit in body, limit

    def test_leads_with_the_withdrawn_claim_not_buries_it(self, body) -> None:
        """The ensemble section must present the counter-evidence, not tuck
        it behind the architecture description. A page voluntarily publishing
        that its headline design is worse than the simpler alternative is the
        most credible thing on the site; burying it defeats the point.
        """
        assert "withdrawn" in body
        section = body.split('id="ensemble-cost"')[1].split("</section>")[0]
        assert "withdrawn" in section
        assert "trails best-base" in section

    def test_external_links_new_tab_internal_links_same_tab(self, body) -> None:
        import re

        for tag in re.findall(r'<a [^>]*href="https://github\.com[^"]*"[^>]*>', body):
            assert 'target="_blank"' in tag and 'rel="noopener"' in tag, tag
        for tag in re.findall(r'<a [^>]*href="/[^"]*"[^>]*>', body):
            assert "target=" not in tag, tag

    def test_focus_indicator_is_opaque(self, body) -> None:
        """--accent-ring at 30% alpha measures ~1.5:1 as an outline, against
        the 3:1 WCAG 1.4.11 requires. Same pin as the other public pages."""
        assert "outline: 2px solid var(--accent-base);" in body
        assert "outline: 2px solid var(--accent-ring);" not in body


class TestLinkGraph:
    def test_about_now_links_here_rather_than_off_domain(self) -> None:
        """The reason this page exists: /about's two "how it works" links
        both pointed at github.com, so the site's strongest page spent its
        authority on a domain that does not need it."""
        about = (_ROOT / "web" / "landing.html").read_text(encoding="utf-8")
        assert 'href="/methodology"' in about
        assert "docs/HOW_IT_WORKS.md" not in about

    def test_page_links_onward(self, body) -> None:
        for href in ('href="/"', 'href="/about"', 'href="/benchmark"'):
            assert href in body, href

    def test_is_in_the_sitemap(self) -> None:
        from seo import PUBLIC_PAGES

        assert "/methodology" in {entry.path for entry in PUBLIC_PAGES}

    def test_is_in_the_noscript_nav(self) -> None:
        """Non-rendering agents reach it from `/` too, not only from /about."""
        import app as app_module

        noscript = app_module.app.index_string.split("<noscript>")[1]
        assert 'href="/methodology"' in noscript
