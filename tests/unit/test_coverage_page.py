"""The /coverage page (landing.py + web/coverage.html).

The table is server-rendered from ``config`` rather than committed as markup,
which is the opposite choice from ``/methodology``. Both are right for their
inputs: ``docs/`` is not in the production image, so a methodology page had to
be committed; ``config.py`` *is*, so generating the 51-row table from it makes
disagreement with what the product covers structurally impossible.

These tests exist mostly to keep that property honest — the table must reflect
``REGION_NAMES``, not a snapshot of it — and to hold the page to the same
posture pins as every other public surface.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from flask import Flask

import landing
from landing import landing_bp

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(landing_bp)
    return app.test_client()


@pytest.fixture
def body(client) -> str:
    return client.get("/coverage").get_data(as_text=True)


def _row_codes(html: str) -> list[str]:
    return re.findall(r'<tr><th scope="row"><code>([A-Z0-9]+)</code></th>', html)


class TestCoverageRoute:
    def test_serves_html_200(self, client) -> None:
        resp = client.get("/coverage")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")

    def test_iterable_cache_header(self, client) -> None:
        resp = client.get("/coverage")
        assert resp.headers["Cache-Control"] == "public, max-age=3600"
        assert "immutable" not in resp.headers["Cache-Control"]

    def test_missing_file_degrades_to_404(self, client, monkeypatch) -> None:
        monkeypatch.setattr(landing, "_COVERAGE_HTML", Path("/nonexistent.html"))
        assert client.get("/coverage").status_code == 404

    def test_sibling_routes_still_serve(self, client) -> None:
        for path in ("/about", "/methodology", "/benchmark"):
            assert client.get(path).status_code == 200, path

    def test_render_failure_never_500s_the_page(self, client, monkeypatch) -> None:
        """Enrichment fails open, as on /benchmark: the prose stands on its
        own, and a page that 500s because a table failed is worse than one
        that is briefly missing it."""
        monkeypatch.setattr(
            landing,
            "_render_coverage_table",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        resp = client.get("/coverage")
        assert resp.status_code == 200
        assert "SSR_COVERAGE_TABLE" not in resp.get_data(as_text=True)


class TestTableMatchesConfig:
    """The whole reason this table is generated rather than written."""

    def test_every_covered_region_has_a_row(self, body) -> None:
        from config import REGION_NAMES

        assert set(_row_codes(body)) == set(REGION_NAMES)

    def test_row_count_matches_the_canonical_count(self, body) -> None:
        from config import REGION_NAMES

        codes = _row_codes(body)
        assert len(codes) == len(REGION_NAMES) == 51
        assert len(set(codes)) == len(codes), "duplicate row"

    def test_rows_are_sorted(self, body) -> None:
        """A stable order keeps the diff readable when coverage changes."""
        codes = _row_codes(body)
        assert codes == sorted(codes)

    def test_peak_derived_rows_are_marked_as_estimates(self, body) -> None:
        """The seven import-heavy authorities whose capacity is peak x 1.15.
        Publishing that as a plain nameplate figure would be the actual
        inaccuracy."""
        from config import PEAK_DERIVED_CAPACITY

        assert len(PEAK_DERIVED_CAPACITY) == 7
        for code in PEAK_DERIVED_CAPACITY:
            row = re.search(rf'<tr><th scope="row"><code>{code}</code></th>.*?</tr>', body, re.S)
            assert row, f"no row for {code}"
            assert 'class="est"' in row.group(0), f"{code} not marked as an estimate"

    def test_nameplate_rows_are_not_marked(self, body) -> None:
        from config import PEAK_DERIVED_CAPACITY, REGION_NAMES

        for code in sorted(set(REGION_NAMES) - set(PEAK_DERIVED_CAPACITY))[:6]:
            row = re.search(rf'<tr><th scope="row"><code>{code}</code></th>.*?</tr>', body, re.S)
            assert 'class="est"' not in row.group(0), f"{code} wrongly marked estimate"

    def test_states_come_from_config(self, body) -> None:
        from config import STATE_TO_BA

        row = re.search(r'<tr><th scope="row"><code>PJM</code></th>.*?</tr>', body, re.S)
        for state in STATE_TO_BA["PJM"]:
            assert state in row.group(0), state


class TestHonestCoverageClaims:
    def test_states_both_coverage_percentages(self, body) -> None:
        """81% by BA count and ~100% by demand describe the same coverage.
        Quoting only the flattering one would be the omission."""
        assert "81%" in body
        assert "100%" in body
        assert "63" in body  # the denominator

    def test_says_the_multiplier_is_not_a_reserve_margin(self, body) -> None:
        """CANONICAL_FACTS is explicit about this and it is the single most
        misreadable number on the page."""
        assert "not a reserve margin" in body.lower()

    def test_names_what_is_not_covered(self, body) -> None:
        assert "Alaska and Hawaii" in body
        assert "not covered" in body.lower() or "What is not covered" in body

    def test_does_not_publish_accuracy(self, body) -> None:
        """Accuracy is per-BA and moves every retrain. Copying it here would
        create a second place to go stale — the failure this whole effort
        started from. The page links to the benchmark instead."""
        assert not re.search(r"\bMAPE\b", body)
        assert 'href="/benchmark"' in body


class TestPosture:
    def test_no_commercial_language(self, body) -> None:
        lowered = body.lower()
        for banned in ("request a demo", "schedule a call", "contact sales", "pricing"):
            assert banned not in lowered, f"posture pin violated: {banned!r}"

    def test_no_combat_claims(self, body) -> None:
        lowered = body.lower()
        for banned in ("beats the", "outperforms", "best-in-class", "state of the art"):
            assert banned not in lowered, f"posture pin violated: {banned!r}"

    def test_external_links_new_tab_internal_links_same_tab(self, body) -> None:
        for tag in re.findall(r'<a [^>]*href="https://github\.com[^"]*"[^>]*>', body):
            assert 'target="_blank"' in tag and 'rel="noopener"' in tag, tag
        for tag in re.findall(r'<a [^>]*href="/[^"]*"[^>]*>', body):
            assert "target=" not in tag, tag

    def test_focus_indicator_is_opaque(self, body) -> None:
        assert "outline: 2px solid var(--accent-base);" in body
        assert "outline: 2px solid var(--accent-ring);" not in body

    def test_wide_table_scrolls_in_its_own_container(self, body) -> None:
        """A 5-column, 51-row table must not make the page body scroll
        horizontally on a phone."""
        assert "table-scroll" in body
        assert "overflow-x: auto" in body


class TestLinkGraph:
    def test_is_in_the_sitemap(self) -> None:
        from seo import PUBLIC_PAGES

        assert "/coverage" in {entry.path for entry in PUBLIC_PAGES}

    def test_is_in_the_noscript_nav(self) -> None:
        import app as app_module

        noscript = app_module.app.index_string.split("<noscript>")[1]
        assert 'href="/coverage"' in noscript

    def test_links_onward(self, body) -> None:
        for href in ('href="/"', 'href="/about"', 'href="/methodology"', 'href="/benchmark"'):
            assert href in body, href
