"""Crawler surfaces: robots.txt, sitemap.xml, and the 404 guard (seo.py).

Route mechanics follow ``test_landing.py``'s idiom — a bare Flask app with the
blueprint registered — so the fast unit tier does not pay ``import app`` for
every assertion. ``TestNotFoundDoesNotBreakTheApp`` is the deliberate
exception: the guard's whole safety argument rests on the real url_map, so
that one class asserts against it.

Several tests here pin decisions rather than behavior. Where a docstring says
a choice is deliberate, flipping it should mean editing the test on purpose.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from flask import Flask

import seo
from config import PUBLIC_BASE_URL
from seo import PUBLIC_PAGES, SITEMAP_EXCLUDED, seo_bp

_CANONICAL_BASE = "https://gridpulse.kristenmartino.ai"
_RUN_APP_BASE = "https://gridpulse-abc123-ue.a.run.app"

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(seo_bp)
    return app.test_client()


@pytest.fixture
def canonical(monkeypatch):
    """Force this deployment to believe it is the indexable production one.

    PUBLIC_BASE_URL is empty off production, so without this the module under
    test correctly refuses to emit anything — which is its own test below,
    not the precondition for the rest of them.
    """
    monkeypatch.setattr(seo, "PUBLIC_BASE_URL", _CANONICAL_BASE)
    monkeypatch.setattr(seo, "SEO_INDEXABLE", True)


class TestRobotsTxt:
    def test_serves_plain_text_200(self, client, canonical) -> None:
        resp = client.get("/robots.txt", base_url=_CANONICAL_BASE)
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "text/plain; charset=utf-8"

    def test_iterable_cache_header(self, client, canonical) -> None:
        """Same reasoning as the landing route: never the assets-folder
        1-year immutable header, which would pin a crawl policy for a year."""
        resp = client.get("/robots.txt", base_url=_CANONICAL_BASE)
        assert resp.headers["Cache-Control"] == "public, max-age=3600"
        assert "immutable" not in resp.headers["Cache-Control"]

    def test_points_at_the_absolute_sitemap_url(self, client, canonical) -> None:
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert f"Sitemap: {_CANONICAL_BASE}/sitemap.xml" in body

    def test_blocks_dash_internals(self, client, canonical) -> None:
        """One prefix rule covers every current and future Dash internal.
        /_dash-layout is a GET returning the whole component tree as JSON."""
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert "Disallow: /_" in body

    def test_public_api_is_crawlable(self, client, canonical) -> None:
        """DECISION (2026-08-11): the read-only JSON API stays crawlable.

        It is a deliberate portfolio artifact (#250), already rate-limited
        and allow-listed, and README + /about both link it.
        """
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert "Disallow: /api" not in body

    def test_assets_stay_crawlable(self, client, canonical) -> None:
        """Blocking /assets/ would break every share card — the unfurlers
        have to be able to fetch og-image.png."""
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert "Disallow: /assets" not in body

    def test_metrics_is_not_advertised(self, client, canonical) -> None:
        """DECISION: /metrics is ABSENT, not disallowed.

        robots.txt is world-readable, so naming a path advertises that it
        exists — and /metrics is the one IP-gated endpoint. It 403s to a
        crawler regardless, so listing it buys nothing and hands a scanner a
        hint. Adding it back should be a conscious edit here.
        """
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert "/metrics" not in body

    def test_ai_crawlers_are_deliberately_unrestricted(self, client, canonical) -> None:
        """DECISION (2026-08-11): no AI-crawler stanzas. Values call, pinned.

        GridPulse earns nothing per pageview, so there is no cannibalization
        to defend against, and an assistant surfacing the project unprompted
        requires the model to know it exists — a corpus property, not a
        retrieval one. Blocking GPTBot would not even remove the site from
        ChatGPT search, which crawls as OAI-SearchBot.
        """
        body = client.get("/robots.txt", base_url=_CANONICAL_BASE).get_data(as_text=True)
        for bot in ("GPTBot", "ClaudeBot", "CCBot", "PerplexityBot", "Google-Extended"):
            assert bot not in body, f"AI-crawler stanza appeared for {bot}"

    def test_non_canonical_host_gets_disallow_all(self, client, canonical) -> None:
        """The *.run.app origin is live, unauthenticated, and serves
        byte-identical content. It must not compete with the custom domain."""
        body = client.get("/robots.txt", base_url=_RUN_APP_BASE).get_data(as_text=True)
        assert "Disallow: /" in body
        assert "Allow: /" not in body
        assert _CANONICAL_BASE not in body

    def test_unconfigured_deployment_disallows_everything(self, client, monkeypatch) -> None:
        """Staging ships an empty PUBLIC_BASE_URL and seo_indexable=False, so
        it cannot out-rank production for production's own content."""
        monkeypatch.setattr(seo, "PUBLIC_BASE_URL", "")
        monkeypatch.setattr(seo, "SEO_INDEXABLE", False)
        body = client.get("/robots.txt", base_url="https://staging.example.com").get_data(
            as_text=True
        )
        assert "Disallow: /" in body
        assert "Allow: /" not in body


class TestSitemap:
    def test_serves_xml_200(self, client, canonical) -> None:
        resp = client.get("/sitemap.xml", base_url=_CANONICAL_BASE)
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "application/xml; charset=utf-8"
        assert resp.headers["Cache-Control"] == "public, max-age=3600"

    def test_is_well_formed(self, client, canonical) -> None:
        body = client.get("/sitemap.xml", base_url=_CANONICAL_BASE).get_data(as_text=True)
        root = ET.fromstring(body)
        assert root.tag.endswith("urlset")

    def test_locs_match_the_registry(self, client, canonical) -> None:
        body = client.get("/sitemap.xml", base_url=_CANONICAL_BASE).get_data(as_text=True)
        locs = {e.text for e in ET.fromstring(body).findall(".//sm:loc", _SITEMAP_NS)}
        assert locs == {_CANONICAL_BASE + entry.path for entry in PUBLIC_PAGES}

    def test_locs_are_host_independent(self, client, canonical) -> None:
        """Fetched from the *.run.app origin, the sitemap must STILL emit only
        canonical-host URLs.

        This is the test that catches building <loc> from request.host_url —
        a sitemap served on the duplicate origin that endorsed that origin
        would actively teach crawlers the wrong canonical.
        """
        body = client.get("/sitemap.xml", base_url=_RUN_APP_BASE).get_data(as_text=True)
        locs = [e.text for e in ET.fromstring(body).findall(".//sm:loc", _SITEMAP_NS)]
        assert locs, "sitemap emitted no URLs"
        assert all(loc.startswith(_CANONICAL_BASE) for loc in locs)
        assert "run.app" not in body

    def test_lastmod_only_where_a_file_backs_it(self, client, canonical) -> None:
        """`/` gets no lastmod: its markup barely changes and the data it
        renders changes hourly, which is not what lastmod means."""
        body = client.get("/sitemap.xml", base_url=_CANONICAL_BASE).get_data(as_text=True)
        backed = {entry.path for entry in PUBLIC_PAGES if entry.source is not None}
        for url in ET.fromstring(body).findall(".//sm:url", _SITEMAP_NS):
            loc = url.find("sm:loc", _SITEMAP_NS).text
            path = loc[len(_CANONICAL_BASE) :]
            has_lastmod = url.find("sm:lastmod", _SITEMAP_NS) is not None
            assert has_lastmod == (path in backed), path

    def test_omits_changefreq_and_priority(self, client, canonical) -> None:
        """Google has ignored both for years, and an unbacked changefreq is
        exactly the unsupported claim CLAUDE.md bans. Pinned so nobody adds
        them back as 'completeness'."""
        body = client.get("/sitemap.xml", base_url=_CANONICAL_BASE).get_data(as_text=True)
        assert "changefreq" not in body
        assert "priority" not in body


class TestSitemapCoverage:
    def test_every_public_route_is_classified(self) -> None:
        """A new public route must be either published or excluded.

        Deriving the sitemap live from url_map would have swept in /health,
        /metrics and every API rule. This forces the decision instead: add
        the route to PUBLIC_PAGES or to SITEMAP_EXCLUDED, on purpose.
        """
        import app as app_module

        published = {entry.path for entry in PUBLIC_PAGES}
        for rule in app_module.server.url_map.iter_rules():
            if "GET" not in rule.methods or "<" in rule.rule:
                continue
            if rule.rule.startswith("/_"):
                continue  # Dash internals, blanket-disallowed in robots.txt
            if rule.rule.startswith("/api/"):
                continue  # the JSON API is its own surface, not sitemap content
            assert rule.rule in published | SITEMAP_EXCLUDED, (
                f"New public route {rule.rule!r} is unclassified. Add it to "
                f"seo.PUBLIC_PAGES or seo.SITEMAP_EXCLUDED — both are "
                f"decisions, neither is a default."
            )


class _FakeDashConfig:
    routes_pathname_prefix = "/"


class _FakeDash:
    config = _FakeDashConfig()


@pytest.fixture
def guarded_client():
    """A synthetic app shaped like Dash's routing, without importing Dash.

    Registers the same ``<path:path>`` catch-all Dash does, plus one real
    route, so the guard can be exercised without the cost of a full app.
    """
    app = Flask(__name__)
    app.register_blueprint(seo_bp)

    @app.get("/")
    def _index():
        return "app shell"

    app.add_url_rule("/<path:path>", "/<path:path>", lambda path: "app shell")

    seo.register_not_found_guard(app, _FakeDash())
    return app.test_client()


class TestNotFoundGuard:
    def test_catch_all_endpoint_is_derived_not_hardcoded(self) -> None:
        """Derived from routes_pathname_prefix so changing it cannot silently
        disarm the guard and restore 200-on-everything."""
        assert seo.catch_all_endpoint(_FakeDash()) == "/<path:path>"

    @pytest.mark.parametrize(
        "path", ["/wp-admin", "/this-page-does-not-exist-12345", "/wp-admin/setup-config.php"]
    )
    def test_unknown_paths_404(self, guarded_client, path) -> None:
        resp = guarded_client.get(path)
        assert resp.status_code == 404
        assert resp.content_type.startswith("text/html")

    def test_real_routes_are_untouched(self, guarded_client) -> None:
        assert guarded_client.get("/").status_code == 200
        assert guarded_client.get("/robots.txt").status_code == 200

    def test_api_paths_get_json_not_html(self, guarded_client) -> None:
        """Handing an API client an HTML error page is a second bug on top of
        the 404 it came for."""
        resp = guarded_client.get("/api/v1/nonsense")
        assert resp.status_code == 404
        assert resp.content_type.startswith("application/json")
        assert resp.get_json()["error"] == "not_found"

    def test_404_does_not_reflect_the_requested_path(self, guarded_client) -> None:
        """api.py already establishes that raw input is never reflected back;
        a 404 page is the classic place to lose that contract."""
        body = guarded_client.get("/<script>alert(1)</script>").get_data(as_text=True)
        assert "script>alert" not in body
        assert "alert(1)" not in body

    def test_404_is_uncached_and_noindex(self, guarded_client) -> None:
        """no-store, not the 1-hour page cache: a path that 404s today may be
        a real page tomorrow, and a cached 404 would outlive the fix."""
        resp = guarded_client.get("/wp-admin")
        assert resp.headers["Cache-Control"] == "no-store"
        assert resp.headers["X-Robots-Tag"] == "noindex"

    def test_missing_404_file_still_404s(self, guarded_client, monkeypatch) -> None:
        """Inverted from landing._serve. That helper degrades TO a 404 when
        its file is missing; here the 404 IS the product, so a missing file
        must never become a 500."""
        monkeypatch.setattr(seo, "_NOT_FOUND_HTML", Path("/nonexistent/404.html"))
        resp = guarded_client.get("/wp-admin")
        assert resp.status_code == 404
        assert resp.content_type.startswith("text/html")
        assert "GridPulse" in resp.get_data(as_text=True)


class TestNotFoundDoesNotBreakTheApp:
    """Against the REAL url_map — the guard's safety argument rests on it."""

    @pytest.fixture(scope="class")
    def client(self):
        import app as app_module

        return app_module.server.test_client()

    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/?region=ERCOT&persona=trader&tab=tab-models",  # the C2 bookmark flow
            "/about",
            "/benchmark",
            "/health",
            "/api/v1/",
            "/robots.txt",
            "/sitemap.xml",
            "/assets/favicon.svg",
            "/_dash-layout",
            "/_dash-dependencies",
        ],
    )
    def test_real_surfaces_are_not_404(self, client, path) -> None:
        assert client.get(path).status_code != 404, path

    def test_scanner_path_404s_against_the_real_app(self, client) -> None:
        """The regression pin for the production defect: before this guard,
        /wp-admin returned 200 and the full 10,918-byte Dash shell."""
        assert client.get("/wp-admin").status_code == 404

    def test_missing_asset_404_is_not_cached_for_a_year(self, client) -> None:
        """_set_cache_headers used to stamp the 1-year immutable header on ANY
        /assets/ response, so a typo'd path had its 404 pinned for a year."""
        resp = client.get("/assets/definitely-missing.png")
        assert resp.status_code == 404
        assert "immutable" not in resp.headers.get("Cache-Control", "")


class TestConfig:
    def test_public_base_url_has_no_trailing_slash(self) -> None:
        """Every absolute URL is built by concatenation, so a trailing slash
        would produce '//about' in canonical tags and sitemap locs."""
        assert not PUBLIC_BASE_URL.endswith("/")
