"""Crawler-facing surfaces: a real 404, ``/robots.txt``, ``/sitemap.xml``.

Separate from ``landing.py`` on purpose. That module's charter is *static
pages read from ``web/`` with a curated head*; these responses are generated,
host-aware and config-driven, and two of them are not HTML.

## The defect this module exists for

Dash registers a ``<path:path>`` catch-all pointed at its index, so **every
unknown path returned HTTP 200 and the full app shell** — ``/robots.txt``,
``/sitemap.xml``, ``/wp-admin``, anything. Verified against production on
2026-08-11: four unrelated paths, four identical 10,918-byte 200s. No error,
no 500, no log line that looked wrong; the only symptom was an unbounded
crawlable URL space where every URL is the same page.

The fix is not a path allowlist. An allowlist duplicates Flask's routing table
by hand and rots the day Dash adds an internal route. A ``@errorhandler(404)``
cannot work at all here — the catch-all returns 200, so ``NotFound`` is never
raised. Instead ask the router the question directly: *did anything real
match, or did we fall through to the catch-all?* That allowlist maintains
itself.

## Why robots/sitemap are generated rather than static files

Three independent reasons, any one sufficient:

1. A file in ``assets/`` inherits the 1-year immutable cache header
   (``app._set_cache_headers``) — the exact trap ``landing.py`` documents.
2. ``robots.txt`` must embed an absolute ``Sitemap:`` URL, which is
   tier-dependent.
3. ``robots.txt`` must be **host-aware**. The Cloud Run ``*.run.app`` origin
   is live, unauthenticated, and serves byte-identical content beside the
   custom domain. A static file cannot tell the two apart; this module serves
   ``Disallow: /`` to the non-canonical one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

import structlog
from flask import Blueprint, Response, jsonify, request

from config import PUBLIC_BASE_URL, SEO_INDEXABLE

log = structlog.get_logger()

seo_bp = Blueprint("seo", __name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"
_NOT_FOUND_HTML = _WEB_DIR / "404.html"

#: Matches ``landing._CACHE_CONTROL`` deliberately — one number to remember,
#: and the same reasoning: long enough that repeat visits are cheap, short
#: enough that a fix lands within an hour of a deploy.
_CACHE_CONTROL = "public, max-age=3600"


class SitemapEntry(NamedTuple):
    """A public page and the file whose mtime dates it.

    ``source=None`` means no ``<lastmod>`` is emitted. That is the honest
    answer for the Dash shell at ``/``: its markup barely changes, and the
    *data* it renders changes hourly — which is not what ``lastmod`` means.
    """

    path: str
    source: Path | None


#: The canonical registry of INDEXABLE public pages.
#:
#: Adding a public page means adding it here — and
#: ``test_seo.py::TestSitemapCoverage`` fails until you do, naming the route.
#: Deriving this from ``server.url_map`` instead would sweep in ``/health``,
#: ``/metrics`` and every API rule, and would happily publish an internal
#: route the day someone adds one.
PUBLIC_PAGES: tuple[SitemapEntry, ...] = (
    SitemapEntry("/", None),
    SitemapEntry("/about", _WEB_DIR / "landing.html"),
    SitemapEntry("/methodology", _WEB_DIR / "methodology.html"),
    SitemapEntry("/coverage", _WEB_DIR / "coverage.html"),
    SitemapEntry("/benchmark", _WEB_DIR / "benchmark.html"),
)

#: Public GET routes deliberately kept OUT of the sitemap. Every entry is a
#: decision; none is a default.
SITEMAP_EXCLUDED: frozenset[str] = frozenset(
    {
        "/health",  # liveness JSON, no reader value
        "/metrics",  # IP-gated; 403s to anyone else
        "/robots.txt",  # infrastructure, not content
        "/sitemap.xml",
        "/static/<path:filename>",  # Flask's default static mount, unused
    }
)

#: robots.txt served to the canonical host.
#:
#: ``Disallow: /_`` is one line covering every current *and future* Dash
#: internal. ``/_dash-layout`` is a GET returning the entire component tree as
#: JSON — the one genuinely worth keeping out of an index.
#:
#: ``/api/v1/`` is deliberately ALLOWED. The public read-only API is an
#: intentional artifact (#250), already rate-limited and allow-listed, and
#: README + /about both point at it.
#:
#: ``/assets/`` is deliberately ALLOWED. Facebook, LinkedIn, Slack and X must
#: be able to fetch ``og-image.png`` or every share card renders imageless.
#:
#: ``/metrics`` is deliberately ABSENT rather than disallowed. robots.txt is
#: world-readable, so naming a path advertises that it exists — and /metrics
#: is the one IP-gated endpoint. It 403s to crawlers regardless, so listing it
#: buys nothing and hands a scanner a hint. Pinned by a test.
#:
#: NO AI-crawler stanzas, and that is a decision rather than an omission
#: (2026-08-11). GridPulse earns nothing per pageview, so there is no
#: cannibalization to defend against, and an assistant surfacing the project
#: unprompted requires the model to know it exists at all — a corpus property,
#: not a retrieval one. Blocking GPTBot would not even remove the site from
#: ChatGPT search, which crawls as OAI-SearchBot. Bot tokens churn constantly;
#: ``Allow: /`` does not. Pinned by a test so flipping it is a conscious edit.
_ROBOTS_ALLOW = """# GridPulse — https://gridpulse.kristenmartino.ai
#
# AI crawlers are deliberately not restricted; see seo.py for the reasoning.
# robots.txt is advisory, not access control.

User-agent: *
Allow: /
Disallow: /_

Sitemap: {sitemap}
"""

_ROBOTS_DENY = """# Non-canonical origin. The indexable deployment is elsewhere.

User-agent: *
Disallow: /
"""

#: Fallback when ``web/404.html`` is unreadable. Inverted from
#: ``landing._serve``: that helper degrades TO a 404 when its file is missing,
#: but here the 404 *is* the product, so a missing file must still 404 rather
#: than 500.
_NOT_FOUND_FALLBACK = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="robots" content="noindex"><title>Not found — GridPulse</title>'
    "</head><body><h1>Not found</h1>"
    '<p><a href="/">GridPulse</a> &middot; <a href="/about">About</a> '
    '&middot; <a href="/benchmark">Benchmark</a></p></body></html>'
)


def _is_canonical_host() -> bool:
    """Whether this request arrived on the host we claim as canonical.

    False off production (empty ``PUBLIC_BASE_URL``) and false on the
    ``*.run.app`` origin, which serves the same content unauthenticated.
    """
    if not SEO_INDEXABLE or not PUBLIC_BASE_URL:
        return False
    return request.host == urlsplit(PUBLIC_BASE_URL).netloc


def _lastmod(source: Path) -> str | None:
    """ISO date from a file's mtime, or None if unreadable.

    In a container this is the **image build time**, not the last content
    edit — every deploy bumps it. Still more honest than a fabricated
    constant, and Google treats lastmod as a hint regardless.
    """
    try:
        ts = source.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")


@seo_bp.get("/robots.txt")
def robots_txt() -> Response:
    """Serve robots.txt, host-aware."""
    if _is_canonical_host():
        body = _ROBOTS_ALLOW.format(sitemap=f"{PUBLIC_BASE_URL}/sitemap.xml")
    else:
        body = _ROBOTS_DENY
    resp = Response(body, mimetype="text/plain")
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Cache-Control"] = _CACHE_CONTROL
    return resp


@seo_bp.get("/sitemap.xml")
def sitemap_xml() -> Response:
    """Serve the sitemap.

    ``<changefreq>`` and ``<priority>`` are deliberately omitted: Google has
    ignored both for years, and an unbacked ``changefreq`` would be exactly
    the unsupported claim CLAUDE.md bans.

    Every ``<loc>`` is built from ``PUBLIC_BASE_URL``, never from the request
    host — otherwise a crawl of the ``*.run.app`` origin would produce a
    sitemap endorsing that origin.
    """
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    for entry in PUBLIC_PAGES:
        lines.append("  <url>")
        lines.append(f"    <loc>{escape(PUBLIC_BASE_URL + entry.path)}</loc>")
        if entry.source is not None:
            stamp = _lastmod(entry.source)
            if stamp:
                lines.append(f"    <lastmod>{stamp}</lastmod>")
        lines.append("  </url>")
    lines.append("</urlset>")

    resp = Response("\n".join(lines) + "\n", mimetype="application/xml")
    resp.headers["Content-Type"] = "application/xml; charset=utf-8"
    resp.headers["Cache-Control"] = _CACHE_CONTROL
    return resp


def _not_found_response(path: str) -> Response:
    """Build the 404.

    The requested path is **never reflected into the body**. ``api.py``
    already establishes that contract for unknown regions, and a 404 page is
    the classic place to lose it — echoing ``/<script>...`` back would be a
    second bug on top of the first.
    """
    if path.startswith("/api/"):
        # The API's contract is JSON in, JSON out. Handing an API client an
        # HTML page is a worse answer than the 404 it came for.
        resp = jsonify({"error": "not_found"})
        resp.status_code = 404
    else:
        try:
            html = _NOT_FOUND_HTML.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("not_found_page_missing", path=str(_NOT_FOUND_HTML), error=str(exc))
            html = _NOT_FOUND_FALLBACK
        resp = Response(html, status=404, mimetype="text/html")

    # no-store, not the 1-hour page cache: a path that 404s today may be a
    # real page tomorrow, and a cached 404 would outlive the fix.
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Robots-Tag"] = "noindex"
    return resp


def catch_all_endpoint(dash_app) -> str:
    """The endpoint name of Dash's catch-all rule.

    Dash's ``_add_url`` sets ``endpoint = routes_pathname_prefix + name``, so
    the catch-all's endpoint is the literal rule string (``/<path:path>`` at
    the default prefix). Derived from the app rather than hardcoded so that
    changing ``routes_pathname_prefix`` cannot silently disarm the guard.
    """
    return f"{dash_app.config.routes_pathname_prefix}<path:path>"


def register_not_found_guard(server, dash_app) -> None:
    """Turn Dash's catch-all 200 into a real 404.

    Safe by construction: the guard fires only on requests whose URL matched
    the catch-all, which means they matched nothing else. ``/``, ``/_dash-*``,
    ``/assets/*`` (a separate blueprint endpoint), ``/api/v1/*``, ``/health``,
    ``/metrics``, ``/about`` and ``/benchmark`` each own a rule and are never
    seen here. Query strings do not participate in rule matching, so the C2
    bookmark flow (``/?region=&persona=&tab=``) is untouched.

    Not handled: ``POST`` to an unknown path, where Werkzeug raises
    ``MethodNotAllowed`` and ``url_rule`` is None. That yields 405 rather than
    404 — only scanners hit it, and reaching into ``routing_exception`` to
    change it would cost more than it buys.
    """
    catch_all = catch_all_endpoint(dash_app)

    @server.before_request
    def _not_found_guard():
        rule = request.url_rule
        if rule is None or rule.endpoint != catch_all:
            return None
        return _not_found_response(request.path)
