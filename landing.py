"""Public static pages — ``/about`` and ``/benchmark``.

Static pages served from ``web/`` via a thin blueprint (the ``api.py``
precedent). Deliberately decoupled from the Dash app: each page re-declares
its own ``<head>`` and embeds a curated copy of the design tokens, so
dashboard chrome changes can never break a public surface.

``/benchmark`` carries no data of its own — it fetches ``/api/v1/benchmark``
in the browser, so the page a reader sees is rendered from the same public
endpoint they can call themselves. That is deliberate: it makes the page
impossible to dress up beyond what the API will admit to.

Placement rationale (docs/internal/landing_page_spec_archive.md): side
path — the dashboard keeps ``/`` and every existing bookmark; the page is
promotable to the front door later. The file lives in ``web/``, not
``assets/`` — Dash auto-serves ``assets/`` with a 1-year immutable cache
header (see ``_set_cache_headers`` in app.py), which would pin a marketing
page for a year with no hash-busting. This route sets an iterable 1-hour
cache instead.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from flask import Blueprint, Response

log = structlog.get_logger()

landing_bp = Blueprint("landing", __name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"
_LANDING_HTML = _WEB_DIR / "landing.html"
_BENCHMARK_HTML = _WEB_DIR / "benchmark.html"
_METHODOLOGY_HTML = _WEB_DIR / "methodology.html"

#: Iterable cache: long enough to keep repeat visits cheap, short enough
#: that copy fixes land within an hour of a deploy.
_CACHE_CONTROL = "public, max-age=3600"


def _serve(path: Path, what: str) -> Response:
    """Serve a static page from ``web/``.

    Read per request (small file, container filesystem) rather than cached
    at import — a missing file must degrade to a loud 404 on its own route,
    never take down the whole app at import time.
    """
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("static_page_missing", page=what, path=str(path), error=str(exc))
        return Response(f"{what} unavailable", status=404, mimetype="text/plain")
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = _CACHE_CONTROL
    return resp


@landing_bp.get("/about")
def about() -> Response:
    """Serve the marketing landing page."""
    return _serve(_LANDING_HTML, "landing page")


@landing_bp.get("/methodology")
def methodology() -> Response:
    """Serve the methodology page.

    Hand-converted from ``docs/HOW_IT_WORKS.md`` and committed, rather than
    rendered from that file at request time — ``docs/`` is in
    ``.dockerignore``, so it does not exist in the production image. A route
    that read it would 404 in every environment that matters.
    """
    return _serve(_METHODOLOGY_HTML, "methodology page")


@landing_bp.get("/benchmark")
def benchmark_page() -> Response:
    """Serve the public forecast-benchmark page.

    The shell only. Numbers arrive from ``/api/v1/benchmark`` client-side,
    which is why this route never touches Redis and cannot render a figure
    the API would not also return.
    """
    return _serve(_BENCHMARK_HTML, "benchmark page")
