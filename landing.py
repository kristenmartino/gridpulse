"""Marketing landing page — ``/about`` (portfolio-neutral posture).

A single static page served from ``web/landing.html`` via a thin blueprint
(the ``api.py`` precedent). Deliberately decoupled from the Dash app: the
page re-declares its own ``<head>`` and embeds a curated copy of the design
tokens, so dashboard chrome changes can never break the public marketing
surface.

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

_LANDING_HTML = Path(__file__).resolve().parent / "web" / "landing.html"

#: Iterable cache: long enough to keep repeat visits cheap, short enough
#: that copy fixes land within an hour of a deploy.
_CACHE_CONTROL = "public, max-age=3600"


@landing_bp.get("/about")
def about() -> Response:
    """Serve the marketing landing page.

    Read per request (small file, container filesystem) rather than cached
    at import — a missing file must degrade to a loud 404 on this route,
    never take down the whole app at import time.
    """
    try:
        html = _LANDING_HTML.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("landing_page_missing", path=str(_LANDING_HTML), error=str(exc))
        return Response("landing page unavailable", status=404, mimetype="text/plain")
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = _CACHE_CONTROL
    return resp
