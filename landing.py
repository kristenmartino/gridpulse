"""Public static pages — ``/about`` and ``/benchmark``.

Static pages served from ``web/`` via a thin blueprint (the ``api.py``
precedent). Deliberately decoupled from the Dash app: each page re-declares
its own ``<head>`` and embeds a curated copy of the design tokens, so
dashboard chrome changes can never break a public surface.

``/benchmark`` carries no data of its own. It server-renders a summary from
``api.build_benchmark_payload()`` — the identical structure, through the
identical allow-list, that ``/api/v1/benchmark`` returns — and then fetches
that endpoint in the browser to hydrate the sortable table over it. There is
no second, friendlier path to the data: the page still cannot show a figure
the API would not also admit to.

The server render exists because the page previously shipped **zero numbers**
in its initial HTML: ``#content`` was hidden and every figure arrived from one
client-side ``fetch``. Googlebot renders JavaScript but defers it; Bingbot and
most LLM fetchers do not. A page whose entire value is a published scoreboard
was invisible to them. It also removes a failure mode — ``_api_rate_limit`` is
per-IP and a render fleet shares egress IPs, so a rendered crawl could be
429'd into a blank page.

Placement rationale (docs/internal/landing_page_spec_archive.md): side
path — the dashboard keeps ``/`` and every existing bookmark; the page is
promotable to the front door later. The file lives in ``web/``, not
``assets/`` — Dash auto-serves ``assets/`` with a 1-year immutable cache
header (see ``_set_cache_headers`` in app.py), which would pin a marketing
page for a year with no hash-busting. This route sets an iterable 1-hour
cache instead.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import structlog
from flask import Blueprint, Response

log = structlog.get_logger()

landing_bp = Blueprint("landing", __name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"
_LANDING_HTML = _WEB_DIR / "landing.html"
_BENCHMARK_HTML = _WEB_DIR / "benchmark.html"
_METHODOLOGY_HTML = _WEB_DIR / "methodology.html"
_COVERAGE_HTML = _WEB_DIR / "coverage.html"

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


#: Replaced on /about with one live benchmark sentence. Left empty when the
#: benchmark is warming or the render fails — the card's qualitative copy
#: stands on its own, and no benchmark number is ever hardcoded in static
#: HTML (the #535 lesson).
_CLAIM_MARKER = "<!--SSR_BENCHMARK_CLAIM-->"


def _render_benchmark_claim(payload: dict) -> str:
    """One derived sentence for /about's benchmark card.

    Derived per request for the same reason the /benchmark verdict is: a
    written-in number becomes a lie the first time the payload moves. Both
    directions handled; the steadiness clause is only claimed when the rows
    actually support it.
    """
    fleet = payload.get("fleet") or {}
    ours = fleet.get("median_gridpulse_mape")
    theirs = fleet.get("median_official_mape")
    n = fleet.get("n")
    if not isinstance(ours, (int, float)) or not isinstance(theirs, (int, float)):
        return ""

    rows = _fleet_rows(payload)
    ours_spread = _spread_ratio(rows, "gridpulse")
    theirs_spread = _spread_ratio(rows, "official")
    if ours < theirs:
        claim = (
            f"Live right now: the closer forecast on the typical operator — "
            f"{_pct(ours)} median error against their {_pct(theirs)}, across "
            f"{_esc(n)} operators"
        )
    else:
        gap = abs(ours - theirs)
        claim = (
            f"Live right now: within {gap:.2f} points of the operators' own "
            f"median error across {_esc(n)} operators — {_pct(ours)} against "
            f"their {_pct(theirs)}"
        )
    if ours_spread and theirs_spread and ours_spread < theirs_spread:
        claim += (
            f", with a fraction of their best-to-worst spread "
            f"(ours {ours_spread}×, theirs {theirs_spread}×)"
        )
    return f'<p class="bench-claim">{claim}.</p>'


@landing_bp.get("/about")
def about() -> Response:
    """Serve the marketing landing page, its benchmark claim rendered live.

    Fails open like the benchmark summary: a render error leaves the marker
    empty and the card's qualitative copy stands on its own.
    """
    resp = _serve(_LANDING_HTML, "landing page")
    if resp.status_code != 200:
        return resp

    claim = ""
    try:
        from api import build_benchmark_payload

        payload = build_benchmark_payload()
        if payload:
            claim = _render_benchmark_claim(payload)
    except Exception as exc:  # noqa: BLE001 — enrichment must never 500 the page
        log.warning("about_benchmark_claim_failed", error=str(exc))

    out = Response(resp.get_data(as_text=True).replace(_CLAIM_MARKER, claim), mimetype="text/html")
    out.headers["Cache-Control"] = _CACHE_CONTROL
    return out


_COVERAGE_MARKER = "<!--SSR_COVERAGE_TABLE-->"


def _render_coverage_table() -> str:
    """The 51-BA coverage table, rendered from ``config`` at request time.

    Server-rendered rather than committed as static markup precisely because
    the source of truth is in the image: ``config.REGION_NAMES`` and friends
    ship in the container, so generating from them makes it structurally
    impossible for this page to disagree with what the product covers. A
    hand-maintained copy of a 51-row list would be stale within a release.
    """
    from config import (
        PEAK_DERIVED_CAPACITY,
        REGION_CAPACITY_MW,
        REGION_GROUPS,
        REGION_NAMES,
        STATE_TO_BA,
    )

    group_of = {code: group for group, codes in REGION_GROUPS.items() for code in codes}

    rows = []
    for code in sorted(REGION_NAMES):
        capacity = REGION_CAPACITY_MW.get(code)
        peak_derived = code in PEAK_DERIVED_CAPACITY
        capacity_cell = "—" if capacity is None else f"{capacity:,}"
        if peak_derived:
            capacity_cell += ' <span class="est" title="Peak demand × 1.15">est.</span>'
        rows.append(
            f'<tr><th scope="row"><code>{_esc(code)}</code></th>'
            f"<td>{_esc(REGION_NAMES.get(code, code))}</td>"
            f"<td>{_esc(group_of.get(code, '—'))}</td>"
            f"<td>{_esc(', '.join(STATE_TO_BA.get(code) or []) or '—')}</td>"
            f'<td class="num">{capacity_cell}</td></tr>'
        )

    return (
        f'<p class="table-note">{len(rows)} balancing authorities, '
        f"{len(PEAK_DERIVED_CAPACITY)} of them carrying a peak-derived capacity "
        f"estimate rather than a summed nameplate figure.</p>"
        "<table><thead><tr>"
        '<th scope="col">Code</th><th scope="col">Name</th>'
        '<th scope="col">Region</th><th scope="col">States served</th>'
        '<th scope="col" class="num">Capacity (MW)</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


@landing_bp.get("/coverage")
def coverage() -> Response:
    """Serve the coverage page with its table already rendered.

    Fails open like the benchmark summary: a render error leaves the marker
    empty and the prose still stands on its own.
    """
    resp = _serve(_COVERAGE_HTML, "coverage page")
    if resp.status_code != 200:
        return resp

    table = ""
    try:
        table = _render_coverage_table()
    except Exception as exc:  # noqa: BLE001 — enrichment must never 500 the page
        log.warning("coverage_table_render_failed", error=str(exc))

    out = Response(
        resp.get_data(as_text=True).replace(_COVERAGE_MARKER, table), mimetype="text/html"
    )
    out.headers["Cache-Control"] = _CACHE_CONTROL
    return out


@landing_bp.get("/methodology")
def methodology() -> Response:
    """Serve the methodology page.

    Hand-converted from ``docs/HOW_IT_WORKS.md`` and committed, rather than
    rendered from that file at request time — ``docs/`` is in
    ``.dockerignore``, so it does not exist in the production image. A route
    that read it would 404 in every environment that matters.
    """
    return _serve(_METHODOLOGY_HTML, "methodology page")


#: Replaced with the server-rendered summary. Left in place (as an empty
#: string) when the benchmark is warming, so the page degrades to exactly its
#: pre-SSR behaviour rather than to a broken half-render.
_SSR_MARKER = "<!--SSR_BENCHMARK_SUMMARY-->"


def _esc(value: object) -> str:
    """Escape a payload value for HTML text.

    Everything rendered here comes from the API allow-list rather than from
    user input, but a region code reaching markup unescaped is the kind of
    assumption that stops being true later.
    """
    return escape("" if value is None else str(value), quote=True)


def _pct(value: object) -> str:
    """Percentage, or an em dash when the payload has no value.

    Two decimals, matching the page's own ``num()`` default. They must agree:
    the client render replaces this one, so a different precision would make
    the same figure visibly change as the page hydrates.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:.2f}%"


def _fleet_rows(payload: dict) -> list[dict]:
    """The fleet population's 24h lead blocks.

    Scored rows minus any BA reported separately (``isolated``) — the same
    walk the client's ``fleetRows()`` does, so the two renders cannot
    disagree about the population they describe.
    """
    isolated = payload.get("isolated") or {}
    rows = []
    for region in payload.get("regions") or []:
        if region.get("region") in isolated:
            continue
        lead = (region.get("leads") or {}).get("24h") or {}
        if (
            region.get("scoreable")
            and lead.get("scoreable")
            and lead.get("official")
            and lead.get("gridpulse")
        ):
            rows.append(lead)
    return rows


def _spread_ratio(rows: list[dict], arm: str) -> float | None:
    """Worst ÷ best mean MAPE across the fleet rows, one decimal.

    Derived from the same rows the table renders rather than from the
    payload's ``fleet`` block — the fleet block can be written a beat apart
    from the per-BA keys, and quoting both sources let two statements of the
    same statistic disagree in the third digit on one screen.
    """
    vals = [
        v
        for row in rows
        for v in [(row.get(arm) or {}).get("mape")]
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
    ]
    if len(vals) < 2:
        return None
    return round(max(vals) / min(vals), 1)


def _render_benchmark_summary(payload: dict) -> str:
    """Server-rendered summary: fleet tiles, the verdict, and a plain table.

    Deliberately NOT the full interactive table. Sorting, the drop-count
    disclosures and the excluded-BA detail stay client-side — this is the
    content a non-rendering agent needs, not a second implementation of the
    page.

    The verdict is derived here for the same reason it is derived in the
    browser: a written-in conclusion becomes a lie the first time the numbers
    move. Both directions are handled, and the order mirrors the client's —
    the shape of the result first (how close, how steady), the median tally
    as the last word rather than the first.
    """
    fleet = payload.get("fleet") or {}
    ours = fleet.get("median_gridpulse_mape")
    theirs = fleet.get("median_official_mape")
    wins, losses, n = fleet.get("wins"), fleet.get("losses"), fleet.get("n")

    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        gap = abs(ours - theirs)
        if ours < theirs:
            shape = (
                f"On the typical grid operator GridPulse is the closer forecast "
                f"— {_pct(ours)} average error against their {_pct(theirs)} — "
                f"built from public data alone."
            )
            tally = (
                f"At the median GridPulse is the closer of the two, closer on "
                f"{_esc(wins)} of {_esc(n)} operators."
            )
        else:
            shape = (
                f"GridPulse runs within {gap:.2f} points of the operators' own "
                f"forecasts on the typical grid — {_pct(ours)} average error "
                f"against their {_pct(theirs)} — built from public data alone."
            )
            tally = (
                f"At the median the operator's own forecast is the closer of "
                f"the two, closer on {_esc(losses)} of {_esc(n)} operators."
            )
        rows = _fleet_rows(payload)
        ours_spread = _spread_ratio(rows, "gridpulse")
        theirs_spread = _spread_ratio(rows, "official")
        steadier = ""
        if ours_spread and theirs_spread:
            steadier = (
                f" It is also the steadier of the two: our error spans "
                f"{ours_spread}× from best operator to worst, theirs "
                f"{theirs_spread}×."
                if ours_spread < theirs_spread
                else f" Theirs is also the steadier of the two — "
                f"{theirs_spread}× from best operator to worst, against our "
                f"{ours_spread}×."
            )
        verdict = f"{shape}{steadier} {tally}"
    else:
        verdict = "Fleet aggregation is still accumulating."

    # Total from config, the same source the coverage table renders from —
    # a bare N gives a reader no way to see the denominator move, which is
    # exactly how #535 hid.
    from config import REGION_COORDINATES

    excluded_entries = payload.get("excluded") or []
    fairness = [e for e in excluded_entries if e.get("reason") != "insufficient-paired-hours"]
    accumulating = [e for e in excluded_entries if e.get("reason") == "insufficient-paired-hours"]
    if payload.get("n_excluded") is None:
        exclusion_sub = "exclusion count unavailable this tick — see the page tables"
    elif accumulating:
        exclusion_sub = (
            f"{len(fairness)} excluded for fairness, {len(accumulating)} still "
            f"accumulating — each with a published reason"
        )
    else:
        exclusion_sub = f"{_esc(payload.get('n_excluded'))} excluded, each with a published reason"

    n_scoreable = payload.get("n_scoreable")
    tiles = [
        (
            "Operators scored",
            "—" if n_scoreable is None else f"{_esc(n_scoreable)} / {len(REGION_COORDINATES)}",
            exclusion_sub,
        ),
        (
            "Their median error",
            _pct(theirs),
            "median across operators of each one's average error (mean MAPE)",
        ),
        ("Our median error", _pct(ours), "same statistic, same hours, same window"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in tiles
    )

    rows = []
    for region in payload.get("regions") or []:
        lead = (region.get("leads") or {}).get("24h") or {}
        # Same rule the client table applies: a row publishes only when its
        # headline lead actually carries a winner. A BA still accumulating
        # paired hours has no verdict, and inventing a blank one reads as a
        # result rather than an absence.
        if not lead.get("winner"):
            continue
        official = lead.get("official") or {}
        gridpulse = lead.get("gridpulse") or {}
        rows.append(
            f'<tr><th scope="row">{_esc(region.get("region"))}</th>'
            f"<td>{_pct(official.get('mape'))}</td>"
            f"<td>{_pct(gridpulse.get('mape'))}</td>"
            f"<td>{_esc(lead.get('winner'))}</td></tr>"
        )

    table = (
        "<table><caption>GridPulse vs each grid operator&rsquo;s own "
        "day-ahead forecast, 24-hour lead, 30-day window. Mean MAPE on paired "
        "hours against settled actuals.</caption><thead><tr>"
        '<th scope="col">BA</th><th scope="col">Their error</th>'
        '<th scope="col">Our error</th><th scope="col">Closer</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        if rows
        else ""
    )

    return (
        f'<div id="ssr-summary" class="ssr-summary">'
        f'<div class="tiles">{tile_html}</div>'
        f'<p class="verdict">{verdict}</p>'
        f"{table}</div>"
    )


@landing_bp.get("/benchmark")
def benchmark_page() -> Response:
    """Serve the public forecast-benchmark page, summary already rendered.

    Fails open: any error building the summary leaves the marker empty and
    the page behaves exactly as it did before SSR existed — a client fetch
    into a hidden container. A benchmark page that 500s because its
    *decoration* failed would be a worse outcome than one a crawler cannot
    read.
    """
    resp = _serve(_BENCHMARK_HTML, "benchmark page")
    if resp.status_code != 200:
        return resp

    summary = ""
    try:
        from api import build_benchmark_payload

        payload = build_benchmark_payload()
        if payload:
            summary = _render_benchmark_summary(payload)
    except Exception as exc:  # noqa: BLE001 — enrichment must never 500 the page
        log.warning("benchmark_ssr_failed", error=str(exc))

    html = resp.get_data(as_text=True).replace(_SSR_MARKER, summary)
    out = Response(html, mimetype="text/html")
    # Shorter than the marketing pages' hour: this route now carries live
    # numbers, and an hour-stale scoreboard is worse than an hour-stale
    # description of the product.
    out.headers["Cache-Control"] = "public, max-age=300"
    return out
