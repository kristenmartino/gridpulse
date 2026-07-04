"""US Grid tab helpers — extracted from ``callbacks.py`` per issue #87.

Owns the data-collection + rendering helpers for the US Grid tab's
three view modes:

- **Cards** — per-region small-multiples grid (``_build_us_grid_region_card``)
- **Map** — Plotly ``scatter_geo`` of BA centroids (``_build_us_grid_map``)
- **Polygons** — Plotly ``Choropleth`` of BA service-territory shapes
  (``_build_us_grid_choropleth``) backed by ``assets/ba_polygons.geojson``

Cross-cutting infrastructure (caches, layout helpers, color tokens,
map basemap constants, ``_latest_real_demand``) lives in
``components/_callbacks_shared.py``. ``REGION_*`` config dicts come
from ``config.py``. Redis access goes through ``data.redis_client``.

The module is imported back into ``components/callbacks.py`` via an
explicit re-import block so existing
``from components.callbacks import <X>`` import sites (notably
``tests/unit/test_us_grid_choropleth.py``,
``tests/unit/test_us_grid_nan_guard.py``,
``tests/unit/test_us_grid_stress_cap.py``) resolve unchanged.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import structlog
from dash import ALL, Input, Output, ctx, dcc, html, no_update

from components._callbacks_shared import (
    _MAP_AXIS_FONT_COLOR,
    _MAP_BORDER_COLOR,
    _MAP_COASTLINE_COLOR,
    _MAP_COLORSCALE,
    _MAP_LAND_COLOR,
    _MAP_SUBUNIT_COLOR,
    _STRESS_RELIABLE_CEILING,
    _latest_real_demand,
)
from components.cards import build_page_title
from config import (
    IS_IMPORT_DOMINATED,
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
    REGION_NAMES,
)
from data.redis_client import redis_get, redis_key

log = structlog.get_logger()


# ── V1.β: US GRID TAB HELPERS ────────────────────────────────


def _collect_us_grid_region_data() -> dict[str, dict]:
    """Pull per-region snapshots from Redis for the US Grid card grid.

    Returns ``{region_code: {"current_mw", "prev_mw", "today_mw",
    "interchange"}}`` for every region in ``REGION_NAMES`` that **passes
    the V3.ζ forecast quality gate**. Regions without a Redis entry
    (cold pipeline, new BAs awaiting their first scoring run) — or
    whose demand tail is NaN / zero (EIA-930 publishing lag for the
    most recent hour) — get a dict with ``current_mw=None`` so the
    caller can render an ``"—"`` placeholder card without ever exposing
    a NaN to downstream sums or divisions. ``interchange`` is the V3.α
    ``gridpulse:interchange:{region}:1h`` payload, or ``None`` if absent.

    Regions failing the quality gate (XGBoost holdout MAPE in the
    ``rollback`` grade per ``mape_grade()``) are omitted from the
    return entirely so all downstream consumers (cards, metrics, map)
    see the same filtered set. The dropdown gate in ``layout.py``
    independently filters the same way; the hidden-count surfaced in
    the page title comes from ``hidden_regions()``.
    """
    from models.model_service import is_forecast_quality_acceptable

    out: dict[str, dict] = {}
    for region in REGION_NAMES:
        if not is_forecast_quality_acceptable(region):
            continue
        actuals = redis_get(redis_key(f"actuals:{region}"))
        demand = (actuals or {}).get("demand_mw") or []
        interchange = redis_get(redis_key(f"interchange:{region}:1h"))
        if not demand:
            out[region] = {"interchange": interchange} if interchange else {}
            continue

        current_mw = _latest_real_demand(demand)
        prev_mw = _latest_real_demand(demand, offset=1) if current_mw is not None else None
        # Sparkline is rendered tolerant of NaN (gp-region-card__sparkline
        # uses raw values), so we keep the trailing-24h slice as-is — but
        # the headline number paths read ``current_mw`` / ``prev_mw``.
        out[region] = {
            "current_mw": current_mw,
            "prev_mw": prev_mw,
            "today_mw": demand[-24:],
            "interchange": interchange,
        }
    return out


def _build_us_grid_title(region_data: dict[str, dict]) -> html.Div:
    """Page title block: 'US Grid' + '<N> BAs · X.X GW total demand'.

    V3.ζ follow-up: when the forecast-quality gate hides one or more
    BAs from the visible set, append a small annotation to the
    subtitle ("· N hidden") with a hover tooltip listing the affected
    codes. Hidden BAs are not in ``region_data`` (the collector skips
    them), so the count is the difference between ``REGION_NAMES`` and
    the gate's view of acceptable regions.
    """
    from models.model_service import hidden_regions

    n_regions = len(REGION_NAMES)
    # Filter to finite, positive values only — ``_collect_us_grid_region_data``
    # already returns ``None`` for regions whose demand tail is NaN, but
    # the explicit ``np.isfinite`` here is a safety net so a future
    # regression in the collector can't poison the sum with NaN.
    real_demands = [
        v for v in (d.get("current_mw") for d in region_data.values()) if _is_real_positive(v)
    ]
    n_with_data = len(real_demands)
    total_mw = sum(real_demands)
    hidden = hidden_regions(REGION_NAMES.keys())
    n_visible = n_regions - len(hidden)

    if total_mw > 0:
        subtitle = (
            f"{n_with_data} of {n_visible} balancing authorities reporting · "
            f"{total_mw / 1000:.1f} GW total demand"
        )
    else:
        subtitle = f"{n_visible} balancing authorities · pipeline warming up"
    tooltip: str | None = None
    if hidden:
        subtitle += f" · {len(hidden)} hidden"
        tooltip = (
            "Hidden by the forecast quality gate (XGBoost holdout MAPE > 22%, "
            "the 7-day rollback grade): " + ", ".join(sorted(hidden))
        )
    return build_page_title(title="US Grid", subtitle=subtitle, subtitle_tooltip=tooltip)


def _is_real_positive(value) -> bool:
    """Strict guard for downstream arithmetic: True only when ``value`` is
    a finite (non-NaN, non-inf) strictly positive number. Used everywhere
    that sums or divides by a region's ``current_mw``.

    Rejects strings outright (no silent coercion) and normalizes numpy
    bool returns to Python ``bool`` so callers can use ``is True / is False``.
    """
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f) and f > 0)


def _simultaneous_national_peak_mw(populated: dict[str, dict]) -> float:
    """True national peak: the max over the last 24h of the cross-BA demand SUM.

    Before #203 this was ``max`` over regions of each region's own peak — the
    largest *single* BA's peak, which the "National Peak" label misrepresents
    (with real data it necessarily reads several times below the adjacent Total
    Demand). The honest metric is the simultaneous national peak: at each hour,
    sum demand across BAs, then take the max.

    ``today_mw`` is each BA's trailing 24h window (``demand[-24:]``), so the
    windows are right-aligned to "now"; we pad shorter windows on the left and
    ``nansum`` per aligned hour (non-positive/NaN readings drop out). Positional
    alignment is a KPI-grade approximation — per-BA EIA publishing lag can shift
    a window by an hour — but it is strictly more correct than the largest-single-
    BA value it replaces.
    """
    cleaned: list[np.ndarray] = []
    for d in populated.values():
        arr = np.array(
            [float(v) if _is_real_positive(v) else np.nan for v in (d.get("today_mw") or [])],
            dtype=float,
        )
        if arr.size:
            cleaned.append(arr)
    if not cleaned:
        return 0.0
    width = max(a.size for a in cleaned)
    mat = np.full((len(cleaned), width), np.nan)
    for i, a in enumerate(cleaned):
        mat[i, width - a.size :] = a  # right-align trailing windows
    col_sums = np.nansum(mat, axis=0)
    return float(np.nanmax(col_sums)) if col_sums.size else 0.0


# ``_STRESS_RELIABLE_CEILING`` lives in ``_callbacks_shared.py`` — re-exported
# via the module-level star-import. Belt-and-braces fallback for any BA not
# explicitly tagged in ``IS_IMPORT_DOMINATED``; structurally above this
# ratio means the capacity figure is unreliable and the row should be
# excluded from stress rankings.


def _is_implausible_demand_artifact(current_mw: float, today_mw: list) -> bool:
    """Check if a BA's latest demand is an implausible artifact vs. recent history.

    Follows the region-relative low-actual filter pattern from models/drift.py
    (#142): if current_mw is below 10% of the rolling 24h median, treat it as
    a data-quality artifact (e.g. EIA reporting glitch, NaN masquerading as
    negative) rather than real utilization.

    Returns True if current_mw should be excluded from stress calculations.
    """
    if not _is_real_positive(current_mw):
        return True  # Already filtered out by _is_real_positive, but be explicit
    if not today_mw:
        return False  # No history available; assume current is real

    # Extract positive values from 24h history
    positive_history = [float(v) for v in today_mw if _is_real_positive(v)]
    if not positive_history:
        return False  # All history is suspect; can't establish scale, assume current is real

    median_24h = float(np.median(positive_history))
    if median_24h <= 0:
        return False  # Degenerate history; can't establish scale, assume current is real

    # 10% threshold, mirroring drift.LOW_ACTUAL_FRACTION
    threshold = 0.10 * median_24h
    return current_mw < threshold


def _build_us_grid_metrics_items(region_data: dict[str, dict]) -> list[dict]:
    """4-up MetricsBar items: Total Demand · National Peak · Top-Stress BA · National Utilization."""
    populated = {r: d for r, d in region_data.items() if _is_real_positive(d.get("current_mw"))}
    if not populated:
        return [
            {"label": "Total Demand", "value": "—", "tone": "primary", "hero": True},
            {"label": "National Peak (24h)", "value": "—"},
            {"label": "Highest-Stress Region", "value": "—"},
            {"label": "National Utilization", "value": "—"},
        ]

    # Filter out regions with implausible current demand artifacts before computing stress
    plausible = {
        r: d
        for r, d in populated.items()
        if not _is_implausible_demand_artifact(d["current_mw"], d.get("today_mw") or [])
    }

    total_mw = sum(d["current_mw"] for d in plausible.values())
    peak_24h_mw = _simultaneous_national_peak_mw(plausible)

    stress_by_region = {
        region: d["current_mw"] / cap
        for region, d in plausible.items()
        if (cap := REGION_CAPACITY_MW.get(region, 0)) > 0
        # V3.η: structurally import-dominated BAs (CPLW, HST, SPA) are
        # never valid candidates for "highest stress" — their capacity
        # is a peak × 1.15 estimate, not a measured plate, so the stress
        # ratio against it is too noisy to rank against truly-stressed
        # vertically integrated BAs. Filter at source rather than at
        # the reliability ceiling.
        and region not in IS_IMPORT_DOMINATED
    }
    # Belt-and-braces fallback: anything above the reliability ceiling
    # is structural (sustained > 100% means the BA imports most of its
    # power), not real stress. Today this cap should never trip after
    # the V3.η filter above, but keeps the KPI honest if a future BA
    # gets misclassified or if EIA-860M data drifts.
    reliable_stress = {r: s for r, s in stress_by_region.items() if s <= _STRESS_RELIABLE_CEILING}
    if reliable_stress:
        top_region = max(reliable_stress, key=reliable_stress.get)
        # Cap displayed stress at 100% so a tight-day reading of 110%
        # doesn't render as e.g. "PJM · 110%". Matches the map's
        # cap (``_build_us_grid_map`` line ~6821).
        top_stress_pct = min(reliable_stress[top_region], 1.0) * 100
        top_tone = "negative" if top_stress_pct >= 85 else "secondary"
        top_value = f"{top_region} · {top_stress_pct:.0f}%"

        # National utilization = summed current demand ÷ summed nameplate capacity
        # over the SAME reliable-capacity BA set (import-dominated BAs, whose
        # capacity is a peak×1.15 estimate, and any BA above the reliability
        # ceiling are excluded — as they are from Highest-Stress). Nameplate-based,
        # NOT a NERC reserve margin (see #243); reads as the national *average*
        # complementing Highest-Stress Region (the per-BA *maximum*). ``util_capacity``
        # is > 0 because every ``reliable_stress`` key has capacity > 0 by construction.
        util_demand = sum(plausible[r]["current_mw"] for r in reliable_stress)
        util_capacity = sum(REGION_CAPACITY_MW[r] for r in reliable_stress)
        util_pct = util_demand / util_capacity * 100
        util_value = f"{util_pct:.0f}%"
        util_tone = "negative" if util_pct >= 85 else "secondary"
    else:
        top_value = "—"
        top_tone = "secondary"
        util_value = "—"
        util_tone = "secondary"

    return [
        {
            "label": "Total Demand",
            "value": f"{total_mw / 1000:.1f}",
            "unit": "GW",
            "tone": "primary",
            "hero": True,
            "help": "Sum of current demand across all reporting balancing authorities (GW).",
        },
        {
            "label": "National Peak (24h)",
            "value": f"{peak_24h_mw / 1000:.1f}",
            "unit": "GW",
            "help": "Highest simultaneous cross-grid demand in the last 24h. BAs peak at different hours, so this exceeds current total demand.",
        },
        {
            "label": "Highest-Stress Region",
            "value": top_value,
            "tone": top_tone,
            "help": "BA with the highest utilization = current demand ÷ estimated capacity (capped 100%). Import-dominated BAs excluded.",
        },
        {
            "label": "National Utilization",
            "value": util_value,
            "tone": util_tone,
            "help": "Current demand ÷ nameplate capacity, aggregated across BAs with a reliable capacity figure (import-dominated BAs are excluded, as they are from Highest-Stress). Nameplate-based, so it is not a NERC reserve margin — it is the national average that complements Highest-Stress Region (the per-BA maximum).",
        },
    ]


def _build_us_grid_sparkline(values: list[float]) -> html.Div:
    """Inline-SVG sparkline for the last ~24h of a region's demand."""
    if not values or len(values) < 2:
        return html.Div("", className="gp-region-card__sparkline gp-region-card__sparkline--empty")

    width, height = 100.0, 28.0
    vmin = min(values)
    vmax = max(values)
    vrange = max(vmax - vmin, 1.0)
    n = len(values)
    points = " ".join(
        f"{i * width / (n - 1):.1f},{height - (v - vmin) / vrange * height:.1f}"
        for i, v in enumerate(values)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
        f'class="gp-region-card__sparkline-svg" preserveAspectRatio="none" '
        f'aria-hidden="true">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />'
        f"</svg>"
    )
    return html.Div(
        dcc.Markdown(svg, dangerously_allow_html=True),
        className="gp-region-card__sparkline",
    )


# Tiny muted prefix that names each card badge inline (util / net / Δ1h), so a
# card is self-describing — no separate legend to cross-reference. The two GW
# values (hero demand vs. the ``net`` badge) and two % values (``util`` vs.
# ``Δ1h``) are otherwise indistinguishable at a glance.
_CHIP_LABEL_STYLE = {
    "fontSize": "0.7em",
    "opacity": 0.55,
    "marginRight": "3px",
    "fontWeight": 400,
    "letterSpacing": "0.03em",
}


def _chip_label(text: str) -> html.Span:
    """Small muted in-badge label (e.g. ``util``, ``net``, ``Δ1h``)."""
    return html.Span(text, className="gp-region-card__chip-label", style=_CHIP_LABEL_STYLE)


def _build_interchange_chip(interchange: dict | None) -> html.Span | None:
    """V3.α: net BA-to-BA interchange chip.

    Renders ``+1.2 GW`` when exporting (positive net), ``-0.8 GW`` when
    importing (negative net), or ``≈0`` when net is below the visual
    threshold. Hover tooltip lists the top counterparty BAs and their
    signed flows.
    """
    if not interchange or interchange.get("net_mw") is None:
        return None

    net_mw = float(interchange["net_mw"])
    counterparties = interchange.get("counterparties") or []

    # Below ±50 MW the sign isn't meaningful at GW resolution; render as ≈0
    # to avoid noise. Real flows in the system are typically 100s–1000s MW.
    if abs(net_mw) < 50:
        label = "≈0"
        tone = "neutral"
    else:
        gw = net_mw / 1000.0
        sign = "+" if gw >= 0 else "−"
        label = f"{sign}{abs(gw):.1f} GW"
        tone = "export" if gw >= 0 else "import"

    if counterparties:
        parts = []
        for cp in counterparties:
            mw = float(cp["mw"])
            cp_sign = "+" if mw >= 0 else "−"
            parts.append(f"{cp_sign}{abs(mw):,.0f} MW → {cp['to_ba']}")
        tooltip = "Top counterparties (last hour): " + " · ".join(parts)
    else:
        tooltip = f"Net interchange (last hour): {net_mw:+,.0f} MW"

    return html.Span(
        [_chip_label("net"), label],
        className=f"gp-region-card__interchange gp-region-card__interchange--{tone}",
        title=tooltip,
    )


def _us_grid_card_sort_value(region: str, data: dict, sort: str):
    """Sort key for one region card under a non-default US Grid sort.

    Numeric sorts return a value ordered so Python's ascending ``sort``
    yields the intended visual order — demand / utilization / hourly-change
    high→low, name A→Z. BAs with missing or unreliable data sort to the
    bottom rather than the top.
    """
    cur = float(data["current_mw"]) if _is_real_positive(data.get("current_mw")) else 0.0
    if sort == "demand":
        return -cur  # high → low
    if sort == "stress":
        cap = REGION_CAPACITY_MW.get(region, 0)
        # Mirror the Highest-Stress KPI: only vertically-integrated BAs with
        # a real capacity plate rank on utilization; import-dominated BAs
        # (capacity is a peak×1.15 estimate) sort to the bottom instead of
        # crowding the top with a noisy ratio.
        if cap > 0 and cur > 0 and region not in IS_IMPORT_DOMINATED:
            return -(cur / cap)  # high → low
        return 1.0  # unreliable → bottom (positive sorts after the negatives)
    if sort == "change":
        prev = data.get("prev_mw")
        if prev and prev > 0 and cur > 0:
            return -abs((cur - prev) / prev)  # biggest movers first
        return 0.0  # no prior hour → bottom
    if sort == "name":
        return REGION_NAMES.get(region, region).lower()  # A → Z
    return region


def _build_us_grid_region_card(region: str, data: dict) -> html.Div:
    """One region card for the US Grid small-multiples grid.

    Empty ``data`` (no Redis row yet) renders an "—" placeholder card that's
    still clickable — drilling down lands on the Forecast tab so the user
    can see the warming state per region rather than guessing.
    """
    name = REGION_NAMES.get(region, region)
    card_id = {"type": "us-grid-region-card", "region": region}

    # ``_collect_us_grid_region_data`` returns ``None`` for unreal demand
    # (NaN / non-finite / non-positive). The strict ``_is_real_positive``
    # check is redundant given that source — kept as a safety net so a
    # future regression in the collector can't render literal "nan" here.
    if not data or not _is_real_positive(data.get("current_mw")):
        return html.Div(
            [
                html.Div(
                    [html.Span(name, className="gp-region-card__name")],
                    className="gp-region-card__header",
                ),
                html.Div("—", className="gp-region-card__demand-empty"),
            ],
            id=card_id,
            n_clicks=0,
            className="gp-region-card gp-region-card--empty",
            title=f"{name} · pipeline warming up",
        )

    current_mw = data["current_mw"]
    prev_mw = data.get("prev_mw")
    today_mw = data.get("today_mw") or []
    capacity_mw = REGION_CAPACITY_MW.get(region, 0)

    delta_chip = None
    if prev_mw and prev_mw > 0:
        delta_pct = (current_mw - prev_mw) / prev_mw * 100
        sign = "+" if delta_pct >= 0 else ""
        direction = "up" if delta_pct >= 0 else "down"
        delta_chip = html.Span(
            [_chip_label("Δ1h"), f"{sign}{delta_pct:.1f}%"],
            className=f"gp-region-card__delta gp-region-card__delta--{direction}",
        )

    stress_chip = None
    # Guard against implausible demand artifacts (#225) — if current_mw is an
    # outlier below 10% of 24h median, treat it as unavailable rather than
    # computing a misleading stress percentage.
    if capacity_mw > 0 and not _is_implausible_demand_artifact(current_mw, today_mw):
        stress_ratio = current_mw / capacity_mw
        # V3.η: structurally import-dominated BAs (CPLW, HST, SPA) get
        # the qualitative "imports" chip whether or not their stress
        # ratio happens to land above the ceiling on a given hour. The
        # capacity figure for these BAs is a peak × 1.15 estimate, not
        # measured plate, so the percentage isn't meaningful even when
        # it's mathematically below 100%.
        is_import_dominated = (
            region in IS_IMPORT_DOMINATED or stress_ratio > _STRESS_RELIABLE_CEILING
        )
        if is_import_dominated:
            stress_chip = html.Span(
                "imports",
                className="gp-region-card__stress gp-region-card__stress--imports",
                title=(
                    f"{name} is an import-dominated balancing authority — "
                    f"its served demand routinely exceeds in-territory generation "
                    f"and is supplied via inter-BA transfers. The utilization % "
                    f"is calculated against an estimated capacity pool, not a "
                    f"measured plate."
                ),
            )
        else:
            # Cap displayed stress at 100% so a tight-day reading of
            # 110% renders as "100%". Matches the choropleth and map.
            stress_pct = min(stress_ratio, 1.0) * 100
            if stress_pct >= 85:
                tone = "high"
            elif stress_pct >= 70:
                tone = "mid"
            else:
                tone = "low"
            stress_chip = html.Span(
                [_chip_label("util"), f"{stress_pct:.0f}%"],
                className=f"gp-region-card__stress gp-region-card__stress--{tone}",
                title=(f"Demand vs. capacity: {current_mw:,.0f} / {capacity_mw:,.0f} MW"),
            )

    demand_row_children: list = [
        html.Span(
            f"{current_mw / 1000:.1f}",
            className="gp-region-card__demand-value tabular",
        ),
        html.Span("GW", className="gp-region-card__demand-unit"),
        html.Span(
            "demand",
            className="gp-region-card__demand-label",
            style={"fontSize": "0.62rem", "opacity": 0.5, "letterSpacing": "0.04em"},
        ),
    ]
    if delta_chip is not None:
        demand_row_children.append(delta_chip)

    interchange_chip = _build_interchange_chip(data.get("interchange"))

    header_children: list = [html.Span(name, className="gp-region-card__name")]
    if stress_chip is not None:
        header_children.append(stress_chip)
    if interchange_chip is not None:
        header_children.append(interchange_chip)

    return html.Div(
        [
            html.Div(header_children, className="gp-region-card__header"),
            html.Div(demand_row_children, className="gp-region-card__demand"),
            _build_us_grid_sparkline(today_mw),
        ],
        id=card_id,
        n_clicks=0,
        className="gp-region-card",
    )


# ── V1.γ: US GRID MAP HELPERS ────────────────────────────────
#
# The ``_MAP_*`` tokens (land / coastline / subunit / border colors +
# the colorscale) live in ``_callbacks_shared.py`` and are re-exported
# via the module-level star-import. Plotly figures need the hex values
# inline because Plotly doesn't read CSS custom properties; the shared
# module keeps them in sync with ``:root`` in ``assets/custom.css``.


def _build_us_grid_map(region_data: dict) -> html.Div:
    """Plotly ``scatter_geo`` of BA centroids — sized by demand, colored by stress.

    Cold or empty regions are dropped from the figure (rather than rendering
    as zero-size markers). When every region is cold the function returns an
    empty-state div — the page already has the warming language in the title.
    """
    # Drop NaN / non-finite / zero demand regions — they would render as
    # zero-size markers and (worse) could push NaN into the colorscale
    # via the stress percentage divide below.
    populated = {r: d for r, d in region_data.items() if _is_real_positive(d.get("current_mw"))}

    if not populated:
        return html.Div(
            html.Div(
                "All regions warming up — switch to Cards view to see per-region status.",
                className="gp-region-map__empty-message",
            ),
            className="gp-region-map gp-region-map--empty",
        )

    regions = list(populated.keys())
    lats = [REGION_COORDINATES[r]["lat"] for r in regions]
    lons = [REGION_COORDINATES[r]["lon"] for r in regions]
    names = [REGION_NAMES.get(r, r) for r in regions]
    demand_gw = [populated[r]["current_mw"] / 1000 for r in regions]

    # Stress = demand / capacity (0..>1). Cap at 100% for the colorscale.
    # Guard against implausible demand artifacts (#225): if a region's current
    # demand is a clear outlier vs. its 24h history, render as 0% (not colored)
    # rather than computing a misleading stress percentage.
    stress_pct: list[float] = []
    for r in regions:
        cap = REGION_CAPACITY_MW.get(r, 0)
        if cap > 0 and not _is_implausible_demand_artifact(
            populated[r]["current_mw"], populated[r].get("today_mw") or []
        ):
            stress_pct.append(min(populated[r]["current_mw"] / cap * 100, 100.0))
        else:
            stress_pct.append(0.0)

    # Marker size: 10–40 px, scaled by demand. Linear is fine for 16 points.
    max_demand = max(demand_gw)
    sizes = [10 + (d / max_demand) * 30 for d in demand_gw]

    # V3.η: append " · imports" suffix to the BA name for structurally
    # import-dominated BAs (CPLW, HST, SPA) so users understand what
    # the utilization % is measured against (a peak × 1.15 estimate,
    # not a measured plate capacity).
    hover_text = [
        f"<b>{name}{' · imports' if r in IS_IMPORT_DOMINATED else ''}</b>"
        f"<br>Demand: {d:.1f} GW<br>Utilization: {s:.0f}%"
        for r, name, d, s in zip(regions, names, demand_gw, stress_pct, strict=False)
    ]

    fig = go.Figure(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            marker={
                "size": sizes,
                "color": stress_pct,
                "colorscale": _MAP_COLORSCALE,
                "cmin": 0,
                "cmax": 100,
                "colorbar": {
                    "title": {
                        "text": "Utilization %",
                        "font": {"color": _MAP_AXIS_FONT_COLOR, "size": 10},
                    },
                    "tickfont": {"color": _MAP_AXIS_FONT_COLOR, "size": 10},
                    "thickness": 10,
                    "len": 0.6,
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": _MAP_BORDER_COLOR,
                    "borderwidth": 1,
                    "x": 1.0,
                },
                "line": {"color": _MAP_BORDER_COLOR, "width": 1},
            },
            customdata=regions,
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor=_MAP_LAND_COLOR,
        showcoastlines=True,
        coastlinecolor=_MAP_COASTLINE_COLOR,
        showsubunits=True,
        subunitcolor=_MAP_SUBUNIT_COLOR,
        showcountries=False,
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        font={"family": "Inter, system-ui, sans-serif", "color": _MAP_AXIS_FONT_COLOR, "size": 11},
        height=480,
        showlegend=False,
        hoverlabel={
            "bgcolor": _MAP_LAND_COLOR,
            "bordercolor": _MAP_COASTLINE_COLOR,
            "font": {"color": "#e4e4e7", "family": "Inter, system-ui, sans-serif", "size": 12},
        },
    )

    return html.Div(
        dcc.Graph(
            id="us-grid-map",
            figure=fig,
            config={"displayModeBar": False, "responsive": True},
            style={"height": "480px"},
        ),
        className="gp-region-map",
    )


# ── V3.β: Choropleth (real BA service-territory polygons) ────


# In-memory cache for the polygon GeoJSON. Read once per process.
# The asset is ~165 KB so the parse is cheap, but we don't want to
# do it on every render.
_BA_POLYGONS_CACHE: dict | None = None


def _load_ba_polygons() -> dict | None:
    """Load and cache the BA-polygon GeoJSON from ``assets/``.

    Returns ``None`` if the file is missing or malformed — the
    choropleth helper falls back to the centroid scatter in that case
    so a corrupt asset doesn't black out the Map view entirely.
    """
    global _BA_POLYGONS_CACHE
    if _BA_POLYGONS_CACHE is not None:
        return _BA_POLYGONS_CACHE

    import json as _json
    from pathlib import Path

    asset_path = Path(__file__).parent.parent / "assets" / "ba_polygons.geojson"
    try:
        with open(asset_path) as f:
            _BA_POLYGONS_CACHE = _json.load(f)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("ba_polygons_load_failed", error=str(exc))
        _BA_POLYGONS_CACHE = None
    return _BA_POLYGONS_CACHE


def _build_us_grid_choropleth(region_data: dict) -> html.Div:
    """Plotly ``Choropleth`` of BA service-territory polygons — colored
    by demand-vs-capacity utilization. Same colorscale + drilldown as
    the centroid scatter so users can switch between them seamlessly.

    V3.β closes the deferral noted in V1.γ — centroids were "80% of the
    visual punch for 10% of the cost"; this delivers the remaining 20%.
    Polygons sourced from electricitymaps-contrib's world.geojson (MIT
    license), filtered to our 51 BA codes via EIA-930 respondent
    suffixes, ~165 KB pre-simplified.

    Falls back to the centroid scatter when the GeoJSON asset is
    missing or unreadable so a corrupt file can't black out the tab.
    """
    geojson = _load_ba_polygons()
    if geojson is None:
        return _build_us_grid_map(region_data)

    populated = {r: d for r, d in region_data.items() if _is_real_positive(d.get("current_mw"))}
    if not populated:
        return html.Div(
            html.Div(
                "All regions warming up — switch to Cards view to see per-region status.",
                className="gp-region-map__empty-message",
            ),
            className="gp-region-map gp-region-map--empty",
        )

    regions = list(populated.keys())
    # Stress = demand / capacity (0..>1). Cap at 100% for the colorscale.
    # Guard against implausible demand artifacts (#225): if a region's current
    # demand is a clear outlier vs. its 24h history, render as 0% (not colored)
    # rather than computing a misleading stress percentage.
    stress_pct: list[float] = []
    for r in regions:
        cap = REGION_CAPACITY_MW.get(r, 0)
        if cap > 0 and not _is_implausible_demand_artifact(
            populated[r]["current_mw"], populated[r].get("today_mw") or []
        ):
            stress_pct.append(min(populated[r]["current_mw"] / cap * 100, 100.0))
        else:
            stress_pct.append(0.0)

    names = [REGION_NAMES.get(r, r) for r in regions]
    demand_gw = [populated[r]["current_mw"] / 1000 for r in regions]
    # V3.η: customdata[3] = " · imports" suffix for import-dominated
    # BAs, empty string for everyone else. Plotly hovertemplate renders
    # the empty string invisibly so we don't need separate templates.
    import_tag = [" · imports" if r in IS_IMPORT_DOMINATED else "" for r in regions]
    customdata = list(zip(regions, names, demand_gw, import_tag, strict=False))

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.region",
            locations=regions,
            z=stress_pct,
            colorscale=_MAP_COLORSCALE,
            zmin=0,
            zmax=100,
            marker={"line": {"color": _MAP_BORDER_COLOR, "width": 1.0}},
            colorbar={
                "title": {
                    "text": "Utilization %",
                    "font": {"color": _MAP_AXIS_FONT_COLOR, "size": 10},
                },
                "tickfont": {"color": _MAP_AXIS_FONT_COLOR, "size": 10},
                "thickness": 10,
                "len": 0.6,
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": _MAP_BORDER_COLOR,
                "borderwidth": 1,
                "x": 1.0,
            },
            # Drilldown reads ``customdata`` (the BA code) — same shape
            # as the scatter so the existing ``us-grid-map.clickData``
            # callback handles it without changes.
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[1]}</b>%{customdata[3]}<br>"
                "Demand: %{customdata[2]:.1f} GW<br>"
                "Utilization: %{z:.0f}%<extra></extra>"
            ),
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor=_MAP_LAND_COLOR,
        showcoastlines=True,
        coastlinecolor=_MAP_COASTLINE_COLOR,
        showsubunits=True,
        subunitcolor=_MAP_SUBUNIT_COLOR,
        showcountries=False,
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        font={"family": "Inter, system-ui, sans-serif", "color": _MAP_AXIS_FONT_COLOR, "size": 11},
        height=480,
        showlegend=False,
        hoverlabel={
            "bgcolor": _MAP_LAND_COLOR,
            "bordercolor": _MAP_COASTLINE_COLOR,
            "font": {"color": "#e4e4e7", "family": "Inter, system-ui, sans-serif", "size": 12},
        },
    )

    # Coverage caption — disambiguates the dark-fill gaps in the
    # Polygons view (Idaho, parts of Montana / Wyoming / the Plains,
    # AK + HI insets) as intentional unmapped territory, not a
    # rendering bug. EIA-930 has 63 BAs in the contiguous U.S. (per
    # the August 2025 Federal Register PRA renewal, stable since 2022);
    # we ship polygons for 51 (~80% of demand). Renders only inside
    # the Polygons view body — Cards / Map don't have coverage gaps
    # to disclose.
    n_covered = len(regions)
    caption = html.Div(
        f"{n_covered} of 63 EIA-930 balancing authorities in the contiguous U.S. mapped · "
        "dark areas are BAs not yet covered here.",
        className="gp-region-map__coverage-caption",
    )

    return html.Div(
        [
            dcc.Graph(
                id="us-grid-map",
                figure=fig,
                config={"displayModeBar": False, "responsive": True},
                style={"height": "480px"},
            ),
            caption,
        ],
        className="gp-region-map",
    )


# ── Callback registration (Step 10b — register_callbacks split) ──────


def register_us_grid_callbacks(app):
    """Register US Grid tab callbacks with the Dash app.

    Step 10b of the ``register_callbacks`` decomposition. Owns the
    three callbacks the US Grid tab needs: snapshot rendering (cards/
    map/polygons toggle), card-click drilldown, and map-click drilldown.
    All three reuse the helpers defined in this module.
    """
    from components.cards import build_metrics_bar
    from config import REGION_GROUPS

    @app.callback(
        [
            Output("us-grid-title", "children"),
            Output("us-grid-metrics-bar", "children"),
            Output("us-grid-region-grid", "children"),
        ],
        [
            Input("dashboard-tabs", "active_tab"),
            Input("refresh-interval", "n_intervals"),
            Input("us-grid-view-toggle", "value"),
            Input("us-grid-sort", "value"),
        ],
    )
    def update_us_grid_snapshot(active_tab, _n_intervals, view, sort):
        """Render the US Grid tabs title, MetricsBar, and body (cards or map)."""
        if active_tab != "tab-us-grid":
            return [no_update] * 3

        view = view or "cards"
        sort = sort or "groups"

        try:
            region_data = _collect_us_grid_region_data()
            title = _build_us_grid_title(region_data)
            metrics_items = _build_us_grid_metrics_items(region_data)
            metrics_bar = build_metrics_bar(metrics_items)
            metrics_bar.className = f"gp-metrics-bar gp-metrics-bar--{len(metrics_items)}up"

            if view == "map":
                body = _build_us_grid_map(region_data)
            elif view == "polygons":
                body = _build_us_grid_choropleth(region_data)
            else:
                # No legend: each card now self-describes via inline badge
                # labels (util / net / Δ1h / demand), so a separate key to
                # cross-reference is redundant.
                grid_children: list = []
                if sort == "groups":
                    for group_name, codes in REGION_GROUPS.items():
                        visible = [c for c in codes if c in region_data]
                        if not visible:
                            continue
                        grid_children.append(
                            html.Div(
                                group_name,
                                className="gp-region-grid__section-header",
                            )
                        )
                        grid_children.extend(
                            _build_us_grid_region_card(code, region_data[code]) for code in visible
                        )
                else:
                    # Non-default sort flattens the regional grouping into one
                    # ranked grid. Universe = the same BAs the grouped view shows.
                    ordered = [
                        c for codes in REGION_GROUPS.values() for c in codes if c in region_data
                    ]
                    ordered.sort(key=lambda c: _us_grid_card_sort_value(c, region_data[c], sort))
                    grid_children.extend(
                        _build_us_grid_region_card(code, region_data[code]) for code in ordered
                    )
                body = html.Div(grid_children, className="gp-region-grid")

            return (title, metrics_bar, body)
        except Exception as exc:
            log.exception("update_us_grid_snapshot_failed")
            err_msg = f"{type(exc).__name__}: {exc}"
            err_div = html.Div(
                err_msg,
                style={"color": "var(--danger)", "fontSize": "0.8rem", "padding": "8px"},
            )
            return (err_div, html.Div(), err_div)

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        ],
        Input({"type": "us-grid-region-card", "region": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def drilldown_from_us_grid(n_clicks_list):
        """Click a region card → open Forecast tab pre-set to that region."""
        if not n_clicks_list or not any(n for n in n_clicks_list if n):
            return no_update, no_update
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or triggered.get("type") != "us-grid-region-card":
            return no_update, no_update
        region = triggered.get("region")
        if not region:
            return no_update, no_update
        return region, "tab-outlook"

    @app.callback(
        [
            Output("region-selector", "value", allow_duplicate=True),
            Output("dashboard-tabs", "active_tab", allow_duplicate=True),
        ],
        Input("us-grid-map", "clickData"),
        prevent_initial_call=True,
    )
    def drilldown_from_us_grid_map(click_data):
        """Click a map point → open Forecast tab pre-set to that region.

        Tolerates two ``customdata`` shapes so the same callback works
        for both the ``scatter_geo`` view (1-D array of region codes)
        and the V3.β ``Choropleth`` view (2-D array where index 0 is
        the region code, indexes 1-2 carry hover text fields).
        """
        if not click_data:
            return no_update, no_update
        points = click_data.get("points") or []
        if not points:
            return no_update, no_update
        cd = points[0].get("customdata")
        if cd is None:
            return no_update, no_update
        region = cd[0] if isinstance(cd, (list, tuple)) else cd
        if not region:
            return no_update, no_update
        return region, "tab-outlook"
