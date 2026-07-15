"""Measure the figures the app BUILDS, not the palette it declares.

Why this file exists
--------------------
Every other color check in this repo asks "does the code match my spec?" All of
them passed while four real defects shipped, because the defects lived where the
SPEC was wrong:

  * ``scripts/verify_palette.py`` measures the accent against a hand-written list
    of "series it shares a figure with". That list omits Wong's sky_blue, citing
    an analysis that says the two never co-occur. They co-occur on the Forecast
    tab, in the default state, at CIEDE2000 9.2. The gate excluded the one rival
    that actually collides.
  * ``test_color_tokens`` pins ``FUEL_COLORS`` to what ``_FUEL_DISPLAY`` renders.
    Both are keyed to the nine fuels I enumerated. Real EIA data yields ``bat``
    and ``snb``, which are in neither, so both fall through to ONE shared
    fallback color and paint two adjacent bands identically.
  * Every existing test of the outlook figure ``@patch``es out
    ``_add_trailing_actuals`` — the function that adds the colliding trace. The
    figure under test cannot contain the defect.

The common shape: my analysis was the input to my checks. So this file does not
consult it. It builds the real figures, reads the traces back out, and measures
every pair WITHIN each figure. It mocks only the I/O boundary (Redis, the EIA
cache); composition runs for real.

What it asserts:

  1. A trace's fill must derive from its own line. A fill in a different color
     from the line it sits under is always a bug.
  2. WCAG 1.4.1: any two traces in one figure that are close in color must be
     separated by a second channel (dash, mode, or marker symbol). This is the
     real rule — color alone is what 1.4.1 forbids, and a floor on CIEDE2000 is
     only a proxy for it.

It also PRINTS every close pair with its measurement, whether or not a second
channel rescues it, so the numbers are visible rather than implied.
"""

from __future__ import annotations

import pathlib
import sys
from itertools import combinations
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from color_science import ciede2000, simulate  # noqa: E402

CVD = ("normal", "protan", "deutan", "tritan")

# Below this, color alone is not doing the work and a second channel must exist.
# Same number the rest of the repo uses (scripts/verify_palette.SHARED_FIGURE_FLOOR).
CLOSE = 12.0


def min_cvd(a: str, b: str) -> float:
    return min(ciede2000(simulate(a, k), simulate(b, k)) for k in CVD)


def _rgb_of(color: str) -> tuple[int, int, int] | None:
    """Channels of a #rrggbb or rgba(...) string, or None if unparseable."""
    if not isinstance(color, str):
        return None
    c = color.strip()
    if c.startswith("#") and len(c) == 7:
        return tuple(int(c[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    if c.startswith("rgb"):
        nums = c[c.index("(") + 1 : c.index(")")].split(",")
        if len(nums) >= 3:
            try:
                return tuple(int(float(n)) for n in nums[:3])  # type: ignore[return-value]
            except ValueError:
                return None
    return None


def _hex_of(color: str) -> str | None:
    rgb = _rgb_of(color)
    return None if rgb is None else "#{:02x}{:02x}{:02x}".format(*rgb)


def _primary(tr) -> str | None:
    """The color a trace is 'in' — its line, else its marker."""
    line = getattr(tr, "line", None)
    if line is not None and isinstance(getattr(line, "color", None), str):
        return line.color
    mk = getattr(tr, "marker", None)
    if mk is not None and isinstance(getattr(mk, "color", None), str):
        return mk.color
    return None


def _channel(tr) -> tuple:
    """The non-color channels distinguishing this trace: dash, mode, symbol."""
    line = getattr(tr, "line", None)
    mk = getattr(tr, "marker", None)
    fp = getattr(tr, "fillpattern", None)
    return (
        getattr(line, "dash", None) if line is not None else None,
        getattr(tr, "mode", None),
        getattr(mk, "symbol", None) if mk is not None else None,
        getattr(tr, "type", None),
        # A fill pattern is a second channel exactly as a dash is — it is what
        # lets a 16-fuel stack exist at all, since no 16 colors separate under CVD.
        (getattr(fp, "shape", None) or None) if fp is not None else None,
    )


# ── Real figures, built the way the app builds them ──────────────────


def _demand_df(n: int = 168) -> pd.DataFrame:
    idx = pd.date_range("2026-07-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": idx, "demand_mw": [20000 + (i % 24) * 400 for i in range(n)]})


def _demand_json(n: int = 168) -> str:
    return _demand_df(n).to_json(date_format="iso", orient="records")


def _forecast_payload(n: int, model: str) -> dict:
    ts = pd.date_range("2026-07-08", periods=n, freq="h", tz="UTC")
    return {
        "scored_at": "2026-07-08T00:00:00Z",
        "forecasts": [
            {
                "timestamp": t.isoformat(),
                model: 30000 + i * 10,
                "predicted_demand_mw": 30000 + i * 10,
            }
            for i, t in enumerate(ts)
        ],
    }


def _generation_df() -> pd.DataFrame:
    """RAW EIA fueltype codes, exactly as the API returns them.

    Raw on purpose: the code -> canonical-name normalisation lives INSIDE
    _fetch_generation_cached, so a fixture of pre-normalised names would skip the
    step under test. The first version of this fixture did exactly that.

    These are the codes the live API returns for FPL (verified 2026-07-15), plus
    the storage-paired and geothermal codes other BAs return, so the check
    exercises the whole vocabulary rather than one region's subset.
    """
    idx = pd.date_range("2026-07-14", periods=24, freq="h", tz="UTC")
    rows = []
    for code, base in [
        ("NG", 12000),
        ("NUC", 3000),
        ("SUN", 2000),
        ("OIL", 40),
        ("OTH", 800),
        ("BAT", 50),
        ("SNB", 120),
        ("COL", 900),
        ("WAT", 400),
        ("WND", 700),
        ("GEO", 60),
        ("WNB", 80),
        ("PS", 30),
        ("OES", 20),
        ("UES", 10),
        ("UNK", 5),
    ]:
        for t in idx:
            rows.append({"timestamp": t, "fuel_type": code, "generation_mw": base, "region": "FPL"})
    return pd.DataFrame(rows)


def _outlook_figures():
    """The Forecast tab chart, for every model the user can select.

    Mocks ONLY redis_get. Every existing test of this figure also patches
    _add_trailing_actuals and _add_confidence_bands — which is why none of them
    could see a collision between the actual line and the model line.
    """
    from components.callbacks import _outlook_tab_from_redis

    out = []
    for model in ("prophet", "arima", "xgboost", "ensemble"):
        with patch("components._callbacks_forecast.redis_get") as rg:
            rg.return_value = _forecast_payload(72, model)
            res = _outlook_tab_from_redis("FPL", 48, model, _demand_json(), None, "grid_ops")
        if res and isinstance(res[0], go.Figure):
            out.append((f"Forecast outlook — {model} selected", res[0]))
    return out


def _generation_figure():
    """The fuel-mix stacked area, driven by the fuel codes real EIA data has."""
    from components import _callbacks_overview as ov

    # Mock the I/O boundary only: no Redis, and the EIA fetch returns raw codes.
    # _fetch_generation_cached then runs its real _EIA_FUEL_MAP normalisation —
    # which is the step that decides whether two fuels share a color.
    ov._GENERATION_CACHE.clear()
    with (
        patch("components._callbacks_overview._generation_df_from_redis", return_value=None),
        patch("components._callbacks_overview.REQUIRE_REDIS", False),
        patch("data.eia_client.fetch_generation_by_fuel", return_value=_generation_df()),
    ):
        panel = ov._build_generation_panel("FPL", _demand_json())
    ov._GENERATION_CACHE.clear()

    figs = []

    def walk(node):
        fig = getattr(node, "figure", None)
        if isinstance(fig, go.Figure):
            figs.append(fig)
        for child in (
            (getattr(node, "children", None) or [])
            if not isinstance(getattr(node, "children", None), str)
            else []
        ):
            walk(child)

    walk(panel)
    return [("Generation fuel mix", f) for f in figs if len(f.data) > 1]


def _hero_figure():
    from components._callbacks_overview import _build_overview_hero_chart

    return [("Overview hero", _build_overview_hero_chart("FPL", _demand_df()))]


def _weather_df(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2026-07-13", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "temperature_2m": [78 + (i % 12) for i in range(n)],
            "wind_speed_80m": [8 + (i % 7) for i in range(n)],
            "shortwave_radiation": [max(0, 700 - abs(12 - i % 24) * 60) for i in range(n)],
        }
    )


def _drivers_figures():
    """The Forecast tab's Drivers strip: one labeled sparkline per driver.

    This is the only surface that draws the weather drivers. It briefly was not
    — a persona "renewables spotlight" overplotted wind and solar as two lines,
    which is a different and stricter problem than three separate cells. That
    figure turned out to be unreachable and was deleted, along with the three
    sibling spotlights, so the drivers are back to being merely adjacent.

    The corpus covers this rather than those, because a check that measures
    something nobody renders reports coverage it does not have — which is the
    exact defect this file was written to replace.
    """
    from components import _callbacks_overview as ov

    panel = ov._build_drivers_panel(_weather_df().to_json(date_format="iso", orient="records"))
    figs = []

    def walk(node):
        fig = getattr(node, "figure", None)
        if isinstance(fig, go.Figure):
            figs.append(fig)
        kids = getattr(node, "children", None)
        for child in kids if isinstance(kids, list) else [kids] if kids is not None else []:
            if not isinstance(child, str):
                walk(child)

    for node in panel if isinstance(panel, list) else [panel]:
        walk(node)
    return [(f"Drivers strip — cell {i + 1}", f) for i, f in enumerate(figs)]


def all_figures() -> list[tuple[str, go.Figure]]:
    figs: list[tuple[str, go.Figure]] = []
    for builder in (_hero_figure, _outlook_figures, _generation_figure, _drivers_figures):
        try:
            figs.extend(builder())
        except Exception as e:  # a builder that cannot run is reported, not skipped silently
            figs.append(
                (f"{builder.__name__} FAILED TO BUILD: {type(e).__name__}: {e}", go.Figure())
            )
    return figs


@pytest.fixture(scope="module")
def figures():
    return all_figures()


# ── The checks ───────────────────────────────────────────────────────


def test_every_builder_ran(figures):
    """A builder that cannot be exercised is a hole, not a pass."""
    broken = [n for n, _ in figures if "FAILED TO BUILD" in n]
    assert not broken, f"figure builders did not run: {broken}"
    assert len(figures) >= 9, (
        f"expected the hero + 4 outlook models + generation + 3 driver cells, got {len(figures)}"
    )


# What a trace CALLING ITSELF X must be painted. Keyed by the trace name the
# builders actually use, lowercased. This is not a second palette — every value
# is read out of the shipped one at call time.
def _identity_of(name: str) -> str | None:
    from components import tokens
    from components._callbacks_overview import _FUEL_LABELS
    from components._callbacks_shared import COLORS

    n = (name or "").strip().lower()
    if n in ("demand", "actual", "actual demand"):
        return COLORS["actual"]
    for model in ("prophet", "arima", "xgboost", "ensemble"):
        if n.startswith(model):
            return COLORS[model]
    # Fuel bands, matched against the label map the panel itself renders from
    # rather than by guessing at the string.
    for fuel, label in _FUEL_LABELS.items():
        if n == label.lower():
            return COLORS[fuel]
    # A physical driver is one quantity, so it is one color wherever it is
    # drawn. The renewables panel drew wind in XGBoost's sky_blue and solar in
    # Prophet's orange while the driver strip drew both from WEATHER_DRIVERS —
    # the same one-series-two-hexes shape as demand, a layer down.
    #
    # Matched on the unit, because "Solar" the fuel band and "Solar (W/m²)" the
    # irradiance trace are different quantities that share a word, and they are
    # deliberately different colors (tokens.WEATHER_DRIVERS says why). A
    # startswith() here reported the fuel bands as defects — the check being
    # wrong about the code, again, and the reason this matches maps instead.
    for driver, unit in (("wind", "(mph)"), ("solar", "(w/m²)"), ("temperature", "(°f)")):
        if n.startswith(driver) and unit in n:
            return tokens.WEATHER_DRIVERS[driver]
    # "Peak"/"Min" are deliberately absent: they annotate whichever curve they
    # were drawn on, so they have no fixed identity of their own. The first
    # draft of this map listed them as demand and duly reported eight failures
    # against a rule nobody had decided on. What they must satisfy is
    # derivation, not identity — test_extremum_markers_derive_from_their_curve.
    return None


def _points(fig) -> list[tuple[str, str]]:
    """(label, color) for every PER-POINT color in a figure.

    A bar/pie trace colors its points via a LIST on marker.color, keyed by x
    rather than by trace name — so ``_primary`` returns None for it and every
    check built on ``_primary`` skips it silently. That is how the accuracy
    panel's hand-written [vermillion, blue, green] went unmeasured while three
    model identities sat right there in its x values.
    """
    out = []
    for tr in fig.data:
        mk = getattr(tr, "marker", None)
        colors = getattr(mk, "color", None) if mk is not None else None
        if isinstance(colors, str) or colors is None:
            continue
        labels = getattr(tr, "x", None) or getattr(tr, "labels", None) or []
        # strict=False: Plotly itself tolerates a short color list by cycling
        # it, so a length mismatch is not an error to raise here.
        for label, color in zip(labels, colors, strict=False):
            if isinstance(label, str) and isinstance(color, str):
                out.append((label, color))
    return out


def test_every_named_series_wears_its_own_identity(figures):
    """A series named for a thing must be painted in that thing's color.

    The pair-distance check cannot see this. It measures traces AGAINST EACH
    OTHER, so a figure whose every trace agrees with itself passes even when
    the whole figure is painting the wrong thing — and a lone trace has no pair
    at all. That blind spot is where "the palette is downloaded, not designed"
    survived the unification: the Overview drew "Demand" in Wong's blue while
    the Forecast tab drew the same series in the accent (the two-hex defect the
    audit named), and the accuracy panel painted prophet/arima/xgboost in
    vermillion/blue/green — two of the three wearing ANOTHER model's identity,
    from a hand-written list that never consulted LINE_STYLES.

    Distance is a property of a pair. Identity is a property of one trace. Both
    have to be checked, and only this one is checked against the meaning.
    """
    from components._callbacks_shared import COLORS

    inverse = {v.lower(): k for k, v in COLORS.items()}
    bad = []

    def check(where: str, label: str, painted: str | None):
        want = _identity_of(label)
        got = _hex_of(painted) if painted else None
        if want is None or got is None or got.lower() == want.lower():
            return
        worn = inverse.get(got.lower())
        bad.append(
            f"{where}: {label!r} is {got} but its identity is {want}"
            + (f" — it is wearing {worn.upper()}'s color" if worn else "")
        )

    for name, fig in figures:
        for tr in fig.data:
            check(name, getattr(tr, "name", None) or "", _primary(tr))
        for label, color in _points(fig):
            check(name, label, color)

    assert not bad, "series painted as something other than themselves:\n  " + "\n  ".join(bad)


def test_extremum_markers_derive_from_their_curve(figures):
    """Peak/Min mark a point ON a curve, so they are drawn IN that curve.

    Same rule as the fill, and for the same reason: they are part of one object.
    They used to be DANGER and ACCENT — a routine daily maximum in the alert
    color, and a min marker sitting at CIEDE2000 0.0 from the demand line.
    """
    bad = []
    for name, fig in figures:
        curves = [
            _hex_of(_primary(t))
            for t in fig.data
            if str(getattr(t, "name", "")).lower().endswith("forecast") and _primary(t)
        ]
        if len(curves) != 1:
            continue  # ambiguous: which curve would it annotate?
        for tr in fig.data:
            if str(getattr(tr, "name", "")) not in ("Peak", "Min"):
                continue
            got = _hex_of(_primary(tr))
            if got and got.lower() != curves[0].lower():
                bad.append(f"{name}: {tr.name!r} is {got} but its curve is {curves[0]}")
    assert not bad, "extremum markers detached from their curve:\n  " + "\n  ".join(bad)


def test_every_fill_derives_from_its_own_line(figures):
    """A fill in a different color from its own line is always a bug.

    The Overview hero drew an accent line over a stock-blue fill; that was fixed
    by routing the literal to a token, which changed the VALUE and left the
    MISMATCH — the Forecast chart still fills a model-colored line with the
    accent.
    """
    bad = []
    for name, fig in figures:
        for tr in fig.data:
            fill = getattr(tr, "fillcolor", None)
            line = _primary(tr)
            if not isinstance(fill, str) or line is None:
                continue
            fh, lh = _hex_of(fill), _hex_of(line)
            if fh is None or lh is None or fh == lh:
                continue
            bad.append(f"{name}: trace {tr.name!r} line={lh} but fill={fh}")
    assert not bad, "fill does not derive from its own line:\n  " + "\n  ".join(bad)


def test_close_pairs_have_a_second_channel(figures, capsys):
    """WCAG 1.4.1 on what actually renders: color is never the only channel.

    This measures every PAIR WITHIN each figure rather than a hand-written list
    of which colors 'share a chart'. That list is what was wrong.
    """
    violations, close_pairs = [], []
    for name, fig in figures:
        traces = [t for t in fig.data if _primary(t) and _hex_of(_primary(t))]
        for a, b in combinations(traces, 2):
            ca, cb = _hex_of(_primary(a)), _hex_of(_primary(b))
            d = 0.0 if ca == cb else min_cvd(ca, cb)
            if d >= CLOSE:
                continue
            close_pairs.append(f"{name}: {a.name!r} {ca} vs {b.name!r} {cb} = {d:.1f}")
            if _channel(a) == _channel(b):
                violations.append(
                    f"{name}: {a.name!r} ({ca}) and {b.name!r} ({cb}) are {d:.1f} apart "
                    f"and share every non-color channel {_channel(a)}"
                )
    with capsys.disabled():
        if close_pairs:
            print("\n  close pairs in real figures (color alone is not doing the work):")
            for p in close_pairs:
                print(f"    {p}")
    assert not violations, (
        "traces distinguished by color ALONE, below the floor:\n  " + "\n  ".join(violations)
    )
