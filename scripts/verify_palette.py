#!/usr/bin/env python3
"""Measure every claim components/tokens.py makes about its palette.

The token module's docstrings assert things — "CVD-safe", "luminance-
monotonic", "clears every series it shares a figure with". Those are the same
kind of claim that was already in this repo and already false: a comment
congratulated the map scale for escaping the deutan trap while the fuel-mix
stacked area two files away painted nuclear and hydro at CIEDE2000 1.0 under
deuteranopia — adjacent bands, indistinguishable, in the palette that actually
shipped. A claim nothing measures is decoration.

This script measures them. ``tests/unit/test_color_tokens.py`` runs the same
invariants in CI so a regression fails the build; this CLI additionally prints
the numbers, for when you are designing rather than defending.

    python scripts/verify_palette.py

Exit code is 0 if every invariant holds, 1 otherwise.
"""

from __future__ import annotations

import math
import pathlib
import sys
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from color_science import (  # noqa: E402
    ciede2000,
    contrast_ratio,
    hex_to_oklch,
    in_gamut,
    lstar,
    oklch_to_hex,
    simulate,
)
from stock_palettes import TAILWIND  # noqa: E402

from components import tokens  # noqa: E402

CVD = ("normal", "protan", "deutan", "tritan")

# The floor for two colors that share a figure. Calibrated to what this repo
# already tolerates rather than to a round number: the weakest co-occurring
# pair in the pre-existing Wong series (ARIMA green vs the EIA gray reference)
# measures 8.2, so 12 is comfortably stricter than the shipped baseline while
# staying achievable inside a palette Wong has already densely packed.
SHARED_FIGURE_FLOOR = 12.0

# A SEPARATE floor for colors that are merely side by side rather than overlaid.
#
# The Overview's three driver sparklines are three independent single-series
# figures (``_driver_sparkline`` builds its own ``go.Figure()`` per driver),
# each in its own labeled cell. Nothing overlaps, and the label — not the hue —
# carries identity. This check originally applied SHARED_FIGURE_FLOOR to them,
# which contradicts that constant's own stated meaning ("two colors that share
# a figure"); they never share one.
#
# Disclosure, since it matters when reading a threshold: the mis-specification
# was noticed because deriving the semantic ramp made the drivers fail 12.0.
# The reason for a lower floor is independent of that (it is a fact about the
# figures, checkable above), but it was not what prompted the look. 10.0 is
# "clearly a different color at a glance"; the pair this governs measures 10.8.
ADJACENT_CHART_FLOOR = 10.0

# WCAG 2.1: 4.5 normal text, 3.0 large text / graphics / UI components.
AA_TEXT = 4.5
AA_GRAPHIC = 3.0

# How far the accent must sit from the nearest OFF-THE-SHELF swatch.
#
# A FLOOR, not a target. With a real corpus the metric nearly saturates: the
# greatest distance from all stock, anywhere in usable accent space (contrast
# >= 4.5 on the ground, chroma >= 0.10), measures 10.8. So a high floor stops
# being a floor and becomes an optimisation target — and optimising distance
# drives you to neon #00fdfd, the least owned color there is.
#
# 4.0 asks only the question worth asking: is this a COPY? CIEDE2000 ~2.3 is
# the just-noticeable difference under careful viewing, so 4.0 is 1.7x JND —
# "clearly not the same color as anything published", with margin.
#
# This number went DOWN from 6.0 and the gate got STRICTER, which is worth
# spelling out because it looks like the opposite. 6.0 was set against a
# Tailwind-only corpus where it was cheap; the accent it passed at 7.8 sat 1.64
# from CSS darkturquoise, below JND. Under the corpus below that same accent
# scores 1.64 and FAILS. The corpus was the thing that was wrong, not the number.
STOCK_FLOOR = 4.0

# Tolerance when re-deriving a token from its rule. Non-zero only because the
# rule runs in OKLCh and the token is stored as 8-bit sRGB, so a round-trip
# quantises.
RAMP_TOL = 1.0

# ACCENT_SOFT is the accent lifted for hover. One number, stated, so it can be
# re-derived rather than trusted.
ACCENT_SOFT_DL = 0.07

# The semantics must share the anchor's OKLab lightness. Tolerance covers 8-bit
# sRGB quantisation only.
SEMANTIC_L_TOL = 0.005

# The ramp's stated curve and lightness architecture. These are the RULE; the
# literals in components/tokens.py are its OUTPUT, and this file exists to prove
# they still agree.
RAMP_CMAX, RAMP_PEAK, RAMP_WIDTH = 0.019, 0.38, 0.34
NEUTRAL_LIGHTNESS = {
    "SURFACE_SUNKEN": 0.123,
    "BG_BASE": 0.145,
    "BG_RAISED": 0.179,
    "BG_HOVER": 0.210,
    "MAP_SUBUNIT": 0.241,
    "MAP_COASTLINE": 0.274,
    "TEXT_DISABLED": 0.442,
    "TEXT_SECONDARY": 0.712,
    "TEXT_PRIMARY": 0.920,
}
# TEXT_TERTIARY is excluded above on purpose: its lightness is not a chosen
# constant but the SOLUTION to "clear WCAG AA on --bg-base", so it is checked by
# contrast rather than by reproduction. See check_contrast.

# The semantic rule. Hue is convention; chroma is the anchor's (or the gamut max
# at that hue); LIGHTNESS IS SOLVED for dichromatic separation and clamped to a
# band where the color still reads as its meaning. See components/tokens.py for
# why constant lightness was an accessibility bug rather than a design position.
SEMANTIC_SPEC = {
    "SUCCESS": (0.86, 150),
    "WARNING": (0.74, 95),
    "DANGER": (0.64, 25),
    "INFO": (0.70, 250),
    "FORECAST": (0.80, 60),
}
SEMANTIC_HUES = {k: v[1] for k, v in SEMANTIC_SPEC.items()}

# Pairs that a reader actually compares, read off the callsites. The severity
# triad is ONE badge that takes one of three colors by threshold, and three such
# badges sit side by side in the weather strip — so they are compared. FORECAST
# and the ACCENT are two lines in the hero figure.
#
# FORECAST vs WARNING is deliberately absent: they share the hero figure, but as
# a dashed LINE and a 10px TEXT annotation. Different mark types are already
# distinguished by form; the floor is for colors encoding the same kind of mark.
SEMANTIC_COMPARED = [
    ("SUCCESS", "WARNING"),
    ("SUCCESS", "DANGER"),
    ("WARNING", "DANGER"),
]


def _chroma(L: float) -> float:
    return RAMP_CMAX * math.exp(-(((L - RAMP_PEAK) / RAMP_WIDTH) ** 2))


def _fit(L: float, C: float, H: float) -> str:
    c = C
    while c > 0 and not in_gamut(L, c, H):
        c -= 0.0005
    return oklch_to_hex(L, c, H)


def _nearest_stock(color: str) -> tuple[str, float]:
    name, hexv = min(TAILWIND.items(), key=lambda kv: ciede2000(color, kv[1]))
    return name, ciede2000(color, hexv)


def min_cvd(a: str, b: str) -> float:
    """Worst-case perceptual distance across normal + all three dichromacies."""
    return min(ciede2000(simulate(a, k), simulate(b, k)) for k in CVD)


def _hdr(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def check_ownership(failures: list[str]) -> None:
    """The palette must be THIS product's, and provably so.

    Three separate claims, checked three different ways, because "owned" is not
    one property:

      1. The ACCENT — the single free choice a human made — must not be a
         near-duplicate of a stock swatch.
      2. The NEUTRALS must reproduce from the stated chroma curve at the
         accent's hue. That is what "derived" means; if they stop reproducing,
         they are just literals again.
      3. The SEMANTICS must sit on the accent's lightness at their stated hues.

    Before this existed, reverting ACCENT to the stock near-duplicate the
    redesign was built to escape passed every gate and all 38 tests. The
    property the whole dimension is scored on was the one property nothing
    measured.
    """
    _hdr("Ownership — the accent is a choice, everything else reproduces from it")

    # 1. The accent is not a copy.
    near, d = _nearest_stock(tokens.ACCENT)
    ok = d >= STOCK_FLOOR
    print(
        f"  accent {tokens.ACCENT} vs nearest stock swatch ({near})  dE={d:5.1f}  "
        f"(floor {STOCK_FLOOR}) {'ok' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(
            f"ACCENT {tokens.ACCENT} is dE {d:.1f} from Tailwind {near} — a near-duplicate. "
            f"The accent is the one color a human chooses; it may not be a stock swatch."
        )

    # 2. The neutrals reproduce from the curve.
    L_a, C_a, H_a = hex_to_oklch(tokens.ACCENT)
    worst = 0.0
    for name, L in NEUTRAL_LIGHTNESS.items():
        regen = _fit(L, _chroma(L), H_a)
        d = ciede2000(getattr(tokens, name), regen)
        worst = max(worst, d)
        if d > RAMP_TOL:
            failures.append(
                f"{name} = {getattr(tokens, name)} does not reproduce from the ramp curve "
                f"at the accent's hue (regenerates to {regen}, dE {d:.1f}) — the neutral "
                f"ramp has stopped being derived."
            )
    print(
        f"  neutral ramp reproduces from chroma curve at hue {H_a:.0f}°  "
        f"worst dE={worst:4.1f}  (tol {RAMP_TOL}) {'ok' if worst <= RAMP_TOL else 'FAIL'}"
    )

    # 2b. ACCENT_SOFT reproduces from the accent. It had no stated rule at all —
    # a second brand color, ungated, sitting dE 5.35 from stock cyan-300, i.e.
    # under the floor the accent itself must clear. Swapping it to stock passed
    # every check.
    soft = _fit(L_a + ACCENT_SOFT_DL, C_a, H_a)
    d = ciede2000(tokens.ACCENT_SOFT, soft)
    ok = d <= RAMP_TOL
    print(
        f"  ACCENT_SOFT reproduces from the accent (+{ACCENT_SOFT_DL} L)  "
        f"dE={d:4.1f}  {'ok' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(
            f"ACCENT_SOFT {tokens.ACCENT_SOFT} does not reproduce from the accent "
            f"(regenerates to {soft}, dE {d:.1f}) — the hover brand color has come "
            f"loose from the brand."
        )

    # 3. The semantics sit on the accent's lightness at their stated hues.
    worst_s = 0.0
    for name, (sem_l, hue) in SEMANTIC_SPEC.items():
        regen = _fit(sem_l, C_a, hue)
        d = ciede2000(getattr(tokens, name), regen)
        worst_s = max(worst_s, d)
        if d > RAMP_TOL:
            failures.append(
                f"{name} = {getattr(tokens, name)} does not reproduce from the semantic rule "
                f"(solved lightness {sem_l:.2f}, hue {hue} → {regen}, dE {d:.1f})."
            )
    print(
        f"  semantic ramp reproduces at the anchor's lightness (L={L_a:.3f})  "
        f"worst dE={worst_s:4.1f}  (tol {RAMP_TOL}) {'ok' if worst_s <= RAMP_TOL else 'FAIL'}"
    )

    # The check that did NOT exist while the constant-lightness rule was
    # shipping, which is why the rule's accessibility cost went unmeasured:
    # the severity triad against ITSELF, under dichromacy.
    _hdr("Severity triad — the pairs a reader compares, under dichromacy")
    for a, b in SEMANTIC_COMPARED:
        m = min_cvd(getattr(tokens, a), getattr(tokens, b))
        ok = m >= SHARED_FIGURE_FLOOR
        print(
            f"  {a:8s} vs {b:8s} minCVD={m:5.1f}  (floor {SHARED_FIGURE_FLOOR}) "
            f"{'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(
                f"severity {a}/{b} minCVD {m:.1f} < {SHARED_FIGURE_FLOOR} — these three "
                f"collapse toward one yellow under deuteranopia and lightness is the only "
                f"channel left; a constant-lightness rule measured 1.5 here."
            )
    contrasts = [contrast_ratio(getattr(tokens, n), tokens.BG_BASE) for n in SEMANTIC_SPEC]
    print(
        f"    contrast band {min(contrasts):.2f}-{max(contrasts):.2f} "
        f"(every semantic still clears AA; see check_contrast)"
    )


def check_contrast(failures: list[str]) -> None:
    _hdr("WCAG contrast on --bg-base")
    # --text-disabled is exempt: WCAG 1.4.3 does not apply to inactive UI.
    for name, floor in [
        ("TEXT_PRIMARY", AA_TEXT),
        ("TEXT_SECONDARY", AA_TEXT),
        ("TEXT_TERTIARY", AA_TEXT),
        ("ACCENT", AA_TEXT),
        ("SUCCESS", AA_TEXT),
        ("WARNING", AA_TEXT),
        ("DANGER", AA_TEXT),
        ("INFO", AA_TEXT),
        ("FORECAST", AA_TEXT),
    ]:
        value = getattr(tokens, name)
        ratio = contrast_ratio(value, tokens.BG_BASE)
        ok = ratio >= floor
        note = ""
        if name == "TEXT_TERTIARY":
            # Its lightness is SOLVED for this number rather than sampled — so
            # the floor is the real one. It was previously pinned at
            # AA_GRAPHIC (3.0), which let stock zinc-500's 4.08 sail through:
            # the one token advertised as "solved for AA" was the one token the
            # gate did not hold to AA.
            note = "  <- lightness is solved for this, not sampled"
        print(f"  {name:16s} {value}  {ratio:5.2f}  (floor {floor}) {'ok' if ok else 'FAIL'}{note}")
        if not ok:
            failures.append(f"{name} contrast {ratio:.2f} < {floor}")


def check_map_scale(failures: list[str]) -> None:
    _hdr("US-Grid stress colorscale — luminance monotonicity (the preserve-guardrail)")
    prev = None
    for stop, color in tokens.MAP_COLORSCALE:
        lum = lstar(color)
        arrow = "" if prev is None else ("  rising" if prev < lum else "  NOT RISING")
        print(f"  {stop:.2f}  {color}  L*={lum:5.1f}{arrow}")
        if prev is not None and prev >= lum:
            failures.append(f"MAP_COLORSCALE not luminance-monotonic at stop {stop}")
        prev = lum


def check_fuel(failures: list[str]) -> None:
    _hdr("Fuel stack — adjacent bands must separate under every CVD type")
    order = tokens.FUEL_STACK_ORDER
    for a, b in zip(order, order[1:], strict=False):  # pairwise: lengths differ by 1
        ha, hb = tokens.FUEL_COLORS[a], tokens.FUEL_COLORS[b]
        m = min_cvd(ha, hb)
        ok = m >= SHARED_FIGURE_FLOOR
        print(
            f"  {a:8s} | {b:8s}  minCVD={m:5.1f}  dL*={abs(lstar(ha) - lstar(hb)):5.1f}  "
            f"{'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"fuel {a}|{b} minCVD {m:.1f} < {SHARED_FIGURE_FLOOR}")

    _hdr("Fuel bands vs the ACCENT (net-load line is drawn OVER the stack)")
    for f in order:
        m = min_cvd(tokens.FUEL_COLORS[f], tokens.ACCENT)
        ok = m >= SHARED_FIGURE_FLOOR
        print(f"  {f:8s} vs accent  minCVD={m:5.1f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"fuel {f} vs accent minCVD {m:.1f} < {SHARED_FIGURE_FLOOR}")


def check_accent_series(failures: list[str]) -> None:
    _hdr("ACCENT (demand/primary) vs every series it shares a figure with")
    # Wong's sky_blue is absent on purpose: it is the XGBoost identity color and
    # is only ever drawn in its own single-trace residual figure, so it never
    # shares a chart with the demand line. If you add a figure that draws both,
    # this list is what you must update — and the numbers will tell you the
    # accent and sky_blue are ~4 apart, which is why they are kept separate.
    rivals = {
        "prophet(orange)": tokens.CB_PALETTE["orange"],
        "arima(green)": tokens.CB_PALETTE["green"],
        "ensemble(vermillion)": tokens.CB_PALETTE["vermillion"],
        "temperature(yellow)": tokens.CB_PALETTE["yellow"],
        "wong_blue(colorway)": tokens.CB_PALETTE["blue"],
        "eia(gray)": tokens.NEUTRAL_SERIES,
        "forecast(orange)": tokens.FORECAST,
    }
    for name, color in rivals.items():
        m = min_cvd(tokens.ACCENT, color)
        ok = m >= SHARED_FIGURE_FLOOR
        print(f"  accent vs {name:22s} minCVD={m:5.1f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"accent vs {name} minCVD {m:.1f} < {SHARED_FIGURE_FLOOR}")


def check_weather_drivers(failures: list[str]) -> None:
    _hdr("Weather driver sparklines — separate labeled figures, merely adjacent")
    items = list(tokens.WEATHER_DRIVERS.items())
    for (na, ca), (nb, cb) in combinations(items, 2):
        m = min_cvd(ca, cb)
        ok = m >= ADJACENT_CHART_FLOOR
        print(
            f"  {na:12s} vs {nb:12s} minCVD={m:5.1f}  (floor {ADJACENT_CHART_FLOOR}) "
            f"{'ok' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(f"weather driver {na}/{nb} minCVD {m:.1f} < {ADJACENT_CHART_FLOOR}")

    # The audit's explicit ask: hydro and temperature must not share a hue.
    t = tokens.WEATHER_DRIVERS["temperature"]
    h = tokens.FUEL_COLORS["hydro"]
    d = ciede2000(t, h)
    print(
        f"  temperature vs hydro (must be distinct hues)  dE={d:5.1f}  "
        f"{'ok' if d >= SHARED_FIGURE_FLOOR else 'FAIL'}"
    )
    if d < SHARED_FIGURE_FLOOR:
        failures.append(f"temperature and hydro too close: {d:.1f}")


def check_wong_intact(failures: list[str]) -> None:
    _hdr("Wong CB_PALETTE integrity (external standard — must not be retuned)")
    expected = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "sky_blue": "#56B4E9",
        "yellow": "#F0E442",
        "purple": "#CC79A7",
        "black": "#000000",
    }
    for k, v in expected.items():
        got = tokens.CB_PALETTE.get(k)
        ok = got == v
        print(f"  {k:11s} {got}  {'ok' if ok else f'FAIL (expected {v})'}")
        if not ok:
            failures.append(f"CB_PALETTE[{k}] = {got}, expected Wong {v}")


def main() -> int:
    failures: list[str] = []
    print("GridPulse palette verification")
    print(
        f"accent {tokens.ACCENT}  OKLCh"
        f"{tuple(round(x, 3) for x in hex_to_oklch(tokens.ACCENT))}  "
        f"L*={lstar(tokens.ACCENT):.1f}"
    )

    check_ownership(failures)
    check_contrast(failures)
    check_wong_intact(failures)
    check_map_scale(failures)
    check_fuel(failures)
    check_accent_series(failures)
    check_weather_drivers(failures)

    print()
    if failures:
        print(f"FAILED — {len(failures)} violation(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("PASS — every palette invariant holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
