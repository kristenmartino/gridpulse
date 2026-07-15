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

import pathlib
import sys
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from color_science import ciede2000, contrast_ratio, hex_to_oklch, lstar, simulate  # noqa: E402

from components import tokens  # noqa: E402

CVD = ("normal", "protan", "deutan", "tritan")

# The floor for two colors that share a figure. Calibrated to what this repo
# already tolerates rather than to a round number: the weakest co-occurring
# pair in the pre-existing Wong series (ARIMA green vs the EIA gray reference)
# measures 8.2, so 12 is comfortably stricter than the shipped baseline while
# staying achievable inside a palette Wong has already densely packed.
SHARED_FIGURE_FLOOR = 12.0

# WCAG 2.1: 4.5 normal text, 3.0 large text / graphics / UI components.
AA_TEXT = 4.5
AA_GRAPHIC = 3.0


def min_cvd(a: str, b: str) -> float:
    """Worst-case perceptual distance across normal + all three dichromacies."""
    return min(ciede2000(simulate(a, k), simulate(b, k)) for k in CVD)


def _hdr(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def check_contrast(failures: list[str]) -> None:
    _hdr("WCAG contrast on --bg-base")
    # --text-disabled is exempt: WCAG 1.4.3 does not apply to inactive UI.
    for name, floor in [
        ("TEXT_PRIMARY", AA_TEXT),
        ("TEXT_SECONDARY", AA_TEXT),
        ("TEXT_TERTIARY", AA_GRAPHIC),
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
        if name == "TEXT_TERTIARY" and ratio < AA_TEXT:
            note = f"  <- {ratio:.2f} clears graphics (3.0) but NOT normal-text AA (4.5)"
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
    _hdr("Weather driver sparklines — mutually distinct")
    items = list(tokens.WEATHER_DRIVERS.items())
    for (na, ca), (nb, cb) in combinations(items, 2):
        m = min_cvd(ca, cb)
        ok = m >= SHARED_FIGURE_FLOOR
        print(f"  {na:12s} vs {nb:12s} minCVD={m:5.1f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"weather driver {na}/{nb} minCVD {m:.1f} < {SHARED_FIGURE_FLOOR}")

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
