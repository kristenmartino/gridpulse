"""Pure-python color science for verifying the GridPulse palette.

Used by ``scripts/verify_palette.py`` (the CLI report) and by
``tests/unit/test_color_tokens.py`` (the CI invariants), so a palette claim is
measured in exactly one place.

Deliberately dependency-free: it must run in CI without numpy/plotly, and a
color check that is expensive to run is a color check that stops being run.

Contents:
  * sRGB <-> linear light, CIE XYZ/Lab (D65), OKLab/OKLCh
  * WCAG 2.1 relative luminance + contrast ratio
  * Dichromacy simulation (Machado, Oliveira & Fernandes 2009, severity 1.0)
  * CIEDE2000 perceptual difference

Why CIEDE2000 and not a naive RGB distance: the question this module answers is
"can a human tell these two chart series apart", which is a perceptual question.
Anchors, measured with this module against the palette actually shipped:
  ~1   invisible   (nuclear vs hydro under deuteranopia, before this pass)
  ~2.5 near-duplicate (the accent vs Tailwind sky-400)
  ~8   the weakest pair the existing Wong series already tolerates
  >=12 the floor this repo enforces for series that share a figure
"""

from __future__ import annotations

import math

# ── sRGB <-> linear ───────────────────────────────────────────────────


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def to_linear(h: str) -> tuple[float, float, float]:
    return tuple(srgb_to_linear(c) for c in hex_to_rgb(h))  # type: ignore


# ── WCAG ──────────────────────────────────────────────────────────────


def relative_luminance(h: str) -> float:
    r, g, b = to_linear(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(h1: str, h2: str) -> float:
    l1, l2 = relative_luminance(h1), relative_luminance(h2)
    lo, hi = sorted((l1, l2))
    return (hi + 0.05) / (lo + 0.05)


# ── CIE Lab (D65) ─────────────────────────────────────────────────────

_M_RGB_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_WP = (0.95047, 1.00000, 1.08883)


def _mv(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def hex_to_xyz(h: str):
    return _mv(_M_RGB_XYZ, to_linear(h))


def _f(t: float) -> float:
    return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29


def hex_to_lab(h: str) -> tuple[float, float, float]:
    x, y, z = hex_to_xyz(h)
    fx, fy, fz = _f(x / _WP[0]), _f(y / _WP[1]), _f(z / _WP[2])
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def lstar(h: str) -> float:
    return hex_to_lab(h)[0]


# ── OKLab / OKLCh ─────────────────────────────────────────────────────


def hex_to_oklab(h: str) -> tuple[float, float, float]:
    r, g, b = to_linear(h)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (
        l ** (1 / 3) if l > 0 else 0,
        m ** (1 / 3) if m > 0 else 0,
        s ** (1 / 3) if s > 0 else 0,
    )
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def hex_to_oklch(h: str) -> tuple[float, float, float]:
    L, a, b = hex_to_oklab(h)
    C = math.hypot(a, b)
    H = math.degrees(math.atan2(b, a)) % 360
    return (L, C, H)


def oklch_to_hex(L: float, C: float, H: float) -> str:
    a = C * math.cos(math.radians(H))
    b = C * math.sin(math.radians(H))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return rgb_to_hex((linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(bb)))


def in_gamut(L: float, C: float, H: float, tol: float = 0.002) -> bool:
    """True if the OKLCh triple round-trips through sRGB without clipping."""
    a = C * math.cos(math.radians(H))
    b = C * math.sin(math.radians(H))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return all(-tol <= c <= 1 + tol for c in (r, g, bb))


# ── CVD simulation (Machado, Oliveira & Fernandes 2009, severity 1.0) ──

_CVD = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def simulate(h: str, kind: str) -> str:
    """Simulate dichromacy. Matrix applies to LINEAR rgb."""
    if kind == "normal":
        return h.lower()
    lin = to_linear(h)
    out = _mv(_CVD[kind], lin)
    return rgb_to_hex(tuple(linear_to_srgb(c) for c in out))


# ── CIEDE2000 ─────────────────────────────────────────────────────────


def ciede2000(h1: str, h2: str) -> float:
    L1, a1, b1 = hex_to_lab(h1)
    L2, a2, b2 = hex_to_lab(h2)
    kL = kC = kH = 1.0
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb**7 / (Cb**7 + 25**7))) if Cb > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (
        1
        - 0.17 * math.cos(math.radians(hbp - 30))
        + 0.24 * math.cos(math.radians(2 * hbp))
        + 0.32 * math.cos(math.radians(3 * hbp + 6))
        - 0.20 * math.cos(math.radians(4 * hbp - 63))
    )
    dTh = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25**7)) if Cbp > 0 else 0
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dTh)) * Rc
    return math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def report(h: str, bg: str = "#0a0a0b") -> str:
    L, C, H = hex_to_oklch(h)
    return (
        f"{h.lower()}  OKLCh({L:.3f}, {C:.3f}, {H:5.1f}°)  "
        f"L*={lstar(h):5.1f}  contrast/{bg}={contrast_ratio(h, bg):.2f}"
    )
