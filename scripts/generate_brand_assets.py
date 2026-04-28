"""Generate GridPulse brand assets (favicon, apple-touch-icon, OG image).

Idempotent: rerun produces identical bytes from the constants below. Run
once after editing brand constants. Outputs land in ``assets/``.

The monogram is an ECG-style pulse line (flat → spike-up → spike-down →
overshoot → recover) on an obsidian rounded square. Path data here matches
``assets/favicon.svg`` exactly so the SVG and the PNG/ICO renders are visually
identical. The icon library (``components/icons.py``, G6) reuses the same
path under the name ``pulse-mono``.

Usage:
    python scripts/generate_brand_assets.py [--all|--favicon|--og]

G1 ships favicon + apple-touch-icon; G2 extends with OG image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Brand constants (must match assets/custom.css + brand spec) ──────
OBSIDIAN = (10, 13, 20)  # #0a0d14
MIDNIGHT = (17, 20, 28)  # #11141c
PULSE_CYAN = (56, 208, 255)  # #38D0FF
TEXT_PRIMARY = (247, 250, 252)  # #F7FAFC
TEXT_SECONDARY = (168, 179, 199)  # #A8B3C7

# ── Pulse path in unit coordinates [0..1] × [0..1] ───────────────────
# Matches the SVG path in assets/favicon.svg exactly.
# Reading coords as (x, y) where y=0 is top, y=1 is bottom.
PULSE_PATH_UNIT: list[tuple[float, float]] = [
    (0.125, 0.500),  # baseline left
    (0.344, 0.500),  # flat run
    (0.438, 0.250),  # sharp up-spike
    (0.500, 0.750),  # sharp down-spike
    (0.562, 0.375),  # up-overshoot
    (0.656, 0.500),  # recover
    (0.875, 0.500),  # baseline right
]

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"


def _draw_pulse(draw: ImageDraw.ImageDraw, size: int, stroke: int, color) -> None:
    """Render the pulse polyline at the given pixel size."""
    pts = [(int(round(x * size)), int(round(y * size))) for x, y in PULSE_PATH_UNIT]
    draw.line(pts, fill=color, width=stroke, joint="curve")
    # Round line caps: paint a circle at each endpoint
    cap_r = max(1, stroke // 2)
    for px, py in (pts[0], pts[-1]):
        draw.ellipse((px - cap_r, py - cap_r, px + cap_r, py + cap_r), fill=color)


def _rounded_bg(size: int, radius: int, color) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=color)
    return img


def _render_icon(size: int, oversample: int = 4) -> Image.Image:
    """Render a single-size icon (rounded obsidian + cyan pulse) at ``size``×``size``."""
    ss = size * oversample
    img = _rounded_bg(ss, radius=int(ss * 6 / 32), color=OBSIDIAN)
    d = ImageDraw.Draw(img)
    stroke = max(2, int(round(ss * 2.25 / 32)))
    _draw_pulse(d, ss, stroke=stroke, color=PULSE_CYAN)
    return img.resize((size, size), Image.LANCZOS)


def make_favicon_svg(out_path: Path) -> None:
    """Write the SVG monogram. Hardcoded path matches PULSE_PATH_UNIT × 32."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none">\n'
        '  <rect width="32" height="32" rx="6" fill="#0a0d14"/>\n'
        '  <path d="M4 16 L11 16 L14 8 L16 24 L18 12 L21 16 L28 16"\n'
        '        stroke="#38D0FF" stroke-width="2.25"\n'
        '        stroke-linecap="round" stroke-linejoin="round" fill="none"/>\n'
        "</svg>\n"
    )
    out_path.write_text(svg, encoding="utf-8")


def make_favicon_ico(out_path: Path) -> None:
    """Multi-resolution ICO (16/32/48/64) for legacy browser support."""
    sizes = [16, 32, 48, 64]
    images = [_render_icon(s) for s in sizes]
    images[0].save(out_path, format="ICO", sizes=[(s, s) for s in sizes])


def make_apple_touch_icon(out_path: Path) -> None:
    """180×180 rounded-square pulse glyph for iOS home screen."""
    size = 180
    ss = size * 4
    img = _rounded_bg(ss, radius=int(ss * 24 / 180), color=OBSIDIAN)
    d = ImageDraw.Draw(img)
    _draw_pulse(d, ss, stroke=int(ss * 5 / 180), color=PULSE_CYAN)
    img.resize((size, size), Image.LANCZOS).save(out_path, format="PNG", optimize=True)


def _load_font(candidates: list[str], size: int) -> ImageFont.ImageFont:
    """First-of candidates that loads, else Pillow's bundled default."""
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_og_image(out_path: Path) -> None:
    """1200×630 OG image: dark gradient + grid lines + monogram + Grid|Pulse + tagline."""
    width, height = 1200, 630
    img = Image.new("RGB", (width, height), OBSIDIAN)

    # Soft gradient: midnight bias toward bottom-right
    gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(gradient)
    for y in range(height):
        # 0 at top, max alpha at bottom
        alpha = int(60 * (y / height))
        g_draw.line([(0, y), (width, y)], fill=(*MIDNIGHT, alpha))
    img.paste(gradient, (0, 0), gradient)

    # Faint horizontal grid lines (4% white)
    grid = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid)
    for y in range(0, height, 90):
        grid_draw.line([(0, y), (width, y)], fill=(255, 255, 255, 10), width=1)
    img.paste(grid, (0, 0), grid)

    # Monogram (left-aligned, 160×160)
    mono_size = 160
    mono = _render_icon(mono_size, oversample=4).convert("RGBA")
    img.paste(mono, (96, (height - mono_size) // 2 - 32), mono)

    # Wordmark "Grid|Pulse" + tagline
    draw = ImageDraw.Draw(img)
    display_candidates = [
        # Prefer brand fonts if pre-staged
        str(ASSETS / "fonts" / "Sora-Bold.ttf"),
        str(ASSETS / "fonts" / "Inter-Bold.ttf"),
        # macOS fallbacks
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    body_candidates = [
        str(ASSETS / "fonts" / "Inter-Medium.ttf"),
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_display = _load_font(display_candidates, 108)
    font_body = _load_font(body_candidates, 32)

    text_x = 96 + mono_size + 48
    grid_y = (height // 2) - 80
    draw.text((text_x, grid_y), "Grid", font=font_display, fill=TEXT_PRIMARY)
    grid_w = draw.textlength("Grid", font=font_display)
    draw.text((text_x + grid_w, grid_y), "Pulse", font=font_display, fill=PULSE_CYAN)

    # Tagline beneath
    draw.text(
        (text_x, grid_y + 140),
        "Energy Intelligence Platform",
        font=font_body,
        fill=TEXT_SECONDARY,
    )

    # Bottom-right small caption
    cap_font = _load_font(body_candidates, 22)
    caption = "See demand sooner. Decide with confidence."
    cap_w = draw.textlength(caption, font=cap_font)
    draw.text(
        (width - cap_w - 64, height - 64),
        caption,
        font=cap_font,
        fill=(168, 179, 199, 200),
    )

    img.save(out_path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["all", "favicon", "og"],
        default="favicon",
        help="Which asset(s) to generate. 'favicon' = G1 set; 'og' = G2 set; 'all' = both.",
    )
    args = parser.parse_args()

    ASSETS.mkdir(exist_ok=True)
    written: list[Path] = []

    if args.target in ("favicon", "all"):
        svg_path = ASSETS / "favicon.svg"
        ico_path = ASSETS / "favicon.ico"
        touch_path = ASSETS / "apple-touch-icon.png"
        make_favicon_svg(svg_path)
        make_favicon_ico(ico_path)
        make_apple_touch_icon(touch_path)
        written.extend([svg_path, ico_path, touch_path])

    if args.target in ("og", "all"):
        og_path = ASSETS / "og-image.png"
        make_og_image(og_path)
        written.append(og_path)

    for path in written:
        size = path.stat().st_size
        print(f"  wrote {path.relative_to(REPO_ROOT)} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
