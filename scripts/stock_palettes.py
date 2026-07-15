"""Off-the-shelf color, kept ONLY so the gate can measure distance from it.

Nothing imports this to paint anything. It exists so ``scripts/verify_palette.py``
can answer one question mechanically: *is a color here one you could have
downloaded?*

Why this file exists, and why it got bigger
-------------------------------------------
The first version held Tailwind alone. An audit then measured the shipped
accent at CIEDE2000 **1.64 from CSS ``darkturquoise``** — a color every browser
ships, no download required — while the gate reported 7.8 and passed it. The
retired accent had been condemned at 2.55. So the gate answered "is this a copy
of a Tailwind swatch", printed it as "is this a copy", and was wrong.

That is this repo's own documented lesson — *"a measurement is only true
against the inputs it was run on"* — recurring one level up, inside the very
check written to enforce it. Fixing the accent alone would have repeated the
mistake a third time; the corpus is the thing that was wrong.

Sources below are the palettes a web product plausibly reaches for. It is not
exhaustive and cannot be — the point is not to prove a color is unprecedented
(no color is), it is to make "I did not take this off a shelf" a claim with
teeth rather than a claim measured against one shelf.

On the asymmetry, because it is the whole design of the check: distance from
stock is a FLOOR, not an optimisation target. We do not ask the accent to be far
from everything — these corpora tile color space densely enough that "far from
all of it" selects for leftovers rather than for good colors. We ask only that
it not be a copy. Every other generated token is defended differently: by
reproducing from the accent through a stated rule, which is what "derived" means
and what a distance check cannot express.
"""

from __future__ import annotations

# Tailwind CSS v3, shades 200-600.
_TAILWIND: dict[str, list[str]] = {
    "slate": ["#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569"],
    "gray": ["#e5e7eb", "#d1d5db", "#9ca3af", "#6b7280", "#4b5563"],
    "zinc": ["#e4e4e7", "#d4d4d8", "#a1a1aa", "#71717a", "#52525b"],
    "neutral": ["#e5e5e5", "#d4d4d4", "#a3a3a3", "#737373", "#525252"],
    "stone": ["#e7e5e4", "#d6d3d1", "#a8a29e", "#78716c", "#57534e"],
    "red": ["#fecaca", "#fca5a5", "#f87171", "#ef4444", "#dc2626"],
    "orange": ["#fed7aa", "#fdba74", "#fb923c", "#f97316", "#ea580c"],
    "amber": ["#fde68a", "#fcd34d", "#fbbf24", "#f59e0b", "#d97706"],
    "yellow": ["#fef08a", "#fde047", "#facc15", "#eab308", "#ca8a04"],
    "lime": ["#d9f99d", "#bef264", "#a3e635", "#84cc16", "#65a30d"],
    "green": ["#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a"],
    "emerald": ["#a7f3d0", "#6ee7b7", "#34d399", "#10b981", "#059669"],
    "teal": ["#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488"],
    "cyan": ["#a5f3fc", "#67e8f9", "#22d3ee", "#06b6d4", "#0891b2"],
    "sky": ["#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7"],
    "blue": ["#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb"],
    "indigo": ["#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5"],
    "violet": ["#ddd6fe", "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed"],
    "purple": ["#e9d5ff", "#d8b4fe", "#c084fc", "#a855f7", "#9333ea"],
    "fuchsia": ["#f5d0fe", "#f0abfc", "#e879f9", "#d946ef", "#c026d3"],
    "pink": ["#fbcfe8", "#f9a8d4", "#f472b6", "#ec4899", "#db2777"],
    "rose": ["#fecdd3", "#fda4af", "#fb7185", "#f43f5e", "#e11d48"],
}

# CSS named colors (CSS Color Level 4). Every browser ships these; a color near
# one is not "invented" by any standard. This omission is what let a
# darkturquoise-adjacent accent past the first gate.
_CSS_NAMED: dict[str, str] = {
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "blanchedalmond": "#ffebcd",
    "blue": "#0000ff",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "cyan": "#00ffff",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgreen": "#006400",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "green": "#008000",
    "greenyellow": "#adff2f",
    "hotpink": "#ff69b4",
    "indianred": "#cd5c5c",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lawngreen": "#7cfc00",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navy": "#000080",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#ff0000",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}

# Material Design 2 — the 300/400/500 band.
_MATERIAL: dict[str, str] = {
    "red-400": "#ef5350",
    "red-500": "#f44336",
    "pink-400": "#ec407a",
    "purple-400": "#ab47bc",
    "deeppurple-400": "#7e57c2",
    "indigo-400": "#5c6bc0",
    "blue-300": "#64b5f6",
    "blue-400": "#42a5f5",
    "blue-500": "#2196f3",
    "lightblue-300": "#4fc3f7",
    "lightblue-400": "#29b6f6",
    "cyan-300": "#4dd0e1",
    "cyan-400": "#26c6da",
    "cyan-500": "#00bcd4",
    "teal-300": "#4db6ac",
    "teal-400": "#26a69a",
    "teal-500": "#009688",
    "green-300": "#81c784",
    "green-400": "#66bb6a",
    "green-500": "#4caf50",
    "lightgreen-400": "#9ccc65",
    "lime-400": "#d4e157",
    "yellow-400": "#ffee58",
    "amber-400": "#ffca28",
    "orange-400": "#ffa726",
    "deeporange-400": "#ff7043",
    "brown-400": "#8d6e63",
    "grey-400": "#bdbdbd",
    "bluegrey-400": "#78909c",
}

# IBM Carbon v11 — the 30/40/50 band.
_CARBON: dict[str, str] = {
    "blue-40": "#78a9ff",
    "blue-50": "#4589ff",
    "cyan-30": "#82cfff",
    "cyan-40": "#33b1ff",
    "cyan-50": "#1192e8",
    "teal-30": "#3ddbd9",
    "teal-40": "#08bdba",
    "teal-50": "#009d9a",
    "green-30": "#42be65",
    "green-40": "#24a148",
    "magenta-40": "#ff7eb6",
    "purple-40": "#be95ff",
    "red-40": "#ff8389",
    "red-50": "#fa4d56",
    "orange-40": "#ff832b",
    "yellow-30": "#f1c21b",
    "gray-30": "#c6c6c6",
    "gray-50": "#8d8d8d",
    "coolgray-40": "#a2a9b0",
    "warmgray-40": "#ada8a8",
}

# Ant Design v5 — the 4/5/6 band.
_ANT: dict[str, str] = {
    "blue-5": "#40a9ff",
    "blue-6": "#1890ff",
    "cyan-4": "#5cdbd3",
    "cyan-5": "#36cfc9",
    "cyan-6": "#13c2c2",
    "green-5": "#73d13d",
    "green-6": "#52c41a",
    "lime-5": "#bae637",
    "gold-5": "#ffc53d",
    "orange-5": "#ffa940",
    "volcano-5": "#ff7a45",
    "red-5": "#ff4d4f",
    "magenta-5": "#f759ab",
    "purple-5": "#9254de",
    "geekblue-5": "#597ef7",
}

# Chakra UI v2 — the 300/400 band.
_CHAKRA: dict[str, str] = {
    "teal-300": "#4fd1c5",
    "teal-400": "#38b2ac",
    "cyan-300": "#76e4f7",
    "cyan-400": "#0bc5ea",
    "blue-300": "#63b3ed",
    "blue-400": "#4299e1",
    "green-300": "#68d391",
    "green-400": "#48bb78",
    "red-300": "#fc8181",
    "red-400": "#f56565",
    "orange-300": "#f6ad55",
    "yellow-300": "#f6e05e",
    "purple-300": "#b794f4",
    "pink-300": "#f687b3",
    "gray-400": "#a0aec0",
}

# Bootstrap 5 theme colors.
_BOOTSTRAP: dict[str, str] = {
    "primary": "#0d6efd",
    "secondary": "#6c757d",
    "success": "#198754",
    "info": "#0dcaf0",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "teal": "#20c997",
    "cyan": "#0dcaf0",
    "indigo": "#6610f2",
    "purple": "#6f42c1",
    "pink": "#d63384",
    "orange": "#fd7e14",
}


def _build() -> dict[str, str]:
    out: dict[str, str] = {}
    for fam, shades in _TAILWIND.items():
        for shade, hexv in zip([200, 300, 400, 500, 600], shades, strict=True):
            out[f"tailwind {fam}-{shade}"] = hexv
    for name, hexv in _CSS_NAMED.items():
        out[f"css {name}"] = hexv
    for src, table in (
        ("material", _MATERIAL),
        ("carbon", _CARBON),
        ("ant", _ANT),
        ("chakra", _CHAKRA),
        ("bootstrap", _BOOTSTRAP),
    ):
        for name, hexv in table.items():
            out[f"{src} {name}"] = hexv
    return out


#: name -> hex, across every corpus above.
STOCK: dict[str, str] = _build()

# Back-compat alias. The old name said "Tailwind" when the question was never
# about Tailwind; kept so nothing breaks, but STOCK is the real name.
TAILWIND = STOCK
