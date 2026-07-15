"""Off-the-shelf palettes, kept ONLY so the gate can measure distance from them.

Nothing imports this to paint anything. It exists so
``scripts/verify_palette.py`` can answer one question mechanically: *is the
brand accent a near-duplicate of a swatch someone could have downloaded?*

Why this file has to exist at all. An independent audit reverted ``ACCENT``
from the derived teal-cyan back to ``#35c6ff`` — the stock-sky near-duplicate
this entire redesign exists to leave — and every check passed green: both
gates, all 38 tests. Nothing measured ownership. That is precisely the failure
this repo already documented once ("nothing in CI failed on a raw hex, which is
why it rotted"), reproduced for the one property the Color dimension is scored
on.

Note the asymmetry, because it is the whole design of the check: distance from
stock is a BAD optimisation target (Tailwind's 242 swatches tile color space,
so chasing pure distance drives you to neon `#00fdfd`, the least owned color
there is) but a GOOD floor. We do not ask the accent to be far from everything.
We ask it not to be a copy. Every OTHER token is defended differently — by
reproducing from the accent through a stated rule, which is what "derived"
means and what a distance check cannot express.

Tailwind CSS v3, shades 200-600 — the plausible band for a bright accent on a
dark ground.
"""

from __future__ import annotations

_RAW: dict[str, list[str]] = {
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

TAILWIND: dict[str, str] = {
    f"{fam}-{shade}": hexv
    for fam, shades in _RAW.items()
    for shade, hexv in zip([200, 300, 400, 500, 600], shades, strict=True)
}
