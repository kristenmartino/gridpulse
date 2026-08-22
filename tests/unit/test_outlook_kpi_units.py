"""The Forecast tab's KPI values must not carry their own unit.

The layout renders a dedicated ``gp-metric-unit`` span beside each of the
four outlook KPIs (``components/tab_demand_outlook.py``). When the callback
*also* embedded "MW" in the value string, every card printed the unit twice
— "75,581 MW MW" — and on a narrow viewport the hero cell truncated the
doubled string to "89,952 ..." so the number itself was cut off.

This pins both halves of the contract, because either half alone is
satisfiable by the bug: the layout still supplies exactly one unit span per
KPI, and neither callback path emits a unit inside the value.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAYOUT = REPO / "components" / "tab_demand_outlook.py"
CALLBACKS = REPO / "components" / "_callbacks_forecast.py"

#: The four KPI cells that pair a value span with a separate unit span.
KPI_IDS = ("outlook-peak", "outlook-avg", "outlook-min", "outlook-range")

#: Value expressions the callbacks return for those cells.
KPI_VALUE_EXPRS = ("peak_val", "avg_val", "min_val", "range_val")


def test_layout_supplies_exactly_one_unit_span_per_kpi():
    """If the layout stops rendering the unit, stripping it from the value
    would leave the number unlabelled — so the fix depends on this."""
    src = LAYOUT.read_text(encoding="utf-8")
    for kpi in KPI_IDS:
        idx = src.find(f'id="{kpi}"')
        assert idx != -1, f"{kpi}: value span not found in layout"
        cell = src[idx : idx + 400]
        units = cell.count('className="gp-metric-unit"')
        assert units == 1, (
            f"{kpi}: expected exactly 1 gp-metric-unit span beside the value, found {units}. "
            "The callback omits the unit on the assumption the layout renders it."
        )


def test_no_callback_path_embeds_a_unit_in_a_kpi_value():
    """Catches the regression on BOTH return paths (Redis-served and the
    inline-compute branch), not just whichever one a fixture happens to hit."""
    src = CALLBACKS.read_text(encoding="utf-8")
    offenders = []
    for expr in KPI_VALUE_EXPRS:
        # f-string formats of the KPI value that also contain a unit token.
        for m in re.finditer(rf'f"\{{{expr}:[^"]*\}}[^"]*"', src):
            literal = m.group(0)
            if re.search(r"\bMW\b", literal):
                line = src[: m.start()].count("\n") + 1
                offenders.append(f"{CALLBACKS.name}:{line}: {literal}")
    assert not offenders, (
        "KPI value strings must not embed a unit — the layout renders a "
        "gp-metric-unit span beside them, so this prints 'MW MW':\n  " + "\n  ".join(offenders)
    )


def test_the_guard_can_actually_see_the_defect():
    """A guard needs a fixture that triggers what it exists to catch.

    Both checks above are greps, so an expression-name drift would make them
    vacuously pass. Assert the pattern matches a known-bad literal.
    """
    bad = 'f"{peak_val:,.0f} MW"'
    assert re.search(r'f"\{peak_val:[^"]*\}[^"]*"', bad)
    assert re.search(r"\bMW\b", bad)


def test_chart_annotations_keep_their_unit():
    """The in-chart Peak/Min labels have no separate unit span, so they must
    still say MW — the fix must not have stripped those too."""
    src = CALLBACKS.read_text(encoding="utf-8")
    annotated = re.findall(r'text=\[f"(?:Peak|Min): \{[a-z_]+:,\.0f\} MW"\]', src)
    assert len(annotated) >= 2, (
        f"expected the chart's Peak/Min annotations to still carry 'MW'; found {len(annotated)}"
    )


def test_callbacks_module_still_parses():
    ast.parse(CALLBACKS.read_text(encoding="utf-8"))
