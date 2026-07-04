"""US Grid region cards self-describe via inline badge labels.

Each card shows two GW values (hero demand + the ``net`` interchange badge) and
two % values (``util`` + ``Δ1h``). Without labels they're indistinguishable, so
a tiny muted tag names each one and the separate legend was dropped. These tests
pin that the labels render (and that the import-dominated ``imports`` chip is not
double-labeled).
"""

from __future__ import annotations

from components._callbacks_us_grid import _build_us_grid_region_card


def _texts(node) -> list[str]:
    out: list[str] = []

    def walk(n):
        c = getattr(n, "children", None)
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, (list, tuple)):
            for x in c:
                walk(x)
        elif c is not None:
            walk(c)

    walk(node)
    return out


class TestCardBadgeLabels:
    def test_all_four_labels_present(self):
        card = _build_us_grid_region_card(
            "PJM",
            {
                "current_mw": 90000.0,
                "prev_mw": 88000.0,
                "today_mw": [90000.0] * 24,
                "interchange": {"net_mw": 700.0},
            },
        )
        texts = _texts(card)
        for label in ("util", "net", "Δ1h", "demand"):
            assert label in texts, f"missing badge label: {label}"

    def test_import_dominated_keeps_imports_without_util_label(self):
        # HST is import-dominated → the stress chip reads "imports", which is
        # self-describing and must NOT also get a "util" label.
        card = _build_us_grid_region_card(
            "HST",
            {"current_mw": 100.0, "prev_mw": 95.0, "today_mw": [100.0] * 24},
        )
        texts = _texts(card)
        assert "imports" in texts
        assert "util" not in texts
        assert "demand" in texts

    def test_empty_card_has_no_labels(self):
        card = _build_us_grid_region_card("PJM", {})
        texts = _texts(card)
        assert "util" not in texts
        assert "net" not in texts
        assert "demand" not in texts
