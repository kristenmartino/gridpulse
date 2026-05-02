"""Tests for the geographic grouping of balancing authorities (V3 polish).

``REGION_GROUPS`` is the single source of truth for both the header
dropdown's optgroup-style separators and the US Grid card grid's section
headers. These tests lock in:

- Total coverage: every BA in ``REGION_COORDINATES`` appears exactly
  once across the groups.
- Group order: A-Z by group name (Central, Northeast, Southeast, West).
- BA order within each group: A-Z by code.
- Dropdown rendering: section options have ``disabled=True`` and value
  ``""``; selectable options interleave correctly between separators.
- Card grid rendering: section headers carry ``gp-region-grid__section-header``
  and appear before each group's cards in the right order.
"""

from __future__ import annotations


class TestRegionGroupsCoverage:
    def test_every_ba_in_exactly_one_group(self):
        from config import REGION_COORDINATES, REGION_GROUPS

        all_grouped = [code for codes in REGION_GROUPS.values() for code in codes]
        assert sorted(all_grouped) == sorted(REGION_COORDINATES.keys())
        # No duplicates
        assert len(all_grouped) == len(set(all_grouped))

    def test_no_unknown_codes_in_groups(self):
        """A code in REGION_GROUPS that doesn't exist in
        REGION_COORDINATES would render an unselectable card."""
        from config import REGION_COORDINATES, REGION_GROUPS

        for codes in REGION_GROUPS.values():
            for code in codes:
                assert code in REGION_COORDINATES, (
                    f"{code} listed in REGION_GROUPS but missing from REGION_COORDINATES"
                )


class TestRegionGroupsOrdering:
    def test_groups_sorted_alphabetically(self):
        from config import REGION_GROUPS

        names = list(REGION_GROUPS.keys())
        assert names == sorted(names)

    def test_codes_within_each_group_sorted(self):
        from config import REGION_GROUPS

        for group_name, codes in REGION_GROUPS.items():
            assert codes == sorted(codes), f"Group '{group_name}' codes not sorted A-Z: {codes}"

    def test_groups_match_recommendation(self):
        """Lock in the exact regional grouping shipped — change with
        intent. If a future refactor wants to reshape the groups, the
        test fail forces an explicit update + design review."""
        from config import REGION_GROUPS

        assert REGION_GROUPS == {
            "Central": ["ERCOT", "MISO", "SPP"],
            "Northeast": ["ISONE", "NYISO", "PJM"],
            "Southeast": ["CPLE", "DUK", "FPL", "SOCO", "TVA"],
            "West": ["AZPS", "BPAT", "CAISO", "NEVP", "PSCO"],
        }


class TestGroupedRegionsHelper:
    def test_returns_list_of_tuples(self):
        from config import grouped_regions

        result = grouped_regions()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_order_matches_region_groups(self):
        from config import REGION_GROUPS, grouped_regions

        assert grouped_regions() == list(REGION_GROUPS.items())


class TestHeaderDropdownGrouping:
    """The header dropdown uses ``REGION_GROUPS`` to render disabled
    separator options between selectable BA options. These tests lock
    in the dropdown's options structure."""

    def _region_options(self):
        """Build region options the same way ``_build_header`` does."""
        from config import REGION_GROUPS, REGION_NAMES

        options: list[dict] = []
        for group_name, codes in REGION_GROUPS.items():
            options.append({"label": f"── {group_name} ──", "value": "", "disabled": True})
            for code in codes:
                options.append({"label": REGION_NAMES.get(code, code), "value": code})
        return options

    def test_total_option_count(self):
        """Total = sum(group_size) + N_groups disabled separators."""
        from config import REGION_GROUPS

        opts = self._region_options()
        n_codes = sum(len(codes) for codes in REGION_GROUPS.values())
        n_separators = len(REGION_GROUPS)
        assert len(opts) == n_codes + n_separators

    def test_separators_are_disabled(self):
        opts = self._region_options()
        separators = [o for o in opts if o.get("disabled")]
        assert len(separators) == 4  # 4 groups
        for sep in separators:
            assert sep["value"] == ""
            assert sep["label"].startswith("── ")
            assert sep["label"].endswith(" ──")

    def test_selectable_options_carry_real_codes(self):
        from config import REGION_COORDINATES

        opts = self._region_options()
        selectable = [o for o in opts if not o.get("disabled")]
        codes = [o["value"] for o in selectable]
        assert sorted(codes) == sorted(REGION_COORDINATES.keys())

    def test_separator_precedes_its_group(self):
        """Each disabled separator must immediately precede the BAs of
        its group — checks that the rendering order isn't scrambled."""
        from config import REGION_GROUPS

        opts = self._region_options()
        i = 0
        for group_name, codes in REGION_GROUPS.items():
            assert opts[i]["disabled"] is True
            assert group_name in opts[i]["label"]
            i += 1
            for code in codes:
                assert opts[i]["value"] == code
                assert opts[i].get("disabled") is None or opts[i]["disabled"] is False
                i += 1


class TestUsGridCardGrouping:
    """The US Grid card grid renders a section header before each
    group's cards. The header is a Div with the
    ``gp-region-grid__section-header`` class, and the grid is a
    pattern-matching container (``gp-region-grid``) so existing CSS
    handles full-row spans via ``grid-column: 1 / -1``."""

    def _patch_redis(self, monkeypatch, redis_state):
        from components import callbacks as cb

        def _get(key):
            return redis_state.get(key)

        monkeypatch.setattr(cb, "redis_get", _get)

    def _render_us_grid_body(self, monkeypatch):
        """Build the cards body the way ``update_us_grid_snapshot`` does."""
        from dash import html

        from components.callbacks import _build_us_grid_region_card, _collect_us_grid_region_data
        from config import REGION_GROUPS

        region_data = _collect_us_grid_region_data()
        grid_children: list = []
        for group_name, codes in REGION_GROUPS.items():
            grid_children.append(html.Div(group_name, className="gp-region-grid__section-header"))
            grid_children.extend(
                _build_us_grid_region_card(code, region_data.get(code, {})) for code in codes
            )
        return html.Div(grid_children, className="gp-region-grid")

    def test_one_section_header_per_group(self, monkeypatch):
        self._patch_redis(monkeypatch, {})  # cold pipeline; cards render placeholders
        body = self._render_us_grid_body(monkeypatch)
        headers = [
            c
            for c in body.children
            if getattr(c, "className", "") == "gp-region-grid__section-header"
        ]
        from config import REGION_GROUPS

        assert len(headers) == len(REGION_GROUPS)

    def test_section_header_text_matches_group_name(self, monkeypatch):
        self._patch_redis(monkeypatch, {})
        body = self._render_us_grid_body(monkeypatch)
        from config import REGION_GROUPS

        header_texts = [
            c.children
            for c in body.children
            if getattr(c, "className", "") == "gp-region-grid__section-header"
        ]
        assert header_texts == list(REGION_GROUPS.keys())

    def test_cards_immediately_follow_their_section_header(self, monkeypatch):
        """Order check: header → its codes → next header → its codes."""
        self._patch_redis(monkeypatch, {})
        body = self._render_us_grid_body(monkeypatch)
        from config import REGION_GROUPS

        children = body.children
        idx = 0
        for group_name, codes in REGION_GROUPS.items():
            assert getattr(children[idx], "className", "") == "gp-region-grid__section-header"
            assert children[idx].children == group_name
            idx += 1
            for code in codes:
                # Cards have pattern-matching dict IDs of shape
                # {"type": "us-grid-region-card", "region": code}
                card = children[idx]
                assert isinstance(card.id, dict)
                assert card.id.get("type") == "us-grid-region-card"
                assert card.id.get("region") == code
                idx += 1
