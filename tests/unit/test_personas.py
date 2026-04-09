"""Unit tests for personas/config.py."""

import pytest

from personas.config import (
    Persona,
    get_persona,
    get_welcome_card,
    list_personas,
)


class TestGetPersona:
    def test_all_four_personas_exist(self):
        for pid in ["grid_ops", "renewables", "trader", "data_scientist"]:
            p = get_persona(pid)
            assert isinstance(p, Persona)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown persona"):
            get_persona("ceo")

    def test_grid_ops_defaults(self):
        p = get_persona("grid_ops")
        assert p.name == "Sarah"
        assert p.default_tab == "tab-overview"
        assert "peak_demand_mw" in p.kpi_metrics

    def test_trader_defaults(self):
        p = get_persona("trader")
        assert p.name == "Maria"
        assert p.default_tab == "tab-overview"


class TestListPersonas:
    def test_returns_four(self):
        result = list_personas()
        assert len(result) == 4

    def test_has_required_keys(self):
        for p in list_personas():
            assert "id" in p
            assert "name" in p
            assert "title" in p
            assert "avatar" in p


class TestWelcomeCard:
    def test_returns_expected_keys(self):
        card = get_welcome_card("grid_ops")
        assert "title" in card
        assert "message" in card
        assert "avatar" in card
        assert "default_tab" in card
        assert "kpis" in card

    def test_persona_specific_content(self):
        card = get_welcome_card("renewables")
        assert "James" in card["message"]
        assert card["default_tab"] == "tab-overview"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_welcome_card("unknown")
