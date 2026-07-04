"""Unit tests for the Risk-tab "Current Conditions" cards.

Covers ``_build_weather_context`` in ``components/_callbacks_overview.py``. It
renders one KPI mini-card per available reading (temperature / wind / humidity /
cloud). Two behaviors this pins:

- **Multiple cards, not just temperature.** In prod the scoring job now ships a
  ``weather_current`` reading with wind/humidity/cloud (previously only the
  temperature series was available, so a lone Temperature card rendered).
- **None AND NaN mean "no reading".** ``pd.Series`` coerces a None to NaN, and
  archive-unstable columns (wind_speed_80m, #164) arrive NaN — a plain
  ``is not None`` check would render a "nan" card. Wind also falls back from the
  80m to the 10m anemometer height when 80m is missing/NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _card_labels(div) -> list[str]:
    """The kpi-label text of every rendered card (in order)."""
    labels: list[str] = []

    def walk(node):
        if getattr(node, "className", None) == "kpi-label" and isinstance(node.children, str):
            labels.append(node.children)
        ch = getattr(node, "children", None)
        if isinstance(ch, (list, tuple)):
            for c in ch:
                walk(c)
        elif ch is not None and not isinstance(ch, str):
            walk(ch)

    walk(div)
    return labels


class TestBuildWeatherContext:
    def test_all_four_cards_when_all_present(self):
        from components._callbacks_overview import _build_weather_context

        s = pd.Series(
            {
                "temperature_2m": 90.0,
                "wind_speed_80m": 15.0,
                "relative_humidity_2m": 60.0,
                "cloud_cover": 40.0,
            }
        )
        assert _card_labels(_build_weather_context(s)) == [
            "TEMPERATURE",
            "WIND SPEED",
            "HUMIDITY",
            "CLOUD COVER",
        ]

    def test_none_and_nan_fields_are_skipped_not_rendered_as_nan(self):
        from components._callbacks_overview import _build_weather_context

        # None (coerced to NaN by pd.Series) and an explicit NaN must both be
        # dropped — never a "nan%" card.
        s = pd.Series(
            {
                "temperature_2m": 90.0,
                "wind_speed_80m": np.nan,
                "wind_speed_10m": np.nan,
                "relative_humidity_2m": None,
                "cloud_cover": 40.0,
            }
        )
        labels = _card_labels(_build_weather_context(s))
        assert labels == ["TEMPERATURE", "CLOUD COVER"]  # no wind, no humidity

    def test_wind_falls_back_from_80m_to_10m(self):
        from components._callbacks_overview import _build_weather_context

        s = pd.Series({"temperature_2m": 90.0, "wind_speed_80m": np.nan, "wind_speed_10m": 12.0})
        labels = _card_labels(_build_weather_context(s))
        assert "WIND SPEED" in labels  # 10m carried it

    def test_temperature_only_renders_one_card(self):
        from components._callbacks_overview import _build_weather_context

        labels = _card_labels(_build_weather_context(pd.Series({"temperature_2m": 90.0})))
        assert labels == ["TEMPERATURE"]

    def test_empty_reading_renders_nothing(self):
        from components._callbacks_overview import _build_weather_context

        div = _build_weather_context(pd.Series({}, dtype=float))
        assert _card_labels(div) == []
