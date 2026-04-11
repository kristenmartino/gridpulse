"""
Unit tests for NEXD-13: Inline 'Why' Tooltips on Model Outputs.

Covers:
- FEATURE_LABELS completeness (all canonical features have human labels)
- get_top_drivers_shap(): per-point SHAP extraction
- get_top_drivers_global(): XGBoost feature importance extraction
- format_driver_line(): tooltip line formatting in shap/global modes
- build_tooltip_strings(): end-to-end tooltip list construction
- Feature flag existence
"""

import numpy as np

from data.explainability import (
    FEATURE_LABELS,
    build_tooltip_strings,
    format_driver_line,
    get_top_drivers_global,
    get_top_drivers_shap,
)

# ── FEATURE_LABELS completeness ───────────────────────────────────


class TestFeatureLabels:
    def test_canonical_features_have_labels(self):
        """All 43 canonical features from get_feature_names() have labels."""
        from data.feature_engineering import get_feature_names

        for name in get_feature_names():
            assert name in FEATURE_LABELS, f"Missing label for: {name}"

    def test_extra_time_features_have_labels(self):
        """Extra time features from _create_future_features have labels."""
        extras = ["hour", "day_of_week", "month", "day_of_year", "is_weekend"]
        for name in extras:
            assert name in FEATURE_LABELS, f"Missing label for: {name}"

    def test_no_empty_labels(self):
        for name, label in FEATURE_LABELS.items():
            assert label.strip(), f"Empty label for: {name}"

    def test_labels_are_strings(self):
        for name, label in FEATURE_LABELS.items():
            assert isinstance(label, str), f"Non-string label for: {name}"


# ── get_top_drivers_shap ──────────────────────────────────────────


class TestGetTopDriversShap:
    def _make_shap(self, n_points=5, n_features=4):
        rng = np.random.RandomState(42)
        return rng.randn(n_points, n_features)

    def test_returns_top_n(self):
        shap_vals = self._make_shap()
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover", "hour"]
        result = get_top_drivers_shap(shap_vals, features, index=0, top_n=3)
        assert len(result) == 3

    def test_sorted_by_abs_value(self):
        shap_vals = np.array([[100.0, -200.0, 50.0, -10.0]])
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover", "hour"]
        result = get_top_drivers_shap(shap_vals, features, index=0, top_n=4)
        abs_vals = [abs(v) for _, v in result]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_uses_human_labels(self):
        shap_vals = np.array([[100.0, -200.0]])
        features = ["temperature_2m", "wind_speed_80m"]
        result = get_top_drivers_shap(shap_vals, features, index=0, top_n=2)
        labels = [lbl for lbl, _ in result]
        assert "Temperature" in labels
        assert "Wind Speed" in labels

    def test_preserves_sign(self):
        shap_vals = np.array([[-500.0, 300.0]])
        features = ["temperature_2m", "wind_speed_80m"]
        result = get_top_drivers_shap(shap_vals, features, index=0, top_n=2)
        vals = dict(result)
        assert vals["Temperature"] == -500.0
        assert vals["Wind Speed"] == 300.0

    def test_different_indices(self):
        shap_vals = np.array([[100.0, -200.0], [300.0, -50.0]])
        features = ["temperature_2m", "wind_speed_80m"]
        r0 = get_top_drivers_shap(shap_vals, features, index=0, top_n=1)
        r1 = get_top_drivers_shap(shap_vals, features, index=1, top_n=1)
        assert r0[0][0] == "Wind Speed"  # |-200| > |100|
        assert r1[0][0] == "Temperature"  # |300| > |-50|

    def test_unknown_feature_gets_title_label(self):
        shap_vals = np.array([[100.0]])
        features = ["unknown_exotic_feature"]
        result = get_top_drivers_shap(shap_vals, features, index=0, top_n=1)
        assert result[0][0] == "Unknown Exotic Feature"


# ── get_top_drivers_global ────────────────────────────────────────


class TestGetTopDriversGlobal:
    def _make_model_dict(self):
        """Create a minimal mock model_dict like train_xgboost() output."""

        class MockModel:
            feature_importances_ = np.array([0.3, 0.1, 0.5, 0.05, 0.05])

        return {
            "model": MockModel(),
            "feature_names": [
                "temperature_2m",
                "wind_speed_80m",
                "cloud_cover",
                "hour",
                "demand_lag_24h",
            ],
        }

    def test_returns_top_n(self):
        result = get_top_drivers_global(self._make_model_dict(), top_n=3)
        assert len(result) == 3

    def test_sorted_by_importance(self):
        result = get_top_drivers_global(self._make_model_dict(), top_n=5)
        importances = [v for _, v in result]
        assert importances == sorted(importances, reverse=True)

    def test_top_feature_is_cloud_cover(self):
        result = get_top_drivers_global(self._make_model_dict(), top_n=1)
        assert result[0][0] == "Cloud Cover"
        assert result[0][1] == 0.5


# ── format_driver_line ────────────────────────────────────────────


class TestFormatDriverLine:
    def test_shap_positive(self):
        line = format_driver_line("Temperature", 1200.0, mode="shap")
        assert line == "Temperature: +1,200 MW"

    def test_shap_negative(self):
        line = format_driver_line("Wind Speed", -300.0, mode="shap")
        assert line == "Wind Speed: -300 MW"

    def test_shap_zero(self):
        line = format_driver_line("Humidity", 0.0, mode="shap")
        assert line == "Humidity: +0 MW"

    def test_global_high(self):
        line = format_driver_line("Temperature", 0.3, mode="global")
        assert line == "Temperature (high)"

    def test_global_zero(self):
        line = format_driver_line("Humidity", 0.0, mode="global")
        assert line == "Humidity"


# ── build_tooltip_strings ─────────────────────────────────────────


class TestBuildTooltipStrings:
    def _make_shap_data(self, n_points=5, n_features=3):
        rng = np.random.RandomState(42)
        return rng.randn(n_points, n_features) * 500

    def _make_model_dict(self):
        class MockModel:
            feature_importances_ = np.array([0.5, 0.3, 0.2])

        return {
            "model": MockModel(),
            "feature_names": ["temperature_2m", "wind_speed_80m", "cloud_cover"],
        }

    def test_xgboost_with_shap(self):
        shap_vals = self._make_shap_data(n_points=5)
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover"]
        result = build_tooltip_strings(
            shap_values=shap_vals,
            feature_names=features,
            model_dict=self._make_model_dict(),
            n_points=5,
            model_name="xgboost",
        )
        assert len(result) == 5
        assert all(isinstance(s, str) for s in result)
        assert all(s != "" for s in result)
        # Each should contain MW values
        assert "MW" in result[0]

    def test_xgboost_without_shap_uses_global(self):
        result = build_tooltip_strings(
            shap_values=None,
            feature_names=None,
            model_dict=self._make_model_dict(),
            n_points=5,
            model_name="xgboost",
        )
        assert len(result) == 5
        # Global fallback — all same static tooltip
        assert result[0] == result[4]
        assert "(high)" in result[0]

    def test_prophet_uses_global_fallback(self):
        result = build_tooltip_strings(
            shap_values=None,
            feature_names=None,
            model_dict=self._make_model_dict(),
            n_points=5,
            model_name="prophet",
        )
        assert len(result) == 5
        assert result[0] == result[4]  # Same static tooltip for all

    def test_no_model_dict_returns_empty(self):
        result = build_tooltip_strings(
            shap_values=None,
            feature_names=None,
            model_dict=None,
            n_points=5,
            model_name="arima",
        )
        assert len(result) == 5
        assert all(s == "" for s in result)

    def test_shap_with_fewer_rows_falls_back(self):
        """If SHAP has fewer rows than n_points, fall back to global."""
        shap_vals = self._make_shap_data(n_points=3)
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover"]
        result = build_tooltip_strings(
            shap_values=shap_vals,
            feature_names=features,
            model_dict=self._make_model_dict(),
            n_points=5,
            model_name="xgboost",
        )
        # Falls to global since shap has 3 rows but we need 5
        assert len(result) == 5
        assert "(high)" in result[0]

    def test_non_xgboost_ignores_shap(self):
        """Even if SHAP data is provided, non-XGBoost model uses global fallback."""
        shap_vals = self._make_shap_data(n_points=5)
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover"]
        result = build_tooltip_strings(
            shap_values=shap_vals,
            feature_names=features,
            model_dict=self._make_model_dict(),
            n_points=5,
            model_name="prophet",
        )
        assert len(result) == 5
        # Prophet uses global, not SHAP
        assert result[0] == result[4]

    def test_per_point_tooltips_vary(self):
        """SHAP tooltips should differ between points (different feature contributions)."""
        shap_vals = np.array(
            [
                [1000.0, -200.0, 50.0],
                [-500.0, 800.0, -100.0],
            ]
        )
        features = ["temperature_2m", "wind_speed_80m", "cloud_cover"]
        result = build_tooltip_strings(
            shap_values=shap_vals,
            feature_names=features,
            model_dict=None,
            n_points=2,
            model_name="xgboost",
        )
        assert result[0] != result[1]


# ── Feature flag ──────────────────────────────────────────────────


class TestFeatureFlag:
    def test_inline_tooltips_flag_exists(self):
        from config import FEATURE_FLAGS

        assert "inline_tooltips" in FEATURE_FLAGS

    def test_inline_tooltips_flag_enabled(self):
        from config import feature_enabled

        assert feature_enabled("inline_tooltips") is True
