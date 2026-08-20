"""#559: recording the temporal-seed arm without serving it.

The offline replay could not settle whether the temporal seed is more accurate,
and cannot be made to — at the rate gaps actually occur a decisive verdict is
1.2-6.6 years out (``docs/POSITIONAL_LAG_SEED_STUDY.md``). So this shadow is a
pre-rollout *safety* instrument, and the properties worth testing are about what
it must not disturb and about the gate that keeps it cheap — not the arithmetic.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from data.feature_engineering import positional_seed_matches_hours
from jobs.phases import (
    RegionData,
    _is_seed_shadow_audit_region,
    _write_seed_shadow,
    predict_and_write_forecast,
    reset_seed_shadow_budget,
)

HOURS = 400


def _featured(hours: int = HOURS, gap_at: int | None = None) -> pd.DataFrame:
    ts = pd.date_range("2026-05-19", periods=hours, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": 18000.0 + 900 * np.sin(2 * np.pi * np.arange(hours) / 24),
        }
    )
    if gap_at is not None:
        df = df.drop(index=range(gap_at, gap_at + 16)).reset_index(drop=True)
    for col, attr in (
        ("hour", "hour"),
        ("day_of_week", "dayofweek"),
        ("month", "month"),
        ("day_of_year", "dayofyear"),
    ):
        df[col] = getattr(df["timestamp"].dt, attr)
    return df


def _region_data(region: str = "FPL", featured: pd.DataFrame | None = None) -> RegionData:
    featured = _featured() if featured is None else featured
    return RegionData(
        region=region,
        demand_df=featured[["timestamp", "demand_mw"]],
        weather_df=pd.DataFrame(),
        featured_df=featured,
    )


def _origin(featured: pd.DataFrame) -> pd.Timestamp:
    return featured["timestamp"].iloc[-1] + pd.Timedelta(hours=1)


@pytest.fixture(autouse=True)
def _fresh_budget():
    """Every tick is a fresh process in production; every test gets that too."""
    reset_seed_shadow_budget()
    yield
    reset_seed_shadow_budget()


@pytest.fixture
def shadow_on(monkeypatch):
    monkeypatch.setitem(config.FEATURE_FLAGS, "temporal_ar_seed_shadow", True)


def _call(featured, *, region="FPL", horizon=24, model=object()):
    rows = [
        {"timestamp": t.isoformat()}
        for t in pd.date_range(_origin(featured), periods=horizon, freq="h", tz="UTC")
    ]
    return _write_seed_shadow(
        region=region,
        model=model,
        featured=featured,
        future_df=pd.DataFrame({"timestamp": [r["timestamp"] for r in rows]}),
        horizon=horizon,
        forecast_start=_origin(featured),
        served_preds=np.full(horizon, 18000.0),
        rows=rows,
        demand_df=featured[["timestamp", "demand_mw"]],
        xgboost_weight=0.6,
    )


class TestThePredicate:
    """The gate is the exact divergence condition, not a proxy for it."""

    def test_a_contiguous_seed_tail_means_the_arms_cannot_differ(self):
        f = _featured()
        assert positional_seed_matches_hours(f["timestamp"], _origin(f)) is True

    def test_a_hole_inside_the_lookback_means_they_will(self):
        f = _featured(gap_at=HOURS - 60)
        assert positional_seed_matches_hours(f["timestamp"], _origin(f)) is False

    def test_a_hole_outside_the_lookback_does_not_count(self):
        """A gap older than the longest lag cannot move any index.

        The first draft of the gapped test fixture in ``test_temporal_ar_seed``
        put its hole here and proved nothing, so this direction is pinned too.
        """
        f = _featured(hours=600, gap_at=100)
        assert positional_seed_matches_hours(f["timestamp"], _origin(f)) is True

    def test_too_little_history_is_treated_as_diverging(self):
        f = _featured(hours=50)
        assert positional_seed_matches_hours(f["timestamp"], _origin(f)) is False

    def test_unusable_input_is_treated_as_diverging(self):
        assert positional_seed_matches_hours(None, pd.Timestamp("2026-05-19", tz="UTC")) is False


class TestTheGateKeepsItCheap:
    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_an_ungapped_ba_runs_no_second_inference(self, _set, _get, shadow_on):
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            _call(_featured(), region="ZZZNOTAREGION")
            assert spy.call_count == 0, "a BA whose arms cannot differ must not be re-run"

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_a_gapped_ba_runs_exactly_one_second_inference(self, _set, _get, shadow_on):
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            spy.return_value = np.full(24, 18500.0)
            _call(_featured(gap_at=HOURS - 60))
            assert spy.call_count == 1
            assert spy.call_args.kwargs["force_temporal"] is True

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_the_flag_off_does_nothing_at_all(self, _set, get, monkeypatch):
        monkeypatch.setitem(config.FEATURE_FLAGS, "temporal_ar_seed_shadow", False)
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            assert _call(_featured(gap_at=HOURS - 60)) is False
            assert spy.call_count == 0
            assert get.call_count == 0


class TestTheRotatingAudit:
    """A gate that silently skips everything looks exactly like a quiet fleet."""

    def test_exactly_one_region_is_audited_per_hour(self):
        origin = pd.Timestamp("2026-08-20T11:00", tz="UTC")
        picked = [
            r for r in sorted(config.REGION_COORDINATES) if _is_seed_shadow_audit_region(r, origin)
        ]
        assert len(picked) == 1

    def test_every_region_comes_round(self):
        base = pd.Timestamp("2026-08-20T00:00", tz="UTC")
        seen = {
            r
            for h in range(len(config.REGION_COORDINATES))
            for r in sorted(config.REGION_COORDINATES)
            if _is_seed_shadow_audit_region(r, base + pd.Timedelta(hours=h))
        }
        assert seen == set(config.REGION_COORDINATES)

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_the_audited_region_is_shadowed_even_though_it_should_not_differ(
        self, _set, _get, shadow_on
    ):
        origin = _origin(_featured())
        region = next(
            r for r in sorted(config.REGION_COORDINATES) if _is_seed_shadow_audit_region(r, origin)
        )
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            spy.return_value = np.full(24, 18000.0)
            _call(_featured(), region=region)
            assert spy.call_count == 1, "the audit region must be run despite the gate"

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_a_divergent_audit_raises_an_alarm_about_the_gate(self, _set, _get, shadow_on):
        origin = _origin(_featured())
        region = next(
            r for r in sorted(config.REGION_COORDINATES) if _is_seed_shadow_audit_region(r, origin)
        )
        with (
            patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy,
            patch("jobs.phases.log") as log,
        ):
            spy.return_value = np.full(24, 19000.0)  # differs, which must not happen
            _call(_featured(), region=region)
            assert any(
                c.args and c.args[0] == "seed_shadow_audit_diverged"
                for c in log.warning.call_args_list
            ), "a divergent audit is evidence the gate is wrong and must be loud"


class TestItDoesNotDisturbWhatShips:
    """The only property here that can cause an incident."""

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_the_served_payload_is_identical_with_and_without_the_shadow(self, _get, mock_set):
        payloads = {}
        for label, on in (("on", True), ("off", False)):
            mock_set.reset_mock()
            with (
                patch.dict(config.FEATURE_FLAGS, {"temporal_ar_seed_shadow": on}),
                patch("jobs.phases._predict_one") as predict,
            ):
                predict.side_effect = lambda m, *a, **k: np.full(720, 18000.0)
                predict_and_write_forecast(
                    _region_data(),
                    models={"xgboost": object(), "prophet": object()},
                    model_mapes={"xgboost": 3.0, "prophet": 6.0},
                )
            payload = next(c.args[1] for c in mock_set.call_args_list if "forecast:" in c.args[0])
            # Wall-clock stamps differ between two runs by construction; every
            # other key is the thing under test.
            payloads[label] = {
                k: v for k, v in payload.items() if k not in {"scored_at", "generated_at"}
            }
        assert payloads["on"] == payloads["off"]

    @patch("data.redis_client.redis_set")
    @patch("data.redis_client.redis_get", return_value=None)
    def test_the_shadow_series_never_enters_the_served_payload(self, _get, mock_set, shadow_on):
        """``drift.extract_one_hour_ahead_predictions`` treats every numeric key
        in a forecast row as a model, so a shadow series leaking into the served
        payload would silently acquire drift records and a place in the
        published rolling MAPE."""
        with patch("jobs.phases._predict_one") as predict:
            predict.side_effect = lambda m, *a, **k: np.full(720, 18000.0)
            predict_and_write_forecast(
                _region_data(),
                models={"xgboost": object(), "prophet": object()},
                model_mapes={"xgboost": 3.0, "prophet": 6.0},
            )
        served = next(c.args[1] for c in mock_set.call_args_list if "forecast:" in c.args[0])
        for row in served["forecasts"]:
            assert "shadow" not in row
            assert "served" not in row


class TestThePayload:
    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_it_records_why_a_tick_computed_nothing(self, mock_set, _get, shadow_on):
        """A quiet key must not be ambiguous between "no gaps" and "not running"."""
        _call(_featured(), region="ZZZNOTAREGION")
        payload = next(c.args[1] for c in mock_set.call_args_list if "seed_shadow:" in c.args[0])
        assert payload["gate"] == "identical"
        assert payload["computed"] is False
        assert payload["audited"] is False

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_a_diverging_tick_records_both_arms_and_the_blend_weight(
        self, mock_set, _get, shadow_on
    ):
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            spy.return_value = np.full(24, 18500.0)
            _call(_featured(gap_at=HOURS - 60))
        payload = next(c.args[1] for c in mock_set.call_args_list if "seed_shadow:" in c.args[0])
        # #624: the gate names WHY the arms differ. This fixture holes the
        # lookback while the seed still reaches origin-1h, so the array is
        # correctly sized and the observation is clean evidence about
        # temporal indexing — not the ``seed_tail_short`` stratum.
        assert payload["gate"] == "hole_in_lookback"
        assert payload["seed_tail_gap_h"] == 0
        assert payload["computed"] is True
        assert payload["divergence_pct"] == pytest.approx(100 * 500 / 18000, rel=1e-6)
        # The served headline is the ensemble; its delta is this weight times
        # the XGBoost delta, so no third arm needs computing.
        assert payload["xgboost_weight"] == 0.6
        assert payload["forecasts"][0].keys() >= {"timestamp", "served", "shadow"}


class TestThePerRunCap:
    """The gate is data-dependent; the cap is what makes the cost bounded.

    `_write_seed_shadow` runs after the served payload persists, so it cannot
    lose its own BA's forecast. What an unbounded enrichment CAN do is push the
    run past the soft deadline and get *later* BAs shed whole — buying shadow
    data with real forecasts. CLAUDE.md's #389 rule is to bound what one run
    costs, not merely to gate it.
    """

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_the_second_recursion_stops_at_the_cap(self, _set, _get, shadow_on, monkeypatch):
        monkeypatch.setattr(config, "SEED_SHADOW_MAX_REGIONS_PER_TICK", 2)
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            spy.return_value = np.full(24, 18500.0)
            for i in range(6):
                _call(_featured(gap_at=HOURS - 60), region=f"R{i}")
        assert spy.call_count == 2, "a data-dependent gate must not be able to run the fleet"

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_a_declined_tick_says_so_rather_than_looking_quiet(
        self, mock_set, _get, shadow_on, monkeypatch
    ):
        """A dropped observation that reads as "no gap" would bias the sample."""
        monkeypatch.setattr(config, "SEED_SHADOW_MAX_REGIONS_PER_TICK", 0)
        _call(_featured(gap_at=HOURS - 60))
        payload = next(c.args[1] for c in mock_set.call_args_list if "seed_shadow:" in c.args[0])
        # #624: the gate names WHY the arms differ. This fixture holes the
        # lookback while the seed still reaches origin-1h, so the array is
        # correctly sized and the observation is clean evidence about
        # temporal indexing — not the ``seed_tail_short`` stratum.
        assert payload["gate"] == "hole_in_lookback"
        assert payload["seed_tail_gap_h"] == 0
        assert payload["computed"] is False
        assert payload["budget_declined"] is True

    @patch("data.redis_client.redis_get", return_value=None)
    @patch("data.redis_client.redis_set")
    def test_skipped_bas_do_not_consume_budget(self, _set, _get, shadow_on, monkeypatch):
        """Only a BA that actually re-runs should cost a slot."""
        monkeypatch.setattr(config, "SEED_SHADOW_MAX_REGIONS_PER_TICK", 1)
        with patch("jobs.phases._predict_xgboost_with_recursive_autoregressive") as spy:
            spy.return_value = np.full(24, 18500.0)
            for i in range(5):
                _call(_featured(), region=f"ZZZ{i}")  # ungapped -> no spend
            _call(_featured(gap_at=HOURS - 60), region="GAPPED")
            assert spy.call_count == 1
