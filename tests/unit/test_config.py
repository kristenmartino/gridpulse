"""Unit tests for config.py — validates all constants and lookups."""

import pytest

from config import (
    BA_FOR_STATE,
    CDD_HDD_BASELINE_F,
    MPH_TO_MS,
    REGION_CAPACITY_MW,
    REGION_COORDINATES,
    STALENESS_THRESHOLDS_SECONDS,
    STATE_TO_BA,
    TAB_IDS,
    TAB_LABELS,
    WEATHER_VARIABLES,
    WIND_CUTOUT_SPEED_MS,
)


class TestRegionConfig:
    def test_fifty_one_regions(self):
        # Original 8 (ISOs/RTOs + FPL) + V1.α's 8 utility/federal BAs +
        # V3.ζ's 35 remaining EIA-930 BAs in the contiguous US.
        # ~99% of US lower-48 demand coverage.
        assert len(REGION_COORDINATES) == 51

    def test_all_regions_have_coordinates(self):
        for region, coords in REGION_COORDINATES.items():
            assert "lat" in coords, f"{region} missing lat"
            assert "lon" in coords, f"{region} missing lon"
            assert "name" in coords, f"{region} missing name"

    def test_fpl_is_nextera(self):
        assert "NextEra" in REGION_COORDINATES["FPL"]["name"]

    def test_all_regions_have_capacity(self):
        for region in REGION_COORDINATES:
            assert region in REGION_CAPACITY_MW, f"{region} missing capacity"
            assert REGION_CAPACITY_MW[region] > 0


class TestStateMapping:
    def test_all_regions_have_states(self):
        for region in REGION_COORDINATES:
            assert region in STATE_TO_BA, f"{region} missing state mapping"
            assert len(STATE_TO_BA[region]) > 0

    def test_reverse_lookup(self):
        assert "FL" in BA_FOR_STATE
        assert "FPL" in BA_FOR_STATE["FL"]

    def test_texas_in_multiple_bas(self):
        assert len(BA_FOR_STATE.get("TX", [])) >= 2


class TestWeatherVariables:
    def test_seventeen_variables(self):
        assert len(WEATHER_VARIABLES) == 17

    def test_key_variables_present(self):
        assert "temperature_2m" in WEATHER_VARIABLES
        assert "wind_speed_80m" in WEATHER_VARIABLES
        assert "shortwave_radiation" in WEATHER_VARIABLES


class TestTabs:
    def test_five_tabs(self):
        # V2.1 dropped 5 hidden tabs (Historical, Backtest, Generation,
        # Weather, Simulator); V1.β added US Grid as a fifth visible tab.
        # Visible set: Overview / US Grid / Forecast / Risk / Models.
        assert len(TAB_IDS) == 5
        assert len(TAB_LABELS) == 5

    def test_tab_ids_match_labels(self):
        for tab_id in TAB_IDS:
            assert tab_id in TAB_LABELS


class TestFeatureFlags:
    """``feature_enabled`` fail-closed semantics (PR-G8 / #145).

    Unknown flags must default to ``False`` so a typo disables rather
    than silently enables behavior. Every flag actually read in the
    codebase has an explicit entry, so this default only fires on a
    genuine mistake.
    """

    def test_known_flag_returns_its_value(self):
        from config import FEATURE_FLAGS, feature_enabled

        # An explicitly-True flag
        assert feature_enabled("tab_forecast") is FEATURE_FLAGS["tab_forecast"] is True
        # An explicitly-False flag (forecast_replay is disabled in the dict)
        assert feature_enabled("forecast_replay") is FEATURE_FLAGS["forecast_replay"] is False

    def test_unknown_flag_defaults_false(self):
        from config import feature_enabled

        assert feature_enabled("definitely_not_a_real_flag") is False
        assert feature_enabled("") is False

    def test_unknown_flag_logs_warning(self):
        """An unknown flag emits a ``feature_flag_unknown`` warning so a
        typo is caught rather than silently swallowed."""
        from unittest.mock import MagicMock, patch

        import config

        mock_logger = MagicMock()
        with patch("structlog.get_logger", return_value=mock_logger):
            result = config.feature_enabled("typo_flag_xyz")

        assert result is False
        mock_logger.warning.assert_called_once()
        # The warning carries the offending flag name for debuggability
        _, kwargs = mock_logger.warning.call_args
        assert kwargs.get("flag") == "typo_flag_xyz"

    def test_known_flag_does_not_log(self):
        """A registered flag must NOT emit the unknown-flag warning."""
        from unittest.mock import MagicMock, patch

        import config

        mock_logger = MagicMock()
        with patch("structlog.get_logger", return_value=mock_logger):
            config.feature_enabled("tab_forecast")

        mock_logger.warning.assert_not_called()

    def test_every_flag_read_in_code_is_defined(self):
        """Regression guard for the fail-closed flip: every flag string
        passed to ``feature_enabled()`` anywhere in the production code
        must exist in ``FEATURE_FLAGS``. If someone adds a
        ``feature_enabled("new_flag")`` call without registering the
        flag, the fail-closed default would silently disable it — this
        test catches that at PR time instead.
        """
        from config import FEATURE_FLAGS

        # Flags read in production code as of PR-G8. Keep in sync when a
        # new feature_enabled() call site is added.
        flags_read_in_code = {
            "forecast_quality_gate",  # models/model_service.py
            "cross_tab_links",  # components/insights.py
            "inline_tooltips",  # components/_callbacks_forecast.py
            "forecast_replay",  # components/_callbacks_forecast.py
            "what_changed",  # components/callbacks.py
            "smart_defaults",  # components/callbacks.py
        }
        missing = flags_read_in_code - set(FEATURE_FLAGS.keys())
        assert not missing, (
            f"These flags are read via feature_enabled() but not defined "
            f"in FEATURE_FLAGS: {missing}. With fail-closed defaults, that "
            f"silently disables them. Register them in config.FEATURE_FLAGS."
        )


class TestConstants:
    def test_cdd_baseline(self):
        assert CDD_HDD_BASELINE_F == 65.0

    def test_wind_cutout(self):
        assert WIND_CUTOUT_SPEED_MS == 25.0

    def test_mph_to_ms(self):
        assert pytest.approx(0.44704) == MPH_TO_MS

    def test_staleness_thresholds(self):
        assert STALENESS_THRESHOLDS_SECONDS["weather"] == 7200
        assert STALENESS_THRESHOLDS_SECONDS["generation"] == 300


class TestScoringRuntimeTiers:
    """The three scoring-runtime tiers must stay ordered (2026-08-04).

    warn (`SCORING_RUNTIME_HEADROOM_FRACTION`) < shed
    (`SCORING_SOFT_DEADLINE_FRACTION`) < 1.0 hard kill. Invert any pair and the
    guard silently stops working: shed below warn means BAs are dropped before
    anyone is told runtime is high, and a shed fraction at or above 1.0 means
    Cloud Run SIGKILLs the task first — which is the exact 2026-08-04 failure
    the deadline exists to prevent.
    """

    def test_warn_fires_before_shedding(self):
        import config

        assert 0 < config.SCORING_RUNTIME_HEADROOM_FRACTION < config.SCORING_SOFT_DEADLINE_FRACTION

    def test_shedding_happens_before_the_hard_kill(self):
        import config

        assert config.SCORING_SOFT_DEADLINE_FRACTION < 1.0

    def test_grace_leaves_room_inside_the_task_timeout(self):
        """In-flight BAs get a grace window; it has to fit in what is left."""
        import config

        remaining = config.SCORING_TASK_TIMEOUT_S * (1 - config.SCORING_SOFT_DEADLINE_FRACTION)
        assert remaining >= config.SCORING_DEADLINE_GRACE_S

    def test_eia_call_budget_exceeds_one_read_attempt(self):
        """A budget below one read timeout would allow zero real attempts."""
        import config

        assert config.EIA_CALL_BUDGET_S > config.EIA_READ_TIMEOUT_S > config.EIA_CONNECT_TIMEOUT_S

    def test_eia_call_budget_fits_the_whole_retry_ladder(self):
        """The budget must not cut a call off mid-ladder — this shipped broken.

        `EIA_CALL_BUDGET_S` went out at 40s against `MAX_RETRIES` (5) x
        `EIA_READ_TIMEOUT_S` (12s) = 60s of reads before any backoff, so every
        call died after ~3 attempts. Under partial degradation a call often
        needs 4-5, so seven hard-failed on the 23:00 tick of 2026-08-04,
        tripped the circuit breaker, and pushed the whole fleet onto
        last-known GCS data — against an EIA three times healthier than the
        run before it. `/health` still read 51/51, because forecasts were
        produced from stale inputs.

        The budget is meant to bound a call that is genuinely failing, not to
        truncate one that is about to succeed. Anything below this floor
        converts the cost control into an availability regression.
        """
        import config
        from data.eia_client import MAX_RETRIES

        reads = MAX_RETRIES * config.EIA_READ_TIMEOUT_S
        # Backoff doubles from INITIAL_BACKOFF_SECONDS, capped per sleep, and
        # there is one sleep fewer than there are attempts.
        from data.eia_client import INITIAL_BACKOFF_SECONDS

        backoff, worst_backoff = INITIAL_BACKOFF_SECONDS, 0.0
        for _ in range(MAX_RETRIES - 1):
            worst_backoff += min(backoff, config.EIA_MAX_BACKOFF_S)
            backoff *= 2

        assert reads + worst_backoff <= config.EIA_CALL_BUDGET_S, (
            f"budget {config.EIA_CALL_BUDGET_S}s cannot fit {MAX_RETRIES} attempts "
            f"({reads}s of reads + {worst_backoff}s of backoff) — calls will be cut "
            "off mid-ladder and fail-fast the fleet onto stale data"
        )
