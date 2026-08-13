"""The first tests in this repo that execute a Dash callback (#399).

Every tier before this one stopped at the seam. Unit tests call the helper
*inside* a callback; the smoke tier calls ``layout()`` and asserts it is not
``None``. Nothing checked that the two are wired to each other — so a renamed
component id, an input list in the wrong order, or a return value that cannot
be serialised all passed CI and broke in a browser.

These dispatch real ``POST /_dash-update-component`` requests through
``app.server.test_client()``. Dash resolves the callback from its own registry
and invokes the actual function. See ``dash_driver.py`` for the mechanics and
for what a driver without a browser still cannot see.
"""

from __future__ import annotations

import json

import pytest

from tests.integration.dash_driver import DashDriver


@pytest.fixture(scope="module")
def driver() -> DashDriver:
    """One app import per module — registration is the expensive part."""
    from app import app

    return DashDriver(app)


class TestCallbackWiring:
    """The layout/callback contract, checked against Dash's own registry."""

    def test_every_callback_output_id_exists_in_the_layout(self, driver: DashDriver) -> None:
        """An ``Output`` with no matching id is a callback that never runs.

        Dash does not raise for this at registration time — with
        ``suppress_callback_exceptions=True`` (set in ``app.py``, and required
        because tab content mounts dynamically) it does not raise at all. The
        panel simply renders empty forever, which looks like a data problem
        and gets debugged as one.

        CLAUDE.md's UI rule is "preserve IDs unless intentionally refactoring
        callbacks". This is that rule as a test, over all callbacks at once,
        read from ``app.callback_map`` rather than grepped out of source.
        """
        missing = driver.callback_output_ids() - driver.layout_ids()
        assert not missing, (
            f"callbacks write to component ids that no layout declares: {sorted(missing)}"
        )

    def test_the_contract_covers_the_whole_app_not_a_sample(self, driver: DashDriver) -> None:
        """Guard the guard: an empty registry would make the check above vacuous.

        If a refactor moved registration behind a flag that defaults off, the
        set difference would be empty and the test would still pass. Pin the
        floor instead — the app registers dozens of callbacks writing dozens
        of components, and a collapse to near-zero is the bug.
        """
        outputs = driver.callback_output_ids()
        assert len(outputs) > 50, f"expected the full callback surface, got {len(outputs)}"
        assert len(driver.layout_ids()) > 50

    def test_dispatching_an_unregistered_output_is_rejected(self, driver: DashDriver) -> None:
        """Characterisation: prove the driver can actually see an orphan.

        Without this, a green run of the whole module proves nothing about
        whether dispatch is checking anything. Dash raises when asked for an
        output it holds no callback for, and the failure surfaces as a 500 —
        which is exactly how an orphaned callback would present once its id
        stopped matching.
        """
        response = driver.client.post(
            "/_dash-update-component",
            json={
                "output": "no-such-component.children",
                "outputs": {"id": "no-such-component", "property": "children"},
                "inputs": [{"id": "region-selector", "property": "value", "value": "CISO"}],
                "changedPropIds": ["region-selector.value"],
            },
        )
        assert response.status_code == 500


class TestRegionSwitch:
    """The most common real interaction: pick a different balancing authority."""

    def test_region_selection_reaches_the_rendered_title(self, driver: DashDriver) -> None:
        """A region switch must change what the Models tab actually renders.

        This is the flow the smoke tier could only assert "is not None" about.
        Here the callback runs, and the assertion is on the text a user would
        read — including that two regions produce two different results, which
        catches a callback that ignores its input and returns a constant.
        """
        ciso = driver.dispatch(
            "models-title.children",
            inputs={"region-selector": "CISO", "dashboard-tabs": "tab-models"},
        )
        ercot = driver.dispatch(
            "models-title.children",
            inputs={"region-selector": "ERCO", "dashboard-tabs": "tab-models"},
        )

        assert ciso.ok and ercot.ok
        assert "CISO" in ciso.text("models-title")
        assert "ERCO" in ercot.text("models-title")
        assert ciso.text("models-title") != ercot.text("models-title")

    def test_an_unknown_region_does_not_crash_the_callback(self, driver: DashDriver) -> None:
        """Region ids arrive from the URL, so they are user-controlled.

        ``restore_bookmark`` validates against ``REGION_COORDINATES`` before
        applying, but nothing stops a hand-edited store value reaching a tab
        callback. It must degrade, not 500 — a stack trace here is a blank
        dashboard for every subsequent interaction.
        """
        result = driver.dispatch(
            "models-title.children",
            inputs={"region-selector": "NOT_A_REGION", "dashboard-tabs": "tab-models"},
        )
        assert result.ok, "an unknown region should render something, not raise"


class TestBookmarkRestore:
    """C2: a shared URL must reproduce the view it was copied from."""

    def test_query_params_restore_region_persona_and_tab(self, driver: DashDriver) -> None:
        """The bookmark contract, end to end through dispatch for the first time.

        ``restore_bookmark`` is one of three callbacks writing
        ``region-selector.value``; the driver selects it by the property it
        watches (``url.search``), which is the only thing distinguishing it
        from the pathname-triggered one.
        """
        result = driver.dispatch(
            "region-selector.value",
            inputs={"url.search": "?region=ERCOT&persona=trader&tab=tab-models"},
        )

        assert result.ok
        assert result.value("region-selector", "value") == "ERCOT"
        assert result.value("persona-selector", "value") == "trader"
        assert result.value("dashboard-tabs", "active_tab") == "tab-models"

    def test_an_invalid_region_param_is_not_applied(self, driver: DashDriver) -> None:
        """CLAUDE.md (C2): "Always validate param values against known sets".

        A URL is attacker-supplied. If an unvalidated region reached the
        stores, every downstream callback would key Redis lookups on it.

        The observed behaviour is stricter than "ignore the bad key": with
        nothing valid left to apply, the callback declines to update at all
        (Dash answers 204). Pinned as-is — a future change that started
        applying a partial restore here would be a real behaviour change and
        should have to edit this assertion.
        """
        result = driver.dispatch(
            "region-selector.value",
            inputs={"url.search": "?region=../../etc/passwd"},
        )

        assert result.prevented, (
            f"an unrecognised region should update nothing; got {result.status_code} "
            f"{result.payload}"
        )

    def test_a_valid_param_still_applies_alongside_an_invalid_one(self, driver: DashDriver) -> None:
        """Rejecting one param must not silently discard the rest.

        Without this, the test above could pass for the wrong reason — a
        callback that bailed out entirely on any unrecognised key would look
        identical, and a user's shared link would quietly lose its tab.
        """
        result = driver.dispatch(
            "region-selector.value",
            inputs={"url.search": "?region=NOT_A_REGION&persona=trader"},
        )

        assert result.ok
        assert result.value("persona-selector", "value") == "trader"


class TestWarmingGateThroughDispatch:
    """The #131 regression, tested at the layer that actually serves it."""

    def test_cold_redis_warms_rather_than_fetching_or_fabricating(
        self, driver: DashDriver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Redis-only web tier with nothing in Redis must report "warming".

        This is the guardrail CLAUDE.md is built around: the Cloud Run Service
        holds no models and must never fetch in the request path, so any
        fallback here either invents a series or blocks on EIA. #131 shipped a
        hardcoded ``MAPE 1.6%`` from exactly that path and #149 strict-gated
        it.

        ``test_callbacks_redis_only.py`` already pins the gate — but it has to
        stand up a second Dash app, scan ``callback_map`` for a function named
        ``load_data``, unwrap the decorator and call it as ``load_data("ERCOT",
        0)``. That reaches the function while stepping around the wiring: the
        positional call would keep passing if the registered ``Input`` order
        were reversed, which is the failure a user would see as the wrong
        region's data. Dispatching puts Dash in charge of the arguments.
        """
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", True)
        monkeypatch.setattr(cbs, "_load_data_from_redis", lambda region: None)

        def _never_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("the web tier fetched upstream during a warming state")

        monkeypatch.setattr("data.eia_client.fetch_demand", _never_called)
        monkeypatch.setattr("data.weather_client.fetch_weather", _never_called)

        result = driver.dispatch(
            "demand-store.data",
            inputs={"region-selector": "ERCOT", "refresh-interval": 0},
        )

        assert result.ok
        freshness = json.loads(result.value("data-freshness-store", "data"))
        assert freshness["demand"] == "warming"
        assert freshness["weather"] == "warming"

        demand = json.loads(result.value("demand-store", "data"))
        assert not demand.get("data"), "a warming web tier must not serve rows"

    def test_the_warming_assertion_is_not_vacuous(
        self, driver: DashDriver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the gate off, the same dispatch must NOT report warming.

        Otherwise the test above would pass on any callback that happened to
        return ``warming`` for unrelated reasons, and would keep passing if
        the gate were deleted outright.
        """
        import components.callbacks as cbs

        monkeypatch.setattr(cbs, "REQUIRE_REDIS", False)
        monkeypatch.setattr(cbs, "_load_data_from_redis", lambda region: None)

        result = driver.dispatch(
            "demand-store.data",
            inputs={"region-selector": "ERCOT", "refresh-interval": 0},
        )

        assert result.ok
        freshness = json.loads(result.value("data-freshness-store", "data"))
        assert freshness["demand"] != "warming"


class TestSerialisation:
    """Return values must survive the trip a browser would make."""

    def test_callback_responses_are_json_encodable(self, driver: DashDriver) -> None:
        """Plotly figures and component trees fail here, not in the browser.

        A ``numpy.float64`` or a ``Timestamp`` left in a return value raises
        during Dash's serialisation. The unit tier never sees it, because the
        helper returns the object and the assertion inspects it in Python.
        """
        result = driver.dispatch(
            "models-leaderboard.children",
            inputs={"region-selector": "CISO", "dashboard-tabs": "tab-models"},
        )
        assert result.ok
        json.dumps(result.payload)  # raises if anything slipped through


class TestOverviewInsightIsRoleAwareThroughDispatch:
    """#523 through the real callback, not the helper.

    The unit tests call ``_build_overview_insight`` directly. This tier is
    what proves the wiring: the callback now parses ``weather-store`` and
    passes it on, so an argument-order mistake or a store the callback cannot
    read is visible here and nowhere else.
    """

    @staticmethod
    def _stores() -> tuple[str, str]:
        import numpy as np
        import pandas as pd

        rng = pd.date_range("2026-08-05", periods=168, freq="h", tz="UTC")
        h = np.arange(168)
        demand = pd.DataFrame(
            {"timestamp": rng, "demand_mw": 20000 + 4000 * np.sin(2 * np.pi * (h - 10) / 24)}
        )
        weather = pd.DataFrame(
            {
                "timestamp": rng,
                "temperature_2m": 70 + 15 * np.sin(2 * np.pi * (h - 10) / 24),
                "wind_speed_80m": np.linspace(4, 26, 168),
                "shortwave_radiation": np.clip(600 * np.sin(2 * np.pi * h / 24), 0, None),
            }
        )
        # Exactly how the store writer serialises them (callbacks.py:254).
        return demand.to_json(date_format="iso"), weather.to_json(date_format="iso")

    def _insight_text(self, driver: DashDriver, persona: str, weather_json: str | None) -> str:
        demand_json, wj = self._stores()
        res = driver.dispatch(
            "overview-insight-card.children",
            inputs={
                "demand-store": demand_json,
                "dashboard-tabs": "tab-overview",
                "persona-selector": persona,
            },
            state={
                "weather-store": wj if weather_json is None else weather_json,
                "region-selector": "PJM",
                "data-freshness-store": None,
            },
        )
        assert res.ok, f"dispatch failed: {res.status}"
        return res.text("overview-insight-card")

    def test_personas_get_different_insight_text_through_dispatch(self, driver: DashDriver) -> None:
        bodies = {
            p: self._insight_text(driver, p, None)
            for p in ("grid_ops", "renewables", "trader", "data_scientist")
        }
        assert all(bodies.values()), "callback returned an empty insight card"
        assert len(set(bodies.values())) == 4, (
            "the persona must change the card body through the real callback, "
            f"not only in a direct helper call: { {k: v[:50] for k, v in bodies.items()} }"
        )

    def test_malformed_weather_store_does_not_blank_the_overview(self, driver: DashDriver) -> None:
        """Weather is an enrichment; a bad payload must cost the sentence, not the page.

        The callback's outer handler returns an error div for all five
        outputs, so an unguarded ``read_json`` here would blank the whole
        Overview over one optional line.
        """
        text = self._insight_text(driver, "renewables", "{not valid json")
        assert "Demand is" in text
        assert "Error" not in text and "error" not in text
