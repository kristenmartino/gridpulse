"""Smoke test for application startup wiring.

Imports ``app.py`` end-to-end, runs callback registration, and asserts
the layout renders. Catches the class of failures that unit tests don't:
missing imports, signature mismatches in callback registrars, circular
imports, NameError at registration time, broken re-export shims.

Background: 2026-05-22 external code review flagged an apparent P0
startup-breaker (claiming ``register_us_grid_callbacks`` was called but
not imported in ``components/callbacks.py``). The claim was a false
positive — the import IS on line 1005 — but the bigger gap was real:
**no CI test imported the app**. The same class of failure (a real
missing import after a refactor, e.g. during issue #87's
in-progress callback decomposition) could ship undetected.

This smoke test closes that gap. It runs in <1s and exercises every
``register_*_callbacks`` invocation chain.

See PR-G1 / issue #143 for the campaign context.
"""

from __future__ import annotations

import pytest


class TestAppSmoke:
    """Application-startup smoke tests.

    Each test verifies a different aspect of the wiring rather than
    rolling them into one massive case — that way a regression in one
    area fails noisily while the others still report green.
    """

    def test_app_module_imports_cleanly(self):
        """``from app import app`` must not raise.

        Catches: missing imports, syntax errors, top-level NameErrors,
        circular import deadlocks.
        """
        # Lazy import inside the test so a failure here is reported with
        # the test name, not as a collection-time crash.
        from app import app  # noqa: F401

    def test_app_object_is_dash_instance(self):
        """The exported ``app`` must be a Dash application object."""
        import dash

        from app import app

        assert isinstance(app, dash.Dash), f"Expected app to be dash.Dash; got {type(app).__name__}"

    def test_app_layout_is_non_empty(self):
        """The layout must be set and have rendered children.

        Catches: a callback registrar that runs but assigns
        ``app.layout = None``, or a layout module that raises during
        import but is swallowed somewhere upstream.
        """
        from app import app

        layout = app.layout
        assert layout is not None, "app.layout is None"
        # Dash layouts can be either a component or a callable that
        # returns one. Either is fine; both must have content.
        if callable(layout):
            layout = layout()
        # A component with no children rendered is a strong signal
        # something went wrong during layout construction.
        children = getattr(layout, "children", None)
        assert children is not None, "app.layout has no children attribute"

    def test_callbacks_registered_without_error(self):
        """Every ``register_*_callbacks`` invocation chain in
        ``components.callbacks.register_callbacks`` completes without
        raising. This is the main regression target.

        ``register_callbacks`` is idempotent in practice because each
        registrar guards against duplicate registrations (Dash itself
        does too), so re-running it during a test is safe.
        """
        from app import app
        from components.callbacks import register_callbacks

        # If any registrar's import is missing, this raises NameError.
        # If a registrar's signature is mismatched, this raises TypeError.
        # If a registrar runs but fails, this raises whatever its
        # underlying exception is.
        try:
            register_callbacks(app)
        except Exception as exc:
            pytest.fail(
                f"register_callbacks raised {type(exc).__name__}: {exc}. "
                "This is the smoke-test signal that wiring is broken — "
                "check the most recent callback-module changes."
            )

    def test_known_callback_registrars_exist(self):
        """Sanity-check that the headline callback registrars are
        actually re-exported from ``components.callbacks``.

        Locks down the surface area against accidental re-export-shim
        regressions during the #87 decomposition. If a future refactor
        drops one of these names, this test fails immediately rather
        than at the app-startup point in CI.
        """
        from components import callbacks

        # The 5 top-level registrars that the app actually invokes —
        # one per visible tab in the shell (Overview, US Grid, Forecast,
        # Risk, Models). Drawn from
        # ``register_callbacks`` in ``components/callbacks.py``. Keep
        # in sync if new tabs are added or registrars are split during
        # the in-progress #87 decomposition.
        required = [
            "register_overview_callbacks",
            "register_us_grid_callbacks",
            "register_models_callbacks",
            "register_alerts_callbacks",
            "register_forecast_callbacks",
        ]
        missing = [name for name in required if not hasattr(callbacks, name)]
        assert not missing, (
            f"components.callbacks is missing these registrars: {missing}. "
            "Likely an import was dropped from the re-export shim."
        )


class TestLandingRouteRegistered:
    def test_about_route_on_the_real_server(self):
        """The landing blueprint is registered on the production Flask
        server — and the Dash index still owns /."""
        from app import server

        rules = {r.rule for r in server.url_map.iter_rules()}
        assert "/about" in rules
        assert "/" in rules  # Dash index untouched
