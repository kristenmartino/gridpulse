"""Data-morph transitions on the demand charts (`_layout(transition=True)`).

Plotly transitions are *declared* in Python and *gated* on the client
(assets/motion.js), because the two conditions that make a morph wrong —
prefers-reduced-motion, and a change in a trace's point count — are only
knowable in the browser. These tests pin the Python half of that contract:
which figures ask for a morph, and what they ask for.

The client half is pinned by `TestClientGateContract`, which asserts the
gate exists rather than re-implementing it — a JS unit runner is out of
scope for this suite, so the test guards against the gate being deleted
while the Python side keeps emitting `layout.transition` (which would ship
motion that ignores the media query).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from components._callbacks_shared import CHART_TRANSITION, _layout

MOTION_JS = Path(__file__).resolve().parents[2] / "assets" / "motion.js"


class TestLayoutTransitionOptIn:
    def test_absent_by_default(self):
        """Opt-in, not global: the 51 small multiples and every other chart
        that never asked for a morph must keep hard-cutting."""
        assert "transition" not in _layout()
        assert "transition" not in _layout(uirevision="DUK")

    def test_present_when_opted_in(self):
        layout = _layout(uirevision="DUK", transition=True)
        assert layout["transition"] == {"duration": 400, "easing": "exp-out"}

    def test_returns_a_copy_not_the_shared_constant(self):
        """A callsite mutating its own layout must not rewrite the module-level
        constant for every other chart in the process."""
        layout = _layout(transition=True)
        layout["transition"]["duration"] = 9999
        assert CHART_TRANSITION["duration"] == 400
        assert _layout(transition=True)["transition"]["duration"] == 400

    def test_transition_does_not_disturb_the_axis_merge(self):
        """Regression: `transition` is a reserved kwarg, not an override — it
        must not leak into the xaxis/yaxis deep-merge or drop shared styling."""
        layout = _layout(transition=True, xaxis=dict(title="t"))
        assert layout["xaxis"]["title"] == "t"
        assert "gridcolor" in layout["xaxis"]  # shared tone survived

    def test_easing_is_a_valid_plotly_enum_member(self):
        """Plotly easing is a d3 easing NAME; a CSS cubic-bezier (or a typo)
        raises at figure-build time. Pin that what we emit actually applies —
        an invalid easing would take the whole chart down, not degrade."""
        fig = go.Figure()
        fig.update_layout(**_layout(transition=True))
        assert fig.layout.transition.easing == "exp-out"
        assert fig.layout.transition.duration == 400

    def test_css_cubic_bezier_is_rejected_by_plotly(self):
        """The reason `exp-out` is used instead of the stylesheet's
        --ease-out-quint verbatim. Documents the constraint so a future edit
        doesn't "fix" the easing back to a CSS curve."""
        with pytest.raises(ValueError):
            go.Figure().update_layout(
                transition=dict(duration=400, easing="cubic-bezier(0.22, 1, 0.36, 1)")
            )


class TestChartsThatOptIn:
    """The two charts the morph is scoped to. Both are demand-over-time views
    whose consecutive states are the same series measured differently."""

    def test_overview_hero_requests_a_transition(self):
        import components._callbacks_overview as ov

        src = Path(ov.__file__).read_text()
        hero = src[src.index("def _build_overview_hero_chart") :]
        hero = hero[: hero.index("\ndef ")]
        assert "transition=True" in hero

    def test_forecast_chart_requests_a_transition(self):
        import components._callbacks_forecast as fc

        assert "transition=True" in Path(fc.__file__).read_text()

    def test_us_grid_small_multiples_do_not(self):
        """Frame-budget guard: 51 concurrent d3 path interpolations per refresh
        is unmeasured on real hardware, so the small multiples stay off."""
        import components._callbacks_us_grid as ug

        assert "transition=True" not in Path(ug.__file__).read_text()


class TestClientGateContract:
    """assets/motion.js is the only thing standing between a Python-declared
    transition and motion that ignores prefers-reduced-motion. These pin its
    existence, not its implementation."""

    @pytest.fixture(scope="class")
    def js(self) -> str:
        return MOTION_JS.read_text()

    def test_patches_plotly_react(self, js):
        assert "P.react = function" in js
        assert "__gpMotionPatched" in js

    def test_the_gate_is_actually_installed_on_startup(self, js):
        """Defining the gate is not the same as running it. Shipped once with
        `watchForPlotly` defined but never called: motion.js loaded, the
        `gp-motion` class applied, and every transition still played straight
        through the reduced-motion preference. Caught only by checking
        `Plotly.__gpMotionPatched` in the live app."""
        body = js[js.index("function start()") :]
        body = body[: body.index("\n    }")]
        assert "watchForPlotly()" in body

    def test_gate_installs_even_under_reduced_motion(self, js):
        """The gate must NOT sit behind an `if (!reduced())` guard the way the
        decorative effects do — under reduced motion, suppressing the
        transition is precisely what it is there to do."""
        body = js[js.index("function start()") :]
        body = body[: body.index("\n    }")]
        install = body[: body.index("watchForPlotly()")]
        assert "reduced()" not in install

    def test_gate_consults_reduced_motion(self, js):
        """The whole reason the gate exists: a CSS media query cannot stop a
        d3/rAF-driven Plotly transition."""
        gate = js[js.index("P.react = function") :][:1200]
        assert "reduced()" in gate

    def test_gate_consults_point_count(self, js):
        gate = js[js.index("P.react = function") :][:1200]
        assert "shapeMatches" in gate

    def test_suppression_zeroes_duration(self, js):
        block = js[js.index("function suppressTransition") :][:700]
        assert "duration: 0" in block

    def test_suppression_copies_rather_than_mutates(self, js):
        """Dash retains the figure across renders; mutating it in place would
        permanently strip the transition after the first suppressed update."""
        block = js[js.index("function suppressTransition") :][:600]
        assert "out[k] = layout[k]" in block

    def test_handles_the_dash_figure_object_signature(self, js):
        """Dash calls react(gd, {data, layout, frames, config}) — the object
        form. A patch that only understood (gd, data, layout) would read
        `layout` as undefined and silently never suppress anything."""
        gate = js[js.index("P.react = function") :][:1400]
        assert "isFigureObj" in gate
        assert "dataOrFigure.layout" in gate

    def test_gate_never_breaks_rendering(self, js):
        """The gate wraps EVERY figure update in the app. If it throws, no chart
        renders — so it must fall through to plotly on any error."""
        gate = js[js.index("P.react = function") :][:1600]
        assert "catch (e)" in gate
        assert "original.apply(this, arguments)" in gate
