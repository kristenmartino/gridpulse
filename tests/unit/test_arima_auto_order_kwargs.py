"""#297 — a kwarg that does nothing, and raises nothing while doing it.

``pm.auto_arima`` accepts ``**fit_args``. So ``exogenous=exog`` — the
pmdarima 1.x spelling of what 2.x calls ``X`` — was accepted, swallowed, and
silently ignored. The stepwise (p,q,P,Q) search ran on a **univariate** view
of demand while the final ``SARIMAX(..., exog=exog)`` fit used all five
weather regressors, so every selected order was chosen for a model we do not
fit. Nothing failed. Nothing logged. It survived a version bump.

A test asserting ``X=`` is passed would only pin the one kwarg that already
bit us. These pin the *class*: every keyword in the call must be a real
parameter of the installed pmdarima, so the next rename cannot hide the same
way.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pmdarima as pm

SOURCE = Path(__file__).resolve().parents[2] / "models" / "arima_model.py"


def _auto_arima_calls() -> list[ast.Call]:
    tree = ast.parse(SOURCE.read_text())
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "auto_arima"
    ]


class TestNoDeadKwargs:
    def test_every_kwarg_is_a_real_parameter_of_the_installed_pmdarima(self):
        """The general guard. Checked against the *installed* signature, so a
        future pmdarima that renames another parameter fails here rather than
        degrading the order search in silence."""
        params = set(inspect.signature(pm.auto_arima).parameters)
        calls = _auto_arima_calls()
        assert calls, "no auto_arima call found — did the module move?"
        for call in calls:
            passed = [kw.arg for kw in call.keywords if kw.arg]
            dead = sorted(k for k in passed if k not in params)
            assert not dead, (
                f"{dead} reach auto_arima's **fit_args and are silently "
                f"ignored. This is #297: the call succeeds and the option "
                f"does nothing."
            )

    def test_auto_arima_really_does_swallow_unknown_kwargs(self):
        """The premise, asserted rather than assumed.

        If pmdarima ever starts rejecting unknown keywords, the guard above
        becomes redundant — and this test is what tells us, instead of the
        guard quietly protecting against nothing.
        """
        sig = inspect.signature(pm.auto_arima)
        assert any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()), (
            "auto_arima no longer takes **kwargs — unknown keywords now raise, "
            "so the dead-kwarg class of bug is gone and this guard can go too."
        )

    def test_the_search_is_univariate_deliberately_not_accidentally(self):
        """The measured decision, pinned.

        `exogenous` (pmdarima 1.x) must never come back — it is the dead kwarg
        that made this univariate by accident. But `X` must not appear either:
        passing the regressors was measured WORSE on every major ISO
        (docs/ARIMA_ORDER_EXOG_STUDY.md — PJM 9.18→18.22, CAISO 5.18→12.42
        sMAPE). If someone "fixes" this by adding `X=`, they are reintroducing
        a regression this study already paid for, and this test is the note
        they will read.
        """
        for call in _auto_arima_calls():
            passed = {kw.arg for kw in call.keywords if kw.arg}
            assert "exogenous" not in passed, "pmdarima 1.x spelling — silently ignored"
            assert "X" not in passed, (
                "the order search is univariate ON PURPOSE; see "
                "docs/ARIMA_ORDER_EXOG_STUDY.md before changing this"
            )

    def test_the_final_fit_still_uses_the_regressors(self):
        """The half that was never broken must stay unbroken.

        Univariate *order selection* is the decision; a univariate *model*
        would be a different and much worse thing.
        """
        tree = ast.parse(SOURCE.read_text())
        sarimax = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "SARIMAX"
        ]
        assert sarimax, "no SARIMAX construction found"
        for call in sarimax:
            assert "exog" in {kw.arg for kw in call.keywords if kw.arg}, (
                "every SARIMAX fit must pass exog — the weather regressors are "
                "in the model even though the order search does not see them"
            )

    def test_the_296_integration_pins_survive(self):
        """`d`/`D` are why #296's long-horizon degeneracy stopped.

        They are real pmdarima parameters — unlike `exogenous` — so they were
        never dead. Pinned here because this PR changes the same call, and
        d + D <= 1 is what keeps SARIMAX from extrapolating a local weather
        trend as a permanent line.
        """
        for call in _auto_arima_calls():
            kw = {k.arg: k.value for k in call.keywords if k.arg}
            assert ast.literal_eval(kw["d"]) == 0
            assert ast.literal_eval(kw["D"]) == 1
            assert ast.literal_eval(kw["max_d"]) == 0
