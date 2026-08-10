"""Convergence facts on the SARIMAX fit log.

``maxiter`` is the largest cost knob in the training job — SARIMAX is ~58% of
it and one fit is ~49s in production — and nothing recorded whether the
optimiser was actually using its 200-iteration budget. Any argument about the
right value was a guess.

Measured on a synthetic 2160-row series: the fit converges at **123
iterations**, ``maxiter=400`` is byte-identical to 200 (same iterations, same
llf, 0.0000% parameter difference), and dropping to 100 saves only 19% while
moving parameters by 88%. But that is synthetic. These fields answer the
question against the 102 real fits a production run performs.

The tests below are about the LOGGING being trustworthy, not about SARIMAX.
``mle_retvals`` is optimiser-dependent — different optimisers spell the
iteration count differently and some paths omit it entirely — so the extractor
must degrade to None rather than raise. A helper that throws inside a log call
would take down a fit that had already succeeded.
"""

from __future__ import annotations

import pytest

from models.arima_model import _convergence_fields


class _Result:
    def __init__(self, retvals):
        self.mle_retvals = retvals


class TestExtractsWhatStatsmodelsProvides:
    def test_converged_and_iterations(self):
        out = _convergence_fields(_Result({"converged": True, "iterations": 123}))
        assert out == {"converged": True, "iterations": 123}

    def test_nit_is_the_other_spelling(self):
        """Some statsmodels optimisers report `nit` instead of `iterations`.
        Missing this would silently log None for every fit on those paths."""
        out = _convergence_fields(_Result({"converged": False, "nit": 200}))
        assert out == {"converged": False, "iterations": 200}

    def test_iterations_wins_when_both_present(self):
        out = _convergence_fields(_Result({"converged": True, "iterations": 87, "nit": 999}))
        assert out["iterations"] == 87

    def test_a_capped_fit_is_distinguishable(self):
        """The reading that matters: converged False at exactly maxiter means
        the optimiser ran out of budget, which is a model-quality signal and
        not merely a slow fit."""
        out = _convergence_fields(_Result({"converged": False, "iterations": 200}))
        assert out["converged"] is False
        assert out["iterations"] == 200


class TestDegradesInsteadOfRaising:
    """A helper that throws inside a log call would fail a fit that succeeded."""

    @pytest.mark.parametrize(
        "obj",
        [
            _Result(None),
            _Result({}),
            _Result("not-a-dict"),
            _Result([("converged", True)]),
            object(),  # no mle_retvals attribute at all
        ],
    )
    def test_unusable_retvals_give_none(self, obj):
        assert _convergence_fields(obj) == {"converged": None, "iterations": None}

    def test_partial_retvals(self):
        assert _convergence_fields(_Result({"converged": True})) == {
            "converged": True,
            "iterations": None,
        }

    def test_return_shape_is_always_the_same_keys(self):
        """Structured logging wants a stable key set — a field that sometimes
        vanishes makes the log-based question unanswerable for those rows."""
        for obj in (_Result({"converged": True, "iterations": 1}), _Result({}), object()):
            assert set(_convergence_fields(obj)) == {"converged", "iterations"}
