"""`region` must ride on every log the training job emits, not just the ones
whose call site had the name in scope.

The gap this closes is concrete. #446 added ``converged`` and ``iterations`` to
``arima_trained``; the first production run carrying them (2026-08-11, 04:00
UTC) logged **102 fits, one of which did not converge** — and the event has no
``region``, so naming the BA meant correlating the failing fit's timestamp
against whichever neighbouring log *did* carry a region. That is archaeology,
not observability, and it degrades to a guess the moment two BAs interleave.

``train_arima`` takes no ``region`` parameter, and neither does ``train_prophet``
or ``train_xgboost``. Threading one into SARIMAX alone would fix a third of the
problem and leave the other two mute, so the fix binds ``region`` once for the
whole per-BA span instead. ``merge_contextvars`` had been sitting in the
processor chain unused since the logging config was written.

Four things have to hold for that to work, and each is pinned below:

1. The span actually binds it.
2. The binding is **reset** on exit — including when the body raises. A log
   emitted between regions that inherits the previous BA's name is worse than
   one with no name at all, because it is wrong rather than absent.
3. ``merge_contextvars`` is still in the configured processor chain. Remove it
   and the bind becomes a silent no-op: nothing fails, no test notices, and the
   fields simply stop appearing in production.
4. An explicit ``region=`` kwarg still wins. Roughly forty call sites already
   pass one; ``merge_contextvars`` uses ``setdefault``, so they are unaffected —
   but that is a property of structlog's implementation, not of ours, and it is
   what makes this change safe to land without touching them.
"""

from __future__ import annotations

import pytest
import structlog

from jobs import training_job
from observability import configure_logging


@pytest.fixture(autouse=True)
def _isolate_structlog():
    """Save and restore global structlog state — both the config and any
    context left behind — so these tests cannot leak into the rest of the run."""
    saved = structlog.get_config().copy()
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.configure(**saved)


def _emit_under(region: str, monkeypatch, *, boom: bool = False) -> dict:
    """Run ``_train_region`` with its body replaced by a probe that reports the
    context a log call would see at that depth."""
    seen: dict = {}

    def fake_inner(region_arg, force, t0, summary):
        seen.update(structlog.contextvars.get_contextvars())
        if boom:
            raise RuntimeError("training blew up")
        return summary

    monkeypatch.setattr(training_job, "_train_region_inner", fake_inner)
    if boom:
        with pytest.raises(RuntimeError):
            training_job._train_region(region)
    else:
        training_job._train_region(region)
    return seen


class TestTheSpanBindsRegion:
    def test_a_log_from_deep_inside_would_carry_region(self, monkeypatch):
        """``train_arima`` logs from three frames down with no region in scope.
        This is the property that makes `arima_trained` attributable."""
        assert _emit_under("NEVP", monkeypatch).get("region") == "NEVP"


class TestTheBindingIsReset:
    """A stale region is a *wrong* label, not a missing one — the failure mode
    worth a test."""

    def test_context_is_clear_after_a_normal_return(self, monkeypatch):
        _emit_under("NEVP", monkeypatch)
        assert "region" not in structlog.contextvars.get_contextvars()

    def test_context_is_clear_after_the_body_raises(self, monkeypatch):
        """Training a BA can and does throw; the next BA must not inherit."""
        _emit_under("NEVP", monkeypatch, boom=True)
        assert "region" not in structlog.contextvars.get_contextvars()

    def test_consecutive_regions_do_not_bleed(self, monkeypatch):
        """`run()` trains sequentially in one thread, so a missing reset would
        show up as every BA after the first being labelled correctly and every
        gap between them being labelled with the previous one."""
        assert _emit_under("NEVP", monkeypatch).get("region") == "NEVP"
        assert _emit_under("CISO", monkeypatch).get("region") == "CISO"


class TestTheProcessorChainStillMergesIt:
    """Without this the bind is a silent no-op."""

    def test_merge_contextvars_is_configured(self):
        configure_logging(json_output=True)
        assert structlog.contextvars.merge_contextvars in structlog.get_config()["processors"], (
            "merge_contextvars was dropped from the processor chain; every "
            "contextvars binding in the job is now silently discarded"
        )

    def test_bound_region_reaches_the_event_dict(self):
        """End-to-end through the real processor, with an event dict shaped
        like the one `arima_trained` actually emits."""
        with structlog.contextvars.bound_contextvars(region="NEVP"):
            out = structlog.contextvars.merge_contextvars(
                None, "info", {"event": "arima_trained", "converged": False, "iterations": 200}
            )
        assert out == {
            "event": "arima_trained",
            "converged": False,
            "iterations": 200,
            "region": "NEVP",
        }


class TestExistingExplicitKwargsAreUnaffected:
    def test_explicit_region_wins_over_the_bound_one(self):
        """~40 call sites already pass `region=`. `merge_contextvars` uses
        `setdefault`, so binding cannot clobber them — which is what makes this
        change safe without editing any of them."""
        with structlog.contextvars.bound_contextvars(region="NEVP"):
            out = structlog.contextvars.merge_contextvars(
                None, "info", {"event": "training_resume_declined", "region": "CISO"}
            )
        assert out["region"] == "CISO"
