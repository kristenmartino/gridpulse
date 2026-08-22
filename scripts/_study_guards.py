"""Self-check guards for study harnesses. Import these; call them early.

**Why this module exists.** Over one investigation (#231, 2026-08-21) five
study harnesses produced results that were later withdrawn. Not one defect was
caught by reading the code — every one was caught by *computing something a
second way and getting a different answer*, usually several rounds later,
after the wrong number had already reached a commit message or a GitHub
comment. The defects:

* A positional rolling-origin slice applied to frames of different lengths and
  start times, so row *i* meant a different calendar hour in each — the
  aggregate forecast was scored against actuals from a different week,
  inventing a 13.25% error where the true figure was 3.65%.
* An aggregate forecast built as ``F.sum(axis=0)``, coherent by construction,
  making four reconciliation arms mathematically identical no-ops.
* A control arm trained on an expanding slice (as little as 35 days against a
  nominal 90), turning a crippled baseline into "21/28 BAs improve".
* A local copy of a production fit path that could silently drift from the
  original.
* Load-weighting columns that must not be weighted (demand lags), because the
  column list was taken from a *featured* frame rather than the raw variables.

CLAUDE.md's own rule: *"Where a rule is fully mechanical, a guard enforces it
instead of a reminder."* These are the mechanical parts. Prose reminders were
already present and did not prevent any of the above.

**The convention: any derived metric a study reports gets computed a second
way, and the two must agree before results are printed.** Cheap — these run in
milliseconds against a harness that runs for an hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StudyGuardError(AssertionError):
    """A harness self-check failed. The results are not trustworthy."""


def assert_frames_aligned(frames: dict[str, pd.DataFrame], col: str = "timestamp") -> int:
    """Every frame must yield the SAME value of ``col`` at the same position.

    Guards the defect that voided the reconciliation study: several frames
    sliced by one positional index while having different lengths and start
    times. Returns the common length so the caller can use it for splits.

    Call this immediately after building the frames and BEFORE computing any
    rolling-origin split, then slice only the aligned frames.
    """
    if not frames:
        raise StudyGuardError("no frames supplied")
    lens = {name: len(df) for name, df in frames.items()}
    if len(set(lens.values())) != 1:
        raise StudyGuardError(f"frames have different lengths: {lens} — align before slicing")
    n = next(iter(lens.values()))
    if n == 0:
        raise StudyGuardError("frames are empty")
    ref_name, ref = next(iter(frames.items()))
    ref_ts = pd.DatetimeIndex(ref[col])
    for name, df in frames.items():
        ts = pd.DatetimeIndex(df[col])
        if not ts.equals(ref_ts):
            # Comparing two DatetimeIndex objects yields a plain ndarray, not a
            # pandas object — calling .to_numpy() here raised AttributeError and
            # the guard failed with the wrong exception type until a test built
            # from the real defect shape caught it.
            diff = np.asarray(ts != ref_ts)
            bad = int(diff.sum())
            first = int(np.argmax(diff))
            raise StudyGuardError(
                f"{name} and {ref_name} disagree on {col} at {bad} of {n} positions; "
                f"first at index {first}: {ts[first]} vs {ref_ts[first]}"
            )
    return n


def assert_agrees(
    name: str, a: float, b: float, *, rtol: float = 0.05, a_label: str = "a", b_label: str = "b"
) -> None:
    """Two independent computations of the SAME quantity must agree.

    The single check that would have caught the alignment bug on its first run
    instead of its fifth: the study's internal aggregate error (13.25%) against
    the same model measured standalone (3.65%).
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise StudyGuardError(f"{name}: non-finite ({a_label}={a}, {b_label}={b})")
    denom = max(abs(a), abs(b), 1e-12)
    if abs(a - b) / denom > rtol:
        raise StudyGuardError(
            f"{name}: {a_label}={a:.4g} vs {b_label}={b:.4g} differ by "
            f"{abs(a - b) / denom:.1%} (> {rtol:.0%}) — one of them is wrong"
        )


def assert_arms_differ(name: str, arms: dict[str, np.ndarray], *, atol: float = 1e-9) -> None:
    """Treatment arms must not be byte-identical to the control.

    Guards the no-op defect: an aggregate built as the sum of its parts is
    already coherent, so every reconciliation arm returned the identical
    result and the study reported four methods that were one method.
    """
    items = list(arms.items())
    if len(items) < 2:
        return
    ctl_name, ctl = items[0]
    identical = [
        n for n, v in items[1:] if v.shape == ctl.shape and np.allclose(v, ctl, atol=atol, rtol=0)
    ]
    if len(identical) == len(items) - 1:
        raise StudyGuardError(
            f"{name}: every arm {identical} is identical to control '{ctl_name}' — "
            f"the treatment is a no-op; check the arms actually differ before scoring"
        )


def assert_fixed_window(name: str, train_slices: list[slice], expected_rows: int) -> None:
    """A fixed-window control must be the SAME size in every window.

    Guards the strawman control: ``rolling_origin_splits`` returns an
    *expanding* train slice by default, so a nominal 90-day control trained on
    as little as 35 days in older windows.
    """
    sizes = {s.stop - s.start for s in train_slices}
    if sizes != {expected_rows}:
        raise StudyGuardError(
            f"{name}: train window varies across windows — sizes {sorted(sizes)}, "
            f"expected all {expected_rows}. An expanding slice is not a fixed control."
        )


def assert_plausible(name: str, value: float, lo: float, hi: float) -> None:
    """Bound a derived metric to a physically sensible range.

    An implausible number is a suspect instrument, not a finding. Two forecasts
    of the same quantity from the same data disagreeing by 24% was reported as
    a property of the hierarchy three times before it was recognised as a bug.
    """
    if not np.isfinite(value) or not (lo <= value <= hi):
        raise StudyGuardError(
            f"{name}={value:.4g} outside plausible [{lo:g}, {hi:g}] — "
            f"treat as a broken instrument until explained, not as a result"
        )
