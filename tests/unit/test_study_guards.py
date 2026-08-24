"""Each guard must actually TRIGGER on the defect it exists to catch.

CLAUDE.md: *"A guard needs a fixture that can trigger what it exists to
catch. A test suite that only confirms a guard runs and returns is not the
same as confirming it can detect its target defect."* So every test here
builds the real defect shape — reproduced from the #231 study failures — and
asserts the guard raises, alongside a clean case asserting it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _study_guards import (  # noqa: E402
    StudyGuardError,
    assert_agrees,
    assert_arms_differ,
    assert_fixed_window,
    assert_frames_aligned,
    assert_plausible,
)


def _frame(start: str, n: int) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="h", tz="UTC")})


class TestFramesAligned:
    def test_clean_frames_pass_and_return_length(self):
        f = {"a": _frame("2026-01-01", 10), "b": _frame("2026-01-01", 10)}
        assert assert_frames_aligned(f) == 10

    def test_different_lengths_raise(self):
        f = {"a": _frame("2026-01-01", 10), "b": _frame("2026-01-01", 8)}
        with pytest.raises(StudyGuardError, match="different lengths"):
            assert_frames_aligned(f)

    def test_the_real_defect_same_length_different_start(self):
        """The reconciliation bug: equal lengths, offset calendar hours.

        This is the shape that got through — a length check alone passes it.
        """
        f = {"ba": _frame("2025-07-20", 100), "agg": _frame("2025-07-27", 100)}
        with pytest.raises(StudyGuardError, match="disagree on timestamp"):
            assert_frames_aligned(f)

    def test_a_single_shifted_hour_is_caught(self):
        a = _frame("2026-01-01", 50)
        b = _frame("2026-01-01", 50)
        b.loc[30, "timestamp"] = b.loc[30, "timestamp"] + pd.Timedelta(hours=1)
        with pytest.raises(StudyGuardError, match="first at index 30"):
            assert_frames_aligned({"a": a, "b": b})


class TestAgrees:
    def test_close_values_pass(self):
        assert_agrees("agg wape", 3.65, 3.71)

    def test_the_real_defect_study_vs_standalone(self):
        """13.25% (in-study) vs 3.65% (standalone) for the same quantity."""
        with pytest.raises(StudyGuardError, match="one of them is wrong"):
            assert_agrees("agg wape", 13.25, 3.65, a_label="in-study", b_label="standalone")

    def test_nonfinite_raises(self):
        with pytest.raises(StudyGuardError, match="non-finite"):
            assert_agrees("x", float("nan"), 1.0)


class TestArmsDiffer:
    def test_differing_arms_pass(self):
        arms = {"base": np.array([1.0, 2.0]), "t": np.array([1.1, 2.2])}
        assert_arms_differ("wape", arms)

    def test_the_real_defect_every_arm_identical(self):
        """Aggregate built as sum-of-parts made all arms mathematically equal."""
        v = np.array([1.0, 2.0, 3.0])
        arms = {"base": v, "bottom_up": v.copy(), "ols": v.copy(), "mint": v.copy()}
        with pytest.raises(StudyGuardError, match="no-op"):
            assert_arms_differ("wape", arms)

    def test_one_identical_arm_is_allowed(self):
        """bottom_up legitimately equals base at the bottom level."""
        v = np.array([1.0, 2.0])
        arms = {"base": v, "bottom_up": v.copy(), "mint": np.array([1.5, 2.5])}
        assert_arms_differ("wape", arms)


class TestFixedWindow:
    def test_fixed_slices_pass(self):
        assert_fixed_window("ctl", [slice(0, 100), slice(50, 150)], 100)

    def test_the_real_defect_expanding_slice(self):
        """rolling_origin_splits' default: slice(0, test_start) grows each window."""
        expanding = [slice(0, 840), slice(0, 1200), slice(0, 2016)]
        with pytest.raises(StudyGuardError, match="not a fixed control"):
            assert_fixed_window("ctl", expanding, 2160)


class TestPlausible:
    def test_in_range_passes(self):
        assert_plausible("incoherence", 2.77, 0, 10)

    def test_the_real_defect_implausible_incoherence(self):
        """24.70% disagreement between two forecasts of the same quantity."""
        with pytest.raises(StudyGuardError, match="broken instrument"):
            assert_plausible("incoherence", 24.70, 0, 10)

    def test_nan_is_not_plausible(self):
        with pytest.raises(StudyGuardError):
            assert_plausible("x", float("nan"), 0, 10)
