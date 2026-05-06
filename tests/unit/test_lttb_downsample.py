"""Unit tests for ``data.preprocessing.lttb_downsample``.

Closes Phase F #32 of the craft-pass tracking issue. The downsampler is
used by the Models-tab residuals chart (where the residual series spans
60-90 days hourly = 1440-2160 points) and is intended to be safe to
apply to any series — input-dtype-preserving, no-op on short series,
guaranteed first/last endpoint preservation, deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestNoOpShortSeries:
    """LTTB MUST be a no-op when the input fits within the threshold —
    callers apply it unconditionally and rely on this contract."""

    def test_returns_input_unchanged_when_under_threshold(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(100, dtype=np.float64)
        y = np.sin(x / 10.0)
        x_out, y_out = lttb_downsample(x, y, threshold=720)

        assert np.array_equal(x_out, x)
        assert np.array_equal(y_out, y)

    def test_returns_input_unchanged_at_exact_threshold(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(720, dtype=np.float64)
        y = x * 2.0
        x_out, y_out = lttb_downsample(x, y, threshold=720)

        assert len(x_out) == 720
        assert np.array_equal(x_out, x)


class TestOutputShape:
    def test_output_length_matches_threshold(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(2160, dtype=np.float64)
        y = np.sin(x / 24.0)
        x_out, _ = lttb_downsample(x, y, threshold=720)
        # Algorithm guarantees exactly threshold output points (first +
        # threshold-2 buckets + last).
        assert len(x_out) == 720

    def test_first_and_last_points_preserved(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(2160, dtype=np.float64)
        y = np.cos(x / 24.0)
        x_out, y_out = lttb_downsample(x, y, threshold=720)

        # First and last must equal the input's first and last —
        # they anchor the chart's x-axis range.
        assert x_out[0] == x[0]
        assert y_out[0] == y[0]
        assert x_out[-1] == x[-1]
        assert y_out[-1] == y[-1]

    def test_output_x_strictly_increasing(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(2160, dtype=np.float64)
        y = np.random.RandomState(7).randn(2160)
        x_out, _ = lttb_downsample(x, y, threshold=720)

        diffs = np.diff(x_out)
        assert (diffs > 0).all(), "Output x must be strictly increasing"


class TestDtypePreservation:
    def test_datetime_x_preserved(self):
        from data.preprocessing import lttb_downsample

        x = np.array(
            np.datetime64("2026-01-01T00:00:00") + np.arange(2160) * np.timedelta64(1, "h"),
            dtype="datetime64[ns]",
        )
        y = np.sin(np.arange(2160) / 24.0)
        x_out, y_out = lttb_downsample(x, y, threshold=720)

        # datetime64 dtype survives the round-trip (helper indexes
        # back into the original array, doesn't coerce dtype).
        assert np.issubdtype(x_out.dtype, np.datetime64)
        assert x_out[0] == x[0]
        assert x_out[-1] == x[-1]

    def test_int_y_preserved(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(2160, dtype=np.float64)
        y = (np.sin(x / 24.0) * 1000).astype(np.int64)
        _, y_out = lttb_downsample(x, y, threshold=720)
        assert y_out.dtype == np.int64


class TestPeakPreservation:
    """The whole point of LTTB over uniform sampling: visually-significant
    extrema survive the downsample. Verifying directly is hard, but we
    can verify the algorithm picks points that include known peaks and
    excludes obvious noise."""

    def test_known_spike_survives(self):
        """Inject a single sharp spike in an otherwise smooth series.
        LTTB should pick it (it makes a huge triangle vs neighbours)."""
        from data.preprocessing import lttb_downsample

        n = 2000
        x = np.arange(n, dtype=np.float64)
        y = np.sin(x / 50.0).copy()
        # Single-point spike at index 1000
        y[1000] = 100.0

        _, y_out = lttb_downsample(x, y, threshold=720)
        # The spike value (or at least something very close) should
        # appear in the output — LTTB picks the peak in its bucket.
        assert y_out.max() > 50.0, "Expected the spike to survive downsampling"

    def test_global_min_max_within_50pct_of_input(self):
        """Output's range should approximate input's range — extrema
        are the highest-priority points for LTTB to keep."""
        from data.preprocessing import lttb_downsample

        rng = np.random.RandomState(7)
        x = np.arange(2160, dtype=np.float64)
        y = 50_000 + 10_000 * np.sin(x / 24.0) + rng.randn(2160) * 500

        _, y_out = lttb_downsample(x, y, threshold=720)
        # Output range covers most of input range (no perfect equality
        # because LTTB picks bucket-aware peaks, not strict extrema).
        assert y_out.max() >= y.max() - 1500
        assert y_out.min() <= y.min() + 1500


class TestDeterminism:
    """Same input → same output. LTTB has no randomness; this is a
    sanity check that our impl doesn't accidentally introduce any."""

    def test_identical_calls_match(self):
        from data.preprocessing import lttb_downsample

        x = np.arange(2160, dtype=np.float64)
        y = np.cos(x / 17.3)
        a_x, a_y = lttb_downsample(x, y, threshold=720)
        b_x, b_y = lttb_downsample(x, y, threshold=720)
        assert np.array_equal(a_x, b_x)
        assert np.array_equal(a_y, b_y)


class TestErrorHandling:
    def test_mismatched_lengths_raises(self):
        from data.preprocessing import lttb_downsample

        with pytest.raises(ValueError, match="same length"):
            lttb_downsample(np.arange(100), np.arange(99), threshold=50)

    def test_threshold_less_than_three_raises(self):
        from data.preprocessing import lttb_downsample

        with pytest.raises(ValueError, match="threshold"):
            lttb_downsample(np.arange(100), np.arange(100), threshold=2)


class TestRealWorldShape:
    """Smoke test against the actual Models-tab residual series shape:
    1500-2160 hourly points with a roughly normal residual distribution.
    We're verifying the helper produces sane output, not asserting on
    specific values — those are sensitive to the algorithm's internals."""

    def test_models_tab_residual_shape(self):
        from data.preprocessing import lttb_downsample

        rng = np.random.RandomState(42)
        n = 2160  # 90 days hourly — Models tab worst case
        x = np.array(
            np.datetime64("2026-02-01T00:00:00") + np.arange(n) * np.timedelta64(1, "h"),
            dtype="datetime64[ns]",
        )
        y = rng.normal(loc=0.0, scale=500.0, size=n)  # residuals in MW

        x_out, y_out = lttb_downsample(x, y, threshold=720)
        assert len(x_out) == 720
        assert np.issubdtype(x_out.dtype, np.datetime64)
        assert np.isfinite(y_out).all()
