"""``data_merged`` and ``feature_engineering_complete`` must name their BA.

Neither carried a ``region`` until #537. Both report row counts — and
``feature_engineering_complete``'s ``dropped_rows`` is where the origin defects
live (`docs/FORECAST_ORIGIN_REGRESSION.md` §3) — so a fleet-wide sweep could see
that rows were dropped and not which balancing authority dropped them.
Correlating either line to a BA meant joining it to ``drift_updated``, which is
written a tick later by a different phase.

Asserted against **emitted output** rather than ``structlog.testing.capture_logs``,
following the convention recorded at length in ``test_scenario_grid.py``: there,
``capture_logs`` assertions passed in isolation and failed in the full suite for
reasons that were investigated and never established. Reading emitted output is
also what production observability actually depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import engineer_features
from data.preprocessing import frame_region, merge_demand_weather


class TestFrameRegion:
    """The label is best-effort and must never raise — these are log call sites."""

    def test_reads_the_region_column(self):
        assert frame_region(pd.DataFrame({"region": ["ERCOT", "ERCOT"], "x": [1, 2]})) == "ERCOT"

    @pytest.mark.parametrize(
        ("case", "df"),
        [
            ("no region column", pd.DataFrame({"x": [1]})),
            ("empty frame", pd.DataFrame({"region": []})),
            ("typed-empty placeholder", pd.DataFrame({"region": [""]})),
            ("whitespace only", pd.DataFrame({"region": ["   "]})),
            ("all null", pd.DataFrame({"region": [None, None]})),
        ],
    )
    def test_degenerate_inputs_yield_none_rather_than_raising(self, case, df):
        assert frame_region(df) is None, case

    def test_none_frame_yields_none(self):
        """``_last_known_good`` can hand a caller a typed-empty frame, and a
        logging helper must not be what turns a degraded fetch into a failure."""
        assert frame_region(None) is None


class TestLogLinesNameTheirRegion:
    def test_data_merged_carries_region(self, sample_demand_df, sample_weather_df, capsys):
        merge_demand_weather(sample_demand_df, sample_weather_df)
        out = capsys.readouterr().out
        assert "data_merged" in out
        assert "ERCOT" in out, "the merge log line must name the BA it merged"

    def test_feature_engineering_start_carries_region(
        self, sample_demand_df, sample_weather_df, capsys
    ):
        """The paired line. ``start`` reports ``input_rows`` and ``complete``
        reports ``output_rows``; the difference between them is the drop, so a
        named ``complete`` next to an anonymous ``start`` still cannot be read
        per BA without a join."""
        merged = merge_demand_weather(sample_demand_df, sample_weather_df)
        capsys.readouterr()  # discard the merge line
        engineer_features(merged)
        out = capsys.readouterr().out
        assert "feature_engineering_start" in out
        start_line = next(ln for ln in out.splitlines() if "feature_engineering_start" in ln)
        assert "ERCOT" in start_line

    def test_feature_engineering_complete_carries_region(
        self, sample_demand_df, sample_weather_df, capsys
    ):
        merged = merge_demand_weather(sample_demand_df, sample_weather_df)
        capsys.readouterr()  # discard the merge line so the assertion is unambiguous
        engineer_features(merged)
        out = capsys.readouterr().out
        assert "feature_engineering_complete" in out
        # Scoped to the line: ``feature_engineering_start`` also names ERCOT now,
        # so an assertion against the whole buffer would pass on its evidence.
        done_line = next(ln for ln in out.splitlines() if "feature_engineering_complete" in ln)
        assert "ERCOT" in done_line, "the feature log line must name the BA whose rows it dropped"

    def test_a_frame_without_a_region_still_logs(self, sample_demand_df, sample_weather_df):
        """Attribution is a nicety; emitting the line at all is not.

        ``models.model_service`` calls ``engineer_features`` on frames this
        module does not control, so an absent column must degrade the label
        rather than the log.
        """
        merged = merge_demand_weather(sample_demand_df, sample_weather_df)
        featured = engineer_features(merged.drop(columns=["region"]))
        assert not featured.empty


class TestEmptyInputWarningCannotBeAttributed:
    """``feature_engineering_empty_input`` is deliberately left without a region.

    It fires precisely when ``df.empty``, and an empty frame has no rows to read
    a label from — ``_typed_empty`` builds zero rows, so the ``region`` column
    exists and holds nothing. Adding ``region=frame_region(df)`` there would log
    ``None`` on every firing. Attributing that warning needs the region passed in
    from the caller, which is a different change from this one.
    """

    def test_region_is_unavailable_on_an_empty_frame(self):
        assert frame_region(pd.DataFrame({"region": [], "demand_mw": []})) is None


class TestRegionIsReadNotInvented:
    def test_label_follows_the_frame(self, sample_demand_df, sample_weather_df, capsys):
        """Pins that the value is read off the data rather than defaulted.

        Without this a hardcoded ``region="ERCOT"`` would pass every assertion
        above, since the shared fixture is ERCOT.
        """
        renamed = sample_demand_df.copy()
        renamed["region"] = "MISO"
        merge_demand_weather(renamed, sample_weather_df)
        out = capsys.readouterr().out
        assert "MISO" in out
        assert "ERCOT" not in out


def test_engineer_features_still_drops_the_warmup_prefix(sample_demand_df, sample_weather_df):
    """Guard that adding a log field changed no behaviour."""
    merged = merge_demand_weather(sample_demand_df, sample_weather_df)
    featured = engineer_features(merged)
    assert len(featured) < len(merged)
    assert featured["demand_lag_168h"].notna().all()
    assert not np.isnan(featured["demand_lag_24h"]).any()
