"""#170: the drift log must describe the number users actually see, and name it.

`drift_updated` reported figures from `models_with_records[0]` — the alphabetical
first model, i.e. always `arima`, typically the weakest and the least-weighted.
The Overview headline (`_resolve_forecast_mape`) and the visibility gate both
read the **ensemble**, so the live log could not confirm the served number:
during the PR-G9 verification LDWP's arima drift read 188% MAPE while the
ensemble's was unobservable from outside the VPC.

The deeper defect was not which model it picked — it was that the line never said
which model it was describing, so the figure could not be interpreted at all.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from jobs import phases


def _call(previous: dict, demand: pd.DataFrame):
    with (
        patch("data.redis_client.redis_set"),
        patch("jobs.phases._read_window_strict", return_value=None),
        patch("jobs.phases.log") as recorder,
    ):
        phases.write_drift_metrics(region="ERCOT", previous_forecast=previous, demand_df=demand)
    calls = [c for c in recorder.info.call_args_list if c.args[0] == "drift_updated"]
    assert calls, f"expected drift_updated; got {[c.args[0] for c in recorder.info.call_args_list]}"
    return calls[0].kwargs


#: ERCOT-scale demand. The magnitudes matter: PR-G9 (#142) added a
#: region-relative low-actual filter, and a 100 MW "ERCOT" hour is excluded by
#: it, leaving the rolling means at None. A fixture that trips a real filter
#: tests the filter, not the thing under test.
ACTUAL_MW = 40_000.0


def _fixture(models: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    # Recent, not fixed: the rolling means are windowed against wall-clock now
    # (`_within_window`), so a hard-coded date falls outside 7d/30d and every
    # figure comes back None — which would make the value assertions below pass
    # vacuously against a broken headline.
    ts = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    row = {"timestamp": ts.isoformat(), **models}
    return (
        {"region": "ERCOT", "forecasts": [row]},
        pd.DataFrame({"timestamp": [ts], "demand_mw": [ACTUAL_MW]}),
    )


class TestTheHeadlineIsTheServedNumber:
    def test_it_reports_the_ensemble_not_the_alphabetical_first(self) -> None:
        """arima sorts first; the ensemble is what ships. Report the ensemble.

        The fixture makes them differ on purpose: arima is 50% wrong and the
        ensemble is exact, so a log describing arima cannot be mistaken for one
        describing the ensemble.
        """
        previous, demand = _fixture({"arima": 60_000.0, "xgboost": 40_000.0, "ensemble": 40_000.0})
        kw = _call(previous, demand)

        assert kw["headline_model"] == "ensemble"
        # The ensemble predicted exactly; arima was 50% out. A non-zero figure
        # here would mean the alphabetical model leaked back in.
        assert kw["rolling_mape_7d"] == 0.0

    def test_the_line_names_the_model_it_describes(self) -> None:
        """The defect underneath #170.

        A drift figure with no model attached cannot be interpreted — it was
        read as the headline for months while being arima's.
        """
        previous, demand = _fixture({"arima": 60_000.0, "ensemble": 40_000.0})
        assert _call(previous, demand)["headline_model"] == "ensemble"

    def test_a_ba_with_no_ensemble_falls_back_and_says_so(self) -> None:
        """Single-model BAs still get a figure, correctly labelled.

        Falling back silently would recreate the original bug for exactly the
        regions least able to afford it.
        """
        previous, demand = _fixture({"arima": 60_000.0, "xgboost": 40_000.0})
        kw = _call(previous, demand)

        assert kw["headline_model"] == "arima"
        assert kw["rolling_mape_7d"] == 50.0


class TestPerModelDepth:
    def test_every_model_reports_its_own_record_count(self) -> None:
        """#512's depth signal, completed.

        That field read the alphabetical sample too — the same defect one field
        over. Models need not agree: one absent from a forecast row grades no
        record that tick and its window runs shallower.
        """
        previous, demand = _fixture({"arima": 60_000.0, "xgboost": 40_000.0, "ensemble": 40_000.0})
        kw = _call(previous, demand)

        assert set(kw["n_records_by_model"]) == {"arima", "xgboost", "ensemble"}
        assert all(v == 1 for v in kw["n_records_by_model"].values())

    def test_the_headline_n_records_is_the_headline_models(self) -> None:
        previous, demand = _fixture({"arima": 60_000.0, "ensemble": 40_000.0})
        kw = _call(previous, demand)
        assert kw["n_records"] == kw["n_records_by_model"]["ensemble"]


class TestTheOldFieldsAreGone:
    def test_no_sample_prefixed_fields_remain(self) -> None:
        """Renamed rather than added beside.

        Keeping `sample_rolling_mape_7d` next to `rolling_mape_7d` would leave
        two similarly-named figures for different models in one line — a worse
        version of the ambiguity this fixes. Verified first that nothing outside
        the function consumed them.
        """
        previous, demand = _fixture({"arima": 60_000.0, "ensemble": 40_000.0})
        kw = _call(previous, demand)
        assert not [k for k in kw if k.startswith("sample_")], sorted(kw)
