"""Differential tests for the tuned row build in ``recursive_autoregressive_forecast``.

That function is the documented single source of truth for multi-step inference —
production scoring, the ADR-010 serve-path gate and holdout evaluation all go
through it (#195/#186). So the property under test is not "the forecast still
looks reasonable"; it is that **the exact frame handed to ``predict_fn`` on every
step is byte-identical**, dtypes included, to what the pre-optimisation
implementation produced.

Two behaviours the fast path can silently break, both pinned below:

1. **The dtype trap.** ``row[col] = <float>`` REPLACED the column and implicitly
   upcast an int one to float64. Positional ``.iloc`` assignment writes into the
   existing block and raises ``TypeError: Invalid value ... for dtype 'int64'``
   instead. This is reachable in production: the tail of
   ``_build_future_feature_frame`` does ``future_df[col] = 0`` for any feature it
   could not fill, which creates an int64 column.
2. **``.ffill().bfill()`` were no-ops.** On a one-row frame there is no
   neighbouring row to fill from, so the chain only ever did ``.fillna(0)``.
   Asserted directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.feature_engineering import (
    compute_autoregressive_snapshot,
    get_feature_names,
    recursive_autoregressive_forecast,
)


def _reference(model, seed_demand, future_df, predict_fn):
    """The pre-optimisation implementation, verbatim, as the oracle."""
    history = [float(v) for v in seed_demand if v is not None and not pd.isna(v) and v > 0]
    preds = []
    for i in range(len(future_df)):
        row = future_df.iloc[[i]].copy()
        for col, val in compute_autoregressive_snapshot(history).items():
            if col in row.columns:
                row[col] = val
        row = row.ffill().bfill().fillna(0)
        pred = float(predict_fn(model, row)[0])
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def _recorder():
    """A predict_fn that keeps every frame it was handed, and whose output
    depends on the autoregressive features so divergence propagates."""
    seen: list[pd.DataFrame] = []

    def fn(_model, row):
        seen.append(row.copy())
        v = row["demand_lag_1h"].iloc[0] if "demand_lag_1h" in row.columns else 100.0
        return [float(v) * 0.99 + 7.0]

    return fn, seen


def _assert_identical(future_df, seed, label):
    f_old, rows_old = _recorder()
    f_new, rows_new = _recorder()
    p_old = _reference(None, seed, future_df, f_old)
    p_new = recursive_autoregressive_forecast(None, seed, future_df, f_new)

    assert len(rows_old) == len(rows_new) == len(future_df)
    for i, (a, b) in enumerate(zip(rows_old, rows_new, strict=True)):
        pd.testing.assert_frame_equal(
            a, b, check_exact=True, check_dtype=True, obj=f"{label} row {i}"
        )
    np.testing.assert_array_equal(p_old, p_new)


def _prod_frame(rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-08-01", periods=rows, freq="h")
    return pd.DataFrame(
        {"timestamp": ts, **{c: rng.normal(0, 1, rows) for c in get_feature_names()}}
    )


SEED = [900.0 + i * 0.5 for i in range(720)]


class TestFrameHandedToPredictIsUnchanged:
    def test_production_shape(self):
        _assert_identical(_prod_frame(384), SEED, "production")

    def test_no_autoregressive_columns_present(self):
        """ar_positions is empty — the assignment must be skipped, not crash."""
        rng = np.random.default_rng(1)
        _assert_identical(pd.DataFrame({"temperature_f": rng.normal(70, 5, 24)}), SEED, "no-ar")

    def test_short_seed_history_produces_nan_lags(self):
        _assert_identical(_prod_frame(48), [900.0, 905.0, 910.0], "short-seed")

    def test_empty_seed_history(self):
        _assert_identical(_prod_frame(24), [], "empty-seed")

    def test_preexisting_nans_in_future_df(self):
        f = _prod_frame(48)
        f.loc[f.index[:5], get_feature_names()[2]] = np.nan
        _assert_identical(f, SEED, "nan-inputs")

    def test_int64_autoregressive_column(self):
        """THE DTYPE TRAP. Positional .iloc assignment raises on an int64 column
        where the old per-column replacement upcast it. Reachable in production
        via `future_df[col] = 0` in _build_future_feature_frame."""
        f = _prod_frame(24)
        f["demand_lag_1h"] = np.arange(24, dtype=np.int64)
        _assert_identical(f, SEED, "int64-ar")

    def test_int64_column_comes_back_as_float(self):
        """Pin the resolution, not just the absence of a crash: the column the
        model sees must be float64, which is what the old code produced too."""
        f = _prod_frame(24)
        f["demand_lag_1h"] = np.arange(24, dtype=np.int64)
        fn, seen = _recorder()
        recursive_autoregressive_forecast(None, SEED, f, fn)
        assert seen[0]["demand_lag_1h"].dtype == np.float64


class TestFillSemantics:
    def test_ffill_and_bfill_are_noops_on_one_row(self):
        """The justification for dropping them. If this ever fails, the fast
        path's `.fillna(0)` is no longer equivalent to the old chain."""
        row = _prod_frame(3).iloc[[1]].copy()
        row.iloc[0, row.columns.get_loc(get_feature_names()[0])] = np.nan
        pd.testing.assert_frame_equal(row.ffill().bfill(), row, check_exact=True)

    def test_nan_snapshot_values_become_zero(self):
        """An empty history makes every lag NaN; fillna(0) must still apply."""
        fn, seen = _recorder()
        recursive_autoregressive_forecast(None, [], _prod_frame(4), fn)
        assert seen[0]["demand_lag_168h"].iloc[0] == 0.0

    def test_caller_frame_is_not_mutated(self):
        """The int64 cast must not write through to the caller's DataFrame."""
        f = _prod_frame(24)
        f["demand_lag_1h"] = np.arange(24, dtype=np.int64)
        before = f["demand_lag_1h"].copy()
        fn, _ = _recorder()
        recursive_autoregressive_forecast(None, SEED, f, fn)
        assert f["demand_lag_1h"].dtype == np.int64
        pd.testing.assert_series_equal(f["demand_lag_1h"], before)
