"""Differential tests for the vectorised climatology baseline (#389 follow-up).

``_build_future_feature_frame``'s climatology fill was ``for col in
numeric_cols: for i in range(horizon)`` — ~49 x 720 scalar ``.loc`` lookups into
a MultiIndex, per BA, per tick. Production sub-phase timing measured it at
**1,081.9s of summed worker time, 33.1% of the whole forecast phase** — its
single largest sub-step. A single-threaded local benchmark said 1.33s/call
(~68s fleet-wide) and was 16x optimistic, because the loop is pure Python and
holds the GIL: at 8 workers the concurrent regions serialise against each other.

These tests pin the two behaviours a naive vectorisation gets wrong, both of
which ``reindex`` alone cannot express:

1. A key **present** in ``group_means`` whose mean is NaN must stay NaN. Only a
   **missing** key falls back to ``last_row``. ``fillna`` would conflate them.
2. Column insertion order must match the loop's, so the resulting frame's
   column order is unchanged for every downstream consumer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.feature_engineering import compute_holiday_flag, get_feature_names
from jobs.phases import (
    _CLIMATOLOGY_MIN_ROWS,
    CLIMATOLOGY_WINDOW_DAYS,
    _build_future_feature_frame,
)

HORIZON = 720


def _featured(n: int, seed: int, nan_col: str | None = None, drop_hour: int | None = None):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-06-01", periods=n, freq="h")
    data = {"timestamp": ts, "demand_mw": rng.normal(3000, 200, n)}
    for c in get_feature_names():
        data[c] = rng.normal(0, 1, n)
    df = pd.DataFrame(data)
    if nan_col is not None:
        # one (hour, dow) group entirely NaN for this column -> its mean is NaN
        mask = (df["timestamp"].dt.hour == 3) & (df["timestamp"].dt.dayofweek == 2)
        df.loc[mask, nan_col] = np.nan
    if drop_hour is not None:
        # remove an hour entirely -> those (hour, dow) keys are ABSENT
        df = df[df["timestamp"].dt.hour != drop_hour].reset_index(drop=True)
    return df


def _reference_frame(featured: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The pre-vectorisation implementation, verbatim, as the oracle."""
    start_ts = featured["timestamp"].max() + pd.Timedelta(hours=1)
    fts = pd.date_range(start=start_ts, periods=horizon, freq="h")
    out = pd.DataFrame({"timestamp": fts})
    out["hour"] = fts.hour
    out["day_of_week"] = fts.dayofweek
    out["month"] = fts.month
    out["day_of_year"] = fts.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_holiday"] = compute_holiday_flag(out["timestamp"]).to_numpy()

    feature_cols = [c for c in featured.columns if c not in ("timestamp", "demand_mw", "region")]
    hist = featured.copy()
    cutoff = hist["timestamp"].max() - pd.Timedelta(days=CLIMATOLOGY_WINDOW_DAYS)
    recent = hist[hist["timestamp"] >= cutoff]
    if len(recent) >= _CLIMATOLOGY_MIN_ROWS:
        hist = recent.copy()
    hist["_hour"] = hist["timestamp"].dt.hour
    hist["_dow"] = hist["timestamp"].dt.dayofweek

    non_time = [c for c in feature_cols if c not in out.columns]
    numeric = [c for c in non_time if c in hist.columns]
    if numeric:
        group_means = hist.groupby(["_hour", "_dow"])[numeric].mean()
        fh, fd = out["timestamp"].dt.hour, out["timestamp"].dt.dayofweek
        last_row = featured.iloc[-1]
        for col in numeric:
            values = np.empty(horizon, dtype=float)
            for i in range(horizon):
                key = (fh.iloc[i], fd.iloc[i])
                if key in group_means.index:
                    values[i] = group_means.loc[key, col]
                else:
                    values[i] = float(last_row[col]) if col in last_row.index else 0.0
            out[col] = values
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0
    return out


class TestVectorisedClimatologyMatchesTheLoop:
    def test_dense_history(self):
        f = _featured(2160, seed=1)
        pd.testing.assert_frame_equal(
            _build_future_feature_frame(f, HORIZON), _reference_frame(f, HORIZON), check_exact=True
        )

    def test_nan_group_mean_stays_nan(self):
        """A PRESENT key whose mean is NaN must NOT fall back to last_row."""
        col = get_feature_names()[3]
        f = _featured(2160, seed=2, nan_col=col)
        new = _build_future_feature_frame(f, HORIZON)
        pd.testing.assert_frame_equal(new, _reference_frame(f, HORIZON), check_exact=True)
        # and the NaNs genuinely survived — otherwise this test proves nothing
        assert new[col].isna().any(), "expected NaN group mean to be preserved"

    def test_missing_keys_fall_back_to_last_row(self):
        """An ABSENT key must take last_row, which is the only fallback path."""
        f = _featured(2160, seed=3, drop_hour=5)
        new = _build_future_feature_frame(f, HORIZON)
        pd.testing.assert_frame_equal(new, _reference_frame(f, HORIZON), check_exact=True)
        col = get_feature_names()[0]
        at_missing = new.loc[new["timestamp"].dt.hour == 5, col]
        assert len(at_missing) > 0
        assert at_missing.notna().all(), "fallback must produce a value, not NaN"

    def test_both_conditions_on_short_history(self):
        f = _featured(400, seed=4, nan_col=get_feature_names()[7], drop_hour=5)
        pd.testing.assert_frame_equal(
            _build_future_feature_frame(f, HORIZON), _reference_frame(f, HORIZON), check_exact=True
        )

    def test_column_order_is_unchanged(self):
        f = _featured(2160, seed=5)
        assert list(_build_future_feature_frame(f, HORIZON).columns) == list(
            _reference_frame(f, HORIZON).columns
        )
