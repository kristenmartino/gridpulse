"""
Feature engineering for energy demand forecasting.

Computes 25+ derived features from raw demand and weather data.
All features are documented in project1-expanded-spec.md §Derived Features.

Key conventions:
- Temperature in Fahrenheit (CDD/HDD baseline = 65°F)
- Wind speed in mph (converted to m/s internally for power calculation)
- All features are numeric, no NaN in final output
- Backward-looking windows only (no future data leakage)
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import (
    AIR_DENSITY_KG_M3,
    CDD_HDD_BASELINE_F,
    MPH_TO_MS,
    SOLAR_RATED_IRRADIANCE,
    WIND_CUTOUT_SPEED_MS,
)
from data.preprocessing import frame_region

log = structlog.get_logger()

AUTOREGRESSIVE_DEMAND_FEATURES = [
    "demand_lag_1h",
    "demand_lag_3h",
    "demand_lag_24h",
    "demand_lag_168h",
    "ramp_rate",
    "demand_momentum_short",
    "demand_momentum_long",
    "demand_ratio_24h",
    "demand_ratio_168h",
    "demand_roll_24h_mean",
    "demand_roll_24h_std",
    "demand_roll_24h_min",
    "demand_roll_24h_max",
    "demand_roll_72h_mean",
    "demand_roll_72h_std",
    "demand_roll_72h_min",
    "demand_roll_72h_max",
    "demand_roll_168h_mean",
    "demand_roll_168h_std",
    "demand_roll_168h_min",
    "demand_roll_168h_max",
]

# US federal holidays
try:
    import holidays as holidays_lib

    US_HOLIDAYS = holidays_lib.US()
except ImportError:
    US_HOLIDAYS = {}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features from a merged demand + weather DataFrame.

    Input must have columns: timestamp, demand_mw, and the 17 weather variables.
    Output adds 20+ new columns and drops rows with NaN.

    Args:
        df: Merged DataFrame from preprocessing.merge_demand_weather().

    Returns:
        Feature-engineered DataFrame ready for model training.
    """
    if df.empty:
        log.warning("feature_engineering_empty_input")
        return df

    df = df.copy()
    log.info("feature_engineering_start", region=frame_region(df), input_rows=len(df))

    df = engineer_exogenous_features(df)
    df = add_autoregressive_demand_features(df)

    # --- Interaction terms ---
    if "temperature_2m" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour"] = compute_temp_hour_interaction(df["temperature_2m"], df["hour_sin"])

    # --- Impute exogenous (weather) features, then drop only on the
    #     demand-derived autoregressive warm-up (#161, 2026-05-29) ---
    #
    # Previously this did ``dropna(subset=<all feature cols>)``, which
    # dropped a row if ANY of the ~50 features was NaN. That made a single
    # sparse exogenous column able to collapse the entire row set: in the
    # 2026-05-29 incident Open-Meteo's /forecast endpoint degraded its
    # historical coverage and ``soil_temperature_0cm`` arrived non-null for
    # only ~100 of ~2100 rows — dragging every region below the 168-row
    # model threshold and taking down forecasts nationwide.
    #
    # A weather provider dropping one variable's coverage should degrade
    # that feature, not zero out all forecasts. So: impute the exogenous
    # (weather, raw + derived) columns — ffill → bfill → 0, the same
    # NaN-defense Prophet/SARIMAX already apply to their regressors at
    # predict time — and ``dropna`` only on the demand-derived
    # autoregressive features, whose sole legitimate NaN is the lag/rolling
    # warm-up prefix we genuinely want gone.
    feature_cols = _get_feature_columns(df)
    autoregressive = [c for c in AUTOREGRESSIVE_DEMAND_FEATURES if c in df.columns]
    impute_cols = [c for c in feature_cols if c not in autoregressive]
    if impute_cols:
        df[impute_cols] = df[impute_cols].ffill().bfill().fillna(0.0)

    initial_rows = len(df)
    drop_subset = autoregressive or feature_cols  # fall back if no AR cols present
    df = df.dropna(subset=drop_subset).reset_index(drop=True)
    dropped = initial_rows - len(df)

    log.info(
        "feature_engineering_complete",
        # #537: without this, the dropped-row count below could not be attributed
        # to a BA — and dropped rows are exactly where the origin defects live.
        region=frame_region(df),
        output_rows=len(df),
        dropped_rows=dropped,
        feature_count=len(feature_cols),
        imputed_exogenous_cols=len(impute_cols),
    )

    return df


def engineer_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features available at prediction time without demand target values."""
    df = df.copy()

    if "temperature_2m" in df.columns:
        df["cooling_degree_days"] = compute_cdd(df["temperature_2m"])
        df["heating_degree_days"] = compute_hdd(df["temperature_2m"])
        df["temperature_deviation"] = compute_temperature_deviation(df["temperature_2m"])
        _add_cooling_response_features(df)

    if "wind_speed_80m" in df.columns:
        df["wind_power_estimate"] = compute_wind_power(df["wind_speed_80m"])

    if "shortwave_radiation" in df.columns:
        df["solar_capacity_factor"] = compute_solar_capacity_factor(df["shortwave_radiation"])

    if "timestamp" in df.columns:
        df["hour_sin"], df["hour_cos"] = compute_cyclical_hour(df["timestamp"])
        df["dow_sin"], df["dow_cos"] = compute_cyclical_dow(df["timestamp"])
        df["is_holiday"] = compute_holiday_flag(df["timestamp"])

    if "temperature_2m" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour"] = compute_temp_hour_interaction(df["temperature_2m"], df["hour_sin"])

    return df


def add_autoregressive_demand_features(
    df: pd.DataFrame, target_col: str = "demand_mw"
) -> pd.DataFrame:
    """Add demand-derived lag/rolling features that require historical demand.

    All features are computed from rows STRICTLY BEFORE the current row —
    never including the current row's ``demand_mw`` value. This matches the
    inference-time behavior of ``compute_autoregressive_snapshot``, which
    sees only ``demand_history[:t]`` when computing features for time t.

    History (2026-05-20, PR-D / #134-followup): the previous version of
    this function leaked the current row's target into ``ramp_rate``
    (``demand[i] - demand[i-1]`` contains ``demand[i]``) and into the
    ``demand_roll_{N}h_*`` features (pandas' default trailing rolling
    window includes the current row). The model was trained on
    contaminated data and learned to over-rely on these features —
    ``demand_roll_24h_min`` was XGBoost's #2 feature for ERCOT per
    ``docs/BACKTEST_RESULTS.md``. At inference time the features were
    computed honestly via ``compute_autoregressive_snapshot``, creating a
    train/inference distribution shift that hurt live drift MAPE.

    The fix is a single ``.shift(1)`` before each rolling/diff: ensures
    ``demand_roll_24h_mean[i]`` aggregates over ``demand[i-24..i-1]`` (the
    24 hours BEFORE row i), exactly matching the inference snapshot.
    """
    df = df.copy()
    if target_col not in df.columns:
        return df

    df["demand_lag_1h"] = compute_lag(df[target_col], periods=1)
    df["demand_lag_3h"] = compute_lag(df[target_col], periods=3)
    df["demand_lag_24h"] = compute_lag(df[target_col], periods=24)
    df["demand_lag_168h"] = compute_lag(df[target_col], periods=168)

    # Pre-shifted series — every downstream rolling / diff reads from
    # this so the current row's demand is structurally invisible. Single
    # source of truth instead of repeating ``.shift(1)`` per feature.
    prior_demand = df[target_col].shift(1)

    # ramp_rate[i] = demand[i-1] - demand[i-2] — matches snapshot's
    # ``lag_1 - lag_2``. Before this fix: demand[i] - demand[i-1] leaked.
    df["ramp_rate"] = compute_ramp_rate(prior_demand)

    for window in [24, 72, 168]:
        prefix = f"demand_roll_{window}h"
        rolling = prior_demand.rolling(window=window, min_periods=1)
        df[f"{prefix}_mean"] = rolling.mean()
        df[f"{prefix}_std"] = rolling.std()
        df[f"{prefix}_min"] = rolling.min()
        df[f"{prefix}_max"] = rolling.max()

    df["demand_momentum_short"] = compute_demand_momentum(df["demand_lag_1h"], df["demand_lag_3h"])
    df["demand_momentum_long"] = compute_demand_momentum(df["demand_lag_1h"], df["demand_lag_24h"])
    df["demand_ratio_24h"] = compute_demand_ratio(df["demand_lag_24h"], df["demand_roll_24h_mean"])
    df["demand_ratio_168h"] = compute_demand_ratio(
        df["demand_lag_168h"], df["demand_roll_168h_mean"]
    )
    return df


def compute_autoregressive_snapshot(demand_history: list[float]) -> dict[str, float]:
    """Compute inference-time autoregressive features using only prior demand history."""

    def _lag(periods: int) -> float:
        return float(demand_history[-periods]) if len(demand_history) >= periods else np.nan

    def _roll(window: int, op: str) -> float:
        if not demand_history:
            return np.nan
        arr = np.array(demand_history[-window:], dtype=float)
        if op == "mean":
            return float(np.mean(arr))
        if op == "std":
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        if op == "min":
            return float(np.min(arr))
        return float(np.max(arr))

    lag_1 = _lag(1)
    lag_3 = _lag(3)
    lag_24 = _lag(24)
    lag_168 = _lag(168)
    lag_2 = _lag(2)
    roll_24_mean = _roll(24, "mean")
    roll_168_mean = _roll(168, "mean")

    return {
        "demand_lag_1h": lag_1,
        "demand_lag_3h": lag_3,
        "demand_lag_24h": lag_24,
        "demand_lag_168h": lag_168,
        "ramp_rate": lag_1 - lag_2 if not np.isnan(lag_1) and not np.isnan(lag_2) else np.nan,
        "demand_momentum_short": lag_1 - lag_3
        if not np.isnan(lag_1) and not np.isnan(lag_3)
        else np.nan,
        "demand_momentum_long": lag_1 - lag_24
        if not np.isnan(lag_1) and not np.isnan(lag_24)
        else np.nan,
        "demand_ratio_24h": lag_24 / max(roll_24_mean, 1.0)
        if not np.isnan(lag_24) and not np.isnan(roll_24_mean)
        else np.nan,
        "demand_ratio_168h": lag_168 / max(roll_168_mean, 1.0)
        if not np.isnan(lag_168) and not np.isnan(roll_168_mean)
        else np.nan,
        "demand_roll_24h_mean": roll_24_mean,
        "demand_roll_24h_std": _roll(24, "std"),
        "demand_roll_24h_min": _roll(24, "min"),
        "demand_roll_24h_max": _roll(24, "max"),
        "demand_roll_72h_mean": _roll(72, "mean"),
        "demand_roll_72h_std": _roll(72, "std"),
        "demand_roll_72h_min": _roll(72, "min"),
        "demand_roll_72h_max": _roll(72, "max"),
        "demand_roll_168h_mean": roll_168_mean,
        "demand_roll_168h_std": _roll(168, "std"),
        "demand_roll_168h_min": _roll(168, "min"),
        "demand_roll_168h_max": _roll(168, "max"),
    }


#: Longest hole an absent point lag may be linearly interpolated across.
#: Reused from ``data.preprocessing`` rather than chosen here — it is this
#: repo's existing judgement about how far demand can be interpolated safely.
ABSENT_HOUR_INTERPOLATION_MAX_H = 6

#: How many days back a long-hole lag may reach for the same clock hour.
#: Matches ``demand_lag_168h``'s own horizon, so this never reaches for context
#: the feature set does not already use.
ABSENT_HOUR_SEASONAL_LOOKBACK_DAYS = 7


class HourIndexedHistory:
    """Demand keyed by hour, stored densely with NaN for hours we do not have.

    This is the same shape the training path already works in — a continuous
    hourly grid where a gap is a NaN rather than an absent row — which is the
    point. ``add_autoregressive_demand_features`` shifts such a grid; this
    offsets into one. Sharing the *representation* is what makes the two paths
    mean the same thing, and it is cheaper than sharing a call: a window is a
    slice here, where a mapping would rebuild a list of up to 168 lookups on
    every step (measured 79x the positional path; this is 0.5x it).

    Holes are load-bearing. ``lag_24h`` at an hour we never observed must be
    NaN, not the nearest thing 24 entries back — that silent reach is #559.
    """

    __slots__ = ("_base", "_dropped_writes", "_values")

    def __init__(self, base: pd.Timestamp, values: np.ndarray) -> None:
        self._base = base
        self._values = values
        self._dropped_writes = 0

    @property
    def dropped_writes(self) -> int:
        """How many writes fell outside the array and were discarded (#624).

        Non-zero means the recursion's own later predictions did not land, so
        every rolling window that should have read them saw a hole instead.
        Counted rather than derived: the arithmetic relating this to the seed's
        tail gap is an off-by-one waiting to happen, and the whole point of the
        counter is to not have to trust it.
        """
        return self._dropped_writes

    @classmethod
    def build(
        cls, timestamps: Any, values: Any, *, extra_hours: int = 0
    ) -> "HourIndexedHistory | None":
        """From parallel timestamp/value sequences, or None if they do not line up.

        ``extra_hours`` reserves room past the last seed hour for predictions the
        recursion will write back.
        """
        try:
            stamps = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
            vals = np.asarray(list(values), dtype=float)
        except (TypeError, ValueError):
            return None
        if len(stamps) != len(vals) or len(stamps) == 0:
            return None

        base = stamps.min().floor("h")
        offsets = ((stamps - base).dt.total_seconds() // 3600).to_numpy(dtype=np.int64)
        size = int(offsets.max()) + 1 + max(0, extra_hours)
        arr = np.full(size, np.nan, dtype=float)
        # #129: a zero or NaN reading poisons the rolling windows, so it is not
        # history. Leaving it NaN is exactly "we do not have this hour".
        keep = np.isfinite(vals) & (vals > 0)
        arr[offsets[keep]] = vals[keep]
        return cls(base, arr)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[pd.Timestamp, float], *, extra_hours: int = 0
    ) -> "HourIndexedHistory | None":
        if not mapping:
            return None
        return cls.build(list(mapping.keys()), list(mapping.values()), extra_hours=extra_hours)

    def index_of(self, ts: pd.Timestamp) -> int:
        return int((pd.Timestamp(ts).tz_convert("UTC") - self._base).total_seconds() // 3600)

    def set(self, ts: pd.Timestamp, value: float) -> None:
        """Record ``value`` at ``ts``, counting the write if it falls outside.

        #624: ``build`` sizes the array from the last *present* seed hour, so a
        trailing gap leaves the recursion writing past the end. The bounds guard
        is correct — writing there would corrupt a neighbouring hour — but it
        used to be indistinguishable from a successful write. A dropped write is
        a counter, never a no-op.
        """
        i = self.index_of(ts)
        if 0 <= i < len(self._values):
            self._values[i] = value
        else:
            self._dropped_writes += 1

    def at(self, index: int) -> float:
        if index < 0 or index >= len(self._values):
            return np.nan
        return float(self._values[index])

    def lag(self, index: int) -> float:
        """Value at ``index``, imputed when we never observed that hour.

        A point lag has to hold *something*: the recursion cannot skip a step,
        and the row build zero-fills whatever is left NaN. Zero is the worst
        available answer — ``demand_lag_24h = 0 MW`` is the #129 poison the seed
        filter exists to exclude, and the injection study measured it firing on
        13% of forecast steps (22.6% on IID), which is the likeliest reason that
        study came back against its own hypothesis
        (``docs/POSITIONAL_LAG_INJECTION_STUDY.md``).

        So an absent hour is imputed, in two regimes chosen to match how demand
        actually behaves:

        1. **Short hole — linear interpolation** between the observations either
           side. The bound is ``MAX_INTERPOLATION_GAP_HOURS`` (6), reused from
           ``data.preprocessing`` rather than invented here, and it covers the
           dominant case: 25 of the 31 real gap runs measured across the fleet
           are a single hour.
        2. **Long hole — same clock hour, previous days.** Interpolating across
           a 16-hour hole would smooth over a diurnal cycle, so instead step back
           in 24-hour multiples and take the first hour we have. Demand's
           dominant structure is diurnal; this preserves phase exactly, which a
           nearest-neighbour fill does not.

        NaN only when neither regime finds anything — a frame that thin has
        bigger problems, and the caller still zero-fills it as before.

        **This is a serve-only estimate and is out of distribution by
        construction.** Training drops any row whose lag source was NaN
        (``engineer_features``), so the model never saw an imputed lag. The goal
        is not to be right about the missing hour, it is to keep the row close
        to the distribution the model was fit on — which a plausible value does
        and zero does not.
        """
        direct = self.at(index)
        if not np.isnan(direct):
            return direct

        # Bounded scan: if either edge of the hole is further than the
        # interpolation limit, the hole is too long for regime 1 by definition,
        # so there is no reason to keep looking.
        span = ABSENT_HOUR_INTERPOLATION_MAX_H
        lo = hi = None
        for step in range(1, span + 1):
            if lo is None and not np.isnan(self.at(index - step)):
                lo = index - step
            if hi is None and not np.isnan(self.at(index + step)):
                hi = index + step
            if lo is not None and hi is not None:
                break
        if lo is not None and hi is not None and (hi - lo - 1) <= span:
            lo_v, hi_v = self.at(lo), self.at(hi)
            return float(lo_v + (hi_v - lo_v) * (index - lo) / (hi - lo))

        for day in range(1, ABSENT_HOUR_SEASONAL_LOOKBACK_DAYS + 1):
            seasonal = self.at(index - 24 * day)
            if not np.isnan(seasonal):
                return float(seasonal)
        return np.nan

    def window(self, index: int, hours: int) -> np.ndarray:
        """The hours actually present in ``[index - hours, index)``."""
        lo = max(0, index - hours)
        hi = max(lo, min(index, len(self._values)))
        seg = self._values[lo:hi]
        return seg[np.isfinite(seg)]


def compute_temporal_autoregressive_snapshot(
    history: "HourIndexedHistory", now: pd.Timestamp
) -> dict[str, float]:
    """``compute_autoregressive_snapshot`` resolved by hour instead of position.

    Same key set, same arithmetic. The only difference is what "24 back" means:
    this reads the hour ``now - 24h`` and returns NaN when we do not have it,
    where the positional version counts back 24 surviving entries and silently
    reaches further whenever the frame has a hole.

    That difference is not hypothetical. ``engineer_features`` drops every row
    whose lag source was null, so ``featured`` carries real discontinuities —
    LGEE's ran 34 hours at a live origin, making ``demand_lag_168h`` read
    2026-08-10T01:00 for an origin of 2026-08-18T11:00. See #559 and
    ``docs/POSITIONAL_LAG_SEED_STUDY.md``.

    Rolling windows cover the hours actually present in ``[now - window, now)``
    — the temporal reading of the training side's ``min_periods=1``, which
    likewise tolerates a gap rather than reaching past the window to fill it.

    Args:
        history: Hour-indexed demand. Real demand for seed hours, the model's
            own predictions for hours already forecast.
        now: The hour being predicted. Lags are resolved relative to this.

    Returns:
        The same 21 keys ``compute_autoregressive_snapshot`` returns.
    """
    idx = history.index_of(now)

    def _lag(periods: int) -> float:
        # Imputed when absent — see ``HourIndexedHistory.lag``. Rolling windows
        # below deliberately do NOT impute: ``min_periods=1`` on the training
        # side skips a NaN inside the window rather than filling it, so
        # ``window()`` matching that is what keeps the two paths equivalent.
        return history.lag(idx - periods)

    def _roll(window: int, op: str) -> float:
        arr = history.window(idx, window)
        if arr.size == 0:
            return np.nan
        if op == "mean":
            return float(arr.mean())
        if op == "std":
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        if op == "min":
            return float(arr.min())
        return float(arr.max())

    lag_1 = _lag(1)
    lag_2 = _lag(2)
    lag_3 = _lag(3)
    lag_24 = _lag(24)
    lag_168 = _lag(168)
    roll_24_mean = _roll(24, "mean")
    roll_168_mean = _roll(168, "mean")

    return {
        "demand_lag_1h": lag_1,
        "demand_lag_3h": lag_3,
        "demand_lag_24h": lag_24,
        "demand_lag_168h": lag_168,
        "ramp_rate": lag_1 - lag_2 if not np.isnan(lag_1) and not np.isnan(lag_2) else np.nan,
        "demand_momentum_short": lag_1 - lag_3
        if not np.isnan(lag_1) and not np.isnan(lag_3)
        else np.nan,
        "demand_momentum_long": lag_1 - lag_24
        if not np.isnan(lag_1) and not np.isnan(lag_24)
        else np.nan,
        "demand_ratio_24h": lag_24 / max(roll_24_mean, 1.0)
        if not np.isnan(lag_24) and not np.isnan(roll_24_mean)
        else np.nan,
        "demand_ratio_168h": lag_168 / max(roll_168_mean, 1.0)
        if not np.isnan(lag_168) and not np.isnan(roll_168_mean)
        else np.nan,
        "demand_roll_24h_mean": roll_24_mean,
        "demand_roll_24h_std": _roll(24, "std"),
        "demand_roll_24h_min": _roll(24, "min"),
        "demand_roll_24h_max": _roll(24, "max"),
        "demand_roll_72h_mean": _roll(72, "mean"),
        "demand_roll_72h_std": _roll(72, "std"),
        "demand_roll_72h_min": _roll(72, "min"),
        "demand_roll_72h_max": _roll(72, "max"),
        "demand_roll_168h_mean": roll_168_mean,
        "demand_roll_168h_std": _roll(168, "std"),
        "demand_roll_168h_min": _roll(168, "min"),
        "demand_roll_168h_max": _roll(168, "max"),
    }


def positional_seed_matches_hours(
    seed_timestamps: Any, origin: pd.Timestamp, *, span: int = 168
) -> bool:
    """Whether positional indexing of the seed lands on the hours it intends to.

    This is the exact condition under which the positional and temporal
    recursions produce **byte-identical** forecasts, not a heuristic proxy for
    it. Use it to decide whether comparing the two arms can tell you anything.

    The recursion appends its own predictions at contiguous hours, so a lag that
    reaches into the *predicted* part is always aligned. The only thing that can
    misalign an index is a hole in the **seed's tail**. Every lag and rolling
    window reaches at most ``span`` entries back, so it is enough that the last
    ``span`` seed entries are contiguous hours ending at ``origin - 1h``.

    Returns ``True`` when the arms cannot differ (nothing to compare), ``False``
    when a hole inside the lookback means they will. Unknown or unusable input
    returns ``False`` — "assume they differ" costs a redundant comparison, where
    the opposite would silently drop a real observation.
    """
    try:
        stamps = pd.to_datetime(pd.Series(list(seed_timestamps)), utc=True)
        origin = pd.Timestamp(origin).tz_convert("UTC")
    except (TypeError, ValueError):
        return False
    if len(stamps) < span:
        return False
    tail = stamps.iloc[-span:]
    # Contiguous and ending where the forecast begins. Distinct sorted hours
    # spanning exactly ``span - 1`` hours cannot have a hole in them.
    return bool(
        tail.iloc[-1] == origin - pd.Timedelta(hours=1)
        and (tail.iloc[-1] - tail.iloc[0]) == pd.Timedelta(hours=span - 1)
    )


def seed_divergence_reason(
    seed_timestamps: Any, origin: pd.Timestamp, *, span: int = 168
) -> tuple[str, int | None]:
    """*Why* the two seed arms differ, not just whether — and by how much (#624).

    :func:`positional_seed_matches_hours` collapses two different situations
    into one ``False``, and they are not equally informative:

    * ``hole_in_lookback`` — the seed reaches ``origin - 1h`` but has a hole
      further back. The arms differ, ``build`` sizes the array correctly, and
      the recursion writes every step it takes. **This is clean evidence about
      temporal indexing.**
    * ``seed_tail_short`` — the seed stops before ``origin - 1h``. The arms
      differ *and* the array is under-sized, so the tail of the horizon is
      silently discarded (#624). An observation here is partly about that bug.

    Both arise from the same upstream cause — production seeds from the
    post-``dropna`` ``featured`` frame — which is why the second is not rare.
    Separating them lets a comparison be stratified rather than discarded.

    Returns:
        ``(reason, tail_gap_hours)``. ``reason`` is one of ``identical``,
        ``hole_in_lookback``, ``seed_tail_short``, ``unusable``.
        ``tail_gap_hours`` is how far short of ``origin - 1h`` the seed stops,
        and is ``None`` when unknown.
    """
    try:
        stamps = pd.to_datetime(pd.Series(list(seed_timestamps)), utc=True)
        origin = pd.Timestamp(origin).tz_convert("UTC")
    except (TypeError, ValueError):
        return "unusable", None
    if len(stamps) == 0:
        return "unusable", None

    gap = int((origin - pd.Timedelta(hours=1) - stamps.iloc[-1]).total_seconds() // 3600)
    if positional_seed_matches_hours(stamps, origin, span=span):
        return "identical", gap
    if len(stamps) < span:
        return "unusable", gap
    return ("seed_tail_short", gap) if gap > 0 else ("hole_in_lookback", gap)


def _temporal_seed_history(
    seed_demand: Any, seed_timestamps: Any, *, extra_hours: int = 0
) -> "HourIndexedHistory | None":
    """Build the hour-indexed seed, or None if temporal mode is unavailable.

    Fail-open by design: every caller that cannot supply timestamps, and every
    frame whose timestamps do not line up with its demand, falls back to the
    positional path unchanged. Flag-off behaviour is byte-identical.
    """
    if seed_timestamps is None:
        return None
    return HourIndexedHistory.build(seed_timestamps, seed_demand, extra_hours=extra_hours)


def _future_step_timestamps(future_df: pd.DataFrame) -> list[pd.Timestamp] | None:
    """Per-step hours for the forecast frame, or None if it carries none."""
    if "timestamp" not in future_df.columns:
        return None
    try:
        return list(pd.to_datetime(future_df["timestamp"], utc=True))
    except (TypeError, ValueError):
        return None


def _warn_dropped_writes(histories: "list[HourIndexedHistory] | None") -> int:
    """Log once per recursion if the temporal history discarded any write (#624).

    ``HourIndexedHistory.build`` reserves the recursion's room from the last
    *present* seed hour, so a trailing gap in the seed leaves the tail of the
    horizon writing past the end of the array. Those predictions are dropped and
    every rolling window that should have read them sees a hole instead.

    Emitted at WARNING because it is silent corruption of the arm's own state,
    not a degraded input. Not fixed here — see #624; the sizing fix is a
    signature change, and this is the half that is correct either way.
    """
    if not histories:
        return 0
    dropped = sum(h.dropped_writes for h in histories if h is not None)
    if dropped:
        log.warning(
            "temporal_seed_writes_dropped",
            dropped_writes=dropped,
            histories=len(histories),
        )
    return dropped


def recursive_autoregressive_forecast(
    model: Any,
    seed_demand: list[float] | np.ndarray | pd.Series,
    future_df: pd.DataFrame,
    predict_fn: Any,
    seed_timestamps: Any = None,
    force_temporal: bool | None = None,
) -> np.ndarray:
    """Multi-step forecast that chains its own predictions as autoregressive lags.

    This is the honest inference protocol: each step recomputes the
    autoregressive features (``compute_autoregressive_snapshot``) from the
    growing history of *predictions* (seeded by real demand), never from
    observed in-window actuals. It is the single source of truth for both
    production scoring (``jobs.phases``) and holdout evaluation
    (``jobs.training_job`` / ``models.training``) so that XGBoost's reported
    accuracy is commensurable with Prophet/SARIMAX's multi-step holdouts and
    matches what production actually serves (#194/#195; #186 parity lever).

    Args:
        model: A trained model accepted by ``predict_fn``.
        seed_demand: Real demand history strictly before the forecast window.
            Zero/NaN readings are filtered (a single 0 would poison the
            rolling-window lags — see #129).
        future_df: One row per forecast step, in order, carrying the non-
            autoregressive features (weather, calendar). Its length sets the
            horizon.
        predict_fn: ``predict_fn(model, single_row_df) -> array-like`` returning
            one prediction for the row.

    Returns:
        1D array of length ``len(future_df)``.
    """
    history: list[float] = [
        float(v) for v in seed_demand if v is not None and not pd.isna(v) and v > 0
    ]
    preds: list[float] = []

    # #559: resolve lags by hour rather than by position in `history`. Off by
    # default and fail-open — without seed timestamps, a future frame that
    # carries no `timestamp`, or a length mismatch, the positional path below
    # runs unchanged and byte-identical.
    from config import feature_enabled

    # ``force_temporal`` lets a caller pin the arm regardless of the flag — the
    # #559 shadow needs the treatment arm while production still serves the
    # control. None means "ask the flag", which is every production path.
    use_temporal = feature_enabled("temporal_ar_seed") if force_temporal is None else force_temporal

    temporal_history: HourIndexedHistory | None = None
    step_stamps: list[pd.Timestamp] | None = None
    if use_temporal:
        step_stamps = _future_step_timestamps(future_df)
        temporal_history = _temporal_seed_history(
            seed_demand, seed_timestamps, extra_hours=len(future_df) + 1
        )
        if temporal_history is None or step_stamps is None:
            temporal_history, step_stamps = None, None

    # Per-step row construction, tuned 2026-08-05. `predict_xgboost` is 49.4%
    # of the scoring job's forecast phase (888.8s of 1,636.3s,
    # `scoring_phase_rollup`), and ~41% of THIS loop is pandas, not the model:
    # 763 ms/BA serial splits ~317 ms row-building / ~328 ms predict, and of
    # that 317 ms roughly 279 ms was the per-column `row[col] = val` loop —
    # ~21 separate DataFrame __setitem__ calls per step, each one re-touching
    # the block manager. Two changes, no contract change:
    #
    #   1. One positional `.iloc` assignment instead of ~21 __setitem__ calls.
    #      Column positions are resolved once, on the first step; the snapshot's
    #      KEY SET is constant across steps (only its values move), so this is
    #      safe. `compute_autoregressive_snapshot` returns floats/NaN only, so
    #      assigning into the float feature columns cannot change a dtype.
    #   2. `.ffill().bfill()` dropped. On a ONE-ROW frame both are no-ops --
    #      they fill along axis 0 and there is no neighbouring row to fill
    #      from -- so the whole chain only ever did the `.fillna(0)`. This is
    #      asserted directly in tests rather than taken on trust.
    #
    # This function is the documented single source of truth for multi-step
    # inference, shared by production scoring, the ADR-010 serve-path gate and
    # holdout evaluation (#195/#186). Its observable behaviour -- the exact
    # frame handed to `predict_fn` on every step -- must not move. A
    # differential test captures every row from both implementations and
    # asserts byte-equality, because "the forecasts still look right" is not
    # evidence at this seam.
    # Resolve the autoregressive columns once. Only the snapshot's VALUES move
    # between steps; its key set is fixed, so one probe call settles it.
    cols = list(future_df.columns)
    if temporal_history is not None and step_stamps is not None:
        _probe = compute_temporal_autoregressive_snapshot(temporal_history, step_stamps[0])
    else:
        _probe = compute_autoregressive_snapshot(history)
    ar_keys = [k for k in _probe if k in cols]
    ar_positions: list[int] = []
    if ar_keys:
        # THE DTYPE TRAP, found by the differential test and not before it:
        # `row[col] = <float>` REPLACED the column and implicitly upcast an int
        # one to float64. Positional `.iloc` assignment instead writes INTO the
        # existing block and raises
        #   TypeError: Invalid value '902.51' for dtype 'int64'
        # This is reachable in production, not hypothetical: the tail of
        # `_build_future_feature_frame` does `future_df[col] = 0` for any
        # feature column it could not fill, which creates an int64 column. Cast
        # up front so the frame `predict_fn` receives is byte-identical to what
        # the old implementation produced — which was float64 either way.
        to_cast = {k: "float64" for k in ar_keys if future_df[k].dtype != np.float64}
        if to_cast:
            future_df = future_df.astype(to_cast)
        ar_positions = [cols.index(k) for k in ar_keys]

    for i in range(len(future_df)):
        row = future_df.iloc[[i]].copy()
        if temporal_history is not None and step_stamps is not None:
            snapshot = compute_temporal_autoregressive_snapshot(temporal_history, step_stamps[i])
        else:
            snapshot = compute_autoregressive_snapshot(history)
        if ar_positions:
            row.iloc[0, ar_positions] = [snapshot[k] for k in ar_keys]
        row = row.fillna(0)
        pred = float(predict_fn(model, row)[0])
        preds.append(pred)
        history.append(pred)
        if temporal_history is not None and step_stamps is not None:
            temporal_history.set(step_stamps[i], pred)
    _warn_dropped_writes([temporal_history] if temporal_history is not None else None)
    return np.asarray(preds, dtype=float)


def batched_recursive_autoregressive_forecast(
    model: Any,
    seed_demand: list[float] | np.ndarray | pd.Series,
    future_frames: list[pd.DataFrame],
    predict_fn: Any,
    seed_timestamps: Any = None,
) -> list[np.ndarray]:
    """``recursive_autoregressive_forecast`` for N frames that share a seed.

    Same protocol, same per-frame results — but one ``predict_fn`` call per
    STEP instead of one per step *per frame*. The scenario grid (#127) runs 80
    weather variants off a single history, so the single-frame helper issued
    1,920 single-row predicts per region against production's 384: five times
    the whole job's predict count, for a side panel. Measured at 2.7x tick
    runtime and reverted (#462). The variants differ only in weather, and
    their step-i rows can travel through the model together.

    The chaining is unchanged and stays per-frame: frame ``j`` appends its own
    prediction to its own history, so no scenario can see another's. Only the
    batching of the model call is shared.

    Parity with the single-frame helper is not asserted by inspection — it is
    a differential test (``test_scenario_grid_batching.py``) that runs both and
    compares byte-for-byte, because this is the seam where "the forecasts still
    look right" is not evidence.

    Args:
        model: A trained model accepted by ``predict_fn``.
        seed_demand: Real demand history strictly before the window. Filtered
            identically to the single-frame helper (#129).
        future_frames: N frames of EQUAL length, one per scenario.
        predict_fn: ``predict_fn(model, frame) -> array-like``, one prediction
            per row. Called with an N-row frame rather than a 1-row one.

    Returns:
        List of N arrays, each ``len(future_frames[0])`` long, in input order.
    """
    if not future_frames:
        return []

    horizon = len(future_frames[0])
    if any(len(f) != horizon for f in future_frames):
        raise ValueError("every scenario frame must have the same length")

    seed = [float(v) for v in seed_demand if v is not None and not pd.isna(v) and v > 0]
    n = len(future_frames)
    histories: list[list[float]] = [list(seed) for _ in range(n)]
    preds: list[list[float]] = [[] for _ in range(n)]

    # #559, mirroring the single-frame helper. Each scenario keeps its own
    # timestamp-keyed history for the same reason it keeps its own list: no
    # scenario may see another's predictions. Fail-open to positional.
    from config import feature_enabled

    temporal_histories: list[HourIndexedHistory] | None = None
    step_stamps: list[pd.Timestamp] | None = None
    if feature_enabled("temporal_ar_seed"):
        step_stamps = _future_step_timestamps(future_frames[0])
        if step_stamps is not None:
            temporal_histories = [
                _temporal_seed_history(seed_demand, seed_timestamps, extra_hours=horizon + 1)
                for _ in range(n)
            ]
            if any(h is None for h in temporal_histories):
                temporal_histories, step_stamps = None, None
        else:
            temporal_histories = None

    cols = list(future_frames[0].columns)
    if temporal_histories is not None and step_stamps is not None:
        _probe = compute_temporal_autoregressive_snapshot(temporal_histories[0], step_stamps[0])
    else:
        _probe = compute_autoregressive_snapshot(seed)
    ar_keys = [k for k in _probe if k in cols]

    # Stack STEP-MAJOR: all N scenarios for step 0, then step 1, and so on. A
    # step's rows are then CONTIGUOUS, so each iteration is a cheap positional
    # slice instead of a fancy-index gather. Measured: fancy-indexing made the
    # batched path slower than cell-at-a-time on pandas overhead alone, which
    # would have wiped out the point of batching.
    stacked = pd.concat(
        [f.iloc[[i]] for i in range(horizon) for f in future_frames], ignore_index=True
    )
    if ar_keys:
        # Same dtype trap as the single-frame helper: `_build_future_feature_frame`
        # can leave an int64 column, and positional assignment writes into the
        # existing block rather than replacing it.
        to_cast = {k: "float64" for k in ar_keys if stacked[k].dtype != np.float64}
        if to_cast:
            stacked = stacked.astype(to_cast)
    # Fill once up front rather than per step. The non-autoregressive columns
    # never change, and the autoregressive ones are overwritten every step from
    # the snapshot (whose own NaNs are filled in the same pass below).
    stacked = stacked.fillna(0)
    ar_positions = [cols.index(k) for k in ar_keys]
    block = np.empty((n, len(ar_keys)), dtype=float) if ar_keys else None

    for i in range(horizon):
        step_rows = stacked.iloc[i * n : (i + 1) * n]
        if ar_positions:
            for j in range(n):
                if temporal_histories is not None and step_stamps is not None:
                    snapshot = compute_temporal_autoregressive_snapshot(
                        temporal_histories[j], step_stamps[i]
                    )
                else:
                    snapshot = compute_autoregressive_snapshot(histories[j])
                block[j] = [snapshot[k] for k in ar_keys]
            np.nan_to_num(block, copy=False, nan=0.0)
            step_rows = step_rows.copy()
            step_rows.iloc[:, ar_positions] = block
        else:
            step_rows = step_rows.copy()

        out = np.asarray(predict_fn(model, step_rows), dtype=float)
        for j in range(n):
            value = float(out[j])
            preds[j].append(value)
            histories[j].append(value)
            if temporal_histories is not None and step_stamps is not None:
                temporal_histories[j].set(step_stamps[i], value)

    _warn_dropped_writes(temporal_histories)
    return [np.asarray(p, dtype=float) for p in preds]


# ---------------------------------------------------------------------------
# Individual feature functions (public, used by scenario engine)
# ---------------------------------------------------------------------------


def compute_cdd(temperature_f: pd.Series) -> pd.Series:
    """
    Cooling Degree Days: max(0, temp - 65°F).

    Standard HVAC demand proxy. Higher CDD = more AC load.

    Args:
        temperature_f: Temperature in Fahrenheit.

    Returns:
        CDD series.
    """
    return np.maximum(0, temperature_f - CDD_HDD_BASELINE_F)


#: Trailing windows for CDD accumulation. 24h captures overnight carry-over,
#: 72h a multi-day heat wave — the two timescales building thermal mass
#: actually operates on.
COOLING_ACCUM_WINDOWS_H = (24, 72)


def _add_cooling_response_features(df: pd.DataFrame) -> None:
    """Cooling-response features, in place, behind ``cooling_response_features``.

    Motivated by measurement, not intuition: `docs/ERROR_ANALYSIS.md` found the
    hottest temperature quintile carries a mean **34.7%** of our error against
    **11.9%** for the coldest, monotone in 7 of 8 BAs analysed. The existing
    representation of cooling load is a single linear ``cooling_degree_days``
    against a fixed 65°F baseline.

    Three things it cannot express, all built from weather variables we already
    fetch — so nothing here touches the fetch path:

    * **Accumulation** — thermal mass across consecutive hot hours/days.
    * **Convexity** — cooling load rises faster than linearly in CDD once
      plant is near capacity.
    * **Humidity** — latent load; 95°F at 70% RH is not 95°F at 20%.

    Flag-gated and default-off until the rolling-eval study says otherwise.
    """
    from config import feature_enabled

    if not feature_enabled("cooling_response_features"):
        return
    if "cooling_degree_days" not in df.columns:
        return

    cdd = df["cooling_degree_days"]
    for window in COOLING_ACCUM_WINDOWS_H:
        df[f"cdd_accum_{window}h"] = compute_cdd_accumulation(cdd, window)
    # Convexity. Squared rather than a learned spline: XGBoost can already
    # split on CDD, so what it lacks is not flexibility but a term that makes
    # the curvature cheap to express in few splits.
    df["cdd_squared"] = cdd.pow(2)
    if "relative_humidity_2m" in df.columns:
        df["heat_index"] = compute_heat_index(df["temperature_2m"], df["relative_humidity_2m"])
        # Latent load only bites when there is sensible load to add it to.
        df["cdd_x_humidity"] = cdd * (df["relative_humidity_2m"] / 100.0)


def compute_heat_index(temperature_f: pd.Series, relative_humidity_pct: pd.Series) -> pd.Series:
    """NWS heat index (Rothfusz regression), °F.

    Cooling load tracks what a building's occupants and its HVAC actually
    experience, which at high temperature is driven as much by humidity as by
    the dry-bulb reading: 95°F at 70% RH is a far larger air-conditioning load
    than 95°F at 20%. The model currently sees temperature and humidity as
    separate columns and has to learn the interaction from data.

    Below 80°F the regression is not valid and the NWS uses a simple average
    form instead; that branch is applied here rather than extrapolating a
    polynomial fitted for hot, humid conditions into cool ones.
    """
    t = pd.to_numeric(temperature_f, errors="coerce")
    r = pd.to_numeric(relative_humidity_pct, errors="coerce").clip(0, 100)

    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    full = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * r
        - 0.22475541 * t * r
        - 0.00683783 * t * t
        - 0.05481717 * r * r
        + 0.00122874 * t * t * r
        + 0.00085282 * t * r * r
        - 0.00000199 * t * t * r * r
    )
    return simple.where(t < 80.0, full)


def compute_cdd_accumulation(cdd: pd.Series, window_h: int) -> pd.Series:
    """Trailing mean CDD — building thermal mass, as a feature.

    The third consecutive 95°F day draws materially more cooling load than the
    first: structures soak heat overnight and start the next day warmer. A
    point-in-time CDD cannot express that, so the model sees two identical
    hours with very different true loads.

    Backward-looking only (``min_periods=1`` so early rows survive rather than
    poisoning the frame with NaN).
    """
    return pd.to_numeric(cdd, errors="coerce").rolling(window_h, min_periods=1).mean()


def compute_hdd(temperature_f: pd.Series) -> pd.Series:
    """
    Heating Degree Days: max(0, 65°F - temp).

    Standard winter heating demand proxy.

    Args:
        temperature_f: Temperature in Fahrenheit.

    Returns:
        HDD series.
    """
    return np.maximum(0, CDD_HDD_BASELINE_F - temperature_f)


def compute_temperature_deviation(temperature_f: pd.Series, window: int = 720) -> pd.Series:
    """
    Temperature deviation from 30-day (720-hour) rolling average.

    Unusual weather = unusual demand.

    Args:
        temperature_f: Temperature in Fahrenheit.
        window: Rolling window in hours (default: 720 = 30 days).

    Returns:
        Deviation series.
    """
    rolling_avg = temperature_f.rolling(window=window, min_periods=1).mean()
    return temperature_f - rolling_avg


def compute_wind_power(wind_speed_mph: pd.Series) -> pd.Series:
    """
    Simplified wind power estimate: 0.5 × ρ × A × v³.

    Converts mph → m/s internally. Applies cutout speed (25 m/s ≈ 56 mph).
    Above cutout, turbines shut down → power = 0.

    Args:
        wind_speed_mph: Wind speed in mph (from Open-Meteo).

    Returns:
        Normalized wind power estimate [0, 1].
    """
    # Convert mph to m/s
    v_ms = wind_speed_mph * MPH_TO_MS

    # Simplified power curve (normalized)
    # P = 0.5 * rho * A * v^3 — we normalize by rated conditions
    # Using v=12 m/s as rated speed (typical for modern turbines)
    rated_speed_ms = 12.0
    rated_power = 0.5 * AIR_DENSITY_KG_M3 * 1.0 * (rated_speed_ms**3)

    raw_power = 0.5 * AIR_DENSITY_KG_M3 * 1.0 * (v_ms**3)
    normalized = raw_power / rated_power

    # Apply cut-in (3 m/s) and cutout (25 m/s) speeds
    cut_in_ms = 3.0
    result = np.where(
        v_ms < cut_in_ms,
        0.0,
        np.where(v_ms > WIND_CUTOUT_SPEED_MS, 0.0, np.minimum(normalized, 1.0)),
    )

    return pd.Series(result, index=wind_speed_mph.index, dtype=float)


def compute_solar_capacity_factor(ghi: pd.Series) -> pd.Series:
    """
    Solar capacity factor: GHI / 1000 W/m², clipped to [0, 1].

    Solar panels rated at standard test conditions (1000 W/m²).

    Args:
        ghi: Global Horizontal Irradiance (shortwave_radiation) in W/m².

    Returns:
        Capacity factor [0, 1].
    """
    return np.clip(ghi / SOLAR_RATED_IRRADIANCE, 0.0, 1.0)


def compute_cyclical_hour(timestamps: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical sin/cos encoding for hour of day.

    hour=0 and hour=24 map to the same point on the unit circle.

    Returns:
        (hour_sin, hour_cos) tuple of Series.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        hour = timestamps.hour
        index = timestamps
    else:
        hour = timestamps.dt.hour
        index = timestamps.index
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return (
        pd.Series(hour_sin, index=index, dtype=float),
        pd.Series(hour_cos, index=index, dtype=float),
    )


def compute_cyclical_dow(timestamps: pd.Series | pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical sin/cos encoding for day of week.

    Monday=0, Sunday=6. Cyclical so Monday and next Monday are identical.

    Returns:
        (dow_sin, dow_cos) tuple of Series.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        dow = timestamps.dayofweek
        index = timestamps
    else:
        dow = timestamps.dt.dayofweek
        index = timestamps.index
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    return (
        pd.Series(dow_sin, index=index, dtype=float),
        pd.Series(dow_cos, index=index, dtype=float),
    )


def compute_holiday_flag(timestamps: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """
    Binary flag: 1 if US federal holiday, 0 otherwise.

    Uses the `holidays` library for accurate holiday detection.
    """
    # Handle both Series and DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
        ts_iter = timestamps
    else:
        index = timestamps.index
        ts_iter = timestamps
    return pd.Series(
        [1.0 if ts.date() in US_HOLIDAYS else 0.0 for ts in ts_iter],
        index=index,
        dtype=float,
    )


def compute_lag(series: pd.Series, periods: int) -> pd.Series:
    """
    Compute lag feature (shift by N periods).

    Uses positive shift so lag_24 = value from 24 hours ago.
    No future data leakage — lagged values only look backward.
    """
    return series.shift(periods)


def compute_ramp_rate(demand: pd.Series) -> pd.Series:
    """
    Ramp rate: demand_t - demand_t-1.

    Critical for grid operations — high ramp rates require fast-responding
    generation (gas peakers).

    Known demand [100, 120, 110] → ramp [NaN, 20, -10].
    """
    return demand.diff()


def compute_demand_momentum(recent_lag: pd.Series, older_lag: pd.Series) -> pd.Series:
    """
    Demand momentum: difference between recent and older lag.

    Captures whether demand is ramping up or down. Positive = increasing,
    negative = decreasing.

    Args:
        recent_lag: More recent demand lag (e.g., lag_1h).
        older_lag: Older demand lag (e.g., lag_3h or lag_24h).

    Returns:
        Momentum series (recent - older).
    """
    return recent_lag - older_lag


def compute_demand_ratio(lag: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    Demand deviation ratio: lag value normalized by rolling mean.

    Values > 1 indicate demand was above average, < 1 below average.
    Captures whether a given period was abnormal relative to its window.

    Args:
        lag: Demand lag value (e.g., lag_24h).
        rolling_mean: Rolling mean over same or longer window.

    Returns:
        Ratio series, clipped to avoid division by near-zero.
    """
    return lag / rolling_mean.clip(lower=1.0)


def compute_temp_hour_interaction(temperature: pd.Series, hour_sin: pd.Series) -> pd.Series:
    """
    Temperature × Hour interaction term.

    Captures the pattern that AC peaks in the afternoon (high temp × high hour_sin)
    while heating peaks in the evening.
    """
    return temperature * hour_sin


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric feature columns (excludes timestamp, metadata)."""
    exclude = {"timestamp", "region", "data_quality", "forecast_mw"}
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude]


def get_feature_names() -> list[str]:
    """
    Return the canonical list of feature names produced by engineer_features().

    Used by models for consistent feature ordering.
    """
    return [
        # Raw weather
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_10m",
        "wind_speed_80m",
        "wind_speed_120m",
        "wind_direction_10m",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
        "cloud_cover",
        "precipitation",
        "snowfall",
        "surface_pressure",
        "soil_temperature_0cm",
        "weather_code",
        # Derived
        "cooling_degree_days",
        "heating_degree_days",
        "temperature_deviation",
        # Cooling-response pack (flag-gated, #ERROR_ANALYSIS). Absent from the
        # frame when the flag is off, which the selector tolerates.
        "cdd_accum_24h",
        "cdd_accum_72h",
        "cdd_squared",
        "heat_index",
        "cdd_x_humidity",
        "wind_power_estimate",
        "solar_capacity_factor",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_holiday",
        "demand_lag_1h",
        "demand_lag_3h",
        "demand_lag_24h",
        "demand_lag_168h",
        "ramp_rate",
        "demand_momentum_short",
        "demand_momentum_long",
        "demand_ratio_24h",
        "demand_ratio_168h",
        "demand_roll_24h_mean",
        "demand_roll_24h_std",
        "demand_roll_24h_min",
        "demand_roll_24h_max",
        "demand_roll_72h_mean",
        "demand_roll_72h_std",
        "demand_roll_72h_min",
        "demand_roll_72h_max",
        "demand_roll_168h_mean",
        "demand_roll_168h_std",
        "demand_roll_168h_min",
        "demand_roll_168h_max",
        "temp_x_hour",
    ]


def get_autoregressive_feature_names() -> list[str]:
    """Return demand-derived autoregressive feature names."""
    return AUTOREGRESSIVE_DEMAND_FEATURES.copy()


def get_exogenous_feature_names() -> list[str]:
    """Return non-demand features available without contemporaneous target values."""
    return [f for f in get_feature_names() if f not in AUTOREGRESSIVE_DEMAND_FEATURES]
