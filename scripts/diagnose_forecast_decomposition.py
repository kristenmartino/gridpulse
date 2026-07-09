"""Quantify the DUK 30-day forecast pathology (Prophet negative demand + summer
decline) by reproducing the real pipeline and decomposing Prophet's components.

Answers two questions with numbers:
  (A) How much of Prophet's downward slope is TREND vs. WEATHER-REGRESSOR?
  (B) How cold is the (hour,dow) climatology baseline vs. current/seasonal temp?
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from jobs.phases import FORECAST_HORIZON_HOURS as H  # noqa: E402
from jobs.phases import (  # noqa: E402
    _build_future_feature_frame,
    engineer_region_features,
    fetch_region_data,
)
from models.prophet_model import PROPHET_REGRESSORS, train_prophet  # noqa: E402

REGION = sys.argv[1] if len(sys.argv) > 1 else "DUK"

OM = 384  # Open-Meteo forecast hours (day-16 boundary)

print(f"== DUK 30-day forecast diagnostic (horizon={H}h, climatology boundary={OM}h) ==\n")

# 1. Real data + features
data = fetch_region_data(REGION)
if data is None:
    raise SystemExit("fetch_region_data returned None (EIA/weather unavailable)")
featured = engineer_region_features(data)
print(
    f"history rows={len(featured)}  span={featured['timestamp'].min()} .. {featured['timestamp'].max()}"
)

# weather_df (historical + forecast) for the overlay
wx = None
try:
    from data.weather_client import fetch_weather

    wx = fetch_weather(REGION)
    print(f"weather_df rows={len(wx)} (Open-Meteo forecast overlay available)")
except Exception as e:  # noqa: BLE001
    print(f"weather fetch failed ({e}); climatology-only future frame")

start_ts = featured["timestamp"].max() + pd.Timedelta(hours=1)
future_df = _build_future_feature_frame(featured, H, weather_df=wx, start_ts=start_ts)

# 2. Train Prophet on real DUK history (same config as production)
model = train_prophet(featured)

# 3. Reconstruct Prophet's future frame the way predict_prophet does, but keep
#    ALL component columns from predict().
future = model.make_future_dataframe(periods=H, freq="h")
future["cap"] = getattr(model, "_demand_cap", 50000)
future["floor"] = 0
reg_cols = [n for n, _ in PROPHET_REGRESSORS if n in future_df.columns]
reg = pd.DataFrame({"ds": future_df["timestamp"].values})
for c in reg_cols:
    reg[c] = future_df[c].values
reg = reg.drop_duplicates("ds", keep="last")
future = future.merge(reg, on="ds", how="left")
for c in reg_cols:
    future[c] = future[c].ffill().bfill().fillna(0)
for n, _ in PROPHET_REGRESSORS:
    if n not in future.columns:
        future[n] = 0.0

fc = model.predict(future)
hz = fc.tail(H).reset_index(drop=True)  # the 720 horizon rows

yhat = hz["yhat"].to_numpy()
trend = hz["trend"].to_numpy()
regs = hz.get("extra_regressors_additive", pd.Series(np.zeros(H))).to_numpy()
seas = hz.get("additive_terms", pd.Series(np.zeros(H))).to_numpy() - regs  # seasonal only

# ---- (1) NEGATIVE DEMAND ----
neg = yhat < 0
print("\n--- (1) Negative-demand check (physical floor violated) ---")
print(f"yhat min={yhat.min():,.0f} MW  max={yhat.max():,.0f} MW")
print(
    f"hours with yhat<0: {neg.sum()} / {H}   first negative at hour {int(np.argmax(neg)) if neg.any() else 'n/a'}"
)
if neg.any():
    i = int(np.argmin(yhat))
    print(f"at the global min (hour {i}, ~day {i // 24}):")
    print(
        f"   yhat={yhat[i]:,.0f} = trend={trend[i]:,.0f} + seasonal={seas[i]:,.0f} + regressors={regs[i]:,.0f}"
    )
    print("   -> floor=0 bounds TREND (>=0) but seasonal+regressor punch the composite below 0")

# ---- (2) SLOPE ATTRIBUTION: trend vs regressor ----
print("\n--- (2) Downward-slope attribution (day 1 -> day 30) ---")
d1 = slice(0, 24)  # first day
d30 = slice(H - 24, H)  # last day
dtrend = trend[d30].mean() - trend[d1].mean()
dreg = regs[d30].mean() - regs[d1].mean()
dseas = seas[d30].mean() - seas[d1].mean()
dyhat = yhat[d30].mean() - yhat[d1].mean()
print(f"Δtrend      = {dtrend:+,.0f} MW")
print(f"Δregressors = {dreg:+,.0f} MW   <- weather reverting to climatology")
print(f"Δseasonal   = {dseas:+,.0f} MW")
print(
    f"Δyhat total = {dyhat:+,.0f} MW  (day-30 mean {yhat[d30].mean():,.0f} vs day-1 mean {yhat[d1].mean():,.0f})"
)
denom = abs(dtrend) + abs(dreg) + abs(dseas) or 1
print(
    f"share of decline: trend {abs(dtrend) / denom:.0%} | regressors {abs(dreg) / denom:.0%} | seasonal {abs(dseas) / denom:.0%}"
)

# ---- boundary step at day 16 (Open-Meteo -> climatology) ----
print("\n--- boundary step at hour 384 (day 16: Open-Meteo -> climatology) ---")
pre = slice(max(0, OM - 48), OM)  # 2 days before boundary
post = slice(OM, min(H, OM + 48))  # 2 days after
print(
    f"regressor contribution: pre-boundary mean {regs[pre].mean():+,.0f} -> post {regs[post].mean():+,.0f}  (step {regs[post].mean() - regs[pre].mean():+,.0f} MW)"
)
print(
    f"yhat:                   pre-boundary mean {yhat[pre].mean():,.0f} -> post {yhat[post].mean():,.0f}"
)

# ---- (B) CLIMATOLOGY TEMPERATURE COLD-BIAS ----
print("\n--- (B) Climatology temperature vs current/seasonal (the cold bias) ---")
if "temperature_2m" in future_df.columns:
    t = future_df["temperature_2m"].to_numpy()
    t_om = t[:OM].mean()  # days 1-16 (Open-Meteo forecast)
    t_clim = t[OM:].mean()  # days 17-30 (hour,dow climatology)
    print(
        f"future temperature_2m: Open-Meteo days1-16 mean {t_om:.1f}F  vs  climatology days17-30 mean {t_clim:.1f}F  (Δ {t_clim - t_om:+.1f}F)"
    )
hist_recent = featured[featured["timestamp"] >= featured["timestamp"].max() - pd.Timedelta(days=14)]
if "temperature_2m" in featured.columns:
    print(
        f"recent-14d actual temperature_2m mean {hist_recent['temperature_2m'].mean():.1f}F  (peak-summer level the climatology regresses away from)"
    )
if "cooling_degree_days" in future_df.columns:
    c = future_df["cooling_degree_days"].to_numpy()
    print(
        f"cooling_degree_days: days1-16 mean {c[:OM].mean():.2f}  vs  days17-30 climatology mean {c[OM:].mean():.2f}  (Δ {c[OM:].mean() - c[:OM].mean():+.2f})"
    )
    print(
        f"recent-14d actual CDD mean {hist_recent['cooling_degree_days'].mean():.2f}"
        if "cooling_degree_days" in featured.columns
        else ""
    )

print("\n== done ==")
