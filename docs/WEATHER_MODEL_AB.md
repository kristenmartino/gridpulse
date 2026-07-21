# Weather-model A/B study

Sample: ERCOT, MISO, PJM, CAISO, AZPS, BPAT, SEC, PNM · window 2026-05-07..2026-07-15 · 11 anchors/BA · arms: A_best_match (best_match), B_gfs_seamless (ncep_gfs_seamless), C_nbm (ncep_nbm_conus)

## Rung 0 — variable audit (empirical, 3 BAs)

| arm | model | vars_missing_everywhere | n_missing |
|---|---|---|---|
| A_best_match | best_match | soil_temperature_0cm | 1 |
| B_gfs_seamless | ncep_gfs_seamless | soil_temperature_0cm | 1 |
| C_nbm | ncep_nbm_conus | diffuse_radiation, direct_normal_irradiance, shortwave_radiation, soil_temperature_0cm, surface_pressure, wind_speed_120m | 6 |

API-wide gaps (all arms → identical climatology fallback, fair): `['soil_temperature_0cm']`; arm-specific fill sets: A_best_match: `(none)`, B_gfs_seamless: `(none)`, C_nbm: `['diffuse_radiation', 'direct_normal_irradiance', 'shortwave_radiation', 'surface_pressure', 'wind_speed_120m']`

## Tier 1 — weather RMSE vs ERA5 truth (pooled)

| arm | var | lead_d | rmse | bias | n |
|---|---|---|---|---|---|
| A_best_match | temperature_2m | 1 | 3.215 | -0.498 | 7872 |
| A_best_match | temperature_2m | 3 | 4.222 | -0.433 | 7872 |
| A_best_match | temperature_2m | 5 | 4.736 | -0.376 | 7872 |
| A_best_match | temperature_2m | 7 | 5.752 | -0.464 | 7872 |
| A_best_match | apparent_temperature | 1 | 3.664 | -1.206 | 7872 |
| A_best_match | apparent_temperature | 3 | 4.848 | -1.849 | 7872 |
| A_best_match | apparent_temperature | 5 | 5.273 | -1.721 | 7872 |
| A_best_match | apparent_temperature | 7 | 6.685 | -1.88 | 7872 |
| A_best_match | shortwave_radiation | 1 | 106.199 | 4.561 | 7872 |
| A_best_match | shortwave_radiation | 3 | 110.411 | 3.748 | 7872 |
| A_best_match | shortwave_radiation | 5 | 112.004 | 5.788 | 7872 |
| A_best_match | shortwave_radiation | 7 | 123.744 | 4.431 | 7872 |
| B_gfs_seamless | temperature_2m | 1 | 3.215 | -0.498 | 7872 |
| B_gfs_seamless | temperature_2m | 3 | 4.222 | -0.433 | 7872 |
| B_gfs_seamless | temperature_2m | 5 | 4.736 | -0.376 | 7872 |
| B_gfs_seamless | temperature_2m | 7 | 5.752 | -0.464 | 7872 |
| B_gfs_seamless | apparent_temperature | 1 | 3.664 | -1.206 | 7872 |
| B_gfs_seamless | apparent_temperature | 3 | 4.848 | -1.849 | 7872 |
| B_gfs_seamless | apparent_temperature | 5 | 5.273 | -1.721 | 7872 |
| B_gfs_seamless | apparent_temperature | 7 | 6.685 | -1.88 | 7872 |
| B_gfs_seamless | shortwave_radiation | 1 | 106.199 | 4.561 | 7872 |
| B_gfs_seamless | shortwave_radiation | 3 | 110.411 | 3.748 | 7872 |
| B_gfs_seamless | shortwave_radiation | 5 | 112.004 | 5.788 | 7872 |
| B_gfs_seamless | shortwave_radiation | 7 | 123.744 | 4.431 | 7872 |
| C_nbm | temperature_2m | 1 | 2.69 | 0.052 | 7872 |
| C_nbm | temperature_2m | 3 | 3.095 | 0.173 | 7872 |
| C_nbm | temperature_2m | 5 | 3.543 | 0.234 | 7872 |
| C_nbm | temperature_2m | 7 | 4.226 | 0.519 | 7872 |
| C_nbm | apparent_temperature | 1 | 2.946 | 0.125 | 7872 |
| C_nbm | apparent_temperature | 3 | 3.381 | 0.038 | 7872 |
| C_nbm | apparent_temperature | 5 | 3.921 | 0.036 | 7872 |
| C_nbm | apparent_temperature | 7 | 4.855 | 0.423 | 7872 |

**Tier-1 gate:** proceed to tier 2

## Tier 2 — B_gfs_seamless paired deltas (positive = beats control)

Mean Δ **+0.040** sMAPE pts over 264 paired (BA, anchor, bucket) cells; big wins (≥1.0 pt): 0 BA(s); worst BA -0.006.

| ba | mean_delta_pts |
|---|---|
| AZPS | 0.119 |
| BPAT | 0.001 |
| CAISO | 0.082 |
| ERCOT | -0.0 |
| MISO | 0.003 |
| PJM | -0.001 |
| PNM | -0.006 |
| SEC | 0.119 |

By bucket: `{'1-24h': 0.0, '25-72h': 0.001, '73-168h': 0.118}`

**Verdict: SKIP**

## Tier 2 — C_nbm paired deltas (positive = beats control)

Mean Δ **+0.921** sMAPE pts over 264 paired (BA, anchor, bucket) cells; big wins (≥1.0 pt): 2 BA(s); worst BA -0.325.

| ba | mean_delta_pts |
|---|---|
| AZPS | 3.699 |
| BPAT | -0.069 |
| CAISO | -0.071 |
| ERCOT | 0.426 |
| MISO | -0.325 |
| PJM | 0.909 |
| PNM | 0.914 |
| SEC | 1.879 |

By bucket: `{'1-24h': 0.518, '25-72h': 1.139, '73-168h': 1.104}`

**Verdict: ADOPT**
