# Anchor-conditioning study (#309)
## Tier 1 — anchor-error proxy
| class | n_hours | placeholder_ties | prod_mean_err_pct | df_mean_err_pct | prod_median_err_pct | df_median_err_pct | df_win_rate | mean_margin_pct | verdict |
|---|---|---|---|---|---|---|---|---|---|
| broken | 103 | 2 | 58.19 | 14.45 | 64.72 | 6.75 | 0.901 | 43.73 | CONDITION |
| churn | 338 | 105 | 3.2 | 4.92 | 0.84 | 3.57 | 0.296 | -1.72 | SKIP |
| bulk | 573 | 50 | 2.56 | 9.03 | 0.51 | 4.86 | 0.149 | -6.47 | SKIP |
| clean | 263 | 6 | 0.0 | 7.01 | 0.0 | 4.05 | 0.004 | -7.01 | NEVER CONDITIONED (policy) |
| unknown | 288 | 5 | 0.25 | 9.91 | 0.0 | 4.5 | 0.011 | -9.66 | NEVER CONDITIONED (policy) |

## Verdicts
- **broken** → CONDITION
- **churn** → SKIP
- **bulk** → SKIP
- **clean** → NEVER CONDITIONED (policy)
- **unknown** → NEVER CONDITIONED (policy)

## Tier 2 — end-to-end model replay
| region | class | n_replays | as_seen_mape | conditioned_mape | delta |
|---|---|---|---|---|---|
| LDWP | broken | 12 | 16.43 | 14.26 | 2.18 |
| IID | broken | 13 | 28.18 | 26.73 | 1.45 |
| BPAT | churn | 12 | 1.98 | 1.45 | 0.53 |
| PSCO | bulk | 10 | 14.81 | 17.68 | -2.88 |
| PNM | unknown | 13 | 0.48 | 0.74 | -0.26 |