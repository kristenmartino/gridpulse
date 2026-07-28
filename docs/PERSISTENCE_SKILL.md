# Forecast skill vs a naive baseline

Does the served forecast beat *yesterday, same hour*? Re-run `python scripts/persistence_skill.py` to refresh.

Baseline: seasonal-naive at a 24h lag — every value it uses is known a full day before the target hour, so it is a fair opponent for a 24h-lead forecast. Our arm is the live benchmark's 24h mean MAPE, so the window, exclusions and settled truth match the published scorecard.

**35 of 44 balancing authorities beat the baseline**, by a median of 0.83 points.

**9 do not.** The forecast that loses by most is **SEC**, at 17.82% against the baseline's 11.46% — 6.36 points of *negative* skill, meaning the model is subtracting information rather than adding it. The rest sit within about a point of the line, which is noise at this sample size; one is not.

## Losing to the baseline

| ba | ours_pct | naive_pct | official_pct | skill | points_vs_naive | n |
|---|---|---|---|---|---|---|
| SEC | 17.82 | 11.46 | 7.83 | -0.5545 | -6.36 | 494 |
| NWMT | 4.8 | 3.77 | 2.85 | -0.2715 | -1.02 | 481 |
| NEVP | 5.59 | 4.65 | 8.11 | -0.2026 | -0.94 | 460 |
| JEA | 6.71 | 6.31 | 4.42 | -0.0634 | -0.4 | 448 |
| SCEG | 5.17 | 4.78 | 5.59 | -0.0805 | -0.39 | 448 |
| FPL | 4.56 | 4.3 | 3.06 | -0.06 | -0.26 | 364 |
| CAISO | 4.88 | 4.7 | 8.77 | -0.0385 | -0.18 | 304 |
| BPAT | 3.78 | 3.75 | 2.04 | -0.0069 | -0.03 | 486 |
| HST | 5.94 | 5.91 | 5.89 | -0.0044 | -0.03 | 436 |

## Every scoreable BA

| ba | ours_pct | naive_pct | official_pct | skill | points_vs_naive | n |
|---|---|---|---|---|---|---|
| SEC | 17.82 | 11.46 | 7.83 | -0.5545 | -6.36 | 494 |
| NWMT | 4.8 | 3.77 | 2.85 | -0.2715 | -1.02 | 481 |
| NEVP | 5.59 | 4.65 | 8.11 | -0.2026 | -0.94 | 460 |
| JEA | 6.71 | 6.31 | 4.42 | -0.0634 | -0.4 | 448 |
| SCEG | 5.17 | 4.78 | 5.59 | -0.0805 | -0.39 | 448 |
| FPL | 4.56 | 4.3 | 3.06 | -0.06 | -0.26 | 364 |
| CAISO | 4.88 | 4.7 | 8.77 | -0.0385 | -0.18 | 304 |
| BPAT | 3.78 | 3.75 | 2.04 | -0.0069 | -0.03 | 486 |
| HST | 5.94 | 5.91 | 5.89 | -0.0044 | -0.03 | 436 |
| PACE | 4.36 | 4.45 | 7.79 | 0.0204 | 0.09 | 448 |
| ERCOT | 2.48 | 2.69 | 1.44 | 0.0766 | 0.21 | 300 |
| SPP | 3.91 | 4.17 | 8.41 | 0.0621 | 0.26 | 375 |
| SOCO | 3.97 | 4.25 | 3.21 | 0.0663 | 0.28 | 430 |
| FMPP | 5.52 | 5.81 | 28.15 | 0.0503 | 0.29 | 523 |
| FPC | 6.41 | 6.73 | 22.0 | 0.047 | 0.32 | 473 |
| PACW | 4.7 | 5.16 | 2.29 | 0.0895 | 0.46 | 465 |
| GCPD | 2.18 | 2.75 | 4.04 | 0.2062 | 0.57 | 537 |
| PNM | 3.36 | 3.97 | 7.15 | 0.1555 | 0.62 | 505 |
| TAL | 5.77 | 6.42 | 5.54 | 0.1009 | 0.65 | 464 |
| SRP | 7.86 | 8.58 | 5.96 | 0.0844 | 0.72 | 467 |
| WALC | 8.17 | 8.89 | 12.82 | 0.0807 | 0.72 | 505 |
| EPE | 4.28 | 5.1 | 3.35 | 0.1605 | 0.82 | 504 |
| TEPC | 5.19 | 6.03 | 3.05 | 0.1391 | 0.84 | 526 |
| AECI | 5.64 | 6.49 | 2.01 | 0.1298 | 0.84 | 486 |
| TVA | 4.56 | 5.65 | 2.39 | 0.1921 | 1.08 | 462 |
| IPCO | 3.83 | 4.95 | 4.52 | 0.2274 | 1.13 | 456 |
| CHPD | 4.11 | 5.47 | 2.77 | 0.2489 | 1.36 | 497 |
| DOPD | 2.72 | 4.19 | 3.29 | 0.3503 | 1.47 | 485 |
| PJM | 4.44 | 5.98 | 3.69 | 0.2579 | 1.54 | 371 |
| CPLE | 5.07 | 6.61 | 3.96 | 0.2332 | 1.54 | 465 |
| CPLW | 4.79 | 6.45 | 4.08 | 0.258 | 1.66 | 457 |
| MISO | 3.47 | 5.19 | 2.43 | 0.3315 | 1.72 | 316 |
| AVA | 3.79 | 5.64 | 1.76 | 0.3272 | 1.85 | 507 |
| BANC | 5.85 | 7.9 | 3.54 | 0.2593 | 2.05 | 529 |
| SCL | 3.72 | 5.81 | 3.5 | 0.3598 | 2.09 | 491 |
| PSEI | 3.59 | 5.69 | 40.9 | 0.3692 | 2.1 | 464 |
| TPWR | 4.12 | 6.29 | 2.4 | 0.3456 | 2.17 | 514 |
| GVL | 6.04 | 8.28 | 12.19 | 0.2706 | 2.24 | 523 |
| DUK | 4.2 | 6.62 | 3.14 | 0.3661 | 2.42 | 456 |
| LGEE | 4.79 | 7.28 | 7.36 | 0.3425 | 2.49 | 431 |
| PGE | 4.14 | 6.8 | 2.07 | 0.391 | 2.66 | 506 |
| TIDC | 5.06 | 8.15 | 3.75 | 0.3791 | 3.09 | 378 |
| ISONE | 7.9 | 11.35 | 3.43 | 0.3039 | 3.45 | 382 |
| NYISO | 5.25 | 8.75 | 2.06 | 0.3997 | 3.5 | 381 |

**Reading.** `skill` is `1 − ours ÷ naive`: positive means the model beats the baseline, negative means it is worse than free. `points_vs_naive` is the same comparison in error points, which is the figure to act on. `official_pct` is the operator's own day-ahead forecast over the same hours, for context — it is not the baseline.

**What this does not say.** A model that beats seasonal-naive is not thereby good; the baseline is the floor, not a target. And skill is measured at one lead (24h) on one metric (mean MAPE), so it inherits every caveat in [`BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md).
