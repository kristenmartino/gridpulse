# Benchmark scoreability — which BAs can be compared, and why not

*Generated 2026-08-18 from the GCS vintage mirror. A dated snapshot, not a standing figure — the current count is whatever [`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark) reports as `n_scoreable`, computed by the same `models.benchmark.scoreability` this script calls. Where the two disagree, this file is the stale one.*

**As measured on 2026-08-18: 46 of 51 balancing authorities clear the scoreability gate.** A BA is excluded only when it cannot be compared *fairly*; the reason is published for every one of them.

This is the **gate** count. The live payload additionally requires at least `MIN_PAIRED_HOURS` comparable hours per lead, so its `n_scoreable` can be lower — a BA that publishes a day-ahead forecast but has too thin a paired sample is reported as `insufficient-paired-hours`, which is a different fact from `df-feed-stopped` and is published as such.

`df_coverage_pct` is the **BA's** publication rate — the share of hours EIA carried a day-ahead forecast for. `df_asissued_pct` is **ours**: the share we observed early enough to score as-issued. Before #535 these were one number, and the second was being published as the first.

**Neither one gates (#549).** `df_stale_hours` does — hours since the BA's most recent published day-ahead forecast, against a 168h ceiling. A rate cannot tell a BA that half-publishes from one that published completely and then stopped, and no BA in this fleet is diffusely sparse: every one with any absence has 92–100% of those hours inside runs of ≥3h.

Among the scoreable set, the operators' *own* day-ahead accuracy spans **1.13% to 33.05% median APE** (a 29× spread), median-of-medians 2.98%. That spread is measured here against settled values with placeholder hours excluded — the same discipline the benchmark applies to both arms.

## Excluded

| ba | class | reason |
|---|---|---|
| SPP | bulk | df-feed-stopped |
| AZPS | broken | broken-feed |
| SPA | broken | broken-feed |
| LDWP | broken | broken-feed |
| IID | broken | broken-feed |

**`broken-feed`** — the feed's provisional readings revise heavily before settling, so intraday scoring is not meaningful; and GridPulse anchors its own forecast on that BA's day-ahead value (ADR-009), which would make the comparison partly self-referential. **`df-feed-stopped`** — the BA has stopped publishing a day-ahead forecast, so every hour we could score predates the stop and the row would describe a different slice of the window than every other row.

Note the direction of the bias: 4 of the exclusions are for feed brokenness, and BAs with sloppy data operations plausibly also forecast sloppily — so excluding them likely removes BAs where GridPulse would win. The exclusion set is conservative against our own claim.

## Scoreable

| ba | class | df_coverage_pct | df_asissued_pct | stub_pct | official_median_ape_pct | n_scoreable_hours | df_stale_hours | absent_bias_pct |
|---|---|---|---|---|---|---|---|---|
| AVA | bulk | 96.7 | 85.4 | 0.97 | 1.13 | 603 | 0.0 | -17.38 |
| ERCOT | bulk | 92.6 | 61.6 | 10.85 | 1.29 | 377 | 5.0 | -2.42 |
| PGE | bulk | 100.0 | 88.6 | 2.92 | 1.41 | 630 | 0.0 |  |
| BPAT | bulk | 96.2 | 77.3 | 3.34 | 1.45 | 527 | 3.0 | -4.64 |
| BANC | bulk | 100.0 | 96.5 | 3.48 | 1.55 | 677 | 0.0 |  |
| PACW | bulk | 99.6 | 82.2 | 2.5 | 1.55 | 587 | 3.0 |  |
| TVA | bulk | 100.0 | 80.5 | 1.95 | 1.55 | 576 | 0.0 |  |
| SOCO | bulk | 96.0 | 64.1 | 1.39 | 1.86 | 451 | 5.0 | -3.36 |
| ISONE | bulk | 99.2 | 63.8 | 6.12 | 1.98 | 426 | 6.0 |  |
| NYISO | bulk | 95.8 | 56.6 | 3.2 | 2.14 | 384 | 6.0 | 2.05 |
| TPWR | bulk | 100.0 | 92.5 | 4.73 | 2.15 | 641 | 0.0 |  |
| CHPD | bulk | 100.0 | 96.5 | 9.18 | 2.27 | 642 | 0.0 |  |
| AECI | bulk | 100.0 | 83.6 | 3.34 | 2.34 | 585 | 0.0 |  |
| SCL | bulk | 96.2 | 76.2 | 2.36 | 2.35 | 524 | 3.0 | -2.86 |
| NWMT | churn | 99.4 | 80.5 | 0.28 | 2.41 | 574 | 4.0 |  |
| CPLE | bulk | 100.0 | 76.4 | 4.59 | 2.45 | 527 | 0.0 |  |
| TEPC | unknown | 100.0 | 92.2 | 0.28 | 2.47 | 661 | 0.0 |  |
| MISO | bulk | 92.6 | 62.9 | 36.44 | 2.52 | 190 | 5.0 | -8.16 |
| TIDC | bulk | 96.7 | 80.5 | 4.74 | 2.56 | 559 | 0.0 | 2.03 |
| FPL | bulk | 100.0 | 75.8 | 2.09 | 2.59 | 540 | 0.0 |  |
| EPE | bulk | 100.0 | 92.6 | 8.34 | 2.72 | 620 | 0.0 |  |
| DUK | bulk | 100.0 | 76.4 | 6.82 | 2.74 | 510 | 0.0 |  |
| PJM | bulk | 100.0 | 78.0 | 4.18 | 2.87 | 541 | 0.0 |  |
| SCEG | bulk | 99.3 | 73.8 | 11.98 | 3.08 | 454 | 5.0 |  |
| SC | bulk | 95.8 | 70.0 | 7.23 | 3.18 | 462 | 6.0 | -2.65 |
| TAL | bulk | 100.0 | 76.5 | 6.26 | 3.29 | 516 | 0.0 |  |
| CPLW | bulk | 100.0 | 76.5 | 6.26 | 3.31 | 506 | 0.0 |  |
| DOPD | bulk | 96.4 | 76.3 | 1.95 | 3.56 | 534 | 2.0 | -7.91 |
| JEA | bulk | 100.0 | 64.8 | 6.82 | 3.64 | 428 | 0.0 |  |
| IPCO | bulk | 99.6 | 81.6 | 5.98 | 3.65 | 558 | 3.0 |  |
| GCPD | bulk | 100.0 | 96.5 | 1.67 | 3.82 | 691 | 0.0 |  |
| TEC | bulk | 80.1 | 49.0 | 1.25 | 4.14 | 343 | 30.0 | 0.83 |
| HST | bulk | 100.0 | 76.5 | 6.68 | 4.44 | 513 | 0.0 |  |
| PNM | bulk | 100.0 | 88.5 | 3.34 | 4.53 | 624 | 0.0 |  |
| SRP | bulk | 96.7 | 85.5 | 2.5 | 5.25 | 611 | 0.0 | -6.51 |
| SEC | bulk | 96.7 | 88.6 | 0.0 | 6.2 | 622 | 0.0 | 11.82 |
| NEVP | bulk | 100.0 | 90.2 | 18.52 | 6.38 | 529 | 0.0 |  |
| CAISO | bulk | 99.6 | 79.4 | 26.56 | 6.82 | 397 | 3.0 |  |
| LGEE | bulk | 100.0 | 75.7 | 2.84 | 8.01 | 513 | 0.0 |  |
| PACE | bulk | 99.4 | 74.8 | 2.36 | 8.42 | 534 | 4.0 |  |
| WALC | bulk | 93.2 | 82.3 | 4.87 | 10.55 | 571 | 0.0 | -19.88 |
| GVL | bulk | 100.0 | 92.6 | 0.56 | 10.73 | 662 | 0.0 |  |
| FPC | bulk | 100.0 | 76.4 | 0.0 | 24.48 | 549 | 0.0 |  |
| PSCO | bulk | 96.1 | 73.2 | 0.0 | 26.5 | 525 | 5.0 | 0.65 |
| FMPP | bulk | 100.0 | 91.7 | 3.76 | 28.84 | 643 | 0.0 |  |
| PSEI | bulk | 99.6 | 73.4 | 0.14 | 33.05 | 527 | 3.0 |  |
