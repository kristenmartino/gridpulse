# Benchmark scoreability — which BAs can be compared, and why not

*Generated 2026-08-18 from the GCS vintage mirror. A dated snapshot, not a standing figure — the current count is whatever [`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark) reports as `n_scoreable`, computed by the same `models.benchmark.scoreability` this script calls. Where the two disagree, this file is the stale one.*

**As measured on 2026-08-18: 46 of 51 balancing authorities clear the scoreability gate.** A BA is excluded only when it cannot be compared *fairly*; the reason is published for every one of them.

This is the **gate** count. The live payload additionally requires at least `MIN_PAIRED_HOURS` comparable hours per lead, so its `n_scoreable` can be lower — a BA that publishes a day-ahead forecast but has too thin a paired sample is reported as `insufficient-paired-hours`, which is a different fact from `df-coverage` and is published as such.

`df_coverage_pct` is the **BA's** publication rate — the share of hours EIA carried a day-ahead forecast for — and is the only figure the exclusion gate acts on. `df_asissued_pct` is **ours**: the share we observed early enough to score as-issued. Before #535 these were one number, and the second was being published as the first.

Among the scoreable set, the operators' *own* day-ahead accuracy spans **1.13% to 33.14% median APE** (a 29× spread), median-of-medians 2.98%. That spread is measured here against settled values with placeholder hours excluded — the same discipline the benchmark applies to both arms.

## Excluded

| ba | class | reason |
|---|---|---|
| SPP | bulk | df-coverage |
| AZPS | broken | broken-feed |
| SPA | broken | broken-feed |
| LDWP | broken | broken-feed |
| IID | broken | broken-feed |

**`broken-feed`** — the feed's provisional readings revise heavily before settling, so intraday scoring is not meaningful; and GridPulse anchors its own forecast on that BA's day-ahead value (ADR-009), which would make the comparison partly self-referential. **`df-coverage`** — the BA publishes a day-ahead forecast too sparsely to score.

Note the direction of the bias: 4 of the exclusions are for feed brokenness, and BAs with sloppy data operations plausibly also forecast sloppily — so excluding them likely removes BAs where GridPulse would win. The exclusion set is conservative against our own claim.

## Scoreable

| ba | class | df_coverage_pct | df_asissued_pct | stub_pct | official_median_ape_pct | n_scoreable_hours |
|---|---|---|---|---|---|---|
| AVA | bulk | 96.7 | 85.3 | 0.97 | 1.13 | 602 |
| ERCOT | bulk | 93.3 | 61.8 | 10.99 | 1.29 | 377 |
| PGE | bulk | 100.0 | 88.5 | 3.06 | 1.41 | 628 |
| BPAT | bulk | 96.7 | 77.3 | 3.34 | 1.45 | 527 |
| TVA | bulk | 100.0 | 80.5 | 1.95 | 1.51 | 576 |
| BANC | bulk | 100.0 | 96.5 | 3.48 | 1.54 | 677 |
| PACW | bulk | 100.0 | 82.2 | 2.5 | 1.55 | 587 |
| SOCO | bulk | 96.7 | 64.1 | 1.39 | 1.86 | 451 |
| ISONE | bulk | 99.9 | 63.8 | 6.12 | 1.98 | 426 |
| NYISO | bulk | 96.5 | 56.6 | 3.2 | 2.14 | 384 |
| TPWR | bulk | 100.0 | 92.3 | 4.73 | 2.16 | 640 |
| CHPD | bulk | 100.0 | 96.5 | 9.04 | 2.27 | 643 |
| AECI | bulk | 100.0 | 83.6 | 3.34 | 2.34 | 585 |
| SCL | bulk | 96.7 | 76.2 | 2.36 | 2.35 | 524 |
| NWMT | churn | 100.0 | 80.5 | 0.28 | 2.41 | 574 |
| CPLE | bulk | 99.9 | 76.4 | 4.73 | 2.45 | 526 |
| TEPC | unknown | 100.0 | 92.1 | 0.28 | 2.45 | 660 |
| MISO | bulk | 93.3 | 63.0 | 36.58 | 2.52 | 190 |
| TIDC | bulk | 96.7 | 80.2 | 4.74 | 2.55 | 558 |
| FPL | bulk | 99.9 | 75.8 | 2.09 | 2.6 | 540 |
| DUK | bulk | 99.9 | 76.4 | 6.95 | 2.72 | 509 |
| EPE | bulk | 100.0 | 92.5 | 8.34 | 2.73 | 619 |
| PJM | bulk | 99.9 | 78.2 | 4.17 | 2.87 | 543 |
| SCEG | bulk | 99.9 | 74.0 | 12.24 | 3.08 | 454 |
| SC | bulk | 96.5 | 70.1 | 7.37 | 3.18 | 462 |
| TAL | bulk | 99.9 | 76.5 | 6.4 | 3.29 | 515 |
| CPLW | bulk | 99.9 | 76.5 | 6.26 | 3.31 | 506 |
| DOPD | bulk | 96.7 | 76.2 | 1.95 | 3.57 | 534 |
| IPCO | bulk | 100.0 | 81.5 | 5.98 | 3.64 | 557 |
| JEA | bulk | 99.9 | 64.7 | 6.82 | 3.65 | 427 |
| GCPD | bulk | 100.0 | 96.5 | 1.67 | 3.83 | 691 |
| TEC | bulk | 80.1 | 49.0 | 1.25 | 4.14 | 343 |
| HST | bulk | 99.9 | 76.5 | 6.95 | 4.42 | 511 |
| PNM | bulk | 100.0 | 88.3 | 3.34 | 4.52 | 623 |
| SRP | bulk | 96.7 | 85.4 | 2.5 | 5.32 | 610 |
| SEC | bulk | 96.5 | 88.5 | 0.0 | 6.18 | 622 |
| NEVP | bulk | 100.0 | 90.1 | 18.5 | 6.2 | 529 |
| CAISO | bulk | 100.0 | 79.4 | 26.56 | 6.82 | 397 |
| LGEE | bulk | 100.0 | 75.7 | 2.84 | 7.97 | 513 |
| PACE | bulk | 100.0 | 74.8 | 2.36 | 8.42 | 534 |
| WALC | bulk | 93.2 | 82.2 | 5.15 | 10.52 | 568 |
| GVL | bulk | 99.9 | 92.5 | 0.56 | 10.72 | 661 |
| FPC | bulk | 99.9 | 76.4 | 0.0 | 24.42 | 549 |
| PSCO | bulk | 96.7 | 73.3 | 0.0 | 26.49 | 526 |
| FMPP | bulk | 99.9 | 91.5 | 3.76 | 28.93 | 642 |
| PSEI | bulk | 100.0 | 73.4 | 0.14 | 33.14 | 527 |
