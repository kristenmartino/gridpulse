# Benchmark scoreability — which BAs can be compared, and why not

*Generated 2026-08-20 from the GCS vintage mirror. A dated snapshot, not a standing figure — the current count is whatever [`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark) reports as `n_scoreable`, computed by the same `models.benchmark.scoreability` this script calls. Where the two disagree, this file is the stale one.*

**As measured on 2026-08-20: 46 of 51 balancing authorities clear the scoreability gate.** A BA is excluded only when it cannot be compared *fairly*; the reason is published for every one of them.

This is the **gate** count. The live payload additionally requires at least `MIN_PAIRED_HOURS` comparable hours per lead, so its `n_scoreable` can be lower — a BA that publishes a day-ahead forecast but has too thin a paired sample is reported as `insufficient-paired-hours`, which is a different fact from `df-feed-gap` and is published as such.

`df_coverage_pct` is the **BA's** publication rate — the share of hours EIA carried a day-ahead forecast for. `df_asissued_pct` is **ours**: the share we observed early enough to score as-issued. Before #535 these were one number, and the second was being published as the first.

**Neither one gates (#549).** `df_longest_gap_hours` does — the longest stretch of the window with no published day-ahead forecast, against a 168h ceiling. `df_stale_hours` is the trailing gap alone and gates nothing; the two diverge once a stopped feed resumes (#587). A rate cannot tell a BA that half-publishes from one that published completely and then stopped, and no BA in this fleet is diffusely sparse: every one with any absence has 92–100% of those hours inside runs of ≥3h.

Among the scoreable set, the operators' *own* day-ahead accuracy spans **1.13% to 32.63% median APE** (a 29× spread), median-of-medians 2.85%. That spread is measured here against settled values with placeholder hours excluded — the same discipline the benchmark applies to both arms.

## Excluded

| ba | class | reason |
|---|---|---|
| SPP | bulk | df-feed-gap |
| AZPS | broken | broken-feed |
| SPA | broken | broken-feed |
| LDWP | broken | broken-feed |
| IID | broken | broken-feed |

**`broken-feed`** — the feed's provisional readings revise heavily before settling, so intraday scoring is not meaningful; and GridPulse anchors its own forecast on that BA's day-ahead value (ADR-009), which would make the comparison partly self-referential. **`df-feed-gap`** — the BA's day-ahead forecast is missing for a long enough stretch of the window that the hours we could score no longer describe the same period as every other row.

Note the direction of the bias: 4 of the exclusions are for feed brokenness, and BAs with sloppy data operations plausibly also forecast sloppily — so excluding them likely removes BAs where GridPulse would win. The exclusion set is conservative against our own claim.

## Scoreable

| ba | class | df_coverage_pct | df_asissued_pct | stub_pct | official_median_ape_pct | n_scoreable_hours | df_stale_hours | df_longest_gap_hours | absent_bias_pct |
|---|---|---|---|---|---|---|---|---|---|
| AVA | bulk | 96.7 | 85.7 | 0.83 | 1.13 | 606 | 0.0 | 24.0 | -17.61 |
| ERCOT | bulk | 92.5 | 61.8 | 7.52 | 1.18 | 402 | 6.0 | 24.0 | -3.19 |
| PGE | bulk | 100.0 | 88.9 | 2.92 | 1.37 | 632 | 0.0 | 0.0 |  |
| BPAT | bulk | 96.7 | 77.9 | 3.34 | 1.46 | 531 | 0.0 | 24.0 | -4.44 |
| PACW | bulk | 100.0 | 83.0 | 2.5 | 1.54 | 589 | 0.0 | 0.0 |  |
| BANC | bulk | 100.0 | 96.5 | 3.34 | 1.54 | 677 | 0.0 | 0.0 |  |
| TVA | bulk | 100.0 | 80.6 | 1.95 | 1.68 | 576 | 0.0 | 0.0 |  |
| NYISO | bulk | 95.5 | 56.5 | 1.53 | 1.95 | 395 | 8.0 | 24.0 | 0.47 |
| ISONE | bulk | 100.0 | 64.5 | 3.48 | 1.97 | 449 | 0.0 | 0.0 |  |
| TPWR | bulk | 100.0 | 92.8 | 4.45 | 2.21 | 645 | 0.0 | 0.0 |  |
| CHPD | bulk | 100.0 | 96.5 | 8.76 | 2.25 | 645 | 0.0 | 0.0 |  |
| NWMT | churn | 100.0 | 81.1 | 0.56 | 2.28 | 574 | 0.0 | 0.0 |  |
| SOCO | bulk | 99.2 | 65.6 | 0.97 | 2.32 | 464 | 6.0 | 6.0 |  |
| TEPC | unknown | 100.0 | 92.5 | 0.28 | 2.38 | 663 | 0.0 | 0.0 |  |
| SCL | bulk | 96.0 | 76.2 | 1.95 | 2.38 | 527 | 5.0 | 24.0 | -4.71 |
| TIDC | bulk | 96.7 | 80.5 | 5.15 | 2.4 | 557 | 0.0 | 25.0 | 1.64 |
| MISO | bulk | 92.5 | 62.8 | 34.96 | 2.47 | 200 | 6.0 | 24.0 | -8.11 |
| CPLE | bulk | 100.0 | 76.6 | 4.46 | 2.59 | 529 | 0.0 | 0.0 |  |
| FPL | bulk | 100.0 | 75.9 | 1.95 | 2.59 | 541 | 0.0 | 0.0 |  |
| AECI | bulk | 100.0 | 83.9 | 3.76 | 2.59 | 585 | 0.0 | 0.0 |  |
| DUK | bulk | 100.0 | 76.6 | 7.1 | 2.69 | 509 | 0.0 | 0.0 |  |
| SCEG | bulk | 100.0 | 74.1 | 11.82 | 2.76 | 458 | 0.0 | 0.0 |  |
| EPE | bulk | 100.0 | 92.9 | 8.62 | 2.81 | 620 | 0.0 | 0.0 |  |
| PJM | bulk | 100.0 | 78.3 | 3.76 | 2.9 | 546 | 0.0 | 0.0 |  |
| DOPD | bulk | 96.0 | 76.2 | 2.78 | 3.1 | 528 | 5.0 | 24.0 | -8.82 |
| SC | bulk | 96.7 | 70.4 | 7.51 | 3.16 | 463 | 0.0 | 24.0 | -1.07 |
| TAL | bulk | 100.0 | 76.7 | 6.96 | 3.25 | 512 | 0.0 | 0.0 |  |
| CPLW | bulk | 100.0 | 76.8 | 6.12 | 3.41 | 508 | 0.0 | 0.0 |  |
| IPCO | bulk | 100.0 | 81.9 | 6.13 | 3.49 | 558 | 0.0 | 0.0 |  |
| JEA | bulk | 98.9 | 64.8 | 7.23 | 3.63 | 425 | 8.0 | 8.0 |  |
| GCPD | bulk | 100.0 | 96.5 | 1.67 | 3.83 | 692 | 0.0 | 0.0 |  |
| TEC | bulk | 82.2 | 51.3 | 1.25 | 4.24 | 360 | 8.0 | 24.0 | -0.88 |
| PNM | bulk | 100.0 | 88.7 | 3.06 | 4.43 | 627 | 0.0 | 0.0 |  |
| HST | bulk | 100.0 | 76.8 | 6.4 | 4.59 | 517 | 0.0 | 0.0 |  |
| SRP | bulk | 100.0 | 88.9 | 2.5 | 5.5 | 635 | 0.0 | 0.0 |  |
| NEVP | bulk | 100.0 | 90.4 | 18.52 | 5.97 | 530 | 0.0 | 0.0 |  |
| SEC | bulk | 96.7 | 88.9 | 0.0 | 6.4 | 623 | 0.0 | 24.0 | 11.77 |
| CAISO | bulk | 99.4 | 79.5 | 23.68 | 6.6 | 418 | 4.0 | 4.0 |  |
| LGEE | bulk | 100.0 | 75.9 | 2.85 | 8.07 | 512 | 0.0 | 16.0 |  |
| PACE | bulk | 100.0 | 78.0 | 2.5 | 8.47 | 555 | 0.0 | 0.0 |  |
| WALC | bulk | 93.2 | 82.6 | 4.87 | 10.06 | 573 | 0.0 | 48.0 | -19.61 |
| GVL | bulk | 100.0 | 92.9 | 0.7 | 10.57 | 663 | 0.0 | 0.0 |  |
| FPC | bulk | 100.0 | 76.6 | 0.0 | 24.39 | 550 | 0.0 | 0.0 |  |
| PSCO | bulk | 96.7 | 73.5 | 0.0 | 26.56 | 527 | 0.0 | 25.0 | 7.22 |
| FMPP | bulk | 100.0 | 91.9 | 3.76 | 28.02 | 644 | 0.0 | 0.0 |  |
| PSEI | bulk | 100.0 | 74.3 | 0.14 | 32.63 | 533 | 0.0 | 0.0 |  |
