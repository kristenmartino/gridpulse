# Benchmark scoreability — which BAs can be compared, and why not

*Generated 2026-08-18 from the GCS vintage mirror. A dated snapshot, not a standing figure — the current count is whatever [`/api/v1/benchmark`](https://gridpulse.kristenmartino.ai/api/v1/benchmark) reports as `n_scoreable`, computed by the same `models.benchmark.scoreability` this script calls. Where the two disagree, this file is the stale one.*

**As measured on 2026-08-18: 25 of 51 balancing authorities are scoreable.** A BA is excluded only when it cannot be compared *fairly*; the reason is published for every one of them.

`df_coverage_pct` is the **BA's** publication rate — the share of hours EIA carried a day-ahead forecast for — and is the only figure the exclusion gate acts on. `df_asissued_pct` is **ours**: the share we observed early enough to score as-issued. Before #535 these were one number, and the second was being published as the first.

Among the scoreable set, the operators' *own* day-ahead accuracy spans **1.14% to 28.97% median APE** (a 25× spread), median-of-medians 2.87%. That spread is measured here against settled values with placeholder hours excluded — the same discipline the benchmark applies to both arms.

## Excluded

| ba | class | reason |
|---|---|---|
| ERCOT | bulk | df-coverage |
| MISO | bulk | df-coverage |
| NYISO | bulk | df-coverage |
| FPL | bulk | df-coverage |
| SPP | bulk | df-coverage |
| ISONE | bulk | df-coverage |
| SOCO | bulk | df-coverage |
| DUK | bulk | df-coverage |
| CPLE | bulk | df-coverage |
| BPAT | bulk | df-coverage |
| AZPS | broken | broken-feed |
| FPC | bulk | df-coverage |
| TEC | bulk | df-coverage |
| JEA | bulk | df-coverage |
| TAL | bulk | df-coverage |
| HST | bulk | df-coverage |
| SC | bulk | df-coverage |
| SCEG | bulk | df-coverage |
| CPLW | bulk | df-coverage |
| SPA | broken | broken-feed |
| PACE | bulk | df-coverage |
| PSEI | bulk | df-coverage |
| SCL | bulk | df-coverage |
| DOPD | bulk | df-coverage |
| LDWP | broken | broken-feed |
| IID | broken | broken-feed |

**`broken-feed`** — the feed's provisional readings revise heavily before settling, so intraday scoring is not meaningful; and GridPulse anchors its own forecast on that BA's day-ahead value (ADR-009), which would make the comparison partly self-referential. **`df-coverage`** — the BA publishes a day-ahead forecast too sparsely to score.

Note the direction of the bias: 4 of the exclusions are for feed brokenness, and BAs with sloppy data operations plausibly also forecast sloppily — so excluding them likely removes BAs where GridPulse would win. The exclusion set is conservative against our own claim.

## Scoreable

| ba | class | df_coverage_pct | df_asissued_pct | stub_pct | official_median_ape_pct | n_scoreable_hours |
|---|---|---|---|---|---|---|
| AVA | bulk | 88.0 | 85.3 | 0.97 | 1.14 | 602 |
| PGE | bulk | 91.2 | 88.5 | 3.06 | 1.42 | 628 |
| TVA | bulk | 82.9 | 80.4 | 1.95 | 1.51 | 575 |
| BANC | bulk | 99.3 | 96.5 | 3.48 | 1.54 | 677 |
| PACW | bulk | 84.4 | 82.2 | 2.5 | 1.55 | 587 |
| TPWR | bulk | 95.3 | 92.3 | 4.73 | 2.16 | 640 |
| CHPD | bulk | 99.3 | 96.5 | 9.04 | 2.26 | 643 |
| AECI | bulk | 86.0 | 83.5 | 3.48 | 2.32 | 583 |
| NWMT | churn | 83.5 | 80.5 | 0.28 | 2.41 | 574 |
| TEPC | unknown | 95.3 | 92.1 | 0.28 | 2.46 | 660 |
| TIDC | bulk | 89.5 | 80.2 | 4.74 | 2.52 | 557 |
| EPE | bulk | 95.3 | 92.5 | 8.48 | 2.74 | 618 |
| PJM | bulk | 81.0 | 78.0 | 4.17 | 2.87 | 542 |
| IPCO | bulk | 83.9 | 81.1 | 5.98 | 3.62 | 554 |
| GCPD | bulk | 99.3 | 96.5 | 1.67 | 3.83 | 691 |
| PNM | bulk | 91.2 | 88.3 | 3.34 | 4.5 | 623 |
| SRP | bulk | 88.2 | 85.4 | 2.5 | 5.37 | 610 |
| NEVP | bulk | 95.1 | 90.1 | 18.5 | 6.12 | 529 |
| SEC | bulk | 91.8 | 88.4 | 0.0 | 6.17 | 621 |
| CAISO | bulk | 82.9 | 79.4 | 26.56 | 6.82 | 397 |
| LGEE | bulk | 80.2 | 75.5 | 2.84 | 7.97 | 512 |
| WALC | bulk | 85.0 | 82.2 | 5.15 | 10.44 | 568 |
| GVL | bulk | 94.8 | 92.5 | 0.56 | 10.73 | 661 |
| PSCO | bulk | 80.4 | 73.1 | 0.0 | 26.5 | 525 |
| FMPP | bulk | 93.9 | 91.5 | 3.76 | 28.97 | 642 |
