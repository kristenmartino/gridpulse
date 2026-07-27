# Benchmark scoreability — which BAs can be compared, and why not

**44 of 51 balancing authorities are scoreable.** A BA is excluded only when it cannot be compared *fairly*; the reason is published for every one of them.

Among the scoreable set, the operators' *own* day-ahead accuracy spans **1.15% to 47.21% median APE** (a 41× spread), median-of-medians 3.05%. That spread is measured here against settled values with placeholder hours excluded — the same discipline the benchmark applies to both arms.

## Excluded

| ba | class | reason |
|---|---|---|
| AZPS | broken | broken-feed |
| PSCO | bulk | df-coverage |
| TEC | unknown | df-coverage |
| SC | bulk | df-coverage |
| SPA | broken | broken-feed |
| LDWP | broken | broken-feed |
| IID | broken | broken-feed |

**`broken-feed`** — the feed's provisional readings revise heavily before settling, so intraday scoring is not meaningful; and GridPulse anchors its own forecast on that BA's day-ahead value (ADR-009), which would make the comparison partly self-referential. **`df-coverage`** — the BA publishes a day-ahead forecast too sparsely to score.

Note the direction of the bias: four of the exclusions are for feed brokenness, and BAs with sloppy data operations plausibly also forecast sloppily — so excluding them likely removes BAs where GridPulse would win. The exclusion set is conservative against our own claim.

## Scoreable

| ba | class | df_coverage_pct | stub_pct | official_median_ape_pct | n_scoreable_hours |
|---|---|---|---|---|---|
| ERCOT | churn | 91.2 | 10.57 | 1.15 | 580 |
| AVA | unknown | 91.4 | 1.25 | 1.28 | 647 |
| PGE | unknown | 96.7 | 1.25 | 1.33 | 686 |
| BPAT | churn | 93.5 | 0.42 | 1.41 | 668 |
| AECI | bulk | 95.0 | 1.95 | 1.51 | 667 |
| NYISO | unknown | 86.5 | 8.07 | 1.59 | 564 |
| TVA | bulk | 90.0 | 0.14 | 1.84 | 645 |
| SOCO | bulk | 89.7 | 4.45 | 1.84 | 613 |
| PACW | bulk | 89.0 | 0.28 | 1.88 | 638 |
| TPWR | unknown | 98.3 | 2.36 | 1.89 | 689 |
| BANC | churn | 99.9 | 1.25 | 2.05 | 708 |
| CHPD | unknown | 100.0 | 7.37 | 2.13 | 666 |
| TEPC | unknown | 98.3 | 0.0 | 2.17 | 707 |
| NWMT | churn | 86.8 | 0.28 | 2.21 | 621 |
| DUK | bulk | 91.7 | 2.23 | 2.27 | 642 |
| ISONE | unknown | 87.6 | 9.46 | 2.38 | 562 |
| MISO | bulk | 89.8 | 20.31 | 2.43 | 500 |
| DOPD | unknown | 95.0 | 3.48 | 2.54 | 658 |
| FPL | bulk | 91.7 | 0.56 | 2.55 | 655 |
| EPE | unknown | 98.3 | 3.76 | 2.75 | 680 |
| CPLE | bulk | 91.7 | 1.39 | 3.0 | 649 |
| IPCO | bulk | 89.6 | 0.97 | 3.01 | 636 |
| PJM | unknown | 92.9 | 1.39 | 3.1 | 658 |
| TIDC | unknown | 96.8 | 2.23 | 3.17 | 678 |
| SCL | churn | 91.8 | 1.25 | 3.34 | 649 |
| JEA | unknown | 91.1 | 1.25 | 3.35 | 646 |
| TAL | unknown | 91.7 | 1.81 | 3.6 | 646 |
| CPLW | bulk | 91.7 | 2.78 | 3.71 | 639 |
| GCPD | churn | 100.0 | 0.28 | 4.07 | 716 |
| HST | clean | 91.7 | 8.07 | 4.11 | 601 |
| SCEG | bulk | 91.7 | 3.89 | 4.19 | 631 |
| SRP | unknown | 87.2 | 0.28 | 4.58 | 625 |
| CAISO | churn | 95.0 | 17.39 | 4.94 | 558 |
| SEC | bulk | 95.1 | 0.28 | 5.79 | 674 |
| PACE | bulk | 88.0 | 0.42 | 6.24 | 630 |
| NEVP | bulk | 98.3 | 6.68 | 6.87 | 659 |
| LGEE | bulk | 90.4 | 0.0 | 7.24 | 650 |
| SPP | unknown | 86.5 | 9.04 | 7.99 | 557 |
| PNM | unknown | 96.7 | 1.39 | 8.23 | 685 |
| GVL | unknown | 98.3 | 0.42 | 11.34 | 704 |
| WALC | churn | 96.7 | 1.53 | 11.72 | 684 |
| FPC | bulk | 91.7 | 0.0 | 22.94 | 659 |
| FMPP | bulk | 98.3 | 0.0 | 28.36 | 707 |
| PSEI | bulk | 89.6 | 0.0 | 47.21 | 644 |
