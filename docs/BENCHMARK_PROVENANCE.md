# Benchmark provenance — what we actually measured

The two questions the benchmark engine could not answer about itself, measured. Re-run `python scripts/benchmark_provenance_probe.py` to refresh.

## Gate 1 — does EIA revise the day-ahead forecast?

`first_seen_df` is the day-ahead value re-read 0–3h *after* the target hour (the vintage window only admits an hour once EIA publishes a metered `D`). If EIA revises DF in between, the scoring choice matters. Below: how often it revises, and the official arm scored **both ways** against settled truth, under the benchmark's own exclusions.

| ba | n_compared | revised_pct | max_revision_pct | official_as_issued_pct | official_as_revised_pct | verdict_shift_pts |
|---|---|---|---|---|---|---|
| PJM | 668 | 0.0 | 0.0 | 3.1 | 3.1 | 0.0 |
| MISO | 646 | 0.0 | 0.0 | 2.43 | 2.43 | 0.0 |
| ERCOT | 656 | 0.0 | 0.0 | 1.15 | 1.15 | 0.0 |
| CAISO | 683 | 0.0 | 0.0 | 4.94 | 4.94 | 0.0 |
| SOCO | 645 | 24.2 | 9.23 | 1.84 | 1.76 | 0.08 |
| PSEI | 644 | 26.4 | 34.17 | 47.16 | 45.74 | 1.42 |
| FMPP | 707 | 5.2 | 11.43 | 28.36 | 28.46 | -0.1 |
| GVL | 707 | 0.0 | 0.0 | 11.34 | 11.34 | 0.0 |
| SPP | 622 | 0.0 | 0.0 | 8.0 | 8.0 | 0.0 |
| NYISO | 622 | 0.0 | 0.0 | 1.59 | 1.59 | 0.0 |

**Reading.** Revision is real but uneven — 7 of 10 sampled BAs never revise at all. The largest effect on any verdict is **1.42 points** (PSEI: 47.16% as-issued vs 45.74% as-revised), which does not flip a single conclusion. The benchmark therefore publishes **both**: as-issued as the fair comparison, as-revised as the conservative one, since a forecast revised after the target hour carries hindsight.

**Limit of this probe.** It cannot see a revision that happened *before* our first capture. That would need DF captured for hours with no `D` yet — a separate instrument, not built. So the phrasing everywhere is *the earliest day-ahead forecast we observed*, never *their day-ahead forecast*.

## Gate 2 — what lead do our forecasts actually carry?

The forecast anchors on the last *real* demand hour, so EIA's publishing lag makes a nominal 24h record shorter than 24h. Measured from live payloads (`scored_at` vs row timestamp).

| ba | nominal_24h_realized_h | nominal_48h_realized_h |
|---|---|---|
| PJM | 22.95 | 46.95 |
| MISO | 22.94 | 46.94 |
| ERCOT | 22.95 | 46.95 |
| CAISO | 22.95 | 46.95 |
| SOCO | 22.92 | 46.92 |
| PSEI | 22.81 | 46.81 |
| FMPP | 22.88 | 46.88 |
| GVL | 22.87 | 46.87 |
| SPP | 22.93 | 46.93 |
| NYISO | 22.94 | 46.94 |
| ISONE | 22.93 | 46.93 |
| BPAT | 22.9 | 46.9 |
| AVA | 22.8 | 46.8 |
| PGE | 22.81 | 46.81 |
| DUK | 22.91 | 46.91 |

**Reading.** A nominal-24h record is a realized **22.80–22.95h** lead — shorter than the label, and shorter than the operators' documented 17–41h day-ahead window, i.e. marginally in our favour. No *N hours ahead* claim should be published without this caveat.

The nominal-48h arm carries a minimum realized **46.80h** — which exceeds their documented maximum of 41h, so publishing it as the *conservative* comparison is supported by measurement rather than assumed.
