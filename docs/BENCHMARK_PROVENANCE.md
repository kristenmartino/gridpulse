# Benchmark provenance — what we actually measured

The two questions the benchmark engine could not answer about itself, measured. Re-run `python scripts/benchmark_provenance_probe.py` to refresh.

## Gate 1 — does EIA revise the day-ahead forecast?

`first_seen_df` is the day-ahead value re-read 0–3h *after* the target hour (the vintage window only admits an hour once EIA publishes a metered `D`). If EIA revises DF in between, the scoring choice matters. Below: how often it revises, and the official arm scored **both ways** against settled truth, under the benchmark's own exclusions.

| ba | n_compared | revised_pct | max_revision_pct | official_as_issued_pct | official_as_revised_pct | verdict_shift_pts |
|---|---|---|---|---|---|---|
| PJM | 668 | 0.0 | 0.0 | 3.1 | 3.1 | 0.0 |
| MISO | 646 | 0.0 | 0.0 | 2.44 | 2.44 | 0.0 |
| ERCOT | 656 | 0.0 | 0.0 | 1.15 | 1.15 | 0.0 |
| CAISO | 683 | 0.0 | 0.0 | 4.94 | 4.94 | 0.0 |
| SOCO | 645 | 24.2 | 9.23 | 1.84 | 1.76 | 0.08 |
| PSEI | 644 | 26.4 | 34.17 | 47.15 | 45.71 | 1.43 |
| FMPP | 707 | 5.2 | 11.43 | 28.36 | 28.46 | -0.1 |
| GVL | 707 | 0.0 | 0.0 | 11.34 | 11.34 | 0.0 |
| SPP | 622 | 0.0 | 0.0 | 8.0 | 8.0 | 0.0 |
| NYISO | 622 | 0.0 | 0.0 | 1.6 | 1.6 | 0.0 |

**Reading.** Revision is real but uneven — 7 of 10 sampled BAs never revise at all. The largest effect on any verdict is **1.43 points** (PSEI: 47.15% as-issued vs 45.71% as-revised), which does not flip a single conclusion. The benchmark therefore publishes **both**: as-issued as the fair comparison, as-revised as the conservative one, since a forecast revised after the target hour carries hindsight.

**Limit of this probe.** It cannot see a revision that happened *before* our first capture. That would need DF captured for hours with no `D` yet — a separate instrument, not built. So the phrasing everywhere is *the earliest day-ahead forecast we observed*, never *their day-ahead forecast*.

## Gate 2 — what lead do our forecasts actually carry?

The forecast anchors on the last *real* demand hour, so EIA's publishing lag makes a nominal 24h record shorter than 24h. Measured from live payloads (`scored_at` vs row timestamp).

| ba | nominal_24h_realized_h | nominal_48h_realized_h |
|---|---|---|
| PJM | 23.95 | 47.95 |
| MISO | 23.92 | 47.92 |
| ERCOT | 23.95 | 47.95 |
| CAISO | 23.95 | 47.95 |
| SOCO | 23.9 | 47.9 |
| PSEI | 23.81 | 47.81 |
| FMPP | 23.87 | 47.87 |
| GVL | 23.85 | 47.85 |
| SPP | 23.92 | 47.92 |
| NYISO | 23.93 | 47.93 |
| ISONE | 23.92 | 47.92 |
| BPAT | 23.88 | 47.88 |
| AVA | 23.8 | 47.8 |
| PGE | 23.81 | 47.81 |
| DUK | 23.9 | 47.9 |

**Reading.** A nominal-24h record is a realized **23.80–23.95h** lead — shorter than its label, and sitting *inside* the operators' documented 17–41h day-ahead window rather than beyond it, so on a typical hour they had at least as much lead as we did. No *N hours ahead* claim should be published without this caveat.

The nominal-48h arm carries a minimum realized **47.80h** — which exceeds their documented maximum of 41h, so publishing it as the *conservative* comparison is supported by measurement rather than assumed.
