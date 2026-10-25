# [forecast-provenance] A forecast label is not a publication record

The evidence audit reproduced API/dashboard series divergence: the API preferred a comparison ensemble while the dashboard followed the published column. It also found forecast-start offsets described as exact issue leads, and bounded chart-callback replay carrying capture clocks instead of original publication provenance.

Prevention: use one published-series resolver; qualify clock/coverage semantics; capture immutable evidence only after a successful required forecast write. Do not relabel legacy replay or reconstruct forecasts from current models as original issues. Regression checks cover source selection, publication ordering, retry identity, and unknown/partial coverage.

A patch application failed during implementation; its result and diff were inspected before retrying. No partial patch or failed check was treated as a completed change.
