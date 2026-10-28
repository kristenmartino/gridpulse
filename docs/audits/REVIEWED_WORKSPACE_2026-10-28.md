# Reviewed forecast workspace — 2026-10-28

This change implements the reviewed US Grid, Forecast, Conditions and Models
views in Dash/Plotly. It preserves the approved Overview layout and existing
model, publication and scoring policies. Implementation and local verification
do not establish deployment, practitioner adoption or improved forecast skill.

## Product and evidence contract

| View | Implemented purpose | Evidence boundary |
| --- | --- | --- |
| US Grid | Search/filter/sort a table or cards, inspect an authority on the real territory map, then explicitly review its forecast | Dynamic coverage and per-authority clocks; no demand/nameplate risk ranking or invented simultaneous national peak |
| Forecast | Inspect the selected source/target window, publication receipts, run changes, weather/generation and bounded sensitivity | Same-series source/error resolution and horizon guards remain; current weather coverage does not reconstruct original weather inputs |
| Conditions | Explore Past, Future or matched Error with linked UTC plots, a shared hour, date/span controls and exact values; inspect associated-state messages separately | No model replay, original-issue reconstruction, station-observation claim or causal weather-error attribution |
| Models | Separate sampled observed error, first-hour records, advanced legacy offsets, training reference and fixed backtest diagnostics | Training filters never trigger inference or relabel the cached residual series; counts, unavailable clocks and thin coverage remain visible |

View links retain allowlisted controls, per-mode Explorer windows and the
inspected hour. They create a URL rather than a frozen review record or an
operational save. Legacy persona state remains compatible while its selector
is hidden in the single workspace. Source-degradation warnings remain visible
in Briefing Mode.

## Approved-design fidelity correction

The first implementation retained much of the callback behavior but missed
approved composition: Forecast was a vertical stack, US Grid lacked its summary
strip/table treatment, Models lacked evidence rails, and shared legacy tokens
changed widths, spacing and colors between views. Green callback tests and the
initial browser review did not establish visual parity. The user's comparison
triggered a fresh review against the approved study at matched viewports.

The correction restores native Dash card/rail layouts, compact toolbars,
source-specific context and bounded directory scrolling. Overview retains its
history/forecast bridge, with the forecast-target boundary kept separate from
that connecting point. Shared gutters, tablet utilities and mobile table/plot
geometry are checked in a real browser. No mockup numbers or regional clocks are
copied into the application to make a screenshot appear complete.

The comparison also checks actual interaction: keyboard target inspection,
source/history/zoom controls, the shared Explorer hour, diagnostic readouts,
explicit map-to-Forecast navigation and empty-region states. Plotly's rendered
values and axis conversions matter here: binary transport arrays are not point
values, and browser-local parsing must not shift declared UTC windows. Hidden
charts must resize when their panel becomes visible as well as when the viewport
changes. Inspected readouts reset with region/source/figure changes, including
pending and empty states. These are presentation and review controls; forecasting,
scoring, publication and source-quality policies remain unchanged.

Final native review and test results are recorded against the final source in
the pull request. Native date/select controls, isolated capture availability and
physical-device touch behavior retain their separate verification boundaries.

## Clock and quality repairs

- `data/weather_windows.py` selects historical/current context at or before an
  explicit UTC clock, and selects forecast weather by the chosen target bounds.
  The producer and consumers no longer use a combined feed's future tail as
  current weather. Historical insight correlation joins exact UTC hours rather
  than similarly sized arrays; a missing peak-hour temperature stays missing.
- New current-weather payloads preserve their sample and producer clocks.
  Clockless legacy headlines are not trusted as current samples.
- Backtest folds originate in UTC but lost their annotation through NumPy.
  Writers now serialize UTC; legacy diagnostics use that meaning only when
  the documented backtest producer lineage is present. Unknown naive clocks
  remain unknown.
- New actuals cache payloads carry optional paired `forecast_mw` values so the
  Explorer can reuse all available retained-hour quality signals. Older caches
  disclose the unavailable day-ahead-ratio input. Original stamped exclusions
  and newly detected retained-window exclusions are deduplicated for reporting;
  raw historical display remains distinct from eligible error comparisons.

The same quality thresholds and publication/scoring writers remain in place.
No model training, forecast-policy change, production archive write or shortened
upstream demand fetch is part of the interface request path.

## Forecast completed-response states

Forecast source, chart and empty statistics share one availability resolver.
Pending callbacks use Dash loading state and hide prior-selection values. A
completed empty-cache read shows "No cached forecast available" with `—` KPIs,
not a permanent loading placeholder. Recent regional input writes support a
warming state only until a completed pass is recorded; a completed pass without
a forecast supports the unavailable state. Strict Redis reads distinguish a
failed read from an absent key without adding retry loops. Usable cached
forecasts still render when weather or actuals are absent; model/horizon guards
and region-keyed inputs remain enforced. Availability handling remains scoped to
Forecast; Overview retains its approved data/history/coverage behavior.

Real callback dispatch covers these states, failures, missing selections and
FPL → empty CAISO → FPL isolation. Native preview checks cover an intentionally
delayed request, settled values and message fit at 375px.
These results do not establish production deployment or physical-device touch
behavior.

## Deliberate limits

Error review requires cached immutable issues and eligible actual matches in
the retained window. An index entry with an unavailable receipt is not a
permanent loading state. Pilot coverage and missing issues are not backfilled.
Selected-window MAE in MW is separate from the official hourly-sampled MAPE.

The current cache does not retain exact training/evaluation dates for each
training-reference metric. Payload generation and diagnostic target periods
cannot substitute for those dates. Weather messages do not establish territory
intersection or a quantified demand effect.

Expected weather as issued was not saved alongside the demand issue. Later
historical estimates provide context; weather-input error cannot be separated
from model error here. Archiving that weather vintage is deferred follow-up.

## Verification

The fidelity correction passes the full local suite: 4,319 tests passed, 3
skipped, with 92.32% measured coverage. Ruff, formatting, JavaScript parsing and
scoped callback checks also pass. The pull request records the final native
comparison result and exact reviewed source.

Focused behavioral and real Dash-dispatch tests cover cache-only readers,
regional isolation, issue selection, time-window validation, gap rendering,
quality inputs, source/lead separation, independent training filters and exact
diagnostic values. The local preview uses captured FPL values and genuine
public-cache issue receipts; it is isolated from production writers and model
execution. Other authorities exercise unavailable states, rather than borrowing
FPL data.

Final full-suite, lint and independent visual-review results are recorded in
the pull request after the final implementation checks complete. Physical
device touch behavior and production deployment are separate verification
boundaries.
