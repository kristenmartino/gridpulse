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

A subsequent expanded-state check reproduced a separate exact-target disclosure
defect: opening it widened the 375px Forecast page to 838px. At 1280px the table
remained contained, but inherited the older metrics-table presentation. The
follow-up checks actual `details.open` state, card and document geometry, local
table scrolling and complete source values at each reviewed width.

A focused supporting-card review also found weather text almost flush with its
border, 9px captions in three narrow mobile columns, and flex gaps compounded
by child margins around disclosures and actions. The spacing follow-up checks
weather values and units, caption readability, advanced controls and the
selected-source/next-step evidence cards as actual rendered states.
The repair adds real weather-card insets and readable captions, stacks those
cards below 600px, and uses one disclosure/action spacing rhythm. The existing
clientside sensitivity toggle now synchronizes its collapsed content and
`aria-expanded` state; the decorative plus/minus follows that state.

The Conditions message review found default white filter fields, a 120px search
box at 375px, border-adjacent disclosure text and numeric-table alignment on
textual message evidence. The follow-up keeps source fields and supplied
offsets intact while separating text-table styling from numeric exact values.
Fresh expanded legacy-offset checks already kept the document within the
375/768/1280 viewports; its wide mobile table needed clearer local-scroll
affordance. Closed/open geometry and keyboard scrolling remain separate checks.

The Overview Review summary follow-up preserves its aligned 12px body text,
line height and paragraph gaps. Its expanded final statement previously ended
only 1px from the border; scoped CSS adds 16px bottom padding and centers the
title/indicator in the toggle row. Narrative values and calculations are
unchanged. This small CSS correction is verified by native geometry/open-state
checks rather than new implementation-mirroring tests.

## Run-to-run window follow-through

The initial comparison exposed every eligible shared target, which could be
nearly 30 days for two stored baseline issues. Run review now defaults to a
24-hour elapsed UTC window from the first eligible shared target; optional 7-day
and advanced 30-day windows remain independent of the main outlook horizon.
Gaps never stretch the requested interval and short overlap is disclosed.

Peak magnitude, earliest peak timing, largest signed hourly revision and
changed-hour counts are calculated from the exact plot/table rows. Ties are
qualified, small real revisions are not labeled zero, and missing comparisons
do not acquire zero-valued metrics. The repeating-baseline note requires both
receipts to identify that source. Later reported demand retains all eligible
actual matches and does not inherit the new run-only control.

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

The full local suite including the run-to-run window follow-through passes:
4,349 tests passed, 3 skipped, with 92.44% measured coverage. Ruff, formatting
and diff checks pass. The preceding fidelity review also checked JavaScript
parsing. The pull request records the final native comparison result and exact
reviewed source.

The subsequent table presentation repair passes 49 affected existing unit and
real Dash-dispatch tests, plus Ruff, formatting and diff checks. Its expanded
mobile/desktop geometry and local-scroll review are recorded separately in the
pull request; the full-suite count above precedes that presentation-only repair.

The supporting-card spacing follow-up passes 52 affected existing unit and
real Dash-dispatch tests, plus Ruff, formatting and diff checks. Native review
separately checks the clientside sensitivity state, since the Flask dispatch
test driver does not execute browser callbacks.

The message/legacy-table presentation follow-up passes 65 affected existing
unit and real Dash-dispatch tests, plus Ruff, formatting and diff checks.
Native review separately covers source-message filtering, open text evidence,
Inspect/Back focus and local scrolling in the populated legacy table.

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
