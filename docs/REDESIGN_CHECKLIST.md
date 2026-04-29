# Shell Redesign — Verification Checklist (R6)

_Closes the [Shell Redesign — v2 alignment](../.claude/plans/shell-redesign-v2.md) work that ran R1 → R5c. Use this checklist for visual review before deploying any subsequent change that touches the shell._

## Automated guards (run on every PR)

`tests/unit/test_redesign_smoke.py` — 19 assertions, runs in <0.2 s:

| Guard | What it pins |
|---|---|
| `TestTabStructure.test_visible_tab_count_is_four` | `_VISIBLE_TABS` exactly = `{tab-overview, tab-outlook, tab-alerts, tab-models}` |
| `TestTabStructure.test_visible_tab_labels_match_v2_naming` | Tab labels = `Overview / Forecast / Risk / Models` |
| `TestTabStructure.test_hidden_tabs_still_render` | Hidden tabs still in DOM so callbacks resolve |
| `TestRemovedIDsStayRemoved.test_overview_cards_removed` | R2 cards (`overview-greeting`, `-briefing`, `-changes`, `-data-health`, `-insight-digest`, `-news-feed`) absent |
| `TestRemovedIDsStayRemoved.test_layout_level_clutter_removed` | R3 cuts (`welcome-card`, `kpi-cards`) absent |
| `TestV2SurfacesPresent.test_overview_v2_ids_present` | Overview's 5 v2 IDs present |
| `TestV2SurfacesPresent.test_forecast_v2_ids_present` | Forecast's 14 v2 IDs present (incl. 3 inline panels + 3 sliders) |
| `TestV2SurfacesPresent.test_risk_v2_ids_present` | Risk's 10 v2 IDs present |
| `TestV2SurfacesPresent.test_models_v2_ids_present` | Models' 10 v2 IDs present |
| `TestV2SurfacesPresent.test_main_landmark_present` | `<main id="main-content">` landmark present |
| `TestBrandIdentity.test_favicon_links_present` | Favicon + apple-touch-icon + mask-icon links present in index_string |
| `TestBrandIdentity.test_og_meta_present` | OG + Twitter card meta tags present |
| `TestBrandIdentity.test_no_pre_v2_cyan_in_index_string` | Stale `#38D0FF` cyan absent |
| `TestHeaderStructure.test_header_has_gp_header_class` | Header has `gp-header` class |
| `TestHeaderStructure.test_header_has_grid_pulse_wordmark` | Color-split `Grid|Pulse` wordmark present |
| `TestCSSTokens.test_v2_color_tokens_present` | All 6 v2 token values in `custom.css` |
| `TestCSSTokens.test_micro_craft_present` | `::selection`, `::-webkit-scrollbar`, `caret-color` rules present |
| `TestCSSTokens.test_briefing_mode_rules_present` | `body.briefing` chrome rules present |
| `TestPlotLayout.test_plot_layout_dict_exists` | `PLOT_LAYOUT.font.color` = v2 `#a1a1aa` |

## Manual perceptual checks (run at deploy time)

Capture each at 1440×900 in light + dark OS modes; compare against [`gridpulse-v2.vercel.app/dashboard`](https://gridpulse-v2.vercel.app/dashboard).

### Identity
- [ ] **Tab favicon** — pulse monogram in v2 blue (`#3b82f6`) renders crisp at 16×16
- [ ] **Apple touch icon** — same monogram on rounded-square obsidian for iOS home-screen pin
- [ ] **OG unfurl** — paste prod URL into Slack draft channel; preview card shows monogram + `Grid|Pulse` wordmark + tagline
- [ ] **Twitter / X unfurl** — same URL renders the full 1200×630 image card
- [ ] **`Grid|Pulse` wordmark in header** — `Grid` neutral, `Pulse` blue accent

### Overview tab
- [ ] **Title block** with region name + 1-line subtitle
- [ ] **5-up MetricsBar**: `Now` (hero) / `7d Peak` / `7d Low` / `Average` / `24h Trend`
- [ ] **Hero forecast chart** — 7d actual demand (blue) + 24h dashed forecast (orange) + confidence band (orange-tinted)
- [ ] **ModelMetricsCard** — horizontal bar, MAPE/RMSE/MAE/R² + `trained|simulated` badge
- [ ] **InsightCard** — 3-sentence narrative; persona affects only the eyebrow
- [ ] **Footer** — data sources row
- [ ] _Cuts verified visually_: no greeting card, no AI briefing card, no what-changed, no data-health card, no quick-nav, no insight digest, no news ribbon

### Forecast tab
- [ ] **Title** with region-aware subtitle
- [ ] **Segmented horizon control** — `24h | 7d | 30d` pills, active state has `bg-hover`
- [ ] **Segmented model selector** — `XGBoost | Prophet | ARIMA | Ensemble`
- [ ] **Hero forecast chart** in `.gp-chart-card` (380px)
- [ ] **"Forecast as of" mono timestamp chip** tucked under the chart
- [ ] **4-up MetricsBar** — Peak Demand (hero) / Average / Min / Range
- [ ] **ModelMetricsCard**
- [ ] **InsightCard**
- [ ] **Toggle strip**: `+ Drivers / Weather`, `+ Generation / Fuel mix`, `+ Scenarios / What-if`
- [ ] **Drivers panel (open)** — 3-up Temperature / Wind / Solar with current value, delta-vs-24h-avg, 60px sparkline
- [ ] **Generation panel (open)** — 3-up sub-MetricsBar (Net Load avg / Renewable Share / Largest Source) + emissions-ordered stacked area chart
- [ ] **Scenarios panel (open)** — 5 preset chips + 3 sliders + 4-up delta KPIs + baseline-vs-scenario chart
  - [ ] Click a preset chip → sliders snap to that preset's deltas
  - [ ] Slider readout updates instantly (clientside)
  - [ ] Chart redraws on slider release (`updatemode="mouseup"`)

### Risk tab
- [ ] **Title** with region-aware subtitle
- [ ] **Risk MetricsBar** — Stress Score (hero) + Components breakdown
- [ ] **Severity timeline** — alert cards with `border-left` thickness coding severity (4px critical / 3px warning / 2px info)
- [ ] **Weather strip** with current conditions
- [ ] **Hero anomaly chart** (320px)
- [ ] **2-up secondary**: Temperature Exceedance + Historical Events
- [ ] **InsightCard**

### Models tab
- [ ] **Title** with region-aware subtitle
- [ ] **Models leaderboard MetricsBar** — 4–5-up with hero highlight on lowest-MAPE model, MAE in sub-line, semantic tones (green ≤2.5%, neutral ≤5%, red >5%)
- [ ] **Compare-Models multi-select** — checked options have `accent-dim` background
- [ ] **Metrics table** with v2 styling (zebra hover, mono numerics, eyebrow column headers)
- [ ] **Residuals 3-up grid** (Time / Distribution / vs Predicted)
- [ ] **2-up**: Error-by-Hour heatmap + SHAP feature importance
- [ ] **InsightCard**

### Briefing Mode (R5c)
- [ ] Click `Briefing Mode` → label flips to `Exit Briefing` (cyan accent)
- [ ] Tab strip hides
- [ ] Hero KPI scales to 56px
- [ ] Section gap doubles to 48px
- [ ] Watermark `GridPulse` appears below the header strip (lower-right)
- [ ] Panel toggle chips, freshness badges, modebar all hide
- [ ] Click `Exit Briefing` → all chrome restores; label flips back

### Micro-craft (R5b)
- [ ] **Skip-to-content link** — `Tab` from address bar → "Skip to main content" pill appears top-left; Enter lands focus on `<main>`
- [ ] **Selection** — highlight any text → background tints blue (`accent-dim`)
- [ ] **Scrollbar** — scroll any overflow region → custom scrollbar (10px thumb on `bg-elevated`)
- [ ] **Caret** — click into a slider/input → caret blinks blue (`accent-base`)
- [ ] **Cursor** — hover any interactive class → `pointer`; hover `:disabled` → `not-allowed` + 60% opacity

### Reduced motion
- [ ] macOS System Settings → "Reduce motion" → animations collapse to ≤0.01ms; layout still works

### Cross-browser
- [ ] Chrome / Edge — all of the above
- [ ] Safari — same; verify scrollbar styling falls back gracefully (Safari uses `::-webkit-scrollbar`)
- [ ] Firefox — verify `scrollbar-color` rule kicks in

## R-phase commit log (chronological)

| R-phase | Commit / PR | What |
|---|---|---|
| R1 | [#36](https://github.com/kristenmartino/gridpulse/pull/36) | v2 token system + repaint favicon to blue |
| R2 | [#37](https://github.com/kristenmartino/gridpulse/pull/37) | Overview rebuild — cut 8 cards, ship 7-section linear stack |
| R3 | [#38](https://github.com/kristenmartino/gridpulse/pull/38) | Tab consolidation 9 → 4 + header rebuild |
| R4a-1 | [#39](https://github.com/kristenmartino/gridpulse/pull/39) | Forecast tab v2 linear stack core |
| R4a-2 | [#40](https://github.com/kristenmartino/gridpulse/pull/40) | Inline panel infrastructure + Drivers panel |
| R4a-3 | [#41](https://github.com/kristenmartino/gridpulse/pull/41) | Generation panel content |
| R4a-4 | [#42](https://github.com/kristenmartino/gridpulse/pull/42) | Scenarios panel content |
| R4b | [#43](https://github.com/kristenmartino/gridpulse/pull/43) | Risk tab v2 linear stack |
| R4c | [#44](https://github.com/kristenmartino/gridpulse/pull/44) | Models tab v2 linear stack + leaderboard |
| R5a | [#45](https://github.com/kristenmartino/gridpulse/pull/45) | OG image + Twitter card meta |
| R5b | [#47](https://github.com/kristenmartino/gridpulse/pull/47) | Icon system + landmarks + micro-craft |
| R5c | [#48](https://github.com/kristenmartino/gridpulse/pull/48) | Briefing Mode chrome + watermark |
| R6 | _this PR_ | Verification — smoke tests + this checklist |
| _hotfix_ | [#46](https://github.com/kristenmartino/gridpulse/pull/46) | Rename duplicate `tab3-insight-card` (R4c regression) |

## Outstanding craft cuts (post-R6 backlog)

- Replace remaining emoji in `components/error_handling.py:267-273` (api-error config dict — low-traffic state)
- Replace freshness/banner emoji icons at `components/callbacks.py:3820-3873`
- Remove the now-fully-folded `components/tab_backtest.py`, `tab_forecast.py` (Historical), `tab_generation.py`, `tab_weather.py`, `tab_simulator.py` files once their callbacks have been verified migrated/dead
- Brand spec doc (`docs/gridpulse_brand_system_spec.md`) still references cyan/blue/teal — needs an addendum noting v2 blue/orange supersedes
- PRD's tab list documents 9 — needs an addendum noting v2 reduction to 4
