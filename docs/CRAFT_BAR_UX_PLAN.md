# GridPulse — Craft-Bar UX & Performance-as-Design Pass

> **Handoff doc.** Drop this into a fresh Claude Code session along with `CLAUDE.md` to resume work.
> Original plan saved here verbatim. Status block at the top tracks completed commits + open issues.

---

## Status (as of commit `4f9ab4b`)

### Completed
| # | Phase | Commit | Title |
|---|---|---|---|
| 15 | A.2-3, A.5 | `02451d6` | feat(ui): expand motion + focus tokens, dual-layer ring |
| 16 | A.1, A.7 | `1f146dc` | feat(ui): typographic refinement (tracking, feature settings, fluid display) |
| 17 | A.6 | `0de671b` | perf(assets): preconnect to fonts + GA4 |
| 18 | C.1, E.1-3 | `6410c48` + `4f9ab4b` | refactor(callbacks): _layout helper + uirevision across all figures |

### Next up (in order)
| # | Phase | Title | GH issue |
|---|---|---|---|
| 19 | C.2-3 | perf(callbacks): cache-key-before-hash + lazy tab guards | https://github.com/kristenmartino/gridpulse/issues/19 |
| 20 | B.1 | feat(ui): sticky header + sticky tab strip | https://github.com/kristenmartino/gridpulse/issues/20 |
| 21 | B.2-4 | feat(state): tab in URL + Save View → clipboard + toast | https://github.com/kristenmartino/gridpulse/issues/21 |
| 22 | B.5 | feat(ui): command palette (⌘K) | https://github.com/kristenmartino/gridpulse/issues/22 |
| 23 | B.6-7 | feat(ui): kbd chips + Briefing Mode chrome rules | https://github.com/kristenmartino/gridpulse/issues/23 |
| 24 | C.5 | perf(http): flask-compress + asset cache headers | https://github.com/kristenmartino/gridpulse/issues/24 |
| 25 | C.4 | perf(callbacks): briefing background callback + diskcache | https://github.com/kristenmartino/gridpulse/issues/25 |
| 26 | E.2-6 | feat(charts): PLOT_CONFIG modebar trim + hover/axis polish | https://github.com/kristenmartino/gridpulse/issues/26 |
| 27 | C.6 | perf(charts): LTTB downsample for 90d | https://github.com/kristenmartino/gridpulse/issues/27 |
| 28 | D | feat(ui): segmented controls in Forecast + Outlook | https://github.com/kristenmartino/gridpulse/issues/28 |
| 29 | D | feat(ui): backtest table cross-highlight | https://github.com/kristenmartino/gridpulse/issues/29 |
| 30 | D | feat(ui): hero overview + sticky KPI row | https://github.com/kristenmartino/gridpulse/issues/30 |
| 31 | D | feat(ui): hand-tune generation, weather, models, alerts, simulator | https://github.com/kristenmartino/gridpulse/issues/31 |
| 32 | F | test: lttb + url state unit coverage | https://github.com/kristenmartino/gridpulse/issues/32 |
| 33 | F | Smoke test + perf measurement | https://github.com/kristenmartino/gridpulse/issues/33 |

Tracking issue: https://github.com/kristenmartino/gridpulse/issues/34

### Operating mode
- One commit per issue, conventional-commit format (`type(scope): description`)
- Each commit must leave the app booting cleanly and CI green (`ruff format --check`, `ruff check`, `pytest`)
- Push only on explicit user request
- Helpers/conventions established so far:
  - `_layout(*, uirevision=None, **overrides)` in `components/callbacks.py:96` — use for all new figure layouts
  - `uirevision` keying:
    - region-only callbacks → `uirevision=region`
    - horizon-aware callbacks → `uirevision=f"{region}:{horizon_hours}"`
    - model-aware callbacks → `uirevision=f"{region}:{horizon_hours}:{model_name}"`
    - selection-aware (model diagnostics) → `uirevision=f"{region}:{','.join(sorted(selected))}"`
    - empty/error placeholders → `uirevision="empty"`

---

## 0. Baseline (verified first-hand)

| Area | Current | Reference |
|---|---|---|
| Token system | `:root` ladder for color/spacing/radius/shadow + `cubic-bezier(0.16, 1, 0.3, 1)` easing | `assets/custom.css:10-93` |
| Fonts | Inter 400/500/600/700 + Sora 600/700, `display=swap` | `assets/custom.css:7` |
| Header | Non-sticky, has `backdrop-filter: blur(8px)` | `components/layout.py:37-129` |
| Tabs | `dbc.Tabs`, 9 pre-rendered, no URL sync | `components/layout.py:144-194` |
| Keyboard | Alt+1–8 / Alt+R / Alt+P **wired** in JS | `assets/accessibility.js:4-41` |
| Cmd palette | None | — |
| Plotly | 47× `go.Scatter`, 0× `uirevision`, 0× `Scattergl`, layout dict copied per figure | `components/callbacks.py:76-92, 1047+` |
| Callbacks | 32 decorators, 6 250 LOC, briefing is sync inside Overview | `components/callbacks.py:2260-2340` |
| Cache strategy | Redis → SQLite → compute, in-memory `_*_CACHE` dicts; **hash computed before cache check** | `components/callbacks.py:484-492` |
| Gunicorn | 2 workers × 2 gthread, no preload, **no gzip, no asset hashing, no cache headers** | `Dockerfile:61-68` |
| URL state | `region`, `persona` only (no `tab`) | `components/callbacks.py` (sync_region_from_url) |
| Toasts | `bookmark-toast` div fixed top-right, no dismiss/animation | `components/layout.py:131-134` |
| Briefing Mode / Save View | Buttons exist, behavior inert / partially wired | `components/layout.py:96-113` |

> Note: items above tagged "Current" reflect the baseline _before_ any work. Phase A items (motion tokens, focus ring, fonts) are now updated — see `assets/custom.css` HEAD.

---

## 1. Phase plan (small, safe increments)

The pass is built in 6 phases. Each phase ends with the app running cleanly and visually coherent. Phases A–C are inheritance-style: every tab benefits with no per-tab edit. Phase D is hand-tuning. Phases E–F are chart and verification.

### Phase A — Foundation polish (token-only, zero callback risk) ✅ DONE

Touches `assets/custom.css` only.

1. **Tracking & rhythm** — add tighter tracking to display sizes:
   - `.dashboard-title`: `letter-spacing: -0.02em` (currently `-0.01em`)
   - Add `--text-display: clamp(20px, 1.6vw + 12px, 28px)` and apply to KPI value + section h1.
   - Add `font-feature-settings: 'cv11','ss01','ss03'` to `body` for Inter's stylistic alternates (the "Linear feel").
2. **Motion vocabulary** — split the single `--ease` token into a small set:
   ```css
   --ease-out-quint: cubic-bezier(0.22, 1, 0.36, 1);   /* enter, default */
   --ease-in-quint:  cubic-bezier(0.64, 0, 0.78, 0);   /* exit, faster */
   --ease-spring:    cubic-bezier(0.16, 1, 0.3, 1);    /* keep — current */
   --ease-emphasized: linear(0, 0.009 1.4%, 0.084 4.4%, 0.36 11.1%, 0.679 19.7%,
                              0.917 28.7%, 1.061 38.4%, 1.121 49.7%, 1.097 60.7%,
                              1.045 71.7%, 1.005 84.7%, 0.992 100%); /* CSS linear() spring */
   ```
   Use `--ease-emphasized` for ⌘K and toast slide-in (CSS `linear()` is supported in modern Chromium/Safari/FF; degrades to `--ease-spring` via `@supports`).
3. **Focus ring upgrade** — dual-layer ring that respects radius:
   ```css
   *:focus-visible {
       outline: none;
       box-shadow:
           0 0 0 2px var(--bg-base),                   /* gap */
           0 0 0 4px var(--accent-base);                /* ring */
   }
   ```
   Drops the awkward `outline-offset` on rounded corners; matches Radix/Vercel ring behavior.
4. **Hover micro-feedback** — replace generic `transform: translateY(-1px)` on `.kpi-card` with a sharper recipe:
   `transform: translateY(-1px); border-color: var(--border-strong); box-shadow: var(--shadow-md);`
   transition `all 120ms var(--ease-out-quint)` on enter, `all 80ms var(--ease-in-quint)` on leave (asymmetric — feels "snappy back").
5. **Sub-pixel borders** — replace `1px solid var(--border-default)` on cards with `box-shadow: 0 0 0 1px var(--border-default)` so hover/focus rings can stack without layout shift.
6. **Font preconnect** — the only HTML hook is `app.index_string`; add `<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>` to remove the 100–200ms font-fetch RTT.
7. **Skeleton style refinement** — current shimmer is good; tighten gradient stops to 30/50/70 and slow to 1.8s for less "jittery" feel.

**Files:** `assets/custom.css`, `app.py` (preconnect only).

---

### Phase B — Shell craft (high-leverage UX)

Touches `components/layout.py`, `components/callbacks.py`, plus a new `assets/02-craft.js`.

1. **Sticky chrome.** Make `.dashboard-header` `position: sticky; top: 0; z-index: 50;` and put the tab strip in its own sticky container right under it:
   ```css
   .dashboard-header { position: sticky; top: 0; z-index: 50; }
   .nav-tabs { position: sticky; top: var(--header-height, 64px); z-index: 40;
               background: var(--bg-overlay); backdrop-filter: blur(8px); }
   ```
   The header already has the blur. KPI row stays in-flow.
2. **URL ↔ tab sync (C2 completion).** Add a `tab` query param. Add one new callback in `components/callbacks.py`:
   ```python
   @app.callback(
       Output("dashboard-tabs", "active_tab", allow_duplicate=True),
       Input("url", "search"),
       prevent_initial_call="initial_duplicate",
   )
   def sync_tab_from_url(search):
       tab = _parse_qs(search).get("tab")
       return tab if tab in _TAB_IDS else no_update

   @app.callback(
       Output("url", "search", allow_duplicate=True),
       Input("dashboard-tabs", "active_tab"),
       Input("region-selector", "value"),
       Input("persona-selector", "value"),
       prevent_initial_call=True,
   )
   def write_state_to_url(tab, region, persona):
       return urlencode({"tab": tab, "region": region, "persona": persona})
   ```
   This finishes the "save view" workflow — Save View button copies `window.location.href`.
3. **Save View → real action.** Wire `bookmark-btn` to a clientside callback that copies URL to clipboard and triggers a toast:
   ```python
   app.clientside_callback(
       """
       function(n) {
           if (!n) return window.dash_clientside.no_update;
           navigator.clipboard.writeText(window.location.href);
           return {visible: true, message: 'View copied to clipboard', kind: 'success', ts: Date.now()};
       }""",
       Output("toast-store", "data"),
       Input("bookmark-btn", "n_clicks"),
       prevent_initial_call=True,
   )
   ```
4. **Toast system rebuild.** Replace the static `bookmark-toast` div with a single dispatcher:
   - new `dcc.Store(id="toast-store")`
   - `<div id="toast-host">` rendered by a small clientside callback that listens to `toast-store` and animates in/out:
     ```css
     .toast { transform: translateY(-12px); opacity: 0;
              transition: transform 220ms var(--ease-emphasized), opacity 160ms var(--ease-out-quint); }
     .toast.in { transform: translateY(0); opacity: 1; }
     ```
   - Auto-dismiss 3.2s; ARIA `role="status"`; keyboard-dismissible with Esc.
5. **Command palette (⌘K).** New file `assets/02-palette.js` (additive, doesn't touch `accessibility.js`). Behavior:
   - `Cmd/Ctrl+K` opens an overlay with a fuzzy-matched list:
     - 8 regions (jump = set `#region-selector` value, dispatch change)
     - 4 personas
     - 9 tabs
     - 5 scenario presets (read from a `data-palette-target` on each preset button)
     - Theme/mode toggles (Briefing Mode, Reduced motion)
   - Tokens-driven CSS in `assets/custom.css`:
     ```css
     .palette { position: fixed; inset: 0; display: grid; place-items: start center; padding-top: 14vh;
                background: rgba(0,0,0,0.55); backdrop-filter: blur(6px); z-index: 100; }
     .palette__panel { width: min(640px, 92vw); background: var(--bg-raised);
                       border: 1px solid var(--border-default); border-radius: var(--radius-xl);
                       box-shadow: var(--shadow-md), 0 24px 80px rgba(0,0,0,0.5);
                       transform: translateY(-8px) scale(0.98); opacity: 0;
                       transition: transform 220ms var(--ease-emphasized), opacity 160ms var(--ease-out-quint); }
     .palette.open .palette__panel { transform: none; opacity: 1; }
     .palette__row[aria-selected="true"] { background: var(--bg-hover); }
     ```
   - Arrow key navigation, `Enter` to select, `Esc` to close, all focus-trapped. Sub-100ms perceived: no Dash round-trip on open; selection sets DOM values then dispatches Dash's native change event.
6. **Header hierarchy.** Replace the inline labels ("Balancing Authority", "View") with subtle `<kbd>R</kbd>` / `<kbd>P</kbd>` chips so power users see the shortcuts inline. Use `--font-mono`, `font-size: 10px`, `padding: 1px 5px`, `border: 1px solid var(--border-default)`.
7. **Briefing Mode actually does something.** Add CSS rules for `.dashboard-header.meeting-mode` (smaller, no buttons) and `body.meeting-mode .nav-tabs` (compact). Toggled via existing `meeting-mode-store`.

**Files:** `components/layout.py` (header restructure, sticky tab container, toast host, palette host), `components/callbacks.py` (URL sync, toast dispatch, persona/region clientside listeners), `assets/custom.css` (sticky, palette, toast, kbd chips, meeting-mode rules), new `assets/02-palette.js`.

---

### Phase C — Performance as design

Goal: shave 250–500ms off every tab/region/persona switch and remove blocking work from first paint.

1. **`uirevision` everywhere.** ✅ DONE (#18). Helper at `components/callbacks.py:96`:
   ```python
   def _layout(*, uirevision: str | None = None, **overrides) -> dict:
       layout = {**PLOT_LAYOUT, **overrides}
       if uirevision is not None:
           layout["uirevision"] = uirevision
       return layout
   ```
   All 51 inline `**PLOT_LAYOUT` calls replaced with `**_layout(uirevision=...)`.
2. **Cache key before hash.** Reorder the early-return in `_run_forecast_outlook` and twins (`callbacks.py:484-492`):
   ```python
   cache_key = (region, horizon_hours, model_name)
   cached = _PREDICTION_CACHE.get(cache_key)
   if cached and (time.time() - cached[3]) < CACHE_TTL_SECONDS:
       return {"timestamps": cached[1], "predictions": cached[0]}
   data_hash = _compute_data_hash(demand_df, weather_df, region)  # only on miss
   ```
3. **Lazy tab rendering.** Today every tab callback re-runs on `demand-store` change. Add `Input("dashboard-tabs", "active_tab")` and an early `if active_tab != "tab-x": return no_update` to the ~7 callbacks that lack it. (Some already have it — only fix the missing ones.)
4. **Background callback for the briefing.** Split `update_overview_tab`:
   - sync part: greeting, data-health, spotlight chart (fast, <80ms)
   - new `update_overview_briefing` decorated with `@app.callback(..., background=True)` writes only `overview-briefing.children`. Skeleton shows until it lands.
   - Requires `DiskcacheManager` configured in `app.py`:
     ```python
     import diskcache, dash
     cache_dir = os.getenv("DASH_BG_CACHE", "/tmp/dash-bg")
     bg_manager = dash.DiskcacheManager(diskcache.Cache(cache_dir))
     app = dash.Dash(..., background_callback_manager=bg_manager)
     ```
   - In production (Cloud Run), prefer `CeleryManager` only if Redis is already wired; otherwise diskcache on `/tmp` is acceptable and survives within a worker.
5. **Gzip + asset versioning.** Add to `requirements.txt`: `flask-compress`. In `app.py`:
   ```python
   from flask_compress import Compress
   Compress(server)
   ```
   For cache-busted assets, use Dash's built-in `assets_url_path` + a content hash suffix. Simplest: a `before_request` that adds `Cache-Control: public, max-age=31536000, immutable` for `/assets/*` and `Cache-Control: no-cache` for `/_dash-update-component`.
6. **Downsample large traces.** For `tab1-timerange == "2160"` (90d × hourly = 2160 points), apply LTTB downsampling to ≤720 visible points before sending to the wire. New helper in `data/preprocessing.py`:
   ```python
   def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int = 720) -> tuple[np.ndarray, np.ndarray]: ...
   ```
   ~80% wire-size reduction on the 90d view; visually indistinguishable.
7. **Modebar trim.** Add to `_layout`:
   ```python
   PLOT_CONFIG = dict(
       displaylogo=False,
       modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
       responsive=True,
   )
   ```
   Apply via `dcc.Graph(config=PLOT_CONFIG, ...)` in tab layouts. Cleaner UI, smaller modebar.
8. **Defer non-visible callbacks for `weather-store`/`generation-store`** behind `active_tab` guards (same pattern as #3).
9. **Preconnect** to fonts (covered in A.6) ✅ and to GA4 (`https://www.googletagmanager.com`) ✅.

**Files:** `components/callbacks.py` (helper, uirevision, cache reorder, lazy guards, briefing split), `app.py` (Compress, bg_manager, cache-control middleware, preconnect), `requirements.txt` (`flask-compress`, `diskcache`), `data/preprocessing.py` (LTTB), every `components/tab_*.py` for `dcc.Graph(config=PLOT_CONFIG)`.

---

### Phase D — Per-tab hand-tuning (all 9)

Each tab gets the same recipe applied with surface-specific judgement: clean section headers, deliberate spacing rhythm, reactive controls, polished empty/loading states. The `_layout()` + `PLOT_CONFIG` from Phase C are mandatory.

| Tab | File | Specific work |
|---|---|---|
| **Overview** | `components/tab_overview.py` | Re-balance grid: hero spotlight chart full-width, KPI strip becomes 4-up sticky row below header. Briefing card uses fade-in already; expand skeleton to match final layout shape. Add quick-nav cards with proper hover (lift + accent border-left). |
| **Historical Demand** | `components/tab_forecast.py` | Time-range RadioItems → segmented control style (`.segmented`, full pill, animated bg via `:has(:checked)` or JS-painted indicator). LTTB downsample on 90d. Weather overlay toggle moves into chart toolbar. |
| **Demand Forecast** | `components/tab_demand_outlook.py` | Horizon selector → segmented. Confidence bands keep 12% opacity; legend → top-right inside plot at 92% opacity to avoid bottom-margin steal. Add inline "Updated 3 min ago" chip pulled from `data-freshness-store`. |
| **Backtest** | `components/tab_backtest.py` | Metrics table → real `<table>` with sticky header, zebra rows at 2% white, hover highlights row + the corresponding model trace via `customdata` + a clientside callback (cross-highlight). |
| **Generation & Net Load** | `components/tab_generation.py` | Stacked area: tighten color ramp (sort fuels by emissions intensity, not alphabetic). Add legend toggle persistence via `uirevision`. Renewable share KPI gets a tiny inline sparkline (Plotly `Scatter`, no axes). |
| **Weather Correlation** | `components/tab_weather.py` | 3 stacked figures share x-axis via `xaxis2="x"`. Scatter-density (heatmap) for hour-of-day × temperature: replace categorical heatmap with `go.Heatmap` + `colorscale=[(0,'#11141c'),(1,'#38D0FF')]`. |
| **Model Diagnostics** | `components/tab_models.py` | Per-model cards with mini-MAPE bar (success/warn/danger via `mape_grade`). Residuals plot keeps colorblind palette + dash patterns. SHAP bars use `--accent-base` only (not full color rotation — too noisy). |
| **Extreme Events** | `components/tab_alerts.py` | Severity-coded vertical timeline (border-left thickness drives perceived weight). Empty-state copy: "All systems nominal." Subscribe button stub OK. |
| **Scenario Simulator** | `components/tab_simulator.py` | Sliders → debounced (250ms) clientside store before triggering recompute. Preset buttons get keyboard arrow-cycle + Enter to apply. Impact KPIs use the same delta chip styling as Overview KPIs (consistent vocabulary). |

**Files:** all `components/tab_*.py` and corresponding callback sections. Each tab change is small, atomic, reviewable independently.

---

### Phase E — Plotly chart polish (cross-cutting)

Goes after D so the helper helpers exist.

1. **Single layout helper** (`_layout` from C.1) — kills 30 inline copies. ✅ DONE (#18).
2. **Hover label theming.** Append to `PLOT_LAYOUT`:
   ```python
   hoverlabel=dict(
       bgcolor="#11141c",
       bordercolor="rgba(255,255,255,0.10)",
       font=dict(family="Inter, system-ui", size=12, color="#F5F7FA"),
       align="left",
   ),
   hovermode="x unified",
   ```
3. **Axis polish.** `gridcolor="rgba(255,255,255,0.04)"`, `zerolinecolor="rgba(255,255,255,0.08)"`, `linecolor="rgba(255,255,255,0.10)"`, `tickfont=dict(color="#8892A5", size=11)`.
4. **Line widths.** Move from 2 → 1.75 for forecast lines, 2.25 for actual demand. Subtle but reads more refined.
5. **Title-less charts** — use the surrounding card's `.chart-title`, drop Plotly's title. Removes a 24px vertical eat.
6. **Legend.** Top-right inside plot, transparent bg, `borderwidth=0`. The current bottom-orientation steals scroll space.

---

### Phase F — Verification

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `python app.py` → smoke load, click each of the 9 tabs.
3. Manual checks (≤ 5 min):
   - ⌘K opens palette in <16ms (one frame); arrow keys + Enter work.
   - Alt+1..8, Alt+R, Alt+P still work.
   - Tab switch persists in URL (`?tab=…&region=…&persona=…`); refresh restores state.
   - Save View → toast appears + auto-dismisses; clipboard contains URL.
   - Briefing card shows skeleton, then content lands without re-rendering the spotlight chart (uirevision held).
   - `prefers-reduced-motion: reduce` (DevTools rendering tab) — palette/toast still work, no springs.
   - Lighthouse: First-Contentful-Paint < 1.0s on local; gzip visible in DevTools Network panel; assets carry `cache-control: max-age=31536000`.
4. Tests:
   - `pytest tests/unit -v` (must stay green; no callback signature changes that break tests).
   - Add `tests/unit/test_lttb_downsample.py` covering monotonic timestamps + length contract.
   - Add `tests/unit/test_url_state.py` covering `_parse_qs`, validation against `_TAB_IDS`/`REGION_NAMES`/`PERSONAS`.
5. `pytest tests/e2e -v` to confirm tab rendering still works (existing e2e covers tab_id values).
6. Manual perf measurement: in DevTools Performance tab, region change → record → confirm scripting < 100ms and JS-thread idle within 200ms (down from current ~500–800ms).

---

## 2. File-by-file change manifest

| File | Phase(s) | Type | Notes |
|---|---|---|---|
| `assets/custom.css` | A, B | Edit | Token additions, focus ring, sticky chrome, palette/toast styles, segmented control, kbd chips, meeting-mode |
| `assets/02-palette.js` | B | New | ⌘K palette, ~150 LOC, vanilla JS, no deps |
| `assets/03-toast.js` | B | New | Toast renderer reading from `toast-store`, ~60 LOC |
| `app.py` | A, C | Edit | Preconnect tags, Compress, background callback manager, cache-control middleware |
| `Dockerfile` | C | Edit | Add `--worker-class gthread --threads 4` (was 2 → 4 to absorb bg callback diskcache I/O) |
| `requirements.txt` | C | Edit | `flask-compress>=1.14`, `diskcache>=5.6` |
| `components/layout.py` | B | Edit | Header restructure, sticky tab container, kbd chips, toast-host, palette-host, toast-store |
| `components/callbacks.py` | B, C | Edit | `_layout()` helper, `PLOT_CONFIG`, URL↔tab sync, toast dispatch, lazy guards, briefing background callback, cache reorder |
| `components/tab_overview.py` | D | Edit | Hero layout, sticky KPI row, skeleton parity |
| `components/tab_forecast.py` | D | Edit | Segmented control, LTTB on 90d, weather overlay chip |
| `components/tab_demand_outlook.py` | D | Edit | Segmented horizon, top-right legend, freshness chip |
| `components/tab_backtest.py` | D | Edit | Real `<table>`, cross-highlight clientside callback |
| `components/tab_generation.py` | D | Edit | Fuel ordering, sparkline, uirevision |
| `components/tab_weather.py` | D | Edit | Shared x-axis, dark heatmap colorscale |
| `components/tab_models.py` | D | Edit | Per-model card grid, MAPE grade colors, SHAP polish |
| `components/tab_alerts.py` | D | Edit | Vertical timeline |
| `components/tab_simulator.py` | D | Edit | Debounced sliders, keyboard preset cycling, delta-chip vocabulary |
| `data/preprocessing.py` | C | Edit | `lttb_downsample()` helper |
| `tests/unit/test_lttb_downsample.py` | F | New | ~30 LOC |
| `tests/unit/test_url_state.py` | F | New | ~50 LOC |

`assets/accessibility.js` — **left untouched**. Existing keyboard shortcuts and ARIA observer are working; the new palette is in a separate file.

---

## 3. Risk register

| Risk | Mitigation |
|---|---|
| `uirevision` accidentally freezes a chart that should refresh | Tie `uirevision` to data identity (e.g. `f"{region}:{horizon}"`) — changes when the user-meaningful state changes, stable when only data values change |
| Background callback manager on Cloud Run uses `/tmp` (ephemeral) | Diskcache survives within a worker; not a correctness issue. If multi-worker invalidation matters later, swap to Celery on Redis |
| `flask-compress` adds CPU on small payloads | Default `min_size=500` keeps it off short responses; figure JSON is exactly the win we want |
| Sticky header breaks tall-laptop layouts | Already responsive in `@media (max-width: 768px)`; verify scroll-behavior on 13" and 4K |
| URL sync triggers feedback loops | `prevent_initial_call="initial_duplicate"` + only writing to `url.search` on user-driven Inputs (not Outputs) |
| `Cmd+K` collides with Chrome's address bar focus | Use `Ctrl+K` only in browsers that already shadow it (Chrome does); listen with `e.preventDefault()` |
| Plotly modebar config affects tests | Existing e2e tests do not assert modebar; verified by reading `tests/TEST_PYRAMID.md` |
| Touching 9 tab files at once = bigger blast radius | Each tab change is independent and self-contained; commit per tab so revert is cheap |

---

## 4. Execution order (commit cadence)

Each bullet = one commit, in order:

1. ✅ `feat(ui): expand motion + focus tokens, dual-layer ring`  *(Phase A.2-3, A.5)* — `02451d6`
2. ✅ `feat(ui): typographic refinement (tracking, feature settings, fluid display)`  *(Phase A.1, A.7)* — `1f146dc`
3. ✅ `perf(assets): preconnect to fonts + GA, refine skeleton`  *(Phase A.6, A.7)* — `0de671b`
4. ✅ `refactor(callbacks): _layout helper + uirevision across all figures`  *(Phase C.1, E.1-3)* — `6410c48`
5. `perf(callbacks): cache-key-before-hash; lazy tab guards`  *(Phase C.2-3)* — **NEXT**
6. `feat(ui): sticky header + sticky tab strip`  *(Phase B.1)*
7. `feat(state): tab in URL + Save View → clipboard + toast`  *(Phase B.2-4)*
8. `feat(ui): command palette (⌘K)`  *(Phase B.5)*
9. `feat(ui): kbd chips + Briefing Mode chrome rules`  *(Phase B.6-7)*
10. `perf(http): flask-compress + asset cache headers`  *(Phase C.5)*
11. `perf(callbacks): briefing background callback + diskcache`  *(Phase C.4)*
12. `feat(charts): PLOT_CONFIG modebar trim + hover/axis polish`  *(Phase E.2-6)*
13. `perf(charts): LTTB downsample for 90d`  *(Phase C.6)*
14. `feat(ui): segmented controls in Forecast + Outlook`  *(Phase D, two tabs)*
15. `feat(ui): backtest table cross-highlight`  *(Phase D, backtest tab)*
16. `feat(ui): hero overview + sticky KPI row`  *(Phase D, overview)*
17. `feat(ui): hand-tune generation, weather, models, alerts, simulator`  *(Phase D, batched 5 tabs)*
18. `test: lttb + url state unit coverage`  *(Phase F)*

Any single commit can be reverted without breaking the others — Phases A and C-1/2/3 are pure refinements, B and D each gate on their own surface.

---

## 5. Verification snapshot (what "done" looks like)

- ⌘K → palette opens in one frame; arrow keys cycle results; Enter switches tab/region/persona/preset.
- Tab switch perceived latency drops from ~500ms to ~150ms; legend zoom/pan persists across data refresh.
- `?tab=tab-simulator&region=ERCOT&persona=trader` deep-links restore full state on reload.
- Briefing skeleton shows, then content lands without flickering the spotlight chart.
- Toasts slide in, auto-dismiss in 3.2s, dismissable with Esc.
- DevTools Network: HTML/CSS/JS gzipped; `/assets/*` carries `Cache-Control: public, max-age=31536000, immutable`.
- DevTools Performance: scripting time on region change < 100ms.
- Lighthouse a11y ≥ 95; reduced-motion users get instant transitions but full functionality.
- All existing tests green; new lttb + url-state unit tests pass.

---

## 6. How to resume in a fresh Claude Code session

1. Open the repo: `cd /Users/rootk/nextera-portfolio/energy-forecast/energy-forecast-final`
2. Tell Claude:
   > Read `CLAUDE.md` and `docs/CRAFT_BAR_UX_PLAN.md`. We're working through the craft-bar UX pass commit-by-commit. Pick up at the next unchecked item in the Execution Order (section 4). Work atomically: one commit per bullet, conventional-commit format, ensure `ruff format --check` and `ruff check` pass before commit, do not push until I say so.
3. After each commit:
   - Run `ruff format --check . && ruff check .`
   - Run `pytest tests/unit -q` (full suite if anything risky)
   - Show the diff summary
4. Push only on explicit user command. CI must be green before moving to the next item.
