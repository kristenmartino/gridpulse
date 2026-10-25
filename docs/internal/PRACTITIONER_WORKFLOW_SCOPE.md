# GridPulse practitioner workflow scope

Date: 2026-10-25

Status: evidence audit A and foundation B deployed and production-verified on 2026-10-25; see `docs/audits/FORECAST_EVIDENCE_2026-10-25.md`. Prospective publication capture is verified for the three pilot regions. Historical workflow C remains gated on adequate verified history. Estimates for remaining work are planning ranges, not delivery commitments.

## 1. Product decision

Working audience: an analyst reviewing regional electricity demand and forecast performance using public data. Power-market research teams are a recruiting hypothesis; utility load forecasters are also valuable domain advisers. Neither group is an established GridPulse customer segment.

First task to test: select a historical operating day, examine the forecasts actually available for it, identify the largest supported misses and data limitations, and prepare a reproducible investigation note.

Primary question: **Where did this demand forecast miss, and what needs follow-up?**

The existing forward outlook remains available as a separate task. Describing a rolling 24-hour forecast as tomorrow's operating-day forecast is outside the contract.

Proposed workflow:

1. Select balancing authority, operating day, forecast series, and forecast selection rule.
2. Establish forecast provenance, target-hour coverage, and the actual-demand version used for evaluation.
3. Inspect paired forecast/actual values, signed errors, peaks, ramps, and missing hours.
4. Review weather and data-quality context as possible explanations; record unresolved questions.
5. Export the evidence and an analyst-authored conclusion with explicit follow-up actions.

Success means an analyst can perform and explain this task with an identifiable work product. Practitioner fit, time savings, willingness to pay, and business impact remain open questions.

## 2. What exists and what needs verification

Code inspection baseline: app commit `d99c7286379ddd0773dec077a67d15c7f1b93e72`; portfolio case-study source in the `portfolio-gridpulse-case-study` checkout. Recheck current main, open work, and actual deployment state before implementation.

| Existing area | Evidence in source | Scope implication |
|---|---|---|
| Published/default versus selected model identity | `data/forecast_identity.py`, `components/forecast_review.py` | Extend the existing resolver; identify GridPulse as producer rather than creating another source-label path. |
| Same-series observed MAPE and separate training reference | `components/forecast_context.py` | Preserve matching by series and lead. The current minimum-count display rule is not proof that a sample is representative. |
| Forecast replay | `data/forecast_history.py`, `components/_callbacks_forecast.py` | Reuse cautiously: storage is bounded to 30 snapshots per combination, the save function assigns its own current `scored_at`, and a chart callback calls it. This does not establish a complete, durable archive of published production vintages. |
| Demand and operator-forecast observations | `data/vintage.py` | Reuse first-seen/latest values and capture timestamps. Earliest observed operator data are not automatically the operator's original issue. Latest demand is not automatically final or settled. |
| Forecast/actual pairing and benchmark calculations | `models/drift.py`, `models/benchmark.py` | Audit and reuse their contracts. Do not create a competing metric implementation or assume historic records contain all needed provenance. |
| Scheduled production writes | `jobs/phases.py`, `jobs/scoring_job.py` | Capture trusted forecast provenance at publication, not as a side effect of visiting the UI. |
| Public project story | `web/landing.html`, `README.md`; portfolio `src/content/work.ts`, `src/content/work/gridpulse.mdx` | Update these together at release boundaries, keeping shipped capabilities separate from planned work. |

These observations establish code paths, not current production coverage, retention durability, or suitability of every historical record. The first work package measures those.

## 3. Work packages, dependencies, and acceptance

### A. Evidence audit and product contract — first

Deliverables:

- Inventory production payloads, forecast records, actual-demand versions, source clocks, retention, and missing-hour reasons using bounded read-only samples.
- Audit a small candidate set: one region with broad coverage, one with gaps/revisions, and one with baseline fallback. Select actual regions from measured availability rather than assuming a particular BA qualifies.
- Distinguish forecast publication time, observation cutoff/origin, target time, archive capture time, and UI refresh time.
- Verify whether operator issue times exist; record unknown times rather than substituting first retrieval.
- Define forecast selection: for example, latest published GridPulse forecast known by an explicit cutoff. Keep fixed-lead evaluation as a separate selection mode where supported; do not blend their scores.
- Bound the pilot to one BA for the first complete review, then three measured coverage/fallback cases. Keep existing forward coverage available; mark unsupported historical selections unavailable. Archive the published series first, with additional model issues added only where their provenance and use justify it.
- Define historical-review mode versus forward-outlook mode, terminology, and a candidate practitioner walkthrough.
- Record which comparisons are possible now, require new capture, or cannot be reconstructed honestly.

Acceptance:

- A field-by-field contract marks provenance as known, derived with a documented rule, or unavailable.
- A sample operating day can be reconstructed from identified records, or the exact missing dependency is documented.
- Retention/storage/runtime measurements support a proposed archive policy. A 30-day pilot retention window may be evaluated, but is not assumed adequate or free.
- Existing UI snapshots are not relabeled as original production issues without evidence.

Dependencies: none. Rough effort: 2–3 engineering days. Practitioner recruitment can begin in parallel when separately authorized.

### B. Source, timing, fallback, and coverage — required foundation

Deliverables:

- Label `GridPulse published forecast`, `GridPulse model comparison`, and `Operator-reported forecast` distinctly. Preserve baseline identity without borrowing model metrics or intervals.
- Present target period and timezone, forecast publication time when known, latest input observation, and retrieval/refresh times with their actual meanings.
- Explain baseline service state: why it was selected, when that state began if known, and whether it is an expected method choice or fallback. Unknown reasons remain unknown. No model-policy changes are implied.
- Make scoring count and coverage prominent alongside error values: eligible target hours, scored hours, unavailable/excluded hours, represented date range, and available reason codes.
- Keep a seven-day reporting window distinct from seven complete days of scored observations.
- Distinguish thin/partial coverage from a score with no usable sample. Establish any headline suppression rule from an explicit reviewed policy, rather than inventing an industry-standard threshold.
- Persist immutable forecast identities and target/value arrays from the scheduled publication path, with schema version and integrity checks. Reuse existing storage where practical after the audit.
- Preserve the actual served target range rather than only a chart's 24-hour slice. A whole-day review requires an eligible forecast issue covering that day; do not manufacture coverage by stitching later issues into an earlier forecast.
- Give existing records an explicit legacy-provenance state. Start prospective capture where history cannot be recovered; do not backfill newly calculated forecasts as historical issues.

Acceptance:

- Baseline, trained model, comparison, missing source, unavailable reason, stale inputs, and missing score states are distinguishable.
- Coverage figures reconcile with the paired records that produced the displayed score.
- Scoring/retry behavior cannot overwrite a prior issue under the same immutable identity or create false changes from duplicate writes.
- The new archive does not depend on page visits and survives the tested service restart/replica lifecycle.
- UI refresh time cannot be mistaken for forecast publication time.

Dependencies: A. Rough effort: 5–8 engineering days; historical recovery or substantial upstream reliability remediation is additional scope.

### C. Historical operating-day review — first usable workflow

Deliverables:

- Introduce a historical review entry point in Overview while keeping the existing Forecast view for the forward outlook. Keep the established visual system and callback/tab identifiers where possible.
- Select BA, operating date, timezone, series, and supported forecast selection rule. Freeze those selections as the review context.
- Show paired forecast and actual curves, hourly signed errors, missing hours, and a sortable discrepancy table.
- Add signed mean error and MAE in MW, largest paired miss, and daily peak magnitude/timing error where evidence supports them. Retain MAPE as a complementary measure with coverage.
- Explain missing evidence and withheld daily peak metrics. Distinguish weather context, data revisions, and model disagreement from established causal explanations.
- Make generic weather-alert counts, nameplate ratios, and model diagnostics secondary to the selected review task.
- Replace occupational persona framing with task-based entry points after auditing its routing, bookmarks, defaults, and callback dependencies. Preserve access to useful existing views.

Acceptance:

- A user can identify which forecast and actual version are compared and which hours are excluded.
- An operating-day review uses the actual local-day duration, including 23- and 25-hour daylight-saving days.
- Daily peak magnitude/timing error is unavailable when the required full-day coverage is missing; it is not silently computed from different subsets.
- Positive signed error consistently means forecast above actual; negative means below actual. Definitions appear near the metrics.
- The same selected paired records drive chart, table, summary, and export inputs.
- No alert declares a forecast wrong solely because it differs from another model or an observation is unusual.
- Saved views and keyboard/mobile navigation are checked after persona/navigation changes.

Dependencies: B and adequate prospective or verified historical data. Rough effort: 4–6 engineering days.

### D. Reproducible review record — first usable workflow

Deliverables:

- Add copyable plain text and downloadable JSON/CSV evidence containing region, operating period, timezone, forecast identity and selection rule, target values, actual-data version/capture time, coverage, metrics, warnings, and schema version.
- Include analyst-authored fields: observation, unresolved question, conclusion, and follow-up. Distinguish those statements from calculated facts.
- Freeze any weather observations or forecast context actually cited in the note, including source and timestamps. If original forecast-weather inputs were not captured, disclose that limit instead of reconstructing an apparent original input from later weather.
- Preserve the evidence shown at export time. A shareable URL pointing only to a changing live view is insufficient.
- Identify later actual revisions without silently changing an exported review. Allow a new review to cite the newer actual version separately.
- Use a minimal stateless download first. Add server-side storage or persistent share links only if walkthroughs demonstrate the need and access/retention behavior is scoped.

Acceptance:

- Reopening the artifact reproduces its values after the live forecast or actual-demand feed changes.
- Downloaded values and metric calculations match the displayed review.
- Unknown publication times and partial coverage remain visible in exports.
- No AI-written causal conclusion is needed for this release; analyst text remains editable and clearly attributed.

Dependencies: C. Rough effort: 2–3 engineering days for the stateless version.

### E. Practitioner walkthroughs and decision gate

Deliverables:

- Prepare a small historical test set: a well-covered day, a peak/ramp miss, a missing/revised-data case, and a baseline/fallback case. Use verified captures; label synthetic edge cases when needed.
- Seek 3–5 initial walkthroughs covering intended analyst users and at least one forecasting domain adviser. This is formative research, not statistical validation or customer adoption.
- Ask participants to review a day, identify the largest supported discrepancy, explain evidence limitations, and produce a follow-up note. Compare with their actual current tool/process when access and permission allow.
- Record task completion, source/time interpretation errors, missing evidence, time spent, assistance required, and the destination/use of their normal work product.
- Separate observed behavior, participant opinion, and product-team interpretation in the findings.

Gate:

- Continue when participants can connect the task to recurring real work and use the available evidence without confusing forecast producer, time, or score scope.
- Refine or stop expansion if public BA-level data are too coarse/late, the task duplicates existing tools without a meaningful advantage, or the review output has no practical destination.
- Make efficiency or accuracy claims only after an appropriate comparison has actually been measured. Interview count alone does not establish fit.

Dependencies: an evaluable C/D prototype and separately authorized participant contact. Rough effort: 2–3 analysis/facilitation days; recruiting and participant availability add calendar time. Drafting this scope sends no outreach.

### F. Portfolio, About, and documentation alignment

First content pass after A/B:

- Use a functional intended-user description instead of treating `Forecast Review Analyst` as an established job title or proven customer segment.
- State the historical-review task as the proposed focus; describe the existing forward outlook accurately.
- Credit Kristen's product scope, workflow, presentation decisions, and delivery of the deployed dashboard/read-only API.
- Replace team-management-sounding wording with: `I took GridPulse from product scope to a deployed dashboard and read-only API, defining the review workflow and how forecast sources, accuracy, and data quality are presented.`
- State that published industry procedures informed the workflow and practitioner testing is pending. Do not call desk research customer discovery or retroactively imply the original build followed interviews.
- Keep one AI-assistance disclosure in Project notes and one concise validation limit.

Release-aligned pass after C/D/E:

- Describe only shipped capabilities as present-tense functionality. Show pending work explicitly as proposed.
- Replace screenshots and captions with the actual released review view, source/coverage context, and capture date. Keep historical screenshots clearly dated if retained.
- Update app About, README, PRD/product framing, relevant architecture/contract documentation, portfolio card, case study, alt text, and metadata together.
- Publish practitioner findings only with accurate attribution, permission where needed, and a clear separation between observed results and plans.

Acceptance: public copy agrees with the actual UI and provenance contract; no unsupported adoption, operational use, measured savings, forecast superiority, or independent authorship of every technical component is implied.

Dependencies: A/B for framing; C/D/E for feature and findings claims. Rough effort: 1–2 engineering/editorial days across the releases.

### G. Forecast changes and fair operator comparisons — conditional extension

Deliverables:

- Compare successive immutable GridPulse issues on their overlapping target hours; show changes in peak magnitude/timing and hourly values with both issue times.
- Compare GridPulse and operator-reported forecasts on identical target hours, explicit selection rules, and a consistent actual-data version.
- Distinguish a legitimate forecast-revision comparison from an accuracy ranking. Different forecast issue times and information sets are material context.
- Use `earliest observed` when that is all the archive establishes. Withhold an as-issued or same-cutoff claim when provenance cannot support it.
- Preserve separate benchmark versus chart-series identities; do not attach an ensemble benchmark result to a baseline or another selected model.

Acceptance: comparison eligibility is explicit; missing target overlap, unknown issue time, differing coverage, or revised operator data cannot silently create a superiority claim.

Dependencies: B's archive, C's pairing, sufficient captured history, and E's demonstrated need. Rough effort: 4–7 additional engineering days.

### H. Weather sensitivity and context — conditional extension

Deliverables:

- Keep existing controls available as clearly labeled sensitivity exploration, showing the reference forecast, calculation method, changed inputs, and held-constant assumptions.
- Separate weather variables influencing demand from changes in renewable generation/net load. Do not relabel demand estimates as generation or reserve forecasts.
- Prioritize weather context relevant to the selected period and region over an undifferentiated alert count.
- Investigate coherent alternative weather trajectories only if practitioner work demonstrates value. Define input provenance and evaluate behavior before calling them operational scenarios or calibrated stress forecasts.

Acceptance: sensitivity outputs do not imply causal attribution, calibrated probability, physical renewable dispatch, or validated operational recommendations.

Dependencies: clarity work can accompany C/F; richer scenarios require E and a separate evaluation design. Rough effort: 1–2 days for disclosure/context work, 3–5 additional days for a limited richer prototype, with evaluation effort to be scoped.

## 4. Data and metric contract

Mandatory distinctions:

- Forecast producer and series: published GridPulse model, GridPulse baseline, explicit comparison, or operator-reported series.
- Forecast publication time versus data origin/cutoff versus local capture time. Unknown operator publication time remains unknown.
- Target timestamps stored in UTC with explicit local display conventions; preserve interval beginning/ending semantics and DST identity.
- Forecast selection cutoff versus review creation time. Later actual observations may evaluate an earlier prediction, but cannot be presented as information available when that prediction was made.
- Actual-demand status: first observed, selected later observation/revision, or explicitly qualified final data. `Latest` alone does not mean final.
- Coverage denominator: target hours required by the selected contract, unique paired hours used, and missing/excluded hours with known reasons. A seven-day fixed-lead denominator differs from one operating day.
- A review package freezes values, not just references to mutable `latest` keys.
- A daily curve belongs to one eligible issue under the selected cutoff rule. A fixed-lead analysis may intentionally combine predictions from different issues, but must be labeled and scored separately. Partial records support only the paired-hour analyses their coverage permits.

Metric definitions:

- Signed hourly error = forecast MW minus actual MW; signed bias = its mean over the stated paired set.
- MAE = mean absolute hourly error over that same set.
- MAPE retains the established actual-value eligibility policy; exclusions and denominators must be visible. A coverage percentage is not a confidence level.
- Daily peak magnitude error = maximum forecast minus maximum actual over a complete, consistently defined operating day. Daily peak timing error compares those peak timestamps, reports ties with a deterministic documented policy, and is withheld if the day cannot support the calculation.
- Largest miss is the maximum absolute error among paired hours, labeled as such when coverage is partial.
- Model holdout, fixed-lead observed error, historical day review, and operator benchmark remain separately labeled products.

## 5. Existing work to coordinate

Checked against GitHub on 2026-10-25; these are related work items, not assertions that every historical diagnosis remains current:

- [Coverage issue 537](https://github.com/kristenmartino/gridpulse/issues/537): audit the seven-day scoring shortfall and unresolved-hour reasons before interpreting coverage. Remediate relevant causes or expose their effect; do not assume one explanation for every gap.
- [Overview/briefing tracker 221](https://github.com/kristenmartino/gridpulse/issues/221): reconcile overlap with historical review and reproducible briefing work rather than building a parallel decision layer.
- [Consistency/persona issue 188](https://github.com/kristenmartino/gridpulse/issues/188): coordinate removal or relabeling of persona behavior and stale claims.
- [Open study PR 632](https://github.com/kristenmartino/gridpulse/pull/632): coordinate any forecast-bias evaluation with its actual study scope; this plan does not merge it or assume its results.

Read the current branch, queue, source contracts, and relevant open reliability work again before implementation. Model-origin, seed-gap, and serve-quality defects may affect candidate days or forecasts without belonging to this workflow release.

## 6. Tests and release evidence

Required functional checks follow the actual change:

- Source matching: baseline/model/operator identities, missing keys, unavailable comparison, and fallback reason states.
- Time alignment: fixed-lead versus issue-cutoff selection, UTC/local conversion, interval convention, duplicate hours, DST 23/25-hour days, partial days, and no hindsight substitution.
- Pairing/metrics: identical eligible sets, signed-error convention, exclusions, ties, missing peaks, sample counts, gaps, and no cross-series score attribution.
- Revision/export: immutable original issues, idempotent retries, legacy provenance, changed live inputs after export, and schema compatibility.
- UI/API: loading/empty/error states, same snapshot across components, accessible keyboard interactions, mobile wrapping, and saved-view compatibility.
- Reliability: bounded read-only request paths, archive failure visibility, storage/runtime impact, and restart/replica behavior. No training in web requests.
- Public copy: sources and metric scope match canonical facts and released features; screenshots are of the actual release.

Use relevant existing unit/contract tests, then the repository's required full checks for material code changes. Check the portfolio production build for its release. After deployment, verify the actual serving revision/traffic and scheduled-job versions with the existing deployment checkers, plus public UI verification. Keep application and portfolio release evidence distinct.

## 7. Delivery sequence and effort

| Sequence | Package | Gate |
|---|---|---|
| 1 | A: evidence audit and contract | Can the selected day and forecast be reconstructed honestly? |
| 2 | B plus first F pass: source/time/fallback/coverage clarity and capture | Does the evidence support the labels and scores? |
| 3 | C, D, limited H disclosure work: historical review and frozen export | Can a complete review be performed and reproduced? |
| 4 | E: formative walkthroughs | Does this task fit recurring practitioner work? |
| 5 | Final F pass and release hardening | Does the public story match the shipped and observed result? |
| Later | G and richer H | Is there demonstrated need and sufficient provenance? |

First usable release: approximately **18–28 focused engineering/editorial days**, including A–F and limited H clarity work, assuming existing services remain usable and no substantial historical recovery or unrelated model remediation is required. Recruiting, prospective history accumulation, and participant availability add calendar time. Refine this range after A; no completion date is committed.

Reserve 2–3 additional days if integration/release hardening exceeds the checks already included in the packages. G and richer H are separately scoped extensions, not hidden first-release requirements.

Suggested merge boundaries: evidence contract; provenance/coverage; historical review; frozen export; release copy/screenshots. Bundle related copy with functional changes to avoid unnecessary deployment churn. Do not publish unimplemented features between those boundaries.

## 8. Explicit exclusions

- Additional forecast model families, training campaigns, or changes to fallback/serve-selection policy just to improve presentation.
- New balancing-authority coverage or a full visual redesign.
- Dispatch control, bidding, procurement recommendations, reserve adequacy, outage probability, or portfolio-specific forecasting without the necessary data and separate scope.
- Automated causal diagnoses, automatic forecast overrides, or a universal alert threshold inferred from a small sample.
- A new account system, collaboration platform, or server-stored analyst notes for the initial stateless export.
- Claims of customer discovery, practitioner validation, adoption, time savings, willingness to pay, or business results based only on documentation research or this plan.

## 9. Recommendation coverage checklist

| Recommendation from the discussion | Package |
|---|---|
| Realistic intended user and concrete recurring task | A, E, F |
| Historical review first; forward outlook remains accessible | A, C |
| Distinct GridPulse/operator producer labels | B, G |
| Explicit source clocks, local operating day, and review cutoff | A, B, C |
| Baseline reason and duration when known | B |
| Prominent count, completeness, and missing-hour evidence | B, C |
| Signed MW error, peak error, timing error, and sample-aware metrics | C, metric contract |
| Fair comparisons without hindsight | A, B, G |
| Changes between forecast releases | G |
| Weather sensitivities with defensible interpretation | H |
| Secondary generic alerts/nameplate context; task-based navigation | C, H |
| Analyst judgment rather than automatic wrong/right labels | C, D |
| Reproducible note, values, provenance, and revision handling | B, D |
| Actual practitioner walkthroughs and measured benefit | E |
| Concrete ownership credit and proportionate AI disclosure | F |
| App, portfolio card/body, metadata, screenshots, and docs aligned | F |

## 10. Research basis

The proposed use case is an inference from documented practice, not an observed GridPulse customer workflow. ISO New England documents forecast review, active-source selection, overrides, timing conventions, and data-quality states. EIA documents public-feed availability and revisions. Neither establishes that GridPulse meets a practitioner's needs.

- [ISO New England: Adjust the Short-Term Load Forecast](https://www.iso-ne.com/static-assets/documents/rules_proceds/operating/sysop/cr_ops/crop_25004.pdf)
- [EIA: About EIA-930 data, availability, and quality](https://www.eia.gov/electricity/930-content/about.html)
- [ERCOT: Load forecasting overview, June 2024](https://www.ercot.com/files/docs/2024/06/11/7%20Board%20Education%20%E2%80%93%20Load%20Forecasting.pdf)
- [Grid Status: EIA Browser](https://docs.gridstatus.io/monitor/eia-browser)
