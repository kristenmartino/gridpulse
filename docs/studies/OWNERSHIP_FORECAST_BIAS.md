# Does forecast bias track ownership? — pre-registration

**Status: pre-registered design. No results have been computed. Nothing below
may be edited after the first measurement runs except by adding a dated
amendment section.**

Written 2026-08-20, before any ownership-split measurement.

---

## 1. The question

Utility load forecasts have historically run high. LBNL's study of IRPs filed
in the 2000s found every utility but one overestimated energy growth, and
eight of eleven overestimated peak demand. Three explanations compete, and
outcome data alone cannot separate them:

1. **Asymmetric loss.** Under-forecasting risks blackouts and reliability
   violations; over-forecasting costs money slowly and diffusely. A rational
   planner biases high. Not misconduct — correct behaviour given the penalty
   structure.
2. **Genuine uncertainty.** Common-mode macroeconomic error, structural
   demand change (efficiency standards, rooftop solar), and inflated
   customer-supplied inputs. Everyone wrong in the same direction, together.
3. **Capital incentive.** Cost-of-service regulation pays a return on rate
   base, so a higher forecast justifies more capex and more earnings
   (Averch–Johnson). The incentive is documented; intent is not observable.

**A high forecast produced by explanation 1 is indistinguishable in the data
from one produced by explanation 3.** This study exists because ownership
structure offers a way to pull them apart.

## 2. The identifying idea

Explanation 3 requires a shareholder return on rate base. That return exists
for investor-owned utilities and does not exist for municipal utilities,
cooperatives, or federal power agencies. If bias is incentive-driven, it
should be stronger in IOUs. If it is loss-function and uncertainty, it should
be similar across ownership classes.

**The confounder is obvious and must be handled:** munis are also smaller,
have smaller forecasting staffs, and serve different load mixes. Size and
sophistication travel with ownership, so a bare IOU-vs-muni gap proves
nothing on its own.

Two design features address it.

**Federal and state-owned agencies are the disentangling control.** BPA,
WAPA, SPA, TVA and Santee Cooper are *large and sophisticated* but earn no
shareholder return. If bias tracks capability, they should resemble IOUs. If
it tracks the capital incentive, they should resemble munis. They are the one
group where the two explanations predict opposite results.

**Day-ahead forecasts are the placebo test.** An EIA-930 day-ahead forecast
schedules tomorrow's operations. It justifies no capital spending, so
explanation 3 cannot act on it. Any ownership gap measured at day-ahead is
therefore *not* the capital incentive — it is capability, load mix, or
something else. That measured gap becomes the baseline subtracted from the
long-horizon result:

```
incentive effect ≈ (IOU − public) long-horizon  −  (IOU − public) day-ahead
```

That difference-in-differences, not either arm alone, is the estimand.

## 3. Hypotheses, registered in advance

- **H0 (day-ahead):** mean signed day-ahead bias does not differ by ownership
  class. *Expected to hold.* If it fails, the confounder is large and the
  long-horizon arm must carry the correction explicitly.
- **H1 (long-horizon):** IOU long-horizon forecasts carry larger positive
  bias than public-power forecasts, after the day-ahead baseline is removed.
- **H2 (the discriminating one):** federal/state agencies pattern with
  municipals rather than with IOUs at long horizon. Support for H1 *without*
  H2 is weak evidence, because size and sophistication would explain it
  equally well.

**Falsification.** If public power over-forecasts at the same rate as IOUs at
long horizon, the capital-incentive explanation loses its best available
evidence and the study says so in those words.

## 4. Metric

**Signed mean percentage error, not MAPE.** The question is direction, and
MAPE deletes it. Per entity:

```
bias_pct = mean( (forecast − actual) / actual ) × 100      positive = over-forecast
```

Reported alongside: median signed error (tail-robust), n, and the share of
periods with positive error. MAPE is reported for comparability with prior
work and decides nothing here.

Aggregation across entities is a **median of per-entity means**, matching the
`/benchmark` fleet convention, so one catastrophic entity moves its own row
rather than the class.

## 5. Population and exclusions, fixed in advance

- **In scope, day-ahead arm:** the 51 EIA-930 balancing authorities GridPulse
  already scores.
- **Excluded — not utilities:** CAISO, ERCOT, ISONE, MISO, NYISO, PJM, SPP.
  These are markets covering mixed-ownership members; they have no ownership
  class.
- **Excluded — cannot be scored fairly:** whatever the existing benchmark
  scoreability gate already excludes, on the published rules, unchanged for
  this study. Exclusions are inherited, not re-derived, so the study cannot
  tune its own population.
- **Minimum sample:** a class with fewer than 5 scoreable entities is
  reported with its n and excluded from significance testing. Cooperatives
  (AECI, SEC) will likely fail this and get folded into "public power" with
  the split published.

## 6. Phases

**Phase 1 — ownership classification.** Assign each of the 51 BAs to
`investor-owned | municipal | cooperative | federal | state-authority |
rto-iso`, one authoritative citation per entity. Edge cases known in advance:
SRP (political subdivision, not a muni utility), BANC (joint powers authority
containing SMUD and others), IID (irrigation district), Santee Cooper
(state-owned corporation), AECI and SEC (generation cooperatives).

**Phase 2 — day-ahead bias, the placebo arm.** Compute signed bias per BA
from the existing vintage archive. Uses data already held; no new collection.
Deliverable is a table of per-BA signed bias plus class aggregates.

**Phase 3 — go / no-go.** If Phase 2 shows a large ownership gap at
day-ahead, the confounder dominates and Phase 4's cost is probably not
justified until it is understood. Decide here, in writing, before spending
months.

**Phase 4 — long-horizon collection.** IRP and load-forecast filings from
state dockets (IOUs), plus BPA/TVA/Santee Cooper planning documents and
whatever municipal filings exist. Extract forecast vintage, horizon,
forecasted peak and energy by year. This is the expensive phase and the one
with a real sample-selection problem: **many municipals do not file public
IRPs at all**, so the public-power sample will be self-selected toward larger,
more formal utilities. That bias runs *against* H1 and must be stated.

**Phase 5 — actuals matching.** EIA-861 annual utility-level sales and peak,
matched on EIA utility ID.

**Phase 6 — analysis and adversarial review.** Nonparametric tests
(Mann-Whitney) given small n and non-normal errors; report effect size and
confidence intervals, not just p-values; pre-commit to reporting
"inconclusive" as a result.

## 7. Known limitations, written before results

- Day-ahead and long-horizon forecasts are produced by different teams with
  different methods. The placebo arm controls for *entity-level* capability
  imperfectly at best.
- Public-power IRP availability is self-selected (Phase 4).
- BA boundaries and utility boundaries do not always coincide.
- n is small in every class. This study can find a large effect or find
  nothing; it cannot resolve a subtle one.
- Nothing here identifies intent, and no phrasing in the write-up may imply
  it. The study measures direction and magnitude of error by ownership class.
  Attribution of motive is for litigants, not for this document.

---

## 8. Provenance and verification

*Added 2026-08-20, still before any measurement has run.*

Every claim in this study is produced by one of four routes, and each route
has a different failure mode and a different check. The routes are listed so
that a reader can ask, of any number in the final write-up, which one it came
from.

### 8.1 Computed quantities — the model never states the number

Anything derivable from data is produced by executing code, never by a model
reporting a figure. An LLM may write the script; the script produces the
value. This removes fabrication as a failure mode for the quantitative core
of the study rather than mitigating it.

**Control:** every computation script must first reproduce a figure the
`/benchmark` payload already publishes for the same BA and window, and print
that comparison before any study result. A script that cannot reproduce a
known-good number is not trusted to produce an unknown one.

### 8.2 Extracted quantities — verbatim span, mechanically checked

Any value read out of a document carries `{value, page, exact_quote,
section_or_table_caption}`. A deterministic post-step confirms the quote
appears on that page of that source. Failures are rejected, not repaired.

**The checker is never a language model.** String matching, PDF text
extraction and HTTP status codes only. Using a model to validate a model's
citation introduces a correlated failure and is prohibited here.

**What this catches and what it does not.** Quote verification catches
fabrication. It does not catch *misattribution* — a real number lifted from
the wrong table, winter peak where summer was wanted. That is why the schema
also captures the table caption, and why human verification (§8.4) is
directed specifically at misattribution rather than at invented numbers. The
two error types are estimated and reported separately. A rejection rate from
the mechanical checker is never presented as a bound on the correctness of
the records that survived it.

### 8.3 Classifications — two independent votes, adjudicated

Ownership classification is performed twice by independently prompted agents.
Agreements stand. Disagreements, plus anything either agent self-flagged, go
to a separate adjudication pass against primary sources, which must state
both the classification it recommends and the defensible alternative.

**Seeded controls.** The classification set includes entities whose ownership
is documented and unambiguous, checked by the harness after the fact. An
agent that misses a control has its remaining output treated as unverified.
Controls are checked before any result is inspected.

**Known limit:** replication catches idiosyncratic error and is blind to
systematic error. Two agents sharing a wrong prior will agree and both be
wrong, which is what the seeded controls and primary-source citations exist
to catch instead.

### 8.4 Human verification — exhaustive, not sampled

**Phase 4 documents are verified in full, not sampled.** The corpus is
plausibly 50–80 filings. Defending a document-level error rate below 5% by
sampling would require checking roughly 60 of them (see the bound below), at
which point sampling saves almost nothing and forfeits the stronger claim.

Where sampling is nonetheless used, two rules apply.

**Report the bound, not the percentage.** With zero errors found in *n*
checks, the 95% upper bound on the true rate is approximately 3/n. A
percentage-of-corpus figure ("we checked 10%") is not a statement about
reliability and may not appear in the write-up.

| checked, 0 errors | 95% upper bound on error rate |
|---:|---:|
| 30 | ~10% |
| 60 | ~5% |
| 100 | ~3% |
| 300 | ~1% |

**Sample documents, not values.** Extraction errors cluster within a
document: one misread column heading corrupts every number drawn from that
filing. Effective sample size is therefore the document count, and checking
many documents shallowly beats checking few exhaustively for the same effort.

**Risk-weight rather than sample uniformly.** Values that drive the headline
result are verified at 100% regardless of any sampling scheme, as are
outliers and filings whose layout differs from the majority. Random sampling
is reserved for establishing the base rate on the remainder.

### 8.5 Interpretive claims

Statements about what the results mean are the researcher's and are not
delegated to a model. No agent output is quoted as analysis.
