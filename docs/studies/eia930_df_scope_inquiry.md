# Draft inquiry to EIA — DF comparability on Form EIA-930

**Status: DRAFT, not sent.** Contact details are from the Form EIA-930
instructions (`QUESTIONS` section): EIA Survey Support Team, `eia4usa@eia.gov`,
1-855-342-4872, Mon–Fri 8:00–18:00 ET.

Notes before sending, below the draft.

---

**To:** eia4usa@eia.gov
**Subject:** Form EIA-930 — is the day-ahead demand forecast (DF) reported on a different basis than demand (D) for some respondents?

Hello,

I maintain a public dashboard that compares hourly demand forecasts against
Form EIA-930 data, and I have a question about how the day-ahead demand
forecast should be interpreted relative to reported demand. I would rather
ask than assume, because the answer changes how I present several balancing
authorities.

I have read the form instructions and understand that comparability is not
required. The instructions state:

> Demand forecast: If you do not produce a day-ahead demand forecast in the
> normal course of business that is directly comparable to actual demand as
> defined for this collection (see discussion of physical vs. commercial
> operations below), you are not required to produce a consistent demand
> forecast for the purposes of EIA-930 reporting. Please report the day-ahead
> demand forecast generated in the normal course of business.

My questions follow from that provision rather than challenging it.

**1.** Does EIA track, or publish anywhere, which respondents report a `DF`
that is not directly comparable to their reported `D`? I have not found such
a list in the instructions, the Hourly Electric Grid Monitor documentation,
or the API metadata, and I would like to avoid publishing a comparison the
data does not support.

**2.** For a respondent whose `DF` is not comparable, is there a documented
convention for what it typically does cover — for example native or retail
load only, load excluding pseudo-tied or dynamically scheduled resources, or
firm load excluding interruptible customers? I am trying to understand
whether a persistent offset has a standard explanation.

**3.** Would EIA consider a persistent one-directional gap between `DF` and
`D` to be within expected reporting practice, or is it something the agency
would follow up with a respondent about? I ask because I want to be careful
not to characterize normal, permitted reporting as a data quality problem.

For context, the pattern that prompted this: across the 30 days ending
2026-08-20, several balancing authorities published a `DF` that sits well
below their `D` in the same direction on nearly every hour — approximately
0.67× for PSEI, 0.74× for PSCO, 0.76× for FPC, and 0.90× for GVL. The
reported `D` for these respondents reconciles with net generation minus net
interchange to within 0.0%, so the demand series itself appears internally
consistent. Some of these ratios have been stable for at least twenty months
while others moved during 2026, which suggests more than one explanation is
in play.

I am not asking EIA to comment on any individual respondent's submissions if
that is not appropriate. A pointer to documentation, or confirmation that
non-comparable `DF` reporting is expected and undocumented, would be entirely
sufficient.

Thank you for your time.

Kristen Martino
https://gridpulse.kristenmartino.ai/benchmark

---

## Before sending

- **Decide how to identify the project.** The draft links the public
  benchmark page. That is honest and gives EIA context for why the question
  matters, but it also makes the inquiry attributable. Drop the link if you
  would rather ask anonymously.
- **Question 3 is the one to reconsider.** It invites EIA to characterize
  respondent behaviour, which they may decline. It is worth asking because
  the answer decides whether the page should describe this as normal practice
  or as an anomaly, but it is the most likely question to go unanswered.
- **The specific BAs are context, not the ask.** Questions 1 and 2 stand on
  their own if EIA cannot discuss individual respondents. That ordering is
  deliberate.
- **Expect a pointer, not an explanation.** The realistic best outcome is a
  link to documentation, or confirmation that none exists. Either resolves
  the immediate problem, which is whether the page may assert a cause.
- **A parallel route worth running at the same time:** the operators
  themselves. Duke Energy Florida and Xcel Colorado both have regulatory
  affairs contacts, and a respondent knows what its own submission covers
  more precisely than EIA does.
- **Every figure quoted above is reproducible** from
  `scripts/ownership_bias_placebo.py` and the reconciliation in
  `docs/BENCHMARK_METHODOLOGY.md` §12, so an answer can be checked against
  the same numbers EIA would see.

---

# Draft inquiry to Duke Energy Florida (FPC)

**Status: DRAFT, not sent. No verified recipient address** — see routing
notes below. Do not send to a guessed address.

FPC is the right respondent to approach first. Its `DF/D` ratio has held at
0.748–0.783 for twenty months, so this is a question about long-standing
practice rather than about anything that recently changed, which makes it
far less likely to read as an accusation.

---

**Subject:** Question about Duke Energy Florida's day-ahead forecast as reported on EIA Form 930

Hello,

I run a public dashboard that publishes hourly electricity demand forecasts
for US balancing authorities, and scores them against the day-ahead forecasts
that balancing authorities themselves report to EIA on Form 930. Duke Energy
Florida appears on it.

I am writing because I think my page may currently be describing your data
incorrectly, and I would like to fix that.

The day-ahead forecast reported for the FPC balancing authority runs at
roughly 0.77× the demand value reported for the same hours, consistently, and
that relationship has been stable for at least twenty months. My page
currently presents that difference as forecast error, which produces a
published accuracy figure for Duke Energy Florida of around 23%.

I do not believe that is a fair characterization, and I would rather correct
it than leave it. EIA's own Form 930 instructions note that a respondent's
day-ahead forecast is not required to be directly comparable to the demand
value reported for the same collection — respondents are asked to report the
forecast they produce in the normal course of business. That suggests the gap
may reflect two different quantities rather than forecast performance.

My question is simply: **what does the day-ahead demand forecast that Duke
Energy Florida submits on Form EIA-930 cover, relative to the demand figure
reported for the same balancing authority?**

Specifically, does the forecast cover Duke Energy Florida's own load
obligation, while the reported demand covers all load physically inside the
balancing authority footprint — which would include the municipal systems and
cooperative load served by others within that boundary?

I ask because the numbers are consistent with that reading. Over calendar
2024, the demand reported for FPC totalled roughly 58,000 GWh, while the
day-ahead forecast totalled roughly 44,400 GWh. I have not been able to
confirm the explanation from public data, and I would rather ask than publish
a guess.

If that is right, I will label the row accordingly and stop presenting the
difference as forecast error. If it is something else, I would be glad to be
corrected — either way the page will be more accurate for it.

Happy to share the underlying figures if useful. The methodology is public.

Thank you for your time.

Kristen Martino
https://gridpulse.kristenmartino.ai/benchmark

---

## Routing — find a real recipient, do not guess

I have not verified an address for Duke Energy Florida and will not invent
one. Routes worth trying, best first:

1. **The Ten-Year Site Plan filing.** Duke Energy Florida files it annually
   with the Florida PSC and it is public. Filings name the preparers and a
   regulatory contact, and whoever assembles the load forecast for that
   document is the person who can answer this in one sentence. This is the
   highest-signal route.
2. **Duke Energy regulatory affairs, Florida.** The corporate site publishes
   a regulatory contact; a data-interpretation question routed there will
   usually reach load forecasting.
3. **Florida PSC docket staff.** They will not answer for Duke, but they
   handle the TYSP filings and can point you at the right contact.
4. **LinkedIn, as a fallback.** Duke Energy's resource-planning and load
   forecasting analysts are findable, and this is a question a practitioner
   would enjoy answering.

## Why this one is likely to get a reply

The email is not asking them for a favour. It is telling a utility that a
public page is currently publishing an unflattering accuracy number about
them that the author believes is wrong, and offering to correct it. That is
in their interest to resolve, which is the honest reason it will probably
work — and the reason the offer to correct has to be genuine.

## What to do with the answer

- **Confirms the scope reading** → label the FPC row with the specific
  mechanism, and check whether the same explanation covers GVL.
- **Contradicts it** → the page keeps saying "cause not established," which
  is what it says today, and the hypothesis is retired in the methodology.
- **No reply** → change nothing. The current wording is already correct in
  claiming no cause, and silence is not evidence for either reading.
