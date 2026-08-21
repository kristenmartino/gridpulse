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
