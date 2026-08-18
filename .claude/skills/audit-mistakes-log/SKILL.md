---
name: audit-mistakes-log
description: Periodic, deliberately decontextualized audit of MISTAKES.md — tallies Worklog candidates by likely root cause, drafts full Analyzed entries and candidate CLAUDE.md rule promotions once a pattern crosses the graduation bar, and spot-checks existing CLAUDE.md rules for staleness. Not a mid-task skill — run it standalone, ideally in a fresh session with none of the memory of whichever session deposited the candidates it's reviewing, on a schedule (weekly, or whenever the Worklog has accumulated a handful of new lines) rather than as part of normal implementation work. Never edits CLAUDE.md directly; drafts proposals for human approval.
---

# Audit the mistakes log

This is the other half of the mistake-logging system in `CLAUDE.md` — the
half that does the actual thinking. `check-past-mistakes` deposits one-line
candidates cheaply and deliberately does no analysis; this skill is where
that analysis happens, on its own pass, separate from whatever session hit
the mistake in the first place. Being decontextualized is the point: a
session that just spent an hour on a bug tends to over- or under-weight it
relative to the rest of the log. Come to this fresh.

If invoked mid-task from an active implementation session, say so and
suggest running it standalone instead (a fresh session, or the next
scheduled pass) unless the user explicitly wants it right now anyway —
running it under someone else's context defeats the reason it's a separate
skill.

## Step 1 — read the whole archive

Read `MISTAKES.md` in full: the `## Worklog (undecided candidates)` section
and the `## Analyzed` section (its pattern tally + entries). This is the one
skill that's supposed to load the whole file — everywhere else in this
system it stays on disk unread.

## Step 2 — tally by root cause, not by tag

Group the Worklog's one-liners by what actually caused them, not by the
literal `[category]` string each one carries — two entries tagged
differently can still share a root cause, and two entries with the same tag
might not. This is a judgment call the depositing sessions deliberately
didn't make; making it well is most of this skill's value. Cross-reference
against categories that already exist in the Analyzed pattern tally — a new
Worklog entry might be the second occurrence of an already-graduated
category, which still matters (it's evidence the rule is or isn't working)
even though it won't itself trigger a *new* promotion.

Update the Pattern tally table's counts to match what you find.

## Step 3 — decide what crosses the graduation bar

Per `CLAUDE.md`, a category graduates on either:
- **Repeat**: the same root cause across ≥2 occurrences.
- **Severity**: a single occurrence costly or high-blast-radius enough
  (production-visible, silently wrong for days, corrupted state) that
  waiting for a repeat isn't worth the risk.

Not everything in the Worklog needs to graduate today. A category with one
mild occurrence just stays open — that's a correct outcome, not a
non-result.

## Step 4 — for anything that crosses the bar, draft two things

**a) A full Analyzed entry**, in the file's existing format (What happened /
Root cause / Prevention / Status), built from the Worklog one-liner(s) plus
whatever additional digging is warranted (check the referenced issue/PR,
read the relevant code) — the one-liner is a pointer, not the whole story.
Move the consumed Worklog line(s) into this entry rather than leaving them
duplicated in both places.

**b) A candidate CLAUDE.md diff.** Follow the same principles CLAUDE.md
states for itself:
- **Phrase it as a positive invariant** — what to do, stated once, plainly
  ("verify X before Y," not "don't forget to check X"). If you catch
  yourself writing "never" or "don't," try inverting it into the
  affirmative version first and see if it reads better; it usually does.
- **Point at the `MISTAKES.md` entry for the story, don't restate it.** The
  CLAUDE.md line should be short enough that adding it doesn't meaningfully
  grow the file's size relative to its value.
- **Prefer strengthening an existing rule over adding a near-duplicate.**
  If the new pattern is a variant of something already codified, propose
  editing that rule rather than appending a sibling.
- Place it in whichever existing CLAUDE.md section it most naturally
  extends; only propose a new top-level section if nothing fits.
- If the actual fix is mechanical (a test, an assertion, a lint rule) rather
  than a judgment call, don't draft a CLAUDE.md line at all — mark the
  Analyzed entry `resolved — enforced by <X>` instead and say so plainly in
  your summary. A rule for something a test already guarantees just gives a
  future agent a second, weaker thing to trust.

## Step 5 — present, don't merge

You do not have standing authority to edit `CLAUDE.md`. Summarize what you
found — the tally, what (if anything) crossed the bar, the drafted Analyzed
entry text, and the exact proposed CLAUDE.md diff — and hand it to the human
for approval. If they approve: add the Analyzed entry to `MISTAKES.md`, add
the invariant to `CLAUDE.md`, and mark the entry
`graduated → CLAUDE.md § <heading> (<date>)`. If they don't: leave
`MISTAKES.md` as-is (the tally update from Step 2 is fine to keep either
way — it's bookkeeping, not a rule change) and note their reasoning
somewhere retrievable in case the pattern recurs again later.

## Step 6 — the reverse check: are the existing rules still true?

Don't only look for things to add. Pick a handful of CLAUDE.md's existing
graduated rules — the ones tied to a specific file, function, or behavior —
and check whether that reality still holds. Has the cited file been
renamed? Does the guarded-against code path still exist? Was the underlying
architecture it describes since replaced? A rule whose premise is gone is
worse than no rule: it spends a future agent's attention on a solved
problem while still sounding authoritative.

If you find one: propose cutting or updating it in the same summary you
give the human in Step 5, with the specific evidence (what changed, when,
in what PR if you can find it) that makes it stale. Don't silently remove
it — that's still a CLAUDE.md edit, subject to the same approval as adding
one.

## What "done" looks like

A short report: tally counts and any category-count changes, zero or more
drafted promotions (each with its Analyzed entry text and CLAUDE.md diff),
zero or more flagged-as-possibly-stale existing rules, and a clear statement
of what still needs the human's yes/no. `MISTAKES.md` may have updated tally
numbers even if nothing graduated this pass — that's still useful output.
