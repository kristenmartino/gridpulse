---
name: check-past-mistakes
description: Cross-check a just-approved plan or just-finished implementation against this repo's known mistake patterns before it ships. Fires automatically via hook after plan approval (ExitPlanMode) and after implementation work ends, but also invoke it manually mid-task any time something feels like it might repeat a past mistake. Reads only CLAUDE.md (already loaded, no extra cost) — never MISTAKES.md's full archive. If it catches something new, it deposits exactly one line to MISTAKES.md's Worklog and stops; it does not diagnose or propose fixes.
---

# Check past mistakes

A cheap, repeatable pass that runs right after a plan is approved and right
after implementation work is done, asking one question: **does this repeat
something CLAUDE.md already tells us not to?** It exists because knowing a
rule and remembering to apply it under task pressure are different things —
this skill is the forcing function for the second one.

## What to check against

Read `CLAUDE.md` in the repo root — specifically the "End-of-PR
explanatory-doc check" section and any prose rules elsewhere in the file
that trace back to a graduated `MISTAKES.md` entry (they're written as
concrete invariants, not vague advice — "verify X before Y," "bound what
one call can cost," "grep for the old literal in `web/`"). That's the
complete checklist. **Do not open `MISTAKES.md`** to look for more —
`MISTAKES.md` is an evidence archive that a separate audit pass mines
periodically; loading it here would both cost real context on every single
plan and duplicate a job that already has an owner. If a rule in CLAUDE.md
feels stale or doesn't match what you're seeing in the code, say so and
flag it (see "regular reassessment" in CLAUDE.md) — don't just skip it.

## When you're checking a plan (after ExitPlanMode)

Walk the plan against CLAUDE.md's rules that a plan can violate before any
code is written:
- Does it reference an issue number (`Closes #N`, `#N` in a description)
  without a note that it was verified via `gh issue view`?
- Does it move a fact that's cited in `docs/CANONICAL_FACTS.md` or similar
  without a step to grep `web/` for the stale literal?
- Does it touch retry/timeout/circuit-breaker logic without a plan to bound
  both per-call and per-run cost, not just add more retries?
- Is it about to open a PR from a branch that's more than a day or two old,
  or that already had a PR merged from it? If so, the plan should include
  diffing against current `origin/main` before pushing, not just pushing.
- Any other CLAUDE.md rule whose trigger condition ("when doing X...") the
  plan's own description matches.

## When you're checking finished work (after implementation)

Same list, but against the actual diff/commits instead of the plan text —
a plan can say the right thing and the implementation can still miss it.
This is also the point to run the "End-of-PR explanatory-doc check" items
themselves if the calling session hasn't already.

## If you find a real match

Flag it plainly, before reporting the work as done — name the CLAUDE.md
rule, the specific line in the plan/diff that triggers it, and what's
missing. This is a stop-and-fix signal, not a footnote.

## If you notice something new

Something can go wrong, or nearly go wrong, in a way that doesn't match any
existing CLAUDE.md rule — that's exactly the kind of thing this whole system
exists to eventually catch. When that happens:

1. Add **exactly one line** to `MISTAKES.md`'s `## Worklog (undecided
   candidates)` section, in its existing format:
   `- YYYY-MM-DD [category] one-line description — <ref: issue/PR/session>`
   **Insert it as the first entry in the list** (right after the section's
   intro text, before the existing top entry) — the section is newest-first,
   and appending to the bottom would silently invert that.
2. Pick a best-guess `[category]` tag. Getting it slightly wrong is fine —
   the audit pass groups by root cause later, not by trusting the tag.
3. **Stop there.** Do not write a root cause. Do not propose a prevention.
   Do not decide whether it's a repeat of something else — don't even read
   the rest of `MISTAKES.md` to check. That restraint is deliberate: a
   session mid-task diagnosing its own mistake carries the same tunnel
   vision that produced it, and analysis is `audit-mistakes-log`'s job,
   done later with none of this session's context. If depositing ever feels
   like it takes real thought, that's a sign the deposit has drifted into
   analysis — pull back to one plain sentence.
4. Never edit `CLAUDE.md` from this skill. Adding an invariant is a
   deliberate, human-approved promotion, not something a same-session
   observation earns on its own.

## What "done" looks like

Either: nothing matched, say so briefly and move on — or: something
matched and got flagged before shipping — or: something new got deposited
as one Worklog line. All three are a complete, successful run of this
skill; none of them require touching `CLAUDE.md`.
