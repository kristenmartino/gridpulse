# Worklog — undecided mistake candidates

One file per deposit. **Never edit an existing file to add a new entry** —
write a new one. That is the entire point of this directory: two sessions
depositing at the same moment write different filenames and can never
conflict, so the cheapest step in the system stays cheap.

## Depositing

Create `<UTC timestamp>-<category>.md`:

```
.mistakes/worklog/2026-08-18T114530Z-guard-decision-without-force.md
```

Filename timestamp is `date -u +%Y-%m-%dT%H%M%SZ`. It is not decoration —
it is what tells the audit which candidates arrived since it last ran, at a
resolution that a date alone does not have.

If that filename already exists (two deposits in the same second, which
happens when one session writes several at once), **advance the seconds
until it is free** — keep the format exact. Do not append a suffix after
the `Z`: the audit and the nudge both parse the leading stamp, and a
`...Z1-` name is a filename they read differently from how it looks. The
first draft of this README omitted that, and the very next deposit hit it.

The file holds exactly one line, in the same format the Worklog list used
before it became a directory:

```
2026-08-18 [category] one-line description — ref: PR #579
```

No root cause, no proposed fix, no decision about whether it repeats
anything. That analysis belongs to `audit-mistakes-log`, running later
without the depositing session's context — see
[`CLAUDE.md`](../../CLAUDE.md) → "Mistake logging & rule graduation" for why
the split is load-bearing rather than bureaucratic.

## Reading

```bash
cat .mistakes/worklog/2026-*.md          # every pending candidate
ls .mistakes/worklog/ | wc -l            # how many are waiting
```

## What happens to these

`audit-mistakes-log` reads the whole directory, groups by root cause, and
promotes anything that crosses the graduation bar into `MISTAKES.md`'s
Analyzed section — deleting the files it consumed. Candidates it looks at
and defers stay here; `.mistakes/last-audit` records when it last ran, so
the SessionStart nudge counts only what arrived after that and a deliberate
"these can wait" buys quiet until genuinely new candidates appear.

Files here are pending, not permanent. `MISTAKES.md` is the archive.
