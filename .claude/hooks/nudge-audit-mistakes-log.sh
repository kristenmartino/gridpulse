#!/bin/bash
# SessionStart hook — quiet staleness check for audit-mistakes-log.
# That skill is deliberately NOT meant to run every session (it wants a
# fresh, decontextualized pass — see its own SKILL.md), so this only
# speaks up when enough undecided Worklog candidates have piled up to be
# worth a separate pass. Pure count threshold, no date arithmetic, to
# avoid GNU-date/BSD-date portability differences across machines.
FILE="MISTAKES.md"
[ -f "$FILE" ] || exit 0

COUNT=$(awk '/^## Worklog/{flag=1; next} /^## /{flag=0} flag && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/' "$FILE" | wc -l | tr -d ' ')
[ "$COUNT" -ge 3 ] || exit 0

OLDEST=$(awk '/^## Worklog/{flag=1; next} /^## /{flag=0} flag && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/' "$FILE" | tail -1)
cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"MISTAKES.md has $COUNT undecided Worklog candidates (oldest: ${OLDEST#- }). Consider running the audit-mistakes-log skill in a fresh session — it's meant to run decontextualized from whatever deposited these, not mid-task here."}}
EOF
