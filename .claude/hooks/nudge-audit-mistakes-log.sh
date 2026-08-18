#!/bin/bash
# SessionStart hook — quiet staleness check for audit-mistakes-log.
# That skill is deliberately NOT meant to run every session (it wants a
# fresh, decontextualized pass — see its own SKILL.md), so this only
# speaks up when enough undecided Worklog candidates have piled up to be
# worth a separate pass. Pure count threshold, no date arithmetic, to
# avoid GNU-date/BSD-date portability differences across machines.
#
# Path resolution is deliberate: an earlier version used a bare relative
# "MISTAKES.md" behind an [ -f ] guard, so running from any subdirectory
# produced silence and exit 0 — indistinguishable from "checked, nothing
# to report." That is the configured-and-inert failure shape CLAUDE.md
# already has a graduated rule about. Resolve from the script's own
# location so cwd cannot make this silently no-op, and prefer the
# harness's project dir when it is set.
if [ -n "$CLAUDE_PROJECT_DIR" ] && [ -f "$CLAUDE_PROJECT_DIR/MISTAKES.md" ]; then
  ROOT="$CLAUDE_PROJECT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
fi

FILE="$ROOT/MISTAKES.md"
# A genuinely absent MISTAKES.md is a legitimate quiet exit (a branch may
# predate it). Path resolution failing is not, and no longer can be.
[ -f "$FILE" ] || exit 0

ENTRIES=$(awk '/^## Worklog/{flag=1; next} /^## /{flag=0} flag && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/' "$FILE")
COUNT=$(printf '%s\n' "$ENTRIES" | grep -c '^- ' )
[ "$COUNT" -ge 3 ] || exit 0

OLDEST=$(printf '%s\n' "$ENTRIES" | tail -1)
cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"MISTAKES.md has $COUNT undecided Worklog candidates (oldest: ${OLDEST#- }). Consider running the audit-mistakes-log skill in a fresh session — it's meant to run decontextualized from whatever deposited these, not mid-task here."}}
EOF
