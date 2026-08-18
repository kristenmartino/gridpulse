#!/bin/bash
# SessionStart hook — quiet staleness check for audit-mistakes-log.
# That skill is deliberately NOT meant to run every session (it wants a
# fresh, decontextualized pass — see its own SKILL.md), so this only
# speaks up when enough NEW undecided Worklog candidates have piled up
# since the last audit to be worth a separate pass.
#
# Dates are compared as ISO strings, never with date(1) arithmetic —
# YYYY-MM-DD sorts lexically, and GNU date and BSD date disagree on the
# flags this would otherwise need.
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

LOG="$ROOT/.claude/hook-activity.log"
log_hook() {
  printf '%s nudge-audit-mistakes-log %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$LOG" 2>/dev/null || true
}

FILE="$ROOT/MISTAKES.md"
# A genuinely absent MISTAKES.md is a legitimate quiet exit (a branch may
# predate it). Path resolution failing is not, and no longer can be.
if [ ! -f "$FILE" ]; then
  log_hook "skipped no-mistakes-file"
  exit 0
fi

# An absent or non-date marker means no audit has ever run, so everything
# counts. Same reading as a file that predates the marker entirely.
MARKER=$(grep -oE 'audited-through:[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2}' "$FILE" \
         | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)
[ -n "$MARKER" ] || MARKER="0000-00-00"

ENTRIES=$(awk '/^## Worklog/{f=1; next} /^## /{f=0} f && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/' "$FILE")
# Strictly after the marker: an audit that ran today has already seen
# today's entries.
NEW=$(printf '%s\n' "$ENTRIES" | awk -v m="$MARKER" '{d=substr($2,1,10); if (d > m) print}')
COUNT=$(printf '%s\n' "$NEW" | grep -c '^- ')

if [ "$COUNT" -lt 3 ]; then
  log_hook "silent new=$COUNT since=$MARKER"
  exit 0
fi

log_hook "nudge new=$COUNT since=$MARKER"
OLDEST=$(printf '%s\n' "$NEW" | tail -1)
cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"MISTAKES.md has $COUNT undecided Worklog candidates newer than the last audit ($MARKER). Oldest unaudited: ${OLDEST#- } — Consider running the audit-mistakes-log skill in a fresh session; it's meant to run decontextualized from whatever deposited these, not mid-task here. It advances the audited-through marker even if it promotes nothing, which silences this until new candidates arrive."}}
EOF
