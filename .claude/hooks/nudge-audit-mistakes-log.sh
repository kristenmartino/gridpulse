#!/bin/bash
# SessionStart hook — quiet staleness check for audit-mistakes-log.
# That skill is deliberately NOT meant to run every session (it wants a
# fresh, decontextualized pass — see its own SKILL.md), so this only
# speaks up when enough NEW undecided Worklog candidates have piled up
# since the last audit to be worth a separate pass.
#
# "New since the last audit" is decided by ENTRY COUNT, not by date.
# The first version compared each entry's date against the marker date and
# counted only entries strictly after it, which meant every deposit made on
# the same calendar day as an audit was invisible — permanently. On a repo
# that lands a dozen PRs a day that is a whole day of candidates the nudge
# would never mention, so the reminder went quiet exactly when there was
# most to report. Dates have no sub-day resolution and the entries carry no
# timestamps, so no comparison of dates can fix it.
#
# The marker therefore records how many entries the audit saw:
#   <!-- audited-through: YYYY-MM-DD | entries-seen: N -->
# and anything beyond N is new. Promotions remove entries and the audit
# rewrites N to whatever it leaves behind, so both directions stay correct.
# A missing entries-seen (or `never`) means nothing has been audited, so
# every entry counts — the safe direction, since erring toward a nudge
# costs one line and erring the other way is silence that looks like health.
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

# Marker date is for humans; entries-seen is what the decision uses.
MARKER=$(grep -oE 'audited-through:[[:space:]]*[0-9]{4}-[0-9]{2}-[0-9]{2}' "$FILE" \
         | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)
[ -n "$MARKER" ] || MARKER="never"

SEEN=$(grep -oE 'entries-seen:[[:space:]]*[0-9]+' "$FILE" | grep -oE '[0-9]+' | head -1)
[ -n "$SEEN" ] || SEEN=0

ENTRIES=$(awk '/^## Worklog/{f=1; next} /^## /{f=0} f && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/' "$FILE")
TOTAL=$(printf '%s\n' "$ENTRIES" | grep -c '^- ')

# Entries are newest-first, so anything beyond the audited count is what
# arrived since. Clamp at zero: promotions can leave TOTAL below SEEN until
# the next audit rewrites it, and a negative would read as "nothing new".
COUNT=$(( TOTAL - SEEN ))
[ "$COUNT" -lt 0 ] && COUNT=0

if [ "$COUNT" -lt 3 ]; then
  log_hook "silent new=$COUNT total=$TOTAL seen=$SEEN since=$MARKER"
  exit 0
fi

log_hook "nudge new=$COUNT total=$TOTAL seen=$SEEN since=$MARKER"
OLDEST=$(printf '%s\n' "$ENTRIES" | sed -n "${COUNT}p")
cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"MISTAKES.md has $COUNT Worklog candidates deposited since the last audit ($MARKER saw $SEEN of the current $TOTAL). Most recent unaudited: ${OLDEST#- } — Consider running the audit-mistakes-log skill in a fresh session; it's meant to run decontextualized from whatever deposited these, not mid-task here. It rewrites the entries-seen count even if it promotes nothing, which silences this until new candidates arrive."}}
EOF
