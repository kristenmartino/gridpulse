#!/bin/bash
# SessionStart hook — quiet staleness check for audit-mistakes-log.
# That skill is deliberately NOT meant to run every session (it wants a
# fresh, decontextualized pass — see its own SKILL.md), so this only
# speaks up when enough NEW candidates have piled up since the last audit
# to be worth a separate pass.
#
# "New since the last audit" is decided by comparing each deposit's
# filename timestamp against .mistakes/last-audit. Two earlier versions of
# this got it wrong in instructive ways:
#
#   1. Compared entry DATES against a marker date, counting only entries
#      strictly after it — so every deposit made on the same calendar day
#      as an audit was invisible permanently, which on this repo is most
#      of them.
#   2. Replaced that with an entries-seen COUNT, which worked but had to be
#      kept truthful by hand: an audit that miscounted made real candidates
#      invisible, and nothing could detect it.
#
# Deposits now carry a second-resolution UTC timestamp in the filename, so
# the comparison is exact and needs no counter to stay honest. The state is
# the filesystem rather than a number someone maintains.
#
# Path resolution is deliberate: an early version used a bare relative path
# behind an [ -f ] guard, so running from any subdirectory produced silence
# and exit 0 — indistinguishable from "checked, nothing to report." Resolve
# from the script's own location so cwd cannot make this silently no-op.
if [ -n "$CLAUDE_PROJECT_DIR" ] && [ -d "$CLAUDE_PROJECT_DIR/.mistakes/worklog" ]; then
  ROOT="$CLAUDE_PROJECT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
fi

LOG="$ROOT/.claude/hook-activity.log"
log_hook() {
  printf '%s nudge-audit-mistakes-log %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$LOG" 2>/dev/null || true
}

WORKLOG="$ROOT/.mistakes/worklog"
# A genuinely absent directory is a legitimate quiet exit (a branch may
# predate it). Path resolution failing is not, and no longer can be.
if [ ! -d "$WORKLOG" ]; then
  log_hook "skipped no-worklog-dir"
  exit 0
fi

# No last-audit file means nothing has ever been audited, so every deposit
# counts. Erring toward a nudge costs one line; erring the other way is
# silence that looks like health.
# Checked with -f rather than relying on 2>/dev/null: bash sets up input
# redirection before the stderr redirect applies, so a missing file still
# prints. A hook that writes to stderr on an ordinary first run is noise
# that trains people to ignore it.
LAST=""
if [ -f "$ROOT/.mistakes/last-audit" ]; then
  LAST=$(tr -d '[:space:]' <"$ROOT/.mistakes/last-audit")
fi
[ -n "$LAST" ] || LAST="0000-00-00T000000Z"

TOTAL=0
COUNT=0
NEWEST=""
for f in "$WORKLOG"/[0-9]*.md; do
  [ -e "$f" ] || continue
  TOTAL=$(( TOTAL + 1 ))
  # Leading YYYY-MM-DDTHHMMSSZ sorts lexically, so string comparison is
  # exact ordering without date(1) — GNU and BSD disagree on its flags.
  stamp=$(basename "$f" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{6}Z')
  [ -n "$stamp" ] || continue
  if [ "$stamp" \> "$LAST" ]; then
    COUNT=$(( COUNT + 1 ))
    [ -z "$NEWEST" ] && NEWEST=$(head -1 "$f" 2>/dev/null)
  fi
done

if [ "$COUNT" -lt 3 ]; then
  log_hook "silent new=$COUNT total=$TOTAL since=$LAST"
  exit 0
fi

log_hook "nudge new=$COUNT total=$TOTAL since=$LAST"

# Deposit text is free-form prose written by whoever hit the mistake, and it
# routinely contains double quotes — the migrated entries alone quote error
# messages and doc headlines. Interpolating it raw produced invalid JSON,
# which the harness cannot parse, so the hook failed silently exactly when it
# had something to say. Escape backslashes then quotes, and flatten newlines.
NEWEST_ESC=$(printf '%s' "$NEWEST" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' ')

cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"$COUNT of the $TOTAL pending candidates in .mistakes/worklog/ were deposited since the last audit ($LAST). Most recent: ${NEWEST_ESC} — Consider running the audit-mistakes-log skill in a fresh session; it's meant to run decontextualized from whatever deposited these, not mid-task here. It stamps .mistakes/last-audit even when it promotes nothing, which silences this until new candidates arrive."}}
EOF
