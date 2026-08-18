#!/bin/bash
# One-shot migration: MISTAKES.md's inline Worklog list -> one file per
# deposit under .mistakes/worklog/.
#
# Written as a script rather than done by hand on purpose. The migration is
# the one step in this refactor where a deposit could genuinely be lost, and
# MISTAKES.md gains new entries from other sessions continuously — so this
# has to run last, against whatever main looks like at merge time, and it
# has to prove it moved everything rather than being trusted.
#
# Safe to re-run: entries already migrated are detected by content and
# skipped, so running it again after main moves picks up only what is new.
#
#   ./scripts/migrate_worklog_to_files.sh --dry-run   # show what would move
#   ./scripts/migrate_worklog_to_files.sh             # move them
#
# Deliberately bash 3.2 compatible — no mapfile, no associative arrays.
# macOS still ships 3.2 as /bin/bash, and the first draft of this script
# used both and died on the machine it was written for.
#
# Timestamps: deposits carry a second-resolution UTC stamp in the filename,
# which is what the audit uses to tell "arrived since I last ran" apart from
# "already reviewed". Historical entries have only a date, so they are
# spread across sequential seconds starting at 00:00:00Z on their own date.
# That preserves their relative order and keeps every one of them strictly
# older than any real deposit made later the same day.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/MISTAKES.md"
DEST="$ROOT/.mistakes/worklog"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f "$SRC" ] || { echo "FATAL: $SRC not found"; exit 1; }
mkdir -p "$DEST"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Pull the Worklog section's entry lines only. Anchored on the section
# headings so prose, the marker comment and the Analyzed section cannot leak in.
awk '
  /^## Worklog/      {f=1; next}
  /^## /             {f=0}
  f && /^- [0-9]{4}-[0-9]{2}-[0-9]{2} \[/ {print}
' "$SRC" > "$TMP"

TOTAL=$(grep -c '^- ' "$TMP" || echo 0)
echo "Found $TOTAL Worklog entries in MISTAKES.md"
[ "$TOTAL" -gt 0 ] || { echo "Nothing to migrate."; exit 0; }

migrated=0
skipped=0
prev_date=""
seq_n=0

while IFS= read -r line; do
  [ -n "$line" ] || continue
  body="${line#- }"                                  # strip list marker
  date="${body%% *}"                                 # YYYY-MM-DD
  cat=$(printf '%s' "$body" | grep -oE '^\S+ \[[a-z0-9-]+\]' | grep -oE '\[[a-z0-9-]+\]' | tr -d '[]')
  [ -n "$cat" ] || cat="uncategorised"

  # Already migrated? Compare on content, not filename — an earlier run may
  # have assigned a different sequence number to the same entry.
  if grep -rqxF "$body" "$DEST" 2>/dev/null; then
    skipped=$((skipped + 1))
    continue
  fi

  # Per-date sequence without associative arrays: entries arrive in file
  # order, so a scalar reset on date change is sufficient.
  if [ "$date" != "$prev_date" ]; then
    prev_date="$date"
    seq_n=0
  fi
  # Skip sequence numbers whose file already exists, so a re-run cannot
  # overwrite a previously migrated entry that happened to share a slot.
  while [ -e "$(printf '%s/%sT%06dZ-%s.md' "$DEST" "$date" "$seq_n" "$cat")" ]; do
    seq_n=$((seq_n + 1))
  done
  out=$(printf '%s/%sT%06dZ-%s.md' "$DEST" "$date" "$seq_n" "$cat")
  seq_n=$((seq_n + 1))

  if [ "$DRY" -eq 1 ]; then
    echo "  would write $(basename "$out")"
  else
    printf '%s\n' "$body" > "$out"
    echo "  wrote $(basename "$out")"
  fi
  migrated=$((migrated + 1))
done < "$TMP"

echo
echo "migrated=$migrated skipped-already-present=$skipped"

# Verification: every source entry must be findable in the directory, matched
# as a whole line. This is the check the script exists for — a count alone
# would pass while silently mangling a line.
if [ "$DRY" -eq 0 ]; then
  missing=0
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    body="${line#- }"
    grep -rqxF "$body" "$DEST" 2>/dev/null || { echo "MISSING: $body"; missing=$((missing + 1)); }
  done < "$TMP"
  if [ "$missing" -gt 0 ]; then
    echo "FAILED: $missing entries did not survive migration"
    exit 1
  fi
  echo "VERIFIED: all $TOTAL entries present in $DEST (exact whole-line match)"
fi
