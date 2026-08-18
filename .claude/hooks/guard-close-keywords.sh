#!/bin/bash
# PreToolUse(Bash) guard for CLAUDE.md's close-keyword invariants.
#
# Why this is a guard and not a nudge: both halves of the rule are
# mechanically decidable, and CLAUDE.md's own policy says a mistake that
# can be made structurally impossible should be, rather than left to an
# agent to remember. GitHub scans commit messages and PR bodies for close
# keywords and ignores backticks and surrounding prose, so a quoted
# `Closes #150` fires exactly like a live one — that is what re-closed the
# issue the 2026-05-29 follow-up commit had just reopened.
#
# Scope is deliberately narrow: only commands that actually write a commit
# message or PR body. Read-only commands that merely mention a keyword
# (git log --grep, gh pr view) are none of this guard's business.
#
# Two decisions, deliberately split:
#
#   backticked keyword -> deny.  Quoting a close keyword is a statement of
#     intent NOT to fire it, and GitHub fires it anyway. There is no case
#     where the author wanted both the backticks and the closure, so this
#     is one of the rare hooks that can be certain. It blocks, and the
#     message carries both escapes (drop the backticks, or break the
#     pattern) so it is never a dead end.
#
#   plain `Closes #N` -> ask.  Legitimate and common; the rule only
#     requires that a human verified the number. Denying it would train
#     people to switch the hook off.
#
# The split exists because "ask" turned out not to gate anything in
# permissive or auto-approving sessions — measured 2026-08-18, the guard
# returned ask on a live reference and the command ran with no prompt. A
# guard whose decision has no force in the mode you actually work in is
# the configured-and-inert failure one layer up, so the case that can be
# decided with certainty now denies.

INPUT=$(cat 2>/dev/null)

if [ -n "$CLAUDE_PROJECT_DIR" ]; then
  ROOT="$CLAUDE_PROJECT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
fi

# Every invocation is logged, not just the ones that speak. A guard that
# never fires and a guard that never ran read identically otherwise, which
# is the failure this repo keeps rediscovering. The audit skill reads this
# to answer "is enforcement actually working, or just installed?"
LOG="$ROOT/.claude/hook-activity.log"
log_hook() {
  printf '%s guard-close-keywords %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"$LOG" 2>/dev/null || true
  # Telemetry, not an audit trail of record — bound it rather than letting
  # a PreToolUse-on-every-Bash hook grow a file forever. Safe to delete.
  if [ "$(wc -l <"$LOG" 2>/dev/null || echo 0)" -gt 2000 ]; then
    tail -n 1000 "$LOG" >"$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG" 2>/dev/null || true
  fi
}

# $1 = log kind, $2 = deny|ask, $3 = reason shown to the caller.
emit_decision() {
  # Escape for JSON embedding: backslashes, quotes, then newlines.
  local reason
  reason=$(printf '%s' "$3" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}')
  log_hook "$2 $1 refs=${REFS// /,}"
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' "$2" "$reason"
  exit 0
}

# A guard that cannot run must say so. Failing open in silence is the
# configured-and-inert shape this repo already has a rule about.
if ! command -v jq >/dev/null 2>&1; then
  log_hook "skipped no-jq"
  printf '%s\n' '{"systemMessage":"close-keyword guard skipped: jq not found on PATH. CLAUDE.md'"'"'s Closes #N rules are unenforced for this command."}'
  exit 0
fi

CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
if [ -z "$CMD" ]; then log_hook "silent no-command"; exit 0; fi

# Only commands that author a commit message or a PR/issue body.
if ! printf '%s' "$CMD" | grep -qE '(^|[;&|[:space:]])(git[[:space:]]+(commit|merge)|gh[[:space:]]+(pr|issue)[[:space:]]+(create|edit))([[:space:]]|$)'; then
  log_hook "silent not-authoring-command"; exit 0
fi

KEYWORD='([Cc]lose[sd]?|CLOSE[SD]?|[Ff]ix(e[sd])?|FIX(E[SD])?|[Rr]esolve[sd]?|RESOLVE[SD]?)'
LIVE_PATTERN="${KEYWORD}[[:space:]]*:?[[:space:]]*#[0-9]+"

if ! printf '%s' "$CMD" | grep -qE "$LIVE_PATTERN"; then
  log_hook "silent authoring-command-no-close-ref"; exit 0
fi

REFS=$(printf '%s' "$CMD" | grep -oE "$LIVE_PATTERN" | grep -oE '#[0-9]+' | sort -u | paste -sd' ' -)

# Case 1 — the documented trap: the keyword sits inside backticks, which
# reads as "I am quoting this, not firing it." GitHub disagrees. Nobody
# wants both the backticks and the closure, so this is decidable with
# certainty and therefore blocks rather than asks.
if printf '%s' "$CMD" | grep -qE "\`[^\`]*${LIVE_PATTERN}[^\`]*\`"; then
  emit_decision backticked deny "BLOCKED — a backticked close keyword still closes ${REFS}.

GitHub scans commit messages and PR bodies for close keywords and ignores backticks, code spans and surrounding prose. Quoting one does not make it inert; this is the trap that re-closed an issue on 2026-05-29 in the very commit written to document it (CLAUDE.md, 'The backtick/quote trap').

Rewrite the message one of two ways, then re-run:
  - You DO mean to close ${REFS}: verify with 'gh issue view <N> --json title,state', then drop the backticks so the intent is plain.
  - You are only REFERRING to it: break the pattern — write the keyword and the number non-adjacently, or use a #NNN placeholder.

To flip issue state, use 'gh issue reopen|close <N>' — a keyword edit cannot undo a keyword."
fi

# Case 2 — a live reference. Legitimate and common, so this asks rather
# than blocks: the rule only requires that a human verified the number,
# and a wrong one closes the wrong issue and corrupts the roadmap.
emit_decision live ask "This will close ${REFS} on merge.

CLAUDE.md requires verifying every close reference before writing it: 'gh issue view <N> --json title,state', confirm the title matches this work. A number written from memory closes the wrong issue and leaves the right one open (PR #165 said 150 when the issue was 148).

Confirm you have verified ${REFS} — or cancel and check."
