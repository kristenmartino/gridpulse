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
# Decision is "ask", never "deny". A live `Closes #N` is legitimate and
# common; what the rule requires is that it was verified first, which is a
# question only the human can answer. Denying would train people to
# disable the hook.

INPUT=$(cat 2>/dev/null)

emit_ask() {
  # Escape for JSON embedding: backslashes, quotes, then newlines.
  local reason
  reason=$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}')
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":"%s"}}\n' "$reason"
  exit 0
}

# A guard that cannot run must say so. Failing open in silence is the
# configured-and-inert shape this repo already has a rule about.
if ! command -v jq >/dev/null 2>&1; then
  printf '%s\n' '{"systemMessage":"close-keyword guard skipped: jq not found on PATH. CLAUDE.md'"'"'s Closes #N rules are unenforced for this command."}'
  exit 0
fi

CMD=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
[ -n "$CMD" ] || exit 0

# Only commands that author a commit message or a PR/issue body.
printf '%s' "$CMD" | grep -qE '(^|[;&|[:space:]])(git[[:space:]]+(commit|merge)|gh[[:space:]]+(pr|issue)[[:space:]]+(create|edit))([[:space:]]|$)' || exit 0

KEYWORD='([Cc]lose[sd]?|CLOSE[SD]?|[Ff]ix(e[sd])?|FIX(E[SD])?|[Rr]esolve[sd]?|RESOLVE[SD]?)'
LIVE_PATTERN="${KEYWORD}[[:space:]]*:?[[:space:]]*#[0-9]+"

printf '%s' "$CMD" | grep -qE "$LIVE_PATTERN" || exit 0

REFS=$(printf '%s' "$CMD" | grep -oE "$LIVE_PATTERN" | grep -oE '#[0-9]+' | sort -u | paste -sd' ' -)

# Case 1 — the documented trap: the keyword sits inside backticks or
# quotes, which reads as "I am quoting this, not firing it." GitHub
# disagrees. This is almost always unintended, so it gets the louder text.
if printf '%s' "$CMD" | grep -qE "\`[^\`]*${LIVE_PATTERN}[^\`]*\`"; then
  emit_ask "BACKTICKED CLOSE-KEYWORD — this will still close ${REFS}.

GitHub scans commit messages and PR bodies for close keywords and ignores backticks, code spans and surrounding prose. Quoting one does not make it inert; this is the trap that re-closed an issue on 2026-05-29 in the very commit written to document it (CLAUDE.md, 'The backtick/quote trap').

If you mean to close ${REFS}: verify with 'gh issue view <N> --json title,state' first, then drop the backticks.
If you are only referring to it: break the pattern — write the keyword and number non-adjacently, or use a #NNN placeholder.
To flip issue state, use 'gh issue reopen|close <N>' — a keyword edit cannot undo a keyword."
fi

# Case 2 — a live reference. Legitimate, but CLAUDE.md requires it be
# verified against the actual issue before it is written, because a
# wrong number closes the wrong issue and silently corrupts the roadmap.
emit_ask "This will close ${REFS} on merge.

CLAUDE.md requires verifying every close reference before writing it: 'gh issue view <N> --json title,state', confirm the title matches this work. A number written from memory closes the wrong issue and leaves the right one open (PR #165 said 150 when the issue was 148).

Confirm you have verified ${REFS} — or cancel and check."
