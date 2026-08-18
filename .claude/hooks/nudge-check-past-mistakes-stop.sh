#!/bin/bash
# Stop hook — nudges the acting agent to run check-past-mistakes against
# finished work.
#
# KNOWN-WEAK TRIGGER, tracked for replacement: the dirty-tree condition is
# a poor proxy for "real work happened this turn." It stays silent right
# after a commit (clean tree) — which is precisely when work is about to
# be pushed — and it fires on turns that changed nothing but inherited a
# dirty tree. The durable fix is a PreToolUse guard at the commit / push /
# PR boundary, where every graduated invariant in CLAUDE.md actually
# applies. Until that lands this remains a best-effort backstop, not the
# primary enforcement point.
#
# Loop guard: check-past-mistakes may deposit a Worklog line to
# MISTAKES.md, which dirties the tree and would re-satisfy the condition
# below. Claude Code sets stop_hook_active when it is already continuing
# because of a Stop hook; bail out in that case. Matched with grep rather
# than jq so a missing jq cannot make this silently inert.
INPUT=$(cat 2>/dev/null)
if printf '%s' "$INPUT" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
  exit 0
fi

if [ -n "$CLAUDE_PROJECT_DIR" ]; then
  ROOT="$CLAUDE_PROJECT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
fi

# -C pins the query to this worktree; a bare `git status` would answer for
# whatever cwd the hook inherited, and this repo runs several worktrees.
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"Stop","additionalContext":"Uncommitted changes are present. Before wrapping up, run the check-past-mistakes skill (.claude/skills/check-past-mistakes/SKILL.md) against the diff. It checks only CLAUDE.md's already-loaded graduated invariants, not the full MISTAKES.md archive, so it's cheap."}}
EOF
fi
