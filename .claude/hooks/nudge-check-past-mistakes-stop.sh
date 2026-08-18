#!/bin/bash
# Stop hook — nudges the acting agent to run check-past-mistakes against
# finished work, but only when there's actually an uncommitted diff to
# check. A clean tree means either nothing changed this turn (purely
# conversational) or the work already landed in a reviewed commit, so
# there's no fresh diff to cross-check and the nudge would be noise.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"Stop","additionalContext":"Uncommitted changes are present. Before wrapping up, run the check-past-mistakes skill (.claude/skills/check-past-mistakes/SKILL.md) against the diff. It checks only CLAUDE.md's already-loaded graduated invariants, not the full MISTAKES.md archive, so it's cheap."}}
EOF
fi
