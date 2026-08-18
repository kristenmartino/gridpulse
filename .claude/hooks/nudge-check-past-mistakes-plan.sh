#!/bin/bash
# PostToolUse(ExitPlanMode) hook — nudges the acting agent to run
# check-past-mistakes against the just-approved plan. See
# .claude/skills/check-past-mistakes/SKILL.md for what that skill checks;
# this script's only job is to make sure the nudge happens.
cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"A plan was just approved. Before implementing, run the check-past-mistakes skill (.claude/skills/check-past-mistakes/SKILL.md) against this plan. It checks only CLAUDE.md's already-loaded graduated invariants, not the full MISTAKES.md archive, so it's cheap."}}
EOF
