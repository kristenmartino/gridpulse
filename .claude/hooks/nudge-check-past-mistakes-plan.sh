#!/bin/bash
# PostToolUse(ExitPlanMode) hook — nudges the acting agent to run
# check-past-mistakes against the just-approved plan. See
# .claude/skills/check-past-mistakes/SKILL.md for what that skill checks;
# this script's only job is to make sure the nudge happens.
#
# A rejected plan is not worth checking, so skip when the tool result says
# the user declined. Matched with grep rather than jq so a missing jq
# cannot make this silently inert; unrecognised payloads fall through to
# nudging, since a redundant nudge costs a sentence and a missed one costs
# the whole point of the hook.
INPUT=$(cat 2>/dev/null)
if printf '%s' "$INPUT" | grep -qiE '"(tool_response|tool_result)"[^}]*(rejected|declined|did not approve|denied)'; then
  exit 0
fi

cat <<'EOF'
{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"A plan was just approved. Before implementing, run the check-past-mistakes skill (.claude/skills/check-past-mistakes/SKILL.md) against this plan. It checks only CLAUDE.md's already-loaded graduated invariants, not the full MISTAKES.md archive, so it's cheap."}}
EOF
