# Interview Prep — GridPulse

> **Status: STUB.** Real STAR stories drafted in PR-C2 (communication
> artifacts, next session per the 2026-05-20 wider replan).

## Format

Each story uses **STAR** (Situation / Task / Action / Result) compressed
to ~90 seconds of speaking time. Stories are drawn from real recent PRs,
not synthesized.

## Seed stories identified from this session + recent merges

Drafts will be expanded in PR-C2. Each tagged with the interview-question
archetype it best answers.

1. **"Tell me about a trade-off you made."**
   *The big-bang Redis flip (PR #114).* Textbook answer is dual-write +
   parallel-read + cutover. I almost built it — then asked why we'd take
   on 4 phases of complexity for a single-tenant portfolio app.
   Accepted ~1 hour of warming downtime instead.

2. **"Walk me through a bug you debugged."**
   *The xaxis collision (PR #117).* Production threw `TypeError: got
   multiple values for keyword argument 'xaxis'`. Tests passed locally.
   Root cause: `update_layout(**_layout(...), xaxis=...)` collided when
   PLOT_LAYOUT contained an xaxis default. Three hypotheses, one right
   answer, regression-test class added so tests catch what tests
   should catch.

3. **"Tell me about a time you chose what to NOT do."**
   *The scenario simulator heuristic (PR #119).* Three hypotheses for
   why wind/solar deltas produced zero ΔPeak. None right. The real
   answer was the panel intentionally doesn't call the full physics
   engine — model loading per slider drag would be too slow. Added
   small coefficients to the heuristic instead. Full-fidelity wiring
   parked as a documented follow-up.

4. **"Tell me about a data-quality decision."**
   *Import-dominated BAs (V3.η).* User-reported "Highest-Stress
   Region: CPLW · 1071%." Root cause: EIA-860M counts in-territory
   generation, but CPLW imports 97% of its power. Built `IS_IMPORT_DOMINATED`
   set + replaced denominator with peak demand × reserve margin.
   `_STRESS_RELIABLE_CEILING` retained as defense-in-depth.

5. **"What's the biggest open issue you'd address with more time?"**
   *Model drift monitoring ([#121](https://github.com/kristenmartino/gridpulse/issues/121)).* The 2026-05-19 PJM walkthrough surfaced
   a 47 GW spread between XGBoost / Ensemble / Prophet / ARIMA on the
   same horizon. Holdout MAPE wouldn't catch this — it's between-training
   drift. Real next investment.

## Practice instructions

After PR-C2 lands:
- Read each story aloud, time it (target ~90 sec)
- Record with Loom or QuickTime; review for verbal stumbles
- Rotate which 3 you rehearse weekly so all 5 stay fresh
