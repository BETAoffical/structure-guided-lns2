# STRIDE StructShell Audit v1 preregistration

## Purpose

This is a read-only audit of why the original four-size StructPool produced useful actions and whether an outcome-blind structural cutpoint can safely replace its hard-coded per-family preferred sizes.

The audit addresses one question only: should the next candidate generator keep all four equal alternatives `8, 16, 24, 32`, or is there already enough frozen evidence for a state-derived structural cutpoint? It does not train a model, modify a runtime pool, execute PP, or make a TTF or long-tail-avoidance claim.

## Frozen evidence

- 98 non-tail development states with the complete four-size family grid and 16 paired PP seeds.
- 78 Maze difficult states with 1,367 V2 candidates, 1,135 StructPool candidates, 209 current-step robust structural actions, and 41 states with current-step structural opportunity.
- 45 outcome-enriched checkpoints with 270 bounded forced-continuation episodes. These episodes are used only to test whether a proposed filter would discard previously observed beneficial first actions; they are not an independent validation cohort.

All input paths and SHA-256 values are frozen in `configs/stride_structshell_audit_v1_registration.json`. The pre-run backup is branch `codex/backup-before-structshell-audit-20260813` and bundle `../backups/structure-guided-lns2-before-structshell-audit-20260813.bundle` with SHA-256 `58c6b0a9911d2a5842183512da500d3f1d5b33f2294973a7fb5a48f249a53e9c`.

## Rules fixed before analysis

The primary rule is `structural_knee`. Within each state and family, every available size receives an equal-weight structural coverage value from incident-event coverage, internal-conflict coverage, component coverage, and a preregistered family-specific signal. Each signal and size are min-max normalized within that state-family group. The rule selects the maximum of:

```text
mean(normalized structural coverage) - normalized nominal size
```

Ties choose the smaller size and then candidate ID. A family with one available size retains it. Missing candidates are never imputed.

The following are descriptive comparators only and cannot be selected post hoc:

- support-nearest size;
- original fixed family sizes;
- the complete equal four-size grid.

## Required analyses

1. For `8→16`, `16→24`, and `24→32`, report set nesting, lost agents, added shell size, Jaccard, structural coverage deltas, current-step PP deltas, family, and map.
2. On the non-tail grid, report each rule's size distribution, current-step regret, exact best-size rate, and two seed-half best-size agreement.
3. On the 78 Maze states, report recall of the 41 old opportunity states and 209 robust actions, overall and per map.
4. On bounded continuation, report how often each rule retains beneficial, neutral, and adverse tested first actions. This is coverage evidence, not proof that the rule would select or avoid the action online.

## Frozen decision

`structural_knee` is design-ready only if every registered gate passes. Comparator results cannot substitute for a primary-rule failure. If any gate fails, the next design must retain `8, 16, 24, 32` as equal size alternatives and remove hard-coded family preferred sizes. This audit cannot authorize training, runtime integration, TTF testing, or a claim that self-loops have been prevented.

The next safe step after a fallback is a separately preregistered, ranker-free budget/reduction audit over the equal four-size pool, followed by independent result-blind bounded forced-continuation confirmation.
