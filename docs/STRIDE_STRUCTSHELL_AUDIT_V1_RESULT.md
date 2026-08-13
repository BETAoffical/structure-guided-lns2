# STRIDE StructShell Audit v1 result

## Decision

The preregistered read-only audit completed with all identity and integrity
checks passing. The outcome-blind `structural_knee` rule failed every quality
readiness gate, so it is not design-ready. The next pool must retain the equal
size alternatives `8, 16, 24, 32`; the old fixed family preferences must not be
reintroduced.

The frozen report is
`build/stride-structshell-audit-v1/structshell_audit_report.json`, with SHA-256
`84bfae2a26ae642da719af67559cf4f0d8ade96b8391f31b524e854939fc1883`.

## What the old size grid contains

All 1,260 adjacent size transitions are exact nested shells. Moving from 8 to
16, 16 to 24, or 24 to 32 adds exactly eight agents and removes none; the mean
Jaccard values are respectively 0.50, 0.667, and 0.75. The four sizes therefore
represent genuine nested alternatives around the same structural seed, not
four unrelated neighborhoods.

Larger is not uniformly better. Mean one-step improvement generally rises for
several families, but no-progress and marginal quality are family- and
state-dependent. The two fixed eight-seed halves agree on the best size in only
68.10% of state-family groups.

## Why a single natural cutpoint was rejected

| Diagnostic | Structural knee | Frozen gate |
|---|---:|---:|
| Maze opportunity-state recall | 37/47 = 78.72% | at least 90% |
| Maze robust-action recall | 70/209 = 33.49% | at least 80% |
| Worst per-map opportunity recall | 60.00% | at least 80% |
| Non-tail mean current-step regret | 0.12064 | at most 0.02 |
| Non-tail seed-half best-size agreement | 68.10% | at least 80% |
| PreTail beneficial-action membership recall | 13/47 = 27.66% | at least 80% |

The original fixed family-size rule also retained only 70/209 robust Maze
actions and 24/47 previously beneficial PreTail actions. It is therefore not a
valid fallback merely because it was the original StructPool implementation.

The complete equal four-size grid retains 47/47 opportunity states, 209/209
robust actions, and all 47 previously beneficial PreTail actions by
construction. These are membership results only: they show that the pool does
not discard observed escape-capable actions, not that an online selector will
choose them or that self-loops are prevented.

## Identity correction

The initial report was rejected before acceptance. Every PreTail `case_id`
occurs once at `first_structural_selection` and once at `first_repeat_stall`.
An unfiltered dictionary allowed the stall state to overwrite the intended
first-selection state. The accepted run filters the frozen checkpoint type and
verifies task, solver seed, state fingerprint, and candidate membership for all
180 paired comparisons. No rule, threshold, state, action, or outcome was
changed.

## Claim boundary and next step

This result changes no runtime code, trains no model, executes no new PP, and
makes no TTF or long-tail-avoidance claim. It establishes only the next
candidate-generation contract:

1. preserve the original structural families;
2. expose `8, 16, 24, 32` symmetrically, without per-family preferred sizes;
3. preserve the complete V2 pool and V2 anchor separately; and
4. preregister a ranker-free budget audit for budgets 6, 8, and 12.

That audit must evaluate an outcome-blind, size-balanced diversity reducer.
If no smaller budget preserves the frozen opportunity, robust-action,
per-map, and bounded-continuation membership gates, the full exact-deduplicated
four-size structural pool remains the candidate contract. No ranker training or
runtime integration is authorized before this decision.
