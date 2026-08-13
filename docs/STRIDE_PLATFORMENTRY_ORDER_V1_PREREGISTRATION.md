# STRIDE PlatformEntry Order v1 preregistration

## Question

Can one explicit, pre-action conflict-priority PP order applied to the same historical neighborhood reduce entry into a persistent rollback platform under the identical frozen continuation controller?

This is a mechanism experiment, not a candidate-pool, runtime, TTF, or generalization claim. It performs exactly one PP call for the forced first action. It never tries one order and then retries another at runtime.

## Why this branch is separate

RepairDependencyPool v1 failed its zero-solver reconstruction gate, but that gate mixed different mechanisms. Across the 45 platform witnesses, the observed escape and causal-intervention annotations cross as follows:

| Observed escape | Set | Order | Joint | Residual PP |
|---|---:|---:|---:|---:|
| same-set order escape | 18 | 10 | 2 | 6 |
| set-change escape | 2 | 3 | 0 | 2 |
| right-censored | 1 | 0 | 0 | 1 |

The six external-blocker-backed set-change cases used by the failed static gate contain only two `set_defect` annotations, plus three `order_defect` and one residual-PP case. Requiring a new neighborhood to reconstruct every escape-added blocker therefore over-demanded an agent-set intervention where the observed causal probe did not support one. The RepairDependencyPool branch remains frozen; no compactness cap or gate is relaxed.

The present cohort is all 36 cases whose real frozen trajectory escaped without changing the agent set but with a changed repair order. It is outcome-enriched and may only answer the mechanism question above.

## Frozen intervention

At each registered `first_structural_selection` state:

- `native_order`: force the historical selected agent set once, with the registered paired PP seed and native seeded shuffle;
- `conflict_priority_order`: force the identical agent set once and explicitly order agents by descending pre-action conflict degree, then agent id;
- after the forced action, return to the exact matching frozen StructPool or SlotPool continuation controller.

Both arms share the restored repair-structure fingerprint (paths and conflict edges), conflict count, candidate id, agent set, trial index, and first-action PP seed. Reset-only bookkeeping is regenerated and is not confused with repair structure. Only `repair_order` differs. Trial indices 0--3 run first for every case and arm. Trial indices 4--7 may run only as a uniform all-case extension after the initial gate passes.

## Platform label

The primary event is entry, within 64 repair decisions, into at least three consecutive transitions that all:

1. report `replan_success=false`;
2. leave the exact repair-structure fingerprint unchanged;
3. therefore leave paths and the conflict-edge set unchanged;
4. remain on the same repair fingerprint throughout the streak.

Each episode is bounded at 64 decisions or 180 seconds, with a 240-second process timeout and a 300-second outer worker fuse. `repair_limit` and `wall_timeout` are valid right-censoring; an external process timeout or execution error stops collection immediately.

## Gates

Initial extension requires complete paired evidence, lower overall platform rate, no success-rate loss, and no map worsening by more than five percentage points.

The eight-trial mechanism gate additionally requires:

- the upper endpoint of a 10,000-replicate case-cluster paired bootstrap interval for `order - native` platform risk to be below zero;
- no success-rate loss;
- lower normalized fixed-horizon conflict AUC;
- lower Kaplan--Meier restricted mean repair decisions through decision 64;
- no map platform-rate worsening by more than five percentage points.

Failure stops the order branch. Passing permits only a separately preregistered result-blind confirmation. It does not authorize training, runtime integration, multiple-PP retry, TTF testing, or a default-controller change.

## Execution identity

- Registered parent: `bbebdb44f4acad14d69e5523c95da657a177417d`.
- Cases: 36.
- Initial episodes: 288.
- Maximum episodes after uniform extension: 576.
- Non-TTF concurrency: 16 independent episode workers with episode-atomic outputs.
- Reset qualification: one current-code V2-full qualification set, built once with 16 workers under the already registered 360-second reset fuse and reused by every isolated episode. The actual continuation episodes retain the stricter 240-second process timeout.
- Registration: `configs/stride_platformentry_order_v1_registration.json`.
