# Warehouse Fixed16 Development Screen v1

## Purpose and claim boundary

This is a development-only Warehouse screen, not a solver speed claim and not
a runtime promotion. It asks whether one fixed-size structural family can beat
frozen V2 consistently enough to justify a separate size study. Official
Adaptive is deliberately absent from development and is added only after one
winner is frozen for a fresh 60-key, three-arm confirmation.

The four single-family arms are all-state fixed-family ablations. They are not
a high-stress router and do not classify a map or conflict structure. Every
valid non-terminal repair state receives the registered family at nominal size
16. The legacy Boundary16 arm directly reuses
`topology_boundary_augmentation`; it is not the StructShell boundary subset.
Although the Boundary algorithm is historical, every result in this screen is
newly rerun under the new producer identity.

## Cohort and execution

- Four checksum-pinned Warehouse layouts.
- Development task master seed `20260808`, task seeds `233/277`, and both
  `opposite_exchange` and `uniform_random` variants.
- Reset-only qualification tests loads `400/600/800/1000`. Each map uses the
  lowest load for which every task-seed/solver-seed reset satisfies both
  variant thresholds: opposite-exchange has at least 16 conflicts and random
  has at least one conflict. There is no replacement or outcome-based filter.
- Half A is `(233,19),(277,21)`; Half B is `(233,20),(277,22)`.
- Formal timing is strict rotating serial: 32 paired keys, six arms, 192
  episodes, one worker. The bounds are 180 seconds solver wall time, 240
  seconds episode process time, and 300 seconds outer job time.

The variants are matched on map and selected load, but endpoint generation is
variant-specific; they are not same-start counterfactuals.

## Arms and unequal candidate budgets

- `v2_only`: frozen V2 candidates.
- `boundary16_static_cache`: legacy Boundary16, at most two added candidates.
- `bottleneck16`, `component16`, `hotspot16`, `path_overlap16`: one registered
  StructShell family at size 16, at most one source candidate per decision.

Boundary therefore has a larger augmentation budget than each family arm.
This is an explicit positive-control difference, not evidence that the family
algorithms were compared under equal candidate counts. Candidate count and
selection cost are reported and used only for the registered near-tie rule.

## Gates and stopping rule

A challenger passes only when all of the following hold:

1. integrity is complete with zero execution errors or process timeouts;
2. success count is not below V2;
3. overall mean restricted TTF is strictly below V2;
4. Half A mean restricted TTF is strictly below V2;
5. Half B mean restricted TTF is strictly below V2.

Normalized wall AUC, repair rounds, PP time, selection time, and paired-faster
fraction are diagnostic only. Per-variant, per-map, and map-by-variant results
are also diagnostic and cannot rescue a failed hard gate.

Among passing arms, the minimum mean restricted TTF wins. Arms within less
than 1% of that minimum are treated as tied; the tie is resolved by lower mean
source-candidate count per controller decision, then lower episode-mean total
selection time, then controller name. If no arm passes, the fixed16 branch
stops and V2 remains the default.

## Reserved final confirmation

This runner must never generate or reset final tasks. It only reserves master
seed `20260831`, task-seed namespace `941/977/1013`, and solver seeds
`1201/1301/1409`. After a development winner exists, exactly 20 final task
definitions must be frozen before any reset, yielding 60 paired keys. The
fresh final comparison is Official Adaptive versus V2 versus the one frozen
winner. The same four available standard Warehouse layouts may be reused, so
the final claim is task/solver result-blind rather than map-disjoint.

## Result template

- Qualification: reset coverage, selected load per map, and both variant
  minima.
- Integrity: coverage, paired initial state/conflicts, clock, censoring,
  actions, semantics, errors, and timeouts.
- Hard gates: success and restricted-TTF deltas overall, Half A, and Half B.
- Diagnostics: normalized wall AUC, repairs, PP, selection, added candidates,
  paired-faster fraction, per variant, per map, and map by variant.
- Decision: eligible arms, near-tie set, development winner or stop.
- Claim: explicitly state that development is not a speed/default claim and
  whether a fresh 60-key three-arm confirmation is authorized.
