# STRIDE SafeSlot Residual Teacher v1

## Scope

This collection reruns one native PP repair only to describe the state produced
immediately after that repair. It does not follow future controller decisions.
The fixed action set contains the exact base-only V2 anchor and all six frozen
SlotPool out-of-fold candidates for every one of the 98 states. One state has an
exact anchor/candidate overlap, leaving 685 unique actions and 10,960 trials.

No action is selected from the readiness result. The complete frozen SlotPool
budget is collected, including negative and ambiguous candidates.

## Residual structure

Each after-state records conflict pairs, conflict events, active conflicting
agents, largest conflict component, bottleneck events, and repeated-event
excess. Bottleneck events touch either a static articulation cell or a free cell
of degree at most two.

Four ratios relative to the same pre-repair state form an equal-weight residual
risk: conflict events, largest component, bottleneck events, and repeated-event
excess. The final label rule is frozen before collection:

- the candidate must already be a pre-residual positive;
- mean residual risk cannot exceed the V2 anchor;
- neither fixed eight-seed half may have higher mean risk;
- no individual component mean may regress by more than 0.02.

These fields are teacher outputs only. They cannot be model inputs at runtime,
because the after-state does not exist before selecting the action.

Training may be designed only if the completed audit retains at least 20
positive states, 40 positive candidates, and eight maps. These gates are frozen
before residual outcomes are collected.

## Integrity and claim boundary

Every rerun must reproduce the previously recorded PP seed, conflicts after
repair, and repair fingerprint. Any mismatch fails the state. The collection
stores no time, TTF, future action, Cost-to-Go, or remaining-round field.

Completion authorizes only a map-grouped SafeSlot label audit. It does not
authorize training, runtime integration, or a speed claim.
