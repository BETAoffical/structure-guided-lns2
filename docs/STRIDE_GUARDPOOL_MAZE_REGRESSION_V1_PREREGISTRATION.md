# STRIDE GuardPool Known-Maze Regression v1

## Status

This is a preregistered regression test, not independent generalization evidence
and not a formal TTF claim.  It is run only after the frozen SlotPool model was
reproduced on all 24 fresh-map confirmation states and the runtime semantics
tests passed.

## Frozen key

- map: `maze-128-128-1`
- task: `derived_opposite_exchange`, task seed 233, 100 agents
- solver seed: 3
- expected initial conflicts: 66
- frozen V2 result: success in 15 repair iterations

The result of this key was excluded from selection of the eight-round stall
threshold.  The threshold was frozen from 29 other V2 episodes and 402 repair
decisions before this regression is read.

## Treatments

1. `v2-full`: exact original candidate pool and frozen V2 ranker.
2. `v2-plus-structpool`: frozen six-candidate Full Speed2 StructPool.
3. `v2-plus-slotpool`: full family-by-size grid reduced to six challengers by
   the frozen SlotPool pairwise model, without a stall guard.
4. `stride-guardpool-v1`: the same SlotPool treatment with an eight-round
   no-progress guard.

All treatments use run-to-completion execution, deterministic paired PP replay,
the same PP/SIPPS implementation, no scientific time limit, and no remaining-
time condition.

## Guard semantics

At each active decision, V2 first chooses its anchor from the exact base pool.
The frozen SlotPool model retains six candidates from the complete 8/16/24/32
structural grid.  V2 then ranks the base pool plus those six challengers.  When
eight consecutive repairs fail to strictly reduce the conflict count, all
structural challengers are suppressed and selection is exactly V2 until a
strict conflict decrease occurs.

## Hard gates

- all four episodes must be present with zero execution errors;
- initial fingerprint and 66 initial conflicts must match;
- no invalid actions or fingerprint mismatches;
- V2 must succeed in exactly 15 repair iterations;
- GuardPool must succeed in at most 30 repair iterations;
- PP seeds must match across all common decision indices.

Full StructPool and unguarded SlotPool are ablations.  Their failure or long
tail is reported but does not weaken the GuardPool safety gate.  GuardPool may
enter a development TTF Quick only if every hard gate passes.
