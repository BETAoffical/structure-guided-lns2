# STRIDE SafeSlot Label Readiness v1 Result

## Outcome

The registered pre-residual audit passed every integrity and opportunity gate.
It compared 1,368 structural candidates with an exact base-only V2 anchor over
98 states from 16 maps. Each comparison used the same 16 PP seeds for the
challenger and anchor, producing 21,888 exact candidate-anchor trial pairs.

The existing current-step evidence contains a real but still provisional
SafeSlot learning signal:

- 84/98 states (85.71%) contain at least one pre-residual positive challenger;
- 504/1,368 structural candidates (36.84%) meet the frozen current-step rule;
- all 16 maps contain at least one such opportunity;
- frozen SlotPool retains a positive on 82/84 positive states (97.62%);
- it retains 332/504 positive candidates (65.87%) within its six-slot budget.

This means the unified StructPool/SlotPool candidate stage does not need to be
discarded: after exact V2 anchoring, it exposes enough stable one-step
alternatives to justify the next measurement. It does not establish that those
alternatives leave an easier residual problem or reduce TTF.

## Decision

Only the registered one-step post-repair residual-structure teacher collection
is authorized. The fixed scope is the exact V2 anchor plus all six frozen
SlotPool candidates for every state, without selecting states or candidates by
their measured outcome. One exact overlap leaves 685 unique actions and 10,960
native PP trials.

No SafeSlot model may be trained until that collection is complete and the
frozen residual-label gates pass. Runtime integration and TTF claims remain
disallowed.

Artifact SHA-256 values:

- report: `884fc6d46316613b13551ac23625215a1b084f9f4de61cf2cbffc0d4c0570275`;
- comparison rows: `95c4c1d0094f6598223f9b1612ec8ef921b3efb734526219f80dd3954d694a25`;
- state rows: `67ebf07620734bbc63f1f1ccf2bff521e23f76b0b7530f0c750a8d240c7d03b2`.
