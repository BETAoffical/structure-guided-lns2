# STRIDE SafeSlot Residual Teacher v1 Result

## Collection integrity

The preregistered collection completed all 98 states, 685 unique actions, and
10,960 native PP trials with zero errors and zero timeouts. Every action has
trial indices 0--15, and every action in the same state uses the same PP seed
at a given trial index. All reruns reproduce the previously recorded conflict
count, repair-success flag, and after-repair structure fingerprint exactly.

The collection contains no TTF, runtime, future trajectory, Cost-to-Go,
Receding-Q, or remaining-repair-round field. It describes only the state
immediately after one paired repair.

Collection artifact SHA-256 values:

- residual trials: `64408d57c052637c134afe4957f5e142dfc2dd152f5a2f8101cab0782acb7a00`;
- action aggregates: `6d6f9d9d22d1ae7e18195297d158672e78d25e7bebdc501a898bbf850747cf88`;
- state manifest: `5e147aa9d2cb4a2f88a6521e12d5160336ec7d43fe95f6b7b124aee239105314`;
- state artifact tree: `a2d7a47ba3be704a4dabcaecf1127f354cac4d04e225e544cbfae7754b5abd9b`.

## Frozen-label result

The analysis evaluates all 588 frozen SlotPool challengers. Before residual
structure is considered, 332 satisfy the paired current-step rule. The frozen
residual teacher rejects 131 of those: their immediate conflict reduction is
promising, but the resulting conflict structure is less stable than the exact
V2 anchor under the registered mean, fixed-half, or component condition.

The final product contains:

- 201 safe-replacement positives out of 588 challengers (34.18%);
- 201/332 pre-residual positives retained after the residual test (60.54%);
- 64 positive states, above the registered minimum of 20;
- all 16 maps with a positive, above the registered minimum of eight.

The final label artifact SHA-256 is
`81e57ecaca26c7b9ef6a8f4c9f38242e85db19196b33c8e4c83a74b575a17baa`.

## Decision and claim boundary

The result authorizes only preregistration of whole-map grouped SafeSlot gate
training. It shows that the unified StructPool/SlotPool candidate stage exposes
enough robust one-step alternatives to train an abstaining challenger-versus-
V2 classifier. It does not show runtime improvement, fewer terminal repair
rounds, or lower TTF.

The next model must use only pre-action fields, keep the exact base-only V2
anchor fixed, prevent maps and states from crossing folds, and abstain to V2
when its calibrated replacement threshold is not met. Runtime integration
remains prohibited until held-out gate quality passes its own preregistered
criteria.
