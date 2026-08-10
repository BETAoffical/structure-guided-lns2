# STRIDE TailSwitch v1 preregistration

## Question

The earlier full-episode evidence shows that structural challengers can create
long tails. Paired first-action replay shows that their first action is often
better in immediate conflict reduction, while ResidualHazard found no single
cross-map scalar post-state target. The remaining causal ambiguity is whether
the tail is caused by the first structural action, by repeatedly applying the
structural policy, or by their interaction.

TailSwitch is a diagnostic, not a new controller and not a TTF promotion test.
It uses every one of the 66 already frozen first-divergence comparisons. No
state may be removed after observing TailSwitch outcomes.

## Two-by-two intervention

Each episode restores the exact repair structure immediately before the frozen
first divergence. It then runs one of four policies:

| Policy | First action | Remaining decisions |
| --- | --- | --- |
| `v2-then-v2` | frozen V2 action | frozen V2 controller |
| `struct-then-v2` | frozen structural challenger action | frozen V2 controller |
| `v2-then-struct` | frozen V2 action | matching StructPool or SlotPool controller |
| `struct-then-struct` | frozen structural challenger action | matching StructPool or SlotPool controller |

The first action is forced exactly once with the already paired PP seed. From
the second action onward, the unchanged online controller generates, ranks and
repairs its own neighborhood. PP, SIPPS, V2 ranking and structural candidate
generation are not changed.

## Fuse and outcome

Every restored branch has at most 200 repair decisions and 300 wall seconds.
`repair_limit` and `wall_timeout` are valid right-censored state trajectories;
process failures are errors. The causal outcomes are fixed-200-step normalized
conflict AUC, fixed-horizon final conflicts and completion/censoring status.
TTF and runtime are reported only as diagnostics and are not labels.

The primary contrasts are the two first-action effects, the two continuation
effects and the factorial interaction registered in
`configs/stride_tailswitch_v1_registration.json`.

## Evidence gate

A mechanism requires an adverse direction in at least two thirds of eligible
paired comparisons, at least eight adverse comparisons, two maps, four tasks,
two solver seeds and two examples from each challenger type. These thresholds
are frozen before TailSwitch execution. Failure to pass a gate means
inconclusive evidence, not evidence that topology is irrelevant.

## Claim boundary

TailSwitch may distinguish an irreversible first-action mechanism from a
repeated-policy feedback mechanism. It cannot establish faster TTF,
generalization, or production safety, and it cannot authorize training or a
default-controller change.
