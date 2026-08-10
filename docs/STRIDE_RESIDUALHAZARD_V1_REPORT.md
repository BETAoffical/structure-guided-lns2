# STRIDE ResidualHazard v1 Result

## Outcome

The preregistered ResidualHazard target-readiness audit completed all 66 frozen
first-divergence states, 132 exact actions, and 2,112 paired PP repairs.  Every
integrity check passed, including exact reproduction of the parent conflict
counts and post-repair fingerprints.  No collection error or timeout occurred.

Target readiness failed.  None of the six structural residual measurements had
the registered tail-versus-control direction on all three maps and for both
challengers.  No target is selected, and `stride-residualhazard-v1` stops
without model training, threshold search, runtime integration, or a TTF test.

## Target-readiness results

| Measurement | Seed-stable tail cases | Stable tail cases with no worse residual conflict count | All-map direction | Both-challenger direction | Ready |
| --- | ---: | ---: | --- | --- | --- |
| New residual-pair ratio | 20 | 18 | no | no | no |
| Selected/unselected residual-pair ratio | 20 | 17 | no | no | no |
| Low-degree residual-event ratio | 4 | 3 | no | no | no |
| Residual conflict-cell Herfindahl | 8 | 7 | no | no | no |
| Largest residual-component agent ratio | 7 | 4 | no | no | no |
| Outside boundary-queue agent ratio | 20 | 17 | no | no | no |

The first, second, and sixth measurements have many seed-stable adverse/severe
states, even after requiring that the challenger leave no more residual
conflict pairs than V2.  That is action-level repeatability, not a usable tail
label: beneficial/control states frequently show the same or stronger effect.

## Cross-map contrasts

Each value is the mean challenger-minus-V2 residual delta in tail states minus
the corresponding mean in control states.  The preregistered hazardous
direction is positive.

| Measurement | `maze-128-128-1` | `maze-128-128-2` | `maze-32-32-4` |
| --- | ---: | ---: | ---: |
| New residual-pair ratio | +0.0991 | -0.1348 | -0.1139 |
| Selected/unselected residual-pair ratio | +0.0676 | -0.2722 | -0.1398 |
| Low-degree residual-event ratio | -0.0473 | +0.0887 | +0.000004 |
| Residual conflict-cell Herfindahl | -0.0057 | -0.0833 | -0.0406 |
| Largest residual-component agent ratio | +0.1926 | -0.2418 | -0.0187 |
| Outside boundary-queue agent ratio | +0.1490 | -0.4554 | -0.0933 |

No measurement is positive on all maps.  In particular, the apparent new-pair
and boundary effects on `maze-128-128-1` reverse on both other Maze maps.

## Challenger contrasts

| Measurement | SlotPool | StructPool |
| --- | ---: | ---: |
| New residual-pair ratio | -0.0383 | -0.0213 |
| Selected/unselected residual-pair ratio | -0.2238 | -0.0743 |
| Low-degree residual-event ratio | +0.0364 | -0.0244 |
| Residual conflict-cell Herfindahl | -0.0227 | -0.0191 |
| Largest residual-component agent ratio | +0.0907 | -0.0651 |
| Outside boundary-queue agent ratio | -0.1140 | -0.1023 |

The low-degree and largest-component measurements reverse between SlotPool and
StructPool.  The remaining measurements are consistently negative rather than
the frozen positive hazard direction.  Reversing a sign or constructing a
weighted target after observing these outcomes is prohibited.

## Interpretation

Together with the parent action replay, this result narrows the failure
mechanism:

1. the first structural action is usually not immediately worse than V2 on
   conflict reduction;
2. several post-PP residual structures are repeatable across PP seeds; but
3. none of the tested one-step structures distinguishes later long tails
   consistently across maps and challenger generators.

The observed long tail is therefore more likely a sequential policy-state
interaction: repeated structural activation, neighborhood overlap, conflict
pair recurrence, or loss of action diversity over multiple decisions.  The
current evidence does not support converting any of those possibilities into
a new label or model.  Existing V2 fallback and historical no-progress guards
remain separate runtime safeguards, not evidence that ResidualHazard works.

## Integrity and artifacts

All registered categories were retained: 24 adverse/severe and 42 controls.
Every state contains exactly two actions and 16 trials per action with strictly
paired PP seeds.  Future trajectory, TTF, runtime, and controller success fields
were not used as features or labels.

- report SHA-256:
  `85ac20a49011ddd74bc4aa5128a4519b6a382b5d73afd39100813d85e3719d56`
- residual trials SHA-256:
  `f3aeb05d2c833a98bd2d36da4857bb993d532c6e88bf5b54718f8573a6101471`
- residual state metrics SHA-256:
  `ffe31204d4aeaeff5bc69e81527c0d893919e42bc08318e62db39527f3d0c05a`
- run configuration SHA-256:
  `640edecce1d05277be126422c173857b504cbb2ead68de24b3e22151b510147d`

## Decision

- Stop `stride-residualhazard-v1`.
- Do not train a residual-hazard model.
- Do not select a threshold, flip a metric direction, combine metrics, or
  filter the cohort after observing this report.
- Do not run a ResidualHazard TTF experiment or claim generalization.
- Any successor sequential-hazard study requires a new scientific rationale
  and preregistration; it is not an automatic continuation of this failed
  target.
