# STRIDE Compact-Blocker Rescue Continuation v1 Result

## Outcome

The registered initial phase completed all 45 frozen `first_repeat_stall`
states, trials 0--3 and five paired arms: 900 atomic episode artifacts. All
identity, pairing, native-order, one-rescue, blocker-cap and completeness checks
passed with zero execution errors and zero process timeouts.

The compact-blocker interaction did **not** pass every registered initial gate.
It substantially reduced next-decision non-escape relative to the full blocker
rescue, while preserving success and improving normalized fixed AUC and
restricted repair decisions. However, its mean rescue PP wall did not decrease.
The uniform trial 4--7 extension is therefore forbidden and this development
branch stops without training, runtime integration or TTF testing.

## Compact-eligible mechanism outcomes

Semantic compaction was applicable to 35 of the 45 states and produced 81
paired rescue-trigger events. On this registered subset:

| Arm | Next-decision unresolved | Escape | Success | Normalized fixed AUC | Restricted repair decisions | Rescue PP wall |
|---|---:|---:|---:|---:|---:|---:|
| full blocker rescue | 75.31% | 24.69% | 49.29% | 0.38569 | 39.33 | 12.933 s |
| compact + blocker rescue | **61.73%** | **38.27%** | 49.29% | **0.38254** | **38.07** | 13.031 s |

The paired compact-blocker minus full-blocker differences were:

- next-decision unresolved: -13.58 percentage points, state-cluster 95% CI
  [-25.84, -2.56] points;
- rescue PP wall: +0.099 seconds, state-cluster 95% CI
  [-0.143, +0.336] seconds.

The first primary mechanism outcome is stable, but the registered wall-time
gate requires a decrease and its upper confidence limit below zero. The point
estimate is instead slightly slower, so the gate fails regardless of the
favorable escape result.

## Whole-cohort outcomes

Across all 45 states and 119 rescue-trigger events, compact plus blocker changed
the full blocker arm as follows:

- next-decision unresolved: 72.27% to 63.03%;
- success: unchanged at 48.89%;
- normalized fixed AUC: 0.42752 to 0.42507;
- restricted repair decisions: 40.32 to 39.33;
- mean rescue size: 36.00 to 34.03 agents;
- mean rescue PP wall: 13.343 to 13.394 seconds.

These measurements include the forced first action and the single rescue cost.
They are concurrent bounded-continuation measurements, not isolated raw TTF.

## Map diagnosis

The compact-blocker minus full-blocker next-decision unresolved differences
were -16.22 points on `maze-128-128-1`, 0.00 on `maze-128-128-2`, and -19.23
points on `maze-32-32-4`. No map exceeded the registered five-point worsening
limit. The hardest `maze-128-128-2` group remained completely right-censored
within 64 repair decisions for every arm, so neither rescue variant established
an end-to-end success gain there.

## Interpretation and stop decision

The interaction evidence is useful: action-visible semantic removal can make a
failure-informed blocker rescue more likely to escape on its next decision
without lowering bounded success. This is stronger than either a generic
"smaller is better" rule or a claim that adding blockers alone is sufficient.

It is nevertheless not a promotable mechanism. The registered objective also
required lower rescue PP wall, and that criterion failed. We must not extend
seeds, select favorable maps, relax the wall gate, train a selector, integrate
the mechanism at runtime or run TTF on this result. Any future revisit requires
a separately motivated implementation that reduces the cost of compact-blocker
repair and a fresh preregistration, rather than retuning on these 45 states.

## Registered artifacts

- initial report SHA-256:
  `26502C21D448FFC7A14395BA4D6A478067498C1A8E5E4D8BE3BE9608F5DE2D74`
- registration SHA-256:
  `59A19D8B03B004E5B4AB3D099CD4CB08F54C725F5BD39A97A41A3FAB3195A810`
- collection run fingerprint:
  `3D1275576BE4CA73D79861F936A0B23306DA20B3F0FE384654AD0B2DD1BBC0C5`

