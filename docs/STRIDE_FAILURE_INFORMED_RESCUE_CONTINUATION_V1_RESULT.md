# STRIDE Failure-Informed Rescue Continuation v1 Result

## Outcome

The registered initial and uniform extension phases completed all 45 frozen
`first_repeat_stall` states, eight paired trials and three arms: 1,080 bounded
continuation episodes. All schedule, state, shared-first-attempt, native-order,
single-rescue, blocker-cap and completeness checks passed with zero execution
errors and zero process timeouts.

The failure-informed blocker arm passed every registered mechanism gate. Among
233 paired first-action exact rollbacks, its probability of remaining on the
same repair fingerprint and conflict-edge signature after the next decision
fell from 73.82% under the frozen controller to 62.66%. The state-cluster
paired risk difference was -0.1116 with a 95% bootstrap interval of
[-0.2010, -0.0180].

This is evidence for one narrow post-failure mechanism: blockers observed in a
real PP rollback can improve the *next* neighborhood decision. It is not yet a
runtime-policy, raw-TTF, generalization or complete long-tail result.

## Aggregate bounded-continuation outcomes

| Arm | Next decision unresolved | Next decision escape | Success | Normalized fixed AUC | Restricted mean repair decisions | Mean repair wall* |
|---|---:|---:|---:|---:|---:|---:|
| frozen controller | 73.82% | 26.18% | 45.56% | 0.45377 | 41.00 | 72.40 s |
| same set, fresh seed | 72.10% | 27.90% | 43.06% | 0.46285 | 41.91 | 74.45 s |
| blockers, fresh paired seed | **62.66%** | **37.34%** | **47.78%** | **0.40951** | **40.21** | **71.40 s** |

`*` The wall measurements include actual bounded repair work collected with 16
concurrent workers. They are mechanism-accounting measurements, not isolated
paired raw TTF.

The blocker arm improved the primary unresolved risk by 11.16 percentage
points versus control and by 9.44 points versus same-set fresh-seed rescue.
Its blocker-versus-same-set 95% interval was [-0.1990, 0.0125], so the data do
not establish a precise independent effect size against that secondary arm.
The registered secondary gate required the blocker arm not to be worse than
same-set on the point estimate, which it satisfied.

Success rose by 2.22 points versus control, normalized fixed AUC fell by 9.75%,
and restricted mean repair decisions fell by 0.80. Same-set reseeding alone
did not reproduce the improvement: it lowered success, worsened AUC and used
more repair decisions than the frozen controller in this continuation test.

## Why the reported platform rate is unchanged

All three arms report an identical historical platform-entry rate of 82.22%.
This is expected and is not the treatment outcome: the rescue is eligible only
after the shared first action has already exact-rolled back. A later decision
cannot erase that earlier event. The preregistered primary estimand is instead
whether the immediately following decision remains on the same repair and
conflict signature.

Consequently this experiment supports faster *post-entry* escape. It does not
show that platform entry was prevented.

## Map diagnosis

| Map | Eligible rescues | Control unresolved | Blocker unresolved | Difference | Control / blocker success |
|---|---:|---:|---:|---:|---:|
| `maze-128-128-1` | 84 | 79.76% | 78.57% | -1.19 pp | 46.09% / 48.44% |
| `maze-128-128-2` | 58 | 68.97% | 48.28% | -20.69 pp | 0.00% / 0.00% |
| `maze-32-32-4` | 91 | 71.43% | 57.14% | -14.29 pp | 77.21% / 80.88% |

No map's primary point estimate worsened. The strongest next-decision effect
occurred on `maze-128-128-2`, but all 96 episodes per arm on that map remained
right-censored at 64 decisions / 180 seconds. Thus leaving the immediate
signature is not sufficient to solve the hardest cases. On
`maze-128-128-1`, the effect on the primary signature was only 1.19 points;
this group must not be hidden by the aggregate result.

The mean observed external-blocker count was 7.50, close to the cap of eight.
That makes the registered cap and selection rule important subjects for an
independent confirmation; it does not authorize retuning them on this cohort.

## Interpretation and next step

The experiment separates membership information from a seed-only retry. After
an exact rollback, reusing the same set with a fresh seed was not enough under
full bounded continuation, whereas adding only blockers reported by that
failure improved immediate escape and all aggregate safety metrics. This is
the first stable evidence in this branch that the *observed failed repair*
contains useful information for the next candidate membership.

The correct next step is a separately preregistered, result-blind bounded
confirmation on new states and maps. It must freeze the current trigger,
eight-blocker cap, one-rescue limit, native order and time budgets. It must
report the hard `maze-128-128-2` right-censoring behavior and the weak
`maze-128-128-1` effect explicitly. Only a successful independent confirmation
could authorize a later isolated paired raw-TTF experiment.

No result here authorizes model training, runtime integration, controller or
candidate-pool replacement, repeated rescue, TTF testing or a claim that the
long tail has been eliminated.

## Validation

- WSL Python test suite with 16 workers: 932 passed, 35 skipped;
- Linux CTest with `-j16`: 11/11 passed;
- Windows native `build/windows/Release/lns2_tests.exe`: passed;
- repository hygiene audit: 0 errors, 24/24 evidence entries verified.

## Registered artifacts

- implementation/protocol commit: `3d09dd33c1ae14df043059efbceab89c7ed8d0a1`;
- registration SHA-256:
  `9C38B04DF5F13C8F33A2C9C14435F2E705B9A1666DC95199F7A3969969567FA1`;
- initial report SHA-256:
  `C230DC4941114412E6E7DA8E90710B458EB223CDE1B6607F989EA59CBCA61EFC`;
- extended report SHA-256:
  `CDB9886EF1DB64CB314D5CEEF71C9C19EBF29BF431AEF33832EED646D1D69E99`;
- initial run configuration SHA-256:
  `45E817C59AC08148596FBDA43F106D63D2E5C44185B20E8443FC4FE966C7FD68`;
- extension run configuration SHA-256:
  `424AEA61B8F760C2A620D904AA2560F470BC0179DA7391B5265ED691C0F00E72`;
- final collection status SHA-256:
  `B6F8176BFF68DF1541CA9B55E4B4DBC0F541B6D2EB8014DC910FBB2CD3BB5341`.
