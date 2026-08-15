# STRIDE Repairability Basin Audit v1 Result

## Decision

The read-only 16-process audit completed all 81 registered compact-blocker
trigger events with every identity and integrity check passing. It supports a
bounded rescue at each *new* platform signature, but rejects accumulating more
blockers repeatedly on the same unchanged platform.

The authorized next protocol is
`bounded_one_rescue_per_new_platform_signature`. It remains a mechanism screen,
not runtime, TTF or promotion evidence.

## Why immediate rescue remains difficult

| Diagnostic | Immediate unresolved (50) | Immediate escape (31) |
| --- | ---: | ---: |
| Residual external blocker present during rescue | 100.0% | 96.77% |
| Trial-local compaction applied | 46.0% | 58.06% |
| Blocker cap reached | 88.0% | 83.87% |
| Rescue neighborhood size, mean / median | 34.86 / 35.5 | 32.71 / 32.0 |
| Initial PP failed-order fraction, mean | 0.898 | 0.928 |
| Rescue PP failed-order fraction, mean | 0.896 | not applicable |
| Rescue internal blockers, mean | 14.40 | 9.35 |
| Rescue new conflict pairs, mean | 51.38 | 38.74 |

Residual external blockers are almost universal in both outcome groups. Their
presence-rate difference is only +3.23 percentage points, below the frozen
+15-point requirement. Repeatedly adding blockers on the same signature would
therefore not target the failures specifically and would enlarge neighborhoods
that are already larger in the unresolved group.

The failed rescue usually reaches about 90% of its native PP order before the
last insertion fails. This supports the interpretation that the state has high
internal coupling and order sensitivity; the exact rollback reveals that
difficulty but does not create it.

## Why a successful escape can still form another platform

Among the 23 complete H=8 observations that escaped immediately, 16 remained
durable and seven formed a new platform.

| Diagnostic | Durable H=8 (16) | New platform H=8 (7) |
| --- | ---: | ---: |
| Original selected-blocker retention after rescue | 22.32% | 6.89% |
| Adjacent exact candidate repeats through H=8 | 1.13 | 3.57 |
| Unique candidate sets through H=8 | 6.13 | 4.29 |
| Mean Jaccard to rescue action | 0.266 | 0.299 |
| New blocker outside original rescue in platform window | not applicable | 100.0% |

All seven new-platform cases expose a blocker outside the original rescue
membership during the three-rollbacks window. They also retain almost none of
the original blocker evidence and repeat candidate sets more often. This passes
the frozen per-new-signature mechanism rule.

## Consequence for the next mechanism

The next screen must use one native PP call per decision and must never retry
inside a decision. The original platform can trigger one compact-blocker rescue.
If that changes the state and a later, different repair fingerprint accumulates
three exact rollbacks, that new signature may trigger one new compact-blocker
rescue from its own PP diagnostics. A signature is never rescued twice and an
episode is capped at three interventions. `time_limit` never triggers rescue.

This design responds to newly observed repair structure without pretending that
all residual blockers should be accumulated indefinitely.

## Frozen artifacts

- Report SHA-256:
  `4bd3aaaf1b48a6eb6ade75965100124846f9d923b262dc01e48affcf3711cf56`.
- Audit rows SHA-256:
  `1ed44f4a7f12cd48253add2f98071cf65c6100158c807074a61baa82d9f1ee68`.
- Run configuration SHA-256:
  `75d97db03b3806460cc33c54545908d41580d8b036b15ab836a1a4fd6f3fd8c7`.
- Registration SHA-256:
  `503d83993d0566a22af98219086177b2bdeee71fe5ca5483d6a7c36eb7617429`.
