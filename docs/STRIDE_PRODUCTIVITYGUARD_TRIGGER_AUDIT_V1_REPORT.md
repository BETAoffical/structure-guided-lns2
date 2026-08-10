# STRIDE-ProductivityGuard v1 trigger audit

## Outcome

The preregistered historical-productivity audit completed all 132 TailSwitch
continuation contrasts with full input identity and no state exclusion. No
variant passed, so `STRIDE-ProductivityGuard-v1` must not be implemented as a
runtime controller from these signals.

The result is stronger than a generic threshold failure: every sequence that
repeats the same structural candidate three times has exactly zero conflict
progress and 100% residual conflict-edge persistence over the two completed
repairs visible before the third proposal. This is true for adverse,
beneficial and neutral sequences alike.

## Frozen trigger results

The base trigger is an exact three-action structural repetition. Each variant
then reads only the outcomes of the two already completed repairs. The current
proposed action is not repaired before triggering.

| Frozen variant | Adverse | Beneficial false triggers | Neutral false triggers | Passed |
| --- | ---: | ---: | ---: | --- |
| Prior two repairs have zero cumulative conflict progress | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Progress <= 5%, mean unresolved edges >= 80% | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Progress <= 5% | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Progress <= 10%, mean unresolved edges >= 80% | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |

All four variants are identical on this cohort. At the first trigger:

| Outcome class | Triggered pairs | Mean prior progress | Mean unresolved-edge fraction |
| --- | ---: | ---: | ---: |
| Adverse | 17 | 0.000 | 1.000 |
| Beneficial | 8 | 0.000 | 1.000 |
| Neutral | 20 | 0.000 | 1.000 |

The registered beneficial false-trigger maximum was 20%, but the observed rate
is 72.73%. The neutral maximum was 10%, but the observed rate is 19.23%.

## Mechanistic interpretation

This rules out a simple explanation of the long tail:

- it is not enough to say that the same neighborhood was selected repeatedly;
- it is not enough to add that conflict count failed to decrease;
- it is not enough to add that the conflict-pair set remained unchanged.

Eight beneficial continuation pairs exhibit all three properties and still
become beneficial later. Therefore PP can alter latent path timing, waiting,
route slack or bottleneck occupancy without immediately changing the conflict
graph. A conflict-graph-only selector cannot observe that hidden preparation.

The neighborhood-selection problem is consequently more specific: the current
memoryless selector cannot tell whether an unchanged conflict graph represents
useful latent path reconfiguration or a genuine policy lock-in. A fixed tabu,
third-repeat fallback or no-progress threshold would conflate the two.

## Decision

- Do not implement `STRIDE-ProductivityGuard-v1` from these rules.
- Do not tune another conflict-progress threshold on this cohort.
- Do not run a TTF promotion test or claim generalization.
- Preserve V2 as the default controller.

The next causal question is whether the original-pool V2 anchor is actually
better at the first exact-repeat/stalled state. That requires a new frozen-state
counterfactual: restore every first trigger state, compare the repeated
structural action with the original-pool V2 anchor under strictly paired PP
seeds, and retain path-level temporal descriptors. This is not authorized by
the current failed trigger audit and requires a separate preregistration before
new solver runs.

## Integrity and claim boundary

- report SHA-256:
  `80bb6c36a7b69e41fb25a0508f11bc999288694ef2a00a9edeb3dd5d2e59d164`
- states: 66;
- continuation contrasts: 132;
- adverse/beneficial/neutral: 17/11/104;
- no solver or PP rerun;
- no result-based exclusion;
- no causal, TTF-improvement or generalization claim.
