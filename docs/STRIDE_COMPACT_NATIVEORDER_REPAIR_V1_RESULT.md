# STRIDE Compact Native-Order Repair v1 Result

## Outcome

The registered initial and uniform extension phases completed all 45 frozen
`first_repeat_stall` states, 16 paired trials and four policies: 2,880 atomic
policy artifacts. All identity, pairing, native-order, attempt-cap and
completeness checks passed with zero execution errors and zero process
timeouts.

Semantic compaction was action-visible and outcome-blind. It was applicable to
14 states, where it reduced the selected neighborhood from a mean of 30.86 to
25.43 agents. It did not request a target size. The other 31 states retained
the exact full neighborhood and remain in the whole-cohort safety summaries.

The compact branch did **not** pass its final registered gate. Its point
estimates favored compaction on the eligible states, but the state-cluster
paired confidence intervals crossed zero for both required primary outcomes.
The branch therefore stops without bounded platform continuation, candidate
pool replacement, runtime integration or TTF testing.

## Eligible-state mechanism outcomes

| Policy | Returned unchanged | Native success | Strict conflict reduction | Mean normalized reduction | Mean size | Mean PP attempt wall* |
|---|---:|---:|---:|---:|---:|---:|
| full set, single attempt | 69.64% | 30.36% | 29.02% | 0.07982 | 30.86 | 1.392 s |
| compact set, single attempt | 62.95% | 37.05% | 36.16% | 0.08680 | 25.43 | 1.306 s |
| full set + fresh native retry | 54.46% | 45.54% | 43.75% | 0.11101 | 30.86 | 2.345 s |
| compact set + fresh native retry | **49.55%** | **50.45%** | **49.11%** | 0.10610 | **25.43** | **2.104 s** |

`*` These are concurrent mechanism-collection measurements, not isolated raw
TTF measurements.

Against the full-set retry, compact retry had the following state-cluster
paired differences across the 14 eligible states:

- returned unchanged: -0.0491, 95% CI [-0.1473, 0.0402];
- PP attempt wall: -0.2406 seconds, 95% CI [-0.5910, 0.0079].

Both upper confidence limits exceed zero, so neither primary improvement is
stable under the registered criterion. The 4.91-point success increase and
5.36-point strict-reduction increase are useful mechanism signals, but they do
not override the failed primary gates.

## Map diagnosis

Eligible states occurred on two maps. On `maze-128-128-1`, compact retry
reduced returned-unchanged from 52.68% to 42.86% and raised success from
47.32% to 57.14%. On `maze-32-32-4`, returned-unchanged and success were both
unchanged at 56.25% and 43.75%; strict conflict reduction rose from 41.07% to
43.75%. The third map had no action-visibly unsupported member, so compaction
correctly fell back to the full set.

This pattern does not support a global "smaller is better" rule. It suggests
that removal of unsupported members may help a subset of large-map repairs,
but the 14-state evidence is insufficiently stable and the benefit is not
uniform across map groups.

## Whole-cohort safety view

Across all 45 states, including exact fallback states, compact retry changed:

- returned unchanged: 45.83% to 44.31%;
- native success: 54.17% to 55.69%;
- strict conflict reduction: 50.00% to 51.67%;
- mean neighborhood size: 28.62 to 26.93.

Concurrent mean attempt wall was effectively unchanged and slightly higher:
26.742 versus 26.786 seconds. This reinforces that the experiment is a
mechanism screen, not end-to-end performance evidence.

## Interpretation and stop decision

Semantic compaction did not damage aggregate repair quality and produced a
promising point improvement where unsupported members existed. Nevertheless,
the registered uncertainty test failed. We must not select the favorable map,
retune the rule on these outcomes, or claim that compaction prevents a
platform or long tail.

The safe conclusion is narrower: blindly retaining all selected agents is not
always necessary, but the present action-visible support rule is not yet a
stable deployable repair mechanism. The compact branch is closed. Any future
revisit requires an independently motivated rule and a new result-blind
cohort, rather than further seed expansion or threshold adjustment on these
45 states.

## Registered artifacts

- initial report SHA-256:
  `EC7DA8575EA556284FF43B9BCDB51A4C13AAC0EBD41D1C56F96D6EA45794D4ED`
- extended report SHA-256:
  `09C06E401546F9BF6EA3A94C536E57B2AECE49C07FAF2ABC393A1D06238DDC96`
- registration SHA-256:
  `5C82D27F5C6C5778FDEA53169152D4CB879C20EB5C9F542CACF2EDC484770EE7`
