# Experiment decisions and regression guards

This file records negative and diagnostic results that must constrain later
controller changes. Build artifacts are evidence, not tracked source. Their
paths identify the preserved local result used when the decision was made.

## Evidence levels

- **End-to-end:** complete paired solver episodes under the same wall budget.
- **Pilot/quick:** bounded evaluation that can reject a method but cannot by
  itself replace the default controller.
- **Diagnostic/Oracle:** same-state, replay, proxy, or counterfactual evidence;
  it does not establish solver-level speed or quality.

## Frozen decisions

| Attempt | Evidence | Decision |
|---|---|---|
| `v2-stall-safe` size backoff and Adaptive fallback | `build/initlns-v2-stall-safe-quick-wall-v1` | Archive as a diagnostic. Early intervention can interrupt useful v2 transitions. |
| learned repair-aware rescue | `build/initlns-v2-repair-aware-quick-wall-v1` | Do not enable by default; local repair gains were not stable end to end. |
| one-step v3, H3, and S3 | `build/initlns-v3-pilot-v1`, `build/initlns-v3-horizon-pilot-v1`, `build/initlns-v3-s3-mixed-load-pilot-v5-adaptive` | Diagnostic successors only. Proxy efficiency or local Oracle gains do not imply lower solver TTF. |
| critical-conflict seed reduction | `build/initlns-v2-critical-wall-clock-600-v1` | Stop deployment: changing the seed/candidate pool changed actions and produced large wall-clock regressions. |
| frozen cost Top-3 reranking | `build/initlns-v2-cost-top3-wall-clock-pilot-v3` | Keep as failed-promotion evidence; it did not reliably beat LNS2 in complete episodes. |
| long LNS2 maze600 feasibility | `build/initlns-maze600-lns2-warm-start-extension-14400-v1` | LNS2 can also reach severe endgame plateaus; a plateau is not unique evidence of a v2 selector defect. |
| historical same-state stalled probe | `build/initlns-stalled-state-probe-v1` | Retain as a v1 descriptive artifact only. Its partial unique-neighborhood coverage and unpaired low-level PP randomness cannot establish `selector_failure` or `candidate_pool_failure`, and it is not resumable by the v2 probe. |
| exact full-pool maze600/seed2 Oracle | `build/initlns-stalled-state-full-pool-probe-v3`, `build/initlns-stall-oracle-maze600-seed2-v1` | Diagnostic `selector_failure`: frozen v2 rank 1 failed 8/8 paired PP attempts while rank 12 (size 8) and rank 17 (size 4) escaped 8/8. This proves one recoverable state, not a population rate or solver gain. |
| threshold-only v2 stall shadow | `build/initlns-v2-stall-shadow-maze-room-seed2-v1`, `build/initlns-v2-stall-shadow-maze-room-seed2-audit-v5` | Do not enable recovery. Thresholds 3/4/6 had premature-trigger rates 33.3%/66.7%/50.0%; none passed the 1% Wilson gate. Across 202 common decisions and 3,636 candidate scores, shadow and v2 had zero semantic mismatches after one legal timeout-boundary post-repair exclusion. |
| frozen generic v3 no-progress head on the exact stall Oracle | `build/initlns-v3-stall-risk-oracle-audit-v1` | Do not reuse the frozen head as a stall selector. It assigned the failed v2 winner high risk, but trial AUC was only 0.6317 and candidate Spearman 0.3567; the two 8/8 stable escapes ranked only 4th and 14th by predicted risk, while the predicted lowest-risk candidate failed 8/8. Redesign stall-specific history labels before any wider shadow controller. |
| twelve-state exact stall-escape cohort | `build/initlns-stall-escape-oracle-cohort-v4` | All 12 included exact states were `selector_failure`: across 216 frozen candidates and 936 paired PP trials, the same pools contained 58 stable alternatives. The generic frozen v3 risk head passed only 3/12 states (mean trial AUC 0.5893; mean candidate Spearman 0.1706), so it must not be reused as the recovery selector. This is exact same-state diagnostic evidence, not an end-to-end solver gain. |
| action-preserving `policy_train` stall extension | `build/initlns-stall-training-extension-60-v1`, `build/initlns-stall-training-extension-120-v1`, `build/initlns-stall-training-source-seeds12-v1`, `build/initlns-stall-training-extension-seeds12-120-v1` | Seed-0 and seed-1/2 extensions preserve frozen v2 actions and expose both natural recoveries and confirmed stalls. The seed-1/2 extension produced 9 confirmed sequences, but paired full-pool Oracle classified only 7 as `selector_failure`; 2 were PP-order-sensitive v2 neighborhoods that escaped under repeated paired attempts. Across all current exact evidence there are 9 training-side selector-failure states from four maps, still too few and too concentrated in the compartmentalized layout for a deployable learned recovery selector. |
| independent mixed-layout stall source and corrected 120-decision extension | `build/initlns-stall-recovery-training-source-v1`, `build/initlns-stall-recovery-training-extension-120-v2` | The 9-map, 54-task `policy_train` source covers compartmentalized, dead-end-aisle, and regular-beltway layouts at 400/600 agents. The corrected action-preserving extension produced 145 new natural recoveries, 20 confirmed long stalls, and 8 unresolved stalls with zero semantic mismatches. Natural recoveries remain the dominant case, so short no-progress runs cannot activate recovery. |
| first 120-decision extension with an unscaled wall budget | `build/initlns-stall-recovery-training-extension-120-v1` | Invalid as semantic-equivalence evidence. It increased 30 to 120 repair decisions but left the 120-second wall/environment limits unchanged; a legal timeout rollback at decision 29 looked like an after-state mismatch. Any decision-budget extension must scale all wall, environment, and process budgets together. |
| mixed-layout exact full-pool stall Oracle | `build/initlns-stall-recovery-training-oracle-v1`, `build/initlns-stall-recovery-training-oracle-report-v1` | All 20 planned same-state jobs have complete paired artifacts. Strict reporting classifies 19 as `selector_failure`, 1 as `no_confirmed_v2_failure`, and 0 as candidate-pool failure, with 155 stable alternatives across all three layouts. This supports a narrow stall rescue experiment, not an end-to-end controller claim. |
| deterministic rescue-order and legacy same-neighborhood trigger audits | `build/initlns-stall-rescue-order-audit-v2`, `build/initlns-stall-trigger-policy-audit-v1` | After an exact selector failure, trying frozen-v2 ranks 2 through 8 escaped in 75/76 paired PP trials and was stable in all 19 states; searching ranks 9 through 18 recovered only one additional trial. The old threshold-9 claim of 0/145 false triggers is invalid: sequence labels were defined with maximum threshold 6 plus a three-decision window, so a threshold-9 natural recovery could not enter the negative class. The audit now rejects thresholds outside the extraction-label range. Rescue order remains useful Oracle evidence, but the old trigger result cannot support deployment. |
| same-neighborhood rank-2-through-8 shadow wall smoke | `build/initlns-v2-stall-rescue-shadow-smoke-v3`, `build/initlns-v2-stall-rescue-shadow-smoke-v3/audit` | One 600-agent, 120-repair paired smoke preserved all 120 actions, 2,160 candidate scores, and final state with zero semantic mismatches or action overrides. Threshold 9 produced 3 resolved trigger events and 20 non-executed rescue suggestions. Direct shadow bookkeeping cost 0.0563 seconds total (0.469 ms/repair); cached repair-structure fingerprint time differed by only 0.0643 seconds. Raw episode time was 92.907 seconds for v2 and 88.942 seconds for shadow, but the identical trajectory and a 3.913-second PP-time difference show that this raw gap is timing noise, not a recovery speedup. The smoke measures real overhead only; it cannot establish TTF or wall-AUC gains because shadow never changes an action. |
| complete mixed-layout threshold-9 runtime shadow | `build/initlns-v2-stall-rescue-shadow-policy-train-120-v1`, `build/initlns-v2-stall-rescue-shadow-policy-train-120-v1/audit` | Across 54 complete `policy_train` episodes, 3,931 common repair decisions, and 70,675 candidate scores, shadow preserved every v2 action and state with zero semantic mismatches. Runtime evidence resolved 13 threshold-9 triggers: 9 confirmed stalls and 4 premature triggers (30.77%; Wilson 95% upper 57.63%). Three premature states had been counted as selector-failure positives by the legacy offline audit and one source-era state was excluded, exposing the label-selection bug. Threshold 9 is rejected; active recovery and training remain disabled. |
| cached-anchor shadow timing after duplicate-fingerprint removal | `build/initlns-v2-stall-rescue-shadow-smoke-v4-token`, `build/initlns-v2-stall-rescue-shadow-smoke-v4-token/audit` | The shadow reuses the repair fingerprint already computed by the normal trace path instead of rescanning paths/conflicts/SOC. A 400-agent, 120-repair paired smoke again had zero action, score, rank, or state mismatches. Direct shadow bookkeeping was 0.0435 seconds total (0.363 ms/repair); the remaining raw episode delta was dominated by normal PP/system timing noise. This is an overhead result, not a solver speedup. |
| order-balanced three-repair trigger counterfactual | `build/initlns-v2-stall-trigger-counterfactual-h3-v2-order-balanced` | All 13 runtime threshold-9 triggers were replayed from the exact repair state with two paired PP seeds and opposite branch execution order (204 branch rollouts, 52 policy comparisons, zero errors). Replacing rank 1 with the first distinct rank-2-through-8 suggestion lowered final conflicts in only 7/26 trials and had mean conflict advantage 0.154, so a fixed next-rank rescue is rejected. A post-hoc best rescue lowered final conflicts in 24/26 trials (mean advantage 91.27) and dominated rank 1 in 17/26, proving substantial full-pool selection headroom. This is diagnostic Oracle evidence only: no online policy can know the post-hoc winner, no branch reached feasibility, and no source action changed. |
| pre-action two-stage stall cohort and paired labels | `build/initlns-stall-preaction-cohort-pilot-v2`, `build/initlns-stall-preaction-oracle-pilot-v1`, `build/initlns-stall-preaction-label-pilot-v4` | A result-blind, map-balanced control sample and a separate outcome-enriched training sample were selected before executing the target action from 71 synthetic `policy_train` v2 episodes. After repeated-state deduplication, the inventory contained 2,969 states from 13 maps; 109 registered states received complete full-pool labels with four paired PP seeds and zero seed, fingerprint, or coverage errors. Frozen v2 rank 1 was a stable failure in 47 states; 44 had a stable rescue among ranks 2 through 8 and the remaining three only beyond rank 8. Candidate-pool failures were zero. Rescue winners were distributed across every rank 2 through 8 (4/7/6/8/8/8/3 states), rejecting any fixed-rank rule and supporting a small stall-specific trigger plus rescue-ranker pilot. Training is restricted to 61 enriched states (32 positive, 29 negative); all four map folds and all six layout/agent cells contain both labels. The 48 outcome-blind controls are excluded from fitting; after exact labels they contain 36 negative controls and 12 true selector failures. A zero-false-positive 1% Wilson gate requires 381 negative controls, so at least 345 additional labelled negatives are still needed and deployment remains forbidden. |
| compact pre-action trigger and rescue learnability pilot | `build/initlns-stall-preaction-features-pilot-v2`, `build/initlns-stall-preaction-model-pilot-v3`, `build/initlns-stall-preaction-model-pilot-v4` | The trace-reconstructed matrix contains 109 states, 872 rank-1-through-8 candidates, and 41 preregistered current-state/history/candidate fields with zero fingerprint or pool mismatches; future class, map id, and measured wall time are excluded. The cohort deduplicated by repair-state fingerprint and retained the first occurrence, however, so every registered history counter is zero. Its four-fold OOF result therefore rejects a static preemptive trigger, not a correctly observed temporal-history trigger: ROC AUC 0.3847, average precision 0.4738, and no threshold achieved zero false positives with useful recall. The rescue ranker improved enriched-state stable escape from fixed rank 2's 34.38% to 62.50%, but failed the result-blind positive-control quality check: escape rose from 41.67% to 66.67% while mean conflict reduction fell from 3.542 to 0.500. A preserved strict-probability artifact showed the same failure mode (1.312 mean reduction). Do not collect 345 additional negative controls for this preemptive model and do not deploy either ranker. High escape probability alone repeats v3's low-impact-neighborhood bias; any future rescue work must start only after conservative observed stall evidence and must retain held-out conflict reduction and complete wall-clock quality. |
| direct frozen-v2 stall-confirmation rule audit | `build/initlns-stall-confirmation-rule-audit-v1` | Directly scanning 71 frozen v2 traces produced 3,256 action-preserving trigger evaluations for thresholds 2/3/4/6/9/12/15. No fixed unchanged-state threshold was safe. In a three-decision future window, threshold 9 was followed by natural v2 recovery in 16/33 resolved triggers (48.48%; Wilson upper 64.78%) and even threshold 15 was premature in 3/11 (27.27%; Wilson upper 56.56%). Six-decision windows were worse. Distinct-attempt minima 2 and 3 were identical in these traces. Stop stall-recovery deployment and training; next profile and optimize PP. This is direct frozen-trajectory evidence, not an end-to-end controller comparison. |

The production default remains optimized `v2-full` until a new controller
passes paired wall-clock promotion gates.

## Non-negotiable stall semantics

1. A repair with `conflicts_after >= conflicts_before` is not automatically a
   failure.
2. `state_changed_no_reduction` is a legitimate transition. It resets stall
   memory and cannot trigger recovery.
3. One `hard_failure` or `accepted_noop` identifies one failed
   `(state, actual agent set, PP seed/order)` arm. It does not prove the whole
   neighborhood or the v2 selector failed.
4. `selector_failure` requires same-state counterfactual evidence: the v2
   neighborhood repeatedly fails while another candidate from the same frozen
   full pool reliably escapes under paired PP attempts.
5. If the complete pool has no stable escape, stop training the selector and
   investigate candidate generation or the repairer.
6. Any new recovery controller must run shadow-only first. Its action override
   count must remain zero until a separate promotion decision.
7. A trigger audit cannot evaluate thresholds outside the range used to label
   its source sequences. Promotion evidence must come from direct runtime
   shadow resolution, not a label partition that makes long natural recovery
   impossible by construction.
8. Future-outcome strata may enrich training, but they cannot estimate trigger
   prevalence or false-trigger rates. Result-blind controls and enriched
   training states must be disjoint and reported separately.
9. Different candidate neighborhoods contain different agent sets, so their
   literal PP repair-order lists cannot be identical. Paired fairness means the
   same PP random seed and the same deterministic order-generation rule; exact
   order equality is required only for repeated trials of the same neighborhood.
10. Continuous escape probability cannot be the sole or lexicographically
    dominant rescue objective. A candidate policy must retain held-out conflict
    reduction and later pass complete wall-clock gates; a higher local escape
    rate alone is failed-promotion evidence.
11. A fixed next rank is not a valid rescue policy. A rescue ranker must learn
    stable escape first, expected conflict reduction second, and real repair
    time only as a tie-breaker. End-to-end TTF and wall-AUC remain the promotion
    criteria.
12. A temporal trigger dataset must identify a state occurrence or decision
    context, not only a repair-state fingerprint. Deduplicating repeated states
    by fingerprint and keeping the first occurrence erases the failure history
    that a stall trigger is supposed to learn. Static-feature conclusions may
    remain valid, but temporal-history claims from such a cohort are invalid.

These rules are enforced by `tests/runtime/test_stall_shadow.py`,
`tests/runtime/test_stall_oracle.py`, and
`tests/evaluation/test_stall_trigger_policy_audit.py`. Paired trigger
counterfactual execution order is additionally guarded by
`tests/evaluation/test_stall_trigger_counterfactual.py`.

## Current shadow protocol

`v2-stall-shadow` executes the exact `v2-full` winner and records only
diagnostics. The legacy threshold audit uses unchanged-state thresholds 3, 4,
and 6. The current conservative rescue shadow instead requires nine distinct
PP attempts for the same actual agent neighborhood and records frozen-v2 ranks
2 through 8 as non-executed suggestions. Both look ahead three unmodified v2
decisions and mark a trigger as premature if v2 changes state during that
window. Every diagnostic configuration rejects `deployment_enabled=true`.
The runtime reuses the repair-structure fingerprint already computed by the
normal trace path and retains that opaque anchor while paths, conflict graph,
and SOC remain unchanged; low-level counters and iteration numbers must not
define a stall.
The external shadow audit requires exact registered-cohort coverage, recomputes
every episode summary from transition evidence, requires all triggers to be
resolved, and applies a 95% Wilson upper confidence bound to the false-trigger
rate. The runtime summary itself never passes a gate.

Full-pool Oracle workflow:

```bash
python3 scripts/probe_stalled_state.py \
  --source <v2-full-collection> \
  --task-id <task> --solver-seed <seed> \
  --auto-terminal-stall --all-candidates --trials 8 \
  --output <probe-output>

python3 scripts/audit_v2_stall_oracle.py \
  --source <probe-output> \
  --output <oracle-output>
```

The v2 probe binds the source trace, frozen controller manifest, complete
candidate pool, producer/native identity, and deterministic
`(state, trial_index)` PP seed. Completed-but-invalid checkpoints are preserved
and rejected. Because this contract is intentionally incompatible with the
historical v1 artifact, use a new output directory rather than `--resume` on
`build/initlns-stalled-state-probe-v1`.

Aggregate a completed shadow collection without enabling recovery:

```bash
python3 scripts/audit_v2_stall_shadow.py \
  --source <v2-stall-shadow-collection> \
  --output <shadow-audit-output>
```

Audit a frozen v3 no-progress head against an exact full-pool Oracle without
training or changing any action:

```bash
python3 scripts/audit_v3_stall_risk.py \
  --probe <full-pool-probe> \
  --oracle <stall-oracle-report.json> \
  --v3-controller <frozen-v3-controller> \
  --output <stall-risk-audit-output>
```

The PP-only Oracle report distinguishes arm, neighborhood, selector, and
candidate-pool failures. It explicitly leaves repairer failure unevaluated;
that claim requires a separate registered stronger-repair experiment.

Current result: the maze600/seed2 terminal state contains stable alternatives,
so it is not a complete-pool failure. However, unchanged-state count alone is
not selective enough to activate them safely. A future rescue experiment must
first identify the failed actual neighborhood and PP attempts, then use
same-state Oracle-supported evidence. It must not revive a generic 3/4/6
counter or treat one non-reducing repair as failure.

The mixed-layout `policy_train` extension now provides 19 exact selector-failure
states from nine independent maps plus 145 action-preserving natural-recovery
controls. This is enough to reject early 3/4/6 triggers and to justify a
diagnostic rank-2-through-8 rescue suggestion, but not enough to pass the 1%
false-trigger confidence gate. Do not count longer observations of the same
repair fingerprint as new states, do not label PP-order-sensitive neighborhoods
as selector failures, and do not reuse the generic frozen v3 no-progress head.
The next controller step must remain action-preserving shadow collection until
the trigger gate passes; it must not rerun the earlier early-intervention logic.

## Current decision after trigger and rescue pilots

Optimized `v2-full` remains the quality anchor and default controller. The
pre-action trigger, rescue ranker, fixed next-rank rescue, and fixed
unchanged-state thresholds have all failed their respective diagnostic gates.
No active recovery controller is promoted or awaiting deployment. The next
engineering direction is a result-preserving PP phase profile and equivalence-
preserving PP optimization; learned recovery should be reconsidered only after
new evidence identifies a selective trigger unavailable to the current traces.

The registered pre-action cohort deliberately separates an outcome-blind,
map-balanced control role from an outcome-enriched training role. The
observational future class is label metadata and must never enter online
features. Because its state-fingerprint deduplication erased repeated-state
history, it may support static-feature and candidate-ranker diagnostics only;
it must not support a temporal trigger claim. MovingAI, formal, and OOD traces
remain excluded from training and threshold selection.
