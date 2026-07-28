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

These rules are enforced by `tests/runtime/test_stall_shadow.py` and
`tests/runtime/test_stall_oracle.py`.

## Current shadow protocol

`v2-stall-shadow` executes the exact `v2-full` winner and records only
diagnostics. It audits unchanged-state thresholds 3, 4, and 6, requires at
least two distinct PP attempts, looks ahead three unmodified v2 decisions, and
marks a trigger as premature if v2 changes state during that window. The
diagnostic configuration rejects `deployment_enabled=true`.
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

The PP-only Oracle report distinguishes arm, neighborhood, selector, and
candidate-pool failures. It explicitly leaves repairer failure unevaluated;
that claim requires a separate registered stronger-repair experiment.
