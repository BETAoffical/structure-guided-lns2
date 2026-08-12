# STRIDE RepairDependency Predictability v1 Report

## Status and scope

This preregistered read-only audit tested whether the external blockers exposed
by failed PP repairs can be predicted compactly before an action is executed.
It evaluated five frozen, outcome-blind predictors on the 45 causal-audit
states, 16 paired trial indices per state, and three Maze maps. Observed PP
blockers were used only as evaluation targets. They were not predictor inputs,
training labels, or runtime features.

Integrity and execution checks passed:

- 45/45 states, 720 state-trials, and 3,600 predictor rows;
- 715 applicable blocker-tail trials and five registered not-applicable trials;
- no runtime, TTF, future-trajectory, or worker-identity fields in scientific
  rows;
- 16 context workers and 16 evaluation workers, with 15 distinct processes in
  the final 32 evaluation tasks;
- single-coordinator deterministic output writes.

The audit is diagnostic only. It is not a candidate-pool implementation, a
trained model, a runtime controller, a PP replay, or a TTF experiment.

## Primary result

The primary `repair_dependency_frontier` did not pass the frozen readiness
gates.

| Metric | Result | Gate | Status |
| --- | ---: | ---: | --- |
| Mean observed-blocker recall | 0.5472 | >= 0.50 | Pass |
| Mean recall lift over matched random | 0.4311 | >= 0.15 | Pass |
| Mean expansion ratio | 0.9706 | <= 0.50 | **Fail** |
| Recall advantage over direct boundary | 0.0844 | >= 0.10 | **Fail** |
| Maximum total-neighborhood fraction | 0.6250 | < 0.80 | Pass |
| First seed-half recall | 0.5457 | >= 0.45 | Pass |
| Second seed-half recall | 0.5488 | >= 0.45 | Pass |

The per-map recall gate also failed:

| Map | Recall | Mean added agents | Expansion ratio | Random lift |
| --- | ---: | ---: | ---: | ---: |
| `maze-128-128-1` | 0.6255 | 14.35 | 0.4693 | 0.4509 |
| `maze-128-128-2` | 0.6771 | 55.75 | 2.4054 | 0.5800 |
| `maze-32-32-4` | 0.3797 | 11.53 | 0.4195 | 0.3057 |

The small Maze missed the preregistered per-map recall floor of 0.40. The
second large Maze achieved high recall only by adding, on average, more than
twice as many agents as the original selected neighborhood. This violates the
purpose of a compact dependency expansion even though the resulting set is not
near-global relative to that map's full agent population.

## What the predictor comparison shows

| Predictor | Recall | Mean added agents | Expansion ratio |
| --- | ---: | ---: | ---: |
| Direct conflict boundary | 0.4629 | 19.71 | 0.7928 |
| Spatial low-degree frontier | 0.0784 | 2.96 | 0.1076 |
| Temporal corridor frontier | 0.1007 | 2.87 | 0.1057 |
| RepairDependency frontier | 0.5472 | 24.41 | 0.9706 |
| Full temporal boundary reference | 0.6831 | 50.76 | 1.8531 |

There is genuine pre-action signal: the primary predictor substantially beats
matched random sets and is stable across seed halves. The failure is the
precision/coverage trade-off. Compact spatial or temporal frontiers miss most
real blockers; broad conflict/temporal closure raises recall by approaching or
exceeding the original neighborhood size. The primary predictor's incremental
recall over the already-large direct boundary is only 0.0844.

This narrows the root cause. A blocker is often determined by the alternative
path and repair-order state created inside a candidate-conditioned PP attempt,
not solely by the current paths and static map geometry. Current-state closure
therefore cannot identify a compact sufficient set reliably enough. Increasing
closure depth or candidate size would reproduce the earlier over-expansion
failure rather than solve it.

## Determinism and 16-core feasibility

The complete analysis was rerun independently with the same frozen config and
16-worker scheduler. The primary and repeat runs produced byte-identical
scientific rows, reports, and execution reports:

- rows SHA-256:
  `6141f3735d706cfb70c64fa4b563b0ff1faadd9f49eb934f18fcbff2b8127f83`;
- report SHA-256:
  `2396310b3ac75ab519aa4015442ba44880c1e159162f50c63a81fd70ea9297e8`;
- execution-report SHA-256:
  `8455c4bed1f73681fbb4a268f68855a3769c24e2096fe7a8cbd0b9d8836dd458`.

Validation after analysis:

- WSL Python, 16-process xdist: 807 passed, 35 skipped in 34.76 seconds;
- Linux CTest, `-j16`: 11/11 passed;
- existing Windows `lns2_tests.exe`: passed;
- repository hygiene: zero errors and 24/24 evidence hashes verified.

Fine-grained `state x trial_index x predictor` dispatch is therefore suitable
for future non-TTF audits, including the tail where fewer states remain. Raw
TTF experiments remain outside this concurrency regime.

## Decision and next step

The frozen hard-stop applies. Do not:

- construct or promote the current RepairDependency pool;
- lower the failed thresholds after observing the result;
- proceed to paired PP replay, model training, runtime integration, or TTF;
- select only successful maps or states;
- expand candidates through unrestricted temporal closure.

The next admissible mechanism study should be candidate-conditioned and
transactional: run a bounded tentative PP repair in a cloned state before the
action is committed, record the actual failure cut and order position, and
either reject the candidate or propose a bounded dependency augmentation.
This is an upfront repairability-aware proposal mechanism, not a post-stall
fallback. It must be separately preregistered with fixed probe budgets,
outcome-blind candidate coverage, paired seeds, strict no-commit semantics,
and explicit overhead accounting before any runtime or TTF claim.

## Reproducibility

- producer milestone: `9963ce6d006c32f79ca0f6938f30dc64d7c4ff14`;
- registration SHA-256:
  `64031db04b85730bc67b69deec8155f4cccef97c46bb3474533dedf4a01f50f5`;
- input-manifest SHA-256:
  `45a2d484ad85797172bdae82f73e978b454c95451bb1eb7a04db88c94c32c125`;
- scientific-row SHA-256:
  `6141f3735d706cfb70c64fa4b563b0ff1faadd9f49eb934f18fcbff2b8127f83`;
- report SHA-256:
  `2396310b3ac75ab519aa4015442ba44880c1e159162f50c63a81fd70ea9297e8`;
- execution-report SHA-256:
  `8455c4bed1f73681fbb4a268f68855a3769c24e2096fe7a8cbd0b9d8836dd458`.
