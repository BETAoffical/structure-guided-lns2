# InitLNS Context Learnability Audit

## Question

The audit tests whether map, static OD, density, and flow context improve action
ranking beyond the current dynamic repair state. It does not imitate the official
Adaptive action and does not collapse the outcome into a hand-tuned scalar reward.

The action is a `(seed agent, Target/Collision/Random, size 4/8/16)` tuple. Trial
outcomes are averaged per candidate, and Horizon-4 Pareto dominance produces
ordered candidate pairs. The Pareto objectives are feasibility, remaining
conflicts, conflict AUC, low-level generated nodes, and branch runtime.

## Features and split

- `action_seed`: requested rule and size plus seed conflict degree, delay, path
  length/stretch, and conflict-component size.
- `dynamic`: action/seed plus collision graph, delay/path distributions, repair
  phase, and accumulated low-level search statistics.
- `full_context`: dynamic plus map dimensions/topology, agent count and density,
  layout/task categories, static OD flow counts, crossing ratios, and route
  statistics from the Pilot v2 sidecars.

Pairwise examples use candidate-feature differences plus the shared state/context
vector. Keeping the shared vector is essential: subtracting two candidates from
the same state would otherwise cancel every contextual feature. Prediction is
anti-symmetrized by averaging `P(left > right)` with `1-P(right > left)`.

Training uses the existing Train split. Validation uses disjoint map and task
instances: 6 Train maps/24 tasks and 3 Validation maps/12 tasks, with zero overlap.
All models use `scikit-learn 1.5.0` HistGradientBoostingClassifier and fixed seed
`20260714`.

## Reproduce

```powershell
python scripts/run_context_audit.py `
  --collection build/repair-experience-calibration-v2 `
  --dataset build/repair-transfer-pilot-v2 `
  --output build/initlns-context-audit-v2
```

Outputs are `candidate_index.jsonl`, three pickled models,
`context_audit.json/.md`, and `closed_loop_gate.json/.md`. A failed gate returns
exit code 2 by design.

## 2026-07-14 result

| Model | Pareto top-1 | Mean AUC regret | Mean remaining-conflict regret |
| --- | ---: | ---: | ---: |
| Action + seed | 15.28% | 1.1102 | 1.2114 |
| Dynamic state | 11.11% | 1.2392 | 1.3251 |
| Full context | 15.28% | 1.2674 | 1.2821 |

The full model gained 4.17 percentage points over the dynamic model, missing the
5-point requirement. Its AUC regret was 2.28% worse, missing the required 5%
reduction. The paired bootstrap intervals were `[0.0000, 0.0972]` for hit-rate
gain and `[-0.2898, 0.2060]` for per-state AUC-regret improvement; they do not
show significant degradation, but they also do not support the required gain.

The offline gate is **FAIL**. Closed-loop evaluation was intentionally not run.

## Interpretation

This is an inconclusive negative result for the current protocol, not proof that
context can never help. Plausible limitations to examine before registering a new
audit are the small number of distinct maps, one-trial labels, only sampled
Adaptive states, coarse handcrafted topology features, and the models' strong
preference for size 4. A revised audit must state its feature/model changes and
thresholds before looking at new Validation outcomes.
