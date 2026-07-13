# Stage 5: Guided Simplified LNS2

Stage 5 tests one narrow claim: whether Stage 4 Repair experience improves
neighborhood selection in this project's simplified LNS2 implementation. It
does not evaluate the complete official MAPF-LNS2 algorithm.

## Closed-loop guidance

The C++ solver still generates its baseline collision-graph neighborhood
first. When guidance is enabled, it sends the seed conflict, baseline
neighborhood, current conflicts, and current paths to a persistent Python
controller.

Python reconstructs the Stage 4 Repair query, runs kNN against Train-only
experience, and maps the retrieved role template to concrete current Agent
IDs. The seed pair is always retained. Remaining Agents are ranked using role
compatibility, conflict-graph proximity, path overlap, and baseline membership.

Guidance falls back to the baseline neighborhood when:

- the query is out of distribution;
- effective probability is below the frozen threshold;
- no effective neighbor exists;
- mapping returns an invalid neighborhood;
- Python communication or inference fails.

Runs that issue at least one guidance request use Trace schema 3, which adds
baseline and selected neighborhoods, confidence, OOD, fallback reason, and
guidance timing. Conflict-free guided runs remain schema 2 because no guidance
interaction occurs. Solver search time excludes Python wait; wall time includes
it.

## Experiment protocol

- Train: builds the memory and is never queried as an evaluation split.
- Validation: selects effective-probability threshold `0.80`.
- Test: uses the frozen Validation configuration once.
- Both variants use neighborhood size 6, at most 500 iterations, a 5000 ms
  search budget, and solver seeds 1, 2, and 3.
- Baseline and guided runs alternate order within each task/seed pair.
- The exact conflict heatmap is not used because Stage 4 found that absolute
  coordinates did not transfer.

The experiment includes 72 Validation pairs and 144 Test pairs. A watchdog
retries a process once if it hangs or exceeds its search budget by more than
one second.

## Test result

| Metric | Baseline | Guided |
| --- | ---: | ---: |
| Solved | 128/144 | 128/144 |
| Final conflicting pairs | 77 | 77 |
| LNS iterations | 168 | 164 |
| Mean wall time | 1192.75 ms | 1212.54 ms |
| Mean solved sum of costs | 1536.92 | 1542.23 |

Guidance was requested 164 times, used 24 times, and fell back 140 times.
Seventeen guided iterations met the effective Repair label.

Paired outcomes were 6 guided wins, 4 baseline wins, and 134 ties. The exact
sign-test p-value is `0.754`; the success McNemar exact p-value is `1.0`.
Bootstrap 95% intervals for success, remaining conflicts, iterations, and
runtime all include zero.

The result is therefore negative at the aggregate level: current Repair
guidance does not significantly improve the simplified LNS2 baseline. There
is a weak pattern of improvement on clustered tasks and dead-end layouts, but
it is offset by losses on dense tasks and is not sufficient for a general
claim.

## Reproduction outputs

`build/stage5-validation-paired/` stores the frozen configuration and
Validation pairs. `build/stage5-test/` stores baseline/guided traces, per-round
guidance decisions, run manifests, paired statistics, and layout/task
breakdowns.

## Stage 5 v2 correction

The v1 result exposed an action-identification problem rather than evidence
that repair experience is intrinsically useless:

- Stage 3 observed only the neighborhood selected by the baseline;
- Stage 4 represented the conflict state but not a candidate action;
- role mapping reproduced the baseline Agent set in 22 of 24 guided rounds;
- effective-probability estimates saturated and did not compare alternatives.

V2 therefore generates eight deterministic candidate neighborhoods for each
state and evaluates all of them with a common deterministic Agent priority.
Trace schema 4 records full current paths, candidate membership, explicit
replanning order, validity, conflict/cost changes, and candidate runtime.
Trials are isolated from the main solver RNG and time budget.

Candidate memory stores 35 continuous map, task, state, and action features,
but Stage 5 v2.1 can fit the kNN index with a smaller feature profile. `full`
keeps the original Train-fitted set after zero-variance filtering, `dedup20`
removes strongly redundant aggregates, and `core12` keeps only a compact
diagnostic subset. Agent IDs, absolute coordinates, candidate generator names,
post-repair paths, and outcome labels are excluded from distance features.
Validation selects `k`, feature-group weights, the OOD threshold, and the
minimum predicted advantage over candidate zero for the chosen profile.

The final protocol has three arms:

1. legacy neighborhood and randomized replanning order;
2. baseline neighborhood and controlled replanning order;
3. candidate-aware kNN neighborhood and the same controlled order.

The primary comparison is arm 2 versus arm 3. Test contains 48 tasks, three
solver seeds, and three arms, for 432 runs. V1 outputs remain unchanged as the
negative historical baseline.

## Stage 5 v2 result

Train produced 253 conflict states and 2024 candidate cases. Validation
produced 84 states and selected `k=3`, candidate-group weight `2`, zero
replacement margin, and OOD threshold `0.689`. Validation's mean offline
utility gain was `0.106`, while Top-1 accuracy was `14.3%` and pairwise
ranking accuracy was `52.0%`.

The final execution order was exactly balanced: controlled preceded guided
72 times and followed it 72 times.

| Metric | Controlled | Candidate-guided |
| --- | ---: | ---: |
| Solved | 112/144 | 110/144 |
| Final conflicting pairs | 104 | 104 |
| LNS iterations | 202 | 227 |
| Mean search time | 1460.1 ms | 1535.6 ms |
| Mean wall time | 1460.2 ms | 1771.2 ms |

Paired outcomes were 9 guided wins, 9 controlled wins, and 126 ties. The
outcome sign-test p-value was `1.0`; the success McNemar exact p-value was
`0.727`. All primary bootstrap intervals included zero. Guidance was used 121
times and all 121 decisions changed neighborhood membership, but only 32 met
the effective-repair criterion.

The aggregate conclusion remains negative: candidate-aware kNN corrected the
v1 action-mapping defect but did not improve simplified LNS2 performance.
Dense tasks showed a favorable subgroup pattern, but it is not statistically
sufficient for a general claim.

The order-control result is also important. Legacy randomized replanning
solved 130/144 runs, versus 112/144 under fixed order. This difference was
significant in the paired experiment, so fixed order should remain a
diagnostic control rather than replace the legacy solver.

## Stage 5 v2.1 feature profiles

Rebuilding the candidate index with lower-dimensional profiles improved the
offline Validation score but did not produce an aggregate Test improvement.

| Profile | Features | Validation top1 gain | Oracle regret | Top-1 acc. | Rank acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 34 | 0.106 | 1.242 | 14.3% | 52.0% |
| `dedup20` | 20 | 0.213 | 1.135 | 14.3% | 50.3% |
| `core12` | 13 | 0.009 | 1.339 | 15.5% | 53.0% |

`dedup20` was selected for Test because it had the best Validation utility and
lower oracle regret. The Test result remained negative overall: controlled
solved 106/144 runs and candidate-guided solved 104/144, with 10 guided wins,
9 controlled wins, and 125 ties. Guided kept one fewer final conflict pair
overall but used more iterations and more wall time. The useful signal is still
local rather than general: compartmentalized and dense cases improved in some
paired outcomes, while regular beltway and clustered tasks regressed.

## Stage 5 v2.2 replan-order labels

V2.2 keeps the eight candidate neighborhoods but evaluates each candidate
under three deterministic replanning orders, `0,1,2`. Trace V5 stores the
per-order trials and the experience builder aggregates them into expected
candidate labels. The collector is resumable: complete traces are reused when
a long collection command is restarted.

Train produced 201 conflict states, 1608 aggregated candidate cases, and 4824
per-order cases. Validation produced 62 states, 496 aggregated candidate
cases, and 1488 per-order cases. The candidate diagnostics confirmed high
label noise: the average utility range across the three orders for the same
state/candidate was `1.79`, and 87.1% of Train states still had some candidate
better than candidate zero under the aggregated label.

| Profile | Features | Validation top1 gain | Oracle regret | Top-1 acc. | Rank acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 34 | 0.246 | 1.113 | 25.8% | 55.5% |
| `dedup20` | 20 | 0.414 | 0.945 | 24.2% | 53.5% |

`dedup20` was selected for the V2.2 Test run. The aggregate Test result is
still negative:

| Metric | Controlled | Candidate-guided |
| --- | ---: | ---: |
| Solved | 107/144 | 106/144 |
| Final conflicting pairs | 162 | 160 |
| LNS iterations | 135 | 157 |
| Mean search time | 1587.8 ms | 1631.7 ms |
| Mean wall time | 1587.9 ms | 1742.3 ms |

Paired outcomes were 8 guided wins, 12 controlled wins, and 124 ties. The
success McNemar p-value was `1.0`; paired outcome sign-test p-value was
`0.503`. Guidance was used 82 times and all 82 choices changed neighborhood
membership, but only 28 were effective repairs. Dense tasks and
compartmentalized layouts retained a useful local signal, while clustered
tasks and regular beltway layouts offset the gains. The conclusion is that
multi-order labels reduce one source of noise, but kNN ranking still is not
strong enough to improve the simplified LNS2 solver overall.

## Stage 5 v3 supervised candidate ranking

V3 keeps the V2.2 candidate traces and labels but changes the decision model.
The task is treated as supervised ranking inside one conflict state: among the
eight candidate neighborhoods, predict which one has the highest aggregated
utility. The implementation trains on Train candidate cases only, selects the
model and replacement margin on Validation only, and reads Test only for the
final frozen experiment.

The default model is `pairwise_linear`, a dependency-free pairwise linear
ranker trained from candidate utility differences. Optional scikit-learn
comparisons are available when the local environment provides sklearn:
`sklearn_logistic`, `sklearn_forest`, and `sklearn_gbdt`. All models use the
same Train-fitted feature normalizer and the same feature profiles as V2.1.
Agent IDs, absolute coordinates, generator names, post-repair paths, and
outcome labels remain excluded from input features.

The V3 final protocol compares four arms when the kNN reference is provided:

1. legacy neighborhood and randomized replanning order;
2. controlled baseline neighborhood and controlled replanning order;
3. V2.2 kNN-guided candidate neighborhood;
4. V3 ranker-guided candidate neighborhood.

The primary comparison is controlled baseline versus ranker-guided. The kNN
arm is a secondary reference to determine whether supervised ranking improves
over retrieval. If Validation does not improve over kNN, the negative result
is still reported rather than hidden.

Reproduction commands:

```powershell
python scripts/train_candidate_ranker.py `
  --memory build/stage5-v2-2-train-experience `
  --output build/stage5-v3-ranker `
  --feature-profile dedup20 `
  --models pairwise_linear,sklearn_logistic,sklearn_forest,sklearn_gbdt

python scripts/evaluate_candidate_ranker.py `
  --ranker build/stage5-v3-ranker `
  --queries build/stage5-v2-2-validation-experience `
  --output build/stage5-v3-validation

python scripts/run_stage5_v3_experiment.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --knn-index build/stage5-v2-2-index-dedup20 `
  --knn-config build/stage5-v2-2-evaluation-dedup20/selected_config.json `
  --ranker build/stage5-v3-ranker `
  --ranker-config build/stage5-v3-validation/selected_config.json `
  --output build/stage5-v3-test `
  --split test
```

The first V3 run selected `sklearn_forest` with zero replacement margin.
Validation was worse than the V2.2 kNN reference: top1 gain `0.165` versus
`0.414`, oracle regret `1.194` versus `0.945`, and Top-1 accuracy `22.6%`
versus `24.2%`.

The Test result remained negative:

| Metric | Controlled | Ranker-guided |
| --- | ---: | ---: |
| Solved | 114/144 | 109/144 |
| Final conflicting pairs | 100 | 102 |
| LNS iterations | 214 | 258 |
| Mean wall time | 1454.1 ms | 1769.1 ms |

Paired outcomes were 19 ranker wins, 23 controlled wins, and 102 ties. The
paired outcome sign-test p-value was `0.644`, and the success McNemar exact
p-value was `0.302`. Ranker guidance was used 207 times and produced 68
effective guided iterations, compared with 130 uses and 33 effective
iterations for the kNN reference in the same V3 run. The local repair signal is
therefore stronger, but it still does not translate into better aggregate
solver quality.

## Stage 5 v4 rollout labels

V4 addresses the V3 label mismatch by replacing one-step repair labels with
closed-loop rollout labels. Trace V6 is enabled with
`--candidate-rollout-horizons`; for every valid candidate/order trial, the C++
solver continues an isolated short LNS rollout from the repaired paths and
records remaining conflicts, cost, solved flag, iterations, accepted
iterations, and runtime at each horizon. The rollout does not consume the main
solver RNG or alter the main trajectory.

`build_rollout_candidate_experience.py` converts Trace V6 into
`rollout_candidate_cases.jsonl` and a compatibility `candidate_cases.jsonl`
for the existing ranker trainer. The `rollout22` feature profile adds rollout
horizon and one-step repair deltas to the previous candidate feature set.
Validation can tune conservative gates with `--conservative-gates`; these add
extra replacement margin for regular beltway maps, low-conflict states, and
clustered tasks.

Fast-loop commands:

```powershell
python scripts/collect_candidate_experience.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --output build/stage5-v4-train-collection `
  --split train `
  --candidate-generator-profile core5 `
  --candidate-replan-order-seeds 0,1 `
  --candidate-rollout-horizons 10,25 `
  --layout-modes compartmentalized,dead_end_aisles `
  --task-variants balanced_dense,balanced_clustered `
  --workers 4

python scripts/build_rollout_candidate_experience.py `
  --dataset build/feasibility-dataset `
  --collection build/stage5-v4-train-collection `
  --output build/stage5-v4-train-experience `
  --split train

python scripts/collect_candidate_experience.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --output build/stage5-v4-validation-collection `
  --split validation `
  --candidate-generator-profile core5 `
  --candidate-replan-order-seeds 0,1 `
  --candidate-rollout-horizons 10,25 `
  --layout-modes compartmentalized,dead_end_aisles `
  --task-variants balanced_dense,balanced_clustered `
  --workers 4

python scripts/build_rollout_candidate_experience.py `
  --dataset build/feasibility-dataset `
  --collection build/stage5-v4-validation-collection `
  --output build/stage5-v4-validation-experience `
  --split validation

python scripts/train_candidate_ranker.py `
  --memory build/stage5-v4-train-experience `
  --output build/stage5-v4-ranker `
  --feature-profile rollout22 `
  --models pairwise_linear,sklearn_logistic,sklearn_forest,sklearn_gbdt

python scripts/evaluate_candidate_ranker.py `
  --ranker build/stage5-v4-ranker `
  --queries build/stage5-v4-validation-experience `
  --output build/stage5-v4-validation `
  --conservative-gates

python scripts/run_stage5_v4_experiment.py `
  --dataset build/feasibility-dataset `
  --solver build/windows/Release/lns2_cli.exe `
  --rollout-ranker build/stage5-v4-ranker `
  --rollout-config build/stage5-v4-validation/selected_config.json `
  --candidate-generator-profile core5 `
  --output build/stage5-v4-test `
  --split test
```

For final reproduction, collect both Train and Validation with
`--candidate-generator-profile full8`, `--candidate-replan-order-seeds 0,1,2`,
and `--candidate-rollout-horizons 10,25,50`. The v4 test runner compares only
controlled baseline and rollout-guided by default; pass optional `--knn-*` and
`--v3-*` arguments only when those secondary arms are needed.

V4 has been smoke-tested on a small dataset. The full `10,25,50` rollout
collection is expected to be substantially more expensive than V2.2 collection
and should be treated as generated build output.
