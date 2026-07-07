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
