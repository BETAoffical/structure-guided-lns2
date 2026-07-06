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
