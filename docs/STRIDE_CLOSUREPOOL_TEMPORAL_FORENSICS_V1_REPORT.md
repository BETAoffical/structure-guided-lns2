# ClosurePool Temporal Forensics v1 Report

## Result

The preregistered read-only supplement completed with every integrity check
passing. It reconstructed ten historical Maze first-divergence comparisons
over five paired task/solver keys, plus the two separately held known
regressions. No solver was run and the known regressions were excluded from
all historical contrasts.

The tested temporal-closure definition is rejected. It is both too dense to be
a useful LNS neighborhood and too inconsistent to identify harmful actions.
No ClosurePool candidate rule, model, runtime controller, or TTF experiment is
authorized from this result.

Artifacts:

- report SHA-256:
  `3d6309211bad566499578eddf675eb40adab3b4d6d560ab884947f0a17c3cc7d`;
- temporal comparison SHA-256:
  `c7ff79519a6c92dd26d5480c8ef6ba2f39e28d025a8c68a26ee4b42f26e33908`;
- persistent-pair location SHA-256:
  `56a5cfa6890fd02c90a34453b69fb9f6326a3dececfe50f523d644c1a1d502da`.

## Cohort

- historical adverse or severe: 3 comparisons;
- historical beneficial or tied: 7 comparisons;
- controllers: SlotPool and StructPool;
- tasks: Maze task seeds 233 and 277;
- solver keys: five historical keys;
- known regression: task seed 233, solver seed 3, reported separately;
- exact persistent-pair locations reconstructed: 1,095 rows.

## Why temporal closure failed

The static Maze decomposition contained 1,462 low-degree segments, with a
maximum segment length of 63 cells. Although individual segments were bounded,
the interval-overlap dependency edges chained them through agents into one
touched dependency component in every historical comparison. A 24- or
32-agent structural action covered only about 24.5%–34.4% of that component,
which implies a full dependency closure of approximately 93–98 of the 100
agents. Such a closure is effectively global replanning rather than a useful
LNS neighborhood.

The preregistered pre-action contrasts were also inconsistent. Across all
historical cases, adverse/severe minus beneficial/tied challenger-minus-V2
means were:

- internal opposing dependency edges: `+5.95`;
- newly created opposing dependency edges after PP: `+4.29`;
- boundary dependency ratio: `-0.054`;
- outside boundary queue agents: `-5.76`.

The apparent opposing-edge signal came only from SlotPool:

| Controller | Internal opposing contrast | New opposing contrast |
| --- | ---: | ---: |
| SlotPool | +10.50 | +14.83 |
| StructPool | -3.50 | -9.25 |

The StructPool historical severe tail therefore moved in the opposite
direction. Beneficial StructPool cases also overlapped or exceeded the severe
case on the same counts. In the known regressions, both challengers had fewer
internal opposing dependencies than V2 (`-6` and `-7`), despite requiring 61
and 301 repairs. Thus neither a boundary-cut metric nor an opposing-dependency
count is a reliable pre-action hazard rule.

## Mechanistic interpretation

The earlier location analysis remains valid: severe runs contain conflict
pairs that repeatedly reappear at low-degree cells, sometimes across the
selected-neighborhood boundary and sometimes with both agents selected. This
supplement shows why that observation is not yet a candidate-selection rule.
The harmful event is an interaction between the chosen neighborhood and the
specific PP realization. Static membership and current-path temporal overlap
alone do not determine whether PP will create a stable ordering or recreate a
bottleneck conflict.

Introducing an arbitrary spatial or temporal cutoff now would make the graph
sparser, but would reintroduce the hand-tuned rule that ClosurePool was meant
to avoid. The current data provide no registered basis for selecting such a
cutoff.

## Decision and next admissible study

Stop the current ClosurePool implementation path before candidate generation,
runtime integration, or TTF evaluation. The evidence gate for freezing an
interpretable dependency-closure rule did not pass.

The next admissible study should isolate PP action uncertainty rather than add
another static closure heuristic. For the exact same candidate neighborhood,
run paired one-step repairs over multiple PP seeds and measure the probability
of producing residual bottleneck concentration, new opposing dependencies,
and no conflict reduction. Only if that robust one-step hazard is repeatable
across maps should it be used as a safety target. This would remain a
one-step, action-specific analysis; it must not use remaining repair rounds,
Cost-to-Go, or future TTF as a label.
