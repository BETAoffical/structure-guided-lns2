# ClosurePool Long-Tail Forensics v1 Report

## Result

The registered read-only reconstruction completed with all integrity checks
passing. It analyzed 38 historical first divergences plus two known Maze
regressions. The historical outcomes were 33 beneficial, two adverse, two
severe tails, and one tie; both known regressions were severe tails.

The analysis supports two distinct failure mechanisms:

1. a structural neighborhood can cut a conflict dependency chain, leaving or
   recreating selected-to-unselected conflicts after PP;
2. even when both agents are selected, PP can repeatedly recreate an internal
   timing conflict at the same low-degree bottleneck.

It does **not** support freezing a ClosurePool rule from the current static
features. Historical adverse SlotPool cases had much larger post-repair
boundary-conflict ratio, new-pair fraction, and persistent-pair counts than
beneficial controls, but their pre-action boundary and component-coverage
statistics overlapped substantially with beneficial Maze cases. StructPool
had only one historical adverse case, which is insufficient for a general
rule.

## Integrity and scope

- 58 historical controller comparisons and 29 paired keys were retained;
- 38 historical comparisons had a genuine first divergence;
- two known-regression comparisons were analyzed separately;
- divergence states, trace hashes, realized actions, fingerprints, conflict
  trajectories, and paired PP seeds matched;
- the known regression was excluded from mechanism contrasts;
- no model, candidate rule, runtime controller, or TTF claim was produced.

Artifacts:

- report SHA-256:
  `a403711973f84659c9d78f6b342d2e288d5c967d3bea8766dbd733a76e28668a`;
- comparison diagnostics SHA-256:
  `534e40edfee81b11b1fea4acc964652aff41de6a67305614a52bba8893289e8d`;
- persistent-pair diagnostics SHA-256:
  `1459b8ff5f39cb70421f003e461b15181347b4569d57a6cc4cd96e0552561668`;
- agent diagnostics SHA-256:
  `4e2c84ec42e9900be2b28eac78ade4f9ab9d905b2dbed99229a120d525b2e084`.

## Historical adverse cases

| Controller | Task / solver seed | V2 repairs | Challenger repairs | Main observation |
| --- | --- | ---: | ---: | --- |
| SlotPool | Maze seed 233 / 1 | 34 | 40 | 70 persistent pairs, including 40 new pairs |
| SlotPool | Maze seed 277 / 3 | 18 | 46 | 77 persistent pairs and 60 new persistent pairs |
| SlotPool | Room seed 4 / 2 | 1 | 3 | small two-round regression, not the severe Maze mechanism |
| StructPool | Maze seed 233 / 2 | 18 | 42 | a pair persisted for 40 consecutive states |

For historical SlotPool adverse/severe cases versus beneficial/tied controls,
the challenger-minus-V2 contrast increased by about 0.380 for post-repair
boundary-pair ratio, 0.160 for new-pair fraction, 30.8 new persistent pairs,
and 24.7 maximum agent conflict states. By contrast, the corresponding
pre-action boundary-pair ratio increased by only 0.045 and minimum component
coverage decreased by only 0.026.

## Known Maze regression

The held-out diagnostic key was
`maze-128-128-1`, task seed 233, 100 agents, solver seed 3.

- V2 completed in 15 repairs.
- SlotPool required 61 repairs. Its dominant internal pair remained for 59
  consecutive states at one bottleneck cell.
- StructPool required 301 repairs. Its dominant internal pair remained for 264
  consecutive states, while several new selected-to-unselected and internal
  pairs persisted for more than 200 states.

These observations are illustrative regression evidence only. They were not
used to choose a rule or threshold.

## Decision

Do not implement a boundary-only or conflict-component-only ClosurePool.
Before a candidate rule can be frozen, run a separately preregistered Maze
supplement based on map-defined low-degree corridor segments and temporal path
overlap. It must test whether selected-to-unselected corridor dependencies and
opposing-direction queue relations separate historical adverse cases from all
historical beneficial Maze controls. The known regression remains excluded
from rule selection.
