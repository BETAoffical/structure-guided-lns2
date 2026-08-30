# Seed25 Same-Set PP vs GCBS One-Step Screen

## Outcome

The screen completed all eight isolated jobs with zero process or integrity
errors.  It restored four frozen seed25 repair states and applied the exact
agent set that Dual16 had selected at that state.  Each state was repaired once
with PP and once with GCBS; no candidate was generated, no controller selected
an action, no continuation was run, and no TTF was measured.

| State | Exact selected action | PP reduction / replan s | GCBS reduction / replan s |
|---|---|---:|---:|
| Maze32 N300, d31 | Component16 | 0 / 0.078 | 41 / 5.283 |
| Random-high, d48 | observed Dual Target8 | 0 / 0.011 | 2 / 0.003 |
| Room, d30 | Component16 | 0 / 0.779 | 7 / 2.977 |
| Warehouse, d0 | Component16 | 59 / 1.041 | 70 / 0.264 |

PP produced a strict conflict decrease in 1/4 states and exact rollback/no-op
in 3/4.  GCBS produced a strict decrease in 4/4.  Mean normalized conflict
reduction was 0.0753 for PP and 0.1362 for GCBS.  Mean native replan time was
0.477 seconds for PP and 2.132 seconds for GCBS.

## Interpretation

This rejects the explanation that recent Dual16 failures are purely ranking
failures.  On Maze and Room, the selected Component16 set was useful to a
stronger repairer even though PP returned it unchanged.  Warehouse was a
positive control: the same structural set was effective under both repairers.

It does not remove the ranking problem.  On the Random checkpoint, GCBS rescued
the selected Target8 set by only two conflicts, whereas the previously replayed
Hotspot16 set reduced eleven conflicts with PP.  The evidence therefore points
to two simultaneous bottlenecks: frozen selection can choose the wrong set, and
PP can fail on a set that another repairer can exploit.

## Qualification boundary

This is an outcome-enriched one-step mechanism diagnostic, not a runtime or TTF
comparison.  GCBS does not currently honor the cooperative action deadline used
by PP; the Maze action exceeded its configured five-second environment budget
and required an external process fuse.  GCBS failure-time rollback has not been
qualified.  PBS was excluded after its first restored-state explicit-neighborhood
smoke terminated with a native core dump.  The algorithms also use different
acceptance rules (PP accepts non-increase, GCBS requires strict decrease).

Consequently no repairer, selector, or controller is promoted.  The next valid
step is native GCBS deadline and exact-failure-rollback qualification, followed
by a small multi-seed same-state screen.  A long TTF run is not yet justified.

## Artifacts

- Config: `configs/stride_seed25_same_set_pp_gcbs_screen_v1.json`
- Runner: `experiments/stride_seed25_same_set_pp_gcbs_screen.py`
- Report: `build/stride-seed25-same-set-pp-gcbs-screen-v1/same_set_repairer_report.json`
- Report SHA-256: `782c0dea27724f28164ea225774452bf4a4486a1a1568dd344ba07faa6ffa7f2`
- Validation: four targeted unit/retention tests passed; the native screen
  completed 8/8 branches with zero errors.
