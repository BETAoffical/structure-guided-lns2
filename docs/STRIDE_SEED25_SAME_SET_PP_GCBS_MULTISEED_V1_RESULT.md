# Seed25 Same-Set PP vs GCBS Multi-Repeat Screen

## Outcome

The registered one-step screen completed 32/32 attempts (four frozen states,
four paired execution salts, PP and GCBS) with zero process, identity, invalid
state, or rollback-integrity errors.  The timed screen itself took 90.24
seconds.  It did not generate candidates, run a controller continuation, or
measure TTF.

The preregistered decision is **stop_same_set_gcbs_repairer_branch**.  GCBS was
stronger on average, but it was not reliable enough to serve as a generic
five-second fallback.

| Frozen state and exact action | PP strict drops | GCBS strict drops | PP mean reduction | GCBS mean reduction | GCBS time limits |
|---|---:|---:|---:|---:|---:|
| Maze32 N300 d31, Component16 | 0/4 | 0/4 | 0.00 | 0.00 | 4 |
| Random-high d48, Dual-selected Target8 | 0/4 | 2/4 | 0.00 | 0.50 | 0 |
| Room d30, Component16 | 0/4 | 3/4 | 0.00 | 6.50 | 1 |
| Warehouse d0, Component16 | 4/4 | 4/4 | 61.00 | 74.25 | 0 |
| **Overall** | **4/16** | **9/16** | **15.25** | **20.31** | **5** |

Mean normalized conflict reduction was 0.07781 for PP and 0.11775 for GCBS.
Mean native replan time was 0.455 seconds for PP and 2.301 seconds for GCBS.
All five GCBS deadline outcomes atomically restored the original repair state.

## Mechanism interpretation

- **Room is a real low-level repairer gap.**  PP failed on the selected
  Component16 set in all four repeats, while GCBS reduced conflicts in three.
- **Warehouse is a good-action positive control.**  Both repairers succeeded
  in all four repeats; GCBS produced the larger mean reduction.
- **Random remains mainly a selection/candidate problem.**  GCBS rescued the
  Dual-selected Target8 set only twice and by one conflict each.  The earlier
  same-state replay found a different Hotspot16 action that reduced conflicts
  much more under PP.
- **Maze is not fixed by changing only the low-level repairer.**  PP returned
  no progress and GCBS exhausted the five-second action budget in all four
  repeats.  This exact Component16 set is unsuitable or too hard under the
  registered budget.

This therefore rejects a universal `same selected set -> GCBS fallback`
policy.  It does not reject GCBS as a state-conditional repairer, and it does
not erase the separate ranking problem.  The observed failure modes are
state-dependent: wrong neighborhood choice on Random, PP repairability on
Room, and an action/complexity gap on Maze.

## Qualification and claim boundary

Before the screen, the native GCBS path was qualified for cooperative action
deadlines, safe root failure, atomic timeout/failure rollback, valid successful
commit, and post-timeout path-table continuity.  The qualification report SHA
is `dcdd15cf48335fc6e3d724fc4bf40ca0de419be9446a948a1c9476f55e10ef48`.

The four GCBS repeats are paired execution/repeatability salts, not four
independent GCBS random seeds.  The actions were selected from known diagnostic
states, so the result estimates neither natural occurrence rate nor map-level
generalization.  No selector, repairer, controller, or default runtime is
promoted, and the failed gate forbids continuation or TTF runs for this branch.

## Artifacts

- Config: `configs/stride_seed25_same_set_pp_gcbs_multiseed_v1.json`
- Runner: `experiments/stride_seed25_same_set_pp_gcbs_multiseed.py`
- Report: `build/stride-seed25-same-set-pp-gcbs-multiseed-v1/same_set_multiseed_report.json`
- Report SHA-256: `8774171d8e4bb53d2f827b957d754a4d550f284e4037bad930848219216d9d16`
- Plan SHA-256: `5b10a573ce756d5c0a848f18234801833c71a73e8eab100d7a319e25a1b01c5d`
- Native module SHA-256: `1231dbc0bbca39cd5d1d2aba49bc68f8c16889ff22e16d5d7eb9afe49ec09221`

