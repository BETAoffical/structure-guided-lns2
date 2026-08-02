# STRIDE-LNS Stage 4R diagnostic protocol

Stage 4R is an exploratory but registered diagnosis after the Stage 4 quality
model missed its primary improvement gate. It cannot promote a controller and
cannot read formal OOD or test data.

## Questions

1. How often does the quality label actually disagree with the old V2 control
   relation?
2. How often does the `0.02` post-structure penalty change the reduction-ratio
   ranking or state winner?
3. How much quality-Oracle opportunity exists beyond the control selection, and
   how much of it did the OOF quality model capture?
4. Are exact winners, Top-3 membership, and candidate ranks stable between PP
   seed halves `0..3` and `4..7`?
5. Is the bottleneck weak label change, noisy labels, or model learnability?

The diagnostic uses only the frozen Stage 3 trial cohort, candidate aggregates,
Stage 4 fold manifest, and OOF predictions. Runtime and generated-node values
remain excluded from label analysis.

## Diagnostic-only runtime boundary

If both final all-development models can be exported with exact portable
selection equivalence, they may be evaluated as `diagnostic_only`:

- frozen `v2-full` remains the deployment anchor;
- `stride-control-v1` tests the effect of expanded data under the old label;
- `stride-quality-v1` tests the incremental new-label effect.

Shadow must run before any action-changing Quick experiment. Shadow records
selection disagreement, inference overhead, feature-range fallback, invalid
actions, and state/fingerprint equivalence while leaving the frozen V2 action
unchanged.

## TTF-first Quick semantics

The exact Quick cohort and wall budget must be committed before execution. It
must be map-disjoint from the 36 Stage 4 training maps and from the formal OOD
set. Controllers use identical instances, solver seeds, budgets, worker count,
and balanced execution order.

Primary metric: mean capped wall TTF. Failures are charged the registered wall
budget. A candidate cannot be called faster if it solves fewer episodes than
`v2-full`. Required secondary reports are:

- common-success paired wall TTF;
- success/failure count;
- repair rounds to feasibility;
- fixed-step and wall-clock conflict AUC;
- PP+SIPPS time;
- proposal, feature, inference, state-export, and controller overhead;
- action validity, fallback, and semantic mismatch counts.

A formal speed claim still requires Stage 5/6 evidence. A Stage 4R Quick result
only decides whether to revise the label, retain a data-only control candidate,
or resume the approved pipeline.

## Registered Quick cohort

The diagnostic Quick is frozen in `configs/stride_stage4r_quick.json` before
any of its controller episodes run:

- 11 tasks from 11 maps, with one seed-1 task per map and initial conflict
  counts from 5 to 55.
- Eight generated maps and three MovingAI development maps, covering
  compartmentalized, dead-end, regular-beltway, game, and maze layouts.
- Zero map overlap with the 36 Stage 4 training maps and the 36-map formal OOD
  cohort. Formal OOD outcomes remain unread.
- `v2-full`, `stride-control-v1`, and `stride-quality-v1` use the same task,
  solver seed, 300-second wall budget, 100-step metric window, candidate pool,
  PP+SIPPS repair, and audit runtime.
- Episodes execute strictly one at a time. Controller order rotates by task so
  each controller appears in every within-task position and the first-position
  counts differ by at most one.
- Task selection used only map/layout coverage and previously recorded initial
  conflicts. No controller outcome was read during selection.

TTF uses `lns2.ttf.reset_inclusive_wall.v1`: its clock and solve budget start
immediately before `env.reset()`, and the clock stops when reset or repair first
returns a feasible state. Environment construction, final trace validation,
and report generation are recorded separately and excluded from TTF.
