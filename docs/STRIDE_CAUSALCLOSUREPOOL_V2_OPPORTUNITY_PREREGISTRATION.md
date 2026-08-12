# STRIDE CausalClosurePool v2 opportunity preregistration

The compact event-time-slice generator is frozen at commit `aa59185`.  This
audit tests whether its 930 outcome-blind candidates add robust one-step native
PP opportunities beyond the complete 1,367-candidate frozen V2 base pool.

- Cohort: all 78 frozen development states; no outcome-based exclusion.
- Trials: indices 0--15 for every candidate, with the same PP seed for every
  action at one state and trial index.
- Execution: 16 workers, one atomic state/candidate/trial artifact, 300-second
  hard limit per job, stop on the first execution error or timeout.
- Preflight: the first two identity-sorted states and trial indices 0--1.
- Stable dominance: seed-mean advantage at least 0.02, no increase in
  no-progress rate, and improvement in both fixed eight-seed halves.
- Gates: at least 10% of states contain a new action robustly better than the
  best V2 base action, and at least 10% contain a stable frontier addition.

The ranker and repair order are not controlled or evaluated.  Runtime, TTF and
future trajectories are excluded.  Passing permits only a separately frozen
continuation confirmation; it is not a solver-speed or long-tail claim.
