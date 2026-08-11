# STRIDE MultiValue productive-recovery amendment

## Reason

The initial collection showed that the registered 1,800-second state window
correctly bounded each worker process, but every productive timeout consumed the
same four-attempt budget as a genuine execution failure.  Four heavy states made
new candidate-level checkpoints in every window yet could not finish their
finite episode matrices within four windows.

This is an execution-orchestration defect, not a label, candidate, teacher,
state, seed, or outcome change.  The collection was stopped with SIGTERM after
candidate-level atomic checkpoints were written.  No completed episode is
discarded or selected by its outcome.

## Frozen scientific semantics

- State manifest, candidates, candidate order, 124-dimensional features,
  teachers, paired PP seeds, horizons, and labels are unchanged.
- Every episode still stops at success, 128 repair decisions, or 300 seconds.
- The process safety timeout remains 360 seconds.
- A state worker window remains 1,800 seconds.
- Result-based state or candidate filtering remains forbidden.

## Corrected recovery semantics

- A timed-out state window that increases its candidate-level episode checkpoint
  is a `productive_window`.  It resets the no-progress streak and does not
  consume the failure-recovery budget.
- A timed-out window with zero new episode checkpoints is a
  `no_progress_window`; it consumes one failure recovery.
- An execution error always consumes one failure recovery, even if earlier
  episodes in the window were checkpointed.
- Two consecutive no-progress windows stop the state for diagnosis.
- Four genuine failed recoveries stop the state.
- Productive windows continue until the finite registered episode matrix is
  complete.  This cannot run without bound because every episode has a
  360-second process fuse and the episode matrix is finite.

## Compatible recovery identity

Only artifacts with the original run fingerprint
`a90bbdb7fb46552e833f3e1e4917bc0aa11ae75bbaf1d3530365029dabc0f05f`
and the unchanged state-record hash are accepted for recovery.  Newly written
partial artifacts use the amended run identity.  The source run fingerprint is
recorded in collection status.
