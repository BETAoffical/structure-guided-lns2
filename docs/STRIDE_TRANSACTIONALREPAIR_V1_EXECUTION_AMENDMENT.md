# STRIDE TransactionalRepair v1 Execution Amendment

## Trigger

The first formal collection used one `state x trial_index` process to run all
three frozen policies serially under a shared 300 second process limit. It was
stopped after 128 completed jobs because two jobs reached that limit. Both were
trial indices 7 and 15 of the same registered 600-agent
`maze-128-128-2` state.

The 300 second limit therefore measured the sum of three policies with maximum
attempt counts 1, 2, and 3. It did not establish that any one policy exceeded
the registered bound. The process wrapper could not attribute the timeout to a
specific policy and created unnecessarily coarse checkpoints.

No policy outcomes or aggregate scientific metrics from the interrupted run
were inspected before this amendment. The 126 completed artifacts and stale
progress record remain preserved under
`build/stride-transactionalrepair-v1`; they will not be imported, selected, or
mixed with the amended run.

## Frozen scientific semantics

The cohort, all three policies, PP seeds, repair attempts, blocker cap and
placement, order retry, outcome definitions, readiness gates, and claim
boundaries remain unchanged.

Only execution packaging changes:

- task granularity becomes `state x trial_index x policy`;
- the same 300 second hard limit applies independently to each policy job;
- each policy job writes one atomic artifact;
- analysis reconstructs the three-policy pairing and verifies byte-level
  equality of their first-attempt scientific signatures;
- the amended run starts in the fresh output
  `build/stride-transactionalrepair-v1-r2`;
- all 2,160 policy jobs are scheduled outcome-blind through the same global
  16-worker queue, including the final tail.

This amendment does not lengthen the limit and does not exclude the two prior
timeout trials. If any single policy job reaches 300 seconds, the zero-timeout
gate fails and the mechanism audit stops.
