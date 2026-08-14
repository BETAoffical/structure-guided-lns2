# Native-Order TransactionalRepair v1 Execution Amendment r2

The first extension stopped exactly as registered when one 600-agent
`preserved_prefix_blocker_tail_upper_bound` job exceeded the 300-second process
limit. Its paired first attempt normally consumed about 140--152 seconds and a
second PP attempt consumed another 116 seconds or more. The process-level
watchdog had no way to pass its remaining time into PP, so it terminated the
worker instead of receiving a native `time_limit` rollback.

The scientific policies, candidate sets, blocker cap, seeds, trial indices,
outcomes and gates do not change. The external 300-second job limit is not
increased. Revision r2 adds a 270-second PP-transaction budget and reserves 30
seconds for state restoration, validation, serialization and worker shutdown.
Before every attempt, the runner passes the real-time remaining transaction
budget to native `step_with_time_limit`. Native `time_limit` with exact rollback
is valid right censoring; a process timeout remains a terminal execution error.

The completed initial phase is reused by registered identity. All 1,440
artifacts passed integrity with zero errors/timeouts, and the maximum recorded
total attempt wall time was 248.315059025 seconds, below the new 270-second
budget. Therefore the added deadline would not have censored any initial
outcome. The failed extension is preserved and is not imported. Trials 8--15
are rerun completely under a fresh r2 run fingerprint.

The exact previously timed-out state/trial is used only for an excluded
four-policy qualification. Qualification must return either a completed native
attempt or a native `time_limit` artifact within the 300-second process limit,
while retaining exact rollback, paired seeds and native order for both
deployable arms.

This remains a mechanism-discovery experiment. It does not authorize runtime
integration, TTF evaluation, model training, controller promotion, candidate-
pool replacement or a long-tail claim.
