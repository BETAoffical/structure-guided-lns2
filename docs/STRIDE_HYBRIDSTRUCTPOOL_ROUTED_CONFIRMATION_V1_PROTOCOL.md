# STRIDE HybridStructPool routed confirmation v1

The development source ablation froze `structshell_only` before this protocol.
The confirmation makes no further source, family, size, gate, ranker, or seed-policy
change.

The result-blind cohort contains 60 paired task/solver-seed keys from ten maps
that did not occur in the eight-key development ablation: three Maze maps, one
Room map, two Warehouse maps, two MovingAI game maps, and two additional DAO
maps.  Each map contributes two tasks and solver seeds 1, 2, and 3.  Task
selection uses only the previously materialized dataset identities; no solver
outcome may remove or replace a key.

Official Adaptive, frozen V2, and frozen StructShell-only run in a rotating,
strictly serial order.  All use native PP and the episode-stream seed policy.
Qualification and non-timed tests may use 16 processes; the 180 timed episodes
use one process.  Raw TTF is reset-inclusive and run-to-completion.

Promotion requires complete paired identity, no execution error, process timeout,
invalid action, semantic mismatch, or capped TTF; success no lower than V2; mean
raw TTF lower than V2; at least half the pairs faster; a positive lower endpoint
of the 95% paired bootstrap interval; no map worse by more than 5%; and no higher
P95 or maximum repair-iteration tail.  Failure keeps V2 as the default and ends
Hybrid runtime tuning.  Passing freezes StructShell-only as the runtime pool but
does not merge the later retry/rescue mechanism into this comparison.
