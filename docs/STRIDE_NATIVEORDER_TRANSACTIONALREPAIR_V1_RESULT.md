# STRIDE Native-Order TransactionalRepair v1 Result

## Outcome

The registered initial phase and the r2 extension jointly contain all 45
states, 16 paired trials and four policies: 2,880 atomic policy artifacts.
Collection and analysis passed every identity, pairing, native-order, rollback,
attempt-cap, deadline and completeness check with zero execution errors and
zero process timeouts.

The first extension stopped on one process timeout and remains preserved but
unused. Revision r2 did not increase the 300-second process limit. It passed a
270-second real-time PP-transaction budget to native `step_with_time_limit`
and reran trials 8--15 completely. The registered initial 1,440 artifacts were
reused because their maximum attempt wall time, 248.315059025 seconds, was
below the new budget. No r2 job reached native `time_limit`; the cooperative
deadline nevertheless removed the uncontrolled process-timeout path.

## Aggregate mechanism outcomes

| Policy | Returned unchanged | Native success | Strict conflict reduction | Mean normalized reduction | Mean attempts | Mean attempt wall* |
|---|---:|---:|---:|---:|---:|---:|
| selected single attempt | 64.17% | 35.83% | 32.64% | 0.06765 | 1.000 | 18.31 s |
| same set + fresh native retry | **45.83%** | **54.17%** | **50.00%** | 0.09636 | 1.642 | 27.23 s |
| blockers + fresh native retry | **42.50%** | **57.50%** | **52.78%** | 0.11214 | 1.642 | 28.22 s |
| preserved-prefix blocker-tail upper bound | 34.31% | 65.69% | 61.81% | 0.13380 | 1.642 | 28.55 s |

`*` These wall measurements came from a 16-worker mechanism collection. They
include real PP attempts but are not isolated raw TTF measurements.

Both deployable native-order arms improved all three maps. Their state-cluster
paired bootstrap risk differences against the single attempt were:

- same-set native retry: -0.1833, 95% CI [-0.2236, -0.1444];
- blocker-augmented native retry: -0.2167, 95% CI [-0.2597, -0.1764];
- controlled-order upper bound: -0.2986, 95% CI [-0.3625, -0.2361].

The blocker-augmented versus same-set difference was -0.0333, but its 95% CI
[-0.0875, 0.0167] crossed zero. The observed 3.33-point advantage is therefore
not stable evidence that adding failed-attempt blockers contributes beyond a
fresh native shuffle. The preregistered mechanism selection is the simpler
`same_set_native_retry`.

## Interpretation

The dominant recoverable mechanism in this discovery cohort is PP order or
low-level randomness, not a proven candidate-membership defect. The exact same
neighborhood succeeds materially more often when native PP receives one fresh
paired seed after an exact `conflict_bound_exceeded` rollback. This retry does
not scan HybridStructPool, run the ranker again, or control the repair order.

The controlled-order diagnostic still has substantial additional headroom,
but official runtime PP order is not known in advance and must not be replaced
with an outcome-derived order. The blocker arm remains scientifically
interesting but is not preferred without independent evidence separating its
small point gain from retry randomness.

This is not yet a fast-solver result. Same-set retry raised concurrent mean
attempt wall from 18.31 to 27.23 seconds, and the 600-agent
`maze-128-128-2` group rose from 64.99 to 96.05 seconds. The mechanism trades
extra PP work for a lower probability of returning unchanged. Whether that
trade shortens platform duration or raw TTF must be tested later under isolated
paired timing, after independent result-blind mechanism confirmation.

## Evidence boundary and next step

The 45 states were selected for prior repair difficulty. Passing permits only
an independently preregistered result-blind confirmation of:

1. one native PP attempt;
2. after exact `conflict_bound_exceeded` rollback only, one same-set retry with
   a fresh paired native PP seed;
3. no retry after native `time_limit`;
4. the same 270/300-second internal/external limits;
5. success, returned-unchanged, conflict reduction and total PP cost reported
   together.

No result here authorizes candidate-pool replacement, model training, runtime
integration, TTF evaluation, controller promotion, or a long-tail claim.

## Registered artifacts

- final report SHA-256:
  `377197A57910A11AE8DCC14396F8752B67EC9602045195A36C1025D74C0DC2A6`
- r2 run configuration SHA-256:
  `075D554C236C0AF5C18CEC749BFCB16C494B8B4E027DA3703B64D17D26E7DE72`
- r2 collection status SHA-256:
  `4DCE3B3E561FA2E78A9594E2023FE1B1845B8620E8306177F35C4491A820B536`
