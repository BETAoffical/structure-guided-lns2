# STRIDE HistoryRank Rule V1 Report

## Status

`stride-historyrank-rule-v1` completed its preregistered offline checkpoint-local audit and **did not pass** the promotion gates. It is not a runtime controller, was not trained, did not read TTF/runtime/future trajectories, and makes no long-term or solver-speed claim.

## Frozen rule

At each of the 45 registered `first_repeat_stall` checkpoints, the audit excluded only the exact neighborhood selected in the two immediately preceding path-level PP no-ops. It then selected the highest frozen V2 score among the remaining runtime-retained candidates, using candidate ID as the deterministic tie-break.

The implementation needed three schema-alignment corrections before any result was produced: use `completed_state_count`, join root checkpoints to logical results by registered `case_id`, and ignore non-retained drafts that have no runtime V2 score. These corrections did not alter the registered cohort, quality gates, candidate outcomes, or fallback ordering.

## Results

- Complete cohort: `45/45`; execution errors: `0`.
- Stable improvement: `21/45 = 46.67%`, below the `60%` gate.
- Both fixed 8-seed halves positive: `23/45 = 51.11%`, below the `60%` gate.
- Mean one-step seed improvement: `+0.05905`, above the `+0.02` gate.
- Mean no-progress-rate delta: `-0.18611`, passing the non-increase gate.
- Mean normalized regret decreased from `0.20805` to `0.14900`.
- Every map had positive mean seed improvement, but no map reached a majority stable-improvement rate:
  - `maze-128-128-1`: `8/16 = 50.00%`, mean improvement `+0.05937`.
  - `maze-128-128-2`: `5/12 = 41.67%`, mean improvement `+0.04684`.
  - `maze-32-32-4`: `8/17 = 47.06%`, mean improvement `+0.06737`.

## Conclusion

The exact-no-op signal is useful but insufficient as a standalone rule. Deterministically taking the next V2-ranked retained candidate improves the average checkpoint and lowers no-progress probability, yet it is not reliable across PP seeds. Therefore this rule must not be installed as a controller or used for TTF evaluation.

The frozen root-cause decision remains: candidate generation is not the immediate bottleneck at these checkpoints; selection requires history-aware features that distinguish which alternative is robust after repeated no-op repair. The next safe step is a separately preregistered `stride-historyrank-v1` feature audit over the same full frozen candidate pool. That audit must remain outcome-blind during feature construction and may not train or modify the solver until feature readiness is established.

## Artifact identity

- Configuration SHA-256: `4531c039517d49dbef04474bf3bd9b4cd4e254159e6a500269ce537b13ed81b3`
- Row manifest SHA-256: `9e070b79c02ae79b2d80f4f882eefc7854c19df7bfcaedec25709d1dda0c05f7`
- Report SHA-256: `375ef66a6e34e7edcc34e578429f15634d286a4fc83b9c4e982757c0b6957157`
- Status SHA-256: `49c2f1ae00e9c5cdb1ca2c22bb7b3d5d81d694047dd8c96ade61c021c9e8ae84`
