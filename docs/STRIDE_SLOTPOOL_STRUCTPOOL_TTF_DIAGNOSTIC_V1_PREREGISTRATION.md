# SlotPool versus Full StructPool Raw-TTF Diagnostic v1

## Purpose

This diagnostic answers whether the frozen SlotPool six-challenger reduction is
faster end to end than the original full StructPool six-candidate rule when the
single already-known catastrophic Maze key is excluded as explicitly requested.

## Frozen scope

- controllers: `v2-plus-structpool` and `v2-plus-slotpool`;
- five fixed maps, ten fixed tasks and solver seeds 1--3;
- one and only one excluded paired key:
  `maze-128-128-1`, derived task seed 233, solver seed 3, 100 agents;
- 29 paired keys and 58 run-to-completion episodes;
- one worker, deterministic PP replay and strict alternating controller order;
- no scientific TTF cap and no result-based deletion after execution starts.

The primary metric is mean reset-inclusive raw wall TTF. Success count, median
TTF, paired faster fraction, repair iterations, PP time, neighborhood-selection
time and per-map results are reported together.

## Claim boundary

The exclusion is based on a previously observed outcome, so this is a focused
diagnostic and cannot establish formal speed, fresh-map generalization or a new
default. Any additional long tail discovered by this run remains in the result.
