# STRIDE GuardPool v1 preregistration

`stride-guardpool-v1` is the first runtime treatment unlocked by the frozen
SlotPool development and six-map confirmation passes. It keeps the exact
`v2-full` base candidate generator, V2 ranker, PP, and SIPPS implementation.

When the existing high-stress gate is active, the complete exact-deduplicated
8/16/24/32 StructPool grid is generated. The frozen SlotPool pairwise model
retains at most six structural challengers. V2 then ranks the unchanged base
pool plus those challengers. SlotPool is therefore a candidate-budget model,
not the final repair-action ranker.

The no-progress guard is frozen before reading the held-out Maze regression.
On 29 historical V2 episodes containing 402 decisions, limits 5, 8, and 12
would first trigger in 5, 3, and 1 episodes. Their decision-level false-trigger
rates are 1.244%, 0.746%, and 0.249%. The lowest limit at or below 1% is therefore
eight consecutive non-decreasing-conflict repairs.

At that limit all structural challengers are suspended and the controller
uses the exact V2 pool until a strict conflict decrease resets the streak. No
remaining-time or wall-limit condition is used. The last structural action is
recorded under the current conflict signature for audit/tabu purposes.

Only after runtime semantics tests pass may the held-out
`maze-128-128-1`, opposite-exchange task seed 233, 100-agent, solver-seed-3
regression run. The fixed requirement is success within 30 repairs (twice the
15-repair V2 reference). Passing that known regression permits paired
development raw-TTF evaluation; it is not independent generalization evidence.
