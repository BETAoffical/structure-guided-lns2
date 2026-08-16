# STRIDE StructShell versus V2 paired confirmation v1

This experiment isolates the runtime-pool decision from the independent
official Adaptive comparison. The only timed arms are frozen V2 and frozen
`structshell_only`; neither implementation, feature, ranker, PP, nor seed
policy changes.

The ten-map cohort is the fully eligible cohort preregistered by routed
confirmation v2, but every timed episode is new: solver seeds 4, 5, and 6
replace the observed seeds 1, 2, and 3. No timed artifact from either previous
confirmation is imported. Every map must first pass six fresh qualification
resets. Failure stops before formal timing and cannot trigger map replacement.

The formal schedule contains 60 paired keys and 120 episodes. V2 and
StructShell alternate first position and run strictly serially. Raw TTF is
reset-inclusive and uncapped; a uniform 900-second process fuse remains a
terminal error rather than a censored or winning outcome.

Promotion still requires complete identity and zero runtime errors; success
noninferiority; lower mean raw TTF; paired faster fraction of at least 50%; a
strictly positive lower endpoint of the 10,000-replicate paired bootstrap;
every map within 5% regression; and noninferior P95 and maximum repair rounds.
Only a complete pass freezes StructShell-only as the runtime pool. Official
Adaptive will be evaluated separately and cannot block this pool decision.
