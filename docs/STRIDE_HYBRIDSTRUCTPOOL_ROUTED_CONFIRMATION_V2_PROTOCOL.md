# STRIDE HybridStructPool routed confirmation v2

Version 2 is a new preregistered confirmation identity. It imports no timed
episode from confirmation v1, r1, or r2. The frozen challenger remains exactly
`structshell_only`; candidate families, sizes, native feature extraction,
Copeland ranking, PP order, and episode-stream seed semantics are unchanged.

The v1 qualification run found three preregistered maps whose six keys were all
initially feasible. Before any controller outcome on those maps was observed,
v2 replaces them with the three unused MovingAI random maps at the maximum
already-materialized load. This is an initial-state eligibility repair, not a
selection based on V2, StructShell, or official controller performance. The
remaining seven maps and all their task and seed identities remain unchanged.

Qualification is now an enforced hard gate: every one of the ten maps must
produce all six registered resets, have at least one nonzero-conflict state,
and report `eligible_for_closed_loop`. Any failed map stops the experiment
before the first timed PP episode. Formal execution remains rotating, strictly
serial, and reset-inclusive for 60 paired keys and three controllers.

The process safety fuse is preregistered at 900 seconds for every controller.
This replaces the v1 300-second fuse after official Adaptive exceeded it on a
600-agent Maze key. A process timeout remains a terminal execution failure; it
is not converted into a win, a capped TTF value, or right-censored performance.

All original promotion gates remain unchanged. Passing is required before
StructShell-only can replace V2. Failure keeps V2 as the default and ends this
runtime-pool confirmation branch.
