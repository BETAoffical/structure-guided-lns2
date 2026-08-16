# Warehouse fixed16 development v2 protocol

## Scope

This is a qualification-conditioned, single narrow-aisle, high-density
Warehouse development ablation. It uses only
`warehouse-10-20-10-2-1` at 600 agents. It is not evidence for four-map or
general Warehouse performance and cannot replace the runtime default.

The previous r2 qualification failed overall and ran zero formal episodes.
Its qualification report hash is recorded only as the design source for the
frozen slice. No r2 reset, manifest, trace, or episode is imported. The v2
runner rejects an output root equal to or nested beneath `build/wh-f16-v1-r2`.

## Frozen cohort

- Task seeds: 233 and 277.
- Task variants: `opposite_exchange` and `uniform_random`.
- Agent count: 600. Loads 800 and 1000 are forbidden.
- Qualification solver seeds: 19, 20, 21, and 22.
- Six formal arms: frozen V2, legacy Boundary16 static-cache, and four
  all-state fixed16 StructShell family ablations.
- Formal pairing: eight paired keys and 48 strictly rotating serial episodes.
  Half A and Half B each contain four paired keys.

The four regenerated scenario and task JSON hashes are pinned in the config.
They detect endpoint-generation drift or byte tampering; they do not by
themselves distinguish a byte-identical copy. The runner has no r2 import path,
and the qualification manifest, report, and later traces must be produced in
the new output root by the current producer.

## Q0 and Q1

Q0 is solver-free. It verifies four deterministic tasks, 600 unique starts,
600 unique goals, no start-goal fixed points, complete four-neighbour
reachability, pinned task hashes, and absence of forbidden loads.

Q1 is a fresh 16-key reset-only qualification. It passes only if every reset
is complete and consistent with no error, every opposite-exchange key has at
least 16 initial conflicts, and every uniform-random key has at least one.
There is no map, load, task, or seed replacement.

Formal collection is blocked unless the stored Q0 audit and a recomputed Q1
selection pass. Every formal schedule row binds the fresh qualification
manifest and report hashes. The runner can proceed directly into the existing
six-arm fixed16 formal mechanism after these checks.

## Formal gates and claims

The only performance gates are complete integrity, challenger success count
not below V2, and strictly lower mean restricted TTF overall and in both
halves. AUC, repair iterations, PP time, selection time, and paired-faster
fraction are diagnostic only. Boundary16 reuses the legacy topology-boundary
algorithm and may add at most two candidates; each single-family arm may add
at most one candidate.

Official Adaptive is excluded from this development ablation. A Boundary16
winner may enter separately preregistered fresh final confirmation directly.
A single-family winner must first enter an independent 8/16/24/32 size study;
only its subsequently frozen size winner may enter fresh final confirmation.
Development success is not a speed claim, a Warehouse generalization claim,
or permission for runtime promotion.

## Commands

`plan` and all `--dry-run` paths are non-solving. `prepare-q0` generates only
the new dataset and Q0 audit. `qualify` performs fresh resets. `collect` refuses
to run before Q0/Q1 pass; `run` performs Q0/Q1 and enters formal only on a pass.
