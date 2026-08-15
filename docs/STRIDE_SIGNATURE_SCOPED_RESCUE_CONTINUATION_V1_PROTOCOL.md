# STRIDE Signature-Scoped Rescue Continuation V1

## Question

After the existing compact-plus-blocker rescue changes the repair state but a
later, distinct platform forms, does one additional rescue based only on that
new failed PP diagnostic improve the bounded trajectory?

The repairability-basin audit rejected accumulating more blockers on an
unchanged signature. It found instead that all seven observed post-rescue new
platforms exposed at least one blocker absent from the original rescue set.
This experiment tests that single remaining mechanism claim directly.

## Frozen mechanism

- A platform signature is the repair-structure fingerprint plus conflict-edge
  set.
- Three consecutive exact `conflict_bound_exceeded` rollbacks form a platform.
- Each distinct signature can schedule at most one action for the next
  decision.
- The action semantically compacts the failed neighborhood and adds at most
  eight external blockers reported by that failed PP.
- It uses native randomized PP order and a deterministic fresh seed.
- It replaces, rather than supplements, the next controller action; there is
  exactly one PP call per decision.
- The episode cap is three distinct-signature rescues. `time_limit` never
  schedules a rescue.

## Paired screen

The 45 frozen `first_repeat_stall` states are tested with trial 0--3 and three
arms: frozen controller, the previous one-shot compact-plus-blocker rescue, and
the signature-scoped mechanism. All arms share the first candidate and PP seed.
Episodes run to feasibility or 64 repair decisions / 180 seconds, with 16
workers for non-timing collection.

Uniform trial 4--7 extension is allowed only when at least one later-signature
rescue executes and the signature-scoped arm is no worse than one-shot on
success, normalized fixed AUC, restricted repair decisions, and charged repair
wall time, with no map losing more than five success-rate points.

Passing remains mechanism evidence only. It does not authorize runtime
integration, candidate-pool replacement, learning, OOD claims, or TTF claims.
