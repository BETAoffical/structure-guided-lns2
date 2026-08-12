# TransactionalRepair confirmation v1: protocol stop

The independent confirmation run stopped after its first execution error.  It
completed 478 registered jobs and wrote 479 atomic policy artifacts, with zero
timeouts.  The failing source state contained a legitimate recorded official
neighborhood that did not touch the current conflict.  The confirmation runner
submitted that neighborhood through `explicit_neighborhood`; the native
contract correctly rejected it and fell back to a different official
neighborhood, after which the exact-neighborhood guard stopped the run.

This is an execution-protocol mismatch, not a PP outcome and not evidence for
or against TransactionalRepair.  The partial artifacts remain diagnostic only
and must not be mixed into another run.

The confirmation will not be restarted in its original form.  The discovery
audit already established that both missing external agents and PP order affect
repair.  Stable repair order is not an allowed runtime control in the next
candidate-pool line, so the deployable question is moved earlier: can an
outcome-blind generator include the externally dependent agents before the
first native PP attempt?  That question is preregistered separately as
`stride-repairclosurepool-v1`.
