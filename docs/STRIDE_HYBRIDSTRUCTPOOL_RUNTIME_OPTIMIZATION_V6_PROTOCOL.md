# STRIDE HybridStructPool runtime optimization v6 protocol

V6 keeps the complete HybridStructPool candidate contract and the frozen V2
pairwise model unchanged.  It does not delete candidates, change structural
sizes, alter CausalClosure evidence, or approximate Copeland scores.

The optimization replaces repeated full conflict scans during candidate audit
with one state-level event/pair incidence index.  For any selected set `S`, the
same counts are recovered exactly:

- `internal` is the indexed weight of conflict edges whose two endpoints are in
  `S`;
- `boundary = sum_incident_degree(S) - 2 * internal`;
- `incident = sum_incident_degree(S) - internal`.

The runtime also avoids copying private occupancy buckets, allocates temporal
contact counters only on first contact, accumulates frontier evidence in
primitive counters, and transfers ownership of locally-created diagnostic
attempts instead of deep-copying them.  Candidate rows remain isolated from all
caller inputs.

Before the paired run, unit tests and a frozen heavy-state shadow check must
show exact `HybridStructPoolResult` equality.  The paired 19-key experiment then
repeats all 152 episodes with 16 workers.  Relative to the registered V5 report,
controller, candidate-generation, and Hybrid-total time must each fall by at
least 10%, while every existing quality, integrity, and per-map gate remains
active.  Failure stops this optimization and does not authorize threshold
relaxation, candidate filtering, raw-TTF testing, or runtime promotion.
