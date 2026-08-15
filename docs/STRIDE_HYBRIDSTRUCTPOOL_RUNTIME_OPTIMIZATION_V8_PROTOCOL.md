# STRIDE HybridStructPool runtime optimization v8 protocol

V8 keeps the complete HybridStructPool candidate universe and frozen V2
pairwise selector unchanged. It does not remove candidates, alter neighborhood
sizes, retune a threshold, change evidence values, or cache across decisions.

The CausalClosure hot path now stores primitive five-counter temporal evidence
until a final candidate is materialized. This removes repeated construction and
immediate decomposition of immutable evidence objects. Localized causal windows
are contiguous by construction, so their inner temporal scan uses bounded
integer iteration; an explicit membership fallback preserves arbitrary
non-contiguous diagnostic inputs.

Targeted tests and a frozen 500-agent Room state must reproduce the V7 serialized
result SHA-256 exactly. Three alternating heavy-state runs must show at least a
5% median reduction relative to frozen V7 before the paired run begins.

The full 19-key, 152-episode engineering cohort is then rerun with 16 workers.
Relative to the frozen V7 report, mean controller, candidate-generation, and
Hybrid-total time must each improve by at least 5%, while all integrity,
success, AUC, repair-decision, platform, and per-map gates remain active.
Failure does not authorize candidate filtering, raw-TTF claims, or promotion.
