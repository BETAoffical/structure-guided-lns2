# Historical research

This tree contains completed, unsupported, or non-promoted InitLNS studies.
They remain importable and tested so negative results can be reproduced. Their
standalone CLIs and configurations are not part of the default workflow.

The frozen controller intentionally reuses a small set of audited feature and
candidate helpers from `studies/`. Those source paths are included in the
controller implementation fingerprint, so duplicating or relocating them again
would weaken exact reproducibility. `engineering/` is likewise imported by the
current evaluation orchestrator for legacy report compatibility, but balanced
and proposal-pruner policies remain hidden from the default CLI and are not
promoted controllers.

`experiments/context_audit.py` is a deliberately tiny compatibility shim. The
canonical v1 sklearn pickle stores that historical qualified class name; new
code must use the `research.studies.context` path instead.

- `studies/` groups scientific audits by question.
- `engineering/` contains v2-balanced, proposal-pruner, and four-way tradeoff
  implementations that were not promoted.
- `scripts/`, `configs/`, and `docs/` retain the corresponding entry points,
  frozen protocols, and conclusions.

The pre-official simplified solver and Stage 3-5 implementation were removed
from the active tree. They remain available from Git tag
`pre-repository-restructure-2026-07-20`.
