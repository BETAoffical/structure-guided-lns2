# STRIDE HybridStructPool runtime optimization v6 report

## Result

V6 completed all 152 registered engineering episodes (76 HybridStructPool and
76 frozen V2) with zero terminal errors or process timeouts.  Every integrity,
quality-preservation, per-map, and runtime gate passed.

The optimization preserves the exact complete candidate contract and frozen
Copeland selector.  A frozen heavy-state shadow comparison produced the same
serialized `HybridStructPoolResult` SHA-256 before and after the change:
`07d6ff9986b0307b6524cbc88abe63b21f4cd402be7e096545e09d06274120d1`.

| Hybrid runtime component | V5 mean (s) | V6 mean (s) | Reduction |
|---|---:|---:|---:|
| Candidate generation | 7.8024 | 6.1524 | 21.15% |
| Controller before repair | 9.9329 | 8.1996 | 17.45% |
| Hybrid generation total | 7.8889 | 6.2206 | 21.15% |

The main change is an exact state-level conflict-incidence index.  Candidate
audit now recovers the same internal, boundary, and incident conflict counts
without rescanning every conflict event for every candidate.  Smaller
allocation changes remove redundant occupancy copies, unused conflict
adjacency, eager temporal-list allocations, and unnecessary diagnostic deep
copies.

## Quality preservation

HybridStructPool retained the registered behavior on the 76-episode engineering
cohort: success was 56/76 (73.68%), normalized fixed AUC was 0.16347,
restricted mean repair decisions were 29.69, and platform entry was 52.63%.
The corresponding frozen V2 values were 44/76 (57.89%), 0.20652, 41.06, and
78.95%.  These are bounded engineering metrics, not raw TTF.

All registered gates passed, including per-map capped-wall and platform
preservation.  The report SHA-256 is
`f89cecfb57e763daa34b4e284db658206ecaf516f57b366ade8914b8e809fce8`.

## Decision boundary

V6 validates an exact engineering acceleration, not candidate compression or
runtime promotion.  It does not establish result-blind generalization, raw-TTF
superiority, a rescue mechanism, or a new default pool.  Those questions remain
separate paired experiments.

## Validation

- WSL Python: 1010 passed, 35 skipped (`pytest -n 16`).
- Linux native: 13/13 passed (`ctest -j16`).
- Windows native: all tests passed.
- Repository hygiene: 0 errors; 24/24 retained evidence entries verified.
