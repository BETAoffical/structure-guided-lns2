# STRIDE HybridStructPool runtime optimization v8 report

## Result

V8 completed all 152 registered engineering episodes (76 HybridStructPool and
76 frozen V2) with zero execution errors or process timeouts. Every integrity,
quality-preservation, per-map, and preregistered engineering gate passed.

The implementation preserves the complete HybridStructPool candidate contract
and frozen V2 Copeland selector. The frozen 500-agent Room state reproduced the
V7 serialized `HybridStructPoolResult` SHA-256 exactly:
`850b7f07a509cf3072fc90f892525056bdb3314be6d3a4417be385e33ac24117`.

| Hybrid runtime component | V7 mean (s) | V8 mean (s) | Reduction |
|---|---:|---:|---:|
| Candidate generation | 5.9412 | 5.2561 | 11.53% |
| Controller before repair | 8.1532 | 7.4541 | 8.57% |
| Hybrid generation total | 6.0120 | 5.3264 | 11.40% |

Relative to the last promoted engineering baseline V6, candidate generation is
14.57% lower, controller time is 9.09% lower, and Hybrid generation total is
14.38% lower.

V8 stores primitive five-counter temporal evidence until final candidate
materialization and iterates constructed contiguous causal windows by bounds.
It retains the general membership fallback for non-contiguous diagnostic input.

## Quality preservation

HybridStructPool success remained 56/76 (73.68%) and platform entry remained
40/76 (52.63%). Normalized fixed AUC was 0.16343 and restricted mean repair
decisions were 29.69. Frozen V2 remained at 44/76 success, 60/76 platform
entries, AUC 0.20652, and 41.06 repair decisions. All per-map capped-wall and
platform gates passed.

The report SHA-256 is
`e70fffb34245c6d3506b713847066870e5a91f8a54c37901ffdf215ec3714f50`.

## Decision boundary

V8 validates an exact engineering acceleration. It is not candidate
compression, ranker retraining, result-blind generalization, raw-TTF evidence,
or default-pool promotion. A fresh preregistered raw-TTF comparison is required
before claiming that the lower overhead closes the remaining V2 gap.

## Validation

- WSL Python: 1013 passed, 35 skipped (`pytest -n 16`).
- Linux native: 13/13 passed (`ctest -j16`).
- Windows native: all tests passed.
- Repository hygiene: 0 errors; 24/24 retained evidence entries verified.
