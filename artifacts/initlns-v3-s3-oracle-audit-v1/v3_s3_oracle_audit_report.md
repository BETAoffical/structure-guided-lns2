# v3-S3 Oracle upper-bound and regret audit

Decision: `proceed_to_long_horizon_value_pilot`

This is a retrospective diagnostic over the existing paired S3 collection. Oracle rows use observed labels and are not deployable models or independent validation.

| Strategy | Reduction | Seconds | Reduction / second | No progress | Feasible |
|---|---:|---:|---:|---:|---:|
| v2_full | 6.5267 | 0.567274 | 11.5053 | 9.333% | 35.333% |
| model_s3 | 5.0100 | 0.398992 | 12.5566 | 7.000% | 39.000% |
| oracle_s3_efficiency | 6.3700 | 0.368171 | 17.3018 | 0.667% | 45.667% |
| oracle_s3_quality_time | 7.1933 | 0.524264 | 13.7208 | 0.000% | 50.000% |
| oracle_s3_reduction | 7.1967 | 0.525599 | 13.6923 | 0.000% | 50.000% |

## Decomposition

- Model exact sequence match with the efficiency Oracle: 16.000%.
- Model first-template match with the efficiency Oracle: 21.333%.
- Observed Oracle efficiency headroom over v2: +50.38%.
- Model captured fraction of positive Oracle-v2 headroom: 18.137%.
- Quality-constrained Oracle reduction retention versus v2: 110.215%.

## Closed-loop evidence

| Report | Pairs | Time change | Iteration change | Wall AUC change |
|---|---:|---:|---:|---:|
| initlns-v3-s3-stable-runtime-v1/runtime_comparison_report.json | 4 | +11.58% | +30.56% | +10.64% |
| initlns-v3-s3-stable-runtime-v2-repeat/runtime_comparison_report.json | 4 | +1.74% | +30.56% | +4.58% |

## Interpretation boundary

- The Oracle is optimistic because it selects after seeing the paired outcomes.
- The collection observes at most the registered three-step prefix. It does not contain true remaining time-to-feasible labels.
- Therefore this audit can diagnose selection regret and a local-versus-closed-loop contradiction, but cannot train or validate a long-horizon value controller.
