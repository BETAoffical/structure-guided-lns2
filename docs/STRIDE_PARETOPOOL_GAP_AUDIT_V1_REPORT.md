# STRIDE ParetoPool candidate-gap audit v1

## Result

The preregistered audit completed with all integrity checks passing:

- 90 temporal checkpoint occurrences and 78 unique state fingerprints;
- complete budgets 6, 8, and 12 for every occurrence;
- a non-empty natural structural pool for every state;
- every budget respected; and
- the frozen V2 base pool and anchor retained separately.

The frozen report is
`build/stride-paretopool-gap-audit-v1/paretopool_gap_audit_report.json`, with
SHA-256
`c220861e972485ba0724cead3fd67c3663d7960d60d52a940e546475f7de9061`.

## Candidate counts

| Budget | Mean raw | Mean selected | Mean novel |
|---:|---:|---:|---:|
| 6 | 10.00 | 5.49 | 5.28 |
| 8 | 10.00 | 6.49 | 6.17 |
| 12 | 10.00 | 7.47 | 7.08 |

The natural closure rules therefore do not reproduce the old fixed 8/16/24/32
grid under another name.  They produce mostly novel agent sets, and many states
naturally contain fewer candidates than the nominal budget.

## Historical overlap boundary

The exact old one-step best was retained in 4.44%, 4.44%, and 6.67% of state
occurrences for budgets 6, 8, and 12.  Only 18, 23, and 23 occurrences had any
labeled overlap candidate.  Within those narrow overlap subsets, mean
normalized one-step regret was 0.00153, 0.00213, and 0.00190.

This is not a failure of the new pool: the historical replay contains no
outcome for most new candidates.  It proves that old one-step labels cannot
select the ParetoPool budget or validate future quality.  The next valid step is
strictly paired, dual-teacher, multi-horizon rollout collection over the new
candidates.  No model training, runtime promotion, or TTF claim is authorized
by this audit.
