# InitLNS ranking-objective alignment audit

## Purpose

This development audit tests whether the policy-visited aggregation failure was caused by a mismatch
between pairwise training and top-1 neighborhood selection. It reuses the registered 442 states, 7,914
explicit neighborhoods and 31,656 PP-order trials. It does not use static context, collect new labels or
train RL.

The fixed comparison contains the existing equal-state pairwise GBDT, an impact-weighted pairwise GBDT
and a dual-outcome GBDT. The impact model preserves total state weight one and weights dominance pairs by
the larger of solved-rate separation and normalized remaining-conflict separation. The dual model
predicts solved rate and normalized residual conflicts; solved predictions are quantized to the observed
four-trial resolution of 0.25 before deterministic selection.

Model selection uses leave-one-map-out predictions from the 12 policy Train maps. The historical 23
states are training-only anchors in every fold. The 154 policy Validation states are not passed to model
selection and may be evaluated only if one challenger passes the Train-map gate.

## Candidate reliability

The registered data support a meaningful neighborhood effect:

- Mean action eta-squared is `0.6047`; 76.24% of states are at least 0.5.
- Split-trial candidate ranking has mean Spearman `0.6321` across 438 valid states.
- The best sets from the two trial halves overlap in 78.28% of states.
- The candidate pool contains a conflict-reducing action in 99.77% of states.
- The oracle one-step reduction is 6.91 conflicts on average and 1.75 at the median.

These diagnostics do not support blaming the failure primarily on an empty action space or overwhelming
PP-order noise.

## Preregistered result

The Train-map gate failed before development Validation:

| Objective | Pareto top-1 | Mean conflict regret | Top-1 delta | Regret improvement | Maps no worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal pairwise | 49.65% | 0.28244 | baseline | baseline | baseline |
| Impact pairwise | 47.22% | 0.29650 | -2.43 pp | -4.98% | 6/12 |
| Dual outcome | 45.49% | 0.32414 | -4.17 pp | -14.76% | 4/12 |

Neither challenger met the 3%/5% improvement gate, the map-bootstrap gate or the required 8/12 maps no
worse. No winner was selected, Validation was not evaluated, no portable winner was registered and the
12-map independent confirmation dataset was not generated.

The result narrows the diagnosis: simply reweighting pairwise examples or replacing pairwise learning
with two direct outcome heads does not fix cross-map top-1 selection. Supervised ranking and RL remain
paused. The next research decision is to redesign the explicit candidate control or make PP repair order
part of the action, rather than tune these three objectives after seeing their result.

## Reproduction

```powershell
python scripts/run_ranking_objective_audit.py `
  --collection build/initlns-policy-visited-natural-v2-collection `
  --training build/initlns-policy-visited-natural-v2-training `
  --offline build/initlns-policy-visited-natural-v2-offline `
  --config configs/ranking_objective_audit.json `
  --output build/initlns-ranking-objective-audit-v1 `
  --phase all
```

The unused confirmation configuration fixes master seed `20270421`, 12 maps, four tasks per map and
three solver seeds, for a theoretical 144 instance-seed cohort. Its gated generator rejects the current
failed audit report. Generated reports and model intermediates remain ignored under `build/`.
