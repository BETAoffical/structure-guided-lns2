# Realized-neighborhood stability probe

## Purpose

The independent-layout probe showed that a nominal `(seed, rule, size)` action
often generates different agent sets. This bounded follow-up separates two random
effects before any model training:

1. Proposal randomness chooses the realized agent neighborhood.
2. Evaluation randomness changes the PP repair order for a fixed agent set.

The probe reuses only the 23 qualified `probe` states and their 6,480 proposal
outcomes. Test/OOD data, supervised learning, and RL remain excluded.

## Candidate construction

For each state, actual neighborhoods are deduplicated by their sorted agent IDs.
Within every `(Target|Collision|Random, 4|8|16)` family, two representatives are
selected deterministically:

- the most frequently generated set;
- the remaining set with maximum Jaccard distance from the first, with frequency
  and neighborhood hash as deterministic tie breakers.

The union across nine families is deduplicated again. Candidate selection reads
only the requested proposal action, proposal seed, and actual neighborhood. It is
forbidden from reading post-repair conflicts, feasibility, generated nodes, or
runtime.

Each selected set is applied through `explicit_neighborhood` for eight new
evaluation seeds. Evaluation seeds use a separate namespace and are checked not to
overlap that candidate's proposal seeds. Reset and prefix replay must reproduce the
source state fingerprint, and the solver must return exactly the requested set.

## Registered gates

- Integrity: no replay errors, missing trials, seed overlap, rejected actions, or
  changed explicit neighborhoods.
- Coverage: all 23 source states and at least nine explicit candidates per state.
- Realized-neighborhood eta-squared at least 0.5.
- Eta-squared improvement over the nominal-action result of 0.404 at least 0.05.
- Trial-split candidate-rank Spearman at least 0.5.
- Trial-split exact Pareto-candidate Jaccard at least 0.5.
- Trial-split best-candidate Jaccard at least 0.5.
- At least 60% of states contain more than one mean effectiveness result.
- No proposal family uniquely supports a Pareto winner in more than 80% of states.

Generated nodes remain a separate compute-aware sensitivity objective. Runtime is
recorded but is not a scientific label.

## Interpretation

- All gates pass: proceed to a dynamic-state plus realized-neighborhood ranking
  audit, with static context added only as an ablation.
- Eta and its improvement pass but top-candidate stability fails: PP order remains
  decision-relevant; model order or use a robust distributional objective.
- Eta does not improve: fixing the agent set is insufficient; include repair order
  in the action and redesign high-level repair control.

The dry-run contains 23 states, 412 selected explicit candidates, and 3,296
evaluation outcomes with four process workers.

## Completed result

The explicit replay completed all 23 states, 412 candidates, and 3,296 evaluation
outcomes with zero errors, missing trials, seed overlaps, or neighborhood changes.
All registered gates passed:

- realized-neighborhood eta-squared: 0.595, up 0.191 from nominal actions;
- map-bootstrap 95% interval for eta-squared: 0.529-0.636;
- trial-split rank Spearman: 0.803;
- exact Pareto-candidate Jaccard: 0.518;
- best-candidate Jaccard: 0.547;
- distinct mean outcomes: 100% of states;
- maximum fixed proposal-family share: 13.0%.

The registered decision is `proceed_to_realized_neighborhood_ranking_audit`.
This establishes that concrete agent sets have a substantially more stable ranking
signal than nominal seed/rule/size distributions. It does not isolate the full eta
gain as a causal effect of fixing the set, because representative selection also
deliberately increases candidate diversity (mean pairwise Jaccard distance 0.909).
The next audit must compare learned rankings on the same candidate pools and keep
static context as an ablation rather than a claimed contribution.

## Commands

```powershell
python research/scripts/neighborhood/collect_realized_neighborhood_probe.py `
  --dataset build/initlns-independent-layout-probe-v1 `
  --source-collection build/initlns-independent-layout-probe-v1-collection `
  --output build/realized-neighborhood-stability-probe-v1 `
  --dry-run
```

Run the same command without `--dry-run`, then analyze with:

```powershell
python research/scripts/neighborhood/analyze_realized_neighborhood_probe.py `
  --collection build/realized-neighborhood-stability-probe-v1 `
  --output build/realized-neighborhood-stability-probe-v1-report
```
