# InitLNS local representation audit

This audit is a stopping-rule experiment over the existing 7,344 controlled
counterfactual outcomes. It does not collect new solver data, train RL, or add a
deep-learning dependency.

## Question

The earlier context audits scored a requested seed, rule, and neighborhood size
without seeing the agent set produced by the official destroy heuristic. This audit
separates two decision surfaces:

- `action_index.jsonl` aggregates trials by `(state, seed, rule, size)` and contains
  only information available before neighborhood generation.
- `realized_index.jsonl` keeps each trial and its exact first-step neighborhood. It
  represents generating legal candidates first and then ranking the realized sets.

Every state rebuilds vertex and edge conflicts from full paths. Paths hold at their
goal after arrival. The reconstructed colliding-agent pairs must exactly equal the
recorded conflict graph before any model is trained.

## Feature boundary

`dynamic_action` is the previous dynamic baseline with categorical neighborhood
size. `local_pre` adds seed conflict timing, waiting, path heat, local obstacle
density, graph degree, articulation exposure, and size ratios. `local_pre_context`
adds the existing map, static OD, and density context. `realized` adds conflict-edge
coverage, component coverage, selected-agent statistics, path overlap, spatial span,
and local topology for the actual agent set. `realized_context` adds static context.

Repair results, repaired paths, conflict reduction, post-action cost, generated
nodes, and runtime are not model inputs. Generated nodes and runtime are available
only as label sensitivity objectives.

## Labels and evaluation

The primary label is the Horizon 1 effectiveness Pareto set over feasibility,
remaining conflicts, and conflict AUC. Horizon 4 uses the same definition as a
long-term sensitivity analysis. A compute-aware label adds generated nodes; a
machine-sensitive label additionally adds branch runtime.

The primary learner is a fixed pairwise scikit-learn GBDT trained on Horizon 1
effectiveness. Horizon 4, compute-aware, and runtime-sensitive analyses rescore the
same fold-local policies, so sensitivity is not confounded by retraining. Three fixed metric GBDT
regressors predict remaining conflicts, AUC, and `log1p(generated)` only as an
auxiliary diagnostic. Evaluation uses map-grouped three-fold cross-validation,
2,000 map bootstraps, and 500 task-level context permutations.

The persisted pairwise models are the three fold-local models for each feature
profile. They are the exact models used to produce the reported held-out choices;
the audit does not fit a separate all-data deployment model.

```powershell
python scripts/run_local_representation_audit.py `
  --collection build/repair-experience-calibration-v2 `
  --dataset build/repair-transfer-pilot-v2 `
  --output build/initlns-local-representation-audit
```

Use `--strict` only in automation that should return exit code 2 when the research
gates fail. A failed gate is a valid experimental result and still writes all
reports and models.

## Interpretation

- Only realized-neighborhood observability passes: rank generated candidate sets and
  defer generation-before-action RL.
- Local pre-generation features and context pass: retain the transfer-oriented
  high-level policy and collect independent dual-trial confirmation data.
- Local features pass but static context fails: retain dynamic high-level control and
  narrow the static transfer claim.
- Horizon 1 passes while Horizon 4 fails: investigate rollout variance with more
  trials.
- All local representations fail: run a small MovingAI mechanism probe before any
  additional collection.

NNS, MAPF-ML-LNS, and the unified MAPF-LNS evaluation framework informed the
representation categories only. No code was copied from repositories whose license
was not sufficiently clear.

## Current result

The formal 7,344-outcome audit did not pass the pre-registered gates. Relative to
`dynamic_action`, `local_pre` gained 2.0 Pareto top-1 percentage points and reduced
AUC regret by 2.1%. Relative to `local_pre`, `realized` reduced AUC regret by 7.3%
but gained only 1.5 top-1 points; both map-bootstrap intervals crossed zero. Static
context changed top-1 by -0.5 points, worsened AUC regret by 4.0%, and reached only
the 31.2% and 41.0% task-permutation percentiles.

There was no size collapse: the strongest pre-generation contextual profile chose
sizes 4, 8, and 16 in 72, 16, and 112 of 200 states. The label sensitivity did expose
the old compute bias. Size 4 was Pareto-supported in 54.5% of states under the
effectiveness label and 95.5% after generated nodes were added.

The registered decision is therefore `run_movingai_mechanism_probe_before_more_collection`.
This result does not justify expanded collection or RL training.
