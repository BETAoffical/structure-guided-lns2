# InitLNS Policy-Visited Sequential Credit Audit

## Question

The frozen `realized_dynamic` v1 controller improved multi-seed closed-loop
conflict AUC, but later Horizon-1 supervised rankers did not improve it. This
audit tests one remaining explanation before RL:

> Does a one-step Pareto label hide stable Horizon-4 action value at states
> actually visited by the frozen v1 policy?

This is a Train-only diagnostic. It does not fit a model, define an RL reward,
use static map/OD/density features, or unseal Validation, Test, or OOD labels.

## Registered Inputs

The audit first verifies the exact SHA256 hashes of the policy-visited source
states, explicit candidate pools, source run configuration, dataset, frozen
portable v1 policy, and the earlier calibration report. The zero-cost legacy
diagnosis must reproduce 200 states, 7,344 outcomes, zero integrity errors, and
mean H1/H4 Pareto Jaccard `0.5060849648717296`.

The formal source contains 288 `policy_train` states on 12 maps. Exactly eight
states per map are selected by deterministic round-robin strata over repair
stage, normalized conflict severity, task variant, and solver seed. The final
96 states are fixed before any Horizon-4 outcome is observed.

## Horizon-4 Trials

Each state replays its full explicit-action prefix and must reproduce the
registered state fingerprint. Its candidate pool is regenerated with the
existing frozen controller and must match the original candidate IDs, agent
sets, proposal seeds, and provenance exactly.

For every explicit candidate, four independent PP repair-order trials run in
separate processes. The candidate is applied first; if the state is not yet
feasible, frozen v1 generates and selects candidates for at most three further
steps. A trial records `C0...C4`, padding with zero after early feasibility,
fixed-budget conflict AUC, final conflicts, low-level nodes, per-step candidate
pools and selections, and diagnostic runtime. Each process has a 180-second
hard timeout. Trial files are atomic and collection supports whole-trial resume.

The formal upper bound is 96 states, 18 candidates per state, four trials, or
6,912 trials and 27,648 repairs. A two-state dry run reports the exact candidate
and trial count before native collection.

## Registered Gates

H1 and H4 effectiveness Pareto sets maximize four-trial feasibility rate and
minimize final conflicts and fixed-budget conflict AUC. Generated nodes are a
compute-aware sensitivity only.

- Split-half trial rank Spearman, Pareto Jaccard, and best-set Jaccard must each
  be at least `0.5`.
- Mean H1/H4 Pareto Jaccard must be at most `0.70`, and at least half the states
  must have different H1 and H4 best sets.
- The H4 oracle must improve conflict AUC over the frozen v1 source choice by at
  least `5%` on average, with positive opportunity in at least `60%` of states,
  at least 8/12 maps non-worse, and a non-negative map-bootstrap lower bound.
- All 96 states and four trials per candidate must complete with no replay,
  action, timeout, split, or unexplained errors.

All gates passing permits one fixed long-term value/ranker experiment before
RL. Unstable H4 labels stop the route. Stable but H1-like labels reject the
credit-mismatch explanation. Stable and different labels without oracle room
mean frozen v1 is already near this candidate space's Horizon-4 limit.

## Commands

```powershell
python research/scripts/sequential/run_sequential_credit_audit.py --phase diagnose
python research/scripts/sequential/run_sequential_credit_audit.py --phase dry-run --smoke-states 2 `
  --output build/initlns-sequential-credit-audit-v1-dry-run
```

Native collection runs in the registered WSL environment:

```bash
python3 research/scripts/sequential/run_sequential_credit_audit.py --phase all --workers 4 \
  --output build/initlns-sequential-credit-audit-v1
```

All generated states, trials, indices, and reports remain under ignored
`build/`; only code, configuration, tests, and the final written conclusion are
versioned.

## Formal Result

The preregistration was pushed at commit `6f3a33b` before formal labels were
generated. Collection then completed all 96 states, 1,718 explicit candidates,
and 6,872 isolated trials in about 24 minutes. All 96 candidate pools reproduced
the source trace exactly. There were zero timeouts, replay mismatches, rejected
or changed actions, unexplained errors, and non-Train labels.

| Gate metric | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Split-half rank Spearman | >= 0.50 | 0.3575 | no |
| Split-half Pareto Jaccard | >= 0.50 | 0.3855 | no |
| Split-half best-set Jaccard | >= 0.50 | 0.3725 | no |
| H1/H4 Pareto Jaccard | <= 0.70 | 0.7214 | no |
| States with changed H1/H4 best set | >= 50% | 35.42% | no |
| H4 oracle AUC improvement | >= 5% | 23.66% | yes |
| States with positive H4 opportunity | >= 60% | 63.54% | yes |
| Maps non-worse | >= 8/12 | 12/12 | yes |

The map-bootstrap 95% interval for apparent oracle AUC improvement was
`[18.09%, 29.73%]`. However, this opportunity cannot be used as a learning
target because the independent trial halves disagree on candidate ranking,
Pareto membership, and the best set. Selecting the best of many candidates
under this instability also makes the raw oracle estimate optimistic.

The policy-visited H1 and H4 labels were more similar than the earlier
calibration labels: mean Pareto Jaccard increased from `0.5061` to `0.7214`, and
only about one third of states changed best sets. Thus the formal decision is
`stop_h4_labels_unstable`. Per the preregistration, the project does not add
trials after seeing this result, train a long-term ranker, or enter RL.

This result rejects the tested four-trial Horizon-4 route, not all possible
sequential control. The next research step must change the action/value
definition or use a lower-variance evaluation design justified independently;
it cannot present the observed oracle gap as evidence that the current H4
labels are learnable.
