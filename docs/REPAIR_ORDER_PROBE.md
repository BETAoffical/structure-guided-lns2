# InitLNS PP Repair Order Mechanism Probe

## Question

The policy-visited sequential-credit audit found stable H1 rank signal but unstable H4 labels. This
Train-only mechanism probe asks whether the missing action variable is the PP order used to replan agents
inside an explicit neighborhood.

The existing 6,872 trials are registered as a zero-cost diagnosis: H1 split-half Spearman is `0.6324`,
H4 Spearman is `0.3575`, 55.5% of candidates vary at C1, and 81.3% vary in H4 AUC. The probe does not fit
a model, use static context, unseal Validation/OOD, or define an RL reward.

## Explicit Order API

An explicit-neighborhood action may include `repair_order`, a complete permutation of `agents`. It is
accepted only with PP repair. The transition reports the requested and actual order. Omitting the field
uses the original random shuffle with the original RNG sequence; official parity must remain unchanged.

A solution fingerprint hashes paths, conflicts, SOC, delay and agent IDs while excluding runtime,
low-level counters and external context. It distinguishes genuine path divergence from machine timing or
search-accounting differences.

## Registered Design

- Select 24 existing `policy_train` states, two from each of 12 maps, balancing repair stage, conflict
  severity, task and solver seed without reading candidate outcomes.
- Select six concrete neighborhoods per state, retaining frozen v1's choice and sizes 4, 8 and 16.
- Evaluate eight random-order trials with a state/trial common random seed shared across candidates.
- Evaluate four deterministic rules twice: ID ascending, conflict degree descending, delay descending and
  path length descending, with agent ID as every tie-break.
- Apply the controlled order only to the first repair. Frozen v1 continues for at most three steps.

The exact formal budget is 24 states, 144 neighborhoods, 16 conditions and 2,304 isolated trials, with at
most 9,216 repairs. Workers remain four and each trial has a 180-second hard timeout. Outputs live under
ignored `build/initlns-repair-order-probe-v1`.

## Gates

- All candidate pools must replay exactly; deterministic duplicates must have identical order, solution
  fingerprints and conflict trajectories; no trial, split or action error is allowed.
- Random CRN split-half Spearman, Pareto Jaccard and best-set Jaccard must each reach `0.5`.
- Repair order is material only if solution divergence reaches 50%, C1 conflict divergence reaches 30%,
  deterministic-order oracle H4 AUC improvement reaches 5%, positive opportunity reaches 60%, and the
  map-bootstrap lower bound is non-negative.
- A fixed order dominates if it is within normalized AUC 5% of the deterministic oracle on at least 80%
  of state-neighborhood pairs.

The registered decisions are: adopt a dominant fixed order; advance to contextual order selection when
order matters without a dominant rule; retain expected-neighborhood learning when only CRN stabilizes;
or stop neighborhood ranking and RL when neither mechanism is stable.

## Commands

```powershell
python scripts/run_repair_order_probe.py --phase diagnose
python scripts/run_repair_order_probe.py --phase dry-run
```

Native collection runs in WSL:

```bash
PYTHONPATH=build/linux/project python3 scripts/run_repair_order_probe.py \
  --phase all --output build/initlns-repair-order-probe-v1
```
