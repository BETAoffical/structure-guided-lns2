# Research Roadmap

## Scoped contribution

The active claim is deliberately narrow:

> Learn an InitLNS high-level policy conditioned on map topology, static OD
> semantics, density, and the current conflict state. The policy jointly selects
> a conflicting seed agent, Target/Collision/Random neighborhood generation, and
> neighborhood size to improve first-feasible-solution efficiency and
> cross-distribution generalization.

The official neighborhood generators and PP+SIPPS repair remain unchanged. The
project does not claim to be the first learned neighborhood method, variable-size
LNS, or RL extension of LNS2. Complete autoregressive agent-subset generation is
outside the current contribution.

## Evidence gate

Before collecting more data or training RL, the project asks whether static
context adds measurable predictive value beyond the action, seed, and dynamic
repair state. `scripts/run_context_audit.py` constructs Horizon-4 Pareto
preferences from the existing counterfactual collection and trains three fixed-
seed pairwise GBDT ablations:

1. action and seed features;
2. action, seed, and dynamic state;
3. dynamic state plus map, static OD, density, and flow context.

The 2026-07-14 audit used 7,344 candidate outcomes from 128 Train and 72
Validation states. Train and Validation contained no shared map or task instance.
The full-context model improved Pareto top-1 hit rate over the dynamic model by
4.17 percentage points, below the 5-point gate, and worsened mean conflict-AUC
regret by 2.28% instead of improving it by 5%. The paired bootstrap did not show
significant degradation, but the overall offline gate failed.

Consequently, Validation closed-loop evaluation, the planned 31,104/62,208-
outcome expansions, semantic v3 data, and RL training are paused. This result
does not prove context is useless; it says the current Pilot v2 representation,
Adaptive-state distribution, and GBDT protocol do not establish the required
incremental value. See `docs/CONTEXT_AUDIT.md`.

## Baseline taxonomy

- **Official MAPF-LNS2 Adaptive and fixed Target/Collision/Random:** isolate the
  InitLNS high-level choice while keeping the low-level solver identical.
- **GPBS:** an independent end-to-end feasibility solver and high-level search
  baseline. It is not represented as an LNS2 destroy heuristic.
- **LNS2+RL:** an end-to-end comparison with a learned low-level repair policy,
  not evidence about this project's high-level action selection alone.
- **DiffLNS:** a paper-level learned-initialization comparison when no public
  implementation is available.
- **BALANCE, ADDRESS, and NNS:** useful policy ideas and code references. Any
  version applied to InitLNS must be labeled an adapted baseline because the
  original method did not directly evaluate this exact InitLNS control problem.

## Conditional next stages

The following stages resume only after a revised, predeclared context audit
passes the offline gate:

1. Run the learned policy closed-loop on 24 Validation instance-seeds. Successes
   must not fall below Adaptive, and conflict AUC or time-to-feasible must improve
   by at least 5%.
2. Create semantic v3 with 12 Train maps, 6 Validation maps, and 132 ID/OOD test
   instances.
3. Collect up to four repair phases, six seeds, three rules, three sizes, and two
   trials per Train/Validation episode, then collect a second round from states
   visited by the supervised policy.
4. Use supervised ranking only as warm start. Train contextual RL with conflict-
   graph and static-context encoders plus conditional seed/rule/size heads.
5. Reward normalized collision reduction and feasibility while penalizing
   low-level generated nodes; select coefficients only on Validation.
6. Evaluate official strategies, random legal actions, contextual bandit,
   ADDRESS-inspired InitLNS, supervised policy, RL, and GPBS on ID and every OOD
   split.

Primary metrics are success, time-to-feasible, conflict AUC, SIPPS calls/nodes,
and runtime. Sum of costs is secondary because the research target ends at the
first feasible solution.

## Public code provenance

- [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2): pinned active solver core.
- [GPBS](https://github.com/shchan13/GPBS): pinned independent feasibility baseline.
- [BALANCE](https://github.com/thomyphan/anytime-mapf): online heuristic/size bandit reference.
- [ADDRESS](https://github.com/JimyZ13/ADDRESS): delayed-agent seed policy reference.
- [NNS](https://github.com/mit-wu-lab/mapf_neural_neighborhood_search): candidate-ranking reference.
- [LNS2+RL](https://github.com/marmotlab/LNS2-RL): learned low-level repair baseline.
- [Unified LNS benchmark](https://github.com/ChristinaTan0704/mapf-lns-benchmark): evaluation reference only; no unlicensed implementation is copied.

No author release was found for DROP-LNS or DiffLNS during the review, so they
remain paper-level comparisons unless provenance changes.
