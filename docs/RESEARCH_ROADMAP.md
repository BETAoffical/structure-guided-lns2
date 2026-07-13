# Research Roadmap

## Contribution target

The contribution is not another classifier that imitates which fixed heuristic MAPF-LNS2 chose. It is
a contextual, sequential repair controller that transfers historical experience across map topology,
task flow, density, and agent count, then optimizes actual time-to-feasible through interaction.

The first policy chooses:

1. a conflicting seed agent;
2. Target, Collision, or Random neighborhood construction;
3. neighborhood size.

The official generator completes the subset, preserving a constrained and interpretable action space.
Only after this controller shows robust out-of-distribution gains should an autoregressive policy produce
complete agent subsets.

## Implementation stages

1. **Trusted baseline:** pin official MAPF-LNS2, verify path-level parity, standardize MovingAI input,
   expose step-wise repair state and raw outcomes.
2. **Experience collection:** run official Adaptive and fixed heuristics, record conflict graphs, paths,
   delays, neighborhoods, low-level search effort, conflict reduction, and elapsed time.
3. **Supervised warm start:** rank seed/heuristic/size actions using counterfactual rollout outcomes. This
   initializes the policy and is not itself the claimed contribution.
4. **Contextual RL:** encode the dynamic conflict graph with a GNN and combine it with map/task context;
   fine-tune masked actions against conflict AUC and time-to-feasible.
5. **Transfer evaluation:** hold out layout families, task flows, densities, and agent-count ranges;
   report zero-shot performance and sample-efficient adaptation separately.
6. **Subset-generation extension:** attempt explicit sequential subset construction only if the smaller
   action space establishes a reliable advantage.

## Public code reviewed

- [MAPF-LNS2](https://github.com/Jiaoyang-Li/MAPF-LNS2): active solver core.
- [BALANCE](https://github.com/thomyphan/anytime-mapf): reference for online heuristic/size bandits;
  originally focused on feasible-solution anytime optimization.
- [ADDRESS](https://github.com/JimyZ13/ADDRESS): reference for restricted Thompson Sampling over delayed
  seed agents; also originally focused on anytime optimization.
- [NNS](https://github.com/mit-wu-lab/mapf_neural_neighborhood_search): candidate-ranking baseline and
  data-design reference.
- [LNS2+RL](https://github.com/marmotlab/LNS2-RL): low-level MARL repair baseline, not a high-level
  neighborhood selector.
- [Unified LNS benchmark](https://github.com/ChristinaTan0704/mapf-lns-benchmark): evaluation reference;
  no top-level license was found during review, so its Python implementation was not copied.

No author code release was found for DROP-LNS or DiffLNS during the review. They remain paper-level
parallel-search and learned-initializer comparisons rather than imported dependencies.

## Evaluation contract

Primary metrics: fixed-budget success, time-to-feasible, and conflict-pair AUC. Secondary metrics:
low-level generated/expanded nodes, repair iterations, neighborhood size, and feasible solution cost.

Required baselines are official Adaptive, fixed Target/Collision/Random, random valid actions, a
contextual bandit, an ADDRESS-inspired seed policy adapted to InitLNS, supervised ranking, and contextual
RL. All methods use identical instances, seeds, time budgets, low-level repair, and failure accounting.

The trusted baseline and first experience-collection infrastructure are implemented. The active pilot
uses transfer-aware ID, unseen-layout, unseen-task, unseen-density, and joint-OOD splits; model training
starts only after collection quality and replay determinism are verified.
