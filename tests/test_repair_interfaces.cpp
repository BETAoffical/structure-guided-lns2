#include "InitLNS.h"

#include <cassert>
#include <iostream>

namespace
{
struct CountingObserver : public RepairObserver
{
    int initial = 0;
    int transitions = 0;
    int finished = 0;

    void onInitialState(const RepairState&) override { initial++; }
    void onTransition(const RepairState&, const RepairTransition&, const RepairState&) override
    {
        transitions++;
    }
    void onFinish(const RepairState&, bool) override { finished++; }
};

struct Snapshot
{
    int conflicts = 0;
    vector<vector<int>> paths;
    vector<int> neighborhood;
};

Snapshot initializeWithSeed(int seed)
{
    Instance instance(TEST_MAP, TEST_SCEN, 80);
    vector<Agent> agents;
    agents.reserve(80);
    for (int id = 0; id < 80; id++)
        agents.emplace_back(instance, id, true);
    srand(seed);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, nullptr, 1);
    assert(solver.initialize());
    RepairState state = solver.getRepairState();
    Snapshot snapshot;
    snapshot.conflicts = state.num_of_colliding_pairs;
    for (const auto& agent : state.agents)
        snapshot.paths.push_back(agent.path);
    return snapshot;
}

Snapshot stepWithActionSeed(int solver_seed, int action_seed)
{
    Instance instance(TEST_MAP, TEST_SCEN, 200);
    vector<Agent> agents;
    agents.reserve(200);
    for (int id = 0; id < 200; id++)
        agents.emplace_back(instance, id, true);
    srand(solver_seed);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, nullptr, 1);
    assert(solver.initialize());
    RepairState state = solver.getRepairState();
    assert(!state.conflict_edges.empty());

    RepairAction action;
    action.mode = RepairActionMode::SEED;
    action.heuristic = RepairHeuristic::COLLISION;
    action.seed_agent = state.conflict_edges.front().first;
    action.neighborhood_size = 8;
    action.random_seed = action_seed;
    assert(solver.step(action));

    Snapshot snapshot;
    state = solver.getRepairState();
    snapshot.conflicts = state.num_of_colliding_pairs;
    snapshot.neighborhood = solver.getLastTransition().neighborhood;
    assert(solver.getLastTransition().requested_action.random_seed == action_seed);
    for (const auto& agent : state.agents)
        snapshot.paths.push_back(agent.path);
    return snapshot;
}
}

int main()
{
    const Snapshot first = initializeWithSeed(7);
    const Snapshot second = initializeWithSeed(7);
    assert(first.conflicts == second.conflicts);
    assert(first.paths == second.paths);

    const Snapshot seeded_first = stepWithActionSeed(29, 12345);
    const Snapshot seeded_second = stepWithActionSeed(29, 12345);
    assert(seeded_first.conflicts == seeded_second.conflicts);
    assert(seeded_first.neighborhood == seeded_second.neighborhood);
    assert(seeded_first.paths == seeded_second.paths);

    Instance instance(TEST_MAP, TEST_SCEN, 80);
    vector<Agent> agents;
    agents.reserve(80);
    for (int id = 0; id < 80; id++)
        agents.emplace_back(instance, id, true);
    CountingObserver observer;
    srand(11);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, &observer, 2);
    assert(solver.initialize());
    assert(observer.initial == 1);
    RepairState initial = solver.getRepairState();
    assert(initial.initialized);
    assert(initial.initial_solution_complete);
    assert(initial.agents.size() == 80);
    assert((int)initial.conflict_edges.size() == initial.num_of_colliding_pairs);

    if (!initial.feasible)
    {
        RepairAction invalid;
        invalid.mode = RepairActionMode::SEED;
        invalid.heuristic = RepairHeuristic::COLLISION;
        invalid.seed_agent = 1000;
        invalid.neighborhood_size = 8;
        assert(solver.step(invalid));
        assert(!solver.getLastTransition().action_valid);
        assert(observer.transitions == 1);
    }

    RepairState current = solver.getRepairState();
    if (!current.done && !current.conflict_edges.empty())
    {
        RepairAction explicit_action;
        explicit_action.mode = RepairActionMode::EXPLICIT_NEIGHBORHOOD;
        explicit_action.agents = {
            current.conflict_edges.front().first,
            current.conflict_edges.front().second
        };
        assert(solver.step(explicit_action));
        assert(solver.getLastTransition().action_valid);
        assert(solver.getLastTransition().neighborhood.size() == 2);
    }

    assert(solver.getRepairState().iteration <= 2);
    std::cout << "repair interface tests passed" << std::endl;
    return 0;
}
