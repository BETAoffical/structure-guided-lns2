#include "InitLNS.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

void requireTimings(const RepairTransition& transition)
{
    require(std::isfinite(transition.native_step_seconds) &&
                transition.native_step_seconds >= 0,
            "native step timing is invalid");
    require(transition.neighborhood_generation_seconds >= 0 &&
                transition.replan_seconds >= 0 &&
                transition.pp_replan_seconds >= 0 &&
                transition.state_snapshot_seconds >= 0 &&
                transition.repair_bookkeeping_seconds >= 0 &&
                transition.native_residual_seconds >= 0,
            "native timing partition contains a negative value");
    require(transition.pp_replan_seconds <= transition.replan_seconds + 1e-6,
            "PP timing exceeds total replanning time");
    const double partition = transition.neighborhood_generation_seconds +
        transition.replan_seconds + transition.state_snapshot_seconds +
        transition.repair_bookkeeping_seconds + transition.native_residual_seconds;
    const double tolerance = std::max(1e-6, 0.01 * transition.native_step_seconds);
    require(std::abs(partition - transition.native_step_seconds) <= tolerance,
            "native timing partition does not close");
}

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
    require(solver.initialize(), "failed to initialize deterministic snapshot");
    RepairState state = solver.getRepairState();
    Snapshot snapshot;
    snapshot.conflicts = state.num_of_colliding_pairs;
    for (const auto& agent : state.agents)
        snapshot.paths.push_back(agent.path);
    return snapshot;
}

bool sameState(const RepairState& left, const RepairState& right);

Snapshot stepWithActionSeed(int solver_seed, int action_seed)
{
    constexpr int AGENT_COUNT = 100;
    Instance instance(PROPOSAL_TEST_MAP, PROPOSAL_TEST_SCEN, AGENT_COUNT);
    vector<Agent> agents;
    agents.reserve(AGENT_COUNT);
    for (int id = 0; id < AGENT_COUNT; id++)
        agents.emplace_back(instance, id, true);
    srand(solver_seed);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, nullptr, 1);
    require(solver.initialize(), "failed to initialize proposal test source");
    RepairState state = solver.getRepairState();
    if (state.conflict_edges.empty())
        throw std::runtime_error("proposal test source unexpectedly has no conflicts");

    RepairAction action;
    action.mode = RepairActionMode::SEED;
    action.seed_agent = state.conflict_edges.front().first;
    action.neighborhood_size = 8;
    const RepairState before = solver.getRepairState();
    vector<int> collision_neighborhood;
    const vector<RepairHeuristic> heuristics = {
        RepairHeuristic::TARGET,
        RepairHeuristic::COLLISION,
        RepairHeuristic::RANDOM
    };
    for (size_t index = 0; index < heuristics.size(); index++)
    {
        action.heuristic = heuristics[index];
        action.random_seed = action_seed + (int)index;
        const RepairProposal first = solver.proposeNeighborhood(action);
        const RepairProposal second = solver.proposeNeighborhood(action);
        require(first.action_valid && first.generated && !first.neighborhood.empty(),
                "valid fixed-heuristic proposal was not generated");
        require(first.neighborhood == second.neighborhood,
                "proposal is not deterministic for a fixed random seed");
        require(sameState(before, solver.getRepairState()),
                "proposal changed the logical repair state");
        require(solver.getLastTransition().iteration == 0,
                "proposal replaced the last repair transition");
        if (heuristics[index] == RepairHeuristic::COLLISION)
            collision_neighborhood = first.neighborhood;
    }
    action.heuristic = RepairHeuristic::COLLISION;
    action.random_seed = action_seed + 1;
    require(solver.step(action), "proposal comparison step did not execute");
    require(solver.getLastTransition().neighborhood == collision_neighborhood,
            "proposal neighborhood differs from a seeded repair step");

    Snapshot snapshot;
    state = solver.getRepairState();
    snapshot.conflicts = state.num_of_colliding_pairs;
    snapshot.neighborhood = solver.getLastTransition().neighborhood;
    require(solver.getLastTransition().requested_action.random_seed == action_seed + 1,
            "repair step did not retain the requested random seed");
    for (const auto& agent : state.agents)
        snapshot.paths.push_back(agent.path);
    return snapshot;
}

Snapshot stepWithExplicitOrder(int solver_seed, int action_seed)
{
    constexpr int AGENT_COUNT = 100;
    Instance instance(PROPOSAL_TEST_MAP, PROPOSAL_TEST_SCEN, AGENT_COUNT);
    vector<Agent> agents;
    agents.reserve(AGENT_COUNT);
    for (int id = 0; id < AGENT_COUNT; id++)
        agents.emplace_back(instance, id, true);
    srand(solver_seed);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, nullptr, 1);
    require(solver.initialize(), "failed to initialize explicit-order source");
    RepairState state = solver.getRepairState();
    require(!state.conflict_edges.empty(), "explicit-order source has no conflicts");

    RepairAction proposal_action;
    proposal_action.mode = RepairActionMode::SEED;
    proposal_action.heuristic = RepairHeuristic::COLLISION;
    proposal_action.seed_agent = state.conflict_edges.front().first;
    proposal_action.neighborhood_size = 8;
    proposal_action.random_seed = action_seed;
    const RepairProposal proposal = solver.proposeNeighborhood(proposal_action);
    require(proposal.action_valid && proposal.generated, "explicit-order proposal failed");

    RepairAction action;
    action.mode = RepairActionMode::EXPLICIT_NEIGHBORHOOD;
    action.agents = proposal.neighborhood;
    action.repair_order = proposal.neighborhood;
    std::reverse(action.repair_order.begin(), action.repair_order.end());
    action.random_seed = action_seed + 1;
    require(solver.step(action), "explicit-order step did not execute");
    const RepairTransition& transition = solver.getLastTransition();
    require(transition.action_valid, "valid explicit repair order was rejected");
    require(transition.repair_order == action.repair_order,
            "actual PP repair order differs from the request");

    Snapshot snapshot;
    state = solver.getRepairState();
    snapshot.conflicts = state.num_of_colliding_pairs;
    snapshot.neighborhood = transition.neighborhood;
    for (const auto& agent : state.agents)
        snapshot.paths.push_back(agent.path);
    return snapshot;
}

void requireInvalidOrderFallback(int solver_seed, int action_seed)
{
    constexpr int AGENT_COUNT = 100;
    Instance instance(PROPOSAL_TEST_MAP, PROPOSAL_TEST_SCEN, AGENT_COUNT);
    vector<Agent> agents;
    agents.reserve(AGENT_COUNT);
    for (int id = 0; id < AGENT_COUNT; id++)
        agents.emplace_back(instance, id, true);
    srand(solver_seed);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, nullptr, 1);
    require(solver.initialize(), "failed to initialize invalid-order source");
    RepairState state = solver.getRepairState();
    require(!state.conflict_edges.empty(), "invalid-order source has no conflicts");

    RepairAction action;
    action.mode = RepairActionMode::EXPLICIT_NEIGHBORHOOD;
    action.agents = {state.conflict_edges.front().first, state.conflict_edges.front().second};
    action.repair_order = {action.agents.front()};
    action.random_seed = action_seed;
    require(solver.step(action), "invalid explicit-order fallback did not execute");
    require(!solver.getLastTransition().action_valid, "incomplete repair order was accepted");
}

bool sameState(const RepairState& left, const RepairState& right)
{
    if (left.initialized != right.initialized ||
        left.initial_solution_complete != right.initial_solution_complete ||
        left.feasible != right.feasible || left.done != right.done ||
        left.iteration != right.iteration || left.rows != right.rows ||
        left.cols != right.cols || left.sum_of_costs != right.sum_of_costs ||
        left.num_of_colliding_pairs != right.num_of_colliding_pairs ||
        left.runtime != right.runtime ||
        left.low_level_expanded != right.low_level_expanded ||
        left.low_level_generated != right.low_level_generated ||
        left.low_level_reopened != right.low_level_reopened ||
        left.low_level_runs != right.low_level_runs ||
        left.obstacles != right.obstacles || left.conflict_edges != right.conflict_edges ||
        left.agents.size() != right.agents.size())
        return false;
    for (size_t i = 0; i < left.agents.size(); i++)
    {
        const auto& a = left.agents[i];
        const auto& b = right.agents[i];
        if (a.id != b.id || a.start != b.start || a.goal != b.goal ||
            a.path_cost != b.path_cost || a.shortest_path_cost != b.shortest_path_cost ||
            a.delay != b.delay || a.conflict_degree != b.conflict_degree || a.path != b.path)
            return false;
    }
    return true;
}

}

int main()
{
    const Snapshot first = initializeWithSeed(7);
    const Snapshot second = initializeWithSeed(7);
    require(first.conflicts == second.conflicts, "reset conflict count is not deterministic");
    require(first.paths == second.paths, "reset paths are not deterministic");

    const Snapshot seeded_first = stepWithActionSeed(0, 12345);
    const Snapshot seeded_second = stepWithActionSeed(0, 12345);
    require(seeded_first.conflicts == seeded_second.conflicts,
            "seeded repair conflict count is not deterministic");
    require(seeded_first.neighborhood == seeded_second.neighborhood,
            "seeded repair neighborhood is not deterministic");
    require(seeded_first.paths == seeded_second.paths,
            "seeded repair paths are not deterministic");

    const Snapshot ordered_first = stepWithExplicitOrder(0, 23456);
    const Snapshot ordered_second = stepWithExplicitOrder(0, 23456);
    require(ordered_first.conflicts == ordered_second.conflicts,
            "explicit repair order conflict count is not deterministic");
    require(ordered_first.neighborhood == ordered_second.neighborhood,
            "explicit repair order changed the neighborhood");
    require(ordered_first.paths == ordered_second.paths,
            "explicit repair order paths are not deterministic");
    requireInvalidOrderFallback(0, 34567);

    Instance instance(TEST_MAP, TEST_SCEN, 80);
    vector<Agent> agents;
    agents.reserve(80);
    for (int id = 0; id < 80; id++)
        agents.emplace_back(instance, id, true);
    CountingObserver observer;
    srand(11);
    InitLNS solver(instance, agents, 30, "PP", "Adaptive", 8, 0, nullptr, &observer, 2);
    require(solver.initialize(), "failed to initialize observer test source");
    require(observer.initial == 1, "initial observer callback count is wrong");
    RepairState initial = solver.getRepairState();
    require(initial.initialized, "initialized state flag is false");
    require(initial.initial_solution_complete, "initial solution is incomplete");
    require(initial.agents.size() == 80, "initial state has the wrong agent count");
    require((int)initial.conflict_edges.size() == initial.num_of_colliding_pairs,
            "initial conflict edge count is inconsistent");

    if (!initial.feasible)
    {
        RepairAction invalid;
        invalid.mode = RepairActionMode::SEED;
        invalid.heuristic = RepairHeuristic::COLLISION;
        invalid.seed_agent = 1000;
        invalid.neighborhood_size = 8;
        require(solver.step(invalid), "invalid-action fallback step did not execute");
        require(!solver.getLastTransition().action_valid,
                "invalid seed action was accepted");
        requireTimings(solver.getLastTransition());
        require(observer.transitions == 1, "invalid fallback transition was not observed");
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
        require(solver.step(explicit_action), "explicit-neighborhood step did not execute");
        require(solver.getLastTransition().action_valid,
                "explicit neighborhood was rejected");
        require(solver.getLastTransition().neighborhood.size() == 2,
                "explicit neighborhood size changed");
        requireTimings(solver.getLastTransition());
    }

    require(solver.getRepairState().iteration <= 2, "repair iteration limit was exceeded");
    std::cout << "repair interface tests passed" << std::endl;
    return 0;
}
