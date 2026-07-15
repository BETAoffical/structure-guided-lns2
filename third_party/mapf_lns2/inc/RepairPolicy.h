#pragma once

#include "common.h"

#include <cstdint>

enum class RepairActionMode
{
    OFFICIAL,
    SEED,
    EXPLICIT_NEIGHBORHOOD
};

enum class RepairHeuristic
{
    ADAPTIVE,
    TARGET,
    COLLISION,
    RANDOM
};

struct RepairAction
{
    RepairActionMode mode = RepairActionMode::OFFICIAL;
    RepairHeuristic heuristic = RepairHeuristic::ADAPTIVE;
    int seed_agent = -1;
    int neighborhood_size = 0;
    int random_seed = -1;
    vector<int> agents;
};

struct RepairProposal
{
    RepairAction requested_action;
    RepairHeuristic applied_heuristic = RepairHeuristic::COLLISION;
    vector<int> neighborhood;
    bool action_valid = false;
    bool generated = false;
};

struct RepairAgentState
{
    int id = -1;
    int start = -1;
    int goal = -1;
    int path_cost = -1;
    int shortest_path_cost = -1;
    int delay = -1;
    int conflict_degree = 0;
    vector<int> path;
};

struct RepairState
{
    bool initialized = false;
    bool initial_solution_complete = false;
    bool feasible = false;
    bool done = false;
    int iteration = 0;
    int rows = 0;
    int cols = 0;
    int sum_of_costs = 0;
    int num_of_colliding_pairs = 0;
    double runtime = 0;
    uint64_t low_level_expanded = 0;
    uint64_t low_level_generated = 0;
    uint64_t low_level_reopened = 0;
    uint64_t low_level_runs = 0;
    vector<unsigned char> obstacles;
    vector<pair<int, int>> conflict_edges;
    vector<RepairAgentState> agents;
};

struct RepairTransition
{
    RepairAction requested_action;
    RepairHeuristic applied_heuristic = RepairHeuristic::COLLISION;
    vector<int> neighborhood;
    bool action_valid = true;
    bool generated = false;
    bool replan_success = false;
    int iteration = 0;
    int conflicts_before = 0;
    int conflicts_after = 0;
    int sum_of_costs_before = 0;
    int sum_of_costs_after = 0;
    double runtime_before = 0;
    double runtime_after = 0;
};

class NeighborhoodPolicy
{
public:
    virtual ~NeighborhoodPolicy() = default;
    virtual RepairAction choose(const RepairState& state) = 0;
};

class RepairObserver
{
public:
    virtual ~RepairObserver() = default;
    virtual void onInitialState(const RepairState&) {}
    virtual void onTransition(const RepairState&, const RepairTransition&, const RepairState&) {}
    virtual void onFinish(const RepairState&, bool) {}
};

inline const char* repairActionModeName(RepairActionMode mode)
{
    switch (mode)
    {
        case RepairActionMode::OFFICIAL: return "official";
        case RepairActionMode::SEED: return "seed";
        case RepairActionMode::EXPLICIT_NEIGHBORHOOD: return "explicit_neighborhood";
    }
    return "unknown";
}

inline const char* repairHeuristicName(RepairHeuristic heuristic)
{
    switch (heuristic)
    {
        case RepairHeuristic::ADAPTIVE: return "adaptive";
        case RepairHeuristic::TARGET: return "target";
        case RepairHeuristic::COLLISION: return "collision";
        case RepairHeuristic::RANDOM: return "random";
    }
    return "unknown";
}
