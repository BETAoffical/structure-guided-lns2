// Built only into the explicitly opted-in PBS diagnostic library.
#include "InitLNS.h"
#include "PBS.h"

bool InitLNS::runPBSDiagnostic(RepairTransition& transition)
{
    vector<SingleAgentSolver*> engines;
    vector<Path> old_paths;
    for (int id : neighbor.agents)
    {
        engines.push_back(agents[id].path_planner);
        old_paths.push_back(agents[id].path);
    }
    auto rebuild = [&]()
    {
        path_table.reset();
        for (auto& agent : agents) path_table.insertPath(agent.id, agent.path);
    };
    auto rollback = [&]()
    {
        for (size_t i = 0; i < neighbor.agents.size(); ++i) agents[neighbor.agents[i]].path = old_paths[i];
        rebuild();
        neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        neighbor.colliding_pairs = neighbor.old_colliding_pairs;
    };
    try
    {
        PBS pbs(engines, path_table, screen - 1);
        pbs.diagnostic = true;
        const auto& action = transition.requested_action;
        if (action.pbs_warm_root)
        {
            vector<const Path*> paths;
            for (const auto& path : old_paths) paths.push_back(&path);
            pbs.setInitialPath(paths);
        }
        const double remaining = max(0.0, time_limit - std::chrono::duration<double>(Time::now() - start_time).count());
        const bool solved = pbs.solve(min(action.pbs_seconds, remaining), action.pbs_node_limit,
                                      neighbor.old_colliding_pairs.size());
        transition.pbs_stop_reason = pbs.diagnostic_stop_reason;
        transition.pbs_expanded = pbs.num_HL_expanded;
        transition.pbs_generated = pbs.num_HL_generated;
        transition.pbs_low_level_calls = pbs.diagnostic_low_level_calls;
        transition.pbs_root_conflicts = pbs.diagnostic_root_conflicts;
        transition.pbs_root_seconds = pbs.diagnostic_root_seconds;
        transition.pbs_best_conflicts = pbs.best_node == nullptr ? -1 : pbs.best_node->getCollidingPairs();
        if (solved && pbs.best_node != nullptr &&
            pbs.best_node->getCollidingPairs() < static_cast<int>(neighbor.old_colliding_pairs.size()))
        {
            neighbor.colliding_pairs.clear();
            neighbor.sum_of_costs = 0;
            for (size_t i = 0; i < neighbor.agents.size(); ++i)
            {
                const int id = neighbor.agents[i];
                if (pbs.paths[i] == nullptr || pbs.paths[i]->empty()) throw std::runtime_error("PBS exported empty path");
                agents[id].path = *pbs.paths[i];
                updateCollidingPairs(neighbor.colliding_pairs, id, agents[id].path);
                path_table.insertPath(id);
                neighbor.sum_of_costs += static_cast<int>(agents[id].path.size()) - 1;
            }
            if (neighbor.colliding_pairs.size() != static_cast<size_t>(pbs.best_node->getCollidingPairs()))
                throw std::runtime_error("PBS exported conflict count mismatch");
            return true;
        }
        rollback();
        ++num_of_failures;
        return false;
    }
    catch (...)
    {
        rollback();
        throw;
    }
}
