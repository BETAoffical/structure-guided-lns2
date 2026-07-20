#include "InitLNS.h"
#include <queue>
#include <algorithm>
#include "GCBS.h"
#include "PBS.h"

namespace
{
class ProposalNeighborScope
{
public:
    explicit ProposalNeighborScope(Neighbor& value) : target(value)
    {
        using std::swap;
        swap(target, scratch);
    }

    ~ProposalNeighborScope()
    {
        using std::swap;
        swap(target, scratch);
    }

    ProposalNeighborScope(const ProposalNeighborScope&) = delete;
    ProposalNeighborScope& operator=(const ProposalNeighborScope&) = delete;

private:
    Neighbor& target;
    Neighbor scratch;
};
}

InitLNS::InitLNS(const Instance& instance, vector<Agent>& agents, double time_limit,
         const string & replan_algo_name, const string & init_destory_name, int neighbor_size, int screen,
         NeighborhoodPolicy* policy, RepairObserver* observer, int max_repair_iterations) :
         BasicLNS(instance, time_limit, neighbor_size, screen), agents(agents), replan_algo_name(replan_algo_name),
         path_table(instance.map_size, agents.size()), collision_graph(agents.size()), goal_table(instance.map_size, -1),
         policy(policy), observer(observer), max_repair_iterations(max_repair_iterations)
 {
     replan_time_limit = time_limit;
     if (init_destory_name == "Adaptive")
     {
         ALNS = true;
         destroy_weights.assign(INIT_COUNT, 1);
         decay_factor = 0.05;
         reaction_factor = 0.05;
     }
     else if (init_destory_name == "Target")
         init_destroy_strategy = TARGET_BASED;
     else if (init_destory_name == "Collision")
         init_destroy_strategy = COLLISION_BASED;
     else if (init_destory_name == "Random")
         init_destroy_strategy = RANDOM_BASED;
     else
     {
         cerr << "Init Destroy heuristic " << init_destory_name << " does not exists. " << endl;
         exit(-1);
     }

     for (auto& i:agents) {
         goal_table[i.path_planner->goal_location] = i.id;
     }

 }

bool InitLNS::initialize()
{
    if (initialized)
        return initial_solution_complete;
    start_time = Time::now();
    initial_solution_complete = getInitialSolution();
    runtime = ((fsec)(Time::now() - start_time)).count();
    iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, "PP", 0, num_of_colliding_pairs);
    initialized = true;
    if (screen >= 3)
        printPath();
    if (screen >= 1)
        cout << "Iteration " << iteration_stats.size() << ", "
             << "group size = " << neighbor.agents.size() << ", "
             << "colliding pairs = " << num_of_colliding_pairs << ", "
             << "solution cost = " << sum_of_costs << ", "
             << "remaining time = " << time_limit - runtime << endl;
    if (observer != nullptr)
        observer->onInitialState(getRepairState());
    if (isDone())
        notifyFinish();
    return initial_solution_complete;
}

bool InitLNS::run()
{
    initialize();
    while (!isDone())
        step();
    printResult();
    notifyFinish();
    return isFeasible();
}

bool InitLNS::step()
{
    RepairAction action;
    if (policy != nullptr)
        action = policy->choose(getRepairState());
    return step(action);
}

RepairProposal InitLNS::proposeNeighborhood(const RepairAction& action)
{
    RepairProposal proposal;
    proposal.requested_action = action;
    proposal.applied_heuristic = action.heuristic;
    if (!initialized || isDone() || action.mode != RepairActionMode::SEED ||
        action.heuristic == RepairHeuristic::ADAPTIVE || action.random_seed < 0 ||
        action.neighborhood_size <= 0 || action.seed_agent < 0 ||
        action.seed_agent >= (int)agents.size() || collision_graph[action.seed_agent].empty())
        return proposal;

    // Proposal generation only writes neighbor.agents. Swap in an empty
    // scratch Neighbor so hundreds of diagnostic proposals do not deep-copy
    // old paths and conflict sets. The formal repair Neighbor is restored
    // even if proposal generation throws.
    ProposalNeighborScope restore_neighbor(neighbor);
    srand(action.random_seed);
    bool generated = false;
    switch (action.heuristic)
    {
        case RepairHeuristic::TARGET:
            generated = generateNeighborByTarget(action.seed_agent, action.neighborhood_size);
            break;
        case RepairHeuristic::COLLISION:
            generated = generateNeighborByCollisionGraph(action.seed_agent, action.neighborhood_size);
            break;
        case RepairHeuristic::RANDOM:
            generated = generateNeighborRandomly(action.seed_agent, action.neighborhood_size);
            break;
        case RepairHeuristic::ADAPTIVE:
            break;
    }
    proposal.action_valid = true;
    proposal.generated = generated && !neighbor.agents.empty();
    if (proposal.generated)
    {
        proposal.neighborhood = neighbor.agents;
        std::sort(proposal.neighborhood.begin(), proposal.neighborhood.end());
    }
    return proposal;
}

vector<RepairProposal> InitLNS::proposeNeighborhoodBatch(
    const vector<RepairAction>& actions)
{
    proposal_collision_component_cache.clear();
    proposal_target_cache.clear();
    proposal_batch_cache_active = true;
    vector<RepairProposal> proposals;
    proposals.reserve(actions.size());
    try
    {
        for (const RepairAction& action : actions)
            proposals.push_back(proposeNeighborhood(action));
    }
    catch (...)
    {
        proposal_batch_cache_active = false;
        proposal_collision_component_cache.clear();
        proposal_target_cache.clear();
        throw;
    }
    proposal_batch_cache_active = false;
    proposal_collision_component_cache.clear();
    proposal_target_cache.clear();
    return proposals;
}

bool InitLNS::step(const RepairAction& action)
{
    const auto native_step_started = Time::now();
    if (!initialized)
        initialize();
    if (isDone())
    {
        notifyFinish();
        return false;
    }

    vector<Path*> paths(agents.size());
    for (size_t i = 0; i < agents.size(); i++)
        paths[i] = &agents[i].path;
    assert(instance.validateSolution(paths, sum_of_costs, num_of_colliding_pairs));

    auto state_snapshot_started = Time::now();
    RepairState before = getRepairState();
    double state_snapshot_seconds =
        ((fsec)(Time::now() - state_snapshot_started)).count();
    RepairTransition transition;
    transition.requested_action = action;
    transition.iteration = repair_iteration + 1;
    transition.conflicts_before = num_of_colliding_pairs;
    transition.sum_of_costs_before = sum_of_costs;
    transition.runtime_before = runtime;

    if (action.random_seed >= 0)
        srand(action.random_seed);

    bool action_valid = true;
    RepairHeuristic applied_heuristic = currentRepairHeuristic();
    const auto neighborhood_started = Time::now();
    bool succ = generateNeighborhood(action, applied_heuristic, action_valid);
    transition.neighborhood_generation_seconds =
        ((fsec)(Time::now() - neighborhood_started)).count();
    repair_iteration++;
    transition.action_valid = action_valid;
    transition.applied_heuristic = applied_heuristic;
    transition.generated = succ && !neighbor.agents.empty();
    transition.neighborhood = neighbor.agents;

    const bool update_alns = ALNS &&
        (action.mode == RepairActionMode::OFFICIAL || !action_valid ||
         (action.mode == RepairActionMode::SEED && action.heuristic == RepairHeuristic::ADAPTIVE));

    auto finishTransition = [&]()
    {
        state_snapshot_started = Time::now();
        RepairState after = getRepairState();
        state_snapshot_seconds +=
            ((fsec)(Time::now() - state_snapshot_started)).count();
        transition.state_snapshot_seconds = state_snapshot_seconds;
        transition.native_step_seconds =
            ((fsec)(Time::now() - native_step_started)).count();
        transition.native_residual_seconds = std::max(
            0.0,
            transition.native_step_seconds -
                transition.neighborhood_generation_seconds -
                transition.replan_seconds - transition.state_snapshot_seconds);
        transition.native_residual_seconds = std::max(
            0.0,
            transition.native_residual_seconds -
                transition.repair_bookkeeping_seconds);
        last_transition = transition;
        if (observer != nullptr)
            observer->onTransition(before, transition, after);
        if (isDone())
            notifyFinish();
    };

    if (!transition.generated)
    {
        runtime = ((fsec)(Time::now() - start_time)).count();
        transition.conflicts_after = num_of_colliding_pairs;
        transition.sum_of_costs_after = sum_of_costs;
        transition.runtime_after = runtime;
        finishTransition();
        return true;
    }

    auto bookkeeping_started = Time::now();
    neighbor.old_colliding_pairs.clear();
    for (int a : neighbor.agents)
    {
        for (auto j: collision_graph[a])
        {
            neighbor.old_colliding_pairs.emplace(min(a, j), max(a, j));
        }
    }
    if (neighbor.old_colliding_pairs.empty())
    {
        if (update_alns)
            destroy_weights[selected_neighbor] = (1 - decay_factor) * destroy_weights[selected_neighbor];
        transition.repair_bookkeeping_seconds +=
            ((fsec)(Time::now() - bookkeeping_started)).count();
        runtime = ((fsec)(Time::now() - start_time)).count();
        transition.conflicts_after = num_of_colliding_pairs;
        transition.sum_of_costs_after = sum_of_costs;
        transition.runtime_after = runtime;
        finishTransition();
        return true;
    }

    neighbor.old_paths.resize(neighbor.agents.size());
    neighbor.old_sum_of_costs = 0;
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        int a = neighbor.agents[i];
        if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
            neighbor.old_paths[i] = agents[a].path;
        path_table.deletePath(neighbor.agents[i]);
        neighbor.old_sum_of_costs += (int) agents[a].path.size() - 1;
    }
    if (screen >= 2)
    {
        cout << "Neighbors: ";
        for (auto a : neighbor.agents)
            cout << a << ", ";
        cout << endl;
        cout << "Old colliding pairs (" << neighbor.old_colliding_pairs.size() << "): ";
        for (const auto & p : neighbor.old_colliding_pairs)
        {
            cout << "(" << p.first << "," << p.second << "), ";
        }
        cout << endl;
    }
    transition.repair_bookkeeping_seconds +=
        ((fsec)(Time::now() - bookkeeping_started)).count();

    const auto replan_started = Time::now();
    if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
    {
        const vector<int> requested_order = action_valid ? action.repair_order : vector<int>();
        succ = runPP(requested_order, transition.repair_order);
    }
    else if (replan_algo_name == "GCBS")
        succ = runGCBS();
    else if (replan_algo_name == "PBS")
        succ = runPBS();
    else
    {
        cerr << "Wrong replanning strategy" << endl;
        exit(-1);
    }
    transition.replan_seconds =
        ((fsec)(Time::now() - replan_started)).count();
    if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
        transition.pp_replan_seconds = transition.replan_seconds;
    transition.replan_success = succ;

    bookkeeping_started = Time::now();
    if (update_alns)
    {
        if (neighbor.colliding_pairs.size() < neighbor.old_colliding_pairs.size())
            destroy_weights[selected_neighbor] =
                    reaction_factor * (double)(neighbor.old_colliding_pairs.size() -
                    neighbor.colliding_pairs.size())
                    + (1 - reaction_factor) * destroy_weights[selected_neighbor];
        else
            destroy_weights[selected_neighbor] =
                    (1 - decay_factor) * destroy_weights[selected_neighbor];
    }
    if (screen >= 2)
        cout << "New colliding pairs = " << neighbor.colliding_pairs.size() << endl;
    if (succ)
    {
        num_of_colliding_pairs += (int)neighbor.colliding_pairs.size() - (int)neighbor.old_colliding_pairs.size();
        for(const auto& agent_pair : neighbor.old_colliding_pairs)
        {
            collision_graph[agent_pair.first].erase(agent_pair.second);
            collision_graph[agent_pair.second].erase(agent_pair.first);
        }
        for(const auto& agent_pair : neighbor.colliding_pairs)
        {
            collision_graph[agent_pair.first].emplace(agent_pair.second);
            collision_graph[agent_pair.second].emplace(agent_pair.first);
        }
        if (screen >= 2)
            printCollisionGraph();
    }

    runtime = ((fsec)(Time::now() - start_time)).count();
    sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
    if (screen >= 1)
        cout << "Iteration " << iteration_stats.size() << ", "
             << "group size = " << neighbor.agents.size() << ", "
             << "colliding pairs = " << num_of_colliding_pairs << ", "
             << "solution cost = " << sum_of_costs << ", "
             << "remaining time = " << time_limit - runtime << endl;
    iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, replan_algo_name,
                                 0, num_of_colliding_pairs);

    transition.conflicts_after = num_of_colliding_pairs;
    transition.sum_of_costs_after = sum_of_costs;
    transition.runtime_after = runtime;
    transition.repair_bookkeeping_seconds +=
        ((fsec)(Time::now() - bookkeeping_started)).count();
    finishTransition();
    return true;
}
bool InitLNS::runGCBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    // build path tables
    vector<PathTable> path_tables(neighbor.agents.size(), PathTable(instance.map_size));
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        int agent_id = neighbor.agents[i];
        for (int j = 0; j < instance.getDefaultNumberOfAgents(); j++)
        {
            if (j != agent_id and collision_graph[agent_id].count(j) == 0)
                path_tables[i].insertPath(j, agents[j].path);
        }
    }

    GCBS gcbs(search_engines, screen - 1, &path_tables);
    gcbs.setDisjointSplitting(false);
    gcbs.setBypass(true);
    gcbs.setTargetReasoning(true);

    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    gcbs.solve(T);
    if (gcbs.best_node->colliding_pairs < (int) neighbor.old_colliding_pairs.size()) // accept new paths
    {
        auto id = neighbor.agents.begin();
        neighbor.colliding_pairs.clear();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *gcbs.paths[i];
            updateCollidingPairs(neighbor.colliding_pairs, agents[*id].id, agents[*id].path);
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = gcbs.best_node->sum_of_costs;
        return true;
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        num_of_failures++;
        return false;
    }
}
bool InitLNS::runPBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    vector<const Path*> initial_paths;
    initial_paths.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
        initial_paths.push_back(&agents[i].path);
    }

    PBS pbs(search_engines, path_table, screen - 1);
    // pbs.setInitialPath(initial_paths);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = pbs.solve(T, (int)neighbor.agents.size(), neighbor.old_colliding_pairs.size());
    if (succ and pbs.best_node->getCollidingPairs() < (int) neighbor.old_colliding_pairs.size()) // accept new paths
    {
        auto id = neighbor.agents.begin();
        neighbor.colliding_pairs.clear();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *pbs.paths[i];
            updateCollidingPairs(neighbor.colliding_pairs, agents[*id].id, agents[*id].path);
            path_table.insertPath(agents[*id].id);
            ++id;
        }
        assert(neighbor.colliding_pairs.size() == pbs.best_node->getCollidingPairs());
        neighbor.sum_of_costs = pbs.best_node->sum_of_costs;
        return true;
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        num_of_failures++;
        return false;
    }
}
bool InitLNS::runPP(const vector<int>& requested_order, vector<int>& applied_order)
{
    vector<int> shuffled_agents;
    if (requested_order.empty())
    {
        shuffled_agents = neighbor.agents;
        std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
    }
    else
        shuffled_agents = requested_order;
    applied_order = shuffled_agents;
    if (screen >= 2) {
        cout<<"Neighbors_set: ";
        for (auto id : shuffled_agents)
            cout << id << ", ";
        cout << endl;
    }
    int remaining_agents = (int)shuffled_agents.size();
    auto p = shuffled_agents.begin();
    neighbor.sum_of_costs = 0;
    neighbor.colliding_pairs.clear();
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = min(time_limit - runtime, replan_time_limit);
    auto time = Time::now();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, nullptr, &path_table);
    while (p != shuffled_agents.end() && ((fsec)(Time::now() - time)).count() < T)
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        assert(agents[id].path_planner->num_collisions > 0 or
            !updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path));
        neighbor.sum_of_costs += (int)agents[id].path.size() - 1;
        remaining_agents--;
        if (screen >= 3)
        {
            runtime = ((fsec)(Time::now() - start_time)).count();
            cout << "After agent " << id << ": Remaining agents = " << remaining_agents <<
                 ", colliding pairs = " << neighbor.colliding_pairs.size() <<
                 ", LL nodes = " << agents[id].path_planner->getNumExpanded() <<
                 ", remaining time = " << time_limit - runtime << " seconds. " << endl;
        }
        if (neighbor.colliding_pairs.size() > neighbor.old_colliding_pairs.size())
            break;
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    if (p == shuffled_agents.end() && neighbor.colliding_pairs.size() <= neighbor.old_colliding_pairs.size()) // accept new paths
    {
        return true;
    }
    else // stick to old paths
    {
        if (p != shuffled_agents.end())
            num_of_failures++;
        auto p2 = shuffled_agents.begin();
        while (p2 != p)
        {
            int a = *p2;
            path_table.deletePath(agents[a].id);
            ++p2;
        }
        if (!neighbor.old_paths.empty())
        {
            p2 = neighbor.agents.begin();
            for (int i = 0; i < (int)neighbor.agents.size(); i++)
            {
                int a = *p2;
                agents[a].path = neighbor.old_paths[i];
                path_table.insertPath(agents[a].id);
                ++p2;
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        return false;
    }
}

bool InitLNS::getInitialSolution()
{
    neighbor.agents.clear();
    neighbor.agents.reserve(agents.size());
    sum_of_costs = 0;
    for (int i = 0; i < (int)agents.size(); i++)
    {
        if (agents[i].path.empty())
            neighbor.agents.push_back(i);
        else
        {
            sum_of_costs += (int)agents[i].path.size() - 1;
            path_table.insertPath(agents[i].id, agents[i].path);
        }
    }
    int remaining_agents = (int)neighbor.agents.size();
    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, nullptr, &path_table);
    set<pair<int, int>> colliding_pairs;
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(colliding_pairs, agents[id].id, agents[id].path);
        sum_of_costs += (int)agents[id].path.size() - 1;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);
        runtime = ((fsec)(Time::now() - start_time)).count();
        if (screen >= 3)
        {
            cout << "After agent " << id << ": Remaining agents = " << remaining_agents <<
                 ", colliding pairs = " << colliding_pairs.size() <<
                 ", LL nodes = " << agents[id].path_planner->getNumExpanded() <<
                 ", remaining time = " << time_limit - runtime << " seconds. " << endl;
        }
        if (runtime > time_limit)
            break;
    }

    num_of_colliding_pairs = colliding_pairs.size();
    for(const auto& agent_pair : colliding_pairs)
    {
        collision_graph[agent_pair.first].emplace(agent_pair.second);
        collision_graph[agent_pair.second].emplace(agent_pair.first);
    }
    if (screen >= 2)
        printCollisionGraph();
    return remaining_agents == 0;
}

// return true if the new p[ath has collisions;
bool InitLNS::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        succ = true;
                        colliding_pairs.emplace(min(agent_id, a1), max(agent_id, a1));
                        break;
                    }
                }
            }
        }
        //auto id = getAgentWithTarget(to, t);
        //if (id >= 0) // this agent traverses the target of another agent
        //    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}

bool InitLNS::isDone() const
{
    if (!initialized)
        return false;
    if (!initial_solution_complete || num_of_colliding_pairs == 0 || runtime >= time_limit)
        return true;
    return max_repair_iterations > 0 && repair_iteration >= max_repair_iterations;
}

RepairState InitLNS::getRepairState() const
{
    RepairState state;
    state.initialized = initialized;
    state.initial_solution_complete = initial_solution_complete;
    state.feasible = initialized && num_of_colliding_pairs == 0;
    state.done = isDone();
    state.iteration = repair_iteration;
    state.rows = instance.num_of_rows;
    state.cols = instance.num_of_cols;
    state.sum_of_costs = sum_of_costs;
    state.num_of_colliding_pairs = num_of_colliding_pairs;
    state.runtime = runtime;
    state.obstacles.reserve(instance.map_size);
    for (int location = 0; location < instance.map_size; location++)
        state.obstacles.push_back(instance.isObstacle(location) ? 1 : 0);

    state.agents.reserve(agents.size());
    for (const auto& agent : agents)
    {
        RepairAgentState agent_state;
        agent_state.id = agent.id;
        agent_state.start = agent.path_planner->start_location;
        agent_state.goal = agent.path_planner->goal_location;
        agent_state.shortest_path_cost =
            agent.path_planner->my_heuristic[agent.path_planner->start_location];
        agent_state.path_cost = agent.path.empty() ? -1 : (int)agent.path.size() - 1;
        agent_state.delay = agent.path.empty() ? -1 : agent.getNumOfDelays();
        agent_state.conflict_degree = collision_graph.empty() ? 0 : (int)collision_graph[agent.id].size();
        agent_state.path.reserve(agent.path.size());
        for (const auto& entry : agent.path)
            agent_state.path.push_back(entry.location);
        state.agents.push_back(std::move(agent_state));

        state.low_level_expanded += agent.path_planner->getTotalNumExpanded();
        state.low_level_generated += agent.path_planner->getTotalNumGenerated();
        state.low_level_reopened += agent.path_planner->getTotalNumReopened();
        state.low_level_runs += agent.path_planner->getTotalNumRuns();
    }
    for (size_t first = 0; first < collision_graph.size(); first++)
    {
        for (int second : collision_graph[first])
        {
            if ((int)first < second)
                state.conflict_edges.emplace_back((int)first, second);
        }
    }
    return state;
}

RepairHeuristic InitLNS::currentRepairHeuristic() const
{
    switch (init_destroy_strategy)
    {
        case TARGET_BASED: return RepairHeuristic::TARGET;
        case COLLISION_BASED: return RepairHeuristic::COLLISION;
        case RANDOM_BASED: return RepairHeuristic::RANDOM;
        default: return RepairHeuristic::COLLISION;
    }
}

void InitLNS::notifyFinish()
{
    if (!finish_notified && observer != nullptr)
        observer->onFinish(getRepairState(), isFeasible());
    finish_notified = true;
}

bool InitLNS::validateExplicitNeighborhood(const vector<int>& values) const
{
    if (values.empty() || values.size() > agents.size())
        return false;
    set<int> unique;
    bool touches_conflict = false;
    for (int agent : values)
    {
        if (agent < 0 || agent >= (int)agents.size() || !unique.insert(agent).second)
            return false;
        touches_conflict = touches_conflict || !collision_graph[agent].empty();
    }
    return touches_conflict;
}

bool InitLNS::validateRepairOrder(const vector<int>& order,
                                  const vector<int>& neighborhood) const
{
    if (order.empty())
        return true;
    if (order.size() != neighborhood.size())
        return false;
    vector<int> ordered = order;
    vector<int> expected = neighborhood;
    std::sort(ordered.begin(), ordered.end());
    std::sort(expected.begin(), expected.end());
    return ordered == expected && std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end();
}

bool InitLNS::generateOfficialNeighborhood(RepairHeuristic& applied_heuristic)
{
    if (ALNS)
        chooseDestroyHeuristicbyALNS();
    applied_heuristic = currentRepairHeuristic();
    switch (init_destroy_strategy)
    {
        case TARGET_BASED: return generateNeighborByTarget();
        case COLLISION_BASED: return generateNeighborByCollisionGraph();
        case RANDOM_BASED: return generateNeighborRandomly();
        default: return false;
    }
}

bool InitLNS::generateNeighborhood(const RepairAction& action, RepairHeuristic& applied_heuristic,
                                   bool& action_valid)
{
    action_valid = true;
    if (!action.repair_order.empty() &&
        (action.mode != RepairActionMode::EXPLICIT_NEIGHBORHOOD || replan_algo_name != "PP"))
    {
        action_valid = false;
        return generateOfficialNeighborhood(applied_heuristic);
    }
    if (action.mode == RepairActionMode::OFFICIAL)
        return generateOfficialNeighborhood(applied_heuristic);

    if (action.mode == RepairActionMode::EXPLICIT_NEIGHBORHOOD)
    {
        if (!validateExplicitNeighborhood(action.agents) ||
            !validateRepairOrder(action.repair_order, action.agents))
        {
            action_valid = false;
            return generateOfficialNeighborhood(applied_heuristic);
        }
        neighbor.agents = action.agents;
        std::sort(neighbor.agents.begin(), neighbor.agents.end());
        applied_heuristic = currentRepairHeuristic();
        return true;
    }

    if (action.seed_agent < 0 || action.seed_agent >= (int)agents.size() ||
        collision_graph[action.seed_agent].empty())
    {
        action_valid = false;
        return generateOfficialNeighborhood(applied_heuristic);
    }

    RepairHeuristic heuristic = action.heuristic;
    if (heuristic == RepairHeuristic::ADAPTIVE)
    {
        if (ALNS)
            chooseDestroyHeuristicbyALNS();
        heuristic = currentRepairHeuristic();
    }
    applied_heuristic = heuristic;
    switch (heuristic)
    {
        case RepairHeuristic::TARGET:
            return generateNeighborByTarget(action.seed_agent, action.neighborhood_size);
        case RepairHeuristic::COLLISION:
            return generateNeighborByCollisionGraph(action.seed_agent, action.neighborhood_size);
        case RepairHeuristic::RANDOM:
            return generateNeighborRandomly(action.seed_agent, action.neighborhood_size);
        case RepairHeuristic::ADAPTIVE:
            break;
    }
    action_valid = false;
    return generateOfficialNeighborhood(applied_heuristic);
}

void InitLNS::chooseDestroyHeuristicbyALNS()
{
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : init_destroy_strategy = TARGET_BASED; break;
        case 1 : init_destroy_strategy = COLLISION_BASED; break;
        case 2 : init_destroy_strategy = RANDOM_BASED; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool InitLNS::generateNeighborByCollisionGraph(int forced_seed, int requested_size)
{
    /*unordered_map<int, list<int>> G;
    for (int i = 0; i < (int)collision_graph.size(); i++)
    {
        if (!collision_graph[i].empty())
            G[i].assign(collision_graph[i].begin(), collision_graph[i].end());
    }
    assert(!G.empty());
    assert(neighbor_size <= (int)agents.size());
    set<int> neighbors_set;
    if ((int)G.size() < neighbor_size)
    {
        for (const auto& node : G)
            neighbors_set.insert(node.first);
        int count = 0;
        while ((int)neighbors_set.size() < neighbor_size && count < 10)
        {
            int a1 = *std::next(neighbors_set.begin(), rand() % neighbors_set.size());
            int a2 = randomWalk(a1);
            if (a2 != NO_AGENT)
                neighbors_set.insert(a2);
            else
                count++;
        }
    }
    else
    {
        int a = -1;
        while ((int)neighbors_set.size() < neighbor_size)
        {
            if (a == -1)
            {
                a = std::next(G.begin(), rand() % G.size())->first;
                neighbors_set.insert(a);
            }
            else
            {
                a = *std::next(G[a].begin(), rand() % G[a].size());
                auto ret = neighbors_set.insert(a);
                if (!ret.second) // no new element inserted
                    a = -1;
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by collision graph" << endl;
    return true;*/

    int target_size = requested_size > 0 ? requested_size : neighbor_size;
    target_size = min((int)agents.size(), max(2, target_size));
    vector<int> all_vertices;
    if (forced_seed < 0)
    {
        all_vertices.reserve(collision_graph.size());
        for (int i = 0; i < (int)collision_graph.size(); i++)
        {
            if (!collision_graph[i].empty())
                all_vertices.push_back(i);
        }
        if (all_vertices.empty())
            return false;
    }
    auto v = forced_seed >= 0 ? forced_seed : all_vertices[rand() % all_vertices.size()];
    unordered_map<int, set<int>> local_component;
    unordered_map<int, set<int>>* component = &local_component;
    if (proposal_batch_cache_active && forced_seed >= 0)
    {
        auto inserted = proposal_collision_component_cache.emplace(
            forced_seed, unordered_map<int, set<int>>()
        );
        component = &inserted.first->second;
        if (inserted.second)
            findConnectedComponent(collision_graph, v, *component);
    }
    else
        findConnectedComponent(collision_graph, v, *component);
    auto& G = *component;
    assert(G.size() > 1);

    set<int> neighbors_set;
    if ((int)G.size() <= target_size)
    {
        for (const auto& node : G)
            neighbors_set.insert(node.first);
        int count = 0;
        while ((int)neighbors_set.size() < target_size && count < 10)
        {
            int a1 = *std::next(neighbors_set.begin(), rand() % neighbors_set.size());
            int a2 = randomWalk(a1);
            if (a2 != NO_AGENT)
                neighbors_set.insert(a2);
            else
                count++;
        }
    }
    else
    {
        int a = forced_seed >= 0 ? forced_seed : std::next(G.begin(), rand() % G.size())->first;
        neighbors_set.insert(a);
        while ((int)neighbors_set.size() < target_size)
        {
            a = *std::next(G[a].begin(), rand() % G[a].size());
            neighbors_set.insert(a);
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by collision graph" << endl;
    return true;

}
bool InitLNS::generateNeighborByTarget(int forced_seed, int requested_size)
{
    int target_size = requested_size > 0 ? requested_size : neighbor_size;
    target_size = min((int)agents.size(), max(2, target_size));
    int a = forced_seed;
    if (a < 0)
    {
        auto r = rand() % (num_of_colliding_pairs * 2);
        int sum = 0;
        for (int i = 0 ; i < (int)collision_graph.size(); i++)
        {
            sum += (int)collision_graph[i].size();
            if (r <= sum and !collision_graph[i].empty())
            {
                a = i;
                break;
            }
        }
    }
    assert(a != -1 and !collision_graph[a].empty());
    ProposalTargetData local_target;
    ProposalTargetData* target_data = &local_target;
    bool populate_target = true;
    if (proposal_batch_cache_active && forced_seed >= 0)
    {
        auto inserted = proposal_target_cache.emplace(
            forced_seed, ProposalTargetData()
        );
        target_data = &inserted.first->second;
        populate_target = inserted.second;
    }
    if (populate_target)
    {
        for(int t = 0 ;t< path_table.table[agents[a].path_planner->start_location].size();t++){
            for(auto id : path_table.table[agents[a].path_planner->start_location][t]){
                if (id!=a)
                    target_data->start_agents.insert(make_pair(t,id));
            }
        }
        agents[a].path_planner->findMinimumSetofColldingTargets(
            goal_table, target_data->target_agents
        );
    }
    const auto& A_start = target_data->start_agents;
    const auto& A_target = target_data->target_agents;


    if (screen >= 3){
        cout<<"     Selected a : "<< a<<endl;
        cout<<"     Select A_start: ";
        for(auto e: A_start)
            cout<<"("<<e.first<<","<<e.second<<"), ";
        cout<<endl;
        cout<<"     Select A_target: ";
        for(auto e: A_target)
            cout<<e<<", ";
        cout<<endl;
    }

    set<int> neighbors_set;

    neighbors_set.insert(a);

    if(A_start.size() + A_target.size() >= target_size-1){
        if (A_start.empty()){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + target_size-1);
        }
        else if (A_target.size() >= target_size){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + target_size-2);

            neighbors_set.insert(A_start.begin()->second);
        }
        else{
            neighbors_set.insert(A_target.begin(), A_target.end());
            for(auto e : A_start){
                //A_start is ordered by time.
                if (neighbors_set.size()>= target_size)
                    break;
                neighbors_set.insert(e.second);

            }
        }
    }
    else if (!A_start.empty() || !A_target.empty()){
        neighbors_set.insert(A_target.begin(), A_target.end());
        for(auto e : A_start){
            neighbors_set.insert(e.second);
        }

        set<int> tabu_set;
        while(neighbors_set.size()<target_size){
            int rand_int = rand() % neighbors_set.size();
            auto it = neighbors_set.begin();
            std::advance(it, rand_int);
            a = *it;
            tabu_set.insert(a);

            if(tabu_set.size() == neighbors_set.size())
                break;

            vector<int> targets;
            for(auto p: agents[a].path){
                if(goal_table[p.location]>-1){
                    targets.push_back(goal_table[p.location]);
                }
            }

            if(targets.empty())
                continue;
            rand_int = rand() %targets.size();
            neighbors_set.insert(*(targets.begin()+rand_int));
        }
    }



    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by target" << endl;
    return true;
}
bool InitLNS::generateNeighborRandomly(int forced_seed, int requested_size)
{
    int target_size = requested_size > 0 ? requested_size : neighbor_size;
    target_size = min((int)agents.size(), max(1, target_size));
    if (target_size >= agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }
    set<int> neighbors_set;
    if (forced_seed >= 0)
        neighbors_set.insert(forced_seed);
    auto total = num_of_colliding_pairs * 2 + agents.size();
    while(neighbors_set.size() < target_size)
    {
        vector<int> r(target_size - neighbors_set.size());
        for (auto i = 0; i < target_size - neighbors_set.size(); i++)
            r[i] = rand() % total;
        std::sort(r.begin(), r.end());
        int sum = 0;
        for (int i = 0, j = 0; i < agents.size() and j < r.size(); i++)
        {
            sum += (int)collision_graph[i].size() + 1;
            if (sum >= r[j])
            {
                neighbors_set.insert(i);
                while (j < r.size() and sum >= r[j])
                    j++;
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors randomly" << endl;
    return true;
}

// Random walk; return the first agent that the agent collides with
int InitLNS::randomWalk(int agent_id)
{
    int t = rand() % agents[agent_id].path.size();
    int loc = agents[agent_id].path[t].location;
    while (t <= path_table.makespan and
           (path_table.table[loc].size() <= t or
           path_table.table[loc][t].empty() or
           (path_table.table[loc][t].size() == 1 and path_table.table[loc][t].front() == agent_id)))
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        int step = rand() % next_locs.size();
        auto it = next_locs.begin();
        loc = *std::next(next_locs.begin(), rand() % next_locs.size());
        t = t + 1;
    }
    if (t > path_table.makespan)
        return NO_AGENT;
    else
        return *std::next(path_table.table[loc][t].begin(), rand() % path_table.table[loc][t].size());
}

void InitLNS::writeIterStatsToFile(const string & file_name) const
{
    std::ofstream output;
    output.open(file_name);
    // header
    output << //"num of agents," <<
           "sum of costs," <<
           "num of colliding pairs," <<
           "runtime" << //"," <<
           //"MAPF algorithm" <<
           endl;

    for (const auto &data : iteration_stats)
    {
        output << //data.num_of_agents << "," <<
               data.sum_of_costs << "," <<
               data.num_of_colliding_pairs << "," <<
               data.runtime << //"," <<
               // data.algorithm <<
               endl;
    }
    output.close();
}

void InitLNS::writeResultToFile(const string & file_name, int sum_of_distances, double preprocessing_time) const
{
    std::ifstream infile(file_name);
    bool exist = infile.good();
    infile.close();
    if (!exist)
    {
        ofstream addHeads(file_name);
        addHeads << "runtime,num of collisions,solution cost,initial collisions,initial solution cost," <<
                 "sum of distances,iterations,group size," <<
                 "runtime of initial solution,area under curve," <<
                 "LL expanded nodes,LL generated,LL reopened,LL runs," <<
                 "preprocessing runtime,solver name,instance name" << endl;
        addHeads.close();
    }
    uint64_t num_LL_expanded = 0, num_LL_generated = 0, num_LL_reopened = 0, num_LL_runs = 0;
    for (auto & agent : agents)
    {
        agent.path_planner->reset();
        num_LL_expanded += agent.path_planner->accumulated_num_expanded;
        num_LL_generated += agent.path_planner->accumulated_num_generated;
        num_LL_reopened += agent.path_planner->accumulated_num_reopened;
        num_LL_runs += agent.path_planner->num_runs;
    }
    double auc = 0;
    if (!iteration_stats.empty())
    {
        auto prev = iteration_stats.begin();
        auto curr = prev;
        ++curr;
        while (curr != iteration_stats.end() && curr->runtime < time_limit)
        {
            auc += prev->num_of_colliding_pairs * (curr->runtime - prev->runtime);
            prev = curr;
            ++curr;
        }
        auc += prev->num_of_colliding_pairs * (time_limit - prev->runtime);
    }

    ofstream stats(file_name, std::ios::app);
    stats << runtime << "," << iteration_stats.back().num_of_colliding_pairs << "," <<
          sum_of_costs << "," << iteration_stats.front().num_of_colliding_pairs << "," <<
          iteration_stats.front().sum_of_costs << "," << sum_of_distances << "," <<
          iteration_stats.size() << "," << average_group_size << "," <<
          iteration_stats.front().runtime << "," << auc << "," <<
          num_LL_expanded << "," << num_LL_generated << "," << num_LL_reopened << "," << num_LL_runs << "," <<
          preprocessing_time << "," << getSolverName() << "," << instance.getInstanceName() << endl;
    stats.close();
}

void InitLNS::printCollisionGraph() const
{
    cout << "Collision graph: ";
    int edges = 0;
    for (size_t i = 0; i < collision_graph.size(); i++)
    {
        for (int j : collision_graph[i])
        {
            if (i < j)
            {
                cout << "(" << i << "," << j << "),";
                edges++;
            }
        }
    }
    cout << endl <<  "|V|=" << collision_graph.size() << ", |E|=" << edges << endl;
}


unordered_map<int, set<int>>& InitLNS::findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                               unordered_map<int, set<int>>& sub_graph)
{
    std::queue<int> Q;
    Q.push(vertex);
    sub_graph.emplace(vertex, graph[vertex]);
    while (!Q.empty())
    {
        auto v = Q.front(); Q.pop();
        for (const auto & u : graph[v])
        {
            auto ret = sub_graph.emplace(u, graph[u]);
            if (ret.second) // insert successfully
                Q.push(u);
        }
    }
    return sub_graph;
}

void InitLNS::printPath() const
{
    for (const auto& agent : agents)
        cout << "Agent " << agent.id << ": " << agent.path << endl;
}

void InitLNS::printResult()
{
    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);
    assert(!iteration_stats.empty());
    cout << "\t" << getSolverName() << ": "
         << "runtime = " << runtime << ", "
         << "iterations = " << iteration_stats.size() << ", "
         << "colliding pairs = " << num_of_colliding_pairs << ", "
         << "initial colliding pairs = " << iteration_stats.front().num_of_colliding_pairs << ", "
         << "solution cost = " << sum_of_costs << ", "
         << "initial solution cost = " << iteration_stats.front().sum_of_costs << ", "
         << "failed iterations = " << num_of_failures << endl;
}

void InitLNS::clear()
{
    path_table.clear();
    collision_graph.clear();
    goal_table.clear();
}


bool InitLNS::validatePathTable() const
{
    for (auto i = 0; i < agents.size(); i++)
        assert(path_table.getPath(i) == &agents[i].path);
    return true;
}
