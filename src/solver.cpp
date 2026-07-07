#include "lns2/solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace lns2 {
namespace {

Location position_at(const Path& path, int timestep) {
    return path[std::min(timestep, static_cast<int>(path.size()) - 1)];
}

struct SearchNode {
    Location location;
    int timestep;
    int conflicts;
    int heuristic;
    int parent;
};

struct QueueEntry {
    int conflicts;
    int estimated_length;
    int heuristic;
    int node;

    bool operator>(const QueueEntry& other) const {
        return std::tie(conflicts, estimated_length, heuristic, node) >
               std::tie(
                   other.conflicts,
                   other.estimated_length,
                   other.heuristic,
                   other.node);
    }
};

int transition_conflicts(
    Location current,
    Location next,
    int timestep,
    const Paths& fixed_paths) {
    int conflicts = 0;
    for (const auto& path : fixed_paths) {
        if (path.empty()) {
            continue;
        }
        if (position_at(path, timestep) == next) {
            ++conflicts;
        }
        if (timestep > 0 &&
            position_at(path, timestep - 1) == next &&
            position_at(path, timestep) == current &&
            current != next) {
            ++conflicts;
        }
    }
    return conflicts;
}

int future_goal_conflicts(
    Location goal,
    int arrival,
    int future_limit,
    const Paths& fixed_paths) {
    int conflicts = 0;
    for (const auto& path : fixed_paths) {
        if (path.empty()) {
            continue;
        }
        for (int t = arrival + 1; t <= future_limit; ++t) {
            if (position_at(path, t) == goal) {
                ++conflicts;
            }
        }
    }
    return conflicts;
}

}  // namespace

Solver::Solver(
    const Instance& instance,
    SolverOptions options,
    GuidanceCallback guidance,
    CandidateGuidanceCallback candidate_guidance)
    : instance_(instance),
      options_(options),
      random_(options.seed),
      guidance_(std::move(guidance)),
      candidate_guidance_(std::move(candidate_guidance)) {
    options_.neighborhood_size =
        std::max(1, std::min(
                        options_.neighborhood_size,
                        static_cast<int>(instance_.agents().size())));
    options_.max_iterations = std::max(0, options_.max_iterations);
    options_.time_limit_ms = std::max(1, options_.time_limit_ms);
    options_.candidate_count = std::max(1, options_.candidate_count);
    options_.candidate_trial_limit_ms =
        std::max(1, options_.candidate_trial_limit_ms);
    if (options_.candidate_mode != CandidateMode::Disabled &&
        options_.neighborhood_size < 2) {
        throw std::runtime_error(
            "candidate modes require a neighborhood of at least two agents");
    }
}

Path Solver::plan_agent(int agent_id, const Paths& fixed_paths) {
    const auto& agent = instance_.agents()[agent_id];
    const int static_distance =
        instance_.shortest_distance(agent.start, agent.goal);
    if (static_distance < 0) {
        return {};
    }

    int reference_horizon = 0;
    for (const auto& path : fixed_paths) {
        reference_horizon =
            std::max(reference_horizon, static_cast<int>(path.size()) - 1);
    }
    const int horizon = std::max(
        static_distance + instance_.size(),
        reference_horizon + instance_.size());

    std::vector<int> distance_to_goal(instance_.size(), -1);
    std::queue<Location> distance_open;
    distance_to_goal[agent.goal] = 0;
    distance_open.push(agent.goal);
    while (!distance_open.empty()) {
        const auto current = distance_open.front();
        distance_open.pop();
        for (const auto next : instance_.neighbors_with_wait(current)) {
            if (next == current || distance_to_goal[next] >= 0) {
                continue;
            }
            distance_to_goal[next] = distance_to_goal[current] + 1;
            distance_open.push(next);
        }
    }

    std::vector<SearchNode> nodes;
    nodes.push_back(
        {agent.start, 0, 0, distance_to_goal[agent.start], -1});
    std::priority_queue<
        QueueEntry,
        std::vector<QueueEntry>,
        std::greater<QueueEntry>>
        open;
    open.push({0, distance_to_goal[agent.start],
               distance_to_goal[agent.start], 0});

    const int state_count = (horizon + 1) * instance_.size();
    std::vector<int> best_conflicts(
        state_count, std::numeric_limits<int>::max());
    best_conflicts[agent.start] = 0;

    int best_goal_node = -1;
    int best_goal_conflicts = std::numeric_limits<int>::max();
    int best_goal_time = std::numeric_limits<int>::max();

    while (!open.empty() && !timed_out()) {
        const auto entry = open.top();
        open.pop();
        const auto current = nodes[entry.node];
        const int state_index =
            current.timestep * instance_.size() + current.location;
        if (current.conflicts != best_conflicts[state_index]) {
            continue;
        }
        if (current.conflicts > best_goal_conflicts) {
            break;
        }

        if (current.location == agent.goal) {
            const int total_conflicts =
                current.conflicts +
                future_goal_conflicts(
                    agent.goal,
                    current.timestep,
                    reference_horizon + instance_.size(),
                    fixed_paths);
            if (std::tie(total_conflicts, current.timestep) <
                std::tie(best_goal_conflicts, best_goal_time)) {
                best_goal_conflicts = total_conflicts;
                best_goal_time = current.timestep;
                best_goal_node = entry.node;
                if (total_conflicts == 0) {
                    break;
                }
            }
        }

        if (current.timestep >= horizon) {
            continue;
        }
        for (const auto next :
             instance_.neighbors_with_wait(current.location)) {
            const int next_time = current.timestep + 1;
            const int next_conflicts =
                current.conflicts +
                transition_conflicts(
                    current.location, next, next_time, fixed_paths);
            const int next_index =
                next_time * instance_.size() + next;
            if (next_conflicts >= best_conflicts[next_index]) {
                continue;
            }
            best_conflicts[next_index] = next_conflicts;
            const int node_index = static_cast<int>(nodes.size());
            nodes.push_back(
                {next, next_time, next_conflicts,
                 distance_to_goal[next], entry.node});
            open.push(
                {next_conflicts,
                 next_time + distance_to_goal[next],
                 distance_to_goal[next],
                 node_index});
        }
    }

    if (best_goal_node < 0) {
        return {};
    }
    Path path;
    for (int node = best_goal_node; node >= 0; node = nodes[node].parent) {
        path.push_back(nodes[node].location);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Paths Solver::initial_solution() {
    const int agent_count = static_cast<int>(instance_.agents().size());
    std::vector<int> order(agent_count);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random_);

    Paths paths(agent_count);
    Paths fixed;
    for (const int agent : order) {
        paths[agent] = plan_agent(agent, fixed);
        if (paths[agent].empty()) {
            return {};
        }
        fixed.push_back(paths[agent]);
    }
    return paths;
}

std::vector<ConflictEvent> Solver::conflict_events(const Paths& paths) {
    std::vector<ConflictEvent> result;
    for (int first = 0; first < static_cast<int>(paths.size()); ++first) {
        for (int second = first + 1;
             second < static_cast<int>(paths.size());
             ++second) {
            if (paths[first].empty() || paths[second].empty()) {
                continue;
            }
            const int horizon = std::max(
                static_cast<int>(paths[first].size()),
                static_cast<int>(paths[second].size()));
            bool conflict = false;
            for (int t = 0; t < horizon; ++t) {
                if (position_at(paths[first], t) ==
                    position_at(paths[second], t)) {
                    result.push_back(
                        {first,
                         second,
                         t,
                         ConflictKind::Vertex,
                         {position_at(paths[first], t)}});
                    conflict = true;
                    break;
                }
                if (t > 0 &&
                    position_at(paths[first], t - 1) ==
                        position_at(paths[second], t) &&
                    position_at(paths[first], t) ==
                        position_at(paths[second], t - 1)) {
                    result.push_back(
                        {first,
                         second,
                         t,
                         ConflictKind::EdgeSwap,
                         {position_at(paths[first], t - 1),
                          position_at(paths[first], t)}});
                    conflict = true;
                    break;
                }
            }
            (void)conflict;
        }
    }
    return result;
}

std::vector<std::pair<int, int>> Solver::conflicting_pairs(
    const Paths& paths) {
    std::vector<std::pair<int, int>> result;
    for (const auto& event : conflict_events(paths)) {
        result.emplace_back(event.first_agent, event.second_agent);
    }
    return result;
}

Solver::NeighborhoodSelection Solver::select_neighborhood(
    const std::vector<std::pair<int, int>>& conflicts) {
    const int agent_count = static_cast<int>(instance_.agents().size());
    std::vector<std::vector<int>> graph(agent_count);
    for (const auto& [first, second] : conflicts) {
        graph[first].push_back(second);
        graph[second].push_back(first);
    }

    std::uniform_int_distribution<int> conflict_choice(
        0, static_cast<int>(conflicts.size()) - 1);
    const auto seed_pair = conflicts[conflict_choice(random_)];
    std::vector<int> neighborhood;
    std::queue<int> open;
    std::vector<bool> selected(agent_count, false);
    const int seed =
        std::uniform_int_distribution<int>(0, 1)(random_) == 0
            ? seed_pair.first
            : seed_pair.second;
    selected[seed] = true;
    open.push(seed);

    while (!open.empty() &&
           static_cast<int>(neighborhood.size()) <
               options_.neighborhood_size) {
        const int current = open.front();
        open.pop();
        neighborhood.push_back(current);
        std::shuffle(
            graph[current].begin(), graph[current].end(), random_);
        for (const int next : graph[current]) {
            if (!selected[next]) {
                selected[next] = true;
                open.push(next);
            }
        }
    }

    std::vector<int> remaining(agent_count);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::shuffle(remaining.begin(), remaining.end(), random_);
    std::stable_sort(
        remaining.begin(), remaining.end(),
        [&](int left, int right) {
            return graph[left].size() > graph[right].size();
        });
    for (const int agent : remaining) {
        if (static_cast<int>(neighborhood.size()) >=
            options_.neighborhood_size) {
            break;
        }
        if (!selected[agent]) {
            selected[agent] = true;
            neighborhood.push_back(agent);
        }
    }
    return {seed_pair, std::move(neighborhood)};
}

Paths Solver::replan_neighborhood(
    const Paths& current,
    const std::vector<int>& neighborhood,
    std::vector<int>* actual_order) {
    auto order = neighborhood;
    std::shuffle(order.begin(), order.end(), random_);
    if (actual_order != nullptr) {
        *actual_order = order;
    }
    return replan_neighborhood_fixed(current, neighborhood, order);
}

Paths Solver::replan_neighborhood_fixed(
    const Paths& current,
    const std::vector<int>& neighborhood,
    const std::vector<int>& order) {
    Paths candidate = current;
    std::vector<bool> selected(current.size(), false);
    for (const int agent : neighborhood) {
        if (agent < 0 || agent >= static_cast<int>(current.size()) ||
            selected[agent]) {
            return {};
        }
        selected[agent] = true;
        candidate[agent].clear();
    }
    if (order.size() != neighborhood.size()) {
        return {};
    }
    std::vector<bool> ordered(current.size(), false);
    for (const int agent : order) {
        if (agent < 0 || agent >= static_cast<int>(current.size()) ||
            !selected[agent] || ordered[agent]) {
            return {};
        }
        ordered[agent] = true;
    }

    Paths fixed;
    for (int agent = 0; agent < static_cast<int>(candidate.size()); ++agent) {
        if (!selected[agent]) {
            fixed.push_back(candidate[agent]);
        }
    }

    for (const int agent : order) {
        candidate[agent] = plan_agent(agent, fixed);
        if (candidate[agent].empty()) {
            return {};
        }
        fixed.push_back(candidate[agent]);
    }
    return candidate;
}

std::vector<CandidateTrial> Solver::generate_candidates(
    int iteration,
    const NeighborhoodSelection& baseline,
    const std::vector<ConflictEvent>& events,
    const Paths& paths) const {
    const int agent_count = static_cast<int>(instance_.agents().size());
    const int target_size = options_.neighborhood_size;
    std::vector<std::vector<int>> graph(agent_count);
    std::vector<int> degree(agent_count, 0);
    std::unordered_set<Location> conflict_cells;
    for (const auto& event : events) {
        graph[event.first_agent].push_back(event.second_agent);
        graph[event.second_agent].push_back(event.first_agent);
        ++degree[event.first_agent];
        ++degree[event.second_agent];
        conflict_cells.insert(event.cells.begin(), event.cells.end());
    }
    for (auto& neighbors : graph) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end());
    }

    std::uint32_t derived_seed =
        options_.seed ^ (0x9e3779b9U * static_cast<std::uint32_t>(iteration));
    derived_seed ^= static_cast<std::uint32_t>(
        baseline.seed_conflict.first * 73856093U);
    derived_seed ^= static_cast<std::uint32_t>(
        baseline.seed_conflict.second * 19349663U);
    std::mt19937 local_random(derived_seed);

    std::vector<int> global_priority(agent_count);
    std::iota(global_priority.begin(), global_priority.end(), 0);
    std::stable_sort(
        global_priority.begin(),
        global_priority.end(),
        [&](int left, int right) {
            return std::tie(degree[left], right) >
                   std::tie(degree[right], left);
        });
    std::unordered_map<int, int> priority_rank;
    for (int index = 0; index < agent_count; ++index) {
        priority_rank[global_priority[index]] = index;
    }

    auto force_seed_pair = [&](const std::vector<int>& ranking) {
        std::vector<int> selected;
        std::vector<bool> seen(agent_count, false);
        const auto add = [&](int agent) {
            if (agent >= 0 && agent < agent_count && !seen[agent] &&
                static_cast<int>(selected.size()) < target_size) {
                seen[agent] = true;
                selected.push_back(agent);
            }
        };
        add(baseline.seed_conflict.first);
        add(baseline.seed_conflict.second);
        for (const int agent : ranking) {
            add(agent);
        }
        for (const int agent : global_priority) {
            add(agent);
        }
        return selected;
    };
    auto fixed_order = [&](const std::vector<int>& agents) {
        auto result = agents;
        std::stable_sort(
            result.begin(), result.end(),
            [&](int left, int right) {
                return std::tie(priority_rank[left], left) <
                       std::tie(priority_rank[right], right);
            });
        return result;
    };

    std::vector<CandidateTrial> result;
    std::set<std::vector<int>> unique_sets;
    auto add_candidate =
        [&](const std::string& generator,
            const std::vector<int>& ranking) {
            if (static_cast<int>(result.size()) >=
                options_.candidate_count) {
                return;
            }
            auto agents = force_seed_pair(ranking);
            if (static_cast<int>(agents.size()) != target_size) {
                return;
            }
            auto key = agents;
            std::sort(key.begin(), key.end());
            if (!unique_sets.insert(key).second) {
                return;
            }
            CandidateTrial candidate;
            candidate.candidate_index =
                static_cast<int>(result.size());
            candidate.generator = generator;
            candidate.agents = std::move(agents);
            candidate.replan_order = fixed_order(candidate.agents);
            result.push_back(std::move(candidate));
        };

    add_candidate("baseline_bfs", baseline.agents);

    auto bfs_ranking = [&](int start, bool reverse) {
        std::vector<int> ranking;
        std::vector<bool> visited(agent_count, false);
        std::queue<int> open;
        visited[start] = true;
        open.push(start);
        while (!open.empty()) {
            const int current = open.front();
            open.pop();
            ranking.push_back(current);
            auto neighbors = graph[current];
            if (reverse) {
                std::reverse(neighbors.begin(), neighbors.end());
            }
            for (const int next : neighbors) {
                if (!visited[next]) {
                    visited[next] = true;
                    open.push(next);
                }
            }
        }
        ranking.insert(
            ranking.end(), global_priority.begin(), global_priority.end());
        return ranking;
    };
    add_candidate(
        "alternate_endpoint_bfs",
        bfs_ranking(baseline.seed_conflict.second, true));

    auto random_walk_ranking = [&](std::uint32_t salt) {
        std::mt19937 walk_random(derived_seed ^ salt);
        std::vector<int> ranking = {
            baseline.seed_conflict.first,
            baseline.seed_conflict.second};
        int current = (salt & 1U) == 0
                          ? baseline.seed_conflict.first
                          : baseline.seed_conflict.second;
        for (int step = 0; step < agent_count * 3; ++step) {
            if (graph[current].empty()) {
                current = std::uniform_int_distribution<int>(
                    0, agent_count - 1)(walk_random);
            } else {
                current = graph[current][
                    std::uniform_int_distribution<int>(
                        0,
                        static_cast<int>(graph[current].size()) - 1)(
                        walk_random)];
            }
            ranking.push_back(current);
        }
        auto remaining = global_priority;
        std::shuffle(remaining.begin(), remaining.end(), walk_random);
        ranking.insert(ranking.end(), remaining.begin(), remaining.end());
        return ranking;
    };
    add_candidate(
        "conflict_random_walk_a",
        random_walk_ranking(0xa511e9b3U));
    add_candidate(
        "conflict_random_walk_b",
        random_walk_ranking(0x63d83595U));

    add_candidate("highest_conflict_degree", global_priority);

    std::vector<int> overlap_ranking(agent_count);
    std::iota(overlap_ranking.begin(), overlap_ranking.end(), 0);
    std::vector<int> overlap(agent_count, 0);
    for (int agent = 0; agent < agent_count; ++agent) {
        for (const auto cell : paths[agent]) {
            overlap[agent] += conflict_cells.count(cell) > 0 ? 1 : 0;
        }
    }
    std::stable_sort(
        overlap_ranking.begin(), overlap_ranking.end(),
        [&](int left, int right) {
            return std::tie(overlap[left], degree[left], right) >
                   std::tie(overlap[right], degree[right], left);
        });
    add_candidate("path_conflict_overlap", overlap_ranking);

    std::vector<int> blocker_ranking(agent_count);
    std::iota(blocker_ranking.begin(), blocker_ranking.end(), 0);
    std::vector<int> blocker_score(agent_count, 0);
    for (int agent = 0; agent < agent_count; ++agent) {
        const auto start = instance_.agents()[agent].start;
        const auto goal = instance_.agents()[agent].goal;
        for (int other = 0; other < agent_count; ++other) {
            if (other == agent) {
                continue;
            }
            blocker_score[agent] += static_cast<int>(
                std::count(paths[other].begin(), paths[other].end(), start));
            blocker_score[agent] += static_cast<int>(
                std::count(paths[other].begin(), paths[other].end(), goal));
        }
    }
    std::stable_sort(
        blocker_ranking.begin(), blocker_ranking.end(),
        [&](int left, int right) {
            return std::tie(
                       blocker_score[left], degree[left], right) >
                   std::tie(
                       blocker_score[right], degree[right], left);
        });
    add_candidate("start_goal_blocker", blocker_ranking);

    std::vector<int> weighted_random;
    weighted_random.reserve(agent_count);
    std::vector<std::pair<double, int>> weighted_keys;
    std::uniform_real_distribution<double> unit(1e-12, 1.0);
    for (int agent = 0; agent < agent_count; ++agent) {
        const double key =
            -std::log(unit(local_random)) /
            static_cast<double>(degree[agent] + 1);
        weighted_keys.emplace_back(key, agent);
    }
    std::sort(weighted_keys.begin(), weighted_keys.end());
    for (const auto& [key, agent] : weighted_keys) {
        (void)key;
        weighted_random.push_back(agent);
    }
    add_candidate("degree_weighted_random", weighted_random);

    for (int attempt = 0;
         static_cast<int>(result.size()) < options_.candidate_count &&
         attempt < 512;
         ++attempt) {
        auto ranking = global_priority;
        std::shuffle(ranking.begin(), ranking.end(), local_random);
        add_candidate(
            "deterministic_fill_" + std::to_string(attempt), ranking);
    }
    if (static_cast<int>(result.size()) != options_.candidate_count) {
        throw std::runtime_error(
            "could not generate the requested number of unique candidates");
    }
    return result;
}

double Solver::run_candidate_trials(
    const Paths& current,
    std::vector<CandidateTrial>* candidates) {
    double total_runtime_ms = 0.0;
    for (auto& trial : *candidates) {
        trial.order_trials.clear();
        auto order_seeds = options_.candidate_replan_order_seeds;
        if (order_seeds.empty()) {
            order_seeds.push_back(0);
        }
        for (const int order_seed : order_seeds) {
            CandidateTrial::OrderTrial order_trial;
            order_trial.order_seed = order_seed;
            order_trial.replan_order = trial.replan_order;
            if (order_seed != 0) {
                std::uint32_t derived_seed =
                    options_.seed ^
                    (0x85ebca6bU *
                     static_cast<std::uint32_t>(
                         trial.candidate_index + 1));
                derived_seed ^= static_cast<std::uint32_t>(
                    order_seed * 0xc2b2ae35U);
                for (const int agent : trial.agents) {
                    derived_seed ^= static_cast<std::uint32_t>(
                        (agent + 1) * 0x27d4eb2dU);
                    derived_seed =
                        (derived_seed << 13U) | (derived_seed >> 19U);
                }
                std::mt19937 order_random(derived_seed);
                std::shuffle(
                    order_trial.replan_order.begin(),
                    order_trial.replan_order.end(),
                    order_random);
            }

            const auto trial_start =
                std::chrono::steady_clock::now();
            const auto main_deadline = deadline_;
            deadline_ = trial_start +
                        std::chrono::milliseconds(
                            options_.candidate_trial_limit_ms);
            auto candidate = replan_neighborhood_fixed(
                current, trial.agents, order_trial.replan_order);
            const auto replan_end = std::chrono::steady_clock::now();
            order_trial.trial_performed = true;
            order_trial.replan_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    replan_end - trial_start)
                    .count();
            if (!candidate.empty()) {
                order_trial.candidate_valid = true;
                order_trial.conflict_events_after =
                    conflict_events(candidate);
                order_trial.conflicting_pairs_after =
                    static_cast<int>(
                        order_trial.conflict_events_after.size());
                order_trial.sum_of_costs_after = path_cost(candidate);
                for (const int agent : trial.agents) {
                    order_trial.neighborhood_paths_after.push_back(
                        candidate[agent]);
                }
            }
            const auto trial_duration =
                std::chrono::steady_clock::now() - trial_start;
            deadline_ = main_deadline + trial_duration;
            order_trial.total_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    trial_duration)
                    .count();
            total_runtime_ms += order_trial.total_runtime_ms;
            trial.order_trials.push_back(std::move(order_trial));
        }
        if (!trial.order_trials.empty()) {
            const auto& primary = trial.order_trials.front();
            trial.trial_performed = primary.trial_performed;
            trial.candidate_valid = primary.candidate_valid;
            trial.conflicting_pairs_after =
                primary.conflicting_pairs_after;
            trial.sum_of_costs_after = primary.sum_of_costs_after;
            trial.replan_runtime_ms = primary.replan_runtime_ms;
            trial.total_runtime_ms = primary.total_runtime_ms;
            trial.conflict_events_after =
                primary.conflict_events_after;
            trial.neighborhood_paths_after =
                primary.neighborhood_paths_after;
        }
    }
    return total_runtime_ms;
}

SolveResult Solver::solve() {
    const auto start_time = std::chrono::steady_clock::now();
    deadline_ = start_time +
                std::chrono::milliseconds(options_.time_limit_ms);

    SolveResult result;
    result.paths = initial_solution();
    if (result.paths.empty()) {
        result.metrics.runtime_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start_time)
                .count();
        result.metrics.search_runtime_ms = result.metrics.runtime_ms;
        return result;
    }

    auto current_events = conflict_events(result.paths);
    std::vector<std::pair<int, int>> conflicts;
    for (const auto& event : current_events) {
        conflicts.emplace_back(event.first_agent, event.second_agent);
    }
    result.metrics.initial_conflicting_pairs =
        static_cast<int>(conflicts.size());

    while (!conflicts.empty() &&
           result.metrics.iterations < options_.max_iterations &&
           !timed_out()) {
        ++result.metrics.iterations;
        auto selection = select_neighborhood(conflicts);
        IterationTrace trace;
        trace.iteration = result.metrics.iterations;
        trace.seed_conflict = selection.seed_conflict;
        trace.baseline_neighborhood = selection.agents;
        const bool candidate_mode =
            options_.candidate_mode != CandidateMode::Disabled;
        if (candidate_mode) {
            const auto capture_start =
                std::chrono::steady_clock::now();
            trace.paths_before = result.paths;
            const auto capture_duration =
                std::chrono::steady_clock::now() - capture_start;
            deadline_ += capture_duration;
            result.metrics.counterfactual_runtime_ms +=
                std::chrono::duration<double, std::milli>(
                    capture_duration)
                    .count();
            if (options_.candidate_mode != CandidateMode::Collect) {
                const auto generation_start =
                    std::chrono::steady_clock::now();
                trace.candidate_trials = generate_candidates(
                    trace.iteration,
                    selection,
                    current_events,
                    result.paths);
                const auto generation_duration =
                    std::chrono::steady_clock::now() -
                    generation_start;
                deadline_ += generation_duration;
                trace.candidate_generation_runtime_ms =
                    std::chrono::duration<double, std::milli>(
                        generation_duration)
                        .count();
                result.metrics.counterfactual_runtime_ms +=
                    trace.candidate_generation_runtime_ms;
                trace.selected_candidate_index = 0;
                selection.agents =
                    trace.candidate_trials.front().agents;
            }
        }
        if (options_.candidate_mode == CandidateMode::Guided) {
            trace.guidance_requested = true;
            ++result.metrics.guidance_requests;
            CandidateGuidanceResponse response;
            const auto guidance_start =
                std::chrono::steady_clock::now();
            try {
                if (!candidate_guidance_) {
                    response.fallback_reason =
                        "candidate_callback_missing";
                } else {
                    response = candidate_guidance_(
                        {trace.iteration,
                         selection.seed_conflict,
                         current_events,
                         result.paths,
                         trace.candidate_trials});
                }
            } catch (const std::exception& error) {
                response.fallback_reason =
                    std::string("callback_error:") + error.what();
            } catch (...) {
                response.fallback_reason = "callback_error";
            }
            const auto guidance_duration =
                std::chrono::steady_clock::now() - guidance_start;
            deadline_ += guidance_duration;
            trace.guidance_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    guidance_duration)
                    .count();
            result.metrics.guidance_runtime_ms +=
                trace.guidance_runtime_ms;
            trace.guidance_out_of_distribution =
                response.out_of_distribution;
            trace.guidance_effective_probability =
                response.predicted_valid_probability;
            trace.guidance_nearest_distance =
                response.nearest_distance;
            trace.guidance_fallback_reason =
                response.fallback_reason;

            bool valid = response.use_guidance;
            if (valid &&
                (response.candidate_index <= 0 ||
                 response.candidate_index >= static_cast<int>(
                     trace.candidate_trials.size()))) {
                valid = false;
                trace.guidance_fallback_reason =
                    "invalid_candidate_index";
            }
            if (valid) {
                trace.selected_candidate_index =
                    response.candidate_index;
                selection.agents = trace.candidate_trials[
                    response.candidate_index]
                                       .agents;
                trace.guidance_used = true;
                ++result.metrics.guidance_used;
            } else {
                trace.selected_candidate_index = 0;
                selection.agents =
                    trace.candidate_trials.front().agents;
                if (trace.guidance_fallback_reason.empty()) {
                    trace.guidance_fallback_reason =
                        response.out_of_distribution
                            ? "out_of_distribution"
                            : "guidance_declined";
                }
                ++result.metrics.guidance_fallbacks;
            }
        } else if (
            options_.candidate_mode == CandidateMode::Disabled &&
            guidance_) {
            trace.guidance_requested = true;
            ++result.metrics.guidance_requests;
            const auto guidance_start =
                std::chrono::steady_clock::now();
            GuidanceResponse response;
            try {
                response = guidance_(
                    {trace.iteration,
                     selection.seed_conflict,
                     selection.agents,
                     current_events,
                     result.paths});
            } catch (const std::exception& error) {
                response.fallback_reason =
                    std::string("callback_error:") + error.what();
            } catch (...) {
                response.fallback_reason = "callback_error";
            }
            const auto guidance_duration =
                std::chrono::steady_clock::now() - guidance_start;
            deadline_ += guidance_duration;
            trace.guidance_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    guidance_duration)
                    .count();
            result.metrics.guidance_runtime_ms +=
                trace.guidance_runtime_ms;
            trace.guidance_out_of_distribution =
                response.out_of_distribution;
            trace.guidance_effective_probability =
                response.effective_probability;
            trace.guidance_nearest_distance =
                response.nearest_distance;
            trace.guidance_fallback_reason =
                response.fallback_reason;

            bool valid = response.use_guidance;
            const int expected_size = options_.neighborhood_size;
            if (valid &&
                static_cast<int>(response.agents.size()) != expected_size) {
                valid = false;
                trace.guidance_fallback_reason = "wrong_size";
            }
            std::vector<bool> seen(instance_.agents().size(), false);
            for (const int agent : response.agents) {
                if (agent < 0 ||
                    agent >= static_cast<int>(seen.size()) ||
                    seen[agent]) {
                    valid = false;
                    trace.guidance_fallback_reason = "invalid_agents";
                    break;
                }
                seen[agent] = true;
            }
            if (valid &&
                (!seen[selection.seed_conflict.first] ||
                 !seen[selection.seed_conflict.second])) {
                valid = false;
                trace.guidance_fallback_reason = "seed_pair_missing";
            }
            if (valid) {
                selection.agents = std::move(response.agents);
                trace.guidance_used = true;
                ++result.metrics.guidance_used;
            } else {
                if (trace.guidance_fallback_reason.empty()) {
                    trace.guidance_fallback_reason =
                        response.out_of_distribution
                            ? "out_of_distribution"
                            : "guidance_declined";
                }
                ++result.metrics.guidance_fallbacks;
            }
        }
        trace.neighborhood = selection.agents;
        trace.conflict_events_before = current_events;
        for (const int agent : selection.agents) {
            trace.neighborhood_paths_before.push_back(
                result.paths[agent]
            );
        }
        trace.conflicting_pairs_before =
            static_cast<int>(conflicts.size());
        trace.sum_of_costs_before = path_cost(result.paths);
        const auto replan_start = std::chrono::steady_clock::now();
        Paths candidate;
        if (options_.candidate_mode == CandidateMode::Controlled ||
            options_.candidate_mode == CandidateMode::Guided) {
            const auto& selected_candidate =
                trace.candidate_trials[
                    trace.selected_candidate_index < 0
                        ? 0
                        : trace.selected_candidate_index];
            trace.replan_order = selected_candidate.replan_order;
            candidate = replan_neighborhood_fixed(
                result.paths,
                selection.agents,
                trace.replan_order);
        } else {
            candidate = replan_neighborhood(
                result.paths,
                selection.agents,
                &trace.replan_order);
        }
        trace.replan_runtime_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - replan_start)
                .count();
        if (candidate.empty()) {
            result.trace.push_back(std::move(trace));
            continue;
        }
        trace.candidate_valid = true;
        auto candidate_events = conflict_events(candidate);
        trace.conflict_events_after = candidate_events;
        for (const int agent : selection.agents) {
            trace.neighborhood_paths_after.push_back(candidate[agent]);
        }
        std::vector<std::pair<int, int>> candidate_conflicts;
        for (const auto& event : candidate_events) {
            candidate_conflicts.emplace_back(
                event.first_agent, event.second_agent
            );
        }
        trace.conflicting_pairs_after =
            static_cast<int>(candidate_conflicts.size());
        trace.sum_of_costs_after = path_cost(candidate);
        const auto current_score =
            std::make_pair(
                static_cast<int>(conflicts.size()),
                trace.sum_of_costs_before);
        const auto candidate_score =
            std::make_pair(
                static_cast<int>(candidate_conflicts.size()),
                trace.sum_of_costs_after);
        if (candidate_score <= current_score) {
            trace.accepted = true;
            result.paths = std::move(candidate);
            conflicts = std::move(candidate_conflicts);
            current_events = std::move(candidate_events);
            ++result.metrics.accepted_iterations;
        }
        result.trace.push_back(std::move(trace));
    }

    if (options_.candidate_mode == CandidateMode::Collect) {
        for (auto& trace : result.trace) {
            const NeighborhoodSelection baseline = {
                trace.seed_conflict,
                trace.baseline_neighborhood};
            const auto generation_start =
                std::chrono::steady_clock::now();
            trace.candidate_trials = generate_candidates(
                trace.iteration,
                baseline,
                trace.conflict_events_before,
                trace.paths_before);
            const auto generation_duration =
                std::chrono::steady_clock::now() - generation_start;
            trace.candidate_generation_runtime_ms =
                std::chrono::duration<double, std::milli>(
                    generation_duration)
                    .count();
            result.metrics.counterfactual_runtime_ms +=
                trace.candidate_generation_runtime_ms;
            result.metrics.counterfactual_runtime_ms +=
                run_candidate_trials(
                    trace.paths_before, &trace.candidate_trials);
        }
    }

    result.metrics.final_conflicting_pairs =
        static_cast<int>(conflicts.size());
    result.metrics.success =
        conflicts.empty() && validate(instance_, result.paths);
    result.metrics.makespan = makespan(result.paths);
    result.metrics.sum_of_costs = path_cost(result.paths);
    result.metrics.runtime_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_time)
                .count();
    result.metrics.search_runtime_ms =
        result.metrics.runtime_ms -
        result.metrics.guidance_runtime_ms -
        result.metrics.counterfactual_runtime_ms;
    return result;
}

bool Solver::validate(
    const Instance& instance,
    const Paths& paths,
    std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (paths.size() != instance.agents().size()) {
        return fail("path count does not match agent count");
    }
    for (int agent = 0; agent < static_cast<int>(paths.size()); ++agent) {
        const auto& path = paths[agent];
        if (path.empty()) {
            return fail("empty path for agent " + std::to_string(agent));
        }
        if (path.front() != instance.agents()[agent].start ||
            path.back() != instance.agents()[agent].goal) {
            return fail(
                "wrong start or goal for agent " + std::to_string(agent));
        }
        for (int t = 0; t < static_cast<int>(path.size()); ++t) {
            if (!instance.traversable(path[t])) {
                return fail(
                    "blocked location in path for agent " +
                    std::to_string(agent));
            }
            if (t > 0) {
                const auto [r1, c1] = instance.coordinate(path[t - 1]);
                const auto [r2, c2] = instance.coordinate(path[t]);
                if (std::abs(r1 - r2) + std::abs(c1 - c2) > 1) {
                    return fail(
                        "non-adjacent move for agent " +
                        std::to_string(agent));
                }
            }
        }
    }
    if (!conflicting_pairs(paths).empty()) {
        return fail("solution contains agent conflicts");
    }
    return true;
}

int Solver::path_cost(const Paths& paths) const {
    int cost = 0;
    for (const auto& path : paths) {
        cost += std::max(0, static_cast<int>(path.size()) - 1);
    }
    return cost;
}

int Solver::makespan(const Paths& paths) const {
    int value = 0;
    for (const auto& path : paths) {
        value = std::max(value, static_cast<int>(path.size()) - 1);
    }
    return value;
}

bool Solver::timed_out() const {
    return std::chrono::steady_clock::now() >= deadline_;
}

void write_paths(
    const std::string& path,
    const Instance& instance,
    const Paths& paths) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write paths: " + path);
    }
    for (int agent = 0; agent < static_cast<int>(paths.size()); ++agent) {
        output << "agent " << agent << ':';
        for (const auto location : paths[agent]) {
            const auto [row, col] = instance.coordinate(location);
            output << " (" << row << ',' << col << ')';
        }
        output << '\n';
    }
}

void write_trace_jsonl(
    const std::string& path,
    const Instance& instance,
    const SolverOptions& options,
    const SolveResult& result) {
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    std::ofstream output(destination);
    if (!output) {
        throw std::runtime_error("cannot write trace: " + path);
    }
    output << std::boolalpha << std::fixed << std::setprecision(3);
    const int schema_version =
        options.candidate_mode != CandidateMode::Disabled
            ? options.candidate_replan_order_seeds.size() > 1 ? 5 : 4
            : result.metrics.guidance_requests > 0 ? 3 : 2;
    const auto write_int_array = [&](const std::vector<int>& values) {
        output << '[';
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                output << ',';
            }
            output << values[index];
        }
        output << ']';
    };
    const auto write_json_string = [&](const std::string& value) {
        output << '"';
        for (const char character : value) {
            if (character == '"' || character == '\\') {
                output << '\\';
            }
            output << character;
        }
        output << '"';
    };
    const auto write_path_json = [&](const Path& value) {
        output << '[';
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index > 0) {
                output << ',';
            }
            const auto [row, col] = instance.coordinate(value[index]);
            output << '[' << row << ',' << col << ']';
        }
        output << ']';
    };
    const auto write_conflict_events =
        [&](const std::vector<ConflictEvent>& events) {
            output << '[';
            for (std::size_t index = 0; index < events.size(); ++index) {
                if (index > 0) {
                    output << ',';
                }
                const auto& event = events[index];
                output << "{\"agents\":[" << event.first_agent << ','
                       << event.second_agent << ']'
                       << ",\"timestep\":" << event.timestep
                       << ",\"type\":\""
                       << (event.kind == ConflictKind::Vertex
                               ? "vertex"
                               : "edge_swap")
                       << "\",\"cells\":[";
                for (std::size_t cell_index = 0;
                     cell_index < event.cells.size();
                     ++cell_index) {
                    if (cell_index > 0) {
                        output << ',';
                    }
                    const auto [row, col] =
                        instance.coordinate(event.cells[cell_index]);
                    output << '[' << row << ',' << col << ']';
                }
                output << "]}";
            }
            output << ']';
        };
    const auto write_neighborhood_paths =
        [&](const std::vector<int>& agents, const Paths& paths) {
            output << '[';
            for (std::size_t index = 0; index < paths.size(); ++index) {
                if (index > 0) {
                    output << ',';
                }
                output << "{\"agent\":" << agents[index]
                       << ",\"path\":";
                write_path_json(paths[index]);
                output << '}';
            }
            output << ']';
        };
    const auto write_paths_json = [&](const Paths& paths) {
        output << '[';
        for (std::size_t index = 0; index < paths.size(); ++index) {
            if (index > 0) {
                output << ',';
            }
            write_path_json(paths[index]);
        }
        output << ']';
    };
    const auto write_candidate_trials =
        [&](const std::vector<CandidateTrial>& candidates) {
            output << '[';
            for (std::size_t index = 0;
                 index < candidates.size();
                 ++index) {
                if (index > 0) {
                    output << ',';
                }
                const auto& candidate = candidates[index];
                output << "{\"candidate_index\":"
                       << candidate.candidate_index
                       << ",\"generator\":";
                write_json_string(candidate.generator);
                output << ",\"agents\":";
                write_int_array(candidate.agents);
                output << ",\"replan_order\":";
                write_int_array(candidate.replan_order);
                output << ",\"trial_performed\":"
                       << candidate.trial_performed
                       << ",\"candidate_valid\":"
                       << candidate.candidate_valid
                       << ",\"conflicting_pairs_after\":"
                       << candidate.conflicting_pairs_after
                       << ",\"sum_of_costs_after\":"
                       << candidate.sum_of_costs_after
                       << ",\"replan_runtime_ms\":"
                       << candidate.replan_runtime_ms
                       << ",\"total_runtime_ms\":"
                       << candidate.total_runtime_ms
                       << ",\"conflict_events_after\":";
                write_conflict_events(
                    candidate.conflict_events_after);
                output << ",\"neighborhood_paths_after\":";
                write_neighborhood_paths(
                    candidate.agents,
                    candidate.neighborhood_paths_after);
                output << ",\"order_trials\":[";
                for (std::size_t order_index = 0;
                     order_index < candidate.order_trials.size();
                     ++order_index) {
                    if (order_index > 0) {
                        output << ',';
                    }
                    const auto& order_trial =
                        candidate.order_trials[order_index];
                    output << "{\"order_seed\":"
                           << order_trial.order_seed
                           << ",\"replan_order\":";
                    write_int_array(order_trial.replan_order);
                    output << ",\"trial_performed\":"
                           << order_trial.trial_performed
                           << ",\"candidate_valid\":"
                           << order_trial.candidate_valid
                           << ",\"conflicting_pairs_after\":"
                           << order_trial.conflicting_pairs_after
                           << ",\"sum_of_costs_after\":"
                           << order_trial.sum_of_costs_after
                           << ",\"replan_runtime_ms\":"
                           << order_trial.replan_runtime_ms
                           << ",\"total_runtime_ms\":"
                           << order_trial.total_runtime_ms
                           << ",\"conflict_events_after\":";
                    write_conflict_events(
                        order_trial.conflict_events_after);
                    output << ",\"neighborhood_paths_after\":";
                    write_neighborhood_paths(
                        candidate.agents,
                        order_trial.neighborhood_paths_after);
                    output << '}';
                }
                output << ']';
                output << '}';
            }
            output << ']';
        };
    for (const auto& trace : result.trace) {
        output << "{\"schema_version\":" << schema_version
               << ",\"event_type\":\"iteration\""
               << ",\"solver_seed\":" << options.seed
               << ",\"iteration\":" << trace.iteration
               << ",\"seed_conflict\":[" << trace.seed_conflict.first
               << ',' << trace.seed_conflict.second << ']'
               << ",\"baseline_neighborhood\":";
        write_int_array(trace.baseline_neighborhood);
        output << ",\"neighborhood\":";
        write_int_array(trace.neighborhood);
        output << ",\"replan_order\":";
        write_int_array(trace.replan_order);
        output
               << ",\"selected_candidate_index\":"
               << trace.selected_candidate_index
               << ",\"candidate_generation_runtime_ms\":"
               << trace.candidate_generation_runtime_ms
               << ",\"conflicting_pairs_before\":"
               << trace.conflicting_pairs_before
               << ",\"conflicting_pairs_after\":"
               << trace.conflicting_pairs_after
               << ",\"sum_of_costs_before\":"
               << trace.sum_of_costs_before
               << ",\"sum_of_costs_after\":"
               << trace.sum_of_costs_after
               << ",\"candidate_valid\":" << trace.candidate_valid
               << ",\"accepted\":" << trace.accepted
               << ",\"replan_runtime_ms\":"
               << trace.replan_runtime_ms
               << ",\"guidance_requested\":"
               << trace.guidance_requested
               << ",\"guidance_used\":" << trace.guidance_used
               << ",\"guidance_out_of_distribution\":"
               << trace.guidance_out_of_distribution
               << ",\"guidance_effective_probability\":"
               << trace.guidance_effective_probability
               << ",\"guidance_nearest_distance\":"
               << trace.guidance_nearest_distance
               << ",\"guidance_runtime_ms\":"
               << trace.guidance_runtime_ms
               << ",\"guidance_fallback_reason\":";
        write_json_string(trace.guidance_fallback_reason);
        output << ",\"conflict_events_before\":";
        write_conflict_events(trace.conflict_events_before);
        output << ",\"conflict_events_after\":";
        write_conflict_events(trace.conflict_events_after);
        output << ",\"paths_before\":";
        write_paths_json(trace.paths_before);
        output << ",\"neighborhood_paths_before\":";
        write_neighborhood_paths(
            trace.neighborhood, trace.neighborhood_paths_before
        );
        output << ",\"neighborhood_paths_after\":";
        write_neighborhood_paths(
            trace.neighborhood, trace.neighborhood_paths_after
        );
        output << ",\"candidate_trials\":";
        write_candidate_trials(trace.candidate_trials);
        output << "}\n";
    }
    const auto& metrics = result.metrics;
    output << "{\"schema_version\":" << schema_version
           << ",\"event_type\":\"summary\""
           << ",\"solver_seed\":" << options.seed
           << ",\"success\":" << metrics.success
           << ",\"initial_conflicting_pairs\":"
           << metrics.initial_conflicting_pairs
           << ",\"final_conflicting_pairs\":"
           << metrics.final_conflicting_pairs
           << ",\"iterations\":" << metrics.iterations
           << ",\"accepted_iterations\":"
           << metrics.accepted_iterations
           << ",\"makespan\":" << metrics.makespan
           << ",\"sum_of_costs\":" << metrics.sum_of_costs
           << ",\"runtime_ms\":" << metrics.runtime_ms
           << ",\"search_runtime_ms\":"
           << metrics.search_runtime_ms
           << ",\"guidance_runtime_ms\":"
           << metrics.guidance_runtime_ms
           << ",\"counterfactual_runtime_ms\":"
           << metrics.counterfactual_runtime_ms
           << ",\"guidance_requests\":"
           << metrics.guidance_requests
           << ",\"guidance_used\":" << metrics.guidance_used
           << ",\"guidance_fallbacks\":"
           << metrics.guidance_fallbacks
           << ",\"candidate_replan_order_seeds\":";
    write_int_array(options.candidate_replan_order_seeds);
    output
           << ",\"candidate_mode\":\""
           << (options.candidate_mode == CandidateMode::Collect
                   ? "collect"
                   : options.candidate_mode == CandidateMode::Controlled
                   ? "controlled"
                   : options.candidate_mode == CandidateMode::Guided
                   ? "guided"
                   : "disabled")
           << "\"}\n";
}

}  // namespace lns2
