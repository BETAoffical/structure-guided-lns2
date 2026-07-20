#include "online_features.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace
{
using FeatureClock = std::chrono::steady_clock;

double elapsedSeconds(const FeatureClock::time_point& started)
{
    return std::chrono::duration<double>(FeatureClock::now() - started).count();
}

struct AgentData
{
    int id = 0;
    std::vector<int> path;
    double conflict_degree = 0;
    double delay = 0;
    double path_cost = 0;
    double shortest_path_cost = 0;
    double wait_ratio = 0;
    double degree_sum = 0;
    int low_degree_count = 0;
    int articulation_count = 0;
    double obstacle_2_sum = 0;
    double obstacle_4_sum = 0;
    double visit_heat_sum = 0;
    std::unordered_set<int> path_set;
};

uint64_t pairKey(int left, int right)
{
    if (left > right) std::swap(left, right);
    return (uint64_t)(uint32_t)left << 32 | (uint32_t)right;
}

uint64_t directedKey(int from, int to)
{
    return (uint64_t)(uint32_t)from << 32 | (uint32_t)to;
}

int keyLeft(uint64_t key) { return (int)(uint32_t)(key >> 32); }
int keyRight(uint64_t key) { return (int)(uint32_t)key; }

double ratio(double numerator, double denominator)
{
    return denominator != 0 ? numerator / denominator : 0.0;
}

double mean(const std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    long double total = 0;
    for (double value : values) total += (long double)value;
    return (double)(total / values.size());
}

double populationStd(const std::vector<double>& values)
{
    if (values.size() <= 1) return 0.0;
    const long double average = mean(values);
    long double squared = 0;
    for (double value : values)
    {
        const long double delta = (long double)value - average;
        squared += delta * delta;
    }
    return std::sqrt((double)(squared / values.size()));
}

double integralPopulationStd(const std::vector<double>& values)
{
    if (values.size() <= 1) return 0.0;
    const int64_t center = (int64_t)values.front();
    int64_t offset_sum = 0;
    long double offset_square_sum = 0;
    for (double value : values)
    {
        const int64_t offset = (int64_t)value - center;
        offset_sum += offset;
        offset_square_sum += (long double)offset * offset;
    }
    const long double count = (long double)values.size();
    const long double variance =
        (offset_square_sum - (long double)offset_sum * offset_sum / count) / count;
    return std::sqrt((double)std::max((long double)0, variance));
}

class DenseFeatureWriter
{
public:
    DenseFeatureWriter(const std::unordered_map<std::string, size_t>& indices,
                       std::vector<double>& values) :
        indices(indices), values(values) {}

    class Slot
    {
    public:
        Slot(DenseFeatureWriter& writer, std::string name) :
            writer(writer), name(std::move(name)) {}

        Slot& operator=(double value)
        {
            writer.set(name, value);
            return *this;
        }

    private:
        DenseFeatureWriter& writer;
        std::string name;
    };

    Slot operator[](const char* name) { return Slot(*this, name); }
    Slot operator[](const std::string& name) { return Slot(*this, name); }
    Slot operator[](const py::str& name)
    {
        return Slot(*this, py::cast<std::string>(name));
    }

private:
    void set(const std::string& name, double value)
    {
        const auto position = indices.find(name);
        if (position != indices.end()) values[position->second] = value;
    }

    const std::unordered_map<std::string, size_t>& indices;
    std::vector<double>& values;
};

template<typename Output>
void aggregate(Output& output, const std::string& prefix,
               const std::vector<double>& values)
{
    double total = 0;
    for (double value : values) total += value;
    const double minimum = values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
    const double maximum = values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
    output[py::str(prefix + "_mean")] = mean(values);
    output[py::str(prefix + "_std")] = populationStd(values);
    output[py::str(prefix + "_min")] = minimum;
    output[py::str(prefix + "_max")] = maximum;
    output[py::str(prefix + "_sum")] = total;
}

double dictDouble(const py::dict& value, const char* key, double fallback = 0.0)
{
    return value.contains(key) ? py::cast<double>(value[key]) : fallback;
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return (char)std::tolower(character);
    });
    return value;
}

struct Analysis
{
    int rows = 0;
    int cols = 0;
    int cell_count = 0;
    int free_cell_count = 0;
    std::vector<double> degrees;
    std::vector<unsigned char> articulation;
    std::vector<double> obstacle_2;
    std::vector<double> obstacle_4;
    std::vector<int> visit_heat;
    std::vector<AgentData> agents;
    std::unordered_map<int, int> agent_index;
    std::unordered_set<uint64_t> conflict_pairs;
    std::unordered_map<uint64_t, int> event_pair_counts;
    std::vector<int> incident_event_counts;
    std::vector<std::vector<int>> adjacency;
    std::vector<int> component_id;
    std::vector<int> component_sizes;
    std::vector<double> event_times;
    int vertex_event_count = 0;
    double input_seconds = 0;
    double conflict_scan_seconds = 0;
    double graph_seconds = 0;
    double path_aggregate_seconds = 0;
};

void addEvent(Analysis& analysis, int time, bool vertex, int left, int right)
{
    const uint64_t key = pairKey(left, right);
    analysis.conflict_pairs.insert(key);
    analysis.event_pair_counts[key]++;
    const auto left_position = analysis.agent_index.find(left);
    const auto right_position = analysis.agent_index.find(right);
    if (left_position == analysis.agent_index.end() || right_position == analysis.agent_index.end())
        throw py::value_error("native feature event references an unknown agent");
    analysis.incident_event_counts[left_position->second]++;
    analysis.incident_event_counts[right_position->second]++;
    analysis.event_times.push_back((double)time);
    if (vertex) analysis.vertex_event_count++;
}

Analysis analyze(const py::dict& state, const py::dict& static_grid,
                 bool include_realized, bool fast_conflict_scan)
{
    const auto input_started = FeatureClock::now();
    Analysis analysis;
    analysis.rows = py::cast<int>(state["rows"]);
    analysis.cols = py::cast<int>(state["cols"]);
    analysis.cell_count = analysis.rows * analysis.cols;
    analysis.free_cell_count = py::cast<int>(static_grid["free_cell_count"]);
    analysis.degrees = py::cast<std::vector<double>>(static_grid["degrees"]);
    analysis.articulation = py::cast<std::vector<unsigned char>>(static_grid["articulation"]);
    analysis.obstacle_2 = py::cast<std::vector<double>>(static_grid["obstacle_rate_2"]);
    analysis.obstacle_4 = py::cast<std::vector<double>>(static_grid["obstacle_rate_4"]);
    if ((int)analysis.degrees.size() != analysis.cell_count ||
        (int)analysis.articulation.size() != analysis.cell_count ||
        (int)analysis.obstacle_2.size() != analysis.cell_count ||
        (int)analysis.obstacle_4.size() != analysis.cell_count)
        throw py::value_error("native static feature grid has the wrong dimensions");

    const py::list agent_values = py::cast<py::list>(state["agents"]);
    analysis.agents.reserve(py::len(agent_values));
    for (const py::handle& handle : agent_values)
    {
        const py::dict source = py::cast<py::dict>(handle);
        AgentData agent;
        agent.id = py::cast<int>(source["id"]);
        agent.path = py::cast<std::vector<int>>(source["path"]);
        if (agent.path.empty()) throw py::value_error("native feature agent path is empty");
        agent.conflict_degree = dictDouble(source, "conflict_degree");
        agent.delay = dictDouble(source, "delay");
        agent.path_cost = dictDouble(source, "path_cost");
        agent.shortest_path_cost = dictDouble(source, "shortest_path_cost");
        int waits = 0;
        for (size_t index = 1; index < agent.path.size(); index++)
            waits += agent.path[index - 1] == agent.path[index];
        agent.wait_ratio = ratio(waits, agent.path.size() > 1 ? agent.path.size() - 1 : 0);
        if (include_realized)
            agent.path_set.insert(agent.path.begin(), agent.path.end());
        if (!analysis.agent_index.emplace(agent.id, (int)analysis.agents.size()).second)
            throw py::value_error("native feature state contains duplicate agent ids");
        analysis.agents.push_back(std::move(agent));
    }
    const int agent_count = (int)analysis.agents.size();
    if (!agent_count) throw py::value_error("native feature state has no agents");
    analysis.incident_event_counts.assign(agent_count, 0);
    analysis.visit_heat.assign(analysis.cell_count, 0);
    size_t horizon = 0;
    for (const AgentData& agent : analysis.agents)
    {
        horizon = std::max(horizon, agent.path.size());
        for (int cell : agent.path)
        {
            if (cell < 0 || cell >= analysis.cell_count)
                throw py::value_error("native feature path cell is out of range");
            analysis.visit_heat[cell]++;
        }
    }
    analysis.input_seconds = elapsedSeconds(input_started);

    const auto conflict_scan_started = FeatureClock::now();
    if (!fast_conflict_scan)
    {
        for (size_t time = 0; time < horizon; time++)
        {
            std::unordered_map<int, std::vector<int>> occupancy;
            occupancy.reserve(agent_count * 2);
            std::unordered_map<uint64_t, std::vector<int>> transitions;
            if (time) transitions.reserve(agent_count * 2);
            for (const AgentData& agent : analysis.agents)
            {
                const int current = agent.path[std::min(time, agent.path.size() - 1)];
                occupancy[current].push_back(agent.id);
                if (time)
                {
                    const int previous = agent.path[std::min(time - 1, agent.path.size() - 1)];
                    if (previous != current)
                        transitions[directedKey(previous, current)].push_back(agent.id);
                }
            }
            for (auto& value : occupancy)
            {
                std::vector<int>& occupants = value.second;
                std::sort(occupants.begin(), occupants.end());
                for (size_t left = 0; left < occupants.size(); left++)
                    for (size_t right = left + 1; right < occupants.size(); right++)
                        addEvent(analysis, (int)time, true, occupants[left], occupants[right]);
            }
            if (time)
            {
                for (const auto& value : transitions)
                {
                    const int from = keyLeft(value.first);
                    const int to = keyRight(value.first);
                    if (from >= to) continue;
                    const auto reverse = transitions.find(directedKey(to, from));
                    if (reverse == transitions.end()) continue;
                    for (int left : value.second)
                        for (int right : reverse->second)
                            addEvent(analysis, (int)time, false, left, right);
                }
            }
        }
    }
    else
    {
        std::vector<int> first_occupant(analysis.cell_count, -1);
        std::vector<int> touched_cells;
        touched_cells.reserve(agent_count);
        std::unordered_map<int, std::vector<int>> colliding_occupants;
        colliding_occupants.reserve(std::max(1, agent_count / 4));
        std::unordered_map<uint64_t, std::vector<int>> transitions;
        transitions.reserve(agent_count * 2);
        std::vector<uint64_t> touched_transitions;
        touched_transitions.reserve(agent_count);
        for (size_t time = 0; time < horizon; time++)
        {
            for (int cell : touched_cells) first_occupant[cell] = -1;
            touched_cells.clear();
            colliding_occupants.clear();
            for (const AgentData& agent : analysis.agents)
            {
                const int current = agent.path[std::min(time, agent.path.size() - 1)];
                const int first = first_occupant[current];
                if (first < 0)
                {
                    first_occupant[current] = agent.id;
                    touched_cells.push_back(current);
                    continue;
                }
                auto inserted = colliding_occupants.emplace(
                    current, std::vector<int>{first}
                );
                std::vector<int>& occupants = inserted.first->second;
                for (int other : occupants)
                    addEvent(analysis, (int)time, true, other, agent.id);
                occupants.push_back(agent.id);
            }
            if (!time) continue;

            for (uint64_t key : touched_transitions) transitions[key].clear();
            touched_transitions.clear();
            for (const AgentData& agent : analysis.agents)
            {
                const int current = agent.path[std::min(time, agent.path.size() - 1)];
                const int previous = agent.path[std::min(time - 1, agent.path.size() - 1)];
                if (previous == current) continue;
                const uint64_t key = directedKey(previous, current);
                const auto reverse = transitions.find(directedKey(current, previous));
                if (reverse != transitions.end())
                    for (int other : reverse->second)
                        addEvent(analysis, (int)time, false, other, agent.id);
                std::vector<int>& movers = transitions[key];
                if (movers.empty()) touched_transitions.push_back(key);
                movers.push_back(agent.id);
            }
        }
    }
    analysis.conflict_scan_seconds = elapsedSeconds(conflict_scan_started);

    const auto graph_started = FeatureClock::now();
    analysis.adjacency.assign(agent_count, std::vector<int>());
    for (uint64_t key : analysis.conflict_pairs)
    {
        const int left = analysis.agent_index.at(keyLeft(key));
        const int right = analysis.agent_index.at(keyRight(key));
        analysis.adjacency[left].push_back(right);
        analysis.adjacency[right].push_back(left);
    }
    analysis.component_id.assign(agent_count, -1);
    for (int start = 0; start < agent_count; start++)
    {
        if (analysis.adjacency[start].empty() || analysis.component_id[start] >= 0) continue;
        const int identifier = (int)analysis.component_sizes.size();
        int size = 0;
        std::queue<int> pending;
        pending.push(start);
        analysis.component_id[start] = identifier;
        while (!pending.empty())
        {
            const int current = pending.front();
            pending.pop();
            size++;
            for (int neighbor : analysis.adjacency[current])
            {
                if (analysis.component_id[neighbor] < 0)
                {
                    analysis.component_id[neighbor] = identifier;
                    pending.push(neighbor);
                }
            }
        }
        analysis.component_sizes.push_back(size);
    }
    analysis.graph_seconds = elapsedSeconds(graph_started);

    const auto path_aggregate_started = FeatureClock::now();
    if (include_realized)
        for (AgentData& agent : analysis.agents)
        {
            for (int cell : agent.path)
            {
                agent.degree_sum += analysis.degrees[cell];
                agent.low_degree_count += analysis.degrees[cell] <= 2;
                agent.articulation_count += analysis.articulation[cell] != 0;
                agent.obstacle_2_sum += analysis.obstacle_2[cell];
                agent.obstacle_4_sum += analysis.obstacle_4[cell];
                agent.visit_heat_sum += analysis.visit_heat[cell];
            }
        }
    analysis.path_aggregate_seconds = elapsedSeconds(path_aggregate_started);
    return analysis;
}

py::dict projectedFeatures(const py::dict& source, const py::dict& required,
                           const char* group)
{
    if (!required.contains(group)) return source;
    py::dict output;
    for (const py::handle& name : py::cast<py::list>(required[group]))
    {
        if (source.contains(name)) output[name] = source[name];
        else output[name] = 0.0;
    }
    return output;
}

template<typename Output>
void fillDynamicFeatures(const py::dict& state, const Analysis& analysis,
                         Output& output)
{
    const int count = (int)analysis.agents.size();
    std::vector<double> degrees, delays, costs, shortest, waits;
    degrees.reserve(count); delays.reserve(count); costs.reserve(count);
    shortest.reserve(count); waits.reserve(count);
    int active = 0;
    for (size_t index = 0; index < analysis.agents.size(); index++)
    {
        const AgentData& agent = analysis.agents[index];
        degrees.push_back(agent.conflict_degree);
        delays.push_back(agent.delay);
        costs.push_back(agent.path_cost);
        shortest.push_back(agent.shortest_path_cost);
        waits.push_back(agent.wait_ratio);
        active += !analysis.adjacency[index].empty();
    }
    double cost_sum = 0, shortest_sum = 0;
    for (double value : costs) cost_sum += value;
    for (double value : shortest) shortest_sum += value;
    int largest_component = 0;
    for (int size : analysis.component_sizes) largest_component = std::max(largest_component, size);
    const py::dict low_level = state.contains("low_level")
        ? py::cast<py::dict>(state["low_level"]) : py::dict();
    output["state.agent_count"] = (double)count;
    output["state.iteration"] = dictDouble(state, "iteration");
    output["state.colliding_pairs"] = dictDouble(state, "num_of_colliding_pairs");
    output["state.conflict_edge_density"] = ratio(analysis.conflict_pairs.size(), count);
    output["state.conflict_event_count"] = (double)analysis.event_times.size();
    output["state.vertex_event_ratio"] = ratio(analysis.vertex_event_count, analysis.event_times.size());
    output["state.conflicting_agent_ratio"] = ratio(active, count);
    output["state.component_count"] = (double)analysis.component_sizes.size();
    output["state.largest_component"] = (double)largest_component;
    output["state.largest_component_ratio"] = ratio(largest_component, count);
    output["state.degree_mean"] = mean(degrees);
    output["state.degree_std"] = populationStd(degrees);
    output["state.degree_max"] = degrees.empty() ? 0.0 : *std::max_element(degrees.begin(), degrees.end());
    output["state.delay_mean"] = mean(delays);
    output["state.delay_std"] = populationStd(delays);
    output["state.delay_max"] = delays.empty() ? 0.0 : *std::max_element(delays.begin(), delays.end());
    output["state.path_cost_mean"] = mean(costs);
    output["state.path_cost_std"] = populationStd(costs);
    output["state.path_stretch_mean"] = ratio(cost_sum, std::max(1.0, shortest_sum));
    output["state.path_wait_ratio_mean"] = mean(waits);
    output["state.conflict_time_mean"] = mean(analysis.event_times);
    output["state.conflict_time_std"] = integralPopulationStd(analysis.event_times);
    output["state.sum_of_costs_per_agent"] = ratio(dictDouble(state, "sum_of_costs"), count);
    output["state.low_level_generated_per_agent"] = ratio(dictDouble(low_level, "generated"), count);
    output["state.low_level_runs_per_agent"] = ratio(dictDouble(low_level, "runs"), count);
}

py::dict dynamicFeatures(const py::dict& state, const Analysis& analysis)
{
    py::dict output;
    fillDynamicFeatures(state, analysis, output);
    return output;
}

std::vector<int> candidateAgents(const py::dict& candidate, const Analysis& analysis)
{
    std::vector<int> values = py::cast<std::vector<int>>(candidate["agents"]);
    std::unordered_set<int> unique(values.begin(), values.end());
    if (values.empty() || unique.size() != values.size())
        throw py::value_error("native candidate neighborhood must be non-empty and unique");
    for (int id : values)
        if (analysis.agent_index.find(id) == analysis.agent_index.end())
            throw py::value_error("native candidate references an unknown agent");
    std::sort(values.begin(), values.end());
    return values;
}

template<typename Output>
void fillProposalFeatures(const py::dict& candidate, const Analysis& analysis,
                          const std::vector<int>& selected, Output& output)
{
    std::vector<int> seed_ids;
    if (candidate.contains("seed_agents"))
        seed_ids = py::cast<std::vector<int>>(candidate["seed_agents"]);
    std::sort(seed_ids.begin(), seed_ids.end());
    seed_ids.erase(std::unique(seed_ids.begin(), seed_ids.end()), seed_ids.end());
    std::vector<double> seed_degrees, seed_delays, seed_costs, seed_components;
    for (int id : seed_ids)
    {
        const auto position = analysis.agent_index.find(id);
        if (position == analysis.agent_index.end())
            throw py::value_error("native proposal seed references an unknown agent");
        const int index = position->second;
        const AgentData& agent = analysis.agents[index];
        seed_degrees.push_back(agent.conflict_degree);
        seed_delays.push_back(agent.delay);
        seed_costs.push_back(agent.path_cost);
        const int component = analysis.component_id[index];
        seed_components.push_back(component < 0 ? 0.0 : analysis.component_sizes[component]);
    }
    py::dict counts;
    if (candidate.contains("proposal_count_by_family"))
        counts = py::cast<py::dict>(candidate["proposal_count_by_family"]);
    int total_count = 0;
    for (const auto& value : counts) total_count += py::cast<int>(value.second);
    std::unordered_set<std::string> selection_families;
    if (candidate.contains("selection_families"))
        for (const py::handle& value : py::cast<py::list>(candidate["selection_families"]))
            selection_families.insert(lower(py::cast<std::string>(value)));
    std::unordered_set<int> proposal_seeds;
    if (candidate.contains("proposal_seeds"))
        for (int value : py::cast<std::vector<int>>(candidate["proposal_seeds"]))
            proposal_seeds.insert(value);
    int support_count = 0;
    for (const auto& value : counts) support_count += py::cast<int>(value.second) > 0;
    const int actual_size = (int)selected.size();
    output["proposal.actual_size"] = (double)actual_size;
    output["proposal.actual_size_ratio_agents"] = ratio(actual_size, analysis.agents.size());
    output["proposal.total_count"] = (double)total_count;
    output["proposal.unique_proposal_seed_count"] = (double)proposal_seeds.size();
    output["proposal.seed_agent_count"] = (double)seed_ids.size();
    output["proposal.selection_family_count"] = (double)selection_families.size();
    output["proposal.support_family_count"] = (double)support_count;
    output[py::str("proposal.actual_size=" + std::to_string(actual_size))] = 1.0;
    for (const std::string& family : selection_families)
        output[py::str("proposal.selection_family=" + family)] = 1.0;
    for (const auto& value : counts)
    {
        const std::string family = lower(py::cast<std::string>(value.first));
        const int count = py::cast<int>(value.second);
        output[py::str("proposal.family_count=" + family)] = (double)count;
        output[py::str("proposal.family_ratio=" + family)] = ratio(count, total_count);
    }
    aggregate(output, "proposal.seed_conflict_degree", seed_degrees);
    aggregate(output, "proposal.seed_delay", seed_delays);
    aggregate(output, "proposal.seed_path_cost", seed_costs);
    aggregate(output, "proposal.seed_component_size", seed_components);
}

py::dict proposalFeatures(const py::dict& candidate, const Analysis& analysis,
                          const std::vector<int>& selected)
{
    py::dict output;
    fillProposalFeatures(candidate, analysis, selected, output);
    return output;
}

size_t intersectionSize(const std::unordered_set<int>& left,
                        const std::unordered_set<int>& right)
{
    const std::unordered_set<int>* smaller = &left;
    const std::unordered_set<int>* larger = &right;
    if (left.size() > right.size()) std::swap(smaller, larger);
    size_t count = 0;
    for (int value : *smaller) count += larger->count(value);
    return count;
}

template<typename Output>
void fillRealizedFeatures(const Analysis& analysis,
                          const std::vector<int>& selected_ids, Output& output)
{
    std::vector<int> selected_indices;
    std::vector<unsigned char> selected(analysis.agents.size(), 0);
    for (int id : selected_ids)
    {
        const int index = analysis.agent_index.at(id);
        selected_indices.push_back(index);
        selected[index] = 1;
    }
    int active_selected = 0;
    int active_total = 0;
    for (size_t index = 0; index < analysis.agents.size(); index++)
    {
        const bool active = !analysis.adjacency[index].empty();
        active_total += active;
        active_selected += active && selected[index];
    }
    int internal_edges = 0, boundary_edges = 0;
    for (int index : selected_indices)
    {
        for (int neighbor : analysis.adjacency[index])
        {
            if (selected[neighbor]) internal_edges += index < neighbor;
            else boundary_edges++;
        }
    }
    std::unordered_map<int, int> component_selected;
    for (int index : selected_indices)
        if (analysis.component_id[index] >= 0)
            component_selected[analysis.component_id[index]]++;
    std::vector<double> component_coverages;
    for (const auto& value : component_selected)
        component_coverages.push_back(ratio(value.second, analysis.component_sizes[value.first]));

    std::vector<double> delays, conflicts, costs, stretches, waits, overlaps;
    std::unordered_set<int> path_union;
    size_t path_entry_count = 0;
    long double degree_sum = 0, obstacle_2_sum = 0, obstacle_4_sum = 0, visit_sum = 0;
    int low_count = 0, articulation_count = 0;
    for (int index : selected_indices)
    {
        const AgentData& agent = analysis.agents[index];
        delays.push_back(agent.delay);
        conflicts.push_back(agent.conflict_degree);
        costs.push_back(agent.path_cost);
        stretches.push_back(ratio(agent.path_cost, std::max(1.0, agent.shortest_path_cost)));
        waits.push_back(agent.wait_ratio);
        path_entry_count += agent.path.size();
        degree_sum += agent.degree_sum;
        low_count += agent.low_degree_count;
        articulation_count += agent.articulation_count;
        obstacle_2_sum += agent.obstacle_2_sum;
        obstacle_4_sum += agent.obstacle_4_sum;
        visit_sum += agent.visit_heat_sum;
        path_union.insert(agent.path_set.begin(), agent.path_set.end());
    }
    for (size_t left = 0; left < selected_indices.size(); left++)
    {
        for (size_t right = left + 1; right < selected_indices.size(); right++)
        {
            const auto& first = analysis.agents[selected_indices[left]].path_set;
            const auto& second = analysis.agents[selected_indices[right]].path_set;
            const size_t intersection = intersectionSize(first, second);
            overlaps.push_back(ratio(intersection, first.size() + second.size() - intersection));
        }
    }
    int min_row = analysis.rows, max_row = -1, min_col = analysis.cols, max_col = -1;
    for (int cell : path_union)
    {
        const int row = cell / analysis.cols;
        const int col = cell % analysis.cols;
        min_row = std::min(min_row, row); max_row = std::max(max_row, row);
        min_col = std::min(min_col, col); max_col = std::max(max_col, col);
    }
    const int bbox_area = path_union.empty() ? 0 :
        (max_row - min_row + 1) * (max_col - min_col + 1);
    int internal_events = 0;
    int incident_events = 0;
    for (int index : selected_indices) incident_events += analysis.incident_event_counts[index];
    for (size_t left = 0; left < selected_indices.size(); left++)
        for (size_t right = left + 1; right < selected_indices.size(); right++)
        {
            const uint64_t key = pairKey(
                analysis.agents[selected_indices[left]].id,
                analysis.agents[selected_indices[right]].id);
            const auto position = analysis.event_pair_counts.find(key);
            if (position != analysis.event_pair_counts.end()) internal_events += position->second;
        }
    incident_events -= internal_events;

    output["realized.actual_size"] = (double)selected_indices.size();
    output["realized.actual_size_ratio_agents"] = ratio(selected_indices.size(), analysis.agents.size());
    output["realized.conflicting_agent_ratio"] = ratio(active_selected, selected_indices.size());
    output["realized.conflicting_agent_coverage"] = ratio(active_selected, active_total);
    output["realized.component_count"] = (double)component_selected.size();
    output["realized.component_coverage_mean"] = mean(component_coverages);
    output["realized.component_coverage_max"] = component_coverages.empty() ? 0.0 :
        *std::max_element(component_coverages.begin(), component_coverages.end());
    output["realized.internal_conflict_edges"] = (double)internal_edges;
    output["realized.boundary_conflict_edges"] = (double)boundary_edges;
    output["realized.incident_conflict_coverage"] = ratio(internal_edges + boundary_edges, analysis.conflict_pairs.size());
    output["realized.internal_conflict_coverage"] = ratio(internal_edges, analysis.conflict_pairs.size());
    output["realized.internal_event_coverage"] = ratio(internal_events, analysis.event_times.size());
    output["realized.incident_event_coverage"] = ratio(incident_events, analysis.event_times.size());
    output["realized.path_overlap_mean"] = mean(overlaps);
    output["realized.path_overlap_max"] = overlaps.empty() ? 0.0 : *std::max_element(overlaps.begin(), overlaps.end());
    output["realized.path_union_cell_ratio"] = ratio(path_union.size(), analysis.free_cell_count);
    output["realized.path_bbox_area_ratio"] = ratio(bbox_area, analysis.rows * analysis.cols);
    output["realized.path_degree_mean"] = ratio((double)degree_sum, path_entry_count);
    output["realized.path_low_degree_ratio"] = ratio(low_count, path_entry_count);
    output["realized.path_articulation_ratio"] = ratio(articulation_count, path_entry_count);
    output["realized.path_visit_heat_mean"] = ratio((double)visit_sum, path_entry_count);
    output["realized.path_obstacle_rate_r2"] = ratio((double)obstacle_2_sum, path_entry_count);
    output["realized.path_obstacle_rate_r4"] = ratio((double)obstacle_4_sum, path_entry_count);
    output["realized.path_wait_ratio_mean"] = mean(waits);
    aggregate(output, "realized.delay", delays);
    aggregate(output, "realized.conflict_degree", conflicts);
    aggregate(output, "realized.path_cost", costs);
    aggregate(output, "realized.path_stretch", stretches);
}

py::dict realizedFeatures(const Analysis& analysis,
                          const std::vector<int>& selected_ids)
{
    py::dict output;
    fillRealizedFeatures(analysis, selected_ids, output);
    return output;
}
}

py::dict batchOnlineFeatures(const py::dict& state, const py::list& candidates,
                             const py::dict& static_grid, bool include_realized,
                             const py::dict& required_features)
{
    const auto analysis_started = FeatureClock::now();
    const Analysis analysis = analyze(
        state, static_grid, include_realized, false
    );
    const double analysis_seconds = elapsedSeconds(analysis_started);
    const auto fill_started = FeatureClock::now();
    py::dict result;
    result["dynamic"] = projectedFeatures(
        dynamicFeatures(state, analysis), required_features, "dynamic");
    py::list proposal_rows;
    py::list realized_rows;
    for (const py::handle& handle : candidates)
    {
        const py::dict candidate = py::cast<py::dict>(handle);
        const std::vector<int> selected = candidateAgents(candidate, analysis);
        proposal_rows.append(projectedFeatures(
            proposalFeatures(candidate, analysis, selected),
            required_features, "proposal"));
        if (include_realized)
            realized_rows.append(projectedFeatures(
                realizedFeatures(analysis, selected),
                required_features, "realized"));
    }
    result["proposal"] = proposal_rows;
    result["realized"] = realized_rows;
    result["candidate_count"] = py::len(candidates);
    result["event_count"] = analysis.event_times.size();
    result["conflict_pair_count"] = analysis.conflict_pairs.size();
    result["state_analysis_seconds"] = analysis_seconds;
    result["state_input_seconds"] = analysis.input_seconds;
    result["state_conflict_scan_seconds"] = analysis.conflict_scan_seconds;
    result["state_graph_seconds"] = analysis.graph_seconds;
    result["state_path_aggregate_seconds"] = analysis.path_aggregate_seconds;
    result["feature_fill_seconds"] = elapsedSeconds(fill_started);
    return result;
}

py::dict batchOnlineFeatureVectors(const py::dict& state,
                                   const py::list& candidates,
                                   const py::dict& static_grid,
                                   const py::list& feature_names)
{
    std::vector<std::string> names;
    names.reserve(py::len(feature_names));
    bool include_realized = false;
    for (const py::handle& value : feature_names)
    {
        const std::string name = py::cast<std::string>(value);
        const bool known = name.rfind("state.", 0) == 0 ||
            name.rfind("context.", 0) == 0 ||
            name.rfind("proposal.", 0) == 0 ||
            name.rfind("realized.", 0) == 0;
        if (!known)
            throw py::value_error("native dense feature request contains an unknown prefix");
        include_realized = include_realized || name.rfind("realized.", 0) == 0;
        names.push_back(name);
    }
    if (names.empty())
        throw py::value_error("native dense feature request is empty");

    std::unordered_map<std::string, size_t> feature_indices;
    feature_indices.reserve(names.size());
    for (size_t index = 0; index < names.size(); index++)
        if (!feature_indices.emplace(names[index], index).second)
            throw py::value_error("native dense feature request contains duplicate names");

    const auto analysis_started = FeatureClock::now();
    const Analysis analysis = analyze(
        state, static_grid, include_realized, true
    );
    const double analysis_seconds = elapsedSeconds(analysis_started);
    const auto fill_started = FeatureClock::now();
    std::vector<double> dynamic_values(names.size(), 0.0);
    DenseFeatureWriter dynamic_writer(feature_indices, dynamic_values);
    fillDynamicFeatures(state, analysis, dynamic_writer);
    py::list vectors;
    for (const py::handle& handle : candidates)
    {
        const py::dict candidate = py::cast<py::dict>(handle);
        const std::vector<int> selected = candidateAgents(candidate, analysis);
        std::vector<double> values = dynamic_values;
        DenseFeatureWriter writer(feature_indices, values);
        fillProposalFeatures(candidate, analysis, selected, writer);
        if (include_realized)
            fillRealizedFeatures(analysis, selected, writer);
        vectors.append(py::cast(std::move(values)));
    }

    py::dict result;
    result["feature_names"] = feature_names;
    result["vectors"] = vectors;
    result["candidate_count"] = py::len(candidates);
    result["event_count"] = analysis.event_times.size();
    result["conflict_pair_count"] = analysis.conflict_pairs.size();
    result["state_analysis_seconds"] = analysis_seconds;
    result["state_input_seconds"] = analysis.input_seconds;
    result["state_conflict_scan_seconds"] = analysis.conflict_scan_seconds;
    result["state_graph_seconds"] = analysis.graph_seconds;
    result["state_path_aggregate_seconds"] = analysis.path_aggregate_seconds;
    result["feature_fill_seconds"] = elapsedSeconds(fill_started);
    return result;
}
