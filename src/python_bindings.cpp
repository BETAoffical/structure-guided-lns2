#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "InitLNS.h"
#include "online_features.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace py = pybind11;

namespace
{
using DiagnosticClock = std::chrono::steady_clock;

double diagnosticSeconds(const DiagnosticClock::time_point& started)
{
    return std::chrono::duration<double>(DiagnosticClock::now() - started).count();
}

struct ProcessGlobalRngState
{
    std::mutex mutex;
    uint64_t epoch = 0;
    uint64_t owner_environment_id = 0;
};

ProcessGlobalRngState& processGlobalRngState()
{
    static ProcessGlobalRngState state;
    return state;
}

uint64_t nextEnvironmentId()
{
    static std::atomic<uint64_t> next_id(1);
    return next_id.fetch_add(1, std::memory_order_relaxed);
}
}

struct PortableTreeNode
{
    double value = 0;
    double threshold = 0;
    int feature = 0;
    int left = 0;
    int right = 0;
    bool missing_go_to_left = false;
    bool is_leaf = false;
};

class PortableTreeEnsemble
{
public:
    PortableTreeEnsemble(double baseline, const py::list& tree_values) : baseline(baseline)
    {
        trees.reserve(py::len(tree_values));
        for (const py::handle& tree_value : tree_values)
        {
            const py::list node_values = py::cast<py::list>(tree_value);
            vector<PortableTreeNode> nodes;
            nodes.reserve(py::len(node_values));
            for (const py::handle& node_value : node_values)
            {
                const py::dict source = py::cast<py::dict>(node_value);
                PortableTreeNode node;
                node.value = py::cast<double>(source["value"]);
                node.is_leaf = py::cast<bool>(source["is_leaf"]);
                if (!node.is_leaf)
                {
                    node.threshold = py::cast<double>(source["num_threshold"]);
                    node.feature = py::cast<int>(source["feature_idx"]);
                    node.left = py::cast<int>(source["left"]);
                    node.right = py::cast<int>(source["right"]);
                    node.missing_go_to_left = py::cast<bool>(source["missing_go_to_left"]);
                }
                nodes.push_back(node);
            }
            if (nodes.empty())
                throw py::value_error("portable tree cannot be empty");
            trees.push_back(std::move(nodes));
        }
    }

    vector<double> predictRaw(const vector<vector<double>>& vectors) const
    {
        vector<double> predictions;
        predictions.reserve(vectors.size());
        for (const auto& values : vectors)
        {
            double raw = baseline;
            for (const auto& nodes : trees)
            {
                int index = 0;
                while (!nodes.at(index).is_leaf)
                {
                    const auto& node = nodes.at(index);
                    if (node.feature < 0 || node.feature >= (int)values.size())
                        throw py::value_error("portable feature vector is too short");
                    const double value = values[node.feature];
                    const bool go_left = (std::isnan(value) && node.missing_go_to_left) ||
                                         (!std::isnan(value) && value <= node.threshold);
                    index = go_left ? node.left : node.right;
                }
                raw += nodes.at(index).value;
            }
            predictions.push_back(raw);
        }
        return predictions;
    }

    vector<double> predictPositive(const vector<vector<double>>& vectors) const
    {
        vector<double> probabilities;
        for (const double raw : predictRaw(vectors))
        {
            if (raw >= 0)
                probabilities.push_back(1.0 / (1.0 + std::exp(-raw)));
            else
            {
                const double exponential = std::exp(raw);
                probabilities.push_back(exponential / (1.0 + exponential));
            }
        }
        return probabilities;
    }

    vector<double> scorePairwiseDense(
        const vector<vector<double>>& rows,
        const vector<int>& modes,
        const vector<int>& feature_indices) const
    {
        if (modes.size() != feature_indices.size())
            throw py::value_error("pairwise input mode/index dimensions differ");
        vector<double> scores(rows.size(), 0.0);
        for (size_t left = 0; left < rows.size(); left++)
            for (size_t right = left + 1; right < rows.size(); right++)
            {
                const double forward = positiveProbability(
                    predictPairRaw(rows[left], rows[right], modes, feature_indices)
                );
                const double reverse = positiveProbability(
                    predictPairRaw(rows[right], rows[left], modes, feature_indices)
                );
                const double probability = (forward + (1.0 - reverse)) / 2.0;
                scores[left] += probability;
                scores[right] += 1.0 - probability;
            }
        return scores;
    }

private:
    static double positiveProbability(double raw)
    {
        if (raw >= 0)
            return 1.0 / (1.0 + std::exp(-raw));
        const double exponential = std::exp(raw);
        return exponential / (1.0 + exponential);
    }

    double predictPairRaw(
        const vector<double>& left_values,
        const vector<double>& right_values,
        const vector<int>& modes,
        const vector<int>& feature_indices) const
    {
        double raw = baseline;
        for (const auto& nodes : trees)
        {
            int node_index = 0;
            while (!nodes.at(node_index).is_leaf)
            {
                const auto& node = nodes.at(node_index);
                if (node.feature < 0 || node.feature >= (int)modes.size())
                    throw py::value_error("portable pairwise feature vector is too short");
                const int base_index = feature_indices[node.feature];
                if (base_index < 0 || base_index >= (int)left_values.size() ||
                    base_index >= (int)right_values.size())
                    throw py::value_error("portable base feature vector is too short");
                const double left = left_values[base_index];
                const double right = right_values[base_index];
                const double value = modes[node.feature] == 0
                    ? left - right
                    : (left + right) / 2.0;
                const bool go_left = (std::isnan(value) && node.missing_go_to_left) ||
                                     (!std::isnan(value) && value <= node.threshold);
                node_index = go_left ? node.left : node.right;
            }
            raw += nodes.at(node_index).value;
        }
        return raw;
    }

    double baseline;
    vector<vector<PortableTreeNode>> trees;
};

namespace
{
py::dict stateToPython(const RepairState& state, const py::dict& context)
{
    py::dict result;
    result["initialized"] = state.initialized;
    result["initial_solution_complete"] = state.initial_solution_complete;
    result["feasible"] = state.feasible;
    result["done"] = state.done;
    result["iteration"] = state.iteration;
    result["rows"] = state.rows;
    result["cols"] = state.cols;
    result["sum_of_costs"] = state.sum_of_costs;
    result["num_of_colliding_pairs"] = state.num_of_colliding_pairs;
    result["runtime"] = state.runtime;
    result["context"] = context;

    py::dict low_level;
    low_level["expanded"] = state.low_level_expanded;
    low_level["generated"] = state.low_level_generated;
    low_level["reopened"] = state.low_level_reopened;
    low_level["runs"] = state.low_level_runs;
    result["low_level"] = low_level;

    py::list obstacles;
    for (unsigned char value : state.obstacles)
        obstacles.append((int)value);
    result["obstacles"] = obstacles;

    py::list edges;
    for (const auto& edge : state.conflict_edges)
        edges.append(py::make_tuple(edge.first, edge.second));
    result["conflict_edges"] = edges;

    py::list agents;
    for (const auto& agent : state.agents)
    {
        py::dict value;
        value["id"] = agent.id;
        value["start"] = agent.start;
        value["goal"] = agent.goal;
        value["path_cost"] = agent.path_cost;
        value["shortest_path_cost"] = agent.shortest_path_cost;
        value["delay"] = agent.delay;
        value["conflict_degree"] = agent.conflict_degree;
        value["path"] = agent.path;
        agents.append(value);
    }
    result["agents"] = agents;
    return result;
}

RepairHeuristic parseHeuristic(const std::string& value)
{
    if (value == "adaptive") return RepairHeuristic::ADAPTIVE;
    if (value == "target") return RepairHeuristic::TARGET;
    if (value == "collision") return RepairHeuristic::COLLISION;
    if (value == "random") return RepairHeuristic::RANDOM;
    throw py::value_error("unknown repair heuristic: " + value);
}

RepairAction parseAction(const py::dict& value)
{
    RepairAction action;
    std::string mode = value.contains("mode") ? py::cast<std::string>(value["mode"]) : "official";
    if (mode == "official")
        action.mode = RepairActionMode::OFFICIAL;
    else if (mode == "seed")
        action.mode = RepairActionMode::SEED;
    else if (mode == "explicit_neighborhood")
        action.mode = RepairActionMode::EXPLICIT_NEIGHBORHOOD;
    else if (mode == "replay_neighborhood")
        action.mode = RepairActionMode::REPLAY_NEIGHBORHOOD;
    else
        throw py::value_error("unknown repair action mode: " + mode);

    if (value.contains("heuristic"))
        action.heuristic = parseHeuristic(py::cast<std::string>(value["heuristic"]));
    if (value.contains("seed_agent"))
        action.seed_agent = py::cast<int>(value["seed_agent"]);
    if (value.contains("neighborhood_size"))
        action.neighborhood_size = py::cast<int>(value["neighborhood_size"]);
    if (value.contains("random_seed"))
    {
        action.random_seed = py::cast<int>(value["random_seed"]);
        if (action.random_seed < 0)
            throw py::value_error("random_seed must be non-negative");
    }
    if (value.contains("pp_random_seed"))
    {
        action.pp_random_seed = py::cast<int>(value["pp_random_seed"]);
        if (action.pp_random_seed < 0)
            throw py::value_error("pp_random_seed must be non-negative");
    }
    if (value.contains("agents"))
        action.agents = py::cast<vector<int>>(value["agents"]);
    if (value.contains("repair_order"))
        action.repair_order = py::cast<vector<int>>(value["repair_order"]);
    return action;
}

py::dict transitionToPython(const RepairTransition& transition)
{
    py::dict result;
    result["iteration"] = transition.iteration;
    result["requested_mode"] = repairActionModeName(transition.requested_action.mode);
    result["requested_heuristic"] = repairHeuristicName(transition.requested_action.heuristic);
    result["requested_random_seed"] = transition.requested_action.random_seed;
    result["requested_pp_random_seed"] = transition.requested_action.pp_random_seed;
    result["applied_pp_random_seed"] = transition.applied_pp_random_seed;
    result["requested_repair_order"] = transition.requested_action.repair_order;
    result["applied_heuristic"] = repairHeuristicName(transition.applied_heuristic);
    result["action_valid"] = transition.action_valid;
    result["generated"] = transition.generated;
    result["replan_success"] = transition.replan_success;
    result["neighborhood"] = transition.neighborhood;
    result["repair_order"] = transition.repair_order;
    result["conflicts_before"] = transition.conflicts_before;
    result["conflicts_after"] = transition.conflicts_after;
    result["conflict_delta"] = transition.conflicts_before - transition.conflicts_after;
    result["sum_of_costs_before"] = transition.sum_of_costs_before;
    result["sum_of_costs_after"] = transition.sum_of_costs_after;
    result["runtime_before"] = transition.runtime_before;
    result["runtime_after"] = transition.runtime_after;
    result["step_runtime"] = transition.runtime_after - transition.runtime_before;
    result["native_step_seconds"] = transition.native_step_seconds;
    result["native_neighborhood_generation_seconds"] =
        transition.neighborhood_generation_seconds;
    result["native_replan_seconds"] = transition.replan_seconds;
    result["pp_replan_seconds"] = transition.pp_replan_seconds;
    result["native_state_snapshot_seconds"] = transition.state_snapshot_seconds;
    result["native_repair_bookkeeping_seconds"] =
        transition.repair_bookkeeping_seconds;
    result["native_residual_seconds"] = transition.native_residual_seconds;
    return result;
}

py::dict proposalToPython(const RepairProposal& proposal)
{
    py::dict result;
    result["requested_mode"] = repairActionModeName(proposal.requested_action.mode);
    result["requested_heuristic"] = repairHeuristicName(proposal.requested_action.heuristic);
    result["requested_random_seed"] = proposal.requested_action.random_seed;
    result["requested_pp_random_seed"] = proposal.requested_action.pp_random_seed;
    result["requested_repair_order"] = proposal.requested_action.repair_order;
    result["applied_heuristic"] = repairHeuristicName(proposal.applied_heuristic);
    result["action_valid"] = proposal.action_valid;
    result["generated"] = proposal.generated;
    result["neighborhood"] = proposal.neighborhood;
    return result;
}
}

class LNS2RepairEnv
{
public:
    LNS2RepairEnv(const std::string& map_path, const std::string& scenario_path,
                  int agent_count, double time_limit, int neighborhood_size,
                  const std::string& destroy_strategy, const std::string& replan_algorithm,
                  bool use_sipp, int max_repair_iterations, int screen, py::dict context) :
        instance(new Instance(map_path, scenario_path, agent_count)),
        time_limit(time_limit), neighborhood_size(neighborhood_size),
        destroy_strategy(destroy_strategy), replan_algorithm(replan_algorithm),
        use_sipp(use_sipp), max_repair_iterations(max_repair_iterations), screen(screen),
        context(std::move(context))
    {
        if (destroy_strategy != "Adaptive" && destroy_strategy != "Target" &&
            destroy_strategy != "Collision" && destroy_strategy != "Random")
            throw py::value_error("destroy_strategy must be Adaptive, Target, Collision, or Random");
        if (replan_algorithm != "PP" && replan_algorithm != "GCBS" && replan_algorithm != "PBS")
            throw py::value_error("replan_algorithm must be PP, GCBS, or PBS");
    }

    py::dict reset(int seed)
    {
        const auto reset_started = DiagnosticClock::now();
        const auto setup_started = DiagnosticClock::now();
        ProcessGlobalRngState& rng_state = processGlobalRngState();
        std::unique_lock<std::mutex> rng_lock(rng_state.mutex);
        double setup_seconds = 0.0;
        double initial_solution_seconds = 0.0;
        double state_snapshot_seconds = 0.0;
        RepairState state;
        try
        {
            solver.reset();
            agents.clear();
            proposal_since_step = false;
            state_revision = 0;
            const int count = instance->getDefaultNumberOfAgents();
            if ((int)agents.capacity() < count)
                agents.reserve(count);
            for (int id = 0; id < count; id++)
                agents.emplace_back(*instance, id, use_sipp);
            srand(seed);
            solver.reset(new InitLNS(*instance, agents, time_limit, replan_algorithm,
                                     destroy_strategy, neighborhood_size, screen,
                                     nullptr, nullptr, max_repair_iterations));
            setup_seconds = diagnosticSeconds(setup_started);
            const auto initialize_started = DiagnosticClock::now();
            solver->initialize();
            initial_solution_seconds = diagnosticSeconds(initialize_started);
            claimRngOwnership(rng_state);
            state_revision = 1;
            const auto snapshot_started = DiagnosticClock::now();
            state = solver->getRepairState();
            state_snapshot_seconds = diagnosticSeconds(snapshot_started);
        }
        catch (...)
        {
            invalidateRngOwnership(rng_state);
            throw;
        }
        rng_lock.unlock();
        const auto export_started = DiagnosticClock::now();
        py::dict observation = stateToPython(state, context);
        const double state_to_python_seconds = diagnosticSeconds(export_started);
        last_reset_timings = py::dict();
        last_reset_timings["agent_and_solver_setup_seconds"] = setup_seconds;
        last_reset_timings["initial_solution_seconds"] = initial_solution_seconds;
        last_reset_timings["state_snapshot_seconds"] = state_snapshot_seconds;
        last_reset_timings["state_to_python_seconds"] = state_to_python_seconds;
        last_reset_timings["reset_total_seconds"] = diagnosticSeconds(reset_started);
        return observation;
    }

    py::dict step(const py::dict& action_value)
    {
        const auto binding_started = DiagnosticClock::now();
        if (!solver)
            throw std::runtime_error("reset() must be called before step()");
        RepairAction action = parseAction(action_value);
        ProcessGlobalRngState& rng_state = processGlobalRngState();
        std::unique_lock<std::mutex> rng_lock(rng_state.mutex);
        if (solver->isDone())
            throw std::runtime_error("the repair episode is already finished");
        if (proposal_since_step && action.random_seed < 0)
            throw py::value_error(
                "step() after propose() requires an explicit random_seed because proposal generation "
                "advances the process-global LNS2 random stream");
        if (action.random_seed < 0 && !ownsRng(rng_state))
            throw py::value_error(
                "step() without random_seed cannot continue because another LNS2RepairEnv changed "
                "the process-global LNS2 random stream; pass an explicit random_seed to recover "
                "RNG ownership");
        double solver_call_seconds = 0.0;
        double state_snapshot_seconds = 0.0;
        RepairState state;
        RepairTransition transition;
        try
        {
            const auto solver_started = DiagnosticClock::now();
            solver->step(action);
            solver_call_seconds = diagnosticSeconds(solver_started);
            claimRngOwnership(rng_state);
            proposal_since_step = false;
            state_revision++;
            const auto snapshot_started = DiagnosticClock::now();
            state = solver->getRepairState();
            state_snapshot_seconds = diagnosticSeconds(snapshot_started);
            transition = solver->getLastTransition();
        }
        catch (...)
        {
            invalidateRngOwnership(rng_state);
            throw;
        }
        rng_lock.unlock();
        const auto export_started = DiagnosticClock::now();
        py::dict observation = stateToPython(state, context);
        const double state_to_python_seconds = diagnosticSeconds(export_started);
        const auto metrics_started = DiagnosticClock::now();
        py::dict metrics = transitionToPython(transition);
        const double metrics_to_python_seconds = diagnosticSeconds(metrics_started);
        py::dict result;
        result["observation"] = observation;
        result["metrics"] = metrics;
        result["terminated"] = state.feasible;
        result["truncated"] = state.done && !state.feasible;
        const double binding_total_seconds = diagnosticSeconds(binding_started);
        metrics["binding_solver_call_seconds"] = solver_call_seconds;
        metrics["binding_state_snapshot_seconds"] = state_snapshot_seconds;
        metrics["state_to_python_seconds"] = state_to_python_seconds;
        metrics["metrics_to_python_seconds"] = metrics_to_python_seconds;
        metrics["binding_total_seconds"] = binding_total_seconds;
        metrics["binding_residual_seconds"] = std::max(
            0.0,
            binding_total_seconds - solver_call_seconds - state_snapshot_seconds -
                state_to_python_seconds - metrics_to_python_seconds);
        return result;
    }

    py::dict propose(const py::dict& action_value)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before propose()");
        proposal_since_step = true;
        const RepairAction action = parseAction(action_value);
        ProcessGlobalRngState& rng_state = processGlobalRngState();
        std::unique_lock<std::mutex> rng_lock(rng_state.mutex);
        RepairProposal proposal;
        try
        {
            proposal = solver->proposeNeighborhood(action);
            if (proposal.action_valid)
                claimRngOwnership(rng_state);
        }
        catch (...)
        {
            invalidateRngOwnership(rng_state);
            throw;
        }
        rng_lock.unlock();
        return proposalToPython(proposal);
    }

    py::list proposeBatch(const py::list& action_values)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before propose_batch()");
        if (!action_values.empty())
            proposal_since_step = true;
        ProcessGlobalRngState& rng_state = processGlobalRngState();
        std::unique_lock<std::mutex> rng_lock(rng_state.mutex);
        vector<RepairProposal> proposals;
        proposals.reserve(py::len(action_values));
        bool rng_touched = false;
        try
        {
            for (const py::handle& value : action_values)
            {
                const py::dict action_value = py::cast<py::dict>(value);
                RepairProposal proposal = solver->proposeNeighborhood(parseAction(action_value));
                rng_touched = rng_touched || proposal.action_valid;
                proposals.push_back(std::move(proposal));
            }
            if (rng_touched)
                claimRngOwnership(rng_state);
        }
        catch (...)
        {
            if (rng_touched)
                claimRngOwnership(rng_state);
            else
                invalidateRngOwnership(rng_state);
            throw;
        }
        rng_lock.unlock();
        py::list results;
        for (const RepairProposal& proposal : proposals)
            results.append(proposalToPython(proposal));
        return results;
    }

    py::list proposeBatchCompact(const py::list& action_values)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before propose_batch_compact()");
        if (!action_values.empty())
            proposal_since_step = true;
        vector<RepairAction> actions;
        actions.reserve(py::len(action_values));
        for (const py::handle& value : action_values)
        {
            const py::dict action_value = py::cast<py::dict>(value);
            actions.push_back(parseAction(action_value));
        }
        ProcessGlobalRngState& rng_state = processGlobalRngState();
        std::unique_lock<std::mutex> rng_lock(rng_state.mutex);
        vector<RepairProposal> proposals;
        try
        {
            proposals = solver->proposeNeighborhoodBatch(actions);
            if (std::any_of(
                    proposals.begin(), proposals.end(),
                    [](const RepairProposal& proposal) { return proposal.action_valid; }))
                claimRngOwnership(rng_state);
        }
        catch (...)
        {
            invalidateRngOwnership(rng_state);
            throw;
        }
        rng_lock.unlock();
        py::list results;
        for (const RepairProposal& proposal : proposals)
        {
            py::tuple compact(3);
            compact[0] = py::bool_(proposal.action_valid);
            compact[1] = py::bool_(proposal.generated);
            compact[2] = py::cast(proposal.neighborhood);
            results.append(std::move(compact));
        }
        return results;
    }

    py::dict getState() const
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before get_state()");
        return stateToPython(solver->getRepairState(), context);
    }

    uint64_t getStateRevision() const
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before get_state_revision()");
        return state_revision;
    }

    py::dict getLastResetTimings() const
    {
        return py::dict(last_reset_timings);
    }

private:
    bool ownsRng(const ProcessGlobalRngState& rng_state) const
    {
        return rng_state.owner_environment_id == environment_id &&
               rng_state.epoch == owned_rng_epoch;
    }

    void claimRngOwnership(ProcessGlobalRngState& rng_state)
    {
        rng_state.epoch++;
        if (rng_state.epoch == 0)
            rng_state.epoch++;
        rng_state.owner_environment_id = environment_id;
        owned_rng_epoch = rng_state.epoch;
    }

    void invalidateRngOwnership(ProcessGlobalRngState& rng_state)
    {
        rng_state.epoch++;
        if (rng_state.epoch == 0)
            rng_state.epoch++;
        rng_state.owner_environment_id = 0;
        owned_rng_epoch = 0;
    }

    const uint64_t environment_id = nextEnvironmentId();
    uint64_t owned_rng_epoch = 0;
    std::unique_ptr<Instance> instance;
    vector<Agent> agents;
    std::unique_ptr<InitLNS> solver;
    double time_limit;
    int neighborhood_size;
    std::string destroy_strategy;
    std::string replan_algorithm;
    bool use_sipp;
    int max_repair_iterations;
    int screen;
    py::dict context;
    bool proposal_since_step = false;
    uint64_t state_revision = 0;
    py::dict last_reset_timings;
};

PYBIND11_MODULE(lns2_env, module)
{
    module.doc() = "Step-wise MAPF-LNS2 collision-repair environment";
    module.attr("repair_timing_schema") = "lns2.repair_timing.v1";
    py::class_<PortableTreeEnsemble>(module, "PortableTreeEnsemble")
        .def(py::init<double, const py::list&>(), py::arg("baseline"), py::arg("trees"))
        .def("predict_raw", &PortableTreeEnsemble::predictRaw, py::arg("vectors"))
        .def("predict_positive", &PortableTreeEnsemble::predictPositive, py::arg("vectors"))
        .def("score_pairwise_dense", &PortableTreeEnsemble::scorePairwiseDense,
             py::arg("rows"), py::arg("modes"), py::arg("feature_indices"));
    py::class_<LNS2RepairEnv>(module, "LNS2RepairEnv")
        .def(py::init<const std::string&, const std::string&, int, double, int,
                      const std::string&, const std::string&, bool, int, int, py::dict>(),
             py::arg("map_path"), py::arg("scenario_path"), py::arg("agent_count") = 0,
             py::arg("time_limit") = 60.0, py::arg("neighborhood_size") = 8,
             py::arg("destroy_strategy") = "Adaptive", py::arg("replan_algorithm") = "PP",
             py::arg("use_sipp") = true, py::arg("max_repair_iterations") = 0,
             py::arg("screen") = 0, py::arg("context") = py::dict())
        .def("reset", &LNS2RepairEnv::reset, py::arg("seed") = 0)
        .def("propose", &LNS2RepairEnv::propose, py::arg("action"))
        .def("propose_batch", &LNS2RepairEnv::proposeBatch, py::arg("actions"))
        .def("propose_batch_compact", &LNS2RepairEnv::proposeBatchCompact,
             py::arg("actions"))
        .def("step", &LNS2RepairEnv::step, py::arg("action"))
        .def("get_state", &LNS2RepairEnv::getState)
        .def("get_state_revision", &LNS2RepairEnv::getStateRevision)
        .def("get_last_reset_timings", &LNS2RepairEnv::getLastResetTimings);
    module.def("batch_online_features", &batchOnlineFeatures,
               py::arg("state"), py::arg("candidates"), py::arg("static_grid"),
               py::arg("include_realized") = true,
               py::arg("required_features") = py::dict());
    module.def("batch_online_feature_vectors", &batchOnlineFeatureVectors,
               py::arg("state"), py::arg("candidates"), py::arg("static_grid"),
               py::arg("feature_names"));
}
