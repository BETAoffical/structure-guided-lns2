#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "InitLNS.h"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

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

    vector<double> predictPositive(const vector<vector<double>>& vectors) const
    {
        vector<double> probabilities;
        probabilities.reserve(vectors.size());
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

private:
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
    if (value.contains("agents"))
        action.agents = py::cast<vector<int>>(value["agents"]);
    return action;
}

py::dict transitionToPython(const RepairTransition& transition)
{
    py::dict result;
    result["iteration"] = transition.iteration;
    result["requested_mode"] = repairActionModeName(transition.requested_action.mode);
    result["requested_heuristic"] = repairHeuristicName(transition.requested_action.heuristic);
    result["requested_random_seed"] = transition.requested_action.random_seed;
    result["applied_heuristic"] = repairHeuristicName(transition.applied_heuristic);
    result["action_valid"] = transition.action_valid;
    result["generated"] = transition.generated;
    result["replan_success"] = transition.replan_success;
    result["neighborhood"] = transition.neighborhood;
    result["conflicts_before"] = transition.conflicts_before;
    result["conflicts_after"] = transition.conflicts_after;
    result["conflict_delta"] = transition.conflicts_before - transition.conflicts_after;
    result["sum_of_costs_before"] = transition.sum_of_costs_before;
    result["sum_of_costs_after"] = transition.sum_of_costs_after;
    result["runtime_before"] = transition.runtime_before;
    result["runtime_after"] = transition.runtime_after;
    result["step_runtime"] = transition.runtime_after - transition.runtime_before;
    return result;
}

py::dict proposalToPython(const RepairProposal& proposal)
{
    py::dict result;
    result["requested_mode"] = repairActionModeName(proposal.requested_action.mode);
    result["requested_heuristic"] = repairHeuristicName(proposal.requested_action.heuristic);
    result["requested_random_seed"] = proposal.requested_action.random_seed;
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
        solver.reset();
        agents.clear();
        proposal_since_step = false;
        const int count = instance->getDefaultNumberOfAgents();
        if ((int)agents.capacity() < count)
            agents.reserve(count);
        for (int id = 0; id < count; id++)
            agents.emplace_back(*instance, id, use_sipp);
        srand(seed);
        solver.reset(new InitLNS(*instance, agents, time_limit, replan_algorithm,
                                 destroy_strategy, neighborhood_size, screen,
                                 nullptr, nullptr, max_repair_iterations));
        solver->initialize();
        return stateToPython(solver->getRepairState(), context);
    }

    py::dict step(const py::dict& action_value)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before step()");
        if (solver->isDone())
            throw std::runtime_error("the repair episode is already finished");
        RepairAction action = parseAction(action_value);
        if (proposal_since_step && action.random_seed < 0)
            throw py::value_error(
                "step() after propose() requires an explicit random_seed because proposal generation "
                "advances the process-global LNS2 random stream");
        solver->step(action);
        proposal_since_step = false;
        RepairState state = solver->getRepairState();
        py::dict result;
        result["observation"] = stateToPython(state, context);
        result["metrics"] = transitionToPython(solver->getLastTransition());
        result["terminated"] = state.feasible;
        result["truncated"] = state.done && !state.feasible;
        return result;
    }

    py::dict propose(const py::dict& action_value)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before propose()");
        proposal_since_step = true;
        return proposalToPython(solver->proposeNeighborhood(parseAction(action_value)));
    }

    py::list proposeBatch(const py::list& action_values)
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before propose_batch()");
        if (!action_values.empty())
            proposal_since_step = true;
        py::list results;
        for (const py::handle& value : action_values)
        {
            const py::dict action_value = py::cast<py::dict>(value);
            results.append(proposalToPython(solver->proposeNeighborhood(parseAction(action_value))));
        }
        return results;
    }

    py::dict getState() const
    {
        if (!solver)
            throw std::runtime_error("reset() must be called before get_state()");
        return stateToPython(solver->getRepairState(), context);
    }

private:
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
};

PYBIND11_MODULE(lns2_env, module)
{
    module.doc() = "Step-wise MAPF-LNS2 collision-repair environment";
    py::class_<PortableTreeEnsemble>(module, "PortableTreeEnsemble")
        .def(py::init<double, const py::list&>(), py::arg("baseline"), py::arg("trees"))
        .def("predict_positive", &PortableTreeEnsemble::predictPositive, py::arg("vectors"));
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
        .def("step", &LNS2RepairEnv::step, py::arg("action"))
        .def("get_state", &LNS2RepairEnv::getState);
}
