#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "InitLNS.h"

#include <memory>
#include <stdexcept>

namespace py = pybind11;

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
        solver->step(parseAction(action_value));
        RepairState state = solver->getRepairState();
        py::dict result;
        result["observation"] = stateToPython(state, context);
        result["metrics"] = transitionToPython(solver->getLastTransition());
        result["terminated"] = state.feasible;
        result["truncated"] = state.done && !state.feasible;
        return result;
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
};

PYBIND11_MODULE(lns2_env, module)
{
    module.doc() = "Step-wise MAPF-LNS2 collision-repair environment";
    py::class_<LNS2RepairEnv>(module, "LNS2RepairEnv")
        .def(py::init<const std::string&, const std::string&, int, double, int,
                      const std::string&, const std::string&, bool, int, int, py::dict>(),
             py::arg("map_path"), py::arg("scenario_path"), py::arg("agent_count") = 0,
             py::arg("time_limit") = 60.0, py::arg("neighborhood_size") = 8,
             py::arg("destroy_strategy") = "Adaptive", py::arg("replan_algorithm") = "PP",
             py::arg("use_sipp") = true, py::arg("max_repair_iterations") = 0,
             py::arg("screen") = 0, py::arg("context") = py::dict())
        .def("reset", &LNS2RepairEnv::reset, py::arg("seed") = 0)
        .def("step", &LNS2RepairEnv::step, py::arg("action"))
        .def("get_state", &LNS2RepairEnv::getState);
}
