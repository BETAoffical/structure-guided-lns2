#include <boost/program_options.hpp>

#include "LNS.h"
#include "PIBT/pibt.h"
#include "structure_guided/instance_validation.hpp"
#include "structure_guided/jsonl_observer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace
{
RepairState terminalState(const Instance& instance, const LNS& solver, bool success)
{
    RepairState state;
    state.initialized = true;
    state.initial_solution_complete = success;
    state.feasible = success;
    state.done = true;
    state.rows = instance.num_of_rows;
    state.cols = instance.num_of_cols;
    state.sum_of_costs = solver.sum_of_costs;
    state.runtime = solver.runtime;
    for (int location = 0; location < instance.map_size; location++)
        state.obstacles.push_back(instance.isObstacle(location) ? 1 : 0);
    for (const auto& agent : solver.agents)
    {
        RepairAgentState value;
        value.id = agent.id;
        value.start = agent.path_planner->start_location;
        value.goal = agent.path_planner->goal_location;
        value.shortest_path_cost =
            agent.path_planner->my_heuristic[agent.path_planner->start_location];
        value.path_cost = agent.path.empty() ? -1 : (int)agent.path.size() - 1;
        value.delay = agent.path.empty() ? -1 : agent.getNumOfDelays();
        for (const auto& entry : agent.path)
            value.path.push_back(entry.location);
        state.agents.push_back(std::move(value));
        state.low_level_expanded += agent.path_planner->getTotalNumExpanded();
        state.low_level_generated += agent.path_planner->getTotalNumGenerated();
        state.low_level_reopened += agent.path_planner->getTotalNumReopened();
        state.low_level_runs += agent.path_planner->getTotalNumRuns();
    }
    return state;
}
}

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    po::options_description options("LNS2 repair-only options");
    options.add_options()
        ("help", "show options")
        ("map,m", po::value<string>()->required(), "MovingAI map file")
        ("agents,a", po::value<string>()->required(), "MovingAI scenario file")
        ("agentNum,k", po::value<int>()->default_value(0), "number of agents")
        ("cutoffTime,t", po::value<double>()->default_value(60), "time limit in seconds")
        ("seed", po::value<int>()->default_value(0), "random seed")
        ("neighborSize", po::value<int>()->default_value(8), "neighborhood size")
        ("maxRepairIterations", po::value<int>()->default_value(0), "repair iteration limit")
        ("replanAlgo", po::value<string>()->default_value("PP"), "PP, GCBS, or PBS")
        ("initDestroyStrategy", po::value<string>()->default_value("Adaptive"),
         "Target, Collision, Random, or Adaptive")
        ("sipp", po::value<bool>()->default_value(true), "use SIPP/SIPPS")
        ("screen,s", po::value<int>()->default_value(0), "logging level")
        ("trace", po::value<string>(), "versioned JSONL repair trace")
        ("output,o", po::value<string>(), "CSV output prefix")
        ("outputPaths", po::value<string>(), "solution paths output");

    po::variables_map values;
    try
    {
        po::store(po::parse_command_line(argc, argv, options), values);
        if (values.count("help"))
        {
            cout << options << endl;
            return 0;
        }
        po::notify(values);
    }
    catch (const std::exception& error)
    {
        cerr << error.what() << endl << options << endl;
        return 2;
    }

    try
    {
        const string map_path = values["map"].as<string>();
        const string scenario_path = values["agents"].as<string>();
        const int agent_count = values["agentNum"].as<int>();
        const double cutoff_time = values["cutoffTime"].as<double>();
        const int neighborhood_size = values["neighborSize"].as<int>();
        const int max_repair_iterations =
            values["maxRepairIterations"].as<int>();
        const string replan_algorithm = values["replanAlgo"].as<string>();
        const string destroy_strategy =
            values["initDestroyStrategy"].as<string>();

        if (!std::isfinite(cutoff_time) || cutoff_time < 0)
            throw std::invalid_argument(
                "cutoffTime must be finite and non-negative"
            );
        if (neighborhood_size <= 0)
            throw std::invalid_argument(
                "neighborSize must be greater than zero"
            );
        if (max_repair_iterations < 0)
            throw std::invalid_argument(
                "maxRepairIterations must be non-negative"
            );
        if (replan_algorithm != "PP" && replan_algorithm != "GCBS" &&
            replan_algorithm != "PBS")
            throw std::invalid_argument(
                "replanAlgo must be PP, GCBS, or PBS"
            );
        if (destroy_strategy != "Adaptive" &&
            destroy_strategy != "Target" &&
            destroy_strategy != "Collision" &&
            destroy_strategy != "Random")
            throw std::invalid_argument(
                "initDestroyStrategy must be Adaptive, Target, Collision, or Random"
            );
        structure_guided::validateInstanceFiles(
            map_path, scenario_path, agent_count
        );

        srand(values["seed"].as<int>());
        Instance instance(map_path, scenario_path, agent_count);
        std::unique_ptr<JsonlRepairObserver> observer;
        if (values.count("trace"))
            observer.reset(
                new JsonlRepairObserver(values["trace"].as<string>())
            );

        PIBTPPS_option pibt_options;
        pibt_options.windowSize = 5;
        pibt_options.winPIBTSoft = true;
        pibt_options.timestepLimit = 0;
        LNS solver(instance, cutoff_time, "PP", replan_algorithm, "Adaptive",
                   neighborhood_size, 0, true, destroy_strategy,
                   values["sipp"].as<bool>(), values["screen"].as<int>(),
                   pibt_options, true, nullptr, observer.get(),
                   max_repair_iterations);
        const bool success = solver.run();
        if (observer && solver.getInitLNS() == nullptr)
        {
            RepairState state = terminalState(instance, solver, success);
            observer->onInitialState(state);
            observer->onFinish(state, success);
        }
        if (success)
        {
            solver.validateSolution();
            if (values.count("outputPaths"))
                solver.writePathsToFile(values["outputPaths"].as<string>());
        }
        if (values.count("output"))
            solver.writeResultToFile(values["output"].as<string>());
        return success ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        cerr << "lns2_repair: " << error.what() << endl;
        return 2;
    }
}
