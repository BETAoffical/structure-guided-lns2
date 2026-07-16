#pragma once
#include "BasicLNS.h"
#include "RepairPolicy.h"

enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };

class InitLNS : public BasicLNS
{
public:
    vector<Agent>& agents;
    int num_of_colliding_pairs = 0;

    InitLNS(const Instance& instance, vector<Agent>& agents, double time_limit,
            const string & replan_algo_name, const string & init_destory_name, int neighbor_size, int screen,
            NeighborhoodPolicy* policy = nullptr, RepairObserver* observer = nullptr,
            int max_repair_iterations = 0);

    bool getInitialSolution();
    bool initialize();
    bool step();
    bool step(const RepairAction& action);
    RepairProposal proposeNeighborhood(const RepairAction& action);
    bool run();
    bool isInitialized() const { return initialized; }
    bool isFeasible() const { return initialized && num_of_colliding_pairs == 0; }
    bool isDone() const;
    bool isInitialSolutionComplete() const { return initial_solution_complete; }
    RepairState getRepairState() const;
    const RepairTransition& getLastTransition() const { return last_transition; }
    const vector<set<int>>& getCollisionGraph() const { return collision_graph; }
    void setPolicy(NeighborhoodPolicy* value) { policy = value; }
    void setObserver(RepairObserver* value) { observer = value; }
    void writeIterStatsToFile(const string & file_name) const;
    void writeResultToFile(const string & file_name, int sum_of_distances, double preprocessing_time) const;
    string getSolverName() const override { return "InitLNS(" + replan_algo_name + ")"; }

    void printPath() const;
    void printResult();
    void clear(); // delete useless data to save memory

private:
    string replan_algo_name;
    init_destroy_heuristic init_destroy_strategy = COLLISION_BASED;

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.

    vector<set<int>> collision_graph;
    vector<int> goal_table;

    NeighborhoodPolicy* policy = nullptr;
    RepairObserver* observer = nullptr;
    int max_repair_iterations = 0;
    int repair_iteration = 0;
    bool initialized = false;
    bool initial_solution_complete = false;
    bool finish_notified = false;
    RepairTransition last_transition;


    bool runPP(const vector<int>& requested_order, vector<int>& applied_order);
    bool runGCBS();
    bool runPBS();

    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const;

    void chooseDestroyHeuristicbyALNS();

    bool generateNeighborhood(const RepairAction& action, RepairHeuristic& applied_heuristic,
                              bool& action_valid);
    bool generateOfficialNeighborhood(RepairHeuristic& applied_heuristic);
    bool validateExplicitNeighborhood(const vector<int>& values) const;
    bool validateRepairOrder(const vector<int>& order, const vector<int>& neighborhood) const;
    bool generateNeighborByCollisionGraph(int forced_seed = -1, int requested_size = 0);
    bool generateNeighborByTarget(int forced_seed = -1, int requested_size = 0);
    bool generateNeighborRandomly(int forced_seed = -1, int requested_size = 0);
    RepairHeuristic currentRepairHeuristic() const;
    void notifyFinish();

    // int findRandomAgent() const;
    int randomWalk(int agent_id);

    void printCollisionGraph() const;

    static unordered_map<int, set<int>>& findConnectedComponent(const vector<set<int>>& graph, int vertex,
            unordered_map<int, set<int>>& sub_graph);

    bool validatePathTable() const;
};
