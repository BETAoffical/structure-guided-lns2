#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <fstream>
#include <memory>
#include <set>
#include "SIPP.h"

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
using TimedConstraint = std::tuple<std::string, int, int, int>;

class NativePathProbe
{
    std::unique_ptr<Instance> instance;
    std::vector<Path> original;
    std::vector<std::unique_ptr<SIPP>> solvers;

    void check_id(int id) const
    {
        if (id < 0 || id >= static_cast<int>(original.size()))
            throw std::invalid_argument("unknown agent ID");
    }

    Path make_path(int id, const std::vector<int>& cells) const
    {
        if (cells.empty() || cells.front() != instance->getStarts().at(id) ||
            cells.back() != instance->getGoals().at(id))
            throw std::invalid_argument("path endpoints disagree with scenario");
        Path path(cells.size());
        for (size_t i = 0; i < cells.size(); ++i)
        {
            const int loc = cells[i];
            if (loc < 0 || loc >= instance->map_size || instance->isObstacle(loc) ||
                (i > 0 && !instance->validMove(cells[i-1], loc)))
                throw std::invalid_argument("illegal path cell or step");
            path[i].location = loc;
        }
        return path;
    }

public:
    NativePathProbe(const std::string& map_file, const std::string& scenario_file,
                    const std::vector<std::vector<int>>& paths)
    {
        if (!std::ifstream(map_file).good() || !std::ifstream(scenario_file).good() || paths.empty())
            throw std::invalid_argument("existing map/scenario and nonempty paths required");
        instance.reset(new Instance(map_file, scenario_file, static_cast<int>(paths.size())));
        original.resize(paths.size());
        solvers.resize(paths.size());
        for (size_t id = 0; id < paths.size(); ++id)
            original[id] = make_path(static_cast<int>(id), paths[id]);
    }

    void seed_rng(unsigned int seed) { srand(seed); }

    py::dict plan(int agent, const std::vector<int>& fixed_ids,
                  const std::map<int, std::vector<int>>& overrides, bool hard,
                  const std::vector<TimedConstraint>& constraints, int max_cost, double seconds)
    {
        const auto started = Clock::now();
        check_id(agent);
        if (!std::isfinite(seconds) || seconds < 0 || max_cost < -1)
            throw std::invalid_argument("invalid cost or wall budget");
        std::set<int> unique;
        for (int id : fixed_ids)
        {
            check_id(id);
            if (id == agent || !unique.insert(id).second)
                throw std::invalid_argument("self or duplicate fixed agent");
        }
        std::vector<Path> paths(original);
        for (const auto& item : overrides)
        {
            check_id(item.first);
            paths[item.first] = make_path(item.first, item.second);
        }
        PathTableWC table(instance->map_size, static_cast<int>(paths.size()));
        ConstraintTable ct(instance->num_of_cols, instance->map_size, nullptr, hard ? nullptr : &table);
        if (max_cost >= 0)
            ct.length_max = max_cost;
        for (int id : fixed_ids)
        {
            if (!hard)
                table.insertPath(id, paths[id]);
            else
            {
                // Preserve every explicit wait and permanent terminal occupancy.
                int from = 0;
                const Path& p = paths[id];
                for (int t = 1; t < static_cast<int>(p.size()); ++t)
                {
                    if (p[t].location == p[t-1].location)
                        continue;
                    ct.insert2CT(p[t-1].location, from, t);
                    ct.insert2CT(p[t].location, p[t-1].location, t, t+1);
                    from = t;
                }
                ct.insert2CT(p.back().location, from, MAX_TIMESTEP);
            }
        }
        for (const auto& constraint : constraints)
        {
            std::string kind;
            int t, u, v;
            std::tie(kind, t, u, v) = constraint;
            if (t < 0 || t >= MAX_TIMESTEP-1 || u < 0 || v < 0 ||
                u >= instance->map_size || v >= instance->map_size ||
                instance->isObstacle(u) || instance->isObstacle(v))
                throw std::invalid_argument("invalid constraint cell or time");
            if (kind == "vertex" && u == v)
                ct.insert2CT(v, t, t+1);
            else if (kind == "edge" && t > 0 && instance->validMove(u, v))
                ct.insert2CT(u, v, t, t+1);
            else
                throw std::invalid_argument("invalid constraint kind or edge");
        }
        if (!solvers[agent])
            solvers[agent].reset(new SIPP(*instance, agent));
        const auto search_start = Clock::now();
        const double preparation = std::chrono::duration<double>(search_start - started).count();
        Path path;
        bool timed_out = preparation >= seconds;
        if (!timed_out)
        {
            path = solvers[agent]->findPath(ct, seconds-preparation);
            timed_out = solvers[agent]->last_find_path_timed_out;
        }
        py::dict result;
        std::vector<int> cells;
        for (const auto& step : path)
            cells.push_back(step.location);
        if (!cells.empty() && max_cost >= 0 && static_cast<int>(cells.size())-1 > max_cost)
            throw std::runtime_error("native path exceeds explicit cost cap");
        result["status"] = timed_out ? "unknown" : (cells.empty() ? "empty" : "path");
        result["path"] = cells;
        result["cost"] = cells.empty() ? -1 : static_cast<int>(cells.size())-1;
        result["expanded"] = preparation >= seconds ? 0 : solvers[agent]->getNumExpanded();
        result["generated"] = preparation >= seconds ? 0 : solvers[agent]->getNumGenerated();
        result["low_level_collisions"] = cells.empty() ? -1 : solvers[agent]->num_collisions;
        result["search_seconds"] = std::chrono::duration<double>(Clock::now()-search_start).count();
        result["wrapper_seconds"] = std::chrono::duration<double>(Clock::now()-started).count();
        return result;
    }
};

PYBIND11_MODULE(lns2_path_probe_native, m)
{
    m.attr("schema") = "lns2.native_path_probe.v1";
    m.attr("semantics") = "isolated_diagnostic_hard_or_CAT_no_InitLNS_step";
    py::class_<NativePathProbe>(m, "NativePathProbe")
        .def(py::init<const std::string&, const std::string&, const std::vector<std::vector<int>>&>())
        .def("seed_rng", &NativePathProbe::seed_rng)
        .def("plan", &NativePathProbe::plan, py::arg("agent"), py::arg("fixed_ids"),
             py::arg("overrides"), py::arg("hard"), py::arg("constraints") = std::vector<TimedConstraint>(),
             py::arg("max_cost") = -1, py::arg("seconds") = 5.0);
}
