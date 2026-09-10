#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "observer.h"
namespace frontier_observer { thread_local std::unique_ptr<Capture> active; }

#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) void register_probe(pybind11::module_& module)
#define NativePathProbe SearchFrontierProbe
#include "../path_probe/native_path_probe.cpp"
#undef NativePathProbe
#pragma pop_macro("PYBIND11_MODULE")

PYBIND11_MODULE(lns2_search_frontier_native, m) {
    register_probe(m);
    m.attr("observer_schema") = "lns2.queued_positive_transitions.v1";
    m.def("begin_observation", [](size_t limit) {
        if (limit > 32768) throw std::invalid_argument("observation limit exceeds 32768");
        if (frontier_observer::active) throw std::logic_error("observation already active");
        frontier_observer::active.reset(new frontier_observer::Capture(limit));
    });
    m.def("end_observation", []() {
        if (!frontier_observer::active) throw std::logic_error("no active observation");
        auto c = std::move(frontier_observer::active);
        pybind11::dict result;
        result["events"] = c->events;
        result["offered"] = c->offered;
        result["enqueued"] = c->enqueued;
        result["popped"] = c->popped;
        result["virtual_goals"] = c->virtual_goals;
        result["goal_conflicts"] = c->goal_conflicts;
        result["goal_cost"] = c->goal_cost;
        result["truncated"] = c->offered > c->events.size();
        return result;
    });
}
