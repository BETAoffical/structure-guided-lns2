#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "observer.h"

namespace occupancy_observer {
thread_local std::unique_ptr<Capture> active;
}

// Reuse the frozen probe body without editing its registered source or binding type.
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, module) void register_probe(pybind11::module_& module)
#define NativePathProbe SearchObserverProbe
#include "../path_probe/native_path_probe.cpp"
#undef NativePathProbe
#pragma pop_macro("PYBIND11_MODULE")

PYBIND11_MODULE(lns2_search_observer_native, m) {
    register_probe(m);
    m.attr("observer_schema") = "lns2.offered_soft_intervals.v1";
    m.def("begin_observation", [](size_t limit) {
        if (limit > 65536) throw std::invalid_argument("observation limit exceeds 65536");
        if (occupancy_observer::active) throw std::logic_error("observation already active");
        occupancy_observer::active.reset(new occupancy_observer::Capture(limit));
    });
    m.def("end_observation", []() {
        if (!occupancy_observer::active) throw std::logic_error("no active observation");
        auto capture = std::move(occupancy_observer::active);
        pybind11::dict result;
        result["events"] = capture->events;
        result["offered"] = capture->offered;
        result["queries"] = capture->queries;
        result["truncated"] = capture->offered > capture->events.size();
        return result;
    });
}
