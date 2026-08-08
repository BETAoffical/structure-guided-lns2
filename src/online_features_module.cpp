#include <pybind11/pybind11.h>

#include "online_features.h"

namespace py = pybind11;

PYBIND11_MODULE(lns2_features_native, module)
{
    module.doc() = "Native batched feature-v2 extractor";
    module.def("batch_online_features", &batchOnlineFeatures,
               py::arg("state"), py::arg("candidates"), py::arg("static_grid"),
               py::arg("include_realized") = true,
               py::arg("required_features") = py::dict());
    module.def("batch_online_feature_vectors", &batchOnlineFeatureVectors,
               py::arg("state"), py::arg("candidates"), py::arg("static_grid"),
               py::arg("feature_names"));
    module.def("topology_conflict_events", &topologyConflictEvents,
               py::arg("state"), py::arg("static_grid"));
    module.def("prepare_online_analysis", &prepareOnlineAnalysis,
               py::arg("state"), py::arg("static_grid"));
    module.def("prepared_topology_conflict_events",
               &preparedTopologyConflictEvents,
               py::arg("prepared_analysis"));
    module.def("batch_online_features_prepared",
               &batchOnlineFeaturesPrepared,
               py::arg("state"), py::arg("candidates"),
               py::arg("prepared_analysis"),
               py::arg("include_realized") = true,
               py::arg("required_features") = py::dict());
    module.def("batch_online_feature_vectors_prepared",
               &batchOnlineFeatureVectorsPrepared,
               py::arg("state"), py::arg("candidates"),
               py::arg("prepared_analysis"), py::arg("feature_names"));
}
