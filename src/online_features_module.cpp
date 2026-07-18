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
}
