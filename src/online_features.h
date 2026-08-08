#pragma once

#include <pybind11/pybind11.h>

pybind11::dict batchOnlineFeatures(
    const pybind11::dict& state,
    const pybind11::list& candidates,
    const pybind11::dict& static_grid,
    bool include_realized,
    const pybind11::dict& required_features);

pybind11::dict batchOnlineFeatureVectors(
    const pybind11::dict& state,
    const pybind11::list& candidates,
    const pybind11::dict& static_grid,
    const pybind11::list& feature_names);

pybind11::dict topologyConflictEvents(
    const pybind11::dict& state,
    const pybind11::dict& static_grid);

pybind11::capsule prepareOnlineAnalysis(
    const pybind11::dict& state,
    const pybind11::dict& static_grid);

pybind11::dict preparedTopologyConflictEvents(
    const pybind11::capsule& capsule);

pybind11::dict batchOnlineFeaturesPrepared(
    const pybind11::dict& state,
    const pybind11::list& candidates,
    const pybind11::capsule& capsule,
    bool include_realized,
    const pybind11::dict& required_features);

pybind11::dict batchOnlineFeatureVectorsPrepared(
    const pybind11::dict& state,
    const pybind11::list& candidates,
    const pybind11::capsule& capsule,
    const pybind11::list& feature_names);
