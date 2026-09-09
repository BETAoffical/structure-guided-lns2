cmake_minimum_required(VERSION 3.16)
project(lns2_path_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LNS2_FROZEN_CORE_FILE "" CACHE FILEPATH "Existing, hash-registered PIC core archive")
if(NOT EXISTS "${LNS2_FROZEN_CORE_FILE}")
    message(FATAL_ERROR "Specify the existing frozen libmapf_lns2_core.a; do not rebuild the core")
endif()
add_library(frozen_core STATIC IMPORTED)
set_target_properties(frozen_core PROPERTIES IMPORTED_LOCATION "${LNS2_FROZEN_CORE_FILE}")
pybind11_add_module(lns2_path_probe_native native_path_probe.cpp)
target_include_directories(lns2_path_probe_native PRIVATE
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc/CBS"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc/PIBT")
target_link_libraries(lns2_path_probe_native PRIVATE frozen_core
    Boost::program_options Boost::system Boost::filesystem Eigen3::Eigen)
