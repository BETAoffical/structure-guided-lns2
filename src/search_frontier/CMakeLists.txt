cmake_minimum_required(VERSION 3.16)
project(lns2_search_frontier LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LNS2_FROZEN_CORE_FILE "" CACHE FILEPATH "Frozen PIC archive")
if(NOT EXISTS "${LNS2_FROZEN_CORE_FILE}")
    message(FATAL_ERROR "Pass the existing frozen PIC core archive")
endif()
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/instrument.py"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/src/SIPP.cpp" "${CMAKE_CURRENT_BINARY_DIR}/observed_sipp.cpp"
    RESULT_VARIABLE GENERATED_STATUS)
if(NOT GENERATED_STATUS EQUAL 0)
    message(FATAL_ERROR "Frozen source observation transform rejected")
endif()
add_library(frozen_core STATIC IMPORTED)
set_target_properties(frozen_core PROPERTIES IMPORTED_LOCATION "${LNS2_FROZEN_CORE_FILE}")
pybind11_add_module(lns2_search_frontier_native module.cpp "${CMAKE_CURRENT_BINARY_DIR}/observed_sipp.cpp")
target_include_directories(lns2_search_frontier_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc/CBS"
    "${REPOSITORY_ROOT}/third_party/mapf_lns2/inc/PIBT")
target_link_libraries(lns2_search_frontier_native PRIVATE frozen_core
    Boost::program_options Boost::system Boost::filesystem Eigen3::Eigen)
