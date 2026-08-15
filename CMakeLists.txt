cmake_minimum_required(VERSION 3.16)
project(structure_guided_lns2 LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type (default: Release, for representative solver timing)"
        FORCE
    )
    message(STATUS "CMAKE_BUILD_TYPE was not set; defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    LNS2_FEATURES_ONLY
    "Build only the standalone Python online-feature extension"
    OFF
)
if(LNS2_FEATURES_ONLY)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    pybind11_add_module(
        lns2_features_native
        src/online_features.cpp
        src/online_features_module.cpp
    )
    target_include_directories(lns2_features_native PRIVATE src)
    enable_testing()
    add_test(
        NAME lns2_features_native_smoke
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_features_native.py"
    )
    set_tests_properties(lns2_features_native_smoke PROPERTIES ENVIRONMENT
        "PYTHONPATH=$<TARGET_FILE_DIR:lns2_features_native>"
    )
    return()
endif()

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(MAPF_LNS2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mapf_lns2")
file(GLOB MAPF_LNS2_SOURCES CONFIGURE_DEPENDS
    "${MAPF_LNS2_ROOT}/src/*.cpp"
    "${MAPF_LNS2_ROOT}/src/CBS/*.cpp"
    "${MAPF_LNS2_ROOT}/src/PIBT/*.cpp"
)
list(REMOVE_ITEM MAPF_LNS2_SOURCES "${MAPF_LNS2_ROOT}/src/driver.cpp")

add_library(mapf_lns2_core ${MAPF_LNS2_SOURCES})
set_target_properties(mapf_lns2_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mapf_lns2_core PUBLIC
    "${MAPF_LNS2_ROOT}/inc"
    "${MAPF_LNS2_ROOT}/inc/CBS"
    "${MAPF_LNS2_ROOT}/inc/PIBT"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(mapf_lns2_core PUBLIC
    Boost::program_options
    Boost::system
    Boost::filesystem
    Eigen3::Eigen
)

add_library(
    structure_guided_instance_validation
    STATIC
    src/instance_validation.cpp
)
set_target_properties(
    structure_guided_instance_validation
    PROPERTIES POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    structure_guided_instance_validation
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(lns_official "${MAPF_LNS2_ROOT}/src/driver.cpp")
target_link_libraries(lns_official PRIVATE mapf_lns2_core)

add_executable(lns2_repair
    src/repair_driver.cpp
    src/jsonl_observer.cpp
)
target_link_libraries(
    lns2_repair
    PRIVATE mapf_lns2_core structure_guided_instance_validation
)

set(GPBS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gpbs")
file(GLOB GPBS_SOURCES CONFIGURE_DEPENDS "${GPBS_ROOT}/src/*.cpp")
add_executable(gpbs_official ${GPBS_SOURCES})
target_compile_features(gpbs_official PRIVATE cxx_std_17)
target_include_directories(gpbs_official PRIVATE "${GPBS_ROOT}/inc")
target_link_libraries(gpbs_official PRIVATE
    Boost::program_options
    Boost::system
    Boost::filesystem
)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
pybind11_add_module(lns2_env src/python_bindings.cpp src/online_features.cpp)
target_link_libraries(
    lns2_env
    PRIVATE mapf_lns2_core structure_guided_instance_validation
)

enable_testing()
set(LNS2_TEST_PYTHONPATH_SEPARATOR ":")
if(WIN32)
    # ENVIRONMENT is itself a CMake list, so keep the Windows path-list
    # separator escaped until CTest sets the variable.
    set(LNS2_TEST_PYTHONPATH_SEPARATOR "\\;")
endif()
add_executable(lns2_interface_tests tests/test_repair_interfaces.cpp)
target_link_libraries(lns2_interface_tests PRIVATE mapf_lns2_core)
target_compile_definitions(lns2_interface_tests PRIVATE
    TEST_MAP="${MAPF_LNS2_ROOT}/random-32-32-20.map"
    TEST_SCEN="${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    PROPOSAL_TEST_MAP="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/proposal_required.map"
    PROPOSAL_TEST_SCEN="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/proposal_required.scen"
)
add_test(NAME lns2_interface_tests COMMAND lns2_interface_tests)
add_test(NAME lns2_repair_smoke COMMAND lns2_repair
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 24
    --cutoffTime 10
    --seed 0
)
add_test(
    NAME lns2_failed_trace_state
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_failed_repair_trace.py"
            "$<TARGET_FILE:lns2_repair>"
            "${MAPF_LNS2_ROOT}/random-32-32-20.map"
            "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
            "${CMAKE_CURRENT_BINARY_DIR}/failed-repair-trace.jsonl"
)
set(GPBS_TEST_STATS "${CMAKE_CURRENT_BINARY_DIR}/gpbs-smoke.csv")
add_test(NAME gpbs_official_smoke COMMAND gpbs_official
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 100
    --cutoffTime 10
    --seed 0
    --screen 0
    --solver GPBS
    --tr true
    --ic true
    --rr true
    --rth 0
    --sipp true
    --output "${GPBS_TEST_STATS}"
)
set_tests_properties(gpbs_official_smoke PROPERTIES FIXTURES_SETUP gpbs_stats)
add_test(NAME gpbs_official_result COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_gpbs_result.py"
    "${GPBS_TEST_STATS}"
)
set_tests_properties(gpbs_official_result PROPERTIES FIXTURES_REQUIRED gpbs_stats)
add_test(NAME lns2_python_env COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/test_python_env.py"
)
set_tests_properties(lns2_python_env PROPERTIES ENVIRONMENT
    "PYTHONPATH=$<TARGET_FILE_DIR:lns2_env>${LNS2_TEST_PYTHONPATH_SEPARATOR}${CMAKE_CURRENT_SOURCE_DIR};LNS2_TEST_MAP=${MAPF_LNS2_ROOT}/random-32-32-20.map;LNS2_TEST_SCEN=${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
)
add_test(NAME lns2_repair_collection_native COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/solver/test_repair_collection_native.py"
)
set_tests_properties(lns2_repair_collection_native PROPERTIES ENVIRONMENT
    "PYTHONPATH=$<TARGET_FILE_DIR:lns2_env>${LNS2_TEST_PYTHONPATH_SEPARATOR}${CMAKE_CURRENT_SOURCE_DIR};LNS2_TEST_MAP=${MAPF_LNS2_ROOT}/random-32-32-20.map;LNS2_TEST_SCEN=${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
)

set(TRACE_TEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/repair-trace-test.jsonl")
add_test(NAME lns2_trace_run COMMAND lns2_repair
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 200
    --cutoffTime 20
    --maxRepairIterations 50
    --seed 29
    --trace "${TRACE_TEST_PATH}"
)
set_tests_properties(lns2_trace_run PROPERTIES FIXTURES_SETUP repair_trace)
add_test(NAME lns2_trace_schema COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_trace.py" "${TRACE_TEST_PATH}"
)
set_tests_properties(lns2_trace_schema PROPERTIES FIXTURES_REQUIRED repair_trace)

set(PARITY_PATH "${CMAKE_CURRENT_BINARY_DIR}/official-parity-paths.txt")
add_test(NAME lns2_official_parity_run COMMAND lns_official
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 200
    --cutoffTime 20
    --maxIterations 0
    --seed 29
    --outputPaths "${PARITY_PATH}"
)
set_tests_properties(lns2_official_parity_run PROPERTIES FIXTURES_SETUP official_paths)
add_test(NAME lns2_official_parity_hash COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_path_hash.py"
    "${PARITY_PATH}" "915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf"
)
set_tests_properties(lns2_official_parity_hash PROPERTIES FIXTURES_REQUIRED official_paths)

# Exercise the official entry point beyond initial path generation.  This hash
# was independently reproduced from the protected 70761ac pre-deadline build;
# unlike the maxIterations=0 fixture above, it covers InitLNS and PP repair.
set(REPAIR_PARITY_PATH "${CMAKE_CURRENT_BINARY_DIR}/official-repair-parity-paths.txt")
add_test(NAME lns2_official_repair_parity_run COMMAND lns_official
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 200
    --cutoffTime 20
    --maxIterations 1
    --seed 29
    --outputPaths "${REPAIR_PARITY_PATH}"
)
set_tests_properties(
    lns2_official_repair_parity_run
    PROPERTIES FIXTURES_SETUP official_repair_paths
)
add_test(NAME lns2_official_repair_parity_hash COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_path_hash.py"
    "${REPAIR_PARITY_PATH}" "031d1bf843ada89f03be6880809bcf7632fdaeba509fd035a15657fcc93aa83a"
)
set_tests_properties(
    lns2_official_repair_parity_hash
    PROPERTIES FIXTURES_REQUIRED official_repair_paths
)
