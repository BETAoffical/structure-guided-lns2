cmake_minimum_required(VERSION 3.16)
project(structure_guided_lns2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_executable(lns_official "${MAPF_LNS2_ROOT}/src/driver.cpp")
target_link_libraries(lns_official PRIVATE mapf_lns2_core)

add_executable(lns2_repair
    src/repair_driver.cpp
    src/jsonl_observer.cpp
)
target_link_libraries(lns2_repair PRIVATE mapf_lns2_core)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
pybind11_add_module(lns2_env src/python_bindings.cpp)
target_link_libraries(lns2_env PRIVATE mapf_lns2_core)

enable_testing()
add_executable(lns2_interface_tests tests/test_repair_interfaces.cpp)
target_link_libraries(lns2_interface_tests PRIVATE mapf_lns2_core)
target_compile_definitions(lns2_interface_tests PRIVATE
    TEST_MAP="${MAPF_LNS2_ROOT}/random-32-32-20.map"
    TEST_SCEN="${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
)
add_test(NAME lns2_interface_tests COMMAND lns2_interface_tests)
add_test(NAME lns2_repair_smoke COMMAND lns2_repair
    --map "${MAPF_LNS2_ROOT}/random-32-32-20.map"
    --agents "${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
    --agentNum 24
    --cutoffTime 10
    --seed 0
)
add_test(NAME lns2_python_env COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_env.py"
)
set_tests_properties(lns2_python_env PROPERTIES ENVIRONMENT
    "PYTHONPATH=$<TARGET_FILE_DIR:lns2_env>;LNS2_TEST_MAP=${MAPF_LNS2_ROOT}/random-32-32-20.map;LNS2_TEST_SCEN=${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
)
add_test(NAME lns2_repair_collection_native COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_repair_collection_native.py"
)
set_tests_properties(lns2_repair_collection_native PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:$<TARGET_FILE_DIR:lns2_env>;LNS2_TEST_MAP=${MAPF_LNS2_ROOT}/random-32-32-20.map;LNS2_TEST_SCEN=${MAPF_LNS2_ROOT}/random-32-32-20-random-1.scen"
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
