cmake_minimum_required(VERSION 3.16)
project(structure_guided_lns2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lns2_core
    src/instance.cpp
    src/solver.cpp
)
target_include_directories(lns2_core PUBLIC include)
if(MSVC)
    target_compile_options(lns2_core PRIVATE /W4)
else()
    target_compile_options(lns2_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(lns2_cli src/main.cpp)
target_link_libraries(lns2_cli PRIVATE lns2_core)

enable_testing()
add_executable(lns2_tests tests/test_solver.cpp)
target_link_libraries(lns2_tests PRIVATE lns2_core)
target_compile_definitions(lns2_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
add_test(NAME lns2_tests COMMAND lns2_tests)
