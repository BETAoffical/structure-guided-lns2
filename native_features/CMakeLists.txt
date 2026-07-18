cmake_minimum_required(VERSION 3.16)
project(lns2_features_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(
  lns2_features_native
  ../src/online_features.cpp
  ../src/online_features_module.cpp)
target_include_directories(lns2_features_native PRIVATE ../src)
