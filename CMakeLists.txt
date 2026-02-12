cmake_minimum_required(VERSION 3.18)
project(fast-boundary-wavelet-packets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- LibTorch ----------
# Torch_DIR must point to the directory containing TorchConfig.cmake.
# For a standard download this is: <libtorch>/share/cmake/Torch
# Set via -DTorch_DIR=... or the TORCH_DIR environment variable.
if(DEFINED ENV{TORCH_DIR} AND NOT DEFINED Torch_DIR)
    set(Torch_DIR $ENV{TORCH_DIR})
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------- Smoke test executable ----------
add_executable(check_build src/check_build.cpp)
target_link_libraries(check_build ${TORCH_LIBRARIES})

# ---------- Library ----------
# Add source files here as the implementation grows.
# file(GLOB_RECURSE LIB_SOURCES src/*.cpp)
# add_library(fbwp ${LIB_SOURCES})
# target_include_directories(fbwp PUBLIC include)
# target_link_libraries(fbwp ${TORCH_LIBRARIES})

# ---------- Tests ----------
enable_testing()
add_test(NAME check_build COMMAND check_build)
# Add further test executables here as tests are written.
