cmake_minimum_required(VERSION 3.18)
project(fast-boundary-wavelet-packets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# ---------- LibTorch ----------
# Torch_DIR must point to the directory containing TorchConfig.cmake.
# For a standard download this is: <libtorch>/share/cmake/Torch
# Set via -DTorch_DIR=... or the TORCH_DIR environment variable.
if(DEFINED ENV{TORCH_DIR} AND NOT DEFINED Torch_DIR)
    set(Torch_DIR $ENV{TORCH_DIR})
endif()

# ---------- Python (early, needed for Torch auto-detection) ----------
if(BUILD_PYTHON_BINDINGS)
    if(DEFINED Python_EXECUTABLE)
        set(Python_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "Path to Python interpreter")
    endif()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

    # Auto-detect Torch_DIR from Python's torch package.
    if(NOT DEFINED Torch_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                "import torch; print(torch.utils.cmake_prefix_path + '/Torch')"
            OUTPUT_VARIABLE Torch_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    # Auto-detect pybind11_DIR from Python's pybind11 package.
    if(NOT DEFINED pybind11_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------- Smoke test executable ----------
add_executable(check_build src/check_build.cpp)
target_link_libraries(check_build ${TORCH_LIBRARIES})

# ---------- Library ----------
add_library(fbwp src/wavelet.cpp src/sparse_math.cpp src/matrix_build.cpp src/orthogonalize.cpp src/transform.cpp)
set_target_properties(fbwp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(fbwp PUBLIC include)
target_link_libraries(fbwp ${TORCH_LIBRARIES})

# ---------- Bindings ----------
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

    pybind11_add_module(fbwp_python bindings/python_bindings.cpp)
    set_target_properties(fbwp_python PROPERTIES
        OUTPUT_NAME fbwp
        INSTALL_RPATH "$ORIGIN/torch/lib")
    target_include_directories(fbwp_python PUBLIC include)
    target_link_libraries(fbwp_python PRIVATE fbwp ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

    install(TARGETS fbwp_python LIBRARY DESTINATION .)
    install(FILES stubs/fbwp.pyi DESTINATION . RENAME fbwp.pyi)
endif()

# ---------- Tests ----------
enable_testing()
add_test(NAME check_build COMMAND check_build)

add_executable(test_wavelet tests/test_wavelet.cpp)
target_link_libraries(test_wavelet fbwp)
add_test(NAME test_wavelet COMMAND test_wavelet)

add_executable(test_sparse_math tests/test_sparse_math.cpp)
target_link_libraries(test_sparse_math fbwp)
add_test(NAME test_sparse_math COMMAND test_sparse_math)

add_executable(test_matrix_build tests/test_matrix_build.cpp)
target_link_libraries(test_matrix_build fbwp)
add_test(NAME test_matrix_build COMMAND test_matrix_build)

add_executable(test_matrix_build_reference tests/test_matrix_build_reference.cpp)
target_link_libraries(test_matrix_build_reference fbwp)
target_include_directories(test_matrix_build_reference PRIVATE tests)
target_compile_definitions(test_matrix_build_reference PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME test_matrix_build_reference COMMAND test_matrix_build_reference)

add_executable(test_orthogonalize tests/test_orthogonalize.cpp)
target_link_libraries(test_orthogonalize fbwp)
add_test(NAME test_orthogonalize COMMAND test_orthogonalize)

add_executable(test_boundary_build_reference tests/test_boundary_build_reference.cpp)
target_link_libraries(test_boundary_build_reference fbwp)
target_include_directories(test_boundary_build_reference PRIVATE tests)
target_compile_definitions(test_boundary_build_reference PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME test_boundary_build_reference COMMAND test_boundary_build_reference)

add_executable(test_transform tests/test_transform.cpp)
target_link_libraries(test_transform fbwp)
add_test(NAME test_transform COMMAND test_transform)

add_executable(test_transform_reference tests/test_transform_reference.cpp)
target_link_libraries(test_transform_reference fbwp)
target_include_directories(test_transform_reference PRIVATE tests)
target_compile_definitions(test_transform_reference PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME test_transform_reference COMMAND test_transform_reference)

add_executable(test_transform_2d_reference tests/test_transform_2d_reference.cpp)
target_link_libraries(test_transform_2d_reference fbwp)
target_include_directories(test_transform_2d_reference PRIVATE tests)
target_compile_definitions(test_transform_2d_reference PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME test_transform_2d_reference COMMAND test_transform_2d_reference)
