cmake_minimum_required(VERSION 3.15)
project(TensorRTTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${TENSORRT_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(NVONNX_PARSER_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT} ${TENSORRT_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# Find CUDA
find_package(CUDA REQUIRED)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(WARNING "TensorRT not found. Using fallback paths.")
    set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
    set(TENSORRT_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
    set(TENSORRT_PLUGIN_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
    set(NVONNX_PARSER_LIBRARY "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")

# Create test executable
add_executable(tensorrt_test 
    test_tensorrt_simple.cpp
    tensorrt_classifier.cpp
)

target_include_directories(tensorrt_test PRIVATE
    .
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tensorrt_test
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${NVONNX_PARSER_LIBRARY}
    ${CUDA_LIBRARIES}
    cudart
)
