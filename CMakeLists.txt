## Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

cmake_minimum_required(VERSION 3.15)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
project(opencv_object_detection_analytics_plugin)

set(metadataSdkVersion "5.1.1.37512")
set(metadataSdkDir "" CACHE PATH
        "Path to unpacked VMS Metadata SDK zip. Tested with version ${metadataSdkVersion}.")

if (metadataSdkDir STREQUAL "")
    message(FATAL_ERROR "metadataSdkDir cache variable is undefined.")
endif ()

if (NOT DEFINED ARTIFACTORY_URL)
    set(ARTIFACTORY_URL "https://artifactory.nxvms.dev/artifactory/artifacts/nx_open_integrations")
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Download OpenCV automatically using Conan
if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake")
endif ()
include(${CMAKE_BINARY_DIR}/conan.cmake)

if (WIN32)
    set(CONAN_VS_RUNTIME_MT_SETTING "compiler.runtime=MT")
endif ()
set(OPENCV_OPTIONS
        opencv:parallel=False
        opencv:contrib=True
        opencv:contrib_freetype=False
        opencv:contrib_sfm=False
        opencv:with_jpeg=libjpeg
        opencv:with_png=True
        opencv:with_tiff=False
        opencv:with_jpeg2000=False
        opencv:with_openexr=False
        opencv:with_eigen=False
        opencv:with_webp=False
        opencv:with_quirc=False
        opencv:with_cuda=False
        opencv:with_cublas=False
        opencv:dnn=True
        opencv:with_ffmpeg=False
)

if (UNIX)
    set(OPENCV_OPTIONS
            ${OPENCV_OPTIONS}
            opencv:fPIC=True
            opencv:with_gtk=False
            opencv:with_cufft=False
            opencv:with_v4l=False
    )
endif ()

conan_cmake_run(BUILD_TYPE "Release")
conan_cmake_configure(REQUIRES opencv/4.1.2
        GENERATORS cmake_find_package
        OPTIONS ${OPENCV_OPTIONS}
        SETTINGS ${CONAN_VS_RUNTIME_MT_SETTING}
)

conan_cmake_install(PATH_OR_REFERENCE .
        OPTIONS ${OPENCV_OPTIONS}
        SETTINGS ${CONAN_VS_RUNTIME_MT_SETTING}
        BUILD missing
)

#set(ONNXRUNTIME_OPTIONS
#        onnxruntime:with_cuda=True
#)
conan_cmake_configure(REQUIRES onnxruntime/1.18.1
        GENERATORS cmake_find_package
        #        OPTIONS ${ONNXRUNTIME_OPTIONS}
        SETTINGS ${CONAN_VS_RUNTIME_MT_SETTING}
)

conan_cmake_install(PATH_OR_REFERENCE .
        #        OPTIONS ${ONNXRUNTIME_OPTIONS}
        SETTINGS ${CONAN_VS_RUNTIME_MT_SETTING}
        BUILD missing
)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

# Add TensorRT support
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

# find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /root/onnxruntime/onnxruntime-linux-x64-1.18.1/include)
# find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /root/onnxruntime/onnxruntime-linux-x64-1.18.1/lib)
include_directories("include/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (UNIX)
    # In Linux, for the plugin .so library, set `rpath` to "$ORIGIN" and do not set `runpath`, thus
    # enabling the lookup of the dependencies in the plugin dir first.
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--disable-new-dtags")
endif ()
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

if (WIN32)
    set(API_EXPORT_MACRO "__declspec(dllexport)")
elseif (UNIX)
    set(API_EXPORT_MACRO "__attribute__((visibility(\"default\")))")
endif ()

#--------------------------------------------------------------------------------------------------
# Define nx_kit lib, static.

set(nxKitLibraryType "STATIC" CACHE STRING "" FORCE)
set(nxKitWithTests "YES" CACHE STRING "" FORCE)

add_subdirectory(${metadataSdkDir}/nx_kit ${CMAKE_CURRENT_BINARY_DIR}/nx_kit)

#--------------------------------------------------------------------------------------------------
# Define nx_sdk lib, static, depends on nx_kit.

set(SDK_SRC_DIR ${metadataSdkDir}/src)
file(GLOB_RECURSE SDK_SRC ${SDK_SRC_DIR}/*)

add_library(nx_sdk STATIC ${SDK_SRC})
target_include_directories(nx_sdk PUBLIC ${SDK_SRC_DIR})
target_link_libraries(nx_sdk PRIVATE nx_kit)

target_compile_definitions(nx_sdk PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO})

#--------------------------------------------------------------------------------------------------
# Define opencv_object_detection_analytics_plugin lib, dynamic, depends on nx_kit and nx_sdk.

set(pluginHeaders
        include/detection.h
        include/device_agent.h
        include/engine.h
        include/exceptions.h
        include/frame.h
        include/geometry.h
        include/object_detector.h
        include/plugin.h
        include/yolo11_classifier.h
        include/yolo11_detector.h
        include/tensorrt_classifier.h
        include/object_tracker.h
        include/object_tracker_utils.h
                include/visualize.h
)

set(pluginSrc ${pluginHeaders}
        src/detection.cpp
        src/device_agent.cpp
        src/engine.cpp
        src/geometry.cpp
        src/object_detector.cpp
        src/plugin.cpp
        src/yolo11_classifier.cpp
        src/yolo11_detector.cpp
        src/tensorrt_classifier.cpp
        src/object_tracker.cpp
        src/object_tracker_utils.cpp
)

message("${OPENCV_OBJECT_DETECTION_ANALYTICS_PLUGIN_SRC}")
add_library(opencv_object_detection_analytics_plugin SHARED ${pluginSrc})
# add_executable(opencv_object_detection_analytics_plugin src/main.cpp ${pluginSrc})

target_include_directories(opencv_object_detection_analytics_plugin PRIVATE
        ${OPENCV_ANALYTICS_PLUGIN_SRC_DIR}
        ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
)
set(CMAKE_EXE_LINKER_FLAGS " -static")
message("ONNX_RUNTIME_LIB: ${ONNX_RUNTIME_LIB}")
target_link_libraries(opencv_object_detection_analytics_plugin
        nx_kit
        nx_sdk
        opencv::core opencv::flann opencv::imgproc opencv::imgcodecs opencv::dnn opencv::opencv_dnn opencv::ml
        opencv::plot opencv::opencv_features2d opencv::opencv_calib3d opencv::datasets opencv::video opencv::tracking
        onnxruntime::onnxruntime
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
        ${NVONNX_PARSER_LIBRARY}
        ${CUDA_LIBRARIES}
        cudart
)
target_compile_definitions(opencv_object_detection_analytics_plugin
        PRIVATE NX_PLUGIN_API=${API_EXPORT_MACRO}
)
