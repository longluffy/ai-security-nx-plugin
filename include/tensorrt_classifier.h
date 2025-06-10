#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <filesystem>

#include "detection.h"
#include "geometry.h"

namespace nx_meta_plugin {

struct ClassificationResult {
    std::string classLabel;      // "CA" or "PN"
    float confidence;            // Classification confidence
    double processTime;          // Processing time in milliseconds
    bool success;               // Whether classification was successful
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTClassifier {
public:
    explicit TensorRTClassifier(std::filesystem::path modelDir);
    ~TensorRTClassifier();

    void ensureInitialized();
    bool isTerminated() const;
    void terminate();

    // Classify a single ROI
    ClassificationResult classifyROI(const cv::Mat& roi);
    
    // Classify all detections in a frame (batch processing)
    void classifyDetections(const cv::Mat& frame, DetectionList& detections);

private:
    void loadEngine();
    cv::Mat preprocessImage(const cv::Mat& image);
    ClassificationResult runInference(const cv::Mat& processedImage);

private:
    bool m_engineLoaded = false;
    bool m_terminated = false;
    std::filesystem::path m_modelDir;
    std::filesystem::path m_enginePath;

    // TensorRT components
    Logger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // CUDA resources
    void* d_input = nullptr;   // Device input buffer
    void* d_output = nullptr;  // Device output buffer
    cudaStream_t stream;
    
    // Model information
    const char* inputName = nullptr;
    const char* outputName = nullptr;
    int inputSize = 0;
    int outputSize = 0;
    
    // Constants
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr int INPUT_CHANNELS = 3;
};

} // namespace nx_meta_plugin
