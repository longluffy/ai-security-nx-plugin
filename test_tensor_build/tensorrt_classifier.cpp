#include "tensorrt_classifier.h"

#include <stdexcept>
#include <iostream>
#include <fstream>

namespace nx_meta_plugin {

TensorRTClassifier::TensorRTClassifier(std::filesystem::path modelDir) 
    : m_modelDir(std::move(modelDir)) {
    
    m_enginePath = m_modelDir / "referModel" / "best_new.engine";
    
    // Initialize CUDA stream
    cudaStreamCreate(&stream);
    std::cout << "TensorRT Classifier initialized with engine path: " << m_enginePath << std::endl;
}

TensorRTClassifier::~TensorRTClassifier() {
    terminate();
}

void TensorRTClassifier::ensureInitialized() {
    if (!m_engineLoaded && !m_terminated) {
        try {
            loadEngine();
        } catch (const std::exception& e) {
            std::cerr << "Failed to load TensorRT engine: " << e.what() << std::endl;
            m_terminated = true;
        }
    }
}

bool TensorRTClassifier::isTerminated() const {
    return m_terminated;
}

void TensorRTClassifier::terminate() {
    if (m_engineLoaded) {
        // Clean up CUDA resources
        if (d_input) {
            cudaFree(d_input);
            d_input = nullptr;
        }
        if (d_output) {
            cudaFree(d_output);
            d_output = nullptr;
        }
        
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        
        // Clean up TensorRT resources
        if (context) {
            delete context;
            context = nullptr;
        }
        if (engine) {
            delete engine;
            engine = nullptr;
        }
        if (runtime) {
            delete runtime;
            runtime = nullptr;
        }
        
        m_engineLoaded = false;
    }
    m_terminated = true;
}

void TensorRTClassifier::loadEngine() {
    if (!std::filesystem::exists(m_enginePath)) {
        throw std::runtime_error("TensorRT engine file not found: " + m_enginePath.string());
    }
    
    std::cout << "Loading TensorRT engine from: " << m_enginePath << std::endl;
    
    // Create runtime
    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    
    // Load engine from file
    std::ifstream file(m_enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + m_enginePath.string());
    }
    
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    
    char* modelStream = new char[size];
    file.read(modelStream, size);
    file.close();
    
    engine = runtime->deserializeCudaEngine(modelStream, size);
    delete[] modelStream;
    
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    
    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }
    
    // Get input and output tensor information using modern TensorRT API
    const char* tempInputName = nullptr;
    const char* tempOutputName = nullptr;
    
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(tensorName);
        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            tempInputName = tensorName;
        } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            tempOutputName = tensorName;
        }
    }
    
    if (!tempInputName || !tempOutputName) {
        throw std::runtime_error("Failed to find input/output tensor names");
    }
    
    // Store tensor names
    inputName = tempInputName;
    outputName = tempOutputName;
    
    // Get tensor dimensions
    nvinfer1::Dims inputDims = engine->getTensorShape(inputName);
    nvinfer1::Dims outputDims = engine->getTensorShape(outputName);
    
    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) {
        inputSize *= inputDims.d[i];
    }
    
    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        outputSize *= outputDims.d[i];
    }
    
    // Allocate GPU memory
    cudaError_t cudaStatus = cudaMalloc(&d_input, inputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for input");
    }
    
    cudaStatus = cudaMalloc(&d_output, outputSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for output");
    }
    
    m_engineLoaded = true;
    std::cout << "TensorRT classifier engine loaded successfully" << std::endl;
    std::cout << "Input size: " << inputSize << ", Output size: " << outputSize << std::endl;
}

cv::Mat TensorRTClassifier::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Resize to model input size (640x640)
    cv::resize(image, processed, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // Convert to RGB if needed
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    // Convert to float and normalize to [0, 1]
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    return processed;
}

ClassificationResult TensorRTClassifier::runInference(const cv::Mat& processedImage) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    ClassificationResult result;
    result.success = false;
    result.confidence = 0.0f;
    result.classLabel = "Unknown";
    
    try {
        // Convert Mat to vector in CHW format
        std::vector<float> inputData(inputSize);
        std::vector<cv::Mat> channels;
        cv::split(processedImage, channels);
        
        int channelSize = INPUT_WIDTH * INPUT_HEIGHT;
        for (int c = 0; c < INPUT_CHANNELS; ++c) {
            std::memcpy(inputData.data() + c * channelSize, 
                       channels[c].ptr<float>(), 
                       channelSize * sizeof(float));
        }
        
        // Copy input data to GPU
        cudaMemcpyAsync(d_input, inputData.data(), 
                       inputSize * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        
        // Set tensor addresses
        context->setTensorAddress(inputName, d_input);
        context->setTensorAddress(outputName, d_output);
        
        // Run inference
        context->enqueueV3(stream);
        
        // Copy output back to host
        std::vector<float> outputData(outputSize);
        cudaMemcpyAsync(outputData.data(), d_output, 
                       outputSize * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        
        // Process YOLO output based on reference implementation
        // YOLO11n typically outputs 8400 detections with 84 values each (4 coords + 80 classes)
        // Find the maximum confidence detection
        float maxConfidence = 0.0f;
        int maxClassIdx = 0;
        
        int numDetections = outputSize / 84; // Assuming 84 values per detection
        
        for (int i = 0; i < numDetections && i < 100; ++i) { // Check first 100 detections
            int offset = i * 84;
            
            // Skip bbox coordinates (first 4 values)
            // Look at class probabilities (values 4-83)
            for (int c = 4; c < 84; ++c) {
                if (outputData[offset + c] > maxConfidence) {
                    maxConfidence = outputData[offset + c];
                    maxClassIdx = c - 4; // Adjust for bbox offset
                }
            }
        }
        
        // Map YOLO classes to binary classification (matching reference logic)
        // Assume class 0 = "normal" (CA), other classes = "abnormal" (PN)
        int predictedClass = (maxClassIdx == 0) ? 0 : 1;
        result.classLabel = (predictedClass == 0) ? "CA" : "PN";
        result.confidence = maxConfidence;
        result.success = true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensorRT inference error: " << e.what() << std::endl;
        result.success = false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    result.processTime = duration.count() / 1000.0; // Convert to milliseconds
    
    return result;
}

ClassificationResult TensorRTClassifier::classifyROI(const cv::Mat& roi) {
    ensureInitialized();
    
    if (m_terminated || !m_engineLoaded) {
        ClassificationResult result;
        result.success = false;
        result.classLabel = "Unknown";
        result.confidence = 0.0f;
        result.processTime = 0.0;
        return result;
    }
    
    if (roi.empty()) {
        ClassificationResult result;
        result.success = false;
        result.classLabel = "Unknown";
        result.confidence = 0.0f;
        result.processTime = 0.0;
        return result;
    }
    
    // Preprocess the ROI
    cv::Mat processedImage = preprocessImage(roi);
    
    // Run inference
    return runInference(processedImage);
}

void TensorRTClassifier::classifyDetections(const cv::Mat& frame, DetectionList& detections) {
    ensureInitialized();
    
    if (m_terminated || !m_engineLoaded) {
        // Set all detections to "Unknown" if classifier is not available
        for (auto& detection : detections) {
            detection->classLabel = "Unknown";
        }
        return;
    }
    
    for (auto& detection : detections) {
        try {
            // For simplified test, just use the whole frame
            ClassificationResult result = classifyROI(frame);
            
            if (result.success) {
                detection->classLabel = result.classLabel;
            } else {
                detection->classLabel = "Unknown";
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error classifying detection: " << e.what() << std::endl;
            detection->classLabel = "Unknown";
        }
    }
}

} // namespace nx_meta_plugin
