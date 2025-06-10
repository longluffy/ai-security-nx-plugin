#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

namespace fs = std::filesystem;

struct InferenceResult {
    std::string imagePath;
    int predictedClass;
    float confidence;
    double processTime; // in milliseconds
    bool isCorrect; // based on folder name (CA=0, PN=1)
};

struct FolderStats {
    std::string folderName;
    int expectedClass;
    int totalImages;
    int correctPredictions;
    double avgProcessTime;
    double avgConfidence;
    std::vector<InferenceResult> results;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTInference {
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    Logger logger;
    
    void* d_input;  // Device input buffer
    void* d_output; // Device output buffer
    cudaStream_t stream;
    
    const char* inputName;
    const char* outputName;
    int inputSize;
    int outputSize;
    
public:
    TensorRTInference(const std::string& enginePath) {
        // Initialize CUDA
        cudaStreamCreate(&stream);
        
        // Create runtime
        runtime = nvinfer1::createInferRuntime(logger);
        
        // Load engine
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file: " + enginePath);
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
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        context = engine->createExecutionContext();
        
        // Get input and output dimensions using modern TensorRT 10.x API
        // Get tensor names
        const char* inputName = nullptr;
        const char* outputName = nullptr;
        
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tensorName = engine->getIOTensorName(i);
            nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(tensorName);
            if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
                inputName = tensorName;
            } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
                outputName = tensorName;
            }
        }
        
        if (!inputName || !outputName) {
            throw std::runtime_error("Failed to find input/output tensor names");
        }
        
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
        
        // Store tensor names
        this->inputName = inputName;
        this->outputName = outputName;
        
        // Allocate GPU memory
        cudaMalloc(&d_input, inputSize * sizeof(float));
        cudaMalloc(&d_output, outputSize * sizeof(float));
    }
    
    ~TensorRTInference() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        
        delete context;
        delete engine;
        delete runtime;
    }
    
    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat processed;
        
        // Resize to YOLO input size (640x640)
        cv::resize(image, processed, cv::Size(640, 640));
        
        // Convert to RGB if needed
        if (processed.channels() == 3) {
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
        }
        
        // Convert to float and normalize to [0, 1] (YOLO preprocessing)
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);
        
        return processed;
    }
    
    InferenceResult processImage(const std::string& imagePath, int expectedClass) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        InferenceResult result;
        result.imagePath = imagePath;
        
        // Load and preprocess image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            result.predictedClass = -1;
            result.confidence = 0.0f;
            result.processTime = 0.0;
            result.isCorrect = false;
            return result;
        }
        
        cv::Mat processedImage = preprocessImage(image);
        
        // Copy image data to GPU
        std::vector<float> inputData;
        inputData.resize(inputSize);
        
        // Convert Mat to vector (CHW format)
        std::vector<cv::Mat> channels;
        cv::split(processedImage, channels);
        
        int channelSize = 640 * 640;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(inputData.data() + c * channelSize, 
                       channels[c].ptr<float>(), 
                       channelSize * sizeof(float));
        }
        
        cudaMemcpyAsync(d_input, inputData.data(), 
                       inputSize * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        
        // Set tensor addresses for modern TensorRT API
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
        
        // Process YOLO output - for object detection
        // YOLO output format: [batch, num_detections, 85] where 85 = 4 bbox coords + 1 objectness + 80 classes
        // For this demo, we'll simulate binary classification based on detection confidence
        
        // Find the maximum confidence detection
        float maxConfidence = 0.0f;
        int maxClassIdx = 0;
        
        // YOLO11n typically outputs 8400 detections with 84 values each (4 coords + 80 classes)
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
        
        // For demo purposes, map YOLO classes to binary classification
        // Assume class 0 = "normal" (CA), other classes = "abnormal" (PN)
        result.predictedClass = (maxClassIdx == 0) ? 0 : 1;
        result.confidence = maxConfidence;
        result.isCorrect = (result.predictedClass == expectedClass);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.processTime = duration.count() / 1000.0; // Convert to milliseconds
        
        return result;
    }
};

std::vector<std::string> getImageFiles(const std::string& folderPath) {
    std::vector<std::string> imageFiles;
    
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

void generateMarkdownReport(const std::vector<FolderStats>& folderStats, 
                          const std::string& outputPath) {
    std::ofstream report(outputPath);
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    report << "# Image Classification Report\n\n";
    report << "Generated on: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n\n";
    
    // Summary Section
    report << "## Summary\n\n";
    report << "| Class | Folder | Total Images | Correct Predictions | Accuracy | Avg Process Time (ms) | Avg Confidence |\n";
    report << "|-------|--------|--------------|---------------------|----------|----------------------|----------------|\n";
    
    int totalImages = 0;
    int totalCorrect = 0;
    double totalProcessTime = 0.0;
    double totalConfidence = 0.0;
    
    for (const auto& stats : folderStats) {
        double accuracy = (stats.totalImages > 0) ? 
                         (static_cast<double>(stats.correctPredictions) / stats.totalImages * 100.0) : 0.0;
        
        report << "| " << stats.expectedClass 
               << " | " << stats.folderName 
               << " | " << stats.totalImages 
               << " | " << stats.correctPredictions 
               << " | " << std::fixed << std::setprecision(2) << accuracy << "% "
               << " | " << std::fixed << std::setprecision(3) << stats.avgProcessTime 
               << " | " << std::fixed << std::setprecision(3) << stats.avgConfidence 
               << " |\n";
        
        totalImages += stats.totalImages;
        totalCorrect += stats.correctPredictions;
        totalProcessTime += (stats.avgProcessTime * stats.totalImages);
        totalConfidence += (stats.avgConfidence * stats.totalImages);
    }
    
    double overallAccuracy = (totalImages > 0) ? 
                            (static_cast<double>(totalCorrect) / totalImages * 100.0) : 0.0;
    double overallAvgProcessTime = (totalImages > 0) ? (totalProcessTime / totalImages) : 0.0;
    double overallAvgConfidence = (totalImages > 0) ? (totalConfidence / totalImages) : 0.0;
    
    report << "| **Total** | **All** | **" << totalImages 
           << "** | **" << totalCorrect 
           << "** | **" << std::fixed << std::setprecision(2) << overallAccuracy << "%** "
           << " | **" << std::fixed << std::setprecision(3) << overallAvgProcessTime 
           << "** | **" << std::fixed << std::setprecision(3) << overallAvgConfidence 
           << "** |\n\n";
    
    // Detailed Results Section
    report << "## Detailed Results\n\n";
    
    for (const auto& stats : folderStats) {
        report << "### " << stats.folderName << " Folder (Class " << stats.expectedClass << ")\n\n";
        report << "| Image | Predicted Class | Confidence | Process Time (ms) | Correct |\n";
        report << "|-------|-----------------|------------|-------------------|---------|\n";
        
        // Show first 25 results, then skip to last 5 if there are many images
        size_t maxDetailedResults = 25;
        
        for (size_t i = 0; i < stats.results.size() && i < maxDetailedResults; ++i) {
            const auto& result = stats.results[i];
            std::string filename = fs::path(result.imagePath).filename().string();
            std::string correctStr = result.isCorrect ? "✅" : "❌";
            
            report << "| " << filename 
                   << " | " << result.predictedClass 
                   << " | " << std::fixed << std::setprecision(3) << result.confidence 
                   << " | " << std::fixed << std::setprecision(3) << result.processTime 
                   << " | " << correctStr << " |\n";
        }
        
        if (stats.results.size() > maxDetailedResults) {
            report << "| ... | ... | ... | ... | ... |\n";
            report << "| *(" << (stats.results.size() - maxDetailedResults) << " more results)* | | | | |\n";
        }
        
        report << "\n";
    }
    
    report << "---\n";
    report << "*Report generated by Image Classification Engine*\n";
    
    report.close();
}

int main(int argc, char** argv) {
    try {
        std::cout << "TensorRT Image Classification\n";
        std::cout << "============================\n\n";
        
        // Initialize TensorRT inference
        std::string enginePath = "best_new.engine";
        TensorRTInference inference(enginePath);
        
        std::vector<FolderStats> folderStats;
        
        // Process CA folder (Class 0)
        {
            std::cout << "Processing CA folder...\n";
            FolderStats caStats;
            caStats.folderName = "CA";
            caStats.expectedClass = 0;
            
            std::vector<std::string> caImages = getImageFiles("CA/");
            caStats.totalImages = caImages.size();
            caStats.correctPredictions = 0;
            caStats.avgProcessTime = 0.0;
            caStats.avgConfidence = 0.0;
            
            if (caStats.totalImages == 0) {
                std::cout << "⚠️  No images found in CA folder\n";
            } else {
                int processed = 0;
                for (const auto& imagePath : caImages) {
                    InferenceResult result = inference.processImage(imagePath, 0);
                    caStats.results.push_back(result);
                    
                    if (result.isCorrect) {
                        caStats.correctPredictions++;
                    }
                    
                    caStats.avgProcessTime += result.processTime;
                    caStats.avgConfidence += result.confidence;
                    
                    processed++;
                    if (processed % 100 == 0 || processed == caStats.totalImages) {
                        std::cout << "  Processed " << processed << "/" << caStats.totalImages 
                                 << " CA images\r" << std::flush;
                    }
                }
                
                if (caStats.totalImages > 0) {
                    caStats.avgProcessTime /= caStats.totalImages;
                    caStats.avgConfidence /= caStats.totalImages;
                }
                
                std::cout << "\n✅ CA folder completed: " << caStats.correctPredictions 
                         << "/" << caStats.totalImages << " correct ("
                         << std::fixed << std::setprecision(1) 
                         << (static_cast<double>(caStats.correctPredictions) / caStats.totalImages * 100.0)
                         << "%)\n\n";
            }
            
            folderStats.push_back(caStats);
        }
        
        // Process PN folder (Class 1)
        {
            std::cout << "Processing PN folder...\n";
            FolderStats pnStats;
            pnStats.folderName = "PN";
            pnStats.expectedClass = 1;
            
            std::vector<std::string> pnImages = getImageFiles("PN/");
            pnStats.totalImages = pnImages.size();
            pnStats.correctPredictions = 0;
            pnStats.avgProcessTime = 0.0;
            pnStats.avgConfidence = 0.0;
            
            if (pnStats.totalImages == 0) {
                std::cout << "⚠️  No images found in PN folder\n";
            } else {
                int processed = 0;
                for (const auto& imagePath : pnImages) {
                    InferenceResult result = inference.processImage(imagePath, 1);
                    pnStats.results.push_back(result);
                    
                    if (result.isCorrect) {
                        pnStats.correctPredictions++;
                    }
                    
                    pnStats.avgProcessTime += result.processTime;
                    pnStats.avgConfidence += result.confidence;
                    
                    processed++;
                    if (processed % 100 == 0 || processed == pnStats.totalImages) {
                        std::cout << "  Processed " << processed << "/" << pnStats.totalImages 
                                 << " PN images\r" << std::flush;
                    }
                }
                
                if (pnStats.totalImages > 0) {
                    pnStats.avgProcessTime /= pnStats.totalImages;
                    pnStats.avgConfidence /= pnStats.totalImages;
                }
                
                std::cout << "\n✅ PN folder completed: " << pnStats.correctPredictions 
                         << "/" << pnStats.totalImages << " correct ("
                         << std::fixed << std::setprecision(1) 
                         << (static_cast<double>(pnStats.correctPredictions) / pnStats.totalImages * 100.0)
                         << "%)\n\n";
            }
            
            folderStats.push_back(pnStats);
        }
        
        // Generate report
        std::cout << "📊 Generating report...\n";
        generateMarkdownReport(folderStats, "classification_report.md");
        
        std::cout << "\n🎉 PROCESSING COMPLETE! 🎉\n";
        std::cout << "📄 Report saved to: classification_report.md\n\n";
        
        // Print summary to console
        std::cout << "📈 SUMMARY:\n";
        std::cout << "----------\n";
        int totalImages = 0;
        int totalCorrect = 0;
        for (const auto& stats : folderStats) {
            totalImages += stats.totalImages;
            totalCorrect += stats.correctPredictions;
            if (stats.totalImages > 0) {
                std::cout << "  " << stats.folderName << " (Class " << stats.expectedClass << "): "
                         << stats.correctPredictions << "/" << stats.totalImages 
                         << " (" << std::fixed << std::setprecision(1) 
                         << (static_cast<double>(stats.correctPredictions) / stats.totalImages * 100.0) 
                         << "%) - Avg time: " << std::fixed << std::setprecision(2) 
                         << stats.avgProcessTime << "ms\n";
            }
        }
        
        if (totalImages > 0) {
            std::cout << "  📊 Overall accuracy: " << totalCorrect << "/" << totalImages 
                     << " (" << std::fixed << std::setprecision(1) 
                     << (static_cast<double>(totalCorrect) / totalImages * 100.0) << "%)\n";
        }
        
        std::cout << "\n💡 To view the detailed report, open: classification_report.md\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
