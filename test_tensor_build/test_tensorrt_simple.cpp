#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

// Include our test classifier
#include "tensorrt_classifier.h"

int main() {
    std::cout << "Testing TensorRT Classifier Integration..." << std::endl;
    
    try {
        // Initialize classifier
        nx_meta_plugin::TensorRTClassifier classifier("/home/longluffy/code/pluggin/ai-security-nx-plugin");
        
        // Create a test image
        cv::Mat testImage = cv::Mat::zeros(640, 640, CV_8UC3);
        
        // Test single ROI classification
        std::cout << "Testing single ROI classification..." << std::endl;
        auto result = classifier.classifyROI(testImage);
        
        std::cout << "Classification Result:" << std::endl;
        std::cout << "  Success: " << (result.success ? "Yes" : "No") << std::endl;
        std::cout << "  Class: " << result.classLabel << std::endl;
        std::cout << "  Confidence: " << result.confidence << std::endl;
        std::cout << "  Process Time: " << result.processTime << "ms" << std::endl;
        
        classifier.terminate();
        std::cout << "Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during test: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
