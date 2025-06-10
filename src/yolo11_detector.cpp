// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/


#include <opencv2/core.hpp>

#include "yolo11_detector.h"
#include "exceptions.h"

namespace nx_meta_plugin {
    using namespace std::string_literals;
    using namespace cv;

    YOLO11Detector::YOLO11Detector(std::filesystem::path modelDir) :
            m_modelDir(std::move(modelDir)) {
    }

/**
* Load the model if it is not loaded, do nothing otherwise. In case of errors terminate the
* plugin and throw a specialized exception.
*/
    void YOLO11Detector::ensureInitialized() {
        if (isTerminated()) {
            throw ObjectDetectorIsTerminatedError(
                    "Object detector initialization error: object detector is terminated.");
        }
        if (m_netLoaded)
            return;

        try {
            loadModel();
        }
        catch (const cv::Exception &e) {
            terminate();
            throw ObjectDetectorInitializationError("Loading model: " + cvExceptionToStdString(e));
        }
        catch (const std::exception &e) {
            terminate();
            throw ObjectDetectorInitializationError("Loading model: Error: "s + e.what());
        }
    }

    bool YOLO11Detector::isTerminated() const {
        return m_terminated;
    }

    void YOLO11Detector::terminate() {
        m_terminated = true;
    }

    DetectionList YOLO11Detector::run(const cv::Mat &frame) {
        if (isTerminated())
            throw ObjectDetectorIsTerminatedError("Detection error: object detector is terminated.");

        try {
            return runImpl(frame);
        }
        catch (const cv::Exception &e) {
            terminate();
            throw ObjectDetectionError(cvExceptionToStdString(e));
        }
        catch (const std::exception &e) {
            terminate();
            throw ObjectDetectionError("Error: "s + e.what());
        }
    }

    void YOLO11Detector::loadModel() {
        // Initialize ONNX Runtime environment with warning level
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
        sessionOptions = Ort::SessionOptions();

        // Set number of intra-op threads for parallelism - reduce for better performance
        sessionOptions.SetIntraOpNumThreads(std::min(4, static_cast<int>(std::thread::hardware_concurrency())));
        sessionOptions.SetInterOpNumThreads(2); // Limit inter-op threads
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Enable memory optimization
        sessionOptions.EnableMemPattern();
        sessionOptions.EnableCpuMemArena();

        // Retrieve available execution providers (e.g., CPU, CUDA)
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(),
                                       "CUDAExecutionProvider");
        OrtCUDAProviderOptions cudaOption;

        // Configure session options based on whether GPU is to be used and available
        if (useGPU && cudaAvailable != availableProviders.end()) {
            std::cout << "Inference device: GPU" << std::endl;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption); // Append CUDA execution provider
        } else {
            if (useGPU) {
                std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
            }
            std::cout << "Inference device: CPU" << std::endl;
        }

        // Load the ONNX model into the session
        std::filesystem::path modelPath = m_modelDir / std::filesystem::path("yolov11n.onnx");
        std::string modelPathStr{modelPath.u8string()};
#ifdef _WIN32
        std::wstring w_modelPath(modelPathStr.begin(), m_modelPath.end());
        session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
        session = Ort::Session(env, modelPathStr.c_str(), sessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        // Retrieve input tensor shape information
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
        std::vector<int64_t> inputTensorShapeVec = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        isDynamicInputShape = (inputTensorShapeVec.size() >= 4) && (inputTensorShapeVec[2] == -1 &&
                                                                    inputTensorShapeVec[3] ==
                                                                    -1); // Check for dynamic dimensions

        // Allocate and store input node names
        auto input_name = session.GetInputNameAllocated(0, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

        // Allocate and store output node names
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

        // Set the expected input image shape based on the model's input tensor
        if (inputTensorShapeVec.size() >= 4) {
            inputImageShape = cv::Size(static_cast<int>(inputTensorShapeVec[3]),
                                       static_cast<int>(inputTensorShapeVec[2]));
        } else {
            m_netLoaded = true;
            throw ObjectDetectorInitializationError("Invalid input tensor shape.");
        }

        // Get the number of input and output nodes
        numInputNodes = session.GetInputCount();
        numOutputNodes = session.GetOutputCount();
        std::cout << "Detection model loaded successfully with " << numInputNodes << " input nodes and "
                  << numOutputNodes
                  << " output nodes." << std::endl;
    }

// Preprocess function implementation
    cv::Mat
    YOLO11Detector::preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape) {
        cv::Mat resizedImage;
        // Resize and pad the image using letterBox utility
        letterBox(image, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape,
                  false, true,
                  32);

        // Update input tensor shape based on resized image dimensions
        inputTensorShape[2] = resizedImage.rows;
        inputTensorShape[3] = resizedImage.cols;

        // Convert image to float and normalize to [0, 1]
        resizedImage.convertTo(resizedImage, CV_32FC3, 1 / 255.0f);

        // Allocate memory for the image blob in CHW format
        blob = new float[resizedImage.cols * resizedImage.rows * resizedImage.channels()];

        // Split the image into separate channels and store in the blob
        std::vector<cv::Mat> chw(resizedImage.channels());
        for (int i = 0; i < resizedImage.channels(); ++i) {
            chw[i] = cv::Mat(resizedImage.rows, resizedImage.cols, CV_32FC1,
                             blob + i * resizedImage.cols * resizedImage.rows);
        }
        cv::split(resizedImage, chw); // Split channels into the blob

        return resizedImage;
    }

// Postprocess function to convert raw model output into detections
    DetectionList YOLO11Detector::postprocess(
            const cv::Size &originalImageSize,
            const cv::Size &resizedImageShape,
            const std::vector<Ort::Value> &outputTensors,
            float confThreshold,
            float iouThreshold
    ) {
        DetectionList detections;
        const float *rawOutput = outputTensors[0].GetTensorData<float>(); // Extract raw output data from the first output tensor
        const std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Determine the number of features and detections
        const size_t num_features = outputShape[1];
        const size_t num_detections = outputShape[2];

        // Early exit if no detections
        if (num_detections == 0) {
            return detections;
        }

        // Calculate number of classes based on output shape
        const int numClasses = static_cast<int>(num_features) - 4;
        if (numClasses <= 0) {
            // Invalid number of classes
            return detections;
        }

        // Reserve memory for efficient appending
        std::vector<cv::Rect> boxes;
        boxes.reserve(num_detections);
        std::vector<float> confs;
        confs.reserve(num_detections);
        std::vector<int> classIds;
        classIds.reserve(num_detections);
        std::vector<cv::Rect> nms_boxes;
        nms_boxes.reserve(num_detections);

        // Constants for indexing
        const float *ptr = rawOutput;

        for (size_t d = 0; d < num_detections; ++d) {
            // Extract bounding box coordinates (center x, center y, width, height)
            float centerX = ptr[0 * num_detections + d];
            float centerY = ptr[1 * num_detections + d];
            float width = ptr[2 * num_detections + d];
            float height = ptr[3 * num_detections + d];

            // Find class with the highest confidence score
            int classId = -1;
            float maxScore = -FLT_MAX;
            for (int c = 0; c < numClasses; ++c) {
                const float score = ptr[d + (4 + c) * num_detections];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            // Proceed only if confidence exceeds threshold
            if (maxScore > confThreshold) {
                // Convert center coordinates to top-left (x1, y1)
                float left = centerX - width / 2.0f;
                float top = centerY - height / 2.0f;

                // Scale to original image size
                cv::Rect scaledBox = scaleCoords(
                        resizedImageShape,
                        cv::Rect(left, top, width, height),
                        originalImageSize,
                        true
                );

                // Round coordinates for integer pixel positions
                cv::Rect roundedBox;
                roundedBox.x = std::round(scaledBox.x);
                roundedBox.y = std::round(scaledBox.y);
                roundedBox.width = std::round(scaledBox.width);
                roundedBox.height = std::round(scaledBox.height);

                // Adjust NMS box coordinates to prevent overlap between classes
                cv::Rect nmsBox = roundedBox;
                nmsBox.x += classId * 7680; // Arbitrary offset to differentiate classes
                nmsBox.y += classId * 7680;

                // Add to respective containers
                nms_boxes.emplace_back(nmsBox);
                boxes.emplace_back(roundedBox);
                confs.emplace_back(maxScore);
                classIds.emplace_back(classId);
            }
        }

        // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        std::vector<int> indices;
        NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);

        // Collect filtered detections into the result vector
        detections.reserve(indices.size());
        for (const int idx: indices) {
            const std::string classLabel = kClasses[(size_t) classIds[idx]];
            bool oneOfRequiredClasses = std::find(
                    kClassesToDetect.begin(), kClassesToDetect.end(), classLabel) != kClassesToDetect.end();
            if (oneOfRequiredClasses) {
                detections.emplace_back(std::make_shared<Detection>(
                        Detection{
                                cvRectToNxRect(boxes[idx], originalImageSize.width, originalImageSize.height),
                                kClasses[(size_t) classIds[idx]],
                                confs[idx],
                                nx::sdk::Uuid() //< Will be filled with real value in ObjectTracker.
                                // nx::sdk::UuidHelper::randomUuid()
                        }
                ));
            }
        }

        return detections;
    }

    DetectionList YOLO11Detector::runImpl(const cv::Mat &frame) {
        if (isTerminated()) {
            throw ObjectDetectorIsTerminatedError(
                    "Object detection error: object detector is terminated.");
        }

        float *blobPtr = nullptr; // Pointer to hold preprocessed image data
        // Define the shape of the input tensor (batch size, channels, height, width)
        std::vector<int64_t> inputTensorShape = {1, 3, inputImageShape.height, inputImageShape.width};

        // Preprocess the image and obtain a pointer to the blob
        cv::Mat preprocessedImage = preprocess(frame, blobPtr, inputTensorShape);

        // Compute the total number of elements in the input tensor
        size_t inputTensorSize = vectorProduct(inputTensorShape);

        // Create a vector from the blob data for ONNX Runtime input
        std::vector<float> inputTensorValues(blobPtr, blobPtr + inputTensorSize);

        delete[] blobPtr; // Free the allocated memory for the blob

        // Create an Ort memory info object (can be cached if used repeatedly)
        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor object using the preprocessed data
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputTensorValues.data(),
                inputTensorSize,
                inputTensorShape.data(),
                inputTensorShape.size()
        );

        // Run the inference session with the input tensor and retrieve output tensors
        std::vector<Ort::Value> outputTensors = session.Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                &inputTensor,
                numInputNodes,
                outputNames.data(),
                numOutputNodes
        );

        // Determine the resized image shape based on input tensor shape
        cv::Size resizedImageShape(static_cast<int>(inputTensorShape[3]),
                                   static_cast<int>(inputTensorShape[2]));

        // Postprocess the output tensors to obtain detections
        DetectionList detections = postprocess(frame.size(), resizedImageShape, outputTensors);
        // NX_PRINT << "size of DetectionList " << detections.size();
        return detections; // Return the vector of detections
    }
}