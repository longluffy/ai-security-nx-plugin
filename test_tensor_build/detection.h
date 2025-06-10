#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace nx_meta_plugin {
    // Simplified detection structure for testing
    struct Detection {
        std::string classLabel = "Unknown";
    };

    using DetectionList = std::vector<std::shared_ptr<Detection>>;
}
