// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#include "device_agent.h"

#include <chrono>
#include <exception>

#include <opencv2/core.hpp>

#include <nx/sdk/analytics/helpers/event_metadata.h>
#include <nx/sdk/analytics/helpers/event_metadata_packet.h>
#include <nx/sdk/analytics/helpers/object_metadata.h>
#include <nx/sdk/analytics/helpers/object_metadata_packet.h>
#include <nx/sdk/helpers/string.h>

#include "detection.h"
#include "exceptions.h"
#include "frame.h"
#include "visualize.h"
#include "tensorrt_classifier.h"

namespace nx_meta_plugin {

    using namespace nx::sdk;
    using namespace nx::sdk::analytics;

    using namespace std::string_literals;

/**
 * @param deviceInfo Various information about the related device, such as its id, vendor, model,
 *     etc.
 */
    DeviceAgent::DeviceAgent(
            const nx::sdk::IDeviceInfo *deviceInfo,
            std::filesystem::path pluginHomeDir) :
    // Call the DeviceAgent helper class constructor telling it to verbosely report to stderr.
            ConsumingDeviceAgent(deviceInfo, /*enableOutput*/ false),
            m_objectDetector(std::make_unique<YOLO11Detector>(pluginHomeDir)),
            m_tensorrtClassifier(std::make_unique<TensorRTClassifier>(pluginHomeDir)),
            m_objectTracker(std::make_unique<ObjectTracker>()) {
    }

    DeviceAgent::~DeviceAgent() {
    }

/**
 *  @return JSON with the particular structure. Note that it is possible to fill in the values
 * that are not known at compile time, but should not depend on the DeviceAgent settings.
 */
    std::string DeviceAgent::manifestString() const {
        // Tell the Server that the plugin can generate the events and objects of certain types.
        // Id values are strings and should be unique. Format of ids:
        // `{vendor_id}.{plugin_id}.{event_type_id/object_type_id}`.
        //
        // See the plugin manifest for the explanation of vendor_id and plugin_id.
        return /*suppress newline*/ 1 + R"json(
{
    "eventTypes": [
        {
            "id": ")json" + kNewTrackEventType + R"json(",
            "name": "New track started"
        }
    ],
    "supportedTypes": [
        {
            "objectTypeId": ")json" + kPersonObjectType + R"json("
        },
        {
            "objectTypeId": ")json" + kCatObjectType + R"json("
        },
        {
            "objectTypeId": ")json" + kDogObjectType + R"json("
        }
    ]
}
)json";
    }

/**
 * Called when the Server sends a new uncompressed frame from a camera.
 */
    bool DeviceAgent::pushUncompressedVideoFrame(const IUncompressedVideoFrame *videoFrame) {
        m_terminated = m_terminated || m_objectDetector->isTerminated() || 
                      m_tensorrtClassifier->isTerminated();
        if (m_terminated) {
            if (!m_terminatedPrevious) {
                pushPluginDiagnosticEvent(
                        IPluginDiagnosticEvent::Level::error,
                        "Plugin is in broken state.",
                        "Disable the plugin.");
                m_terminatedPrevious = true;
            }
            return true;
        }

        m_lastVideoFrameTimestampUs = videoFrame->timestampUs();

        // Detecting objects only on every `kDetectionFramePeriod` frame.
        if (m_frameIndex % kDetectionFramePeriod == 0) {
            const MetadataPacketList metadataPackets = processFrame(videoFrame);
            for (const Ptr<IMetadataPacket> &metadataPacket: metadataPackets) {
                metadataPacket->addRef();
                pushMetadataPacket(metadataPacket.get());
            }
        }

        ++m_frameIndex;

        return true;
    }

    void DeviceAgent::doSetNeededMetadataTypes(
            nx::sdk::Result<void> *outValue,
            const nx::sdk::analytics::IMetadataTypes * /*neededMetadataTypes*/) {
        if (m_terminated)
            return;

        try {
            m_objectDetector->ensureInitialized();
            m_tensorrtClassifier->ensureInitialized();
        }
        catch (const ObjectDetectorInitializationError &e) {
            *outValue = {ErrorCode::otherError, new String(e.what())};
            m_terminated = true;
        }
        catch (const ObjectDetectorIsTerminatedError & /*e*/) {
            m_terminated = true;
        }
        catch (const std::runtime_error &e) {
            *outValue = {ErrorCode::otherError, new String(std::string("TensorRT initialization error: ") + e.what())};
            m_terminated = true;
        }
        catch (const std::exception &e) {
            *outValue = {ErrorCode::otherError, new String(std::string("Initialization error: ") + e.what())};
            m_terminated = true;
        }
    };

//-------------------------------------------------------------------------------------------------
// private

    Ptr<IMetadataPacket> DeviceAgent::generateEventMetadataPacket() {
        // Generate event every kTrackFrameCount'th frame.
        if (m_frameIndex % kTrackFrameCount != 0)
            return nullptr;

        // EventMetadataPacket contains arbitrary number of EventMetadata.
        const auto eventMetadataPacket = makePtr<EventMetadataPacket>();
        // Bind event metadata packet to the last video frame using a timestamp.
        eventMetadataPacket->setTimestampUs(m_lastVideoFrameTimestampUs);
        // Zero duration means that the event is not sustained, but momental.
        eventMetadataPacket->setDurationUs(0);

        // EventMetadata contains an information about event.
        const auto eventMetadata = makePtr<EventMetadata>();
        // Set all required fields.
        eventMetadata->setTypeId(kNewTrackEventType);
        eventMetadata->setIsActive(true);
        eventMetadata->setCaption("New sample plugin track started");
        eventMetadata->setDescription("New track #" + std::to_string(m_trackIndex) + " started");

        eventMetadataPacket->addItem(eventMetadata.get());

        // Generate index for the next track.
        ++m_trackIndex;

        return eventMetadataPacket;
    }

    Ptr<ObjectMetadataPacket> DeviceAgent::detectionsToObjectMetadataPacket(
            const DetectionList &detections,
            int64_t timestampUs) {
        if (detections.empty())
            return nullptr;

        const auto objectMetadataPacket = makePtr<ObjectMetadataPacket>();

        for (const std::shared_ptr<Detection> &detection: detections) {
            const auto objectMetadata = makePtr<ObjectMetadata>();

            objectMetadata->setBoundingBox(detection->boundingBox);
            objectMetadata->setConfidence(detection->confidence);
            objectMetadata->setTrackId(detection->trackId);

            // Convert class label to object metadata type id.
            if (detection->classLabel == "person")
                objectMetadata->setTypeId(kPersonObjectType);
            else if (detection->classLabel == "cat")
                objectMetadata->setTypeId(kCatObjectType);
            else if (detection->classLabel == "dog")
                objectMetadata->setTypeId(kDogObjectType);
            else if (detection->classLabel == "CA") {
                objectMetadata->setTypeId(kPersonObjectType);
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "Type", "CA"));
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "TrackID",
                                                                         UuidHelper::toStdString(detection->trackId)));
                objectMetadata->addAttribute(makePtr<Attribute>(Attribute::Type::string, "nx.sys.color", "Green"));
            } else if (detection->classLabel == "PN") {
                objectMetadata->setTypeId(kPersonObjectType);
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "Type", "PN"));
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "TrackID",
                                                                         UuidHelper::toStdString(detection->trackId)));
                objectMetadata->addAttribute(makePtr<Attribute>(Attribute::Type::string, "nx.sys.color", "Red"));
            } else if (detection->classLabel == "Unknown") {
                objectMetadata->setTypeId(kPersonObjectType);
                objectMetadata->setSubtype("Unknown");
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "Type", "Unknown"));
                objectMetadata->addAttribute(nx::sdk::makePtr<Attribute>(IAttribute::Type::string, "TrackID",
                                                                         UuidHelper::toStdString(detection->trackId)));
                objectMetadata->addAttribute(makePtr<Attribute>(Attribute::Type::string, "nx.sys.color", "Yellow"));
            }
            // There is no "else", because only the detections with those types are generated.

            objectMetadataPacket->addItem(objectMetadata.get());
        }
        objectMetadataPacket->setTimestampUs(timestampUs);

        return objectMetadataPacket;
    }

    void DeviceAgent::reinitializeObjectTrackerOnFrameSizeChanges(const Frame &frame) {
        const bool frameSizeUnset = m_previousFrameWidth == 0 && m_previousFrameHeight == 0;
        if (frameSizeUnset) {
            m_previousFrameWidth = frame.width;
            m_previousFrameHeight = frame.height;
            return;
        }

        const bool frameSizeChanged = frame.width != m_previousFrameWidth ||
                                      frame.height != m_previousFrameHeight;
        if (frameSizeChanged) {
            m_objectTracker = std::make_unique<ObjectTracker>();
            m_previousFrameWidth = frame.width;
            m_previousFrameHeight = frame.height;
        }
    }

    DeviceAgent::MetadataPacketList DeviceAgent::processFrame(
            const IUncompressedVideoFrame *videoFrame) {
        const Frame frame(videoFrame, m_frameIndex);
        reinitializeObjectTrackerOnFrameSizeChanges(frame);

        try {
            cv::Mat image = frame.cvMat;
            
            DetectionList detections;
            
            // Performance optimization: Use cached detections for some frames
            bool useCachedDetections = !m_lastDetections.empty() && 
                                     (m_frameIndex - m_lastDetectionFrameIndex) < kDetectionCacheFrames;
            
            if (useCachedDetections) {
                // Use cached detections and update with tracker
                detections = m_lastDetections;
                detections = m_objectTracker->run(frame, detections);
            } else {
                // Stage 1: Object Detection using YOLO (expensive operation)
                detections = m_objectDetector->run(image);
                detections = m_objectTracker->run(frame, detections);
                
                // Cache the detections
                m_lastDetections = detections;
                m_lastDetectionFrameIndex = m_frameIndex;
            }
            
            // Stage 2: Classification using TensorRT for each detected object
            if (!detections.empty()) {
                // Limit the number of detections to classify for performance
                const size_t maxDetectionsToClassify = 5; // Process max 5 objects per frame
                
                // Filter detections for classification (skip small/low-confidence ones for performance)
                DetectionList detectionsToClassify;
                size_t processedCount = 0;
                
                for (const auto& detection : detections) {
                    if (processedCount >= maxDetectionsToClassify) {
                        // Set remaining detections to Unknown for performance
                        detection->classLabel = "Unknown";
                        continue;
                    }
                    
                    // Only classify if bounding box is large enough and confidence is high enough
                    float boxArea = detection->boundingBox.width * detection->boundingBox.height;
                    if (boxArea > 0.01f && detection->confidence > 0.6f) { // At least 1% of frame and 60% confidence
                        detectionsToClassify.push_back(detection);
                        processedCount++;
                    } else {
                        // Set default classification for small/low-confidence detections
                        detection->classLabel = "Unknown";
                    }
                }
                
                // Only run TensorRT if we have detections worth classifying
                if (!detectionsToClassify.empty()) {
                    m_tensorrtClassifier->classifyDetections(image, detectionsToClassify);
                }
            }

            // Convert detections to metadata packets for UI display
            const auto &objectMetadataPacket =
                    detectionsToObjectMetadataPacket(detections, frame.timestampUs);
            MetadataPacketList result;
            if (objectMetadataPacket)
                result.push_back(objectMetadataPacket);
            return result;
        }
        catch (const ObjectDetectionError &e) {
            pushPluginDiagnosticEvent(
                    IPluginDiagnosticEvent::Level::error,
                    "Object detection error.",
                    e.what());
            m_terminated = true;
        }
        catch (const ObjectTrackingError &e) {
            pushPluginDiagnosticEvent(
                    IPluginDiagnosticEvent::Level::error,
                    "Object tracking error.",
                    e.what());
            m_terminated = true;
        }
        catch (const std::exception &e) {
            pushPluginDiagnosticEvent(
                    IPluginDiagnosticEvent::Level::error,
                    "Classification error.",
                    e.what());
            m_terminated = true;
        }

        return {};
    }

}
