#!/bin/bash

echo "🚀 Quick Deploy Optimized Plugin"
echo "================================="

# Copy YOLO model to deployment directory
echo "📦 Step 1: Copying YOLO model..."
if [ -f "/home/longluffy/code/yolo11n.onnx" ]; then
    cp /home/longluffy/code/yolo11n.onnx /home/longluffy/code/pluggin/ai-security-nx-plugin/deploy/plugin/
    echo "   ✅ yolo11n.onnx copied to deployment directory"
else
    echo "   ❌ YOLO model not found at /home/longluffy/code/yolo11n.onnx"
    exit 1
fi

# Verify deployment package has all required files
echo "📋 Step 2: Verifying deployment package..."
cd /home/longluffy/code/pluggin/ai-security-nx-plugin/deploy/plugin

missing_files=()
[ ! -f "libopencv_object_detection_analytics_plugin.so" ] && missing_files+=("libopencv_object_detection_analytics_plugin.so")
[ ! -f "manifest.json" ] && missing_files+=("manifest.json")
[ ! -f "referModel/best_new.engine" ] && missing_files+=("referModel/best_new.engine")
[ ! -f "yolov11n.onnx" ] && missing_files+=("yolov11n.onnx")

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "   ✅ All required files present"
    echo ""
    echo "📋 Deployment package contents:"
    echo "   ✓ $(ls -lh libopencv_object_detection_analytics_plugin.so | awk '{print $9, $5}')"
    echo "   ✓ $(ls -lh manifest.json | awk '{print $9, $5}')"
    echo "   ✓ $(ls -lh referModel/best_new.engine | awk '{print $9, $5}' 2>/dev/null || echo 'best_new.engine (not found)')"
    echo "   ✓ $(ls -lh yolov11n.onnx | awk '{print $9, $5}')"
    
    echo ""
    echo "🎉 DEPLOYMENT PACKAGE READY!"
    echo ""
    echo "🚀 Deploy with one of these commands:"
    echo ""
    echo "📍 OPTION 1 - System deployment (recommended):"
    echo "   sudo ./deploy.sh system /opt/networkoptix-metavms/mediaserver"
    echo ""
    echo "📍 OPTION 2 - Manual plugin directory deployment:"
    echo "   sudo ./deploy.sh plugin /opt/networkoptix-metavms/mediaserver/bin/plugins"
    echo ""
    echo "📍 OPTION 3 - Quick update (replace existing):"
    echo "   sudo cp libopencv_object_detection_analytics_plugin.so /opt/networkoptix-metavms/mediaserver/bin/plugins/"
    echo "   sudo cp yolov11n.onnx /opt/networkoptix-metavms/mediaserver/bin/plugins/"
    echo "   sudo systemctl restart networkoptix-metavms-mediaserver"
    echo ""
    echo "⚡ Performance optimizations in the current code:"
    echo "   • Ultra-aggressive frame skipping (processing every 12th frame)"
    echo "   • Extended detection caching (11 frames reuse)"
    echo "   • Classification persistence (120 frames)" 
    echo "   • Multi-stream TensorRT processing (4 concurrent streams)"
    echo "   • CPU affinity and high priority scheduling"
    echo "   • Aggressive filtering (70% confidence, 0.5% frame area minimum)"
    echo ""
    echo "📊 Expected performance improvements:"
    echo "   • ~91.7% reduction in frame processing load"
    echo "   • ~95% reduction in classification operations"
    echo "   • Elimination of frame queue overflow issues"
    echo "   • Smooth video playback with real-time analytics"
    
else
    echo "   ❌ Missing required files: ${missing_files[*]}"
    exit 1
fi
