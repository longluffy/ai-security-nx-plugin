#!/bin/bash

echo "🚀 Building Optimized AI Security Plugin"
echo "========================================"

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Performance-optimized compilation flags
export CXXFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG -fopenmp"

echo "🔧 Step 1: Building NX SDK dependencies..."
cd /home/longluffy/code/metadata_sdk
if [ ! -f "/home/longluffy/code/metadata_sdk-build/nx_kit/libnx_kit.a" ]; then
    echo "   Building NX SDK..."
    ./build_samples.sh >/dev/null 2>&1
    echo "   ✅ NX SDK built successfully"
else
    echo "   ✅ NX SDK already built"
fi

echo "🔧 Step 2: Setting up build environment..."
cd "$SCRIPT_DIR"

# Create optimized build directory
rm -rf build_optimized
mkdir build_optimized
cd build_optimized

echo "🎯 Step 3: Configuring with optimizations..."

# Use the working cmake configuration from metadata_sdk
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG -fopenmp -flto" \
    -DmetadataSdkDir=/home/longluffy/code/metadata_sdk \
    .. 2>&1 | grep -E "(Error|error|failed|Failed)" || echo "   ✅ Configuration successful"

echo "⚙️ Step 4: Compiling optimized plugin..."

# Build with all CPU cores
make -j$(nproc) 2>&1 | grep -E "(Error|error|failed|Failed)" || echo "   ✅ Compilation successful"

# Check if plugin was built
if [ -f "libopencv_object_detection_analytics_plugin.so" ]; then
    echo "✅ Plugin built successfully!"
    
    # Show file info
    echo "📊 Plugin details:"
    ls -lh libopencv_object_detection_analytics_plugin.so
    
    echo "🚚 Step 5: Copying to deployment directory..."
    cp libopencv_object_detection_analytics_plugin.so ../deploy/plugin/
    
    echo "📦 Step 6: Verifying deployment package..."
    cd ../deploy/plugin
    
    # Verify all required files are present
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
        echo "   ✓ $(ls -lh yolov11n.onnx | awk '{print $9, $5}' 2>/dev/null || echo 'yolov11n.onnx (checking...')"
        
        # Check for YOLO model in various locations
        if [ ! -f "yolov11n.onnx" ]; then
            if [ -f "/home/longluffy/code/yolo11n.onnx" ]; then
                cp /home/longluffy/code/yolo11n.onnx .
                echo "   ✓ yolov11n.onnx (copied from /home/longluffy/code/)"
            elif [ -f "../yolov11n.onnx" ]; then
                cp ../yolov11n.onnx .
                echo "   ✓ yolov11n.onnx (copied from parent)"
            elif [ -f "../../yolov11n.onnx" ]; then
                cp ../../yolov11n.onnx .
                echo "   ✓ yolov11n.onnx (copied from root)"
            else
                echo "   ❌ yolov11n.onnx not found in expected locations"
                missing_files+=("yolov11n.onnx")
            fi
        else
            echo "   ✓ yolov11n.onnx (already present)"
        fi
        
        echo ""
        echo "🎉 BUILD COMPLETED SUCCESSFULLY!"
        echo ""
        echo "🚀 Ready to deploy! Run the following command:"
        echo "   cd $(pwd) && sudo ./deploy.sh system /opt/networkoptix-metavms/mediaserver"
        echo ""
        echo "⚡ Performance optimizations included:"
        echo "   • Ultra-aggressive frame skipping (91.7% reduction)"
        echo "   • Extended detection caching (11 frames)"
        echo "   • Classification persistence (120 frames)" 
        echo "   • Multi-stream TensorRT processing"
        echo "   • CPU affinity and priority optimizations"
        echo "   • Maximum compiler optimizations (-O3, LTO, native)"
        
    else
        echo "❌ Missing required files: ${missing_files[*]}"
        exit 1
    fi
    
else
    echo "❌ Plugin build failed - library not found"
    exit 1
fi
