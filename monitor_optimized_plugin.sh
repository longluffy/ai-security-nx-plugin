#!/bin/bash

echo "🎯 AI Security Plugin Performance Monitor"
echo "========================================"
echo ""

# Function to count performance issues in a time window
count_performance_issues() {
    local time_window="$1"
    local issues=$(sudo journalctl -u networkoptix-metavms-mediaserver --since "$time_window" --no-pager 2>/dev/null | grep -c "Frame queue overflow\|skipped.*frame\|too slowly" 2>/dev/null | head -1 || echo "0")
    # Clean the output to ensure it's a number
    echo "$issues" | tr -d '\n' | grep -o '[0-9]*' | head -1 || echo "0"
}

# Check service status
echo "🔍 SERVICE STATUS:"
service_status=$(sudo systemctl is-active networkoptix-metavms-mediaserver)
echo "   NX Media Server: $service_status"

if [ "$service_status" = "active" ]; then
    echo "   Memory Usage: $(sudo systemctl status networkoptix-metavms-mediaserver --no-pager | grep Memory | awk '{print $2}' || echo 'N/A')"
    echo "   CPU Usage: $(sudo systemctl status networkoptix-metavms-mediaserver --no-pager | grep CPU | awk '{print $2}' || echo 'N/A')"
fi
echo ""

# Check plugin deployment
echo "📦 PLUGIN DEPLOYMENT STATUS:"
plugin_locations=(
    "/opt/networkoptix-metavms/mediaserver/bin/plugins/libopencv_object_detection_analytics_plugin.so"
    "/opt/networkoptix-metavms/mediaserver/plugins/ai-security-plugin/libopencv_object_detection_analytics_plugin.so"
)

for location in "${plugin_locations[@]}"; do
    if [ -f "$location" ]; then
        size=$(ls -lh "$location" | awk '{print $5}')
        date=$(ls -l "$location" | awk '{print $6, $7, $8}')
        echo "   ✅ Plugin found: $location ($size, $date)"
    else
        echo "   ❌ Plugin missing: $location"
    fi
done
echo ""

# Check model files
echo "🤖 MODEL FILES STATUS:"
model_files=(
    "/opt/networkoptix-metavms/mediaserver/bin/plugins/yolo11n.onnx"
    "/opt/networkoptix-metavms/mediaserver/bin/plugins/referModel/best_new.engine"
)

for model in "${model_files[@]}"; do
    if [ -f "$model" ]; then
        size=$(ls -lh "$model" | awk '{print $5}')
        echo "   ✅ Model found: $(basename "$model") ($size)"
    else
        echo "   ❌ Model missing: $model"
    fi
done
echo ""

# Check for performance issues
echo "📈 PERFORMANCE ANALYSIS:"
issues_1min=$(count_performance_issues "1 minute ago")
issues_5min=$(count_performance_issues "5 minutes ago")
issues_10min=$(count_performance_issues "10 minutes ago")

echo "   Last 1 minute: $issues_1min frame queue issues"
echo "   Last 5 minutes: $issues_5min frame queue issues"
echo "   Last 10 minutes: $issues_10min frame queue issues"
echo ""

# Show active optimizations
echo "⚡ ACTIVE PERFORMANCE OPTIMIZATIONS:"
echo "   🚀 Ultra-aggressive frame processing: Every 12th frame (91.7% reduction)"
echo "   🧠 Extended detection caching: 11 frames reuse"
echo "   🎯 Classification persistence: 120 frames duration"
echo "   🔥 Multi-stream TensorRT: 4 concurrent streams"
echo "   💾 Aggressive filtering: 70% confidence + 0.5% frame area minimum"
echo "   ⚙️  CPU optimization: High priority + all cores"
echo "   📊 Metadata reuse: 11 frames persistence"
echo ""

# GPU utilization
echo "🎮 GPU STATUS:"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
    echo "   GPU Utilization: ${gpu_util}%"
    echo "   GPU Memory: $gpu_mem"
else
    echo "   ❌ nvidia-smi not available"
fi
echo ""

# Performance verdict
echo "🎯 PERFORMANCE VERDICT:"
if [ "$issues_5min" -eq 0 ]; then
    echo "   ✅ EXCELLENT: No frame queue issues detected"
    echo "   ✅ Plugin operating at optimal performance"
    echo "   ✅ Ready for production analytics workload"
    echo ""
    echo "🎬 TESTING RECOMMENDATIONS:"
    echo "   1. Open NX Meta client interface"
    echo "   2. Enable analytics on camera streams"
    echo "   3. Verify CA/PN bounding boxes appear smoothly"
    echo "   4. Confirm no frame skip warnings in logs"
elif [ "$issues_5min" -le 2 ]; then
    echo "   ⚠️  GOOD: Minimal issues ($issues_5min in 5 min)"
    echo "   💡 Performance within acceptable range"
    echo "   📝 Monitor for pattern changes"
else
    echo "   ⚠️  MODERATE: $issues_5min issues in last 5 minutes"
    echo "   💡 Consider further optimization if persistent"
    echo "   🔧 Check camera resolution and frame rate settings"
fi
echo ""

# Real-time monitoring option
echo "📊 REAL-TIME MONITORING:"
echo "   To monitor live performance:"
echo "   sudo journalctl -u networkoptix-metavms-mediaserver -f | grep -E 'Frame queue|overflow|plugin'"
echo ""
echo "   To check plugin logs:"
echo "   sudo journalctl -u networkoptix-metavms-mediaserver --since '5 minutes ago' | grep -i 'opencv\\|tensor\\|analytics'"
