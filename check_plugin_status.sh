#!/bin/bash

echo "🔍 NX Meta Plugin Status Check"
echo "=============================="
echo ""

# Check if plugin files are in place
echo "📁 Plugin Files:"
if [ -f "/opt/networkoptix-metavms/mediaserver/bin/plugins/libopencv_object_detection_analytics_plugin.so" ]; then
    echo "✅ Plugin library: Found"
    ls -lh /opt/networkoptix-metavms/mediaserver/bin/plugins/libopencv_object_detection_analytics_plugin.so
else
    echo "❌ Plugin library: Not found"
fi

if [ -f "/opt/networkoptix-metavms/mediaserver/bin/plugins/referModel/best_new.engine" ]; then
    echo "✅ TensorRT engine: Found"
    ls -lh /opt/networkoptix-metavms/mediaserver/bin/plugins/referModel/best_new.engine
else
    echo "❌ TensorRT engine: Not found"
fi

echo ""

# Check service status
echo "🔧 NX Meta Service Status:"
sudo systemctl is-active networkoptix-metavms-mediaserver.service

echo ""

# Check recent logs for plugin loading
echo "📋 Recent Plugin Logs:"
echo "Recent logs showing plugin activity:"
sudo journalctl -u networkoptix-metavms-mediaserver.service --since="5 minutes ago" | grep -i opencv | tail -5

echo ""

# Check for any error logs
echo "⚠️  Recent Error Logs:"
sudo journalctl -u networkoptix-metavms-mediaserver.service --since="5 minutes ago" | grep -i error | tail -3

echo ""

# GPU Status
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

echo ""

# Next steps
echo "🚀 Next Steps:"
echo "1. Open NX Meta Client/Web Interface"
echo "2. Go to: System → Analytics → Plugins"
echo "3. Look for: 'AI Security Object Detection with TensorRT Classification'"
echo "4. Enable it for your camera devices"
echo ""
echo "🌐 Web Interface usually available at:"
echo "   http://localhost:7001"
echo "   https://localhost:7001"
echo ""
echo "📱 Desktop Client:"
echo "   Run NX Meta Desktop Client and connect to localhost"
