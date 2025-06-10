#!/bin/bash

echo "🧪 Testing Plugin Direct Access"
echo "==============================="

# Check if the plugin library can be loaded
echo "📚 Testing plugin library loading..."
if command -v ldd >/dev/null 2>&1; then
    echo "Dependencies for plugin library:"
    ldd /opt/networkoptix-metavms/mediaserver/bin/plugins/libopencv_object_detection_analytics_plugin.so | head -10
else
    echo "ldd not available, skipping dependency check"
fi

echo ""
echo "🔍 Checking NX Meta plugin registration..."

# Look for our plugin in the system
if pgrep -f mediaserver >/dev/null; then
    echo "✅ NX Meta mediaserver is running"
    
    # Check plugin loading in real-time logs
    echo "📋 Recent plugin initialization logs:"
    sudo journalctl -u networkoptix-metavms-mediaserver.service --since "5 minutes ago" | grep -i "opencv_object_detection_analytics_plugin" | tail -5
    
    echo ""
    echo "🔄 Checking plugin manifest registration..."
    # Check if the plugin manifest is being read
    sudo journalctl -u networkoptix-metavms-mediaserver.service --since "5 minutes ago" | grep -i "manifest\|analytics\|plugin" | tail -10
    
else
    echo "❌ NX Meta mediaserver is not running"
fi

echo ""
echo "🌐 NX Meta Web Interface Access:"
echo "   Primary:   http://localhost:7001"
echo "   Secondary: https://localhost:7001"
echo "   Admin UI:  http://localhost:7001/admin"
echo ""

echo "🎯 Manual Plugin Verification Steps:"
echo "1. Open NX Meta client or web interface"
echo "2. Check System → Analytics → Plugins"
echo "3. Look for cameras in System → Cameras"
echo "4. Right-click camera → Analytics Settings"
echo "5. Enable 'AI Security Object Detection with TensorRT Classification'"
echo ""

echo "📊 If plugin still not visible, try:"
echo "   sudo systemctl restart networkoptix-metavms-mediaserver.service"
echo "   # Wait 30 seconds, then check again"
