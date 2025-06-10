#!/bin/bash

# AI Security NX Plugin Deployment Script
# Usage: ./deploy.sh [system|user] [nx_meta_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_NAME="ai-security-plugin"

# Default paths
SYSTEM_PLUGIN_DIR="/opt/networkoptix/nx_meta/plugins"
USER_PLUGIN_DIR="$HOME/.nx/plugins"

# Parse arguments
INSTALL_TYPE="${1:-user}"  # Default to user installation
NX_META_PATH="${2:-}"

# Determine target directory
if [ "$INSTALL_TYPE" = "system" ]; then
    if [ -n "$NX_META_PATH" ]; then
        TARGET_DIR="$NX_META_PATH/plugins/$PLUGIN_NAME"
    else
        TARGET_DIR="$SYSTEM_PLUGIN_DIR/$PLUGIN_NAME"
    fi
    NEED_SUDO=true
elif [ "$INSTALL_TYPE" = "user" ]; then
    TARGET_DIR="$USER_PLUGIN_DIR/$PLUGIN_NAME"
    NEED_SUDO=false
else
    echo "❌ Invalid install type. Use 'system' or 'user'"
    exit 1
fi

echo "🚀 AI Security NX Plugin Deployment"
echo "=================================="
echo "Install type: $INSTALL_TYPE"
echo "Target directory: $TARGET_DIR"
echo "Plugin source: $SCRIPT_DIR"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if NX Meta is installed (for system install)
if [ "$INSTALL_TYPE" = "system" ] && [ -z "$NX_META_PATH" ]; then
    if [ ! -d "/opt/networkoptix" ]; then
        echo "❌ NX Meta not found at /opt/networkoptix"
        echo "   Please specify NX Meta path: ./deploy.sh system /path/to/nx_meta"
        exit 1
    fi
fi

# Check CUDA/TensorRT
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: NVIDIA driver not found. GPU acceleration may not work."
fi

# Check plugin files
if [ ! -f "$SCRIPT_DIR/libopencv_object_detection_analytics_plugin.so" ]; then
    echo "❌ Plugin library not found: libopencv_object_detection_analytics_plugin.so"
    echo "   Please build the plugin first."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/referModel/best_new.engine" ]; then
    echo "❌ TensorRT engine not found: referModel/best_new.engine"
    echo "   Please ensure the model file exists."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Create target directory
echo "📁 Creating target directory..."
if [ "$NEED_SUDO" = true ]; then
    sudo mkdir -p "$TARGET_DIR"
else
    mkdir -p "$TARGET_DIR"
fi

# Copy plugin files
echo "📦 Copying plugin files..."
if [ "$NEED_SUDO" = true ]; then
    sudo cp -r "$SCRIPT_DIR"/* "$TARGET_DIR/"
    sudo chmod +x "$TARGET_DIR/libopencv_object_detection_analytics_plugin.so"
    sudo chown -R nx:nx "$TARGET_DIR" 2>/dev/null || echo "⚠️  Could not set nx:nx ownership"
else
    cp -r "$SCRIPT_DIR"/* "$TARGET_DIR/"
    chmod +x "$TARGET_DIR/libopencv_object_detection_analytics_plugin.so"
fi

echo "✅ Plugin files copied successfully"
echo ""

# Display installation summary
echo "📋 Installation Summary"
echo "======================"
echo "Plugin installed to: $TARGET_DIR"
echo "Files installed:"
echo "  ✓ libopencv_object_detection_analytics_plugin.so"
echo "  ✓ referModel/best_new.engine"
echo "  ✓ manifest.json"
echo "  ✓ README.md"
echo ""

# Next steps
echo "🔄 Next Steps"
echo "============"
if [ "$INSTALL_TYPE" = "system" ]; then
    echo "1. Restart NX Meta Server:"
    echo "   sudo systemctl restart nx_meta_server"
    echo ""
fi
echo "2. Enable plugin in NX Meta Web UI:"
echo "   - Go to System → Analytics → Plugins"
echo "   - Find 'AI Security Object Detection with TensorRT Classification'"
echo "   - Enable for desired cameras"
echo ""
echo "3. Monitor logs:"
echo "   tail -f /var/log/nx_meta/nx_meta.log"
echo ""

echo "🎉 Deployment completed successfully!"

# Offer to restart service (system install only)
if [ "$INSTALL_TYPE" = "system" ]; then
    echo ""
    read -p "Would you like to restart NX Meta Server now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Restarting NX Meta Server..."
        sudo systemctl restart nx_meta_server
        echo "✅ NX Meta Server restarted"
    fi
fi
