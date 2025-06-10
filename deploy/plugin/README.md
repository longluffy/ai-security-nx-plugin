# AI Security NX Plugin - Deployment Guide

## 🚀 Quick Deployment

### Prerequisites
- NX Meta Server running on Linux
- NVIDIA GPU with CUDA support
- TensorRT 10.x installed
- Minimum 4GB RAM, 2GB GPU memory

### Installation Steps

#### 1. Copy Plugin Files
```bash
# Copy to NX Meta plugins directory
sudo cp -r /path/to/plugin/* /opt/networkoptix/nx_meta/plugins/ai-security-plugin/

# Or copy to user plugins directory
cp -r /path/to/plugin/* ~/.nx/plugins/ai-security-plugin/
```

#### 2. Set Permissions
```bash
sudo chmod +x /opt/networkoptix/nx_meta/plugins/ai-security-plugin/libopencv_object_detection_analytics_plugin.so
sudo chown nx:nx -R /opt/networkoptix/nx_meta/plugins/ai-security-plugin/
```

#### 3. Restart NX Meta Server
```bash
sudo systemctl restart nx_meta_server
```

#### 4. Enable Plugin in NX Web UI
1. Open NX Meta web interface
2. Go to System → Analytics → Plugins
3. Find "AI Security Object Detection with TensorRT Classification"
4. Enable the plugin for desired cameras
5. Configure detection parameters if needed

### Verification

#### Check Plugin Loading
```bash
# Check NX Meta logs for plugin loading
sudo tail -f /var/log/nx_meta/nx_meta.log | grep "ai-security"
```

#### Expected Log Messages
```
[INFO] Loading plugin: AI Security Object Detection with TensorRT Classification
[INFO] TensorRT classifier engine loaded successfully
[INFO] Two-stage AI pipeline initialized
```

### Configuration

#### Plugin Settings (via NX UI)
- **Detection Frame Period**: Every N frames (default: 2)
- **Confidence Threshold**: Minimum detection confidence (default: 0.4)
- **Classification Threshold**: Minimum classification confidence (default: 0.5)

#### Object Classes
- **CA** (Class 0): Displayed with green bounding boxes
- **PN** (Class 1): Displayed with red bounding boxes
- **Unknown**: Displayed with yellow bounding boxes (fallback)

### Performance Tuning

#### GPU Memory Optimization
The plugin allocates ~1.2MB GPU memory per classification session.
For multiple cameras, monitor GPU memory usage:
```bash
nvidia-smi -l 1
```

#### CPU/GPU Load Balancing
- Adjust `kDetectionFramePeriod` in plugin config
- Higher values = lower CPU load, less frequent detection
- Lower values = higher accuracy, more resource usage

### Troubleshooting

#### Plugin Not Loading
1. Check file permissions
2. Verify TensorRT installation
3. Check CUDA driver compatibility
4. Review NX Meta logs

#### Poor Performance
1. Reduce detection frame period
2. Lower input resolution
3. Monitor GPU memory usage
4. Check CUDA/TensorRT versions

#### Classification Errors
1. Verify `best_new.engine` file exists and is readable
2. Check GPU memory availability
3. Review TensorRT compatibility

### File Structure
```
ai-security-plugin/
├── libopencv_object_detection_analytics_plugin.so  # Main plugin
├── manifest.json                                    # Plugin metadata
├── referModel/
│   ├── best_new.engine                             # TensorRT classifier
│   └── ...
└── README.md                                       # This file
```

### Support
- Check logs: `/var/log/nx_meta/nx_meta.log`
- GPU diagnostics: `nvidia-smi`
- Plugin status: NX Meta web interface → System → Analytics → Plugins
