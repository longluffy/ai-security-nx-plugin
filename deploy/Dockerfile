# Use NVIDIA CUDA runtime as base
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=11.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT
ARG TENSORRT_VERSION=10.5.0
RUN wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.5.0/tars/TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && tar -xzf TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && cp -r TensorRT-${TENSORRT_VERSION}/lib/* /usr/lib/x86_64-linux-gnu/ \
    && cp -r TensorRT-${TENSORRT_VERSION}/include/* /usr/include/ \
    && rm -rf TensorRT-${TENSORRT_VERSION}*

# Install OpenCV
ARG OPENCV_VERSION=4.8.0
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip \
    && mkdir -p opencv-${OPENCV_VERSION}/build \
    && cd opencv-${OPENCV_VERSION}/build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D INSTALL_PYTHON_EXAMPLES=ON \
             -D INSTALL_C_EXAMPLES=OFF \
             -D OPENCV_ENABLE_NONFREE=ON \
             -D BUILD_EXAMPLES=OFF .. \
    && make -j$(nproc) \
    && make install \
    && cd / && rm -rf opencv*

# Install NX Meta SDK (placeholder - replace with actual installation)
COPY metadata_sdk/ /opt/nx_meta_sdk/

# Create application directory
WORKDIR /app

# Copy plugin files
COPY deploy/plugin/ /app/plugin/

# Set up environment
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create plugin directory
RUN mkdir -p /opt/networkoptix/nx_meta/plugins/ai-security-plugin

# Copy plugin to NX location
RUN cp -r /app/plugin/* /opt/networkoptix/nx_meta/plugins/ai-security-plugin/

# Expose NX Meta ports
EXPOSE 7001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi && echo "GPU OK" || exit 1

# Start NX Meta (adjust command as needed)
CMD ["/opt/networkoptix/nx_meta/bin/nx_meta_server"]
