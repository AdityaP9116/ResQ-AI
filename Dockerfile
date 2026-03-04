# ============================================================================
# ResQ-AI Isaac Sim Docker Image
# 
# Builds a containerized environment with:
# - NVIDIA CUDA 12.2
# - Python 3.10
# - Isaac Sim SDK (headless compatible)
# - All ResQ-AI dependencies
#
# Usage:
#   docker build -t resq-ai:latest .
#   docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
#     python test_isaacsim_minimal.py
# ============================================================================

FROM ubuntu:22.04

LABEL maintainer="ResQ-AI Team"
LABEL description="NVIDIA Isaac Sim environment for ResQ-AI autonomous drone"

# ============================================================================
# System Dependencies
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    curl \
    wget \
    ca-certificates \
    git \
    vim \
    \
    # Python 3.10 stack
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    \
    # System libraries for Isaac Sim & graphics
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    libnuma1 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shm0 \
    libxkbcommon0 \
    libdbus-1-3 \
    libfontconfig1 \
    \
    # Required for video/encoding
    ffmpeg \
    libavcodec-extra \
    \
    # Utilities
    sudo \
    less \
    htop \
    \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ============================================================================
# Workspace Setup
# ============================================================================

WORKDIR /workspace

# Create directories
RUN mkdir -p /workspace/resq-ai \
    && mkdir -p /workspace/isaac-sim \
    && mkdir -p /opt/isaac_env

# ============================================================================
# Python Virtual Environment & Dependencies
# ============================================================================

# Create venv
RUN python3.10 -m venv /opt/isaac_env

# Activate venv for all subsequent RUN commands
ENV PATH="/opt/isaac_env/bin:$PATH" \
    VIRTUAL_ENV="/opt/isaac_env" \
    PYTHONUNBUFFERED=1

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies for ResQ-AI
RUN pip install --no-cache-dir \
    # Core ML/CV stack
    ultralytics==8.0.202 \
    opencv-python==4.8.1.78 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    \
    # Core utilities
    numpy==1.24.3 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    pillow==10.1.0 \
    \
    # Web/API
    requests==2.31.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    flask==3.0.0 \
    \
    # Data & visualization
    folium==0.14.0 \
    \
    # Environment & logging
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    \
    # Optional but useful
    ipython==8.17.2 \
    jupyter==1.0.0 \
    pytest==7.4.3

# ============================================================================
# Isaac Sim Installation (full pip-based, from NVIDIA PyPI)
# ============================================================================

# The full Isaac Sim SDK is distributed across several packages on
# https://pypi.nvidia.com. We add it as an extra index so standard
# PyPI packages still resolve normally.
#
# Package sizes:
#   isaacsim-extscache-kit        ~2.5 GB
#   isaacsim-extscache-kit-sdk    ~1.3 GB
#   isaacsim-extscache-physics    ~248 MB
#   isaacsim-core                 ~67 MB
RUN pip install --no-cache-dir \
    isaacsim==4.5.0.0 \
    isaacsim-core==4.5.0.0 \
    isaacsim-extscache-physics==4.5.0.0 \
    isaacsim-extscache-kit==4.5.0.0 \
    isaacsim-extscache-kit-sdk==4.5.0.0 \
    --extra-index-url https://pypi.nvidia.com

# Accept the NVIDIA EULA on first import so containers don't stall
RUN echo "Yes" | python3 -c "import isaacsim" || true

# ============================================================================
# PegasusSimulator Installation
# ============================================================================

# Clone and install PegasusSimulator (Isaac Sim drone extension)
RUN git clone --depth=1 https://github.com/PegasusSimulator/PegasusSimulator.git /tmp/PegasusSimulator && \
    ISAACSIM_PATH=$(python3 -c "import isaacsim,os; print(os.path.dirname(os.path.abspath(isaacsim.__file__)))") \
    pip install --no-cache-dir -e /tmp/PegasusSimulator/extensions/pegasus.simulator && \
    rm -rf /tmp/PegasusSimulator

# ============================================================================
# ResQ-AI Source Code
# ============================================================================

COPY . /workspace/resq-ai/

# Set permissions
RUN chmod -R 755 /workspace/resq-ai \
    && find /workspace/resq-ai -name "*.py" -exec chmod +x {} \;

# ============================================================================
# Environment Configuration
# ============================================================================

# Create .env if it doesn't exist
RUN if [ ! -f /workspace/resq-ai/.env ]; then \
    echo "# Isaac Sim Configuration" > /workspace/resq-ai/.env && \
    echo "HEADLESS_MODE=true" >> /workspace/resq-ai/.env && \
    echo "NVIDIA_VISIBLE_DEVICES=all" >> /workspace/resq-ai/.env && \
    echo "" >> /workspace/resq-ai/.env && \
    echo "# VLM Backend Configuration" >> /workspace/resq-ai/.env && \
    echo "VLM_BACKEND=mock" >> /workspace/resq-ai/.env && \
    echo "" >> /workspace/resq-ai/.env && \
    echo "# Simulation Configuration" >> /workspace/resq-ai/.env && \
    echo "SCENE_RESOLUTION=1280x720" >> /workspace/resq-ai/.env && \
    echo "OUTPUT_DIR=/workspace/resq-ai/outputs" >> /workspace/resq-ai/.env && \
    echo "" >> /workspace/resq-ai/.env && \
    echo "# Python path" >> /workspace/resq-ai/.env && \
    echo "PYTHONPATH=/workspace/resq-ai:\$PYTHONPATH" >> /workspace/resq-ai/.env; \
    fi

# ============================================================================
# Entry Point Configuration
# ============================================================================

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'set -e' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Activate virtual environment' >> /entrypoint.sh && \
    echo '. /opt/isaac_env/bin/activate' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Change to ResQ-AI directory' >> /entrypoint.sh && \
    echo 'cd /workspace/resq-ai' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Create outputs directory if it doesn'"'"'t exist' >> /entrypoint.sh && \
    echo 'mkdir -p outputs' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Execute the command' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh

RUN chmod +x /entrypoint.sh

# ============================================================================
# Final Configuration
# ============================================================================

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Document exposed ports (even in headless mode)
# Port 8000: VLM Server
# Port 8211: Isaac Sim WebRTC livestream  (http://localhost:8211/streaming/webrtc-demo/)
# Port 8001: Optional dashboard/monitoring
EXPOSE 8000 8001 8211

# Volume mount points
VOLUME ["/workspace/resq-ai", "/workspace/outputs"]

# ============================================================================
# Build metadata
# ============================================================================

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=5.1

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source=https://github.com/AdityaP9116/ResQ-AI \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.title="ResQ-AI Isaac Sim" \
      org.opencontainers.image.description="NVIDIA Isaac Sim environment for ResQ-AI autonomous disaster response drone"
