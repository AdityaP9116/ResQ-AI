# Isaac Sim Setup Guide for ResQ-AI

This guide provides complete instructions for setting up **NVIDIA Isaac Sim 5.1+** to run the ResQ-AI autonomous disaster response drone simulation.

## ⚠️ CRITICAL: Read This First

**Isaac Sim CANNOT be installed with `pip install isaacsim`.** 

It MUST be installed via **NVIDIA Omniverse Launcher** or manual tarball extraction. If you're seeing errors like:
```
PYTHONPATH: path doesn't exist
[✗] Isaac Sim verification failed
```

→ See [ISAAC_SIM_INSTALLATION_FIX.md](ISAAC_SIM_INSTALLATION_FIX.md) for immediate solutions.

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Local Setup (Recommended for Development)](#local-setup)
3. [Docker Setup](#docker-setup)
4. [Verifying Installation](#verifying-installation)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware (Minimum)
- **GPU**: NVIDIA GPU with CUDA support (GeForce RTX 3080+, A100, or equivalent)
- **RAM**: 16 GB (32+ GB recommended for optimal performance)
- **Storage**: 100+ GB free (Isaac Sim: ~30GB, models: ~20GB, datasets: variable)
- **Network**: Internet connection for downloading assets and models

### Software
- **OS**: Ubuntu 22.04 LTS or 20.04 LTS
- **NVIDIA Driver**: Version 550+ (check with `nvidia-smi`)
- **CUDA**: 12.0+
- **Python**: 3.10 or 3.11

### Verify Prerequisites
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA is working
nvidia-smi --query-gpu=compute_cap --format=csv

# Check if you have 100+ GB free
df -h /
```

---

## Local Setup

### Step 1: Install NVIDIA Omniverse Launcher (REQUIRED)

The **ONLY supported way** to install Isaac Sim is via the **NVIDIA Omniverse Launcher**.

```bash
# Download the Omniverse Launcher
curl -fsSL https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage -o ~/Downloads/omniverse-launcher.AppImage

# Make it executable
chmod +x ~/Downloads/omniverse-launcher.AppImage

# Run it
~/Downloads/omniverse-launcher.AppImage
```

**In the Launcher GUI:**
1. Sign in with your NVIDIA account (free)
2. Search for **"Isaac Sim"**
3. Select **Version 5.1+**
4. Click **Install**
5. Wait for installation to complete (~30-45 minutes)

Default installation path: `~/.local/share/ov/pkg/isaac-sim-5.*`

### Step 2: Verify Isaac Sim Installation

```bash
# Test basic Isaac Sim functionality
~/.local/share/ov/pkg/isaac-sim-5.*/python.sh -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"
```

### Step 3: Install ResQ-AI Dependencies

Navigate to the ResQ-AI root directory:

```bash
cd /path/to/ResQ-AI

# Create a Python virtual environment
python3.10 -m venv isaac_env
source isaac_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install \
  ultralytics \
  opencv-python \
  numpy \
  scipy \
  requests \
  torch \
  torchvision \
  torchaudio \
  pyyaml \
  pillow \
  dotenv \
  flask \
  fastapi \
  uvicorn \
  folium \
  matplotlib
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the ResQ-AI root directory:

```bash
cat > /path/to/ResQ-AI/.env << 'EOF'
# Isaac Sim Configuration
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-5.1
ISAAC_SIM_PYTHON=${ISAAC_SIM_PATH}/python.sh

# NVIDIA Omniverse & Omniverse Nucleus (optional, for asset streaming)
OMNIVERSE_NUCLEUS_PATH=omniverse://localhost/NVIDIA/Assets/

# VLM Backend Configuration (for Cosmos Reason 2)
# Options: mock, nim, vllm
VLM_BACKEND=mock
NVIDIA_API_KEY=your_api_key_here

# Simulation Configuration
HEADLESS_MODE=false
SCENE_RESOLUTION=1280x720

# Data Output Paths
OUTPUT_DIR=./outputs
YOLO_WEIGHTS_PATH=./Phase1_SituationalAwareness/best.pt

# Python Path for Isaac Sim
PYTHONPATH=${ISAAC_SIM_PATH}/python:$PYTHONPATH
EOF

source .env
```

### Step 5: Download Required Assets

ResQ-AI requires Omniverse asset packs for buildings, characters, and environments.

#### Option A: Manual Download from Omni

Visit [NVIDIA Omniverse Content](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/)
and download these packs to `assets/`:

```bash
mkdir -p assets

# Download from Omniverse website and extract to:
# - assets/Architecture/    (CityEngine buildings)
# - assets/Characters/      (Reallusion characters)
# - assets/Particles/       (Fire/smoke effects)
# - assets/BaseMaterials/   (Textures & materials)
```

#### Option B: Programmatic Download (if available)

```bash
# From within the ResQ-AI directory with Isaac Sim active:
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh utils/model_downloader.py
```

### Step 6: Download Pre-trained Models

```bash
cd /path/to/ResQ-AI

# Download YOLO weights (trained on AIDER dataset)
# Place in Phase1_SituationalAwareness/
# From: https://github.com/AdityaP9116/ResQ-AI (releases or weights folder)

# Download best.pt and convert to ONNX/TensorRT
cd Phase1_SituationalAwareness
python export_trt.py
cd ..
```

### Step 7: Install Pegasus Simulator Plugin

The Pegasus Simulator is required for drone dynamics.

```bash
# Clone the Pegasus repository
git clone https://github.com/PegasusSimulator/PegasusSimulator.git

# Install it to Isaac Sim extensions directory
ISAAC_SIM_EXT_PATH=~/.local/share/ov/pkg/isaac-sim-5.1/exts
mkdir -p $ISAAC_SIM_EXT_PATH
cp -r PegasusSimulator/pegasus $ISAAC_SIM_EXT_PATH/

# Enable the extension in Isaac Sim
# Menu: Window → Extensions → Search "pegasus" → Enable
```

Or link it for development:

```bash
ln -sf $(pwd)/PegasusSimulator/pegasus ~/.local/share/ov/pkg/isaac-sim-5.1/exts/pegasus
```

---

## Docker Setup

### Option A: Using Provided Dockerfile

We'll create a Docker image with Isaac Sim pre-installed.

#### Step 1: Create Dockerfile

Create this file at `/path/to/ResQ-AI/Dockerfile`:

```dockerfile
# Use NVIDIA CUDA base image compatible with Isaac Sim
FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /workspace

# Copy ResQ-AI repository
COPY . /workspace/resq-ai
WORKDIR /workspace/resq-ai

# Create virtual environment and install Python dependencies
RUN python3.10 -m venv /opt/isaac_env && \
    . /opt/isaac_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
    ultralytics \
    opencv-python \
    numpy \
    scipy \
    requests \
    torch \
    torchvision \
    python-dotenv \
    fastapi \
    uvicorn \
    folium \
    matplotlib

# Install Isaac Sim from Omniverse (requires user interaction)
# For automated setup, use the kit-cli approach or OCI image
RUN mkdir -p /workspace/isaac-sim

# Copy Pegasus Simulator (if available)
COPY PegasusSimulator /workspace/resq-ai/PegasusSimulator

# Set environment variables
ENV PATH="/opt/isaac_env/bin:$PATH" \
    PYTHONPATH="/workspace/resq-ai:$PYTHONPATH" \
    NVIDIA_VISIBLE_DEVICES=all

# Create entrypoint script
RUN echo '#!/bin/bash\n\
. /opt/isaac_env/bin/activate\n\
cd /workspace/resq-ai\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
```

#### Step 2: Build the Docker Image

```bash
cd /path/to/ResQ-AI

docker build \
  -t resq-ai:latest \
  -f Dockerfile \
  --build-arg NVIDIA_VISIBLE_DEVICES=all \
  .
```

#### Step 3: Run Docker Container

```bash
# Interactive shell with GPU support
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace/resq-ai \
  -v ~/.local/share/ov:/root/.local/share/ov \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  resq-ai:latest bash

# Or run a specific script
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace/resq-ai \
  resq-ai:latest \
  python test_isaacsim_minimal.py
```

### Option B: Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  resq-ai:
    image: resq-ai:latest
    build:
      context: .
      dockerfile: Dockerfile
    container_name: resq-ai-sim
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - DISPLAY=${DISPLAY}
      - PYTHONUNBUFFERED=1
    volumes:
      - .:/workspace/resq-ai
      - ~/.local/share/ov:/root/.local/share/ov
      - /tmp/.X11-unix:/tmp/.X11-unix
    ports:
      - "8000:8000"  # VLM server
      - "8001:8001"  # Optional dashboard
    stdin_open: true
    tty: true
    entrypoint: /entrypoint.sh
    command: bash
```

Run with Docker Compose:

```bash
docker-compose up -d resq-ai
docker-compose exec resq-ai bash
```

### Option C: Bare Metal Isaac Sim within Docker

If you already have Isaac Sim installed locally and want to use it in Docker:

```bash
# Mount Isaac Sim into container
docker run --rm -it \
  --gpus all \
  -v /home/ubuntu/.local/share/ov/pkg:/opt/ov/pkg \
  -v $(pwd):/workspace/resq-ai \
  resq-ai:latest bash

# Then set ISAAC_SIM_PATH
export ISAAC_SIM_PATH=/opt/ov/pkg/isaac-sim-5.1
```

---

## Verifying Installation

### Test 1: Isaac Sim Basic Import

```bash
# Local setup
source isaac_env/bin/activate
python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"

# Docker setup
docker run --rm --gpus all resq-ai:latest python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"
```

### Test 2: Full ResQ-AI Import Test

```bash
# Local
source isaac_env/bin/activate
python test_isaacsim_minimal.py

# Docker
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest python test_isaacsim_minimal.py
```

Expected output:
```
[TEST] Step 1: stdlib imports OK
[TEST] Step 2: isaacsim imported OK
...
[TEST] Step 11: orchestrator_bridge OK — ALL IMPORTS PASSED!
[TEST] Done!
```

### Test 3: Run Minimal Isaac Sim Scene

```bash
# Local (with GUI)
source isaac_env/bin/activate
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh sim_bridge/spawn_drone.py

# Local (headless)
source isaac_env/bin/activate
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh sim_bridge/spawn_drone.py --headless

# Docker (headless only)
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  /workspace/isaac-sim/python.sh sim_bridge/spawn_drone.py --headless
```

### Test 4: Run Full Demo Flight

```bash
# Terminal 1: Start VLM server (mock backend, no GPU needed)
source isaac_env/bin/activate
cd orchestrator
python vlm_server.py --backend mock

# Terminal 2: Run simulation
source isaac_env/bin/activate
cd sim_bridge
python demo_flight.py --vlm-url http://localhost:8000/analyze

# Expected output:
# - Flight_Report.json
# - Hazard_Map.html
# - Annotated video (if configured)
```

---

## Troubleshooting

### Issue 1: "isaacsim" Module Not Found

**Cause**: Isaac Sim Python is not in your PATH.

**Solution**:
```bash
# Add to your .env or ~/.bashrc
export ISAAC_SIM_PYTHON=~/.local/share/ov/pkg/isaac-sim-5.1/python.sh
alias isaac-python=$ISAAC_SIM_PYTHON

# Use it
isaac-python -c "from isaacsim import SimulationApp"
```

### Issue 2: GPU Out of Memory

**Cause**: Simulation + YOLO + VLM all running simultaneously.

**Solution**:
```bash
# Run with reduced resolution
python demo_flight.py --resolution 640 360

# Or offload model to CPU for preprocessing
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

### Issue 3: Missing Pegasus Simulator Extension

**Cause**: Extension directory not found.

**Solution**:
```bash
# Verify extension path
ls -la ~/.local/share/ov/pkg/isaac-sim-5.1/exts/

# If missing, manually enable in Isaac Sim:
# 1. Open Isaac Sim GUI
# 2. Window → Extensions
# 3. Search "pegasus"
# 4. Click Settings (gear icon)
# 5. Enable
```

### Issue 4: Docker GPU Access Not Working

**Cause**: NVIDIA Container Runtime not installed.

**Solution**:
```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-runtime

# Verify
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi
```

### Issue 5: Asset Pack Missing or Not Found

**Cause**: Assets downloaded to wrong location.

**Solution**:
```bash
# Verify asset structure
ls -la assets/Architecture/  # Should contain .usd/.usda files
ls -la assets/Characters/    # Should contain character assets
ls -la assets/Particles/     # Should contain fire/smoke effects

# If missing, download from:
# https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
```

### Issue 6: Slow Performance in Docker

**Cause**: X11 forwarding or insufficient GPU allocation.

**Solution**:
```bash
# Run headless (faster)
python sim_bridge/main_sim_loop.py --headless

# Or use hardware acceleration in Docker
docker run --rm \
  --gpus all \
  --device /dev/dri \
  --device /dev/nvidia* \
  resq-ai:latest bash
```

---

## Quick Start Commands

### Local Setup
```bash
# One-time setup
cd /path/to/ResQ-AI
python3.10 -m venv isaac_env
source isaac_env/bin/activate
pip install -r requirements.txt

# Run simulation
source isaac_env/bin/activate
python test_isaacsim_minimal.py
python sim_bridge/demo_flight.py --headless
```

### Docker Setup
```bash
# Build
docker build -t resq-ai:latest .

# Run tests
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  python test_isaacsim_minimal.py

# Run demo
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  python sim_bridge/demo_flight.py --headless
```

---

## Next Steps

1. **Verify Installation**: Run the verification tests above
2. **Download Assets**: Get the Omniverse asset packs
3. **Train/Download Models**: Get YOLO weights
4. **Run Demo**: Execute `demo_flight.py`
5. **Check Output**: Look for `Flight_Report.json` and `Hazard_Map.html`

For more details on the simulation architecture, see [walkthrough.md.resolved](walkthrough.md.resolved).
