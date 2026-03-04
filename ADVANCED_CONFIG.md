# Advanced Isaac Sim Configuration Guide

Comprehensive troubleshooting, optimization, and advanced configuration for ResQ-AI with Isaac Sim.

## Table of Contents
1. [GPU Memory Optimization](#gpu-memory-optimization)
2. [Performance Tuning](#performance-tuning)
3. [Advanced Docker Setup](#advanced-docker-setup)
4. [Isaac Sim Extensions](#isaac-sim-extensions)
5. [VLM Backend Configuration](#vlm-backend-configuration)
6. [Debugging & Logging](#debugging--logging)
7. [Common Issues & Solutions](#common-issues--solutions)

---

## GPU Memory Optimization

### Check GPU Memory Usage

```bash
# Monitor in real-time
nvidia-smi -l 1

# Get detailed utilization
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,nounits

# Python script to check memory in code
python << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
EOF
```

### Reduce Memory Usage

#### A. Lower Simulation Resolution

```python
# In your simulation script (e.g., spawn_drone.py)
# Modify camera resolution:

CAMERA_RESOLUTION = (480, 360)  # Instead of (640, 480)
SEMANTIC_RES = (480, 360)
DEPTH_RES = (480, 360)
```

#### B. Batch Processing with Memory Cleanup

```python
# In orchestrator_bridge.py
import torch

# Process frames and clean up
for frame_idx in range(num_frames):
    rgb = get_frame()
    result = model(rgb)
    
    # Clear GPU cache
    if frame_idx % 10 == 0:
        torch.cuda.empty_cache()
```

#### C. Model Quantization

```python
# Use quantized YOLO for lower memory
from ultralytics import YOLO

# Less memory than full precision
model = YOLO('best.pt')
model_int8 = model.export(format='onnx', half=True)  # FP16 precision
```

---

## Performance Tuning

### Simulation Speed

```bash
# Measure FPS
python << 'EOF'
import time
from sim_bridge.main_sim_loop import main

start = time.time()
main(num_steps=1000)
elapsed = time.time() - start
fps = 1000 / elapsed
print(f"Average FPS: {fps:.2f}")
EOF
```

### Optimize Physics Engine

```python
# In your simulation initialization
from omni.isaac.core.world import World

world = World()

# Adjust physics timestep (default 1/60)
world.set_physics_dt(1/60)  # Slower = more accurate but slower
world.set_rendering_dt(1/30)  # Rendering frequency

# Reduce physics substeps for speed
world.set_physics_prim_path("/physicsScene")
```

### Parallel Processing

```python
# Use ThreadPoolExecutor for inference
from concurrent.futures import ThreadPoolExecutor
import numpy as np

executor = ThreadPoolExecutor(max_workers=2)

# Inference in parallel
def process_frames_parallel(frames):
    futures = [executor.submit(model, f) for f in frames]
    results = [f.result() for f in futures]
    return results
```

---

## Advanced Docker Setup

### Build with Custom CUDA Version

```dockerfile
# Dockerfile (custom CUDA)
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# Rest of Dockerfile...
```

### Multi-Stage Build (Smaller Image)

```dockerfile
# Stage 1: Builder
FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y python3.10 python3.10-venv
RUN python3.10 -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime (smaller)
FROM nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04

COPY --from=builder /opt/venv /opt/venv
COPY . /workspace/resq-ai

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace/resq-ai
CMD ["/bin/bash"]
```

### Docker with NVIDIA Container Runtime

```bash
# Verify nvidia-docker runtime
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi

# Run container with GPU
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v $(pwd):/workspace/resq-ai \
  resq-ai:latest bash
```

### Push to Docker Registry

```bash
# Tag image
docker tag resq-ai:latest your-registry/resq-ai:v1.0

# Login and push
docker login your-registry
docker push your-registry/resq-ai:v1.0

# Pull and run
docker pull your-registry/resq-ai:v1.0
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai your-registry/resq-ai:v1.0
```

---

## Isaac Sim Extensions

### Install Custom Extensions

```bash
# List installed extensions
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c "from omni.kit.extension_manager import get_extensions; print(get_extensions())"

# Install extension from folder
# 1. Copy extension to: ~/.local/share/ov/pkg/isaac-sim-5.1/exts/
# 2. Enable in Isaac Sim GUI: Window → Extensions → Search → Enable

# Or programmatically enable
python << 'EOF'
from omni.kit.extension_manager import ext_manager
ext_manager().set_extension_enabled_immediate("namespace.extension_name", True)
EOF
```

### Pegasus Simulator Advanced Configuration

```python
# Advanced drone configuration
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Create interface with custom settings
interface = PegasusInterface()

# Configure drone physics
drone = interface.spawn_multirotor(
    spawn_translation=(0, 0, 1),
    spawn_orientation=(0, 0, 0, 1),  # wxyz quaternion
    controller_type="velocity",  # or "attitude"
    physics_dt=0.0016,  # 1/625 Hz
    max_rotation_velocity=10.0,
    max_linear_velocity=10.0,
)

# Configure sensors
drone.attach_lidar(sensor_type="2d_gpu", range=100.0)
drone.attach_camera(camera_type="rgb")
```

---

## VLM Backend Configuration

### 1. Mock Backend (No GPU)

```bash
# Fast, deterministic responses
python vlm_server.py --backend mock
```

### 2. NVIDIA NIM Cloud Backend

Requires NVIDIA API key from: https://build.nvidia.com/

```bash
# Setup
export NVIDIA_API_KEY="your-api-key-here"

# Run
python vlm_server.py --backend nim

# In Python
import requests
import json

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "image_base64": "...",
        "context": {
            "hazards": ["Collapsed Building"],
            "drone_position": [0, 0, 10]
        }
    }
)
result = response.json()
print(result["waypoint"])
print(result["reasoning"])
```

### 3. Local vLLM Backend (Highest Performance)

```bash
# Install vLLM
pip install vllm

# Download Cosmos Reason 2 model
huggingface-cli login
huggingface-cli download nvidia/Cosmos-1.0-Reasoning-7.5B

# Run VLM server with vLLM
python vlm_server.py --backend vllm \
  --model-id nvidia/Cosmos-1.0-Reasoning-7.5B \
  --tensor-parallel-size 2  # Use 2 GPUs if available
```

---

## Debugging & Logging

### Enable Verbose Logging

```python
# In your simulation script
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.debug("Starting simulation...")
```

### Isaac Sim Debug Mode

```bash
# Run with debug info
CARB_APP_DEBUG=1 \
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh sim_bridge/main_sim_loop.py

# Enable USD debugging
export OMNI_USD_DEBUG=1
```

### Save Detailed Logs

```python
# In orchestrator_bridge.py
import json
from datetime import datetime

def log_frame_data(frame_idx, detections, hazards, waypoint):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "frame_idx": frame_idx,
        "detections": detections,
        "active_hazards": hazards,
        "waypoint": waypoint,
    }
    
    with open(f"outputs/logs/frame_{frame_idx:06d}.json", "w") as f:
        json.dump(log_entry, f, indent=2)
```

### Profile Code Performance

```python
# Using cProfile
import cProfile
import pstats

pr = cProfile.Profile()
pr.enable()

# Your code here
simulation_main_loop()

pr.disable()
ps = pstats.Stats(pr)
ps.sort_stats('cumulative')
ps.print_stats(10)  # Top 10 functions
```

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# 1. Reduce batch size
python demo_flight.py --batch-size 1

# 2. Lower resolution
python demo_flight.py --resolution 480 360

# 3. Use smaller model
python demo_flight.py --yolo-model nano  # Instead of 'small'

# 4. Run in headless mode
python demo_flight.py --headless  # No GUI rendering

# 5. Clear GPU cache periodically
# Add to orchestrator_bridge.py:
import torch
torch.cuda.empty_cache()  # Call periodically
```

### Issue 2: Slow Simulation (< 5 FPS)

**Cause**: Physics calculations too expensive or model inference slow

**Solutions**:
```python
# 1. Reduce physics substeps in main_sim_loop.py
world.set_physics_dt(1/30)  # Lower timestep

# 2. Skip frames for inference
INFERENCE_EVERY_N_FRAMES = 3
if frame_idx % INFERENCE_EVERY_N_FRAMES == 0:
    detections = model(frame)

# 3. Use TensorRT for faster YOLO
# First export: python Phase1_SituationalAwareness/export_trt.py
# Then load: model = YOLO("best.engine")

# 4. Reduce number of simulation objects
num_buildings = 10  # Instead of 50
```

### Issue 3: VLM Server Connection Refused

**Error**: `Connection refused` when calling `http://localhost:8000/analyze`

**Solutions**:
```bash
# 1. Check if server is running
curl http://localhost:8000/docs

# 2. Wait for server to start
sleep 5
python sim_bridge/demo_flight.py

# 3. Check port binding
netstat -tulpn | grep 8000

# 4. Use different port
python vlm_server.py --host 0.0.0.0 --port 8001
# Then in demo_flight.py:
# --vlm-url http://localhost:8001/analyze
```

### Issue 4: Pegasus Simulator Not Found

**Error**: `ModuleNotFoundError: No module named 'pegasus.simulator'`

**Solutions**:
```bash
# 1. Check if extension is enabled
ls ~/.local/share/ov/pkg/isaac-sim-5.1/exts/ | grep pegasus

# 2. Clone and link Pegasus
git clone https://github.com/PegasusSimulator/PegasusSimulator.git
ln -sf $(pwd)/PegasusSimulator/pegasus \
  ~/.local/share/ov/pkg/isaac-sim-5.1/exts/pegasus

# 3. Verify importable
python -c "from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface; print('OK')"

# 4. Add to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$HOME/.local/share/ov/pkg/isaac-sim-5.1/exts"
```

### Issue 5: Asset Packs Not Loading

**Error**: Asset paths not found or wrong format

**Solutions**:
```bash
# 1. Verify asset structure
ls -la assets/Architecture/  # Should have .usd files
find assets -name "*.usd" -o -name "*.usda" | head -10

# 2. Check file permissions
chmod -R 755 assets/

# 3. Verify paths in generate_urban_scene.py
# Edit: ASSET_ROOT_PATH in the script

# 4. Use absolute paths
import os
ASSET_PATH = os.path.abspath("assets/Architecture/")
print(f"Loading from: {ASSET_PATH}")
```

### Issue 6: Docker GPU Not Accessible

**Error**: `could not select device driver "" with capabilities`

**Solutions**:
```bash
# 1. Install nvidia-container-runtime
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install nvidia-container-runtime

# 2. Restart Docker
sudo systemctl restart docker

# 3. Test
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi

# 4. Run ResQ-AI with GPU
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  python test_isaacsim_minimal.py
```

---

## Performance Benchmarks

Expected performance on different GPUs:

| GPU | YOLO FPS | Physics FPS | Memory Used |
|-----|----------|------------|-------------|
| RTX 3090 | 120+ | 60+ | ~8GB |
| RTX 4090 | 200+ | 100+ | ~6GB |
| A100 | 300+ | 120+ | ~12GB |
| RTX 3080 | 80-100 | 40-50 | ~9GB |

---

## Optimization Checklist

- [ ] Use headless mode for 2-3x speedup
- [ ] Export YOLO to TensorRT for inference speedup
- [ ] Reduce camera resolution to 480x360
- [ ] Use `--backend mock` for VLM if not testing reasoning
- [ ] Enable GPU memory optimization in PyTorch
- [ ] Profile your code and identify bottlenecks
- [ ] Use Docker for reproducible environments
- [ ] Monitor GPU usage with `nvidia-smi`
- [ ] Consider quantized models (INT8/FP16)
- [ ] Cache pre-computed assets

---

## Advanced Scripting

### Run Multiple Simulations in Parallel

```python
# parallel_sims.py
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

def run_sim(sim_id, output_dir):
    cmd = [
        "python", "sim_bridge/demo_flight.py",
        "--headless",
        "--output-dir", f"{output_dir}/sim_{sim_id}",
        "--seed", str(sim_id),
    ]
    subprocess.run(cmd)

# Run 4 simulations in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    for i in range(4):
        executor.submit(run_sim, i, "outputs")
```

### Automated Testing

```bash
#!/bin/bash
# run_tests.sh

set -e

echo "Running Isaac Sim Tests..."

# Test 1: Import test
python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"

# Test 2: Full integration
python test_isaacsim_minimal.py

# Test 3: VLM server
python vlm_server.py --backend mock &
VLM_PID=$!
sleep 3

# Test 4: Demo flight
python sim_bridge/demo_flight.py --headless --max-steps 100

# Cleanup
kill $VLM_PID

echo "✓ All tests passed!"
```

---

## Next Steps

- Explore the [Architecture Guide](walkthrough.md.resolved)
- Customize simulation parameters in `generate_urban_scene.py`
- Train your own YOLO model on custom datasets
- Integrate real sensor data from an actual drone

Good luck with your advanced Isaac Sim setup!
