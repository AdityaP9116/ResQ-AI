# ResQ-AI Isaac Sim Quick Start Guide

Get a fully functional ResQ-AI simulation environment running via pip — no Omniverse Launcher required.

## Install Isaac Sim via pip (Recommended)

Isaac Sim 4.5.0.0 is distributed as Python wheels on [pypi.nvidia.com](https://pypi.nvidia.com).
The full install is **~4 GB** and only requires `pip` and internet access.

```bash
# Full Isaac Sim installation (one command)
pip install \
  isaacsim==4.5.0.0 \
  isaacsim-core==4.5.0.0 \
  isaacsim-extscache-physics==4.5.0.0 \
  isaacsim-extscache-kit==4.5.0.0 \
  isaacsim-extscache-kit-sdk==4.5.0.0 \
  --extra-index-url https://pypi.nvidia.com

# Accept EULA on first import (cached after first run)
echo "Yes" | python3 -c "import isaacsim"

# Verify
python3 -c "from isaacsim import SimulationApp; print('✓ Isaac Sim ready')"
```

---

## Prerequisites Checklist

- [ ] NVIDIA GPU with drivers 525+ (`nvidia-smi` shows your GPU)
- [ ] Ubuntu 20.04/22.04 LTS
- [ ] Python 3.10
- [ ] ~60 GB free disk space (for packages + sim data)
- [ ] Internet access to [pypi.nvidia.com](https://pypi.nvidia.com)

---

## 🚀 Option 1: Fastest Setup (Automated)

Run the setup script:

```bash
cd /path/to/ResQ-AI

# Make script executable
chmod +x setup_isaac_sim.sh

# Run setup
./setup_isaac_sim.sh
```

This automatically:
- ✓ Checks prerequisites
- ✓ Creates Python virtual environment
- ✓ Installs all dependencies
- ✓ Installs Isaac Sim
- ✓ Configures environment
- ✓ Runs verification tests

**Time: ~15-20 minutes** (depends on internet speed)

---

## 🐳 Option 2: Docker (Easiest)

If you already have Docker installed with GPU support:

```bash
cd /path/to/ResQ-AI

# Build image (one-time)
docker build -t resq-ai:latest .

# Run tests
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  python test_isaacsim_minimal.py

# Interactive shell
docker run --rm -it --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest bash
```

**Time: ~10-15 minutes** (image download + build)

### Docker Compose (Even Easier)

```bash
# Build and start
docker-compose up -d --build

# Enter container
docker-compose exec resq-ai bash

# Run tests
docker-compose exec resq-ai python test_isaacsim_minimal.py

# View logs
docker-compose logs -f resq-ai
```

---

## ✅ Verify Installation

After setup completes, verify everything works:

### Test 1: Basic Import (30 seconds)
```bash
# Use Isaac Sim's Python (not venv!)
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c \
  "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"

# Or create an alias for convenience
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
isaac-python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"
```

### Test 2: Full Integration Test (1-2 minutes)
```bash
# Use Isaac Sim's Python
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
isaac-python test_isaacsim_minimal.py
```

Expected output:
```
[TEST] Step 1: stdlib imports OK
[TEST] Step 2: isaacsim imported OK
...
[TEST] Step 11: orchestrator_bridge OK — ALL IMPORTS PASSED!
[TEST] Done!
```

### Test 3: Spawn Drone (2-3 minutes)
```bash
# Use Isaac Sim's Python
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
isaac-python sim_bridge/spawn_drone.py --headless
```

---

## 📦 Download Assets & Models

### 1. NVIDIA Omniverse Asset Packs (Required)

Download CityEngine buildings, character assets, and effects:

1. Visit: https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/
2. Download these packs:
   - **Architecture** (CityEngine) → extract to `assets/Architecture/`
   - **Characters** (Reallision) → extract to `assets/Characters/`
   - **Particles** (Fire/smoke) → extract to `assets/Particles/`
   - **BaseMaterials** (Textures) → extract to `assets/BaseMaterials/`

### 2. YOLO Model Weights

Download the trained YOLO model:

```bash
# Download best.pt from ResQ-AI releases
# https://github.com/AdityaP9116/ResQ-AI/releases

# Place in Phase 1 directory
cp best.pt Phase1_SituationalAwareness/

# Export to TensorRT (optional, for faster inference)
# Use Isaac Sim's Python
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
cd Phase1_SituationalAwareness
isaac-python export_trt.py
cd ..
```

---

## 🎮 Run Your First Simulation

**Important: Always use Isaac Sim's Python for running scripts:**

```bash
# Create convenient alias
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
```

### Option A: Quick Test (Mock VLM Backend)

```bash
# Terminal 1: Run VLM server (mock doesn't need GPU)
cd orchestrator
isaac-python vlm_server.py --backend mock

# Terminal 2: Run simulation
cd sim_bridge
isaac-python main_sim_loop.py --headless
```

### Option B: Full Demo Flight

```bash
# Terminal 1: VLM server
cd orchestrator
isaac-python vlm_server.py --backend mock

# Terminal 2: Full demo with reporting
cd sim_bridge
isaac-python demo_flight.py --headless --vlm-url http://localhost:8000/analyze
```

Expected output:
- `Flight_Report.json` — Hazard detections with 3D coordinates
- `Hazard_Map.html` — Interactive map of detected hazards
- Annotated video (if enabled)

---

## 📁 File Structure After Setup

```
ResQ-AI/
├── isaac_env/                    # Python virtual environment
├── outputs/                      # Simulation outputs
│   ├── Flight_Report.json
│   ├── Hazard_Map.html
│   └── videos/
├── assets/
│   ├── Architecture/             # Must download
│   ├── Characters/               # Must download
│   ├── Particles/                # Must download
│   └── BaseMaterials/            # Must download
├── Phase1_SituationalAwareness/
│   ├── best.pt                   # Must download
│   ├── best.engine               # Generated by export_trt.py
│   └── ...
├── sim_bridge/                   # Simulation modules
├── orchestrator/                 # AI reasoning pipeline
├── .env                          # Configuration (auto-created)
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── setup_isaac_sim.sh            # Setup script
```

---

## 🔧 Troubleshooting

### ⚠️ "Isaac Sim verification failed" or "path doesn't exist"

**This means Isaac Sim is not installed via Omniverse Launcher.**

✅ **Solution:**
See [ISAAC_SIM_INSTALLATION_FIX.md](ISAAC_SIM_INSTALLATION_FIX.md) for complete instructions.

Quick version:
```bash
# 1. Download launcher
curl -fsSL https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage \
  -o ~/Downloads/omniverse-launcher.AppImage

# 2. Run it
chmod +x ~/Downloads/omniverse-launcher.AppImage
~/Downloads/omniverse-launcher.AppImage

# 3. In launcher: Search "Isaac Sim" → Install version 5.1+
# 4. Wait 15-30 minutes for installation
# 5. Verify: ~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c "from isaacsim import SimulationApp"
```

### "isaacsim module not found"
```bash
# You MUST use Isaac Sim's Python, not a virtual environment!

# ✅ Correct way:
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh test_isaacsim_minimal.py

# ❌ Wrong way (won't work):
source isaac_env/bin/activate
python test_isaacsim_minimal.py
```

### GPU out of memory
```bash
# Run with reduced resolution
isaac-python demo_flight.py --headless --resolution 640 360

# Or specify GPU
export CUDA_VISIBLE_DEVICES=0
isaac-python demo_flight.py --headless
```

### Docker GPU not working
```bash
# Verify nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi

# If not, install:
sudo apt-get install nvidia-container-runtime
sudo systemctl restart docker
```

### Assets not found
```bash
# Verify asset structure
ls -la assets/Architecture/  # Should contain .usd files
ls -la assets/Characters/    # Should contain .usd files

# If empty, download from:
# https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/
```

---

## 📚 Next Steps

1. **Read the Architecture Guide**: See [walkthrough.md.resolved](walkthrough.md.resolved)
2. **Understand the Phases**:
   - **Phase 1**: YOLO detection (completed)
   - **Phase 2**: Semantic segmentation (in progress)
   - **Phase 3**: Cosmos Reason 2 VLM integration (done, use `--backend mock` or `--backend nim`)
3. **Experiment**: Modify scene generation, adjust simulation parameters
4. **Train Custom Models**: Use Colab notebooks in each Phase directory
5. **Deploy**: Package as Docker container for production use

---

## 💡 Tips & Tricks

### Faster First Run
Use `--backend mock` for VLM to avoid downloading heavy models:
```bash
python vlm_server.py --backend mock
```

### Optimize Performance
```python
# In your simulation scripts:
PHYSICS_STEPS = 60  # Lower = faster
SIMULATION_RESOLUTION = (640, 480)  # Reduce if slow
HEADLESS_MODE = True  # Always use for speed
```

### Batch Processing
```bash
# Run multiple simulations
for i in {1..5}; do
  echo "Run $i"
  python demo_flight.py --headless --output-dir outputs/run_$i
done
```

### Monitor Resources
```bash
# In another terminal, monitor GPU/CPU/RAM
watch -n 1 nvidia-smi
# or
btop  # if installed
```

---

## 🆘 Getting Help

- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/
- **ResQ-AI GitHub**: https://github.com/AdityaP9116/ResQ-AI
- **NVIDIA Forums**: https://forums.developer.nvidia.com/c/omniverse/isaac/

---

## ✨ Success!

You're now ready to:
- ✓ Simulate autonomous drone flights
- ✓ Test hazard detection in urban environments
- ✓ Integrate AI reasoning for navigation
- ✓ Generate hazard maps and reports

Happy simulating! 🚁
