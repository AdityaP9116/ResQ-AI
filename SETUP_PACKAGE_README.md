# Isaac Sim Setup Package Summary

Complete setup documentation and automation for ResQ-AI with NVIDIA Isaac Sim.

## 📦 Files Included

### 📖 Documentation

#### 1. **ISAAC_SIM_SETUP.md** (Primary Setup Guide)
Complete instructions for setting up Isaac Sim both locally and in Docker.

**Covers:**
- System requirements and hardware prerequisites
- Local installation via Omniverse Launcher
- Step-by-step dependency installation
- Environment configuration
- Docker setup (multiple approaches)
- Verification tests
- Troubleshooting common issues

**When to use:** First-time setup, reference for installation steps

---

#### 2. **QUICKSTART.md** (Getting Started Fast)
Quick reference guide to get up and running in 5 minutes.

**Covers:**
- Prerequisites checklist
- Three fastest setup options (Automated, Docker, Docker Compose)
- Verification commands
- Asset and model downloads
- First simulation run
- File structure overview
- Common troubleshooting

**When to use:** Quick setup reference, fast problem solving

---

#### 3. **ADVANCED_CONFIG.md** (Advanced Users)
Optimization, performance tuning, and advanced configurations.

**Covers:**
- GPU memory optimization
- Performance benchmarking and tuning
- Advanced Docker setups (multi-stage builds, registries)
- Isaac Sim extensions
- VLM backend configuration (mock, NIM, vLLM)
- Debugging and logging
- Detailed issue solutions
- Performance benchmarks
- Parallel simulation running

**When to use:** Optimization, performance debugging, advanced customization

---

### 🚀 Automation Scripts

#### 4. **setup_isaac_sim.sh** (Automated Setup - Linux)
Complete automated setup script that handles all installation steps.

**Features:**
- Checks system prerequisites
- Creates Python virtual environment
- Installs all dependencies
- Installs Isaac Sim
- Configures environment
- Runs verification tests
- Generates .env file

**Usage:**
```bash
chmod +x setup_isaac_sim.sh
./setup_isaac_sim.sh [--headless] [--docker] [--skip-download]
```

**Time:** ~15-20 minutes

**When to use:** First-time local setup

---

### 🐳 Docker Configuration

#### 5. **Dockerfile** (Docker Image Definition)
Production-ready Dockerfile with optimized multi-stage build.

**Includes:**
- CUDA 12.2 base image
- Python 3.10 and virtual environment
- All ResQ-AI dependencies pre-installed
- Isaac Sim SDK
- Proper entry point configuration
- GPU support ready
- Volume mount points configured

**Usage:**
```bash
docker build -t resq-ai:latest .
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest bash
```

**Size:** ~5-6 GB (optimized)

---

#### 6. **docker-compose.yml** (Docker Orchestration)
Docker Compose configuration for easy container management.

**Features:**
- Automatic GPU allocation
- Volume mounting (code, outputs, cache)
- Port mapping (VLM server on 8000)
- Health checks
- Environment variable management
- Optional VLM service profile
- Named volumes for caching

**Usage:**
```bash
docker-compose up -d --build
docker-compose exec resq-ai bash
docker-compose down
```

**When to use:** Multi-service setup, persistent development

---

#### 7. **.dockerignore** (Docker Build Optimization)
Specifies files to exclude from Docker build context.

**Excludes:**
- Large model files (*.pt, *.engine, *.onnx)
- Generated datasets and outputs
- Git and IDE files
- Redundant documentation

**Effect:** Faster builds, smaller context size

---

### 📦 Dependency Management

#### 8. **requirements.txt** (Python Dependencies)
Complete list of Python package dependencies with versions.

**Includes:**
- Core ML/CV stack (PyTorch, YOLOv8, OpenCV)
- Scientific computing (NumPy, SciPy)
- Web frameworks (FastAPI, Flask)
- Visualization (Folium, Matplotlib)
- Development tools (Pytest, IPython)
- Optional: LLM/VLM support, Isaac Sim SDK

**Usage:**
```bash
pip install -r requirements.txt
```

**Alternative for Docker:**
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

---

## 🎯 Quick Navigation

**I want to:**

### ✅ Get started quickly
→ **QUICKSTART.md** + **setup_isaac_sim.sh**

### 🐳 Use Docker
→ **QUICKSTART.md** (Option 2) + **Dockerfile** + **docker-compose.yml**

### 🔧 Optimize performance
→ **ADVANCED_CONFIG.md** (Performance Tuning section)

### 🐛 Debug an issue
→ **ADVANCED_CONFIG.md** (Common Issues section) + **ISAAC_SIM_SETUP.md** (Troubleshooting)

### 📚 Learn the architecture
→ **ISAAC_SIM_SETUP.md** (System Requirements) + **walkthrough.md.resolved**

### 🎓 Advanced customization
→ **ADVANCED_CONFIG.md** (all sections)

### 👥 Deploy for team
→ **docker-compose.yml** + push to registry (see ADVANCED_CONFIG.md)

---

## 🚀 Three Setup Paths

### Path 1: Automated Local Setup (Recommended for Beginners)
```
1. Clone ResQ-AI
2. Run: ./setup_isaac_sim.sh
3. Done! (15-20 min)
```
- ✅ Automatic checking
- ✅ One command
- ❌ Takes longer
- ❌ Requires more disk space upfront

### Path 2: Docker Compose (Easiest)
```
1. Clone ResQ-AI
2. Ensure nvidia-docker installed
3. Run: docker-compose up
4. Done! (10-15 min)
```
- ✅ Reproducible across machines
- ✅ Isolated environment
- ✅ Easy to share
- ❌ Requires Docker/nvidia-docker

### Path 3: Manual Local Setup (Most Control)
```
1. Clone ResQ-AI
2. Follow ISAAC_SIM_SETUP.md step-by-step
3. Manually verify each step
```
- ✅ Full control
- ✅ Understand each step
- ❌ Time-consuming
- ❌ Error-prone

---

## 📋 Setup Checklist

- [ ] Read QUICKSTART.md
- [ ] Verify GPU and prerequisites
- [ ] Choose setup path (automated/Docker/manual)
- [ ] Run setup script or Docker build
- [ ] Verify installation with test scripts
- [ ] Download Omniverse asset packs
- [ ] Download YOLO model weights
- [ ] Run first simulation
- [ ] Check output files (Flight_Report.json, Hazard_Map.html)
- [ ] Read ADVANCED_CONFIG.md for optimization

---

## 🎓 Learning Resources

### For Setup & Configuration
1. Start with **QUICKSTART.md** for overview
2. Use **ISAAC_SIM_SETUP.md** for detailed steps
3. Refer to **ADVANCED_CONFIG.md** for issues

### For Understanding ResQ-AI
1. Read **walkthrough.md.resolved** (architecture overview)
2. Examine **README.md** (project structure)
3. Study source code in `sim_bridge/` and `orchestrator/`

### For Isaac Sim
1. Official docs: https://docs.omniverse.nvidia.com/isaacsim/
2. Pegasus Simulator: https://github.com/PegasusSimulator/PegasusSimulator
3. NVIDIA Forums: https://forums.developer.nvidia.com/

---

## 💻 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 | RTX 4090 / A100 |
| VRAM | 8 GB | 16+ GB |
| RAM | 16 GB | 32+ GB |
| Storage | 100 GB | 200+ GB |
| OS | Ubuntu 20.04 | Ubuntu 22.04 LTS |
| Driver | 540+ | 550+ |
| Python | 3.9 | 3.10, 3.11 |

---

## 🔄 File Relationships

```
QUICKSTART.md (START HERE)
    ↓
setup_isaac_sim.sh (Automation)
    OR
(Dockerfile + docker-compose.yml) (Docker path)
    OR
ISAAC_SIM_SETUP.md (Manual path)
    ↓
requirements.txt (Dependencies)
    ↓
test_isaacsim_minimal.py (Verify)
    ↓
sim_bridge/demo_flight.py (Run simulation)
    ↓
outputs/ (Results)

ADVANCED_CONFIG.md (Optimization & Debugging)
walkthrough.md.resolved (Architecture)
README.md (Project overview)
```

---

## 📞 Support

If you encounter issues:

1. **Check QUICKSTART.md** for quick solutions
2. **Check ADVANCED_CONFIG.md** for detailed troubleshooting
3. **Read ISAAC_SIM_SETUP.md** for prerequisites
4. **Search GitHub issues**: https://github.com/AdityaP9116/ResQ-AI/issues
5. **NVIDIA Forums**: https://forums.developer.nvidia.com/c/omniverse/isaac/
6. **Isaac Sim Docs**: https://docs.omniverse.nvidia.com/isaacsim/

---

## ✨ Success Indicators

You'll know everything is working when:

- ✅ `test_isaacsim_minimal.py` runs without errors
- ✅ `python sim_bridge/spawn_drone.py --headless` completes successfully
- ✅ `python sim_bridge/demo_flight.py --headless` generates output files
- ✅ `Flight_Report.json` and `Hazard_Map.html` are created
- ✅ GPU memory usage is reasonable (~4-8 GB)
- ✅ Simulation runs at 30+ FPS

---

## 🎯 Next Steps After Setup

1. **Understand the system**: Read walkthrough.md.resolved
2. **Modify simulation**: Edit sim_bridge/generate_urban_scene.py
3. **Train custom models**: Use Phase1_SituationalAwareness/ notebooks
4. **Integrate real data**: Connect to actual drone or camera feeds
5. **Deploy**: Package as Docker image for production
6. **Optimize**: Use ADVANCED_CONFIG.md tips for your hardware

---

**Happy Simulating! 🚁**

For questions, refer to the appropriate guide file above.
