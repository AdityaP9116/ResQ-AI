# Isaac Sim Setup Checklist

Complete checklist for setting up Isaac Sim with ResQ-AI. Print this or check off items as you complete them.

## 📋 Pre-Setup Planning

- [ ] **Choose your setup method:**
  - [ ] Automated script (fastest, recommended)
  - [ ] Docker Compose (reproducible)
  - [ ] Manual local setup (most control)
  - [ ] Docker single image (isolated)

- [ ] **Verify your system:**
  - [ ] Linux OS (Ubuntu 20.04 or 22.04)
  - [ ] NVIDIA GPU with 8+ GB VRAM
  - [ ] 100+ GB free disk space
  - [ ] NVIDIA drivers 550+ installed
  - [ ] Internet connection available

---

## 🔧 Phase 1: System Prerequisites

### Hardware Check
- [ ] Run `nvidia-smi` and verify GPU appears
- [ ] Check free disk space: `df -h /`
- [ ] Verify disk has 100+ GB available
- [ ] Check RAM: `free -h`
- [ ] Confirm at least 16 GB RAM available

### Software Check
- [ ] Verify NVIDIA driver: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
- [ ] Check driver version is 550+
- [ ] Verify Python 3.10: `python3.10 --version`
- [ ] Check if git is installed: `git --version`
- [ ] Verify internet connectivity: `ping nvidia.com`

### For Docker Setup Only
- [ ] Install Docker: `docker --version`
- [ ] Install NVIDIA Container Runtime
  - [ ] Verify: `docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi`

---

## 🚀 Phase 2: Setup Installation

### Option A: Automated Setup Script (Recommended)

- [ ] Navigate to ResQ-AI directory: `cd /path/to/ResQ-AI`
- [ ] Make script executable: `chmod +x setup_isaac_sim.sh`
- [ ] Run setup: `./setup_isaac_sim.sh`
  - [ ] Let it check prerequisites
  - [ ] Create virtual environment
  - [ ] Install dependencies
  - [ ] Install Isaac Sim
  - [ ] Generate .env file
  - [ ] Run verification tests
- [ ] Wait for completion (15-20 minutes)
- [ ] Note the success message and next steps

---

### Option B: Docker Compose Setup

- [ ] Navigate to ResQ-AI directory: `cd /path/to/ResQ-AI`
- [ ] Build Docker image: `docker build -t resq-ai:latest .`
  - [ ] Wait for build to complete (10-15 minutes)
  - [ ] Check for successful "Successfully tagged" message
- [ ] Verify image created: `docker images | grep resq-ai`
- [ ] Start container: `docker-compose up -d --build`
- [ ] Verify container running: `docker ps | grep resq-ai`

---

### Option C: Manual Local Setup

- [ ] Navigate to ResQ-AI directory: `cd /path/to/ResQ-AI`

#### Create Virtual Environment
- [ ] Create venv: `python3.10 -m venv isaac_env`
- [ ] Activate: `source isaac_env/bin/activate`
- [ ] Verify activation (prompt should show `(isaac_env)`)

#### Install Dependencies
- [ ] Upgrade pip: `pip install --upgrade pip setuptools wheel`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Wait for all packages to install (may take 5-10 minutes)

#### Install Isaac Sim
- [ ] Install via pip: `pip install isaacsim`
  - [ ] Or: Download from Omniverse Launcher and install manually
  - [ ] Or: Use pre-installed `~/.local/share/ov/pkg/isaac-sim-5.1`

#### Setup Configuration
- [ ] Copy .env template: `cp .env.example .env`
- [ ] Edit .env with your paths: `vi .env`
  - [ ] Set ISAAC_SIM_PATH
  - [ ] Set OUTPUT_DIR
  - [ ] Set VLM_BACKEND (default: mock)

---

## ✅ Phase 3: Verification

### Test 1: Basic Isaac Sim Import (30 seconds)

```bash
# Activate environment (if using local setup)
source isaac_env/bin/activate

# Run Python test
python -c "from isaacsim import SimulationApp; print('✓ Isaac Sim OK')"
```

- [ ] Command completes without errors
- [ ] See "✓ Isaac Sim OK" message

### Test 2: Full Integration Test (1-2 minutes)

```bash
# Activate environment (if using local setup)
source isaac_env/bin/activate

# Run comprehensive test
python test_isaacsim_minimal.py
```

- [ ] Script runs to completion
- [ ] See multiple "[TEST] Step X: ... OK" messages
- [ ] Final message: "[TEST] Step 11: orchestrator_bridge OK — ALL IMPORTS PASSED!"

### Test 3: Docker Verification (if using Docker)

```bash
# Run test in Docker container
docker run --rm --gpus all -v $(pwd):/workspace/resq-ai resq-ai:latest \
  python test_isaacsim_minimal.py
```

- [ ] Docker container starts successfully
- [ ] Test completes without GPU errors
- [ ] All imports pass

---

## 📦 Phase 4: Download Assets

### Omniverse Asset Packs (Required)

- [ ] Visit: https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/
- [ ] Download **Architecture** pack
  - [ ] Download ZIP file
  - [ ] Extract to `assets/Architecture/`
  - [ ] Verify files exist: `ls assets/Architecture/ | head`
  
- [ ] Download **Characters** pack
  - [ ] Download ZIP file
  - [ ] Extract to `assets/Characters/`
  - [ ] Verify files exist: `ls assets/Characters/ | head`

- [ ] Download **Particles** pack
  - [ ] Download ZIP file
  - [ ] Extract to `assets/Particles/`
  - [ ] Verify files exist: `ls assets/Particles/ | head`

- [ ] Download **BaseMaterials** pack (optional but recommended)
  - [ ] Download ZIP file
  - [ ] Extract to `assets/BaseMaterials/`
  - [ ] Verify files exist: `ls assets/BaseMaterials/ | head`

---

### YOLO Model Weights (Optional but Recommended)

- [ ] Download `best.pt` from ResQ-AI GitHub releases
  - [ ] URL: https://github.com/AdityaP9116/ResQ-AI/releases
  - [ ] Or train your own using Colab notebooks
  
- [ ] Place in Phase 1 directory:
  - [ ] `cp best.pt Phase1_SituationalAwareness/`

- [ ] Verify file exists:
  - [ ] `ls -lh Phase1_SituationalAwareness/best.pt`

- [ ] (Optional) Export to TensorRT for faster inference:
  ```bash
  cd Phase1_SituationalAwareness
  python export_trt.py
  cd ..
  ```
  - [ ] Check for `best.engine` file created
  - [ ] Verify file size is reasonable

---

## 🎮 Phase 5: First Simulation

### Spawn Drone Test (2-3 minutes)

```bash
# Activate environment
source isaac_env/bin/activate

# Run drone spawner
python sim_bridge/spawn_drone.py --headless
```

- [ ] Script runs without errors
- [ ] No GPU out of memory errors
- [ ] Script completes successfully
- [ ] Check terminal for success message

### Main Simulation Loop (3-5 minutes)

```bash
# Terminal 1: Start VLM server
source isaac_env/bin/activate
cd orchestrator
python vlm_server.py --backend mock

# Wait a moment, then...

# Terminal 2: Run simulation
source isaac_env/bin/activate
cd sim_bridge
python main_sim_loop.py --headless
```

- [ ] VLM server starts on port 8000
- [ ] Simulation loop starts without errors
- [ ] Simulation runs for 100+ steps
- [ ] Script completes successfully

### Full Demo Flight (5-10 minutes)

```bash
# Terminal 1: Start VLM server
source isaac_env/bin/activate
cd orchestrator
python vlm_server.py --backend mock

# Wait a moment, then...

# Terminal 2: Run full demo
source isaac_env/bin/activate
cd sim_bridge
python demo_flight.py --headless --vlm-url http://localhost:8000/analyze
```

- [ ] Both services start without errors
- [ ] Simulation runs for multiple minutes
- [ ] No GPU memory errors
- [ ] VLM server responds to requests
- [ ] Script completes successfully

---

## 📊 Phase 6: Verify Output

After running `demo_flight.py`, check for these output files:

- [ ] `Flight_Report.json` exists
  - [ ] Size: > 1 KB
  - [ ] Contains hazard detections
  - [ ] Contains drone positions
  - [ ] Contains VLM analysis

- [ ] `Hazard_Map.html` exists (optional)
  - [ ] Can open in browser
  - [ ] Shows map with hazard markers
  - [ ] Markers have popups with analysis

- [ ] Output directory structure:
  ```
  outputs/
  ├── Flight_Report.json
  ├── Hazard_Map.html
  ├── videos/
  │   └── (annotated video if enabled)
  └── logs/
      └── (simulation logs)
  ```

- [ ] Verify files:
  ```bash
  ls -lh outputs/
  cat outputs/Flight_Report.json | head -20
  ```

---

## 🔍 Phase 7: Performance Check

### Monitor GPU Usage

- [ ] Open new terminal
- [ ] Run: `nvidia-smi -l 1` (updates every second)
- [ ] During simulation, check:
  - [ ] GPU memory usage (should be < 90% of total)
  - [ ] GPU utilization (should be 70-95%)
  - [ ] Temperature (should be < 85°C)

### Measure FPS

- [ ] Check simulation output for FPS metrics
- [ ] Expected performance:
  - [ ] RTX 3080+: 30+ FPS
  - [ ] RTX 4090: 50+ FPS
  - [ ] If < 20 FPS, see ADVANCED_CONFIG.md for optimization

### Check Memory Usage

- [ ] Run: `free -h` to check total RAM
- [ ] During simulation:
  - [ ] CPU memory usage should be < 50% of total RAM
  - [ ] Avoid using swap (check: `cat /proc/swaps`)

---

## 🛡️ Phase 8: Backup & Documentation

- [ ] Create snapshot of working state:
  ```bash
  cd /path/to/ResQ-AI
  git status  # Note any uncommitted changes
  git diff > my_changes.patch  # Save custom changes
  ```

- [ ] Document your configuration:
  ```bash
  cp .env .env.backup
  nvidia-smi > gpu_info.txt
  python --version > python_version.txt
  ```

- [ ] Save important files:
  - [ ] `.env` (environment variables)
  - [ ] `best.pt` (YOLO weights)
  - [ ] `Hazard_Map.html` (example output)

---

## 📚 Phase 9: Understanding the System

- [ ] Read documentation:
  - [ ] `QUICKSTART.md` (overview)
  - [ ] `walkthrough.md.resolved` (architecture)
  - [ ] `ADVANCED_CONFIG.md` (optimization)

- [ ] Explore source code:
  - [ ] `sim_bridge/main_sim_loop.py` (simulation loop structure)
  - [ ] `orchestrator/orchestrator_bridge.py` (AI pipeline)
  - [ ] `sim_bridge/generate_urban_scene.py` (scene generation)

- [ ] Understand the data flow:
  - [ ] How camera frames are captured
  - [ ] How YOLO detects hazards
  - [ ] How 3D projection works
  - [ ] How VLM makes decisions

---

## 🎯 Phase 10: Advanced Setup (Optional)

- [ ] Performance optimization:
  - [ ] Follow ADVANCED_CONFIG.md optimization tips
  - [ ] Measure FPS improvements
  - [ ] Document changes made

- [ ] Custom modifications:
  - [ ] Modify `generate_urban_scene.py` to create custom scenes
  - [ ] Adjust simulation parameters
  - [ ] Add custom sensors or cameras

- [ ] Model training (optional):
  - [ ] Review Phase1 Colab notebooks
  - [ ] Train custom YOLO model
  - [ ] Export and integrate model

- [ ] VLM backend selection:
  - [ ] Try different backends (mock, nim, vllm)
  - [ ] Compare response quality and speed
  - [ ] Choose best for your use case

---

## 🚀 Final Checklist

- [ ] All tests passing
- [ ] Asset packs downloaded and in correct locations
- [ ] YOLO model weights available
- [ ] First simulation ran successfully
- [ ] Output files generated correctly
- [ ] GPU performance acceptable
- [ ] Environment variables configured
- [ ] Documentation reviewed
- [ ] System ready for use

---

## 📞 Troubleshooting Quick Links

If any step fails, check these in order:

1. **Import errors** → ISAAC_SIM_SETUP.md (Troubleshooting section)
2. **GPU errors** → ADVANCED_CONFIG.md (GPU Memory optimization)
3. **Missing files** → QUICKSTART.md (Download Assets section)
4. **Slow performance** → ADVANCED_CONFIG.md (Performance Tuning)
5. **Docker issues** → ADVANCED_CONFIG.md (Advanced Docker Setup)

---

## 🎉 Success Criteria

You're ready to use ResQ-AI when:

✅ All 3 tests pass (Test 1, 2, 3)
✅ Asset packs are present in `assets/` directories
✅ Demo flight completes successfully
✅ `Flight_Report.json` and `Hazard_Map.html` are generated
✅ GPU memory usage is reasonable (< 90%)
✅ Simulation runs at 30+ FPS
✅ You've read the architecture documentation
✅ You can modify and understand the code

---

**Congratulations! 🎊**

Your Isaac Sim + ResQ-AI setup is complete and ready for development.

**Next steps:**
1. Modify scenes in `generate_urban_scene.py`
2. Experiment with different VLM backends
3. Train custom YOLO models
4. Integrate real drone data
5. Deploy as Docker container

---

**Estimated Total Time:** 45 minutes to 2 hours
- Setup & dependencies: 15-20 min
- Asset download: 15-30 min
- First test runs: 10-15 min
- Documentation review: 5-10 min
