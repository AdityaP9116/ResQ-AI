# Isaac Sim Installation Fix Guide

If you're seeing this error:
```
PYTHONPATH: path doesn't exist
[✗] Isaac Sim verification failed. Check installation.
```

**This is expected!** The issue is that Isaac Sim **must be installed via the NVIDIA Omniverse Launcher** - it cannot be installed with `pip install isaacsim`.

## ✅ Correct Installation Steps

### Step 1: Download Omniverse Launcher

```bash
# Download the Launcher
curl -fsSL https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage \
  -o ~/Downloads/omniverse-launcher.AppImage

# Make executable
chmod +x ~/Downloads/omniverse-launcher.AppImage

# Run it
~/Downloads/omniverse-launcher.AppImage
```

### Step 2: Install Isaac Sim through the UI

1. **Sign in** with your free NVIDIA account (create one at developer.nvidia.com if needed)
2. **Search** for "Isaac Sim" in the launcher
3. **Select** version 5.1 or latest
4. **Click Install** and wait 15-30 minutes
5. **Installation path** will be: `~/.local/share/ov/pkg/isaac-sim-5.1`

### Step 3: Verify Installation

```bash
# Check if Isaac Sim is installed
ls ~/​.local/share/ov/pkg/ | grep isaac

# Should show something like: isaac-sim-5.1 or isaac-sim-4.0
```

### Step 4: Run Setup Script Again

```bash
cd /path/to/ResQ-AI

# Now run the setup script - it will find Isaac Sim
chmod +x setup_isaac_sim.sh
./setup_isaac_sim.sh
```

### Step 5: Test Installation

```bash
# After setup completes, test with Isaac Sim's Python
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c "from isaacsim import SimulationApp; print('✓ Success!')"
```

---

## 🚨 Common Issues & Solutions

### Issue: "Isaac Sim not found in standard locations"

**Solution:**
```bash
# Check if it's installed elsewhere
find ~/.local/share/ov/pkg -name "isaac-sim-*" -type d

# If found, set the correct path:
export ISAAC_SIM_PATH="/your/path/to/isaac-sim-5.1"
```

### Issue: Omniverse Launcher won't run

**Solution:**
```bash
# Ensure required dependencies
sudo apt-get install -y \
  libxcomposite1 \
  libxdamage1 \
  libxrandr2 \
  libxinerama1 \
  libxcursor1 \
  libxi6

# Try running again
~/Downloads/omniverse-launcher.AppImage
```

### Issue: "Connection refused" when running launcher

**Solution:**
```bash
# Clear launcher cache and try again
rm -rf ~/.config/NVIDIA
rm -rf ~/.cache/nvidia

# Restart launcher
~/Downloads/omniverse-launcher.AppImage
```

### Issue: Installation stuck or slow

**Solution:**
- Close other applications to free up bandwidth
- Check internet connection: `ping nvidia.com`
- Use a wired connection if WiFi is slow
- Installation can take 20-60 minutes depending on internet speed

---

## 📱 Manual Installation (Advanced)

If the launcher doesn't work, you can manually download Isaac Sim:

```bash
# 1. Create directory
mkdir -p ~/.local/share/ov/pkg

# 2. Download Isaac Sim (>30GB file)
# Visit: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html
# Download the .tar.xz file to ~/.local/share/ov/pkg/

# 3. Extract
cd ~/.local/share/ov/pkg
tar xf isaac-sim-*.tar.xz

# 4. Verify
ls -la isaac-sim-5.1/python.sh
```

---

## 🔧 Correct Way to Run ResQ-AI Scripts

After Isaac Sim is installed, **use Isaac Sim's Python**, not a virtual environment's Python:

### ✅ Correct Way

```bash
# Use Isaac Sim's embedded Python
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh test_isaacsim_minimal.py

# Or create an alias for convenience
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"
isaac-python test_isaacsim_minimal.py
```

### ❌ Wrong Way (Don't Do This)

```bash
# DON'T activate a venv and run directly
source isaac_env/bin/activate
python test_isaacsim_minimal.py  # ← This won't work!
```

---

## 📝 Updated Commands

After Isaac Sim is installed, use these commands:

```bash
# Define alias once
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"

# Test import
isaac-python -c "from isaacsim import SimulationApp; print('✓ OK')"

# Run mini tests
isaac-python test_isaacsim_minimal.py

# Run simulations
isaac-python sim_bridge/main_sim_loop.py --headless

# Run demo (with VLM server in another terminal)
isaac-python orchestrator/vlm_server.py --backend mock
isaac-python sim_bridge/demo_flight.py --headless --vlm-url http://localhost:8000/analyze
```

---

## 🔄 Updated Setup Script

The setup script has been updated to:
1. ✅ Check for Isaac Sim installed via Omniverse Launcher
2. ✅ Provide clear instructions if not found
3. ✅ Use Isaac Sim's Python for verification
4. ✅ Skip problematic `pip install isaacsim`

---

## 📋 Verification Checklist

After following these steps:

- [ ] Omniverse Launcher downloaded and installed
- [ ] Isaac Sim installed via launcher UI
- [ ] Isaac Sim path exists: `~/.local/share/ov/pkg/isaac-sim-5.1/`
- [ ] `isaac-python` alias works
- [ ] Basic import test passes
- [ ] `test_isaacsim_minimal.py` runs successfully
- [ ] First demo flight works

---

## ⚡ Quick Recovery

If you've already run the failed setup:

```bash
cd /path/to/ResQ-AI

# 1. Make sure Isaac Sim is installed (see above)

# 2. Clean up the old venv if needed
rm -rf isaac_env

# 3. Re-run setup script
chmod +x setup_isaac_sim.sh
./setup_isaac_sim.sh

# 4. Install ResQ-AI dependencies manually
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -m pip install -r requirements.txt

# 5. Test again
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh test_isaacsim_minimal.py
```

---

## 🎓 Why This Happens

**Why not just `pip install isaacsim`?**

Isaac Sim is a complex application with:
- Custom C++ extensions for physics, graphics, rendering
- Omniverse Nucleus integration
- Pre-compiled binaries for GPUs
- Specific system dependencies

It's too complex to package as a simple pip wheel. It **requires the full Omniverse stack** which is only properly installed via the Launcher or manual tarball extraction.

---

## 📞 Need Help?

1. **Verify Isaac Sim installed:**
   ```bash
   ls -la ~/.local/share/ov/pkg/isaac-sim-5.1/
   # Should show: python.sh, kit, exts, etc.
   ```

2. **Test Isaac Sim directly:**
   ```bash
   ~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c \
     "from isaacsim import SimulationApp; from omni.isaac.core.world import World; print('✓ All imports OK')"
   ```

3. **Check official docs:**
   - https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html

4. **NVIDIA Forums:**
   - https://forums.developer.nvidia.com/c/omniverse/isaac/

---

Good luck! Once Isaac Sim is properly installed via the Launcher, everything will work smoothly. 🚀
