# How to Fix the Isaac Sim Installation Error

You're seeing this error:
```
PYTHONPATH: path doesn't exist (/home/ubuntu/ResQ-AI/isaac_env/lib/python3.10/site-packages/isaacsim/exts/isaacsim.simulation_app)
PYTHONPATH: path doesn't exist (/home/ubuntu/ResQ-AI/isaac_env/lib/python3.10/site-packages/isaacsim/extsDeprecated/omni.isaac.kit)
[✗] Isaac Sim verification failed. Check installation.
```

## What's Wrong?

The setup script attempted to install Isaac Sim with `pip install isaacsim`, which **doesn't work**. Isaac Sim is a complex application that requires the full Omniverse stack and must be installed via the official NVIDIA Omniverse Launcher.

The pip package doesn't include all the necessary extensions and components.

## Solution (5 steps, 30 minutes)

### Step 1: Download Omniverse Launcher

```bash
# Download the launcher
curl -fsSL https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage \
  -o ~/Downloads/omniverse-launcher.AppImage

# Make it executable
chmod +x ~/Downloads/omniverse-launcher.AppImage

# Run it
~/Downloads/omniverse-launcher.AppImage
```

### Step 2: Sign In

In the launcher window:
1. Click "Sign In"
2. Create a free NVIDIA account (or use existing)
3. Complete the login flow

### Step 3: Install Isaac Sim

1. Search for "Isaac Sim"
2. Select version **5.1** or higher
3. Click the **Install** button
4. Wait 15-30 minutes for download and installation

The installer will show progress. **Installation path will be:** 
```
~/.local/share/ov/pkg/isaac-sim-5.1
```

### Step 4: Verify Installation Works

```bash
# Test that Isaac Sim is properly installed
~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c \
  "from isaacsim import SimulationApp; print('✓ Isaac Sim installed successfully!')"

# You should see: ✓ Isaac Sim installed successfully!
```

**If you see an error, Isaac Sim installation failed.** Try:
- Check disk space: `df -h /`
- Verify launcher downloaded correctly: `ls -la ~/Downloads/omniverse-launcher.AppImage`
- Try installing again from the launcher UI

### Step 5: Run Setup Script Again

Now that Isaac Sim is installed, the setup script will work:

```bash
cd /path/to/ResQ-AI

# Clean up the failed venv (optional, but recommended)
rm -rf isaac_env

# Run setup script again
chmod +x setup_isaac_sim.sh
./setup_isaac_sim.sh
```

The script will now:
- ✓ Find Isaac Sim at `~/.local/share/ov/pkg/isaac-sim-5.1`
- ✓ Create a Python virtual environment
- ✓ Install ResQ-AI dependencies
- ✓ Verify everything works

---

## Quick Commands After Installation

Once Isaac Sim is installed, use it like this:

```bash
# Create convenient alias
alias isaac-python="~/.local/share/ov/pkg/isaac-sim-5.1/python.sh"

# Test import
isaac-python -c "from isaacsim import SimulationApp; print('✓ OK')"

# Run ResQ-AI tests
isaac-python test_isaacsim_minimal.py

# Run simulations
isaac-python sim_bridge/main_sim_loop.py --headless
isaac-python sim_bridge/demo_flight.py --headless
```

**Important:** Always use `isaac-python` (which points to `~/.local/share/ov/pkg/isaac-sim-5.1/python.sh`), not `python` from a virtual environment.

---

## Verification Checklist

After following these steps:

- [ ] Launcher downloaded and installed
- [ ] Isaac Sim downloaded via Launcher (15-30 minutes)
- [ ] Path exists: `ls ~/.local/share/ov/pkg/isaac-sim-5.1/`
- [ ] Import works: `~/.local/share/ov/pkg/isaac-sim-5.1/python.sh -c "from isaacsim import SimulationApp"`
- [ ] Setup script runs successfully
- [ ] Test passes: `isaac-python test_isaacsim_minimal.py`

---

## If You Still Have Issues

Check the full troubleshooting guide: [ISAAC_SIM_INSTALLATION_FIX.md](ISAAC_SIM_INSTALLATION_FIX.md)

Key things to verify:
1. **Disk space:** `df -h /` (need 100+ GB)
2. **GPU drivers:** `nvidia-smi` (need 550+)
3. **Isaac Sim path:** `ls ~/.local/share/ov/pkg/isaac-sim-5.1/python.sh`
4. **Direct import:** `~/.local/share/ov/pkg/isaac-sim-5.1/python.sh << 'EOF'`
   ```python
   from isaacsim import SimulationApp
   from omni.isaac.core.world import World
   from pxr import Gf, UsdGeom
   print("✓ All imports work!")
   EOF
   ```

---

## Why This Happened

The original setup script tried to use `pip install isaacsim`, but:

- Isaac Sim is a complex application with custom C++ extensions
- It requires the full Omniverse SDK
- It has GPU-specific binaries
- It cannot be packaged as a simple Python package

**The ONLY supported installation method is via Omniverse Launcher.**

---

## Next Steps

1. ✅ Download and run Omniverse Launcher
2. ✅ Install Isaac Sim (5.1+)
3. ✅ Verify installation
4. ✅ Run setup script
5. ✅ Download asset packs
6. ✅ Download YOLO weights
7. ✅ Run first simulation

Good luck! 🚀
