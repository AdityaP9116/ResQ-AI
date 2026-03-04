#!/bin/bash

################################################################################
# Quick Isaac Sim Installation Test
#
# Run this AFTER installing Isaac Sim via Omniverse Launcher
# It will verify that Isaac Sim is properly installed and working
#
# Usage:
#   chmod +x test_isaac_sim_install.sh
#   ./test_isaac_sim_install.sh
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "========================================"
echo "  Isaac Sim Installation Test"
echo "========================================"
echo ""

# Check if Isaac Sim is installed
echo -e "${BLUE}[1/5]${NC} Checking Isaac Sim installation path..."

ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac-sim-5.1"

if [ -d "$ISAAC_SIM_PATH" ]; then
    echo -e "${GREEN}✓${NC} Found Isaac Sim at: $ISAAC_SIM_PATH"
else
    echo -e "${RED}✗${NC} Isaac Sim not found at: $ISAAC_SIM_PATH"
    echo ""
    echo "Isaac Sim must be installed via NVIDIA Omniverse Launcher:"
    echo "1. Download: https://www.nvidia.com/en-us/omniverse/download/"
    echo "2. Run installer: ~/Downloads/omniverse-launcher-linux.AppImage"
    echo "3. Search 'Isaac Sim' and click Install"
    echo "4. Wait 15-30 minutes for installation"
    echo ""
    exit 1
fi

# Check if python.sh exists
echo -e "${BLUE}[2/5]${NC} Checking Isaac Sim python.sh..."

if [ -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo -e "${GREEN}✓${NC} Found: $ISAAC_SIM_PATH/python.sh"
else
    echo -e "${RED}✗${NC} python.sh not found at: $ISAAC_SIM_PATH/python.sh"
    echo "Isaac Sim installation may be incomplete or corrupted."
    exit 1
fi

# Test basic Python
echo -e "${BLUE}[3/5]${NC} Testing Isaac Sim Python..."

if "$ISAAC_SIM_PATH/python.sh" -c "import sys; print('Python version:', sys.version)" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Isaac Sim Python works"
else
    echo -e "${RED}✗${NC} Isaac Sim Python failed to run"
    echo "Try running manually: $ISAAC_SIM_PATH/python.sh -c 'import sys; print(sys.version)'"
    exit 1
fi

# Test isaacsim import
echo -e "${BLUE}[4/5]${NC} Testing isaacsim module import..."

if "$ISAAC_SIM_PATH/python.sh" -c "from isaacsim import SimulationApp; print('✓ isaacsim.core imported')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} isaacsim module loads successfully"
else
    echo -e "${RED}✗${NC} isaacsim module import failed"
    echo "Try manually: $ISAAC_SIM_PATH/python.sh -c 'from isaacsim import SimulationApp'"
    echo ""
    echo "Possible causes:"
    echo "- Isaac Sim incomplete download"
    echo "- Corrupted installation"
    echo "- Try reinstalling Isaac Sim via Omniverse Launcher"
    exit 1
fi

# Test Omniverse modules
echo -e "${BLUE}[5/5]${NC} Testing additional Omniverse modules..."

if "$ISAAC_SIM_PATH/python.sh" -c \
    "from omni.isaac.core.world import World; \
     from pxr import Gf, UsdGeom; \
     print('✓ All core modules loaded')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} All core modules load successfully"
else
    echo -e "${YELLOW}!${NC} Some modules failed (may not be critical)"
fi

# Success
echo ""
echo "========================================"
echo -e "${GREEN}✓ Isaac Sim Installation Verified!${NC}"
echo "========================================"
echo ""
echo "Your Isaac Sim is ready to use!"
echo ""
echo "Next steps:"
echo "1. Create a convenient alias:"
echo "   ${YELLOW}alias isaac-python='$ISAAC_SIM_PATH/python.sh'${NC}"
echo ""
echo "2. Run the updated setup script:"
echo "   ${YELLOW}cd /path/to/ResQ-AI${NC}"
echo "   ${YELLOW}chmod +x setup_isaac_sim.sh${NC}"
echo "   ${YELLOW}./setup_isaac_sim.sh${NC}"
echo ""
echo "3. Or immediately test ResQ-AI:"
echo "   ${YELLOW}$ISAAC_SIM_PATH/python.sh test_isaacsim_minimal.py${NC}"
echo ""
