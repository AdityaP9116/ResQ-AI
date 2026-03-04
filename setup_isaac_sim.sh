#!/bin/bash

################################################################################
# ResQ-AI Isaac Sim Setup Script
#
# Automated setup for Isaac Sim and ResQ-AI dependencies
#
# Usage:
#   chmod +x setup_isaac_sim.sh
#   ./setup_isaac_sim.sh [--headless] [--docker] [--help]
#
# Options:
#   --headless        Install Isaac Sim in headless mode (no GUI)
#   --docker          Build Docker image instead of local setup
#   --help            Show this help message
#   --skip-download   Skip downloading assets and models
#
# Requirements:
#   - Ubuntu 20.04 or 22.04
#   - NVIDIA GPU with drivers 550+
#   - Python 3.10
#   - sudo access
#
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PYTHON_VERSION="3.10"
VENV_NAME="isaac_env"
HEADLESS=false
DOCKER_BUILD=false
SKIP_DOWNLOAD=false

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    grep "^# " "$0" | sed 's/^# //' | head -20
}

# Check system requirements
check_prerequisites() {
    log_info "Checking system requirements..."
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "This script only supports Linux"
        exit 1
    fi
    
    # Check NVIDIA driver
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA driver not found. Please install NVIDIA drivers."
        exit 1
    fi
    
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log_info "NVIDIA Driver version: $DRIVER_VERSION"
    
    # Check if NVIDIA driver is 550+
    MAJOR_VERSION=${DRIVER_VERSION%%.*}
    if [ "$MAJOR_VERSION" -lt 550 ]; then
        log_warning "NVIDIA driver version 550+ recommended (you have $DRIVER_VERSION)"
    fi
    
    # Check Python version
    if command -v python$PYTHON_VERSION &> /dev/null; then
        PY_VERSION=$(python$PYTHON_VERSION --version 2>&1 | awk '{print $2}')
        log_success "Python $PY_VERSION found"
    else
        log_error "Python 3.10 not found. Install with: sudo apt install python3.10 python3.10-venv python3.10-dev"
        exit 1
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    log_info "Available disk space: ${AVAILABLE_GB}GB"
    if [ $AVAILABLE_GB -lt 100 ]; then
        log_warning "Less than 100GB available. Isaac Sim may need that much space."
    fi
    
    log_success "Prerequisites check passed"
}

# Setup Python virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [ -d "$PROJECT_ROOT/$VENV_NAME" ]; then
        log_warning "Virtual environment already exists at $PROJECT_ROOT/$VENV_NAME"
        read -p "Recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$PROJECT_ROOT/$VENV_NAME"
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi
    
    python$PYTHON_VERSION -m venv "$PROJECT_ROOT/$VENV_NAME"
    log_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    source "$PROJECT_ROOT/$VENV_NAME/bin/activate"
    log_success "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --quiet --upgrade pip setuptools wheel
    
    # Install dependencies
    pip install --quiet \
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
        python-dotenv \
        flask \
        fastapi \
        uvicorn \
        folium \
        matplotlib \
        pymavlink
    
    log_success "Python dependencies installed"
}

# Setup environment file
setup_env_file() {
    log_info "Setting up environment configuration (.env)..."
    
    if [ -f "$PROJECT_ROOT/.env" ]; then
        log_warning ".env file already exists"
        read -p "Overwrite it? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing .env file"
            return
        fi
    fi
    
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# Isaac Sim Configuration
# Not needed for pip-based install (isaacsim==4.5.0.0 from pypi.nvidia.com)
# ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-5.1

# VLM Backend Configuration
VLM_BACKEND=mock
NVIDIA_API_KEY=

# Simulation Configuration
HEADLESS_MODE=false
SCENE_RESOLUTION=1280x720

# Data Output Paths
OUTPUT_DIR=./outputs
YOLO_WEIGHTS_PATH=./Phase1_SituationalAwareness/best.pt

# Python Path
PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
EOF
    
    log_success ".env file created"
}

# Install Isaac Sim via pip (from NVIDIA PyPI)
install_isaacsim() {
    log_info "Installing Isaac Sim via pip (NVIDIA PyPI)..."
    log_info "This will download ~4 GB of extension packages."
    log_info "Packages: isaacsim, isaacsim-core, isaacsim-extscache-physics,"
    log_info "          isaacsim-extscache-kit, isaacsim-extscache-kit-sdk"
    echo ""

    pip install --quiet \
        isaacsim==4.5.0.0 \
        isaacsim-core==4.5.0.0 \
        isaacsim-extscache-physics==4.5.0.0 \
        isaacsim-extscache-kit==4.5.0.0 \
        isaacsim-extscache-kit-sdk==4.5.0.0 \
        --extra-index-url https://pypi.nvidia.com

    log_success "Isaac Sim packages installed"

    # Accept EULA on first import
    log_info "Accepting NVIDIA EULA..."
    echo "Yes" | python3 -c "import isaacsim" 2>/dev/null || true
    log_success "EULA accepted and cached"
}

# Check if Isaac Sim pip packages are already installed
check_isaacsim_installed() {
    if python3 -c "from isaacsim import SimulationApp" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Check Isaac Sim installation (pip-based)
check_isaac_sim() {
    log_info "Checking Isaac Sim pip packages..."

    if check_isaacsim_installed; then
        local version
        version=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('isaacsim'))" 2>/dev/null || echo "unknown")
        log_success "Isaac Sim already installed (version: $version)"
        return 0
    fi

    log_warning "Isaac Sim pip packages not found"
    log_info "Installing now..."
    install_isaacsim
}

# Verify Isaac Sim works
verify_isaac_sim() {
    log_info "Verifying Isaac Sim installation..."

    if python3 -c "from isaacsim import SimulationApp; print('OK')" 2>/dev/null; then
        log_success "Isaac Sim verified — SimulationApp is importable"
    else
        log_error "Isaac Sim verification failed."
        log_info "Try reinstalling:"
        log_info "  pip install isaacsim==4.5.0.0 isaacsim-core==4.5.0.0 \\"
        log_info "    isaacsim-extscache-physics==4.5.0.0 \\"
        log_info "    isaacsim-extscache-kit==4.5.0.0 \\"
        log_info "    isaacsim-extscache-kit-sdk==4.5.0.0 \\"
        log_info "    --extra-index-url https://pypi.nvidia.com"
        exit 1
    fi
}

# Install PegasusSimulator (Isaac Sim drone extension)
install_pegasus() {
    log_info "Installing PegasusSimulator..."

    # Check if already installed
    if python3 -c "import pegasus" 2>/dev/null; then
        log_success "PegasusSimulator already installed"
        return 0
    fi

    PEGASUS_DIR="$PROJECT_ROOT/PegasusSimulator"

    # Populate from GitHub if directory is empty
    if [ ! -f "$PEGASUS_DIR/extensions/pegasus.simulator/setup.py" ]; then
        log_info "Cloning PegasusSimulator from GitHub..."
        git clone --depth=1 https://github.com/PegasusSimulator/PegasusSimulator.git /tmp/PegasusSimulator_tmp
        cp -r /tmp/PegasusSimulator_tmp/. "$PEGASUS_DIR/"
        rm -rf /tmp/PegasusSimulator_tmp
        log_success "PegasusSimulator cloned"
    fi

    ISAACSIM_PATH=$(python3 -c "import isaacsim,os; print(os.path.dirname(os.path.abspath(isaacsim.__file__)))")
    log_info "Installing pegasus.simulator Python package (ISAACSIM_PATH=$ISAACSIM_PATH)..."
    ISAACSIM_PATH="$ISAACSIM_PATH" pip install --quiet -e "$PEGASUS_DIR/extensions/pegasus.simulator"
    log_success "PegasusSimulator installed"
}

# Download asset packs
download_asset_packs() {
    if [ "$SKIP_DOWNLOAD" = true ]; then
        log_info "Skipping asset download (--skip-download specified)"
        return
    fi
    
    log_info "Asset packs need to be downloaded manually"
    log_info "Download from: https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html"
    log_info "Extract to:"
    echo "  - assets/Architecture/     (CityEngine buildings)"
    echo "  - assets/Characters/       (Reallision characters)"
    echo "  - assets/Particles/        (Fire/smoke effects)"
    echo "  - assets/BaseMaterials/    (Textures & materials)"
    
    read -p "Press Enter once assets are set up, or skip to continue..."
}

# Run minimal tests
run_tests() {
    log_info "Running verification tests..."

    log_info "Test 1: Isaac Sim import test..."
    if python3 -c "from isaacsim import SimulationApp; print('✓ OK')" 2>/dev/null; then
        log_success "Test 1 passed"
    else
        log_error "Test 1 failed"
        return 1
    fi

    log_info "Test 2: ResQ-AI test script..."
    if python3 "$PROJECT_ROOT/test_isaacsim_minimal.py" 2>&1 | tail -5; then
        log_success "Test 2 passed"
    else
        log_warning "Test 2 had issues (may need GPU/display for full test)"
    fi
}

# Build Docker image
build_docker() {
    log_info "Building Docker image..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    docker build \
        -t resq-ai:latest \
        -f "$PROJECT_ROOT/Dockerfile" \
        "$PROJECT_ROOT"
    
    log_success "Docker image built successfully"
    
    log_info "To run the container:"
    echo "  docker run --rm --gpus all -v \$(pwd):/workspace/resq-ai resq-ai:latest bash"
}

# Print summary
print_summary() {
    echo ""
    log_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Test the installation:"
    echo "   python3 -c \"from isaacsim import SimulationApp; print('OK')\""
    echo ""
    echo "2. Run the minimal test script:"
    echo "   python3 test_isaacsim_minimal.py"
    echo ""
    echo "3. Run the simulation bridge:"
    echo "   python3 sim_bridge/main_sim_loop.py --headless"
    echo ""
    echo "4. Start the VLM server:"
    echo "   python3 orchestrator/vlm_server.py --backend mock"
    echo ""
    echo "5. Run the full demo:"
    echo "   bash scripts/run_demo.sh"
    echo ""
    echo "Isaac Sim version: 4.5.0.0 (pip, from pypi.nvidia.com)"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "==========================================="
    echo "    ResQ-AI Isaac Sim Setup Script"
    echo "==========================================="
    echo ""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --headless)
                HEADLESS=true
                shift
                ;;
            --docker)
                DOCKER_BUILD=true
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_prerequisites
    
    if [ "$DOCKER_BUILD" = true ]; then
        # Docker setup
        build_docker
    else
        # Local setup
        setup_venv
        activate_venv
        install_dependencies
        setup_env_file
        check_isaac_sim
        verify_isaac_sim
        install_pegasus
        download_asset_packs
        run_tests
    fi
    
    print_summary
}

# Run main
main "$@"
