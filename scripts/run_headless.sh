#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ResQ-AI Headless Simulation Launcher
#
# Runs the simulation in headless mode with optional WebRTC streaming or
# video recording.
#
# USAGE
#   # WebRTC live stream (open browser at http://localhost:8211/streaming/webrtc-demo/)
#   bash scripts/run_headless.sh --stream
#
#   # Record to video file (/tmp/resqai_demo.mp4)
#   bash scripts/run_headless.sh --record
#
#   # Both stream and record
#   bash scripts/run_headless.sh --stream --record
#
#   # Use NIM backend for Cosmos Reason 2
#   NVIDIA_API_KEY=nvapi-... bash scripts/run_headless.sh --stream --vlm nim
#
# STREAMING OUTPUT
#   WebRTC browser URL : http://localhost:8211/streaming/webrtc-demo/
#   Recorded video     : /tmp/resqai_demo.mp4
#   Flight report      : orchestrator/Flight_Report.json
#   Hazard map         : orchestrator/Hazard_Map.html
# ═══════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Defaults
STREAM=false
RECORD=false
VLM_BACKEND="mock"
NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --stream)   STREAM=true  ; shift ;;
        --record)   RECORD=true  ; shift ;;
        --vlm)      VLM_BACKEND="$2" ; shift 2 ;;
        --help|-h)
            sed -n '/^# /p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1" ; exit 1 ;;
    esac
done

# Build SimulationApp livestream flag
LIVESTREAM_FLAG=0
if $STREAM; then
    LIVESTREAM_FLAG=2  # WebRTC
fi

# Build demo_flight args
DEMO_ARGS="--headless --livestream $LIVESTREAM_FLAG"
$RECORD && DEMO_ARGS="$DEMO_ARGS --record"

# ─── Step 1: Start VLM server ───────────────────────────────────────────────
echo "[1/2] Starting VLM server (backend: $VLM_BACKEND)..."
if [[ "$VLM_BACKEND" == "nim" ]]; then
    if [[ -z "$NVIDIA_API_KEY" ]]; then
        echo "  ERROR: NVIDIA_API_KEY not set. Export it or use mock backend."
        exit 1
    fi
    NVIDIA_API_KEY="$NVIDIA_API_KEY" python3 orchestrator/vlm_server.py \
        --backend nim &
elif [[ "$VLM_BACKEND" == "vllm" ]]; then
    python3 orchestrator/vlm_server.py --backend vllm &
else
    python3 orchestrator/vlm_server.py --backend mock &
fi

VLM_PID=$!
echo "  VLM server PID: $VLM_PID"
echo "  Waiting for server to start..."
sleep 3

# Check server is up
if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  WARNING: VLM server may not be ready yet (continuing anyway)"
fi

# ─── Step 2: Launch simulation ──────────────────────────────────────────────
echo ""
echo "[2/2] Launching simulation (headless=$STREAM, record=$RECORD)..."
if $STREAM; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  WebRTC stream ready once simulation starts:         │"
    echo "  │  http://localhost:8211/streaming/webrtc-demo/        │"
    echo "  │                                                       │"
    echo "  │  Forward the port if running remotely:               │"
    echo "  │  ssh -L 8211:localhost:8211 <user>@<host>            │"
    echo "  └─────────────────────────────────────────────────────┘"
    echo ""
fi

# Trap Ctrl-C to cleanly stop VLM server too
trap "echo ''; echo 'Stopping...'; kill $VLM_PID 2>/dev/null; exit 0" SIGINT SIGTERM

python3 sim_bridge/demo_flight.py $DEMO_ARGS

# Wait for VLM server
kill $VLM_PID 2>/dev/null || true
echo ""
echo "Done."
if $RECORD; then
    echo "  Video: /tmp/resqai_demo.mp4"
fi
echo "  Report: orchestrator/Flight_Report.json"
