#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ResQ-AI Headless End-to-End Test
#
# Bash equivalent of:   & "C:\isaacsim\python.bat" sim_bridge/headless_e2e_test.py
# Linux equivalent:     python3 sim_bridge/headless_e2e_test.py
#
# USAGE
#   bash scripts/run_e2e_test.sh
#
# OPTIONAL ENV VARS (all have defaults)
#   RESQAI_MAX_STEPS=300         Number of physics steps to run
#   RESQAI_DEBUG_DIR=./debug_output    Where to save frames + JSON artifacts
#   RESQAI_SCENE=./resqai_urban_disaster.usda    USD scene to load
#   RESQAI_VLM_URL_E2E=http://localhost:8000/analyze
#   RESQAI_YOLO_WEIGHTS=Phase1_SituationalAwareness/best.pt
#   RESQAI_SURVEY_ALT=45.0       Drone survey altitude (metres)
#
# EXAMPLE (shorter run + custom debug dir)
#   RESQAI_MAX_STEPS=100 RESQAI_DEBUG_DIR=./my_debug bash scripts/run_e2e_test.sh
#
# LIVE STREAMING (view in browser while running)
#   On this server:  bash scripts/run_e2e_test.sh --stream
#   On your laptop:  ssh -L 8211:localhost:8211 <brev-instance>
#   Then open:       http://localhost:8211/streaming/webrtc-demo/
#
# OUTPUTS
#   debug_output/frames/*_rgb.jpg         RGB camera frames
#   debug_output/frames/*_thermal.jpg     Thermal overlays
#   debug_output/frames/*_yolo.json       YOLO detections
#   debug_output/frames/*_cosmos_prompt.json  Cosmos Reason 2 prompts
#   debug_output/vlm_response*.json       VLM responses
#   debug_output/resqai_sim.mp4           Video of RGB frames (auto-generated)
# ═══════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# ─── Parse flags ─────────────────────────────────────────────────────────────
STREAM=0
for arg in "$@"; do
    case "$arg" in
        --stream) STREAM=2 ;;
    esac
done

if [[ "$STREAM" -gt 0 ]]; then
    export RESQAI_LIVESTREAM=2
fi

# ─── Activate Isaac Sim 5.1 Python 3.11 venv ────────────────────────────────
VENV_DIR="$SCRIPT_DIR/isaacsim_env"
if [[ -d "$VENV_DIR" ]]; then
    echo "[0/3] Activating Isaac Sim 5.1 venv ($VENV_DIR)..."
    source "$VENV_DIR/bin/activate"
    export ACCEPT_EULA=Y
else
    echo "ERROR: Isaac Sim venv not found at $VENV_DIR"
    echo "  Create it: python3.11 -m venv isaacsim_env && source isaacsim_env/bin/activate"
    echo "  pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com"
    exit 1
fi

# ─── Virtual display (Isaac Sim requires a display even in headless mode) ─────
# Always restart Xvfb fresh to avoid stale-display hangs at GLFW init
echo "[1/3] (Re)starting Xvfb virtual display on :99..."
pkill -9 Xvfb 2>/dev/null || true
sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1280x720x24 -ac > /tmp/xvfb_resqai.log 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

# ─── Check prerequisites ────────────────────────────────────────────────────
python3 -c "from isaacsim import SimulationApp" 2>/dev/null || {
    echo "ERROR: isaacsim not found. Run:"
    echo "  pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com"
    exit 1
}

# ─── Start VLM server in background ─────────────────────────────────────────
VLM_BACKEND="${VLM_BACKEND:-mock}"
echo "[2/3] Starting VLM server (backend: $VLM_BACKEND)..."
python3 orchestrator/vlm_server.py --backend "$VLM_BACKEND" &
VLM_PID=$!
# Give it a moment
sleep 2
curl -sf http://localhost:8000/health > /dev/null 2>&1 || echo "  (VLM server still starting, continuing...)"
echo "  VLM server PID: $VLM_PID"

# ─── Run the headless test ───────────────────────────────────────────────────
echo ""
echo "[3/3] Running headless E2E test..."
echo "      Debug output: ${RESQAI_DEBUG_DIR:-$SCRIPT_DIR/debug_output}"
echo "      Max steps   : ${RESQAI_MAX_STEPS:-300}"
if [[ "$STREAM" -gt 0 ]]; then
    echo ""
    echo "  ┌─── LIVE STREAM ──────────────────────────────────────────────────┐"
    echo "  │  Camera stream will be available at port 8211 (WebRTC)           │"
    echo "  │  On your laptop run:  ssh -L 8211:localhost:8211 <brev-name>     │"
    echo "  │  Then open:  http://localhost:8211/streaming/webrtc-demo/        │"
    echo "  └──────────────────────────────────────────────────────────────────┘"
fi
echo "      Ctrl-C to stop early."
echo ""

trap "kill $VLM_PID 2>/dev/null; kill ${XVFB_PID:-0} 2>/dev/null; exit 0" SIGINT SIGTERM

# PYTHONPATH ensures all project modules (orchestrator, sim_bridge, etc.) resolve
# DISPLAY=:99 uses the Xvfb virtual framebuffer started above
PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" DISPLAY=:99 python3 sim_bridge/headless_e2e_test.py

STATUS=$?
kill $VLM_PID 2>/dev/null || true

echo ""
if [[ $STATUS -eq 0 ]]; then
    echo "E2E test completed. Debug artifacts: ${RESQAI_DEBUG_DIR:-$SCRIPT_DIR/debug_output}"

    # ─── Auto-generate video from captured RGB frames ─────────────────────
    DEBUG_DIR="${RESQAI_DEBUG_DIR:-$SCRIPT_DIR/debug_output}"
    FRAMES_DIR="$DEBUG_DIR/frames"
    VIDEO_OUT="$DEBUG_DIR/resqai_sim.mp4"
    THERMAL_OUT="$DEBUG_DIR/resqai_thermal.mp4"

    if command -v ffmpeg > /dev/null 2>&1 && ls "$FRAMES_DIR"/*_rgb.jpg > /dev/null 2>&1; then
        echo "Encoding video..."
        ffmpeg -y -framerate 15 \
            -pattern_type glob -i "$FRAMES_DIR/*_rgb.jpg" \
            -vf "scale=960:540" -c:v libx264 -pix_fmt yuv420p "$VIDEO_OUT" \
            > /dev/null 2>&1 && echo "  RGB video  : $VIDEO_OUT"

        ffmpeg -y -framerate 15 \
            -pattern_type glob -i "$FRAMES_DIR/*_thermal.jpg" \
            -vf "scale=960:540" -c:v libx264 -pix_fmt yuv420p "$THERMAL_OUT" \
            > /dev/null 2>&1 && echo "  Thermal vid: $THERMAL_OUT"

        # Re-encode external overview with ffmpeg (higher quality than cv2)
        EXT_VID="$DEBUG_DIR/external_overview.mp4"
        EXT_VID_HQ="$DEBUG_DIR/external_overview_hq.mp4"
        if [[ -f "$EXT_VID" ]]; then
            ffmpeg -y -i "$EXT_VID" -c:v libx264 -pix_fmt yuv420p -crf 23 "$EXT_VID_HQ" \
                > /dev/null 2>&1 && echo "  Ext cam vid: $EXT_VID_HQ"
        fi

        # Re-encode forward FPV video
        FWD_VID="$DEBUG_DIR/forward_fpv.mp4"
        FWD_VID_HQ="$DEBUG_DIR/forward_fpv_hq.mp4"
        if [[ -f "$FWD_VID" ]]; then
            ffmpeg -y -i "$FWD_VID" -c:v libx264 -pix_fmt yuv420p -crf 23 "$FWD_VID_HQ" \
                > /dev/null 2>&1 && echo "  FPV cam vid: $FWD_VID_HQ"
        fi

        echo ""
        echo "  Copy to your laptop with:"
        echo "    scp $(hostname):$VIDEO_OUT ."
        echo "    scp $(hostname):$THERMAL_OUT ."
    fi
else
    echo "E2E test exited with code $STATUS"
fi
exit $STATUS
