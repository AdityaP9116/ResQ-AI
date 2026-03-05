#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ResQ-AI Pipeline v2 — Full Orchestrator
#
# Runs the complete v2 pipeline:
#   Step 1: Render aerial view with real Flow particle fires (Isaac Sim)
#   Step 2: Run Cosmos Phase 1 on the rendered aerial image
#   Step 3: Run a full 5-zone drone simulation (Isaac Sim)
#   Step 4: Run YOLO + Cosmos pipeline on all 5 zones
#
# USAGE:
#   bash scripts/run_pipeline_v2.sh            # full pipeline
#   bash scripts/run_pipeline_v2.sh --skip-sim # skip Isaac Sim steps (use existing data)
#
# OUTPUTS:
#   debug_output_2/aerial_rendered.jpg       Rendered aerial image
#   debug_output_2/phase1_annotated.jpg      Aerial with fire zone annotations
#   debug_output_2/frames/                   Drone camera frames (all 5 zones)
#   debug_output_2/external_overview_hq.mp4  External camera overview video
#   debug_output_2/forward_fpv_hq.mp4        Forward FPV video
#   debug_output_2/phase2_results.json       YOLO results
#   debug_output_2/phase3_results.json       Cosmos assessment + ranking
#   debug_output_2/mission_report.json       Final mission report
# ═══════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

SKIP_SIM=0
for arg in "$@"; do
    case "$arg" in
        --skip-sim) SKIP_SIM=1 ;;
    esac
done

# ─── Activate Isaac Sim venv ────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/isaacsim_env"
if [[ -d "$VENV_DIR" ]]; then
    echo "[0/4] Activating Isaac Sim venv..."
    source "$VENV_DIR/bin/activate"
    export ACCEPT_EULA=Y
else
    echo "ERROR: Isaac Sim venv not found at $VENV_DIR"
    exit 1
fi

# ─── Output directory ───────────────────────────────────────────────────
DEBUG_DIR="$SCRIPT_DIR/debug_output_2"
mkdir -p "$DEBUG_DIR"

# ─── Virtual display ────────────────────────────────────────────────────
echo "[1/4] (Re)starting virtual display..."
pkill -9 Xvfb 2>/dev/null || true
sleep 1
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
Xvfb :99 -screen 0 1920x1080x24 -ac > /tmp/xvfb_resqai_v2.log 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99

# ─── Start VLM server (for headless sim Cosmos waypoint requests) ───────
VLM_BACKEND="${VLM_BACKEND:-mock}"
echo "  Starting VLM server (backend: $VLM_BACKEND)..."
python3 orchestrator/vlm_server.py --backend "$VLM_BACKEND" &
VLM_PID=$!
sleep 2

trap "kill $VLM_PID 2>/dev/null; kill ${XVFB_PID:-0} 2>/dev/null; exit 0" SIGINT SIGTERM

if [[ "$SKIP_SIM" -eq 0 ]]; then
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Render aerial view with real fires
    # ═══════════════════════════════════════════════════════════════════
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  STEP 1: Rendering aerial view with real Flow fires        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    export RESQAI_AERIAL_OUTPUT="$DEBUG_DIR/aerial_rendered.jpg"
    export RESQAI_SCENE="$SCRIPT_DIR/resqai_urban_disaster.usda"
    export RESQAI_WARMUP_STEPS=80
    export RESQAI_FIRE_GROW_STEPS=120
    export RESQAI_CAM_ALT=180

    PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" DISPLAY=:99 \
        python3 sim_bridge/render_aerial_view.py

    if [[ ! -f "$DEBUG_DIR/aerial_rendered.jpg" ]]; then
        echo "ERROR: Aerial render failed — no output image"
        kill $VLM_PID 2>/dev/null || true
        exit 1
    fi
    echo "  ✓ Aerial image rendered: $DEBUG_DIR/aerial_rendered.jpg"

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Run 5-zone drone simulation
    # ═══════════════════════════════════════════════════════════════════
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  STEP 2: Running 5-zone drone simulation                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    export RESQAI_MAX_STEPS=700
    export RESQAI_DEBUG_DIR="$DEBUG_DIR"
    export RESQAI_SCENE="$SCRIPT_DIR/resqai_urban_disaster.usda"
    export RESQAI_SURVEY_ALT=110.0
    export RESQAI_ZONE_ORDER="FZ_1,FZ_2,FZ_0,FZ_3,FZ_4"
    export RESQAI_FOCUS_DURATION=45
    export RESQAI_COOLDOWN_STEPS=25

    PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" DISPLAY=:99 \
        python3 sim_bridge/headless_e2e_test.py

    FRAME_COUNT=$(ls "$DEBUG_DIR/frames/"*_rgb.jpg 2>/dev/null | wc -l)
    echo "  ✓ Simulation complete: $FRAME_COUNT frames captured"

    # Re-encode videos with ffmpeg
    if command -v ffmpeg > /dev/null 2>&1; then
        echo "  Encoding videos..."
        ffmpeg -y -framerate 15 \
            -pattern_type glob -i "$DEBUG_DIR/frames/*_rgb.jpg" \
            -vf "scale=960:540" -c:v libx264 -pix_fmt yuv420p "$DEBUG_DIR/resqai_sim.mp4" \
            > /dev/null 2>&1 && echo "    RGB video encoded"

        for raw_vid in "external_overview" "forward_fpv"; do
            if [[ -f "$DEBUG_DIR/${raw_vid}.mp4" ]]; then
                ffmpeg -y -i "$DEBUG_DIR/${raw_vid}.mp4" \
                    -c:v libx264 -pix_fmt yuv420p -crf 23 \
                    "$DEBUG_DIR/${raw_vid}_hq.mp4" > /dev/null 2>&1 \
                    && echo "    ${raw_vid}_hq.mp4 encoded"
            fi
        done
    fi

else
    echo ""
    echo "  --skip-sim: Skipping Isaac Sim steps. Using existing data in $DEBUG_DIR"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Run Pipeline v2 (Cosmos Phase 1 + YOLO + Cosmos Phase 3)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  STEP 3: Running Cosmos + YOLO pipeline (all 5 zones)      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python3 run_cosmos_pipeline_v2.py

# ─── Cleanup ────────────────────────────────────────────────────────────
kill $VLM_PID 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Pipeline v2 complete!"
echo "  Outputs: $DEBUG_DIR"
echo "  View:    page3.html (aerial) / page4.html (dashboard)"
echo "═══════════════════════════════════════════════════════════════"
