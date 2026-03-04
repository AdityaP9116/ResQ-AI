#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# ResQ-AI Full Demo Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# Step 1: Start the VLM server (pick one backend):
#
#   Mock (no GPU, no API key — for testing):
#     python orchestrator/vlm_server.py
#
#   NVIDIA NIM Cloud (requires API key):
#     export NVIDIA_API_KEY='nvapi-...'
#     python orchestrator/vlm_server.py --backend nim
#
#   Local vLLM (requires GPU + model download):
#     vllm serve nvidia/Cosmos-Reason2-2B --port 8001
#     python orchestrator/vlm_server.py --backend vllm
#
# Step 2: Run the demo flight (requires Isaac Sim):
#     /isaac-sim/python.sh sim_bridge/demo_flight.py --record
#
# Step 3: Generate hazard map (after demo):
#     python orchestrator/generate_map.py --report orchestrator/Flight_Report.json
#
# Outputs:
#   /tmp/resqai_demo.mp4              Annotated demo video
#   /tmp/resqai_urban_disaster.usda   Generated scene (with fixed materials)
#   orchestrator/Flight_Report.json   Flight path + hazards + Cosmos decisions
#   orchestrator/Hazard_Map.html      Interactive Folium map
# ═══════════════════════════════════════════════════════════════════════════

echo "See instructions above. Run each step in a separate terminal."
