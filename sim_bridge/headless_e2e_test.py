#!/usr/bin/env python3
"""ResQ-AI headless end-to-end integration test.

Runs the full Sense pipeline (scene -> drone -> YOLO -> 3-D projection -> Cosmos
prompt) in headless mode and dumps every intermediate artifact to a debug
directory for manual inspection.

Configuration is via ENVIRONMENT VARIABLES (not CLI args) because Isaac Sim's
kit.exe crashes on unrecognized command-line flags.

Usage::

    # Start VLM server first (in another terminal):
    python orchestrator/vlm_server.py

    # Run the headless test (uses defaults):
    C:\\isaacsim\\python.bat sim_bridge/headless_e2e_test.py

    # Override settings via env vars:
    $env:RESQAI_MAX_STEPS = "200"
    $env:RESQAI_DEBUG_DIR = "./my_debug"
    C:\\isaacsim\\python.bat sim_bridge/headless_e2e_test.py
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must precede ALL third-party / Omniverse imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

# Force headless — this is the core purpose of this script
simulation_app = SimulationApp({"headless": True})

# ---------------------------------------------------------------------------
# Third-party + Omniverse imports (valid only AFTER SimulationApp)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np

import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from pxr import Gf, UsdGeom

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from sim_bridge.generate_urban_scene import main as generate_scene
from sim_bridge.spawn_drone import spawn_resqai_drone
from sim_bridge.thermal_sim import generate_synthetic_thermal
from sim_bridge.projection_utils import make_intrinsics_from_fov

from orchestrator.orchestrator_bridge import OrchestratorBridge


# ---------------------------------------------------------------------------
# Configuration from environment variables (with sensible defaults)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCENE_PATH = os.environ.get(
    "RESQAI_SCENE",
    os.path.join(_PROJECT_ROOT, "resqai_urban_disaster.usda"),
)
VLM_URL = os.environ.get("RESQAI_VLM_URL_E2E", "http://localhost:8000/analyze")
YOLO_WEIGHTS = os.environ.get("RESQAI_YOLO_WEIGHTS", None)
MAX_STEPS = int(os.environ.get("RESQAI_MAX_STEPS", "300"))
DEBUG_DIR = os.environ.get(
    "RESQAI_DEBUG_DIR",
    os.path.join(_PROJECT_ROOT, "debug_output"),
)
SURVEY_ALT = float(os.environ.get("RESQAI_SURVEY_ALT", "45.0"))


# ---------------------------------------------------------------------------
# Sensor extraction helpers (same as demo_flight.py)
# ---------------------------------------------------------------------------

def _extract_rgb(drone) -> np.ndarray | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "RGBCamera":
            state = gs.state
            if state and "rgba" in state:
                rgba = state["rgba"]
                if rgba is not None and len(rgba) > 0:
                    # rgba from annotator is often uint8 [H, W, 4]
                    rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
                    return rgb[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
            return None
    return None


def _extract_semantic(drone) -> dict | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "SemanticSegmentationCamera":
            state = gs.state
            if state and "semantic_segmentation" in state:
                return state["semantic_segmentation"]
    return None


def _extract_depth(drone) -> np.ndarray | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "DepthCamera":
            state = gs.state
            if state and "depth" in state:
                raw = state["depth"]
                if isinstance(raw, dict):
                    raw = raw.get("data")
                if raw is None:
                    return None
                d = np.asarray(raw, dtype=np.float32)
                if d.ndim == 3 and d.shape[2] == 1:
                    d = d[:, :, 0]
                return d
    return None


def _get_camera_world_pose(drone):
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "DepthCamera":
            cam = gs._camera
            if cam is None:
                return None
            try:
                return cam.get_world_pose(camera_axes="world")
            except Exception:
                return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    debug_dir = os.path.abspath(DEBUG_DIR)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"[E2E Test] Config:")
    print(f"  Scene      : {SCENE_PATH}")
    print(f"  VLM URL    : {VLM_URL}")
    print(f"  Max steps  : {MAX_STEPS}")
    print(f"  Survey alt : {SURVEY_ALT}m")
    print(f"  Debug dir  : {debug_dir}")
    print()

    timeline = omni.timeline.get_timeline_interface()

    pg = PegasusInterface()
    # Use Pegasus' own method to create the World (handles the singleton correctly)
    pg.initialize_world()
    world = pg.world

    # Isaac Sim 5.x: World.__init__ may skip _scene creation due to singleton
    # guards. Ensure _scene exists before anything accesses world.scene.
    if not hasattr(world, '_scene') or world._scene is None:
        from isaacsim.core.api.scenes.scene import Scene
        world._scene = Scene()
        world._task_scene_built = False
        world._current_tasks = dict()
        from isaacsim.core.api.loggers import DataLogger
        world._data_logger = DataLogger()

    # ---- Load or generate scene -------------------------------------------
    if SCENE_PATH and os.path.exists(SCENE_PATH):
        print(f"[E2E Test] Loading scene from {SCENE_PATH}")
        omni.usd.get_context().open_stage(SCENE_PATH)
    else:
        print("[E2E Test] Generating urban disaster scene ...")
        generate_scene()

    # ---- Spawn drone at survey altitude -----------------------------------
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=[0.0, 0.0, SURVEY_ALT],
    )

    # ---- Set up orchestrator with DEBUG MODE ON ---------------------------
    K = make_intrinsics_from_fov(640, 480, hfov_deg=70.0)
    orchestrator = OrchestratorBridge(
        yolo_weights=YOLO_WEIGHTS,
        vlm_url=VLM_URL,
        debug_dir=debug_dir,
    )

    world.reset()  # Initialise physics + Scene before playing
    timeline.play()

    # ---- Simulation loop --------------------------------------------------
    warmup_steps = 120
    step = 0
    frames_processed = 0
    total_hazards = 0
    prev_time = time.perf_counter()

    print(f"[E2E Test] Running {MAX_STEPS} steps (+ {warmup_steps} warmup) at alt={SURVEY_ALT}m")
    print(f"[E2E Test] Ctrl-C to stop early.\n")

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            if step < warmup_steps:
                if step % 30 == 0:
                    print(f"[E2E Test] Warming up... step {step}/{warmup_steps}")
                continue

            # Sensor extraction
            rgb = _extract_rgb(drone)
            seg_raw = _extract_semantic(drone)
            depth = _extract_depth(drone)

            if rgb is None or depth is None:
                if step % 30 == 0:
                    print(f"[{step}] rgb is None: {rgb is None}, depth is None: {depth is None}")
                continue

            thermal = generate_synthetic_thermal(seg_raw) if seg_raw else None
            cam_pose = _get_camera_world_pose(drone)
            pos = imu_backend.latest_state["position"]

            # Run the full orchestrator pipeline (debug saves happen inside)
            frame_result = orchestrator.process_frame(
                rgb_frame=rgb,
                thermal_frame=thermal,
                depth_map=depth,
                camera_intrinsics=K,
                camera_world_pose=cam_pose,
                frame_idx=step,
                drone_position=pos,
            )

            frames_processed += 1

            if frame_result and frame_result.get("hazards"):
                n = len(frame_result["hazards"])
                total_hazards += n
                if frames_processed % 20 == 0:
                    print(f"[E2E Test] step={step} | {n} active hazard(s) | pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
            elif frames_processed % 50 == 0:
                print(f"[E2E Test] step={step} | no detections | pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

            if step >= warmup_steps + MAX_STEPS:
                print(f"\n[E2E Test] Reached max steps ({MAX_STEPS})")
                break

    except KeyboardInterrupt:
        print("\n[E2E Test] Interrupted by user.")

    timeline.stop()

    # ---- Summary ----------------------------------------------------------
    elapsed = time.perf_counter() - prev_time
    debug_files = []
    frames_dir = os.path.join(debug_dir, "frames")
    if os.path.isdir(frames_dir):
        debug_files = os.listdir(frames_dir)

    print("\n" + "=" * 60)
    print("  ResQ-AI Headless E2E Test - Summary")
    print("=" * 60)
    print(f"  Sim steps run    : {step}")
    print(f"  Frames processed : {frames_processed}")
    print(f"  Total hazards    : {total_hazards}")
    print(f"  Wall time        : {elapsed:.1f}s")
    print(f"  Debug dir        : {debug_dir}")
    print(f"  Debug files      : {len(debug_files)}")

    # Breakdown by file type
    rgb_count = sum(1 for f in debug_files if f.endswith("_rgb.jpg"))
    thermal_count = sum(1 for f in debug_files if f.endswith("_thermal.jpg"))
    yolo_count = sum(1 for f in debug_files if f.endswith("_yolo.json"))
    cosmos_count = sum(1 for f in debug_files if f.endswith("_cosmos_prompt.json"))
    seg_count = sum(1 for f in debug_files if f.endswith("_seg.jpg"))
    vlm_count = sum(1 for f in debug_files if f.startswith("vlm_response"))
    print(f"    RGB frames     : {rgb_count}")
    print(f"    Thermal frames : {thermal_count}")
    print(f"    Seg overlays   : {seg_count}")
    print(f"    YOLO JSONs     : {yolo_count}")
    print(f"    Cosmos prompts : {cosmos_count}")
    print(f"    VLM responses  : {vlm_count}")
    print("=" * 60)

    simulation_app.close()
    print("[E2E Test] Done.")


if __name__ == "__main__":
    main()
