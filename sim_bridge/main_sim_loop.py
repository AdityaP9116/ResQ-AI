#!/usr/bin/env python3
"""ResQ-AI end-to-end simulation loop.

Ties together every component of the sim_bridge:

1. Initialises ``SimulationApp`` and loads the urban disaster scene.
2. Spawns the Pegasus Iris drone with RGB / Semantic / Depth cameras + IMU.
3. On each render step, extracts live sensor matrices and pipes them through
   the orchestrator (YOLO → 3-D projection → Cosmos Reason 2 JSON prompt).

Usage::

    /isaac-sim/python.sh sim_bridge/main_sim_loop.py
    /isaac-sim/python.sh sim_bridge/main_sim_loop.py --headless
    /isaac-sim/python.sh sim_bridge/main_sim_loop.py --scene /tmp/resqai_urban_disaster.usda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so orchestrator/sim_bridge are importable
# regardless of how this script is invoked (python3, python.bat, pytest, etc.)
# ---------------------------------------------------------------------------
_PROJECT_ROOT_EARLY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_EARLY not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_EARLY)

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must precede all Omniverse imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI main simulation loop")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Path to a .usda scene. If omitted, generates one on-the-fly.",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/analyze",
        help="Cosmos Reason 2 / VLM endpoint.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLO .pt weights.  Defaults to Phase1 best.pt → yolov8n.pt fallback.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N physics steps (0 = run until window closes / Ctrl-C).",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Livestream mode: 0=off, 1=native (Omni Streaming), 2=WebRTC (browser at :8211/streaming/webrtc-demo/)",
    )
    return parser.parse_args()


_args = _parse_args()
simulation_app = SimulationApp({
    "headless": _args.headless,
    "livestream": _args.livestream,
})

# ---------------------------------------------------------------------------
# Omniverse / Pegasus imports (valid only after SimulationApp)
# ---------------------------------------------------------------------------
import carb
import omni.timeline
import omni.usd
from omni.isaac.core.world import World

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# sim_bridge modules
from sim_bridge.generate_urban_scene import generate_scene
from sim_bridge.spawn_drone import spawn_resqai_drone
from sim_bridge.thermal_sim import generate_synthetic_thermal
from sim_bridge.projection_utils import (
    batch_pixel_to_3d_world,
    make_intrinsics_from_fov,
    pixel_to_3d_world,
)

# Orchestrator (refactored)
from orchestrator.orchestrator_bridge import (
    OrchestratorBridge,
    build_cosmos_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════
# Sensor frame extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_rgb(drone) -> np.ndarray | None:
    """Return the latest RGB frame as an ``(H, W, 3)`` uint8 BGR array
    suitable for OpenCV / YOLO, or ``None`` if not ready."""
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "MonocularCamera":
            cam = gs._camera
            if cam is None:
                return None
            try:
                rgba = cam.get_rgba()
                if rgba is None or rgba.size == 0:
                    return None
                rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
                return rgb[:, :, ::-1].copy()  # RGBA→BGR for OpenCV
            except Exception:
                return None
    return None


def _extract_semantic(drone) -> dict | None:
    """Return the raw Replicator semantic segmentation annotator output."""
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "SemanticSegmentationCamera":
            state = gs.state
            if state and "semantic_segmentation" in state:
                return state["semantic_segmentation"]
    return None


def _extract_depth(drone) -> np.ndarray | None:
    """Return the depth map as a float ``(H, W)`` array (metres)."""
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
    """Return ``(position, quaternion_wxyz)`` for the depth camera."""
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
    timeline = omni.timeline.get_timeline_interface()

    # ---- Pegasus world setup ----------------------------------------------
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    # ---- Load or generate the urban disaster scene ------------------------
    if _args.scene and os.path.exists(_args.scene):
        print(f"[ResQ-AI] Loading pre-built scene from {_args.scene}")
        omni.usd.get_context().open_stage(_args.scene)
    else:
        print("[ResQ-AI] Generating urban disaster scene on-the-fly …")
        generate_scene()

    # ---- Spawn the drone --------------------------------------------------
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=[0.0, 0.0, 15.0],
    )

    # ---- Camera intrinsics (matches MonocularCamera / ReplicatorCamera) ---
    K = make_intrinsics_from_fov(640, 480, hfov_deg=70.0)

    # ---- Make the drone body kinematic so we can precisely control its translation ---
    from pxr import UsdPhysics, Gf
    stage = omni.usd.get_context().get_stage()
    body_prim = stage.GetPrimAtPath("/World/ResQDrone/body")
    if body_prim.IsValid():
        rb = UsdPhysics.RigidBodyAPI(body_prim)
        rb.CreateKinematicEnabledAttr(True)
        print("[ResQ-AI] Configured Drone rigid body as Kinematic for AI Control.")

    root_prim = stage.GetPrimAtPath("/World/ResQDrone")

    # ---- Orchestrator bridge (YOLO + logic gates) -------------------------
    orchestrator = OrchestratorBridge(
        yolo_weights=_args.yolo_weights,
        vlm_url=_args.vlm_url,
    )

    world.reset()
    timeline.play()

    print("[ResQ-AI] Simulation running.  Press Ctrl-C or close window to exit.\n")

    step = 0
    warmup_steps = 120  # let cameras warm up before reading

    flight_report: list[dict] = []
    
    # Flight Controller State
    current_target_wp = np.array([0.0, 0.0, 15.0]) # Default hover target
    drone_speed = 0.2  # meters per frame (approx 12 m/s at 60Hz physics)

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            if step < warmup_steps:
                continue

            # ---- Extract sensor matrices ----------------------------------
            rgb = _extract_rgb(drone)
            seg_raw = _extract_semantic(drone)
            depth = _extract_depth(drone)

            if rgb is None or depth is None:
                continue

            # ---- Generate synthetic thermal from semantics ----------------
            if seg_raw is not None:
                thermal = generate_synthetic_thermal(seg_raw)
            else:
                thermal = None

            # ---- Get camera world pose for projection ---------------------
            cam_pose = _get_camera_world_pose(drone)

            # ---- Run orchestrator (YOLO → 3D → Cosmos JSON) --------
            frame_result = orchestrator.process_frame(
                rgb_frame=rgb,
                thermal_frame=thermal,
                depth_map=depth,
                camera_intrinsics=K,
                camera_world_pose=cam_pose,
                frame_idx=step,
                drone_position=imu_backend.latest_state["position"],
            )

            if frame_result and frame_result.get("hazards"):
                flight_report.extend(frame_result["hazards"])

                # Print waypoint if Cosmos returned one
                wp = frame_result.get("target_waypoint")
                if wp:
                    print(f"[Step {step}] Cosmos waypoint: {wp}  reason: {frame_result.get('reasoning', '')[:80]}")
                    current_target_wp = np.array(wp)

                # Print Cosmos prompt for this frame
                if frame_result.get("cosmos_prompt"):
                    print(f"\n[Step {step}] Cosmos Reason 2 prompt:")
                    print(frame_result["cosmos_prompt"][:500])
                    print("…\n" if len(frame_result["cosmos_prompt"]) > 500 else "\n")

            # ---- Physicalize Cosmos Flight Command (Kinematic Movement) ----
            current_pos = imu_backend.latest_state["position"]
            direction = current_target_wp - current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.5: # Deadband to prevent jitter when target reached
                velocity_vector = (direction / distance) * drone_speed
                new_pos = current_pos + velocity_vector
                
                # Apply the translation to the USD Node directly
                if root_prim.IsValid():
                    translate_attr = root_prim.GetAttribute("xformOp:translate")
                    if translate_attr and translate_attr.IsValid():
                        translate_attr.Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
                if body_prim.IsValid():
                    body_translate_attr = body_prim.GetAttribute("xformOp:translate")
                    if body_translate_attr and body_translate_attr.IsValid():
                        body_translate_attr.Set(Gf.Vec3d(0.0, 0.0, 0.0)) # keep body centered on root

            if _args.max_steps and step >= _args.max_steps + warmup_steps:
                print(f"[ResQ-AI] Reached --max-steps={_args.max_steps}, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[ResQ-AI] Interrupted by user.")

    # ---- Save flight report -----------------------------------------------
    timeline.stop()

    report_path = os.path.join(os.path.dirname(__file__), "..", "orchestrator", "Flight_Report.json")
    with open(report_path, "w") as f:
        json.dump(flight_report, f, indent=4)
    print(f"[ResQ-AI] Flight report saved → {report_path}  ({len(flight_report)} hazard entries)")

    simulation_app.close()
    print("[ResQ-AI] Simulation finished.")


if __name__ == "__main__":
    main()
