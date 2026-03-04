#!/usr/bin/env python3
"""ResQ-AI autonomous demo flight — Sense-Think-Act loop with Cosmos-driven waypoints.

Master demo script that:
1. Generates the urban disaster scene with corrected OmniPBR materials
2. Spawns the Pegasus Iris drone with RGB/Semantic/Depth cameras
3. Runs SURVEY → INVESTIGATE → RETURN state machine
4. Uses target_waypoint from Cosmos Reason 2 / VLM to drive the drone
5. Records annotated video with YOLO boxes, thermal, HUD and flight report

Usage::

    # Start VLM server first (in another terminal):
    #   Mock:  python orchestrator/vlm_server.py
    #   NIM:   NVIDIA_API_KEY=nvapi-... python orchestrator/vlm_server.py --backend nim
    #   vLLM:  python orchestrator/vlm_server.py --backend vllm

    # Run demo (requires Isaac Sim):
    /isaac-sim/python.sh sim_bridge/demo_flight.py
    /isaac-sim/python.sh sim_bridge/demo_flight.py --headless --record
    /isaac-sim/python.sh sim_bridge/demo_flight.py --scene /tmp/resqai_urban_disaster.usda --record
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so orchestrator/sim_bridge are importable
# regardless of how this script is invoked (python3, python.bat, pytest, etc.)
# ---------------------------------------------------------------------------
import os as _os, sys as _sys
_PROJECT_ROOT_EARLY = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_EARLY not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT_EARLY)

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must precede ALL third-party / Omniverse imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI autonomous demo flight")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "resqai_urban_disaster.usda"),
        help="Path to .usda scene (default: resqai_urban_disaster.usda).",
    )
    parser.add_argument(
        "--vlm-url",
        type=str,
        default="http://localhost:8000/analyze",
        help="VLM / Cosmos Reason 2 endpoint.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record annotated video to /tmp/resqai_demo.mp4",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after N steps (0 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--survey-alt",
        type=float,
        default=45.0,
        help="Survey altitude (m).",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Livestream mode: 0=off, 1=native (Omni Streaming), 2=WebRTC (browser at :8211/streaming/webrtc-demo/)",
    )
    args, _remaining = parser.parse_known_args()
    return args, _remaining


_args, _remaining_argv = _parse_args()
sys.argv = [sys.argv[0]] + _remaining_argv
simulation_app = SimulationApp({
    "headless": _args.headless,
    "livestream": _args.livestream,
})

# ---------------------------------------------------------------------------
# Third-party + Omniverse imports (valid only AFTER SimulationApp)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import cv2
import numpy as np

import carb
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

# Global for position control (set in main)
imu_backend_ref: list = [None]


# ═══════════════════════════════════════════════════════════════════════════
# Flight state machine
# ═══════════════════════════════════════════════════════════════════════════

class FlightState(Enum):
    SURVEY = "SURVEY"
    INVESTIGATE = "INVESTIGATE"
    RETURN = "RETURN"


SURVEY_POSITION = [0.0, 0.0, 45.0]  # [x, y, z] m
WAYPOINT_REACHED_DIST = 5.0  # m
INVESTIGATE_HOVER_SEC = 3.0
FLIGHT_SPEED = 8.0  # m/s for position lerp
MAX_INVESTIGATIONS = 5  # max hazards to investigate before returning


# ═══════════════════════════════════════════════════════════════════════════
# Sensor extraction (from main_sim_loop)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_rgb(drone) -> np.ndarray | None:
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
                return rgb[:, :, ::-1].copy()
            except Exception:
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
# Drone position control (lerp toward target)
# ═══════════════════════════════════════════════════════════════════════════

def _set_drone_position(drone, target: list[float], dt: float) -> float:
    """Move drone toward target via USD prim transform. Returns distance to target."""
    stage = omni.usd.get_context().get_stage()
    if stage is None or imu_backend_ref[0] is None:
        return float("inf")

    # Try common Pegasus Multirotor prim paths
    for path in ["/World/ResQDrone", "/World/ResQDrone/body", "/World/ResQDrone/base_link"]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue

        pos = np.array(imu_backend_ref[0].latest_state["position"], dtype=np.float64)
        err = np.array(target, dtype=np.float64) - pos
        dist = float(np.linalg.norm(err))

        if dist < 0.1:
            return dist

        step = min(FLIGHT_SPEED * dt, dist)
        new_pos = pos + err * (step / dist) if dist > 0 else pos

        ops = xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
                return float(np.linalg.norm(np.array(target) - new_pos))

        # No translate op — add one
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))
        return float(np.linalg.norm(np.array(target) - new_pos))

    return float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# Video recording and annotation
# ═══════════════════════════════════════════════════════════════════════════

def _draw_annotated_frame(
    rgb: np.ndarray,
    frame_result: dict | None,
    state: FlightState,
    target_waypoint: list[float] | None,
    reasoning: str,
    drone_pos: np.ndarray,
    thermal: np.ndarray | None,
) -> np.ndarray:
    """Draw YOLO boxes, thermal thumbnail, state, reasoning, HUD on frame."""
    out = rgb.copy()

    # YOLO boxes
    if frame_result and frame_result.get("hazards"):
        for h in frame_result["hazards"]:
            bbox = h.get("bbox")
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{h.get('class_name', '?')} {h.get('confidence', 0):.2f}"
                cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Thermal thumbnail (top-right)
    if thermal is not None and thermal.size > 0:
        th_small = cv2.resize(thermal, (120, 90))
        th_color = cv2.applyColorMap(th_small, cv2.COLORMAP_JET)
        out[5:95, out.shape[1] - 125 : out.shape[1] - 5] = th_color

    # State label (top-left)
    cv2.putText(out, f"[{state.value}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, f"[{state.value}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 1)

    # Target waypoint
    if target_waypoint:
        wp_str = f"WP: [{target_waypoint[0]:.1f}, {target_waypoint[1]:.1f}, {target_waypoint[2]:.1f}]"
        cv2.putText(out, wp_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Reasoning (bottom, wrapped)
    if reasoning:
        for i, line in enumerate(reasoning[:80].split("\n")[:2]):
            cv2.putText(out, line[:60], (10, out.shape[0] - 50 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Drone position HUD
    hud = f"pos=[{drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}]m"
    cv2.putText(out, hud, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    global imu_backend_ref
    timeline = omni.timeline.get_timeline_interface()
    imu_backend_ref[0] = None  # Will be set after spawn

    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    # Load or generate scene
    if _args.scene and os.path.exists(_args.scene):
        print(f"[ResQ-AI] Loading scene from {_args.scene}")
        omni.usd.get_context().open_stage(_args.scene)
    else:
        print("[ResQ-AI] Generating urban disaster scene …")
        generate_scene()

    # Spawn drone at survey altitude
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=[0.0, 0.0, _args.survey_alt],
    )
    imu_backend_ref[0] = imu_backend

    K = make_intrinsics_from_fov(640, 480, hfov_deg=70.0)
    orchestrator = OrchestratorBridge(
        yolo_weights=_args.yolo_weights,
        vlm_url=_args.vlm_url,
    )

    world.reset()
    timeline.play()

    # Export the freshly-generated scene so it can be reloaded later
    output_usda = "/tmp/resqai_urban_disaster.usda"
    if not (_args.scene and os.path.exists(_args.scene)):
        stage = omni.usd.get_context().get_stage()
        if stage:
            stage.GetRootLayer().Export(output_usda)
            print(f"[ResQ-AI] Scene exported → {output_usda}")

    # State machine
    state = FlightState.SURVEY
    target_waypoint: list[float] | None = list(SURVEY_POSITION)
    reasoning = ""
    investigate_start_time: float | None = None
    investigations_done: int = 0
    last_dt = 1.0 / 60.0

    # Outputs
    flight_path: list[dict] = []
    cosmos_decisions: list[dict] = []
    hazards_accumulated: list[dict] = []
    video_writer: cv2.VideoWriter | None = None

    if _args.record:
        video_path = "/tmp/resqai_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        print(f"[ResQ-AI] Recording to {video_path}")

    step = 0
    warmup_steps = 120
    prev_time = time.perf_counter()

    print("[ResQ-AI] Autonomous demo running. Ctrl-C to stop.\n")
    print(f"[SURVEY] alt={_args.survey_alt}m | scanning...")

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1
            curr_time = time.perf_counter()
            last_dt = curr_time - prev_time
            prev_time = curr_time

            if step < warmup_steps:
                continue

            pos = imu_backend.latest_state["position"]
            flight_path.append({"step": step, "position": pos.tolist()})

            # Extract sensors
            rgb = _extract_rgb(drone)
            seg_raw = _extract_semantic(drone)
            depth = _extract_depth(drone)
            if rgb is None or depth is None:
                continue

            thermal = generate_synthetic_thermal(seg_raw) if seg_raw else None
            cam_pose = _get_camera_world_pose(drone)

            # Run orchestrator
            frame_result = orchestrator.process_frame(
                rgb_frame=rgb,
                thermal_frame=thermal,
                depth_map=depth,
                camera_intrinsics=K,
                camera_world_pose=cam_pose,
                frame_idx=step,
                drone_position=pos,
            )

            # Update target from Cosmos / VLM
            n_detections = 0
            if frame_result:
                if frame_result.get("hazards"):
                    hazards_accumulated.extend(frame_result["hazards"])
                    n_detections = len(frame_result["hazards"])
                wp = frame_result.get("target_waypoint")
                if wp is not None and isinstance(wp, (list, tuple)) and len(wp) >= 3:
                    new_wp = [float(wp[0]), float(wp[1]), float(wp[2])]
                    new_reasoning = frame_result.get("reasoning", "")
                    cosmos_decisions.append({
                        "step": step,
                        "waypoint": new_wp,
                        "reasoning": new_reasoning,
                    })
                    print(f'[COSMOS] "{new_reasoning[:80]}"')
                    if state == FlightState.SURVEY:
                        target_waypoint = new_wp
                        reasoning = new_reasoning
                        state = FlightState.INVESTIGATE
                        investigate_start_time = None
                        print(f"[INVESTIGATE] target={target_waypoint} | hazard=detected")
                    elif state == FlightState.INVESTIGATE and investigate_start_time is not None:
                        # New waypoint while hovering — update target for next investigation
                        target_waypoint = new_wp
                        reasoning = new_reasoning

            # State machine logic
            dist_to_target = float(np.linalg.norm(np.array(target_waypoint) - pos))

            if state == FlightState.SURVEY:
                target_waypoint = list(SURVEY_POSITION)
                _set_drone_position(drone, target_waypoint, last_dt)
                if step % 100 == 0:
                    print(f"[SURVEY] alt={pos[2]:.1f}m | detections={n_detections} | scanning...")

            elif state == FlightState.INVESTIGATE:
                _set_drone_position(drone, target_waypoint, last_dt)
                if dist_to_target < WAYPOINT_REACHED_DIST:
                    if investigate_start_time is None:
                        investigate_start_time = curr_time
                        print(f"[INVESTIGATE] Arrived at waypoint, hovering for {INVESTIGATE_HOVER_SEC}s...")
                    elif curr_time - investigate_start_time >= INVESTIGATE_HOVER_SEC:
                        investigations_done += 1
                        investigate_start_time = None
                        if investigations_done >= MAX_INVESTIGATIONS:
                            state = FlightState.RETURN
                            target_waypoint = list(SURVEY_POSITION)
                            print(f"[RETURN] Max investigations reached, flying back to survey position")
                        else:
                            # Check if Cosmos gave us a new waypoint while hovering
                            # If not, return to survey
                            state = FlightState.RETURN
                            target_waypoint = list(SURVEY_POSITION)
                            print(f"[RETURN] Investigation {investigations_done} complete, returning to base")
                if step % 50 == 0 and target_waypoint:
                    print(f"[INVESTIGATE] target=[{target_waypoint[0]:.1f}, {target_waypoint[1]:.1f}, {target_waypoint[2]:.1f}] | dist={dist_to_target:.1f}m")

            elif state == FlightState.RETURN:
                _set_drone_position(drone, target_waypoint, last_dt)
                if dist_to_target < WAYPOINT_REACHED_DIST:
                    print("[ResQ-AI] Mission complete.")
                    break
                if step % 50 == 0:
                    print(f"[RETURN] dist={dist_to_target:.1f}m")

            # Record frame
            if video_writer and rgb is not None:
                ann = _draw_annotated_frame(
                    rgb, frame_result, state, target_waypoint, reasoning, pos, thermal
                )
                video_writer.write(ann)

            if _args.max_steps and step >= _args.max_steps + warmup_steps:
                print(f"[ResQ-AI] Reached --max-steps={_args.max_steps}")
                break

    except KeyboardInterrupt:
        print("\n[ResQ-AI] Interrupted by user.")

    timeline.stop()

    # Save outputs
    report_dir = os.path.join(os.path.dirname(__file__), "..", "orchestrator")
    report_path = os.path.join(report_dir, "Flight_Report.json")
    report = {
        "flight_path": flight_path,
        "cosmos_decisions": cosmos_decisions,
        "hazards": hazards_accumulated,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ResQ-AI] Flight report → {report_path}")

    if video_writer:
        video_writer.release()
        print("[ResQ-AI] Video saved → /tmp/resqai_demo.mp4")

    simulation_app.close()
    print("[ResQ-AI] Demo finished.")


if __name__ == "__main__":
    main()
