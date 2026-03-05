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
# Ensure project root is on sys.path (makes 'orchestrator', 'sim_bridge', etc.
# importable regardless of how this script is invoked — python.bat, python3, or
# pytest — matching the Windows behviour of C:\isaacsim\python.bat)
# ---------------------------------------------------------------------------
_PROJECT_ROOT_EARLY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_EARLY not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_EARLY)

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must precede ALL third-party / Omniverse imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

# RESQAI_LIVESTREAM: 0=none (default), 2=WebRTC (view in browser via SSH tunnel)
#   To stream: RESQAI_LIVESTREAM=2 bash scripts/run_e2e_test.sh --stream
#   SSH tunnel: ssh -L 8211:localhost:8211 <brev-instance>
#   Then open:  http://localhost:8211/streaming/webrtc-demo/
_LIVESTREAM = int(os.environ.get("RESQAI_LIVESTREAM", "0"))

_sim_config = {
    "headless": True,
    "anti_aliasing": 0,        # 0=disabled (no DLSS shader stall on first run)
    "renderer": "RaytracedLighting",
    "width": 960,
    "height": 540,
    "sync_loads": True,
}
if _LIVESTREAM:
    _sim_config["livestream"] = _LIVESTREAM

simulation_app = SimulationApp(_sim_config)

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

from sim_bridge.spawn_drone import spawn_resqai_drone
from sim_bridge.thermal_sim import generate_synthetic_thermal, generate_thermal_from_rgb
from sim_bridge.projection_utils import make_intrinsics_from_fov
from sim_bridge.drone_controller import DroneController

from orchestrator.orchestrator_bridge import OrchestratorBridge

# --- New subsystem imports (Prompts 1-5) ----------------------------------
from sim_bridge.yolo_detector import DualYOLODetector
from sim_bridge.thermal_processor import ThermalProcessor
from sim_bridge.civilian_tracker import CivilianTracker
from sim_bridge.report_generator import ReportGenerator

import random
import math


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
SURVEY_ALT = float(os.environ.get("RESQAI_SURVEY_ALT", "110.0"))

# ---- Optional zone order override ----
# If set, drone visits these zones in this exact order instead of alphabetical.
# Example: RESQAI_ZONE_ORDER="FZ_2,FZ_1,FZ_4,FZ_0,FZ_3"
ZONE_ORDER_STR = os.environ.get("RESQAI_ZONE_ORDER", "")
CUSTOM_FOCUS_DURATION = int(os.environ.get("RESQAI_FOCUS_DURATION", "0"))  # 0 = use default
CUSTOM_COOLDOWN_STEPS = int(os.environ.get("RESQAI_COOLDOWN_STEPS", "0"))  # 0 = use default


# ---------------------------------------------------------------------------
# Sensor extraction helpers (same as demo_flight.py)
# ---------------------------------------------------------------------------

def _unwrap_annotator_data(raw):
    """Replicator annotators return either a raw ndarray or {"data": array, "info": {...}}.
    This helper normalises both formats to a plain ndarray (or None)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get("data")
    return raw


def _extract_rgb(drone, _debug_once=[True]) -> np.ndarray | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "RGBCamera":
            state = gs.state
            if not state:
                return None
            rgba = _unwrap_annotator_data(state.get("rgba"))
            if rgba is None or rgba.size == 0:
                return None
            # Debug: print dtype/range once
            if _debug_once[0]:
                _debug_once[0] = False
                print(f"[RGB debug] dtype={rgba.dtype} shape={rgba.shape} min={rgba.min():.4f} max={rgba.max():.4f}")
            # rgba may be float32 [0,1] or uint8 [0,255] depending on annotator version
            rgba_arr = np.asarray(rgba)
            if rgba_arr.dtype in (np.float32, np.float64):
                rgba_arr = np.clip(rgba_arr * 255.0, 0, 255).astype(np.uint8)
            else:
                rgba_arr = rgba_arr.astype(np.uint8)
            # Drop alpha, convert RGB→BGR for OpenCV
            rgb = rgba_arr[:, :, :3]
            return rgb[:, :, ::-1].copy()
    return None


def _extract_semantic(drone) -> dict | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "SemanticSegmentationCamera":
            state = gs.state
            if not state:
                return None
            raw = state.get("semantic_segmentation")
            if isinstance(raw, dict) and "data" in raw:
                return raw  # pass the full dict downstream (contains labels too)
            return raw if raw is not None else None
    return None


def _extract_depth(drone) -> np.ndarray | None:
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "DepthCamera":
            state = gs.state
            if not state:
                return None
            d = _unwrap_annotator_data(state.get("depth"))
            if d is None:
                return None
            d = np.asarray(d, dtype=np.float32)
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
# Flow fire integration (real particle fires via omni.flowusd)
# ═══════════════════════════════════════════════════════════════════════════

# Fire zone coords from the USDA
_FIRE_ZONES = [
    {"name": "FZ_0", "pos": (-94.8, -16.5, 0.15), "radius": 7.3,  "intensity": 0.72},
    {"name": "FZ_1", "pos": (-98.6,  46.5, 0.15), "radius": 5.8,  "intensity": 0.77},
    {"name": "FZ_2", "pos": ( -1.8,  46.5, 0.15), "radius": 7.3,  "intensity": 0.99},
    {"name": "FZ_3", "pos": ( 87.6, -16.5, 0.15), "radius": 4.2,  "intensity": 0.89},
    {"name": "FZ_4", "pos": ( 79.7,  46.5, 0.15), "radius": 6.5,  "intensity": 0.91},
]

_FLOW_AVAILABLE = False


def _enable_flow_extension() -> bool:
    """Enable omni.flowusd + omni.usd.schema.flow for real particle fire/smoke.
    Returns True if Flow is available."""
    global _FLOW_AVAILABLE
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        # Enable schema first, then the main Flow USD extension
        mgr.set_extension_enabled_immediate("omni.usd.schema.flow", True)
        result = mgr.set_extension_enabled_immediate("omni.flowusd", True)
        if result:
            print("[Fire] omni.flowusd extension enabled successfully!")
            _FLOW_AVAILABLE = True
            return True
        else:
            print("[Fire] WARNING: omni.flowusd could not be enabled")
            return False
    except Exception as e:
        print(f"[Fire] Flow extension load failed: {e}")
        return False


def _create_flow_fire_at_zone(stage, zone_idx: int) -> bool:
    """Create a single Flow particle fire at the given zone index.
    Called initially for zone 0 only; FireManager creates fires at
    other zones as they spread.
    Returns True on success."""
    if not _FLOW_AVAILABLE:
        return False
    if zone_idx < 0 or zone_idx >= len(_FIRE_ZONES):
        return False

    import omni.kit.commands
    fz = _FIRE_ZONES[zone_idx]
    name = fz["name"]
    pos = fz["pos"]

    parent_path = "/World/FlowFires"
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    fire_path = f"{parent_path}/{name}"
    xform = UsdGeom.Xform.Define(stage, fire_path)
    xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

    try:
        omni.kit.commands.execute(
            "FlowCreatePresets",
            preset_name="Fire",
            paths=[fire_path],
            create_copy=True,
            layer=-1,
        )
        print(f"[Fire] Created initial Flow fire at {name} ({pos[0]:.1f}, {pos[1]:.1f})")
        return True
    except Exception as e:
        print(f"[Fire] FlowCreatePresets failed for {name}: {e}")
        return False


def _create_flow_fires(stage) -> int:
    """Prepare for fires — the actual fire creation is handled by FireManager.
    FireManager._ignite_zone() calls _add_flow_effect() internally,
    so we only set up the parent Xform here.
    Returns 0 (fires are created by FireManager after IGNITE_DELAY)."""
    if not _FLOW_AVAILABLE:
        print("[Fire] Flow not available — skipping real fire creation")
        return 0

    # Ensure the parent Xform exists for any fires FireManager creates
    parent_path = "/World/FlowFires"
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Xform.Define(stage, parent_path)

    print("[Fire] Flow ready — FireManager will create fires after ignition delay")
    return 0


# ---- Smooth external camera state (module-level to persist across calls) ----
_ext_cam_pos = np.array([0.0, -40.0, 145.0])


def _setup_external_camera(stage) -> str:
    """Create a 3rd-person camera that orbits behind the drone.
    Returns the camera prim path."""
    cam_path = "/World/ExternalCamera"

    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(28.0)          # moderate zoom to see drone clearly
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 2000.0))

    # Position behind and above drone start — this gets updated every frame
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -40.0, 145.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, 0.0, 0.0, 0.0))  # identity — will be set per frame

    print(f"[Camera] External 3rd-person camera created at {cam_path}")
    return cam_path


def _update_external_camera(stage, cam_path: str, drone_pos: np.ndarray) -> None:
    """Smoothly follow a 3rd-person chase camera behind the drone."""
    global _ext_cam_pos
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        return
    try:
        # Chase camera: 30m behind, 15m above, slightly to the side
        target_cam_pos = np.array([
            drone_pos[0] + 8.0,
            drone_pos[1] - 30.0,
            drone_pos[2] + 15.0,
        ])

        # Smooth follow (cinematic lag)
        alpha = 0.06
        _ext_cam_pos = _ext_cam_pos + alpha * (target_cam_pos - _ext_cam_pos)

        # Write translate
        ta = cam_prim.GetAttribute("xformOp:translate")
        if ta and ta.IsValid():
            ta.Set(Gf.Vec3d(float(_ext_cam_pos[0]), float(_ext_cam_pos[1]), float(_ext_cam_pos[2])))

        # Orient to look at drone (slightly below drone to see it + ground)
        look_at = np.array([drone_pos[0], drone_pos[1], drone_pos[2] - 5.0])
        fwd = look_at - _ext_cam_pos
        fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / r_norm
        up = np.cross(right, fwd)
        R_cam = np.column_stack([right, up, -fwd])
        from scipy.spatial.transform import Rotation as _RotUpd
        q = _RotUpd.from_matrix(R_cam).as_quat()  # [x, y, z, w]
        oa = cam_prim.GetAttribute("xformOp:orient")
        if oa and oa.IsValid():
            oa.Set(Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    except Exception as _e:
        import sys as _s
        _s.__stdout__.write(f"[EXT_CAM_ERR] {_e}\n")
        _s.__stdout__.flush()


def _capture_external_frame(annotator) -> np.ndarray | None:
    """Capture one frame from the external camera render product."""
    try:
        data = annotator.get_data()
        if data is None:
            return None
        arr = np.asarray(data)
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]  # drop alpha
        return arr[:, :, ::-1].copy()  # RGB→BGR for OpenCV
    except Exception:
        return None


def _generate_fire_report(fire_mgr, yolo_fire_detections: list, debug_dir: str):
    """Generate a comprehensive fire output report as JSON.

    Includes: fire zones, spread events, YOLO fire confirmations,
    intensity, estimated containment time.
    """
    import json

    report = {
        "title": "ResQ-AI Fire Situation Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fire_zones": [],
        "yolo_fire_detections": yolo_fire_detections,
        "summary": {},
    }

    if fire_mgr is not None:
        fire_data = fire_mgr.get_fire_report()
        report["fire_zones"] = fire_data.get("active_fires", [])
        report["summary"] = {
            "total_active_fires": len(fire_data.get("active_fires", [])),
            "total_area_burning_m2": fire_data.get("total_area_burning_m2", 0),
            "spread_rate_m2_per_min": fire_data.get("spread_rate_m_per_min", 0),
            "estimated_containment_min": fire_data.get("estimated_containment_time_min", 0),
            "total_yolo_fire_detections": len(yolo_fire_detections),
            "yolo_confirmed_fires": sum(1 for d in yolo_fire_detections if d.get("confirmed")),
        }
    else:
        report["summary"] = {
            "total_active_fires": 0,
            "total_yolo_fire_detections": len(yolo_fire_detections),
        }

    # Write report
    report_path = os.path.join(debug_dir, "fire_situation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Fire Report] Written to {report_path}")

    # Also print summary
    print("\n" + "=" * 60)
    print("  FIRE SITUATION REPORT")
    print("=" * 60)
    for zone in report["fire_zones"]:
        print(f"  Zone {zone['zone']}: intensity={zone['intensity']}, "
              f"radius={zone['radius']}m, burning={zone['time_burning_seconds']:.0f}s")
    print(f"  Total area burning  : {report['summary'].get('total_area_burning_m2', 0):.1f} m²")
    print(f"  Spread rate         : {report['summary'].get('spread_rate_m2_per_min', 0):.2f} m²/min")
    print(f"  YOLO fire detections: {report['summary'].get('total_yolo_fire_detections', 0)}")
    print(f"  YOLO confirmed fires: {report['summary'].get('yolo_confirmed_fires', 0)}")
    print(f"  Est. containment    : {report['summary'].get('estimated_containment_min', 0):.1f} min")
    print("=" * 60)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Procedural scene builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_procedural_scene() -> None:
    """Add a colorful urban disaster scene using Isaac Sim VisualCuboid/VisualSphere.

    Called after stage open so real geometry is always present, regardless of
    whether the USDA asset references resolved (they won't on Linux — they use
    Windows-absolute paths).

    Layout matches the USDA scene:
      • Scene extent: X=-168..168, Y=-84..84
      • Fire zones at: (-95,-17), (-99,47), (-2,47), (88,-17), (80,47)
      • Buildings across the full scene grid
      • Up-axis = Z, metersPerUnit = 1
    """
    try:
        from omni.isaac.core.objects import VisualCuboid, VisualSphere
    except ImportError:
        print("[E2E Test] WARNING: omni.isaac.core.objects not available — skipping procedural scene")
        return

    print("[E2E Test] Building procedural urban disaster scene (full USDA layout) ...")

    # --- Ground plane (matches USDA extent) --------------------------------
    VisualCuboid(
        prim_path="/World/ProceduralGround",
        position=np.array([0.0, 0.0, -0.25]),
        scale=np.array([340.0, 170.0, 0.5]),
        color=np.array([0.35, 0.33, 0.30]),   # dark asphalt grey
    )

    # --- Standing buildings (spread across scene) -------------------------
    # These supplement the USDA brownstones — visible whether or not the ref
    # assets loaded.  Positioned along the major block grid.
    buildings = [
        # Block NE quadrant
        ([70.0,  25.0, 8.0],  [15.0, 30.0, 16.0], [0.31, 0.28, 0.24]),
        ([50.0,  25.0, 10.0], [12.0, 28.0, 20.0], [0.28, 0.26, 0.23]),
        ([90.0,  25.0, 7.5],  [14.0, 25.0, 15.0], [0.34, 0.30, 0.26]),
        # Block NW quadrant
        ([-70.0, 25.0, 9.0],  [15.0, 28.0, 18.0], [0.30, 0.27, 0.24]),
        ([-50.0, 30.0, 7.5],  [12.0, 25.0, 15.0], [0.35, 0.32, 0.27]),
        ([-90.0, 30.0, 8.5],  [14.0, 30.0, 17.0], [0.33, 0.29, 0.25]),
        # Block SE quadrant
        ([70.0,  -25.0, 9.5], [15.0, 30.0, 19.0], [0.29, 0.27, 0.23]),
        ([50.0,  -30.0, 8.0], [12.0, 28.0, 16.0], [0.32, 0.28, 0.24]),
        ([90.0,  -20.0, 7.0], [14.0, 25.0, 14.0], [0.36, 0.32, 0.28]),
        # Block SW quadrant
        ([-70.0, -25.0, 8.5], [15.0, 30.0, 17.0], [0.30, 0.27, 0.23]),
        ([-50.0, -25.0, 10.0],[12.0, 28.0, 20.0], [0.28, 0.26, 0.22]),
        ([-90.0, -25.0, 7.0], [14.0, 25.0, 14.0], [0.34, 0.30, 0.26]),
        # Centre area
        ([0.0,   0.0,  6.0],  [10.0, 10.0, 12.0], [0.29, 0.27, 0.23]),
        ([15.0,  10.0, 7.5],  [12.0, 12.0, 15.0], [0.31, 0.28, 0.24]),
        ([-15.0, -10.0, 8.0], [11.0, 11.0, 16.0], [0.33, 0.29, 0.25]),
        ([30.0,  0.0,  5.5],  [10.0, 15.0, 11.0], [0.35, 0.31, 0.27]),
        ([-30.0, 5.0,  6.5],  [10.0, 12.0, 13.0], [0.30, 0.28, 0.24]),
    ]
    for i, (pos, scale, color) in enumerate(buildings):
        VisualCuboid(
            prim_path=f"/World/Building_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # --- Collapsed building rubble piles ----------------------------------
    # Near fire-zone locations + in building areas
    collapses = [
        ([-95.0, -17.0, 1.0],  [18.0, 14.0, 2.0], [0.56, 0.48, 0.35]),   # near FZ_0
        ([-99.0,  46.0, 1.2],  [16.0, 12.0, 2.4], [0.52, 0.45, 0.33]),   # near FZ_1
        ([-2.0,   46.0, 0.8],  [14.0, 16.0, 1.6], [0.50, 0.44, 0.32]),   # near FZ_2
        ([88.0,  -17.0, 1.0],  [18.0, 12.0, 2.0], [0.54, 0.46, 0.34]),   # near FZ_3
        ([80.0,   46.0, 0.9],  [15.0, 14.0, 1.8], [0.51, 0.43, 0.31]),   # near FZ_4
        ([20.0,   0.0,  0.6],  [12.0, 10.0, 1.2], [0.48, 0.42, 0.30]),   # centre
        ([-40.0, -20.0, 0.7],  [14.0, 12.0, 1.4], [0.53, 0.46, 0.33]),   # extra
    ]
    for i, (pos, scale, color) in enumerate(collapses):
        VisualCuboid(
            prim_path=f"/World/Collapse_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # --- Fire zones: REMOVED (now using real Flow particle fires) ----------
    # Flow fires are created by _create_flow_fires() after scene load.

    # --- Flooded areas (blue/dark-blue puddles on ground) ----------------
    floods = [
        ([0.0,  -40.0, 0.07],  [50.0, 25.0, 0.14], [0.10, 0.18, 0.45]),   # south road
        ([40.0,   0.0, 0.06],  [30.0, 40.0, 0.12], [0.08, 0.15, 0.40]),   # east road
        ([-60.0,  0.0, 0.08],  [35.0, 30.0, 0.16], [0.12, 0.20, 0.48]),   # west road
        ([0.0,   40.0, 0.05],  [40.0, 20.0, 0.10], [0.09, 0.16, 0.42]),   # north road
    ]
    for i, (pos, scale, color) in enumerate(floods):
        VisualCuboid(
            prim_path=f"/World/FloodZone_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # --- Vehicles (coloured boxes on roads) -------------------------------
    vehicles = [
        ([25.0,  -35.0, 0.8],  [4.5, 2.2, 1.6], [0.70, 0.10, 0.10]),   # red car
        ([-20.0, -38.0, 0.8],  [4.5, 2.2, 1.6], [0.15, 0.30, 0.70]),   # blue car
        ([55.0,   5.0,  0.8],  [4.5, 2.2, 1.6], [0.80, 0.80, 0.80]),   # white car
        ([-55.0, -5.0,  0.8],  [4.5, 2.2, 1.6], [0.10, 0.10, 0.10]),   # black car
        ([10.0,  -42.0, 0.8],  [5.5, 2.5, 1.8], [0.90, 0.55, 0.00]),   # orange SUV
        ([-35.0,  30.0, 0.8],  [4.5, 2.2, 1.6], [0.30, 0.60, 0.15]),   # green car
        ([65.0,  -10.0, 0.8],  [6.0, 2.5, 2.0], [0.85, 0.80, 0.10]),   # yellow truck
        ([-80.0,  10.0, 0.8],  [4.5, 2.2, 1.6], [0.50, 0.10, 0.50]),   # purple car
        # Overturned/crashed vehicles (tilted or flat) near disaster zones
        ([-93.0, -12.0, 0.5],  [4.5, 2.2, 1.0], [0.60, 0.20, 0.05]),   # tipped near FZ_0
        ([85.0,  -20.0, 0.5],  [5.0, 2.5, 1.0], [0.40, 0.15, 0.10]),   # tipped near FZ_3
    ]
    for i, (pos, scale, color) in enumerate(vehicles):
        VisualCuboid(
            prim_path=f"/World/Vehicle_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # --- Survivors (bright, larger markers visible from 110 m) ---------------
    # Radius increased to 2.0 m so YOLO can detect them at bird's eye altitude.
    # Bright orange/red colors stand out against grey rubble.
    survivors = [
        ([-92.0, -15.0, 2.5], 2.0, [1.00, 0.25, 0.05]),   # near FZ_0
        ([-97.0,  48.0, 2.8], 2.0, [1.00, 0.15, 0.05]),   # near FZ_1
        ([0.0,    48.0, 2.2], 2.0, [1.00, 0.25, 0.05]),   # near FZ_2
        ([90.0,  -15.0, 2.5], 2.0, [1.00, 0.15, 0.05]),   # near FZ_3
        ([22.0,   1.0,  1.5], 2.0, [1.00, 0.25, 0.05]),   # centre rubble
        ([-38.0, -18.0, 1.8], 2.0, [1.00, 0.15, 0.05]),   # extra
    ]
    for i, (pos, radius, color) in enumerate(survivors):
        VisualSphere(
            prim_path=f"/World/Survivor_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            radius=float(radius),
            color=np.array(color),
        )
        # Add bright beacon/marker cube next to each survivor for aerial visibility
        VisualCuboid(
            prim_path=f"/World/SurvivorMarker_{i:02d}",
            position=np.array([pos[0], pos[1], pos[2] + 2.5], dtype=np.float64),
            size=1.5,
            color=np.array([1.0, 0.9, 0.0]),  # bright yellow
        )

    print("[E2E Test] Procedural scene: 17 buildings, 7 collapsed,")
    print("           4 flood zones, 10 vehicles, 6 survivors added.")
    print("           (Fire zones use real Flow particles — created separately)")

    # Lighting: The USDA scene already provides a sun + sky dome.
    # When SCENE_PATH is empty we add fallback lights; otherwise rely on USDA lights.
    if not SCENE_PATH:
        try:
            from pxr import UsdLux, UsdGeom as _UsdGeom
            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath("/World/SunLight").IsValid():
                sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
                sun.CreateIntensityAttr(5000.0)
                sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
                xform_api = _UsdGeom.XformCommonAPI(sun.GetPrim())
                xform_api.SetRotate(Gf.Vec3f(25.0, 0.0, 0.0))
            if not stage.GetPrimAtPath("/World/SkyDome").IsValid():
                dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
                dome.CreateIntensityAttr(800.0)
                dome.CreateColorAttr(Gf.Vec3f(0.53, 0.70, 1.0))
        except Exception as _le:
            print(f"[E2E Test] Warning: could not add lights: {_le}")


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

    # ---- Load or generate scene FIRST (before creating World) ------------
    # Opening a new USD stage invalidates any existing World object, so the
    # stage must be ready before World / PegasusInterface are constructed.
    if SCENE_PATH and os.path.exists(SCENE_PATH):
        print(f"[E2E Test] Loading scene from {SCENE_PATH}")
        omni.usd.get_context().open_stage(SCENE_PATH)
        simulation_app.update()  # let the app process the stage open event
    else:
        print("[E2E Test] No scene file found — starting with a new stage ...")
        omni.usd.get_context().new_stage()
        simulation_app.update()

    # Always add procedural geometry so the camera sees real content.
    # The USDA uses Windows-absolute paths that don't resolve on Linux,
    # leaving an empty ground plane.  Procedural prims are always present.
    _build_procedural_scene()
    simulation_app.update()

    # ---- Fire integration: enable Flow extension + create real particle fires
    _enable_flow_extension()
    _stage_fire = omni.usd.get_context().get_stage()
    n_fires = _create_flow_fires(_stage_fire)
    print(f"[E2E Test] Created {n_fires} Flow particle fires (Flow={'YES' if _FLOW_AVAILABLE else 'NO'})")

    # ---- External overview camera ----------------------------------------
    ext_cam_path = _setup_external_camera(_stage_fire)
    simulation_app.update()

    # ---- Fire spread manager (from fire_system.py) -----------------------
    fire_mgr = None
    try:
        from sim_bridge.fire_system import FireManager
        fire_mgr = FireManager()
        # FireManager._ignite_zone() already calls _add_flow_effect() to create
        # visual fires — no external callback needed.
        print(f"[E2E Test] FireManager: {len(fire_mgr._zones)} zones discovered")
    except Exception as _fme:
        print(f"[E2E Test] FireManager init failed: {_fme}")

    # Track YOLO fire detections for the output report
    yolo_fire_log = []  # list of {frame, confidence, bbox, confirmed}

    simulation_app.update()

    pg = PegasusInterface()
    # Use Pegasus' own method to create the World (handles the singleton correctly)
    pg.initialize_world()
    world = pg.world

    # Isaac Sim 4.5: World.__init__ may skip _scene creation when the singleton
    # already holds a stale reference.  Patch defensively.
    if not hasattr(world, '_scene') or world._scene is None:
        from isaacsim.core.api.scenes.scene import Scene
        world._scene = Scene()
        world._task_scene_built = False
        world._current_tasks = dict()
        from isaacsim.core.api.loggers import DataLogger
        world._data_logger = DataLogger()

    # ---- Spawn drone at survey altitude -----------------------------------
    # Iris body forward (body +X) = world +Y when init_yaw=0 (confirmed by
    # camera world-pose diagnostic).  Scene buildings placed along +Y.
    # Spawn at scene centre matching USDA SpawnPoint (0, 0, 50).
    # Start drone at high altitude over scene center for bird's eye overview.
    # The drone will survey the area in a circular orbit until YOLO detects fire,
    # then Cosmos reasoning directs it to the highest-priority area.
    _DRONE_START_POS = [0.0, 15.0, SURVEY_ALT]
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=_DRONE_START_POS,
        init_yaw_deg=0.0,
    )

    # ---- Set up orchestrator with DEBUG MODE ON ---------------------------
    K = make_intrinsics_from_fov(640, 480, hfov_deg=70.0)
    orchestrator = OrchestratorBridge(
        yolo_weights=YOLO_WEIGHTS,
        vlm_url=VLM_URL,
        debug_dir=debug_dir,
    )

    # ---- New subsystems (Prompts 1-5) ------------------------------------
    print("[E2E Test] Initialising new subsystems...")
    try:
        dual_yolo = DualYOLODetector()
        print("[E2E Test]   DualYOLODetector ready (fire + person)")
    except Exception as _e:
        dual_yolo = None
        print(f"[E2E Test]   DualYOLODetector failed: {_e}")

    thermal_proc = ThermalProcessor()
    print("[E2E Test]   ThermalProcessor ready")

    civ_tracker = None
    try:
        civ_tracker = CivilianTracker()
        print(f"[E2E Test]   CivilianTracker ready ({civ_tracker.get_civilian_report()['total']} civilians)")
    except Exception as _e:
        print(f"[E2E Test]   CivilianTracker failed: {_e}")

    report_gen = ReportGenerator(
        reports_dir=os.path.join(debug_dir, "reports"),
    )
    print(f"[E2E Test]   ReportGenerator ready (mission {report_gen.get_mission_id()})")

    world.reset()  # Initialise physics + Scene before playing

    # ---- Save initial ROOT pose BEFORE warmup ----------------------------
    # simulation_app.update() runs physics, so the drone will fall and tumble.
    # We save the ROOT pose now and restore it after warmup.
    _stage_w = omni.usd.get_context().get_stage()
    _root_path = "/World/ResQDrone"
    _root_prim = _stage_w.GetPrimAtPath(_root_path)
    _saved_root_translate = None
    _saved_root_orient = None
    _body_prim_save = _stage_w.GetPrimAtPath("/World/ResQDrone/body")
    _saved_body_translate = None
    _saved_body_orient = None
    if _root_prim.IsValid():
        _saved_root_translate = _root_prim.GetAttribute("xformOp:translate").Get()
        _saved_root_orient = _root_prim.GetAttribute("xformOp:orient").Get()
    if _body_prim_save.IsValid():
        _saved_body_translate = _body_prim_save.GetAttribute("xformOp:translate").Get()
        _saved_body_orient = _body_prim_save.GetAttribute("xformOp:orient").Get()
    import sys as _sy
    _sy.__stdout__.write(f"[DIAG] SAVED root translate={_saved_root_translate}, orient={_saved_root_orient}\n")
    _sy.__stdout__.write(f"[DIAG] SAVED body translate={_saved_body_translate}, orient={_saved_body_orient}\n")
    _sy.__stdout__.flush()

    # ---- RTX render warmup -----------------------------------------------
    for _ in range(200):
        simulation_app.update()

    # ---- Restore poses + make kinematic ---------------------------------
    try:
        from pxr import UsdPhysics
        _body_prim_k = _stage_w.GetPrimAtPath("/World/ResQDrone/body")
        if _body_prim_k.IsValid():
            _rb = UsdPhysics.RigidBodyAPI(_body_prim_k)
            _rb.CreateKinematicEnabledAttr(True)
        # Restore ROOT position (physics moved the root, not just body)
        if _root_prim.IsValid() and _saved_root_translate is not None:
            _root_prim.GetAttribute("xformOp:translate").Set(_saved_root_translate)
        if _root_prim.IsValid() and _saved_root_orient is not None:
            _root_prim.GetAttribute("xformOp:orient").Set(_saved_root_orient)
        # Restore body local pose
        if _body_prim_save.IsValid() and _saved_body_translate is not None:
            _body_prim_save.GetAttribute("xformOp:translate").Set(_saved_body_translate)
        if _body_prim_save.IsValid() and _saved_body_orient is not None:
            _body_prim_save.GetAttribute("xformOp:orient").Set(_saved_body_orient)
        print("[E2E Test] Drone: kinematic + restored to initial pose")
    except Exception as _ke:
        print(f"[E2E Test] Warning: could not restore body: {_ke}")

    # Let the new position settle for the renderer
    for _ in range(10):
        simulation_app.update()

    # ---- NOW start physics + sensors -------------------------------------
    timeline.play()

    # Force-start all graphical sensors explicitly.
    for gs in drone._graphical_sensors:
        try:
            if not getattr(gs, "_ready", False):
                gs.start()
        except Exception as e:
            print(f"[E2E Test] Warning: could not start sensor {gs.sensor_type}: {e}")

    # Re-register render callback for graphical sensors in case it was cleared
    cb_name = drone._stage_prefix + "/GraphicalSensors"
    if cb_name not in getattr(world, "_render_callback_functions", {}):
        try:
            world.add_render_callback(cb_name, drone.update_graphical_sensors)
        except Exception:
            pass

    # ---- Set up external camera render product ---------------------------
    ext_cam_annotator = None
    ext_cam_frames = []
    try:
        import omni.replicator.core as rep
        ext_rp = rep.create.render_product(ext_cam_path, (960, 540))
        ext_cam_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        ext_cam_annotator.attach([ext_rp])
        print("[E2E Test] External camera render product attached")
    except Exception as _ext_err:
        print(f"[E2E Test] External camera setup failed: {_ext_err}")
        print("[E2E Test] Will capture external view after main loop instead")

    # ---- Start FireManager for fire spread simulation --------------------
    if fire_mgr is not None:
        fire_mgr.start()
        print("[E2E Test] FireManager started — fire spread simulation active")

    # ---- Compute + apply correct camera orientation at runtime -------------
    # The Iris USD model has its own body-hierarchy rotations that we cannot
    # predict analytically.  Instead, read the body's ACTUAL world rotation,
    # then compute the camera local quaternion that produces the desired
    # world-space optical axis (45° forward-down along world +Y).
    import sys as _sys
    from scipy.spatial.transform import Rotation as _Rot
    stage = omni.usd.get_context().get_stage()

    try:
        from pxr import UsdGeom as _UG2
        body_prim = stage.GetPrimAtPath("/World/ResQDrone/body")
        if body_prim.IsValid():
            xfc2 = _UG2.XformCache()
            mtx2 = xfc2.GetLocalToWorldTransform(body_prim)
            t2 = mtx2.ExtractTranslation()
            _sys.__stdout__.write(f"[DIAG] Drone body world pos: ({t2[0]:.2f}, {t2[1]:.2f}, {t2[2]:.2f})\n")

            # Extract body world rotation as 3x3 numpy array
            rot_b = mtx2.ExtractRotationMatrix()
            R_body = np.array([[rot_b[i][j] for j in range(3)] for i in range(3)])
            _sys.__stdout__.write(f"[DIAG] Body world R row0: {R_body[0]}\n")
            _sys.__stdout__.write(f"[DIAG] Body world R row1: {R_body[1]}\n")
            _sys.__stdout__.write(f"[DIAG] Body world R row2: {R_body[2]}\n")

            # Desired camera frame in world:
            #   optical axis (-Z_cam) at 70° below horizontal, along +Y
            #   = (0, cos(70°), -sin(70°)) = (0, 0.3420, -0.9397)
            #   camera up   (+Y_cam) ≈ (0, sin(70°), cos(70°))
            #   camera right(+X_cam) = (1, 0, 0)            = world +X
            import math as _math
            _tilt_deg = 80.0  # degrees below horizontal (near-vertical bird's eye)
            _tilt_rad = _math.radians(_tilt_deg)
            _cos_t = _math.cos(_tilt_rad)
            _sin_t = _math.sin(_tilt_rad)
            c2 = np.array([0.0, -_cos_t, _sin_t])       # +Z_cam (negated = optical axis)
            c1 = np.array([0.0,  _sin_t, _cos_t])       # up
            c0 = np.cross(c1, c2)                          # right
            c0 /= np.linalg.norm(c0)
            R_desired = np.column_stack([c0, c1, c2])

            # R_cam_local (standard math) = R_body_standard_inv × R_desired
            # USD convention: v' = v × M, so R_standard = M_usd.T
            # R_body_standard = R_body_USD.T → R_body_standard_inv = R_body_USD
            R_cam_local = R_body @ R_desired  # R_body_USD @ R_desired

            # Convert to quaternion [x,y,z,w] (scipy) then to USD [w,x,y,z]
            q_scipy = _Rot.from_matrix(R_cam_local).as_quat()  # [x, y, z, w]
            from pxr import Gf as _Gf2
            q_usd = _Gf2.Quatd(float(q_scipy[3]), float(q_scipy[0]),
                                float(q_scipy[1]), float(q_scipy[2]))

            _sys.__stdout__.write(f"[DIAG] Computed cam local quat (USD wxyz): w={q_scipy[3]:.4f} x={q_scipy[0]:.4f} y={q_scipy[1]:.4f} z={q_scipy[2]:.4f}\n")

            # Apply to all cameras
            for gs in drone._graphical_sensors:
                cam_p = stage.GetPrimAtPath(gs._stage_prim_path)
                if cam_p.IsValid():
                    oa = cam_p.GetAttribute("xformOp:orient")
                    if oa and oa.IsValid():
                        oa.Set(q_usd)
            _sys.__stdout__.write("[DIAG] Camera orientation set dynamically\n")
            _sys.__stdout__.flush()
    except Exception as _e2:
        _sys.__stdout__.write(f"[DIAG] Camera orient error: {_e2}\n")
        _sys.__stdout__.flush()

    # Verify final camera orientation
    for gs in drone._graphical_sensors:
        if gs.sensor_type == "RGBCamera" and gs._camera is not None:
            try:
                from pxr import UsdGeom as _UG
                cam_prim = stage.GetPrimAtPath(gs._stage_prim_path)
                if cam_prim.IsValid():
                    # Force cache refresh
                    simulation_app.update()
                    xfc = _UG.XformCache()
                    mtx = xfc.GetLocalToWorldTransform(cam_prim)
                    trans = mtx.ExtractTranslation()
                    rot = mtx.ExtractRotationMatrix()
                    # USD row-vector convention: row i = where +i axis goes in world
                    look = (rot[2][0], rot[2][1], rot[2][2])  # row 2 = camera +Z in world
                    _sys.__stdout__.write(f"[DIAG] Camera prim: {gs._stage_prim_path}\n")
                    _sys.__stdout__.write(f"[DIAG] World pos  : ({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f})\n")
                    _sys.__stdout__.write(f"[DIAG] +Z look dir: ({look[0]:.3f}, {look[1]:.3f}, {look[2]:.3f})\n")
                    _sys.__stdout__.flush()
            except Exception as _e:
                _sys.__stdout__.write(f"[DIAG] Verify error: {_e}\n")
                _sys.__stdout__.flush()
            break

    # ---- Simulation loop --------------------------------------------------
    warmup_steps = 10   # sensor warmup is now just 5 frames
    step = 0
    frames_processed = 0
    total_hazards = 0
    prev_time = time.perf_counter()

    # ---- MPC-inspired smooth dynamics controller --------------------------
    # Replaces the old constant-speed linear interpolation with a velocity /
    # acceleration-based system that produces natural, cinematic drone flight.
    from orchestrator.orchestrator_bridge import _latest_vlm_waypoint, _latest_vlm_reasoning

    _stage_for_flight = omni.usd.get_context().get_stage()
    _root_path_str = "/World/ResQDrone"  # animate root, not body

    # Flight state machine (kept for high-level decisions; motion is now smooth)
    _FLIGHT_SURVEY   = "survey"      # circular orbit, surveying the area
    _FLIGHT_APPROACH = "approach"    # flying above a priority fire/people area
    _FLIGHT_FOCUS    = "focus"       # hovering over priority area, observing
    _FLIGHT_TRACKING = "tracking"    # Cosmos-directed repositioning

    # Create MPC controller
    _ctrl = DroneController(
        _DRONE_START_POS,
        max_vel=4.5,           # m/step — smooth but not sluggish
        max_accel=0.6,         # gentle acceleration limit
        drag=0.04,             # realistic air drag
        hover_drift=0.12,      # subtle hover oscillation
        wind_strength=0.25,    # noticeable but not distracting wind
        altitude=SURVEY_ALT,
    )
    _ctrl.configure_orbit(
        center=[0.0, 15.0],   # scene center
        radius=55.0,
        speed=0.025,
    )
    _ctrl.start_orbit()        # begin with smooth survey orbit

    _flight_state = _FLIGHT_SURVEY
    _cosmos_last_reasoning = ""
    _last_cosmos_wp_check = None
    _all_yolo_fires = []             # aggregated fire positions from YOLO
    _all_yolo_people = []            # aggregated people positions from YOLO
    _yolo_fire_detect_count = 0      # consecutive frames with fire detection
    _FOCUS_DURATION = CUSTOM_FOCUS_DURATION if CUSTOM_FOCUS_DURATION > 0 else 55
    _focus_counter = 0

    # ---- Fire zone itinerary: visit ALL zones in order -------------------
    # Populated once FireManager has discovered zones
    _fire_zone_positions = {}  # zone_name -> [x, y, z]
    try:
        if fire_mgr is not None:
            for _zn, _zi in fire_mgr._zones.items():
                _fire_zone_positions[_zn] = [_zi["position"][0], _zi["position"][1], SURVEY_ALT]
    except Exception:
        pass

    # Zone order: use custom order if env var is set, otherwise alphabetical
    if ZONE_ORDER_STR:
        _fire_zone_names = [z.strip() for z in ZONE_ORDER_STR.split(",") if z.strip()]
        # Ensure all requested zones exist; add missing zone positions from _FIRE_ZONES
        for zn in _fire_zone_names:
            if zn not in _fire_zone_positions:
                for fz in _FIRE_ZONES:
                    if fz["name"] == zn:
                        _fire_zone_positions[zn] = [fz["pos"][0], fz["pos"][1], SURVEY_ALT]
                        break
        print(f"[Flight] Custom zone order: {_fire_zone_names}")
    else:
        _fire_zone_names = sorted(_fire_zone_positions.keys())  # FZ_0, FZ_1, ...

    _visited_fire_zones = set()         # zones the drone has already focused on
    _itinerary_idx = 0                  # next zone in itinerary to visit
    _cooldown_after_focus = 0           # steps to stay in survey after a focus
    _COOLDOWN_STEPS = CUSTOM_COOLDOWN_STEPS if CUSTOM_COOLDOWN_STEPS > 0 else 40

    # ---- Cosmos reasoning log (written to file at end) -------------------
    _cosmos_reasoning_log = []          # list of {step, reasoning, target, state}

    # ---- Forward-facing camera (FPV) setup --------------------------------
    fwd_cam_path = "/World/ForwardCamera"
    fwd_cam_annotator = None
    fwd_cam_frames = []
    try:
        fwd_cam = UsdGeom.Camera.Define(_stage_for_flight, fwd_cam_path)
        fwd_cam.CreateFocalLengthAttr(18.0)
        fwd_cam.CreateHorizontalApertureAttr(24.0)
        fwd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 2000.0))
        fwd_xform = UsdGeom.Xformable(fwd_cam.GetPrim())
        fwd_xform.AddTranslateOp().Set(Gf.Vec3d(
            float(_DRONE_START_POS[0]),
            float(_DRONE_START_POS[1]),
            float(_DRONE_START_POS[2]),
        ))
        # Identity orient — will be set per frame
        fwd_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1.0, 0.0, 0.0, 0.0))

        import omni.replicator.core as rep
        fwd_rp = rep.create.render_product(fwd_cam_path, (960, 540))
        fwd_cam_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        fwd_cam_annotator.attach([fwd_rp])
        print("[Camera] Forward-facing FPV camera created")
    except Exception as _fwd_err:
        print(f"[Camera] Forward camera setup failed: {_fwd_err}")

    def _update_forward_camera(drone_pos, drone_yaw):
        """Move forward camera to drone position and orient along flight direction.
        Camera looks 25° below horizontal in the yaw direction."""
        try:
            fp = _stage_for_flight.GetPrimAtPath(fwd_cam_path)
            if not fp.IsValid():
                return
            ta = fp.GetAttribute("xformOp:translate")
            if ta and ta.IsValid():
                ta.Set(Gf.Vec3d(float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])))

            # Orient: 25° below horizontal, facing yaw direction
            _tilt = math.radians(25.0)
            _cy, _sy = math.cos(drone_yaw), math.sin(drone_yaw)
            fwd_dir = np.array([
                _cy * math.cos(_tilt),
                _sy * math.cos(_tilt),
                -math.sin(_tilt),
            ])
            right = np.array([-_sy, _cy, 0.0])
            up = np.cross(right, fwd_dir)
            up_norm = np.linalg.norm(up)
            if up_norm > 1e-8:
                up /= up_norm
            R = np.column_stack([right, up, -fwd_dir])
            from scipy.spatial.transform import Rotation as _RotF2
            q = _RotF2.from_matrix(R).as_quat()
            oa = fp.GetAttribute("xformOp:orient")
            if oa and oa.IsValid():
                oa.Set(Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
        except Exception:
            pass

    print(f"[E2E Test] Running {MAX_STEPS} steps (+ {warmup_steps} warmup)")
    print(f"[E2E Test] Flight: MPC dynamics — Survey orbit → YOLO detect → Cosmos approach → Focus")
    print(f"[E2E Test] Smooth acceleration/deceleration, wind perturbation, hover drift")
    print(f"[E2E Test] Drone starts at {_DRONE_START_POS}, altitude {SURVEY_ALT}m")
    print(f"[E2E Test] Ctrl-C to stop early.\n")

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            # ---- MPC flight controller + state machine -----------------------
            import orchestrator.orchestrator_bridge as _orch_mod
            _cur_cosmos_wp = _orch_mod._latest_vlm_waypoint
            _cur_cosmos_reason = _orch_mod._latest_vlm_reasoning

            # Check for new Cosmos waypoint (VLM priority reasoning)
            if _cur_cosmos_wp is not None and _cur_cosmos_wp != _last_cosmos_wp_check:
                _last_cosmos_wp_check = list(_cur_cosmos_wp)
                _cosmos_target = np.array(_cur_cosmos_wp, dtype=np.float64)
                _cosmos_target[2] = SURVEY_ALT  # enforce altitude
                _cosmos_last_reasoning = _cur_cosmos_reason or ""

                # Log Cosmos reasoning
                _cosmos_reasoning_log.append({
                    "step": step,
                    "reasoning": _cosmos_last_reasoning,
                    "target": [float(_cosmos_target[0]), float(_cosmos_target[1]), float(_cosmos_target[2])],
                    "state": _flight_state,
                })

                # Smoothly redirect drone (not mid-focus, not in cooldown)
                if _flight_state in (_FLIGHT_SURVEY, _FLIGHT_TRACKING) and _cooldown_after_focus <= 0:
                    # Only follow Cosmos if target is near an unvisited zone
                    _cosmos_near_unvisited = False
                    for _zn, _zp in _fire_zone_positions.items():
                        if _zn not in _visited_fire_zones:
                            if np.linalg.norm(_cosmos_target[:2] - np.array(_zp[:2])) < 30.0:
                                _cosmos_near_unvisited = True
                                break
                    if _cosmos_near_unvisited or len(_visited_fire_zones) >= len(_fire_zone_names):
                        _flight_state = _FLIGHT_TRACKING
                        _ctrl.go_to(_cosmos_target, speed_factor=0.8)
                        if step > warmup_steps:
                            print(f"[Cosmos Nav] Priority target → ({_cosmos_target[0]:.1f}, {_cosmos_target[1]:.1f}) — {_cosmos_last_reasoning[:100]}")

            # Tick cooldown counter
            if _cooldown_after_focus > 0:
                _cooldown_after_focus -= 1

            # State transitions (high-level decisions; MPC handles smooth motion)
            if _flight_state == _FLIGHT_SURVEY:
                # MPC controller handles smooth circular orbit with wind
                # Auto-advance itinerary: if we've been orbiting long enough,
                # fly to the next unvisited fire zone automatically
                if _cooldown_after_focus <= 0 and _fire_zone_names:
                    _attempts = 0
                    while _attempts < len(_fire_zone_names):
                        _next_zone = _fire_zone_names[_itinerary_idx % len(_fire_zone_names)]
                        if _next_zone not in _visited_fire_zones:
                            _target_pos = np.array(_fire_zone_positions[_next_zone], dtype=np.float64)
                            _ctrl.go_to(_target_pos, speed_factor=0.7)
                            _flight_state = _FLIGHT_APPROACH
                            print(f"[Flight] Itinerary → approaching {_next_zone} at ({_target_pos[0]:.1f}, {_target_pos[1]:.1f})")
                            break
                        _itinerary_idx += 1
                        _attempts += 1
                    else:
                        # All zones visited — reset to allow revisits
                        if len(_visited_fire_zones) >= len(_fire_zone_names):
                            print(f"[Flight] All {len(_fire_zone_names)} fire zones visited — resetting itinerary")
                            _visited_fire_zones.clear()
                            _itinerary_idx = 0

            elif _flight_state == _FLIGHT_APPROACH:
                # Controller automatically decelerates near target
                if _ctrl.at_target:
                    _flight_state = _FLIGHT_FOCUS
                    _focus_counter = 0
                    _ctrl.slow_down(0.25)  # slow down over fire area
                    print(f"[Flight] Over priority area — focusing (smooth hover)")

            elif _flight_state == _FLIGHT_FOCUS:
                # Hovering with gentle drift (MPC hover perturbation)
                _focus_counter += 1
                if _focus_counter >= _FOCUS_DURATION:
                    # Mark current zone as visited
                    _cur_zone_name = None
                    _cur_drone = _ctrl.pos[:2]
                    _best_dist = 999.0
                    for _zn, _zp in _fire_zone_positions.items():
                        _d = np.linalg.norm(_cur_drone - np.array(_zp[:2]))
                        if _d < _best_dist:
                            _best_dist = _d
                            _cur_zone_name = _zn
                    if _cur_zone_name:
                        _visited_fire_zones.add(_cur_zone_name)
                        _itinerary_idx = (_fire_zone_names.index(_cur_zone_name) + 1) % len(_fire_zone_names) if _cur_zone_name in _fire_zone_names else _itinerary_idx + 1
                        print(f"[Flight] Focus complete on {_cur_zone_name} — visited {len(_visited_fire_zones)}/{len(_fire_zone_names)} zones")
                    else:
                        print(f"[Flight] Focus complete — resuming survey")

                    _flight_state = _FLIGHT_SURVEY
                    _cooldown_after_focus = _COOLDOWN_STEPS
                    _ctrl.resume_speed()
                    _ctrl.start_orbit()

            elif _flight_state == _FLIGHT_TRACKING:
                # Cosmos-directed smooth approach
                if _ctrl.at_target:
                    _flight_state = _FLIGHT_FOCUS
                    _focus_counter = 0
                    _ctrl.slow_down(0.25)
                    print(f"[Cosmos Nav] Over priority target — focusing")

            # ---- MPC physics step (always runs) ----
            _pos_now = _ctrl.step()

            # ---- Update external camera to track drone ----
            _update_external_camera(_stage_for_flight, ext_cam_path, _pos_now)

            # ---- Update forward camera position + orientation ----
            _update_forward_camera(_pos_now, _ctrl.yaw)

            # ---- Capture external camera frame (every 3rd step) ----
            if ext_cam_annotator is not None and step % 3 == 0:
                ext_frame = _capture_external_frame(ext_cam_annotator)
                if ext_frame is not None:
                    ext_cam_frames.append(ext_frame)

            # ---- Capture forward camera frame (every 3rd step) ----
            if fwd_cam_annotator is not None and step % 3 == 0:
                fwd_frame = _capture_external_frame(fwd_cam_annotator)
                if fwd_frame is not None:
                    fwd_cam_frames.append(fwd_frame)

            # ---- Apply position to USD prims ----
            try:
                _rp = _stage_for_flight.GetPrimAtPath(_root_path_str)
                if _rp.IsValid():
                    _ta = _rp.GetAttribute("xformOp:translate")
                    if _ta and _ta.IsValid():
                        _ta.Set(Gf.Vec3d(float(_pos_now[0]), float(_pos_now[1]), float(_pos_now[2])))
                # Also set body translate to zero to prevent physics drift
                _bp2 = _stage_for_flight.GetPrimAtPath("/World/ResQDrone/body")
                if _bp2.IsValid():
                    _bta = _bp2.GetAttribute("xformOp:translate")
                    if _bta and _bta.IsValid():
                        _bta.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            except Exception:
                pass

            # Runtime diagnostics (every 10 steps)
            if step % 10 == 0:
                try:
                    import sys as _sd
                    _sd.__stdout__.write(
                        f"[DIAG step={step}] state={_flight_state} "
                        f"pos=({_pos_now[0]:.1f},{_pos_now[1]:.1f},{_pos_now[2]:.1f}) "
                        f"vel={_ctrl.speed:.2f} mode={_ctrl.mode} "
                        f"spd_factor={_ctrl.speed_factor:.2f}\n"
                    )
                    _sd.__stdout__.flush()
                except Exception:
                    pass

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

            thermal = None
            if seg_raw is not None:
                thermal = generate_synthetic_thermal(seg_raw)
                # If all pixels are nearly the same value, semantic labels were
                # empty (everything = unlabeled) → fall back to RGB-based thermal
                if thermal is not None and thermal.std() < 5:
                    thermal = generate_thermal_from_rgb(rgb)
            else:
                thermal = generate_thermal_from_rgb(rgb)
            cam_pose = _get_camera_world_pose(drone)
            pos = imu_backend.latest_state["position"]

            # Get fire report before orchestrator call so Cosmos has fire context
            _fire_rpt = fire_mgr.get_fire_report() if fire_mgr else None

            # Run the full orchestrator pipeline (debug saves happen inside)
            frame_result = orchestrator.process_frame(
                rgb_frame=rgb,
                thermal_frame=thermal,
                depth_map=depth,
                camera_intrinsics=K,
                camera_world_pose=cam_pose,
                frame_idx=step,
                drone_position=pos,
                fire_report=_fire_rpt,
            )

            frames_processed += 1

            # ---- New subsystems: dual YOLO + thermal + civilian tracking ----
            dual_detections = []
            thermal_hotspots = []

            # Thermal hotspot analysis
            if thermal is not None:
                thermal_hotspots = thermal_proc.process(thermal)

            # Dual YOLO: person + fire detection with thermal cross-validation
            if dual_yolo is not None:
                try:
                    dual_detections = dual_yolo.detect(
                        rgb, thermal_hotspots=thermal_hotspots)
                except Exception as _yd_err:
                    if frames_processed % 100 == 1:
                        print(f"[E2E Test] DualYOLO error: {_yd_err}")

            # Track fire detections for the fire situation report
            for _det in dual_detections:
                if _det.get("class") == "fire":
                    yolo_fire_log.append({
                        "frame": step,
                        "confidence": _det.get("confidence", 0),
                        "bbox": _det.get("bbox"),
                        "confirmed": _det.get("confirmed", False),
                    })

            # ---- YOLO fire/people → aggregate + trigger Cosmos priority ----
            # YOLO decides WHERE fire/people are. Cosmos decides WHAT to do next.
            _yolo_has_fire = any(d.get("class") == "fire" for d in dual_detections)
            _yolo_has_person = any(d.get("class") == "person" for d in dual_detections)

            # Aggregate fire positions from orchestrator's 3D projections
            if _yolo_has_fire and frame_result and frame_result.get("hazards"):
                for _hz in frame_result["hazards"]:
                    if _hz.get("class_name") == "fire" and _hz.get("world_xyz") is not None:
                        _all_yolo_fires.append({
                            "pos": list(_hz["world_xyz"]),
                            "confidence": _hz.get("confidence", 0),
                            "frame": step,
                        })

            # Aggregate people positions
            if _yolo_has_person and frame_result and frame_result.get("hazards"):
                for _hz in frame_result["hazards"]:
                    if _hz.get("class_name") == "person" and _hz.get("world_xyz") is not None:
                        _all_yolo_people.append({
                            "pos": list(_hz["world_xyz"]),
                            "confidence": _hz.get("confidence", 0),
                            "frame": step,
                        })

            # Transition from survey to approach when YOLO confirms fire
            # (only if not in cooldown and the detected zone hasn't been visited yet)
            if _yolo_has_fire:
                _yolo_fire_detect_count += 1
                if (_yolo_fire_detect_count >= 2
                        and _flight_state == _FLIGHT_SURVEY
                        and _cooldown_after_focus <= 0):
                    if _all_yolo_fires:
                        _latest = _all_yolo_fires[-1]["pos"]
                        # Check if this is near an already-visited zone
                        _yolo_near_visited = False
                        for _vzn in _visited_fire_zones:
                            _vzp = _fire_zone_positions.get(_vzn, [0, 0, 0])
                            if np.linalg.norm(np.array(_latest[:2]) - np.array(_vzp[:2])) < 25.0:
                                _yolo_near_visited = True
                                break
                        if not _yolo_near_visited:
                            _target = np.array([_latest[0], _latest[1], SURVEY_ALT])
                            _ctrl.go_to(_target, speed_factor=0.7)
                            _flight_state = _FLIGHT_APPROACH
                            print(f"[Flight] YOLO fire at ({_latest[0]:.1f}, {_latest[1]:.1f}) — "
                                  f"smooth approach from {SURVEY_ALT:.0f}m altitude")
            else:
                _yolo_fire_detect_count = 0

            # Also react to people detection near fire for criticality
            if _yolo_has_person and _yolo_has_fire and step % 30 == 0:
                print(f"[Flight] CRITICAL: People detected near fire at step {step}!")

            # Civilian tracking
            _fire_rpt = fire_mgr.get_fire_report() if fire_mgr else None
            if civ_tracker is not None:
                civ_tracker.update(
                    detections=dual_detections,
                    fire_report=_fire_rpt,
                    frame_idx=step,
                )

            # Mission reporting (every 5 seconds)
            if report_gen.should_generate():
                civ_report = civ_tracker.get_civilian_report() if civ_tracker else None
                drone_pos_list = [float(pos[0]), float(pos[1]), float(pos[2])]

                # Count person/fire from dual detections
                _n_people = sum(1 for d in dual_detections if d["class"] == "person")
                _n_fire = sum(1 for d in dual_detections if d["class"] == "fire" and d.get("confirmed"))

                # Build Cosmos decision log for reporting
                _cosmos_decisions = []
                if _flight_state in (_FLIGHT_TRACKING, _FLIGHT_APPROACH, _FLIGHT_FOCUS):
                    _cosmos_decisions.append({
                        "frame": step,
                        "reasoning": _cosmos_last_reasoning or f"Fire detected — state: {_flight_state}",
                        "target": list(_ctrl.target),
                        "mode": _flight_state,
                    })
                else:
                    _cosmos_decisions.append({
                        "frame": step,
                        "reasoning": f"Surveying at ({_ctrl.pos[0]:.1f}, {_ctrl.pos[1]:.1f}, {SURVEY_ALT:.0f}m) — scanning for fire",
                        "target": list(_ctrl.pos),
                        "mode": "survey",
                    })

                report = report_gen.generate(
                    fire_report=_fire_rpt,
                    civilian_report=civ_report,
                    detections=dual_detections,
                    cosmos_decisions=_cosmos_decisions,
                    drone_position=drone_pos_list,
                    drone_battery=max(0, 100.0 - step * 0.01),
                    drone_status=_flight_state,
                )
                if frames_processed % 50 == 0:
                    print(f"[E2E Test] Report: people={_n_people} fire={_n_fire} "
                          f"urgency={report.get('urgency_level', '?')}")

            # ---- Original hazard logging ------------------------------------
            if frame_result and frame_result.get("hazards"):
                n = len(frame_result["hazards"])
                total_hazards += n
                if frames_processed % 20 == 0:
                    print(f"[E2E Test] step={step} | {n} active hazard(s) | "
                          f"people={len([d for d in dual_detections if d['class']=='person'])} | "
                          f"pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
            elif frames_processed % 50 == 0:
                print(f"[E2E Test] step={step} | no detections | pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")

            if step >= warmup_steps + MAX_STEPS:
                print(f"\n[E2E Test] Reached max steps ({MAX_STEPS})")
                break

    except KeyboardInterrupt:
        print("\n[E2E Test] Interrupted by user.")

    timeline.stop()

    # ---- Stop FireManager + generate fire report ---------------------------
    if fire_mgr is not None:
        fire_mgr.stop()
    _generate_fire_report(fire_mgr, yolo_fire_log, debug_dir)

    # ---- Write Cosmos reasoning log to text file --------------------------
    try:
        cosmos_log_path = os.path.join(debug_dir, "cosmos_reasoning.txt")
        with open(cosmos_log_path, "w") as _clf:
            _clf.write("=" * 70 + "\n")
            _clf.write("  ResQ-AI — Cosmos Reasoning Log\n")
            _clf.write("=" * 70 + "\n\n")
            _clf.write(f"Total Cosmos decisions: {len(_cosmos_reasoning_log)}\n")
            _clf.write(f"Fire zones visited: {sorted(_visited_fire_zones)}\n")
            _clf.write(f"Fire zones total: {_fire_zone_names}\n\n")
            for _entry in _cosmos_reasoning_log:
                _clf.write("-" * 50 + "\n")
                _clf.write(f"Step {_entry['step']} | State: {_entry['state']}\n")
                _clf.write(f"Target: ({_entry['target'][0]:.1f}, {_entry['target'][1]:.1f}, {_entry['target'][2]:.1f})\n")
                _clf.write(f"Reasoning: {_entry['reasoning']}\n\n")
            _clf.write("=" * 70 + "\n")
            _clf.write("END OF LOG\n")
        print(f"[E2E Test] Cosmos reasoning log: {cosmos_log_path} ({len(_cosmos_reasoning_log)} entries)")
    except Exception as _log_err:
        print(f"[E2E Test] Cosmos log write failed: {_log_err}")

    # ---- Encode external camera overview video ----------------------------
    if ext_cam_frames:
        try:
            import cv2
            ext_vid_path = os.path.join(debug_dir, "external_overview.mp4")
            h, w = ext_cam_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(ext_vid_path, fourcc, 10, (w, h))
            for f in ext_cam_frames:
                writer.write(f)
            writer.release()
            print(f"[E2E Test] External camera video: {ext_vid_path} ({len(ext_cam_frames)} frames)")
        except Exception as _vid_err:
            print(f"[E2E Test] External video encode failed: {_vid_err}")
            # Fallback: save frames as images
            ext_frames_dir = os.path.join(debug_dir, "ext_cam_frames")
            os.makedirs(ext_frames_dir, exist_ok=True)
            for i, f in enumerate(ext_cam_frames):
                import cv2 as _cv
                _cv.imwrite(os.path.join(ext_frames_dir, f"ext_{i:04d}.jpg"), f)
            print(f"[E2E Test] Saved {len(ext_cam_frames)} ext cam frames to {ext_frames_dir}")
    else:
        print("[E2E Test] No external camera frames captured")

    # ---- Encode forward camera (FPV) video ---------------------------------
    if fwd_cam_frames:
        try:
            import cv2
            fwd_vid_path = os.path.join(debug_dir, "forward_fpv.mp4")
            h, w = fwd_cam_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(fwd_vid_path, fourcc, 10, (w, h))
            for f in fwd_cam_frames:
                writer.write(f)
            writer.release()
            print(f"[E2E Test] Forward FPV video: {fwd_vid_path} ({len(fwd_cam_frames)} frames)")
        except Exception as _fwd_vid_err:
            print(f"[E2E Test] Forward video encode failed: {_fwd_vid_err}")
    else:
        print("[E2E Test] No forward camera frames captured")

    # ---- Generate final report ---------------------------------------------
    _final_fire_rpt = fire_mgr.get_fire_report() if fire_mgr else None
    if report_gen is not None:
        civ_report = civ_tracker.get_civilian_report() if civ_tracker else None
        report_gen.generate(
            fire_report=_final_fire_rpt,
            civilian_report=civ_report,
            detections=[],
            cosmos_decisions=[],
            drone_position=[0, 0, 50],
            drone_battery=max(0, 100.0 - step * 0.01),
            drone_status="shutdown",
        )

    # ---- Summary ----------------------------------------------------------
    elapsed = time.perf_counter() - prev_time
    debug_files = []
    frames_dir = os.path.join(debug_dir, "frames")
    if os.path.isdir(frames_dir):
        debug_files = os.listdir(frames_dir)

    total_reports = len(report_gen.get_all_reports()) if report_gen else 0

    print("\n" + "=" * 60)
    print("  ResQ-AI Headless E2E Test - Summary")
    print("=" * 60)
    print(f"  Sim steps run    : {step}")
    print(f"  Frames processed : {frames_processed}")
    print(f"  Total hazards    : {total_hazards}")
    print(f"  Mission reports  : {total_reports}")
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
    report_count = sum(1 for f in debug_files if f.startswith("mission_"))
    print(f"    RGB frames     : {rgb_count}")
    print(f"    Thermal frames : {thermal_count}")
    print(f"    Seg overlays   : {seg_count}")
    print(f"    YOLO JSONs     : {yolo_count}")
    print(f"    Cosmos prompts : {cosmos_count}")
    print(f"    VLM responses  : {vlm_count}")
    print(f"    Mission reports: {total_reports}")
    if civ_tracker is not None:
        cr = civ_tracker.get_civilian_report()
        print(f"  Civilians tracked: {cr['total']} (danger: {cr['critical_danger']})")
    print("=" * 60)

    simulation_app.close()
    print("[E2E Test] Done.")


if __name__ == "__main__":
    main()
