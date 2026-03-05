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

from orchestrator.orchestrator_bridge import OrchestratorBridge

# --- New subsystem imports (Prompts 1-5) ----------------------------------
from sim_bridge.yolo_detector import DualYOLODetector
from sim_bridge.thermal_processor import ThermalProcessor
from sim_bridge.civilian_tracker import CivilianTracker
from sim_bridge.report_generator import ReportGenerator


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

    # --- Fire / hazard zones (tall flame columns + glowing bases) ---------
    # Positioned at USDA fire-zone coordinates for visual consistency
    fire_zones = [
        ([-95.0, -17.0], 7.3),   # FZ_0
        ([-99.0,  46.0], 5.8),   # FZ_1
        ([-2.0,   46.0], 7.3),   # FZ_2
        ([88.0,  -17.0], 4.2),   # FZ_3
        ([80.0,   46.0], 5.0),   # FZ_4
    ]
    for i, ((fx, fy), radius) in enumerate(fire_zones):
        # Tall flame column (bright orange-red)
        VisualCuboid(
            prim_path=f"/World/FireColumn_{i:02d}",
            position=np.array([fx, fy, 4.0], dtype=np.float64),
            scale=np.array([radius, radius, 8.0], dtype=np.float64),
            color=np.array([1.0, 0.25, 0.0]),
        )
        # Flame top (extra bright)
        VisualSphere(
            prim_path=f"/World/FireTop_{i:02d}",
            position=np.array([fx, fy, 9.0], dtype=np.float64),
            radius=float(radius * 0.6),
            color=np.array([1.0, 0.50, 0.0]),
        )
        # Fire glow base sphere
        VisualSphere(
            prim_path=f"/World/FireGlow_{i:02d}",
            position=np.array([fx, fy, 2.5], dtype=np.float64),
            radius=float(radius * 0.8),
            color=np.array([0.95, 0.15, 0.0]),
        )

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

    # --- Survivors (small red spheres on rubble) --------------------------
    survivors = [
        ([-92.0, -15.0, 2.5], 0.6, [0.85, 0.05, 0.05]),   # near FZ_0
        ([-97.0,  48.0, 2.8], 0.5, [0.90, 0.10, 0.05]),   # near FZ_1
        ([0.0,    48.0, 2.2], 0.6, [0.85, 0.05, 0.05]),   # near FZ_2
        ([90.0,  -15.0, 2.5], 0.5, [0.90, 0.10, 0.05]),   # near FZ_3
        ([22.0,   1.0,  1.5], 0.5, [0.85, 0.05, 0.05]),   # centre rubble
        ([-38.0, -18.0, 1.8], 0.6, [0.90, 0.10, 0.05]),   # extra
    ]
    for i, (pos, radius, color) in enumerate(survivors):
        VisualSphere(
            prim_path=f"/World/Survivor_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            radius=float(radius),
            color=np.array(color),
        )

    print("[E2E Test] Procedural scene: 17 buildings, 7 collapsed, 5 fire columns,")
    print("           4 flood zones, 10 vehicles, 6 survivors added.")

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
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=[0.0, 0.0, SURVEY_ALT],   # scene centre
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
            _tilt_deg = 70.0  # degrees below horizontal
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

    # Flight path: patrol over the USDA urban disaster scene using the
    # waypoints defined in the USDA (DroneOps/Waypoints).
    # WP0: (0,0,50) → WP1: (92,36,40) → WP2: (92,-36,35) → WP3: (-92,-36,35)
    # → WP4: (-92,36,40) → WP5: (0,0,25)
    # Each leg gets an equal fraction of total steps.
    _WAYPOINTS = [
        np.array([0.0,   0.0,  45.0]),   # start centre (use SURVEY_ALT)
        np.array([85.0,  30.0, 40.0]),   # NE - near fire zones FZ_3, FZ_4
        np.array([85.0, -30.0, 35.0]),   # SE
        np.array([-85.0,-30.0, 35.0]),   # SW - near fire zones FZ_0
        np.array([-85.0, 30.0, 40.0]),   # NW - near fire zones FZ_1
        np.array([0.0,   0.0,  30.0]),   # return centre, lower pass
    ]
    _stage_for_flight = omni.usd.get_context().get_stage()
    _root_path_str = "/World/ResQDrone"  # animate root, not body
    _n_legs = len(_WAYPOINTS) - 1

    def _get_flight_pos(progress: float) -> tuple:
        """Interpolate position along waypoint path. progress: 0→1."""
        progress = max(0.0, min(1.0, progress))
        leg_f = progress * _n_legs
        leg_idx = min(int(leg_f), _n_legs - 1)
        leg_t = leg_f - leg_idx
        p0 = _WAYPOINTS[leg_idx]
        p1 = _WAYPOINTS[leg_idx + 1]
        return p0 + leg_t * (p1 - p0)

    print(f"[E2E Test] Running {MAX_STEPS} steps (+ {warmup_steps} warmup)")
    print(f"[E2E Test] Flight path: {_n_legs}-leg waypoint patrol, alt=30-45m")
    print(f"[E2E Test] Ctrl-C to stop early.\n")

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            # ---- Animate drone position along waypoint path ----
            total_active = max(warmup_steps + MAX_STEPS, 1)
            t = min(step / total_active, 1.0)  # 0 → 1 over entire run
            _pos_now = _get_flight_pos(t)
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

            # Runtime position diagnostics (every 10 steps)
            if step % 10 == 0:
                try:
                    from pxr import UsdGeom as _UGd
                    _xfc_d = _UGd.XformCache()
                    _bp_d = _stage_for_flight.GetPrimAtPath("/World/ResQDrone/body")
                    if _bp_d.IsValid():
                        _mtx_d = _xfc_d.GetLocalToWorldTransform(_bp_d)
                        _td = _mtx_d.ExtractTranslation()
                        _rd = _mtx_d.ExtractRotationMatrix()
                        _z_row = (_rd[2][0], _rd[2][1], _rd[2][2])
                        _cam_d = _stage_for_flight.GetPrimAtPath("/World/ResQDrone/body/rgb_cam")
                        _cam_pos = "N/A"
                        if _cam_d.IsValid():
                            _cam_mtx = _xfc_d.GetLocalToWorldTransform(_cam_d)
                            _cam_t = _cam_mtx.ExtractTranslation()
                            _cam_r = _cam_mtx.ExtractRotationMatrix()
                            _cam_z = (_cam_r[2][0], _cam_r[2][1], _cam_r[2][2])
                            _cam_pos = f"({_cam_t[0]:.1f},{_cam_t[1]:.1f},{_cam_t[2]:.1f}) +Z=({_cam_z[0]:.2f},{_cam_z[1]:.2f},{_cam_z[2]:.2f})"
                        import sys as _sd
                        _sd.__stdout__.write(f"[DIAG step={step}] t={t:.2f} pos=({_pos_now[0]:.1f},{_pos_now[1]:.1f},{_pos_now[2]:.1f}) body=({_td[0]:.1f},{_td[1]:.1f},{_td[2]:.1f}) cam={_cam_pos}\n")
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

            # Civilian tracking
            if civ_tracker is not None:
                civ_tracker.update(
                    detections=dual_detections,
                    fire_report=None,
                    frame_idx=step,
                )

            # Mission reporting (every 5 seconds)
            if report_gen.should_generate():
                civ_report = civ_tracker.get_civilian_report() if civ_tracker else None
                drone_pos_list = [float(pos[0]), float(pos[1]), float(pos[2])]

                # Count person/fire from dual detections
                _n_people = sum(1 for d in dual_detections if d["class"] == "person")
                _n_fire = sum(1 for d in dual_detections if d["class"] == "fire" and d.get("confirmed"))

                report = report_gen.generate(
                    fire_report=None,
                    civilian_report=civ_report,
                    detections=dual_detections,
                    cosmos_decisions=[],
                    drone_position=drone_pos_list,
                    drone_battery=max(0, 100.0 - step * 0.01),
                    drone_status="patrolling",
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

    # ---- Generate final report ---------------------------------------------
    if report_gen is not None:
        civ_report = civ_tracker.get_civilian_report() if civ_tracker else None
        report_gen.generate(
            fire_report=None,
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
