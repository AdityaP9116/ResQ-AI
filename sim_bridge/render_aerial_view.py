#!/usr/bin/env python3
"""ResQ-AI — Render a static aerial view with ALL Flow particle fires.

Launches Isaac Sim headless, loads the scene + procedural geometry,
ignites all 5 fire zones simultaneously with real Flow particle fires,
and captures a high-resolution aerial frame from a static overhead camera.

This replaces the manual fire compositing approach — fires are real
Isaac Sim rendered Flow particles visible from the aerial camera.

Environment variables:
    RESQAI_AERIAL_OUTPUT   Output image path        (default: debug_output_2/aerial_rendered.jpg)
    RESQAI_SCENE           USD scene path            (default: ./resqai_urban_disaster.usda)
    RESQAI_WARMUP_STEPS    RTX warmup steps          (default: 80)
    RESQAI_FIRE_GROW_STEPS Steps for fire growth     (default: 120)
    RESQAI_CAM_ALT         Camera altitude in metres (default: 180)
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

_RES_W = int(os.environ.get("RESQAI_RENDER_W", "960"))
_RES_H = int(os.environ.get("RESQAI_RENDER_H", "540"))

simulation_app = SimulationApp({
    "headless": True,
    "anti_aliasing": 0,
    "renderer": "RaytracedLighting",
    "width": _RES_W,
    "height": _RES_H,
    "sync_loads": True,
})

# ---------------------------------------------------------------------------
# Imports (available after SimulationApp init)
# ---------------------------------------------------------------------------
import numpy as np
import cv2

import omni.usd
import omni.timeline
import omni.kit.commands

from pxr import Gf, UsdGeom

try:
    from omni.isaac.core.objects import VisualCuboid, VisualSphere
except ImportError:
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENE_PATH = os.environ.get(
    "RESQAI_SCENE",
    os.path.join(_PROJECT_ROOT, "resqai_urban_disaster.usda"),
)
OUTPUT_PATH = os.environ.get(
    "RESQAI_AERIAL_OUTPUT",
    os.path.join(_PROJECT_ROOT, "debug_output_2", "aerial_rendered.jpg"),
)
WARMUP_STEPS = int(os.environ.get("RESQAI_WARMUP_STEPS", "80"))
FIRE_GROW_STEPS = int(os.environ.get("RESQAI_FIRE_GROW_STEPS", "120"))
CAM_ALT = float(os.environ.get("RESQAI_CAM_ALT", "180"))

# Fire zone definitions (matching headless_e2e_test.py)
_FIRE_ZONES = [
    {"name": "FZ_0", "pos": (-94.8, -16.5, 0.15), "radius": 7.3,  "intensity": 0.72},
    {"name": "FZ_1", "pos": (-98.6,  46.5, 0.15), "radius": 5.8,  "intensity": 0.77},
    {"name": "FZ_2", "pos": ( -1.8,  46.5, 0.15), "radius": 7.3,  "intensity": 0.99},
    {"name": "FZ_3", "pos": ( 87.6, -16.5, 0.15), "radius": 4.2,  "intensity": 0.89},
    {"name": "FZ_4", "pos": ( 79.7,  46.5, 0.15), "radius": 6.5,  "intensity": 0.91},
]

_FLOW_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Flow extension helpers
# ═══════════════════════════════════════════════════════════════════════════

def _enable_flow_extension() -> bool:
    global _FLOW_AVAILABLE
    try:
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        mgr.set_extension_enabled_immediate("omni.usd.schema.flow", True)
        result = mgr.set_extension_enabled_immediate("omni.flowusd", True)
        if result:
            print("[Aerial] omni.flowusd extension enabled")
            _FLOW_AVAILABLE = True
            return True
        else:
            print("[Aerial] WARNING: omni.flowusd could not be enabled")
            return False
    except Exception as e:
        print(f"[Aerial] Flow extension load failed: {e}")
        return False


def _create_flow_fire_at_zone(stage, zone_idx: int) -> bool:
    """Create a Flow particle fire at the given zone index."""
    if not _FLOW_AVAILABLE:
        return False
    if zone_idx < 0 or zone_idx >= len(_FIRE_ZONES):
        return False

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
        print(f"[Aerial] Created Flow fire at {name} ({pos[0]:.1f}, {pos[1]:.1f})")
        return True
    except Exception as e:
        print(f"[Aerial] FlowCreatePresets failed for {name}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Procedural scene builder (copied from headless_e2e_test.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_procedural_scene() -> None:
    """Build the same procedural urban scene as the main simulation."""
    print("[Aerial] Building procedural urban disaster scene...")

    # Ground plane
    VisualCuboid(
        prim_path="/World/ProceduralGround",
        position=np.array([0.0, 0.0, -0.25]),
        scale=np.array([340.0, 170.0, 0.5]),
        color=np.array([0.35, 0.33, 0.30]),
    )

    # Standing buildings
    buildings = [
        ([70.0, 25.0, 8.0],   [15.0, 30.0, 16.0], [0.31, 0.28, 0.24]),
        ([50.0, 25.0, 10.0],  [12.0, 28.0, 20.0], [0.28, 0.26, 0.23]),
        ([90.0, 25.0, 7.5],   [14.0, 25.0, 15.0], [0.34, 0.30, 0.26]),
        ([-70.0, 25.0, 9.0],  [15.0, 28.0, 18.0], [0.30, 0.27, 0.24]),
        ([-50.0, 30.0, 7.5],  [12.0, 25.0, 15.0], [0.35, 0.32, 0.27]),
        ([-90.0, 30.0, 8.5],  [14.0, 30.0, 17.0], [0.33, 0.29, 0.25]),
        ([70.0, -25.0, 9.5],  [15.0, 30.0, 19.0], [0.29, 0.27, 0.23]),
        ([50.0, -30.0, 8.0],  [12.0, 28.0, 16.0], [0.32, 0.28, 0.24]),
        ([90.0, -20.0, 7.0],  [14.0, 25.0, 14.0], [0.36, 0.32, 0.28]),
        ([-70.0, -25.0, 8.5], [15.0, 30.0, 17.0], [0.30, 0.27, 0.23]),
        ([-50.0, -25.0, 10.0],[12.0, 28.0, 20.0], [0.28, 0.26, 0.22]),
        ([-90.0, -25.0, 7.0], [14.0, 25.0, 14.0], [0.34, 0.30, 0.26]),
        ([0.0, 0.0, 6.0],     [10.0, 10.0, 12.0], [0.29, 0.27, 0.23]),
        ([15.0, 10.0, 7.5],   [12.0, 12.0, 15.0], [0.31, 0.28, 0.24]),
        ([-15.0, -10.0, 8.0], [11.0, 11.0, 16.0], [0.33, 0.29, 0.25]),
        ([30.0, 0.0, 5.5],    [10.0, 15.0, 11.0], [0.35, 0.31, 0.27]),
        ([-30.0, 5.0, 6.5],   [10.0, 12.0, 13.0], [0.30, 0.28, 0.24]),
    ]
    for i, (pos, scale, color) in enumerate(buildings):
        VisualCuboid(
            prim_path=f"/World/Building_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # Collapsed building rubble piles
    collapses = [
        ([-95.0, -17.0, 1.0],  [18.0, 14.0, 2.0], [0.56, 0.48, 0.35]),
        ([-99.0,  46.0, 1.2],  [16.0, 12.0, 2.4], [0.52, 0.45, 0.33]),
        ([-2.0,   46.0, 0.8],  [14.0, 16.0, 1.6], [0.50, 0.44, 0.32]),
        ([88.0,  -17.0, 1.0],  [18.0, 12.0, 2.0], [0.54, 0.46, 0.34]),
        ([80.0,   46.0, 0.9],  [15.0, 14.0, 1.8], [0.51, 0.43, 0.31]),
        ([20.0,   0.0,  0.6],  [12.0, 10.0, 1.2], [0.48, 0.42, 0.30]),
        ([-40.0, -20.0, 0.7],  [14.0, 12.0, 1.4], [0.53, 0.46, 0.33]),
    ]
    for i, (pos, scale, color) in enumerate(collapses):
        VisualCuboid(
            prim_path=f"/World/Collapse_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # Flooded areas
    floods = [
        ([0.0, -40.0, 0.07],  [50.0, 25.0, 0.14], [0.10, 0.18, 0.45]),
        ([40.0,  0.0, 0.06],  [30.0, 40.0, 0.12], [0.08, 0.15, 0.40]),
        ([-60.0, 0.0, 0.08],  [35.0, 30.0, 0.16], [0.12, 0.20, 0.48]),
        ([0.0,  40.0, 0.05],  [40.0, 20.0, 0.10], [0.09, 0.16, 0.42]),
    ]
    for i, (pos, scale, color) in enumerate(floods):
        VisualCuboid(
            prim_path=f"/World/FloodZone_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # Vehicles
    vehicles = [
        ([25.0, -35.0, 0.8],  [4.5, 2.2, 1.6], [0.70, 0.10, 0.10]),
        ([-20.0, -38.0, 0.8], [4.5, 2.2, 1.6], [0.15, 0.30, 0.70]),
        ([55.0,   5.0, 0.8],  [4.5, 2.2, 1.6], [0.80, 0.80, 0.80]),
        ([-55.0, -5.0, 0.8],  [4.5, 2.2, 1.6], [0.10, 0.10, 0.10]),
        ([10.0, -42.0, 0.8],  [5.5, 2.5, 1.8], [0.90, 0.55, 0.00]),
        ([-35.0, 30.0, 0.8],  [4.5, 2.2, 1.6], [0.30, 0.60, 0.15]),
        ([65.0, -10.0, 0.8],  [6.0, 2.5, 2.0], [0.85, 0.80, 0.10]),
        ([-80.0, 10.0, 0.8],  [4.5, 2.2, 1.6], [0.50, 0.10, 0.50]),
        ([-93.0, -12.0, 0.5], [4.5, 2.2, 1.0], [0.60, 0.20, 0.05]),
        ([85.0, -20.0, 0.5],  [5.0, 2.5, 1.0], [0.40, 0.15, 0.10]),
    ]
    for i, (pos, scale, color) in enumerate(vehicles):
        VisualCuboid(
            prim_path=f"/World/Vehicle_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            scale=np.array(scale, dtype=np.float64),
            color=np.array(color),
        )

    # Survivors
    survivors = [
        ([-92.0, -15.0, 2.5], 2.0, [1.00, 0.25, 0.05]),
        ([-97.0,  48.0, 2.8], 2.0, [1.00, 0.15, 0.05]),
        ([0.0,    48.0, 2.2], 2.0, [1.00, 0.25, 0.05]),
        ([90.0,  -15.0, 2.5], 2.0, [1.00, 0.15, 0.05]),
        ([22.0,   1.0,  1.5], 2.0, [1.00, 0.25, 0.05]),
        ([-38.0, -18.0, 1.8], 2.0, [1.00, 0.15, 0.05]),
    ]
    for i, (pos, radius, color) in enumerate(survivors):
        VisualSphere(
            prim_path=f"/World/Survivor_{i:02d}",
            position=np.array(pos, dtype=np.float64),
            radius=float(radius),
            color=np.array(color),
        )
        VisualCuboid(
            prim_path=f"/World/SurvivorMarker_{i:02d}",
            position=np.array([pos[0], pos[1], pos[2] + 2.5], dtype=np.float64),
            size=1.5,
            color=np.array([1.0, 0.9, 0.0]),
        )

    # Lighting (if no USDA lights)
    try:
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/SunLight").IsValid():
            sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
            sun.CreateIntensityAttr(5000.0)
            sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
            xform_api = UsdGeom.XformCommonAPI(sun.GetPrim())
            xform_api.SetRotate(Gf.Vec3f(25.0, 0.0, 0.0))
        if not stage.GetPrimAtPath("/World/SkyDome").IsValid():
            dome = UsdLux.DomeLight.Define(stage, "/World/SkyDome")
            dome.CreateIntensityAttr(800.0)
            dome.CreateColorAttr(Gf.Vec3f(0.53, 0.70, 1.0))
    except Exception as e:
        print(f"[Aerial] Warning: could not add lights: {e}")

    print("[Aerial] Procedural scene built")


# ═══════════════════════════════════════════════════════════════════════════
# Static overhead camera
# ═══════════════════════════════════════════════════════════════════════════

def _setup_aerial_camera(stage, altitude: float) -> str:
    """Create a static overhead camera looking straight down at the scene center.
    Returns the camera prim path."""
    cam_path = "/World/AerialCamera"

    cam = UsdGeom.Camera.Define(stage, cam_path)
    # Wide-angle lens to capture all fire zones (-99..88 in X, -17..47 in Y)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.5, 2000.0))

    # Position: center of scene at [0, 15] (average of fire zone Y coords),
    # high altitude for full coverage
    xform = UsdGeom.Xformable(cam.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 15.0, altitude))

    # Looking straight down: camera -Z = world -Z, camera +Y = world +Y
    # This is identity rotation for USD camera convention
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    print(f"[Aerial] Static overhead camera at [0, 15, {altitude}], 18mm focal")
    return cam_path


def _capture_frame(annotator) -> np.ndarray | None:
    """Capture one frame from a render product annotator."""
    try:
        data = annotator.get_data()
        if data is None:
            return None
        arr = np.asarray(data)
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr[:, :, ::-1].copy()  # RGB→BGR for OpenCV
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  ResQ-AI — Aerial View Renderer (Real Flow Particle Fires)")
    print("=" * 70)
    print(f"  Scene   : {SCENE_PATH}")
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Warmup  : {WARMUP_STEPS} steps")
    print(f"  Fire grow: {FIRE_GROW_STEPS} steps")
    print(f"  Cam alt : {CAM_ALT}m")
    print()

    timeline = omni.timeline.get_timeline_interface()

    # ---- Load scene ----
    if SCENE_PATH and os.path.exists(SCENE_PATH):
        print(f"[Aerial] Loading scene from {SCENE_PATH}")
        omni.usd.get_context().open_stage(SCENE_PATH)
        simulation_app.update()
    else:
        print("[Aerial] No scene file — creating new stage")
        omni.usd.get_context().new_stage()
        simulation_app.update()

    # ---- Build procedural geometry ----
    _build_procedural_scene()
    simulation_app.update()

    # ---- Enable Flow and create ALL fires immediately ----
    _enable_flow_extension()
    stage = omni.usd.get_context().get_stage()

    n_created = 0
    for i in range(len(_FIRE_ZONES)):
        if _create_flow_fire_at_zone(stage, i):
            n_created += 1
    print(f"[Aerial] Created {n_created}/{len(_FIRE_ZONES)} Flow fires")

    # Also use FireManager to ignite all zones immediately for visual growth
    fire_mgr = None
    try:
        from sim_bridge.fire_system import FireManager
        fire_mgr = FireManager()
        for zname in fire_mgr._zones:
            fire_mgr._ignite_zone(zname)
        print(f"[Aerial] FireManager ignited all {len(fire_mgr._active)} zones")
    except Exception as e:
        print(f"[Aerial] FireManager: {e}")

    simulation_app.update()

    # ---- Setup aerial camera ----
    cam_path = _setup_aerial_camera(stage, CAM_ALT)
    simulation_app.update()

    # Create render product
    import omni.replicator.core as rep
    rp = rep.create.render_product(cam_path, (_RES_W, _RES_H))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([rp])
    simulation_app.update()

    # ---- Start timeline ----
    timeline.play()
    simulation_app.update()

    # ---- RTX warmup ----
    print(f"[Aerial] Running {WARMUP_STEPS} RTX warmup steps...")
    for i in range(WARMUP_STEPS):
        simulation_app.update()
        if (i + 1) % 20 == 0:
            print(f"  Warmup step {i + 1}/{WARMUP_STEPS}")

    # ---- Fire growth phase ----
    print(f"[Aerial] Running {FIRE_GROW_STEPS} fire growth steps...")
    frames_captured = []
    for i in range(FIRE_GROW_STEPS):
        simulation_app.update()

        # Capture intermediate frames at regular intervals
        if (i + 1) % 30 == 0 or i == FIRE_GROW_STEPS - 1:
            frame = _capture_frame(annotator)
            if frame is not None:
                step_path = os.path.join(output_dir, f"aerial_step_{i + 1:04d}.jpg")
                cv2.imwrite(step_path, frame)
                frames_captured.append(step_path)
                h, w = frame.shape[:2]
                nonzero = np.count_nonzero(frame.sum(axis=2) > 30)
                coverage = nonzero / (h * w)
                print(f"  Fire growth step {i + 1}/{FIRE_GROW_STEPS} — "
                      f"frame {w}x{h}, non-black coverage: {coverage:.1%}")

    # ---- Capture final aerial frame ----
    print("[Aerial] Capturing final aerial frame...")

    # Do a few extra render updates for final quality
    for _ in range(10):
        simulation_app.update()

    final_frame = _capture_frame(annotator)
    if final_frame is not None:
        cv2.imwrite(OUTPUT_PATH, final_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = final_frame.shape[:2]
        print(f"[Aerial] Saved final aerial image: {OUTPUT_PATH} ({w}x{h})")

        # Also save a copy at lower res for web display
        web_path = os.path.join(output_dir, "aerial_rendered_web.jpg")
        web_frame = cv2.resize(final_frame, (960, 540))
        cv2.imwrite(web_path, web_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[Aerial] Saved web version: {web_path}")
    else:
        print("[Aerial] ERROR: Could not capture final frame!")
        # Fallback: try to use last intermediate frame
        if frames_captured:
            import shutil
            shutil.copy(frames_captured[-1], OUTPUT_PATH)
            print(f"[Aerial] Using last intermediate frame as fallback")

    # ---- Cleanup ----
    timeline.stop()
    elapsed = time.time() - t_start
    print(f"\n[Aerial] Done in {elapsed:.1f}s")

    simulation_app.close()


if __name__ == "__main__":
    main()
