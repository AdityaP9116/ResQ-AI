#!/usr/bin/env python3
"""Generate a realistic 3D urban disaster environment for ResQ-AI.

Uses referenced CityEngine USD assets when available, falling back to
procedural buildings with OmniPBR materials, recessed windows, and roof
overhangs.  Vegetation uses UsdGeom.PointInstancer with four distinct
tree/bush prototypes arranged in a forest ring, city sidewalks, and
building edges.

Semantic class mapping
----------------------
building geometry  -> 'building'
road / asphalt     -> 'road'
trees / bushes     -> 'vegetation'
sidewalk / grass   -> 'terrain'
car / truck        -> 'vehicle'
fire meshes        -> 'fire'
pedestrian assets  -> 'person'

Usage (Isaac Sim python environment)::

    /isaac-sim/python.sh sim_bridge/generate_urban_scene.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResQ-AI urban disaster scene generator")
    p.add_argument("--headless", action="store_true",
                   help="Run without a viewport window")
    p.add_argument("--num-pedestrians", type=int, default=12,
                   help="Number of indoor victims to spawn")
    p.add_argument("--wind-direction", type=float, nargs=3,
                   default=[1.0, 0.0, 0.0],
                   help="Wind direction vector (X Y Z), default: 1 0 0")
    p.add_argument("--wind-speed", type=float, default=5.0,
                   help="Wind speed in m/s (affects fire spread bias)")
    p.add_argument("--fire-spread-radius", type=float, default=35.0,
                   help="Distance threshold (m) for fire spreading")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after N physics steps (0 = unlimited)")
    return p.parse_args()


_args = _parse_args()
simulation_app = SimulationApp({"headless": _args.headless})

import omni.usd
import omni.timeline
from omni.isaac.core import World
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CITY_ASSEMBLY_USD = (
    PROJECT_ROOT
    / "assets" / "CityDemo" / "Demos" / "AEC" / "TowerDemo"
    / "CityDemopack" / "Assemblies" / "assembly_City.usd"
)
ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1"
)

# ---------------------------------------------------------------------------
# Local character asset paths
# ---------------------------------------------------------------------------

_CHARACTER_USDS = [
    str(PROJECT_ROOT / "assets" / "Characters" / "Assets" / "Characters"
        / "Reallusion" / "Worker" / "Worker.usd"),
    str(PROJECT_ROOT / "assets" / "Characters" / "Assets" / "Characters"
        / "Reallusion" / "ActorCore" / "Business_F_0002" / "Actor"
        / "business-f-0002" / "business-f-0002.usd"),
    str(PROJECT_ROOT / "assets" / "Characters" / "Assets" / "Characters"
        / "Reallusion" / "ActorCore" / "Uniform_F_0001" / "Actor"
        / "uniform_f_0001" / "uniform_f_0001.usd"),
    str(PROJECT_ROOT / "assets" / "Characters" / "Assets" / "Characters"
        / "Reallusion" / "ActorCore" / "Uniform_M_0001" / "Actor"
        / "uniform_m_0001" / "uniform_m_0001.usd"),
    str(PROJECT_ROOT / "assets" / "Characters" / "Assets" / "Characters"
        / "Reallusion" / "ActorCore" / "Party_M_0001" / "Actor"
        / "party-m-0001" / "party-m-0001.usd"),
]

# ---------------------------------------------------------------------------
# HDRI sky & Flow preset paths
# ---------------------------------------------------------------------------

_HDRI_DIR = PROJECT_ROOT / "assets" / "Environments" / "Assets" / "Skies"
_FLOW_SAMPLES_DIR = (
    PROJECT_ROOT / "assets" / "Particles" / "Assets" / "Extensions"
    / "Samples" / "Flow" / "samples"
)

# Optional anim extensions
try:
    import omni.anim.people as _anim_people
    HAS_ANIM_PEOPLE = True
except (ModuleNotFoundError, ImportError):
    _anim_people = None
    HAS_ANIM_PEOPLE = False

try:
    import omni.anim.navigation as _anim_nav
    HAS_ANIM_NAV = True
except (ModuleNotFoundError, ImportError):
    _anim_nav = None
    HAS_ANIM_NAV = False

# ---------------------------------------------------------------------------
# Semantic labelling
# ---------------------------------------------------------------------------

def _apply_semantic_label(prim: Usd.Prim, label: str) -> None:
    try:
        import omni.replicator.core.functional as rep_f
        rep_f.modify.semantics(prim, {"class": [label]}, mode="replace")
        return
    except (ModuleNotFoundError, ImportError, AttributeError):
        pass
    try:
        import Semantics
        sem = Semantics.SemanticsAPI.Apply(prim, f"class_{label}")
        sem.CreateSemanticTypeAttr().Set(f"class_{label}")
        sem.CreateSemanticDataAttr().Set(label)
    except Exception:
        pass


def _label_recursive(stage: Usd.Stage, root_path: str, label: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Xformable):
            _apply_semantic_label(prim, label)

# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------

def _set_xform(xformable: UsdGeom.Xformable,
               translate: tuple = (0, 0, 0),
               rotate_deg: tuple = (0, 0, 0),
               scale: tuple = (1, 1, 1)):
    # Clear any pre-existing ops to avoid "already exists" errors
    if xformable.GetOrderedXformOps():
        xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_deg))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))

# ---------------------------------------------------------------------------
# Prim creation
# ---------------------------------------------------------------------------

def _xform(stage, path, **kw):
    xf = UsdGeom.Xform.Define(stage, path)
    _set_xform(xf, **kw)
    return xf

def _cube(stage, path, size=1.0, **kw):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(size)
    _set_xform(c, **kw)
    return c

def _cylinder(stage, path, radius=0.5, height=1.0, axis="Z", **kw):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    _set_xform(c, **kw)
    return c

def _sphere(stage, path, radius=0.5, **kw):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    _set_xform(s, **kw)
    return s

def _cone(stage, path, radius=0.5, height=1.0, axis="Z", **kw):
    c = UsdGeom.Cone.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    _set_xform(c, **kw)
    return c

# ---------------------------------------------------------------------------
# OmniPBR material
# ---------------------------------------------------------------------------

_mat_cache: dict[str, str] = {}

def _mat(stage: Usd.Stage, name: str, *,
         albedo: tuple = (0.5, 0.5, 0.5),
         roughness: float = 0.5,
         metallic: float = 0.0,
         emissive: tuple = (0.0, 0.0, 0.0),
         emissive_intensity: float = 0.0,
         opacity: float = 1.0) -> str:
    if name in _mat_cache:
        return _mat_cache[name]
    mat_path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = f"{mat_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)

    use_opacity = opacity < 1.0
    mdl_file = "OmniPBR_Opacity.mdl" if use_opacity else "OmniPBR.mdl"
    mdl_name = "OmniPBR_Opacity" if use_opacity else "OmniPBR"

    shader.GetPrim().CreateAttribute(
        "info:implementationSource", Sdf.ValueTypeNames.Token, True).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, True).Set(mdl_file)
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, True).Set(mdl_name)

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*albedo))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)

    if use_opacity:
        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive_intensity > 0:
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(emissive_intensity)

    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    _mat_cache[name] = mat_path
    return mat_path


def _bind(stage, prim_path: str, mat_path: str):
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)

# ---------------------------------------------------------------------------
# Terrain — 200m × 200m with grass + sidewalks + physics collision
# ---------------------------------------------------------------------------

def _build_terrain(stage):
    _xform(stage, "/World/Terrain")

    # Main ground — 400×400 m, slightly below origin
    ground = _cube(stage, "/World/Terrain/Ground", size=1.0,
                   translate=(0, 0, -0.25), scale=(400, 400, 0.5))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    grass = _mat(stage, "GrassMat", albedo=(0.12, 0.26, 0.08), roughness=0.95)
    _bind(stage, "/World/Terrain/Ground", grass)

    # Soil/dirt underlayer (visible at edges)
    dirt = _mat(stage, "DirtMat", albedo=(0.35, 0.25, 0.15), roughness=0.92)
    _cube(stage, "/World/Terrain/DirtLayer", size=1.0,
          translate=(0, 0, -1.5), scale=(420, 420, 2.5))
    _bind(stage, "/World/Terrain/DirtLayer", dirt)

    # Terrain rim hills — 8 mounds along edges to break flat horizon
    hill_mat = _mat(stage, "HillMat", albedo=(0.14, 0.22, 0.08), roughness=0.9)
    random.seed(999)
    for i in range(8):
        angle = i * (2 * math.pi / 8) + random.uniform(-0.2, 0.2)
        dist = random.uniform(150, 180)
        hx = dist * math.cos(angle)
        hy = dist * math.sin(angle)
        sx = random.uniform(40, 80)
        sy = random.uniform(40, 80)
        sz = random.uniform(4, 12)
        _sphere(stage, f"/World/Terrain/Hill_{i}", radius=1.0,
                translate=(hx, hy, sz * 0.2), scale=(sx, sy, sz))
        _bind(stage, f"/World/Terrain/Hill_{i}", hill_mat)

    # Sidewalks along main road
    sw = _mat(stage, "SidewalkMat", albedo=(0.6, 0.6, 0.55), roughness=0.8)
    for tag, tx, ty, ssx, ssy in [
        ("N",  0,  8, 200, 4), ("S", 0, -8, 200, 4),
        ("E",  8,  0, 4, 200), ("W", -8, 0, 4, 200),
    ]:
        _cube(stage, f"/World/Terrain/Sidewalk_{tag}", size=1.0,
              translate=(tx, ty, 0.03), scale=(ssx, ssy, 0.06))
        _bind(stage, f"/World/Terrain/Sidewalk_{tag}", sw)

    _label_recursive(stage, "/World/Terrain", "terrain")

# ---------------------------------------------------------------------------
# CityEngine reference
# ---------------------------------------------------------------------------

def _build_city_reference(stage) -> bool:
    """Reference CityEngine assembly USD. Returns True on success."""
    if not CITY_ASSEMBLY_USD.exists():
        print(f"[ResQ-AI] CityEngine USD not found at {CITY_ASSEMBLY_USD}")
        return False
    try:
        ref_prim = stage.DefinePrim("/World/CityReference", "Xform")
        ref_prim.GetReferences().AddReference(
            str(CITY_ASSEMBLY_USD.as_posix()))
        # Quick validity check
        if not ref_prim.IsValid() or not ref_prim.GetChildren():
            print("[ResQ-AI] CityEngine reference loaded but has no children — using fallback")
            stage.RemovePrim("/World/CityReference")
            return False
        print("[ResQ-AI] CityEngine city referenced successfully")
        return True
    except Exception as exc:
        print(f"[ResQ-AI] CityEngine reference failed ({exc}) — using fallback")
        try:
            stage.RemovePrim("/World/CityReference")
        except Exception:
            pass
        return False


def _discover_city_buildings(stage):
    """Traverse CityReference and alias buildings under /World/Buildings."""
    _xform(stage, "/World/Buildings")
    ref = stage.GetPrimAtPath("/World/CityReference")
    if not ref.IsValid():
        return
    idx = 0
    for prim in Usd.PrimRange(ref):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetName()
        if not any(k in name.lower() for k in ("bldg", "building", "lot", "block", "house")):
            continue
        # Compute bounding box
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        sx = mx[0] - mn[0]
        sy = mx[1] - mn[1]
        h  = mx[2] - mn[2]
        cx = (mn[0] + mx[0]) / 2.0
        cy = (mn[1] + mx[1]) / 2.0
        if sx < 1 or sy < 1 or h < 1:
            continue

        bld_name = f"CityBldg_{idx:03d}"
        bp = f"/World/Buildings/{bld_name}"
        _xform(stage, bp, translate=(cx, cy, 0))
        body = _xform(stage, f"{bp}/Body", scale=(max(sx, 1), max(sy, 1), max(h, 1)))
        _apply_semantic_label(stage.GetPrimAtPath(bp), "building")
        idx += 1

    if idx == 0:
        print("[ResQ-AI] No buildings discovered in CityEngine scene — using fallback")
    else:
        print(f"[ResQ-AI] Aliased {idx} CityEngine buildings under /World/Buildings")
    return idx

# ---------------------------------------------------------------------------
# Fallback buildings with OmniPBR facades & recessed windows
# ---------------------------------------------------------------------------

_WALL_MATS = None

def _get_wall_materials(stage):
    global _WALL_MATS
    if _WALL_MATS is None:
        _WALL_MATS = {
            "concrete": _mat(stage, "WallConcrete", albedo=(0.52, 0.50, 0.47), roughness=0.85),
            "brick":    _mat(stage, "WallBrick",    albedo=(0.55, 0.25, 0.18), roughness=0.80),
            "sandstone":_mat(stage, "WallSandstone", albedo=(0.65, 0.55, 0.40), roughness=0.82),
        }
    return _WALL_MATS


def _build_buildings_fallback(stage):
    root = "/World/Buildings"
    if not stage.GetPrimAtPath(root).IsValid():
        _xform(stage, root)

    walls = _get_wall_materials(stage)
    glass = _mat(stage, "WindowGlass", albedo=(0.1, 0.15, 0.2),
                 metallic=0.7, roughness=0.05, opacity=0.5)
    roof_m = _mat(stage, "RoofDark", albedo=(0.25, 0.23, 0.21), roughness=0.75)

    wall_keys = list(walls.keys())
    specs = [
        ("OfficeA",  30,  30, 16, 12, 32), ("TowerB",   58,  30, 10, 14, 55),
        ("AptA",     30,  58, 14, 14, 24), ("AptB",     58,  58, 12, 10, 18),
        ("OfficeC", -30,  30, 18, 14, 40), ("ShopA",   -58,  30, 10,  8, 10),
        ("AptC",    -30,  58, 14, 12, 26), ("OfficeD", -58,  58, 12, 12, 38),
        ("OfficeE",  30, -30, 14, 16, 30), ("ShopB",    58, -30,  8, 10, 10),
        ("AptD",     30, -58, 14, 14, 20), ("Warehouse",58, -58, 18, 18, 12),
        ("OfficeF", -30, -30, 16, 12, 34), ("AptE",    -58, -30, 12, 10, 16),
        ("ShopC",   -30, -58, 10, 10,  8), ("AptF",    -58, -58, 14, 14, 22),
    ]

    for bi, (name, x, y, sx, sy, h) in enumerate(specs):
        bp = f"{root}/{name}"
        wmat = walls[wall_keys[bi % len(wall_keys)]]
        _xform(stage, bp, translate=(x, y, 0))
        _cube(stage, f"{bp}/Body", size=1.0,
              translate=(0, 0, h / 2.0), scale=(sx, sy, h))
        _bind(stage, f"{bp}/Body", wmat)

        # Roof with overhang
        _cube(stage, f"{bp}/Roof", size=1.0,
              translate=(0, 0, h + 0.15), scale=(sx + 0.6, sy + 0.6, 0.3))
        _bind(stage, f"{bp}/Roof", roof_m)

        # Recessed windows on two faces at 4m intervals
        if h > 14:
            floors = int(h / 4)
            inset = 0.3
            for fi, (dx, dy, wsx, wsy) in enumerate([
                (sx / 2.0 + 0.01, 0, inset, sy * 0.8),
                (0, sy / 2.0 + 0.01, sx * 0.8, inset),
            ]):
                for fl in range(floors):
                    fz = 3.0 + fl * (h - 4.0) / max(floors - 1, 1)
                    _cube(stage, f"{bp}/Win_{fi}_{fl}", size=1.0,
                          translate=(dx, dy, fz), scale=(wsx, wsy, 1.8))
                    _bind(stage, f"{bp}/Win_{fi}_{fl}", glass)

    _label_recursive(stage, root, "building")

# ---------------------------------------------------------------------------
# Roads — asphalt grid with lane markings
# ---------------------------------------------------------------------------

def _build_roads(stage, city_ref_ok: bool):
    root = "/World/Roads"
    _xform(stage, root)
    if city_ref_ok:
        # Roads included in CityEngine reference
        _label_recursive(stage, "/World/CityReference", "road")
        return

    asphalt = _mat(stage, "AsphaltMat", albedo=(0.15, 0.15, 0.15), roughness=0.85)
    _cube(stage, f"{root}/EW", size=1.0, translate=(0, 0, 0.01), scale=(200, 14, 0.06))
    _bind(stage, f"{root}/EW", asphalt)
    _cube(stage, f"{root}/NS", size=1.0, translate=(0, 0, 0.01), scale=(14, 200, 0.06))
    _bind(stage, f"{root}/NS", asphalt)

    marking = _mat(stage, "MarkingMat", albedo=(0.95, 0.85, 0.15), roughness=0.6)
    for i, off in enumerate(range(-90, 91, 12)):
        _cube(stage, f"{root}/MarkEW_{i}", size=1.0,
              translate=(off, 0, 0.05), scale=(5, 0.25, 0.02))
        _bind(stage, f"{root}/MarkEW_{i}", marking)
    for i, off in enumerate(range(-90, 91, 12)):
        _cube(stage, f"{root}/MarkNS_{i}", size=1.0,
              translate=(0, off, 0.05), scale=(0.25, 5, 0.02))
        _bind(stage, f"{root}/MarkNS_{i}", marking)

    _label_recursive(stage, root, "road")

# ---------------------------------------------------------------------------
# Vegetation — PointInstancer with 4 prototypes
# ---------------------------------------------------------------------------

def _build_vegetation(stage):
    root = "/World/Vegetation"
    _xform(stage, root)

    # Materials
    bark   = _mat(stage, "BarkMat",       albedo=(0.28, 0.16, 0.08), roughness=0.92)
    green1 = _mat(stage, "CanopyDark",    albedo=(0.08, 0.28, 0.06), roughness=0.85)
    green2 = _mat(stage, "CanopyLime",    albedo=(0.25, 0.50, 0.12), roughness=0.80)
    green3 = _mat(stage, "CanopyOlive",   albedo=(0.18, 0.30, 0.10), roughness=0.82)
    greens = [green1, green2, green3]

    inst_path = f"{root}/TreeInstancer"
    instancer = UsdGeom.PointInstancer.Define(stage, inst_path)
    proto_rel = instancer.CreatePrototypesRel()
    proto_root = f"{inst_path}/prototypes"
    _xform(stage, proto_root)

    # --- Prototype A: Tall Pine ---
    pa = f"{proto_root}/Pine"
    _xform(stage, pa)
    _cylinder(stage, f"{pa}/Trunk", radius=0.15, height=5.0, translate=(0, 0, 2.5))
    _bind(stage, f"{pa}/Trunk", bark)
    _cone(stage, f"{pa}/Canopy", radius=1.8, height=4.0, translate=(0, 0, 7.0))
    _bind(stage, f"{pa}/Canopy", green1)
    proto_rel.AddTarget(pa)

    # --- Prototype B: Oak ---
    pb = f"{proto_root}/Oak"
    _xform(stage, pb)
    _cylinder(stage, f"{pb}/Trunk", radius=0.25, height=3.0, translate=(0, 0, 1.5))
    _bind(stage, f"{pb}/Trunk", bark)
    _sphere(stage, f"{pb}/Canopy", radius=2.5, translate=(0, 0, 5.5))
    _bind(stage, f"{pb}/Canopy", green2)
    proto_rel.AddTarget(pb)

    # --- Prototype C: Bush ---
    pc = f"{proto_root}/Bush"
    _xform(stage, pc)
    _sphere(stage, f"{pc}/Canopy", radius=1.2, translate=(0, 0, 0.72),
            scale=(1, 1, 0.6))
    _bind(stage, f"{pc}/Canopy", green3)
    proto_rel.AddTarget(pc)

    # --- Prototype D: Tall thin tree ---
    pd = f"{proto_root}/TallThin"
    _xform(stage, pd)
    _cylinder(stage, f"{pd}/Trunk", radius=0.1, height=6.0, translate=(0, 0, 3.0))
    _bind(stage, f"{pd}/Trunk", bark)
    _sphere(stage, f"{pd}/Canopy", radius=1.0, translate=(0, 0, 7.0))
    _bind(stage, f"{pd}/Canopy", green1)
    proto_rel.AddTarget(pd)

    # Prototype indices: 0=Pine, 1=Oak, 2=Bush, 3=TallThin
    positions = []
    proto_indices = []
    orientations = []
    scales = []
    random.seed(2024)

    # Building footprint rects to avoid  (x, y, half_sx, half_sy)
    bldg_rects = [
        (30, 30, 8, 6), (58, 30, 5, 7), (30, 58, 7, 7), (58, 58, 6, 5),
        (-30, 30, 9, 7), (-58, 30, 5, 4), (-30, 58, 7, 6), (-58, 58, 6, 6),
        (30, -30, 7, 8), (58, -30, 4, 5), (30, -58, 7, 7), (58, -58, 9, 9),
        (-30, -30, 8, 6), (-58, -30, 6, 5), (-30, -58, 5, 5), (-58, -58, 7, 7),
    ]

    def _in_road(px, py):
        return abs(px) < 8 or abs(py) < 8

    def _in_building(px, py):
        for bx, by, hx, hy in bldg_rects:
            if abs(px - bx) < hx + 2 and abs(py - by) < hy + 2:
                return True
        return False

    def _add(px, py, proto, sc):
        if _in_building(px, py) or _in_road(px, py):
            return
        positions.append(Gf.Vec3f(px, py, 0))
        proto_indices.append(proto)
        yaw = random.uniform(0, 360)
        rad = math.radians(yaw / 2.0)
        orientations.append(Gf.Quath(math.cos(rad), 0, 0, math.sin(rad)))
        scales.append(Gf.Vec3f(sc, sc, sc))

    # FOREST RING: 400 trees, radius 80-140m, mostly pines
    for _ in range(400):
        r = random.uniform(80, 140)
        a = random.uniform(0, 2 * math.pi)
        px, py = r * math.cos(a), r * math.sin(a)
        proto = random.choices([0, 3], weights=[0.7, 0.3])[0]
        _add(px, py, proto, random.uniform(0.8, 1.5))

    # CITY TREES: 60 oaks along sidewalks, r=15-70m
    for _ in range(60):
        r = random.uniform(15, 70)
        a = random.uniform(0, 2 * math.pi)
        px, py = r * math.cos(a), r * math.sin(a)
        proto = random.choices([1, 0], weights=[0.8, 0.2])[0]
        _add(px, py, proto, random.uniform(0.6, 1.2))

    # BUSHES: 80 near building edges
    for _ in range(80):
        r = random.uniform(12, 72)
        a = random.uniform(0, 2 * math.pi)
        px, py = r * math.cos(a), r * math.sin(a)
        _add(px, py, 2, random.uniform(0.6, 1.3))

    instancer.CreatePositionsAttr(positions)
    instancer.CreateProtoIndicesAttr(Vt.IntArray(proto_indices))
    instancer.CreateOrientationsAttr(orientations)
    instancer.CreateScalesAttr(scales)

    _label_recursive(stage, root, "vegetation")
    print(f"[ResQ-AI] Placed {len(positions)} trees (4 types).")
    return positions

# ---------------------------------------------------------------------------
# Lighting — Sun + Sky dome + 4 corner fill lights
# ---------------------------------------------------------------------------

def _build_lighting(stage):
    root = "/World/Lights"
    _xform(stage, root)

    # Sun — DistantLight ~30° elevation from southeast
    sun = UsdLux.DistantLight.Define(stage, f"{root}/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    _set_xform(sun, translate=(0, 0, 100), rotate_deg=(-60, 0, -45))

    # Sky dome — prefer local HDRI from Environments asset pack
    dome = UsdLux.DomeLight.Define(stage, f"{root}/Sky")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.7, 0.75, 0.85))

    hdri_loaded = False
    # Storm/dramatic sky for disaster atmosphere
    storm_candidates = [
        _HDRI_DIR / "2022_1" / "Skies" / "Storm" / "approaching_storm.hdr",
        _HDRI_DIR / "Cloudy" / "champagne_castle_1_4k.hdr",
        _HDRI_DIR / "Clear" / "noon_grass_4k.hdr",
        _HDRI_DIR / "2022_1" / "Skies" / "Cloudy" / "kloofendal_48d_partly_cloudy.hdr",
    ]
    for hdr_path in storm_candidates:
        if hdr_path.exists():
            dome.CreateTextureFileAttr(str(hdr_path))
            print(f"[ResQ-AI] HDRI sky: {hdr_path.name}")
            hdri_loaded = True
            break

    if not hdri_loaded:
        # Fallback: try any 4K HDR in the environments pack
        if _HDRI_DIR.exists():
            for hdr in _HDRI_DIR.rglob("*.hdr"):
                dome.CreateTextureFileAttr(str(hdr))
                print(f"[ResQ-AI] HDRI sky (fallback): {hdr.name}")
                hdri_loaded = True
                break

    if not hdri_loaded:
        # Last resort: remote URL
        try:
            dome.CreateTextureFileAttr(
                f"{ASSET_ROOT}/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr")
            print("[ResQ-AI] HDRI sky: remote URL")
        except Exception:
            print("[ResQ-AI] WARNING: No HDRI sky loaded")

    # 4× SphereLight fill at corners
    corners = [(80, 80), (-80, 80), (80, -80), (-80, -80)]
    for i, (cx, cy) in enumerate(corners):
        fl = UsdLux.SphereLight.Define(stage, f"{root}/Fill_{i}")
        fl.CreateIntensityAttr(500.0)
        fl.CreateRadiusAttr(2.0)
        fl.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 0.95))
        _set_xform(fl, translate=(cx, cy, 40))

# ---------------------------------------------------------------------------
# Vehicles — 8-12 with varied paints
# ---------------------------------------------------------------------------

def _build_vehicles(stage):
    root = "/World/Vehicles"
    _xform(stage, root)

    paints = {
        "Red":    _mat(stage, "PaintRed",    albedo=(0.65, 0.08, 0.08), roughness=0.3, metallic=0.7),
        "Blue":   _mat(stage, "PaintBlue",   albedo=(0.08, 0.15, 0.65), roughness=0.3, metallic=0.7),
        "White":  _mat(stage, "PaintWhite",  albedo=(0.88, 0.88, 0.86), roughness=0.25, metallic=0.6),
        "Yellow": _mat(stage, "PaintYellow", albedo=(0.82, 0.72, 0.08), roughness=0.3, metallic=0.65),
        "Black":  _mat(stage, "PaintBlack",  albedo=(0.05, 0.05, 0.06), roughness=0.2, metallic=0.8),
        "Silver": _mat(stage, "PaintSilver", albedo=(0.6, 0.6, 0.62),   roughness=0.2, metallic=0.85),
    }
    tire   = _mat(stage, "TireMat",       albedo=(0.03, 0.03, 0.03), roughness=0.95)
    chrome = _mat(stage, "ChromeMat",     albedo=(0.7, 0.7, 0.72),   roughness=0.08, metallic=1.0)
    wind   = _mat(stage, "WindshieldMat", albedo=(0.25, 0.4, 0.55),  roughness=0.05, metallic=0.4, opacity=0.4)

    cars = [
        ("Car01",  22,   7.5, 0,   "Red"),   ("Car02",  38,   7.5, 0,   "Blue"),
        ("Car03",  54,  -7.5, 180, "White"),  ("Car04", -22,   7.5, 0,   "Yellow"),
        ("Car05", -42,  -7.5, 180, "Black"),  ("Car06",  7.5,  28,  90,  "Silver"),
        ("Car07", -7.5,  44,  270, "Blue"),   ("Car08",  7.5, -28,  90,  "White"),
        ("Car09", -7.5, -48,  270, "Yellow"), ("Car10", -68,   7.5, 0,   "Black"),
        ("Car11",  74,  -7.5, 180, "Red"),    ("Car12",  7.5,  74,  90,  "Silver"),
    ]

    for vid, x, y, rz, color in cars:
        vp = f"{root}/{vid}"
        _xform(stage, vp, translate=(x, y, 0), rotate_deg=(0, 0, rz))
        _cube(stage, f"{vp}/Body", size=1.0, translate=(0, 0, 0.55), scale=(4.4, 1.85, 0.75))
        _bind(stage, f"{vp}/Body", paints[color])
        _cube(stage, f"{vp}/Cabin", size=1.0, translate=(0.2, 0, 1.15), scale=(2.4, 1.7, 0.65))
        _bind(stage, f"{vp}/Cabin", wind)
        _cube(stage, f"{vp}/Hood", size=1.0, translate=(-1.5, 0, 0.85), scale=(1.2, 1.75, 0.12))
        _bind(stage, f"{vp}/Hood", paints[color])
        _cube(stage, f"{vp}/Bumper_F", size=1.0, translate=(-2.25, 0, 0.35), scale=(0.2, 1.6, 0.3))
        _bind(stage, f"{vp}/Bumper_F", chrome)
        _cube(stage, f"{vp}/Bumper_R", size=1.0, translate=(2.25, 0, 0.35), scale=(0.2, 1.6, 0.3))
        _bind(stage, f"{vp}/Bumper_R", chrome)
        for wi, (wx, wy) in enumerate([(-1.3, 1.0), (-1.3, -1.0), (1.3, 1.0), (1.3, -1.0)]):
            _cylinder(stage, f"{vp}/Wheel_{wi}", radius=0.34, height=0.22, axis="Y",
                      translate=(wx, wy, 0.34))
            _bind(stage, f"{vp}/Wheel_{wi}", tire)
            _cylinder(stage, f"{vp}/Hub_{wi}", radius=0.18, height=0.24, axis="Y",
                      translate=(wx, wy, 0.34))
            _bind(stage, f"{vp}/Hub_{wi}", chrome)

    _label_recursive(stage, root, "vehicle")

# ---------------------------------------------------------------------------
# Disaster zones
# ---------------------------------------------------------------------------

def _build_disaster_collapsed(stage):
    dz = "/World/DisasterZones/CollapsedBuilding"
    _xform(stage, "/World/DisasterZones")
    _xform(stage, dz, translate=(-30, -58, 0))

    rubble = _mat(stage, "RubbleMat", albedo=(0.42, 0.40, 0.36), roughness=0.92)
    rebar  = _mat(stage, "RebarMat",  albedo=(0.32, 0.20, 0.12), roughness=0.7, metallic=0.6)

    walls = [(-3, 0, 3.5, 8, 0.5, 7, 0), (3, 2, 2.5, 0.5, 6, 5, 0),
             (0, -3, 1.5, 6, 0.5, 3, 15)]
    for i, (dx, dy, dz_off, sx, sy, sz, rz) in enumerate(walls):
        _cube(stage, f"{dz}/Wall_{i}", size=1.0,
              translate=(dx, dy, dz_off), scale=(sx, sy, sz), rotate_deg=(0, 0, rz))
        _bind(stage, f"{dz}/Wall_{i}", rubble)

    random.seed(42)
    for i in range(25):
        s = random.uniform(0.3, 2.0)
        _cube(stage, f"{dz}/Rubble_{i:02d}", size=1.0,
              translate=(random.uniform(-7, 7), random.uniform(-7, 7), s / 2),
              scale=(s, s * random.uniform(0.4, 1.5), s * random.uniform(0.2, 1.0)),
              rotate_deg=(random.uniform(-30, 30), random.uniform(-30, 30), random.uniform(0, 360)))
        _bind(stage, f"{dz}/Rubble_{i:02d}", rubble if i % 3 else rebar)

    for i in range(8):
        _cylinder(stage, f"{dz}/Rebar_{i}", radius=0.04, height=random.uniform(2, 6),
                  translate=(random.uniform(-6, 6), random.uniform(-6, 6), 0.5),
                  rotate_deg=(random.uniform(-45, 45), random.uniform(-45, 45), 0))
        _bind(stage, f"{dz}/Rebar_{i}", rebar)

    _label_recursive(stage, dz, "building")


def _build_disaster_fire(stage):
    dz = "/World/DisasterZones/VehicleFire"
    _xform(stage, dz, translate=(54, -7.5, 0))

    burnt = _mat(stage, "BurntMetalMat", albedo=(0.08, 0.06, 0.04), roughness=0.85, metallic=0.3)
    _xform(stage, f"{dz}/BurntCar")
    _cube(stage, f"{dz}/BurntCar/Body", size=1.0, translate=(0, 0, 0.55), scale=(4.4, 1.85, 0.75))
    _bind(stage, f"{dz}/BurntCar/Body", burnt)
    _cube(stage, f"{dz}/BurntCar/Cabin", size=1.0, translate=(0.2, 0, 1.15), scale=(2.2, 1.6, 0.5))
    _bind(stage, f"{dz}/BurntCar/Cabin", burnt)
    _label_recursive(stage, f"{dz}/BurntCar", "vehicle")

    fire_root = f"{dz}/Flames"
    _xform(stage, fire_root, translate=(0, 0, 1.0))
    flame_outer = _mat(stage, "FlameOuter", albedo=(1.0, 0.3, 0.0),
                       emissive=(1.0, 0.4, 0.0), emissive_intensity=5000.0, roughness=1.0)
    flame_core  = _mat(stage, "FlameCore", albedo=(1.0, 0.9, 0.3),
                       emissive=(1.0, 0.9, 0.3), emissive_intensity=8000.0, roughness=1.0)

    flames = [
        ("Core",  "cone",   dict(radius=0.5,  height=2.8), (0, 0, 1.4),      (1,1,1),       (0,0,0),   True),
        ("Left",  "cone",   dict(radius=0.38, height=2.0), (-0.6, 0.3, 1.0), (1,1,1),       (0,0,12),  False),
        ("Right", "cone",   dict(radius=0.38, height=2.0), (0.5, -0.2, 1.0), (1,1,1),       (0,0,-10), False),
        ("Back",  "cone",   dict(radius=0.3,  height=1.6), (-0.9, 0, 0.8),   (1,1,1),       (0,0,18),  False),
        ("EmberA","sphere", dict(radius=0.22),              (0.3, 0.4, 2.6),  (1,1,1),       (0,0,0),   True),
        ("EmberB","sphere", dict(radius=0.18),              (-0.4,-0.3,2.8),  (1,1,1),       (0,0,0),   False),
        ("EmberC","sphere", dict(radius=0.15),              (0.1, 0.1, 3.1),  (1,1,1),       (0,0,0),   True),
        ("Glow",  "sphere", dict(radius=0.9),               (0, 0, 0.2),     (1.6,1.6,0.4), (0,0,0),   True),
    ]
    for fname, shape, params, t, s, r, is_core in flames:
        fp = f"{fire_root}/{fname}"
        if shape == "cone":
            _cone(stage, fp, **params, translate=t, scale=s, rotate_deg=r)
        else:
            _sphere(stage, fp, **params, translate=t, scale=s, rotate_deg=r)
        _bind(stage, fp, flame_core if is_core else flame_outer)

    smoke = _mat(stage, "SmokeMat", albedo=(0.12, 0.12, 0.12), roughness=1.0, opacity=0.3)
    for i in range(6):
        rad = 0.5 + i * 0.35
        _sphere(stage, f"{fire_root}/Smoke_{i}", radius=rad,
                translate=(i * 0.18, i * 0.12, 3.2 + i * 1.3))
        _bind(stage, f"{fire_root}/Smoke_{i}", smoke)

    _label_recursive(stage, fire_root, "fire")

# ---------------------------------------------------------------------------
# Street furniture
# ---------------------------------------------------------------------------

def _build_street_props(stage):
    root = "/World/StreetProps"
    _xform(stage, root)
    pole_mat   = _mat(stage, "PoleMat",    albedo=(0.4, 0.4, 0.42),  roughness=0.4, metallic=0.9)
    hydrant_m  = _mat(stage, "HydrantMat", albedo=(0.7, 0.15, 0.1),  roughness=0.5, metallic=0.7)
    barrier_m  = _mat(stage, "BarrierMat", albedo=(0.9, 0.45, 0.0),  roughness=0.7)
    dumpster_m = _mat(stage, "DumpsterMat",albedo=(0.15, 0.3, 0.15), roughness=0.65, metallic=0.5)

    for i, (x, y) in enumerate([
        (20, 10), (50, 10), (-20, 10), (-50, 10),
        (20,-10), (50,-10), (-20,-10), (-50,-10),
        (10, 20), (10, 50), (-10, 20), (-10, 50),
    ]):
        _cylinder(stage, f"{root}/Pole_{i}", radius=0.08, height=6.0, translate=(x, y, 3.0))
        _bind(stage, f"{root}/Pole_{i}", pole_mat)
        _sphere(stage, f"{root}/Lamp_{i}", radius=0.25, translate=(x, y, 6.2))
        lamp_m = _mat(stage, f"LampGlow_{i}", albedo=(1.0, 0.95, 0.8),
                      emissive=(1.0, 0.95, 0.8), emissive_intensity=800.0)
        _bind(stage, f"{root}/Lamp_{i}", lamp_m)

    for i, (x, y) in enumerate([(12, 10), (-12, 10), (10, -12), (-10, -38)]):
        hp = f"{root}/Hydrant_{i}"
        _xform(stage, hp, translate=(x, y, 0))
        _cylinder(stage, f"{hp}/Base", radius=0.15, height=0.6, translate=(0, 0, 0.3))
        _bind(stage, f"{hp}/Base", hydrant_m)
        _sphere(stage, f"{hp}/Cap", radius=0.18, translate=(0, 0, 0.65))
        _bind(stage, f"{hp}/Cap", hydrant_m)

    for i in range(4):
        _cube(stage, f"{root}/Barrier_{i}", size=1.0,
              translate=(-22 + i * 5, -50, 0.4), scale=(1.5, 0.6, 0.8))
        _bind(stage, f"{root}/Barrier_{i}", barrier_m)

    _cube(stage, f"{root}/Dumpster", size=1.0,
          translate=(-14, -42, 0.6), scale=(2.5, 1.5, 1.2))
    _bind(stage, f"{root}/Dumpster", dumpster_m)
    _label_recursive(stage, root, "terrain")

# ---------------------------------------------------------------------------
# Pedestrians — character USDs with geometry fallback
# ---------------------------------------------------------------------------

def _spawn_geometry_person(stage, path, x, y, z,
                           skin_mat, shirt_mat, pants_mat):
    """Fallback person from basic geometry (~1.75 m tall)."""
    _xform(stage, path, translate=(x, y, z))
    # Torso
    _cylinder(stage, f"{path}/Torso", radius=0.22, height=0.50,
              translate=(0, 0, 1.15))
    _bind(stage, f"{path}/Torso", shirt_mat)
    # Head
    _sphere(stage, f"{path}/Head", radius=0.12, translate=(0, 0, 1.63))
    _bind(stage, f"{path}/Head", skin_mat)
    # Neck
    _cylinder(stage, f"{path}/Neck", radius=0.06, height=0.12,
              translate=(0, 0, 1.47))
    _bind(stage, f"{path}/Neck", skin_mat)
    # Hips
    _cylinder(stage, f"{path}/Hips", radius=0.20, height=0.15,
              translate=(0, 0, 0.85))
    _bind(stage, f"{path}/Hips", pants_mat)
    # Legs
    for li, ly in enumerate([-0.10, 0.10]):
        _cylinder(stage, f"{path}/Leg{li}", radius=0.08, height=0.55,
                  translate=(0, ly, 0.33))
        _bind(stage, f"{path}/Leg{li}", pants_mat)
    # Arms
    for ai, ay in enumerate([-0.30, 0.30]):
        _cylinder(stage, f"{path}/Arm{ai}", radius=0.06, height=0.55,
                  translate=(0, ay, 1.05))
        _bind(stage, f"{path}/Arm{ai}", shirt_mat)
    # Feet
    for fi, fy in enumerate([-0.10, 0.10]):
        _cube(stage, f"{path}/Foot{fi}", size=1.0,
              translate=(0.05, fy, 0.05), scale=(0.18, 0.08, 0.06))
        _bind(stage, f"{path}/Foot{fi}", pants_mat)
    _label_recursive(stage, path, "person")


def _build_pedestrians(stage, buildings_info, count=12):
    """Spawn character USDs inside buildings. Falls back to geometry."""
    root = "/World/Pedestrians"
    _xform(stage, root)

    skin = _mat(stage, "SkinMat", albedo=(0.72, 0.55, 0.45), roughness=0.85)
    shirts = [
        _mat(stage, "ShirtBlue", albedo=(0.10, 0.15, 0.40), roughness=0.75),
        _mat(stage, "ShirtRed", albedo=(0.55, 0.12, 0.10), roughness=0.75),
        _mat(stage, "ShirtGreen", albedo=(0.12, 0.35, 0.12), roughness=0.75),
        _mat(stage, "ShirtWhite", albedo=(0.80, 0.80, 0.78), roughness=0.7),
    ]
    pants = _mat(stage, "PantsMat", albedo=(0.12, 0.12, 0.15), roughness=0.8)

    ped_idx = 0
    bldg_list = list(buildings_info.values()) if isinstance(buildings_info, dict) \
                else buildings_info
    per_bldg = max(1, count // max(len(bldg_list), 1))

    for bi, binfo in enumerate(bldg_list):
        for j in range(per_bldg):
            if ped_idx >= count:
                break

            ped_path = f"{root}/Ped{ped_idx:02d}"
            shirt_mat = shirts[ped_idx % len(shirts)]

            # Position inside building
            bx, by = binfo["x"], binfo["y"]
            hsx, hsy = binfo["hsx"], binfo["hsy"]
            ox = bx + random.uniform(-hsx + 1.5, hsx - 1.5)
            oy = by + random.uniform(-hsy + 1.5, hsy - 1.5)
            oz = 0.0

            char_usd = _CHARACTER_USDS[ped_idx % len(_CHARACTER_USDS)]

            if os.path.isfile(char_usd):
                print(f"[ResQ-AI] Character: {os.path.basename(char_usd)} → {ped_path}")
                xf = _xform(stage, ped_path, translate=(ox, oy, oz))
                prim = stage.GetPrimAtPath(ped_path)
                prim.GetReferences().AddReference(char_usd)
                _apply_semantic_label(prim, "person")
            else:
                print(f"[ResQ-AI] Character: geometry fallback → {ped_path}")
                _spawn_geometry_person(stage, ped_path, ox, oy, oz,
                                       skin, shirt_mat, pants)

            binfo.setdefault("ped_paths", []).append(ped_path)
            ped_idx += 1

    print(f"[ResQ-AI] Placed {ped_idx} pedestrians across "
          f"{len(bldg_list)} buildings.")


# ---------------------------------------------------------------------------
# Flow fire preset discovery
# ---------------------------------------------------------------------------

def _find_fire_preset() -> Optional[str]:
    """Find a fire preset USD in the Particles pack."""
    if not _FLOW_SAMPLES_DIR.exists():
        return None
    for candidate in ["Fire.usda", "WispyFire.usda"]:
        path = _FLOW_SAMPLES_DIR / candidate
        if path.exists():
            return str(path)
    for fname in os.listdir(str(_FLOW_SAMPLES_DIR)):
        if "fire" in fname.lower() and fname.endswith((".usd", ".usdc", ".usda")):
            return str(_FLOW_SAMPLES_DIR / fname)
    return None


def _find_smoke_preset() -> Optional[str]:
    """Find a smoke preset USD in the Particles pack."""
    if not _FLOW_SAMPLES_DIR.exists():
        return None
    for candidate in ["DenseSmoke.usda", "DarkSmoke.usda"]:
        path = _FLOW_SAMPLES_DIR / candidate
        if path.exists():
            return str(path)
    for fname in os.listdir(str(_FLOW_SAMPLES_DIR)):
        if "smoke" in fname.lower() and fname.endswith((".usd", ".usdc", ".usda")):
            return str(_FLOW_SAMPLES_DIR / fname)
    return None


# ---------------------------------------------------------------------------
# Fire system — Flow presets + vegetation/building emitters
# ---------------------------------------------------------------------------

def _spawn_flow_fire(stage, path, wx, wy, wz,
                     fire_preset, smoke_preset,
                     emitter_radius=1.5, fuel=1.8, visible=True):
    """Create a fire emitter at the given position.

    Strategy: reference Flow preset USD → manual Flow hierarchy → emissive geometry.
    Always adds a SphereLight for glow.
    """
    _xform(stage, path, translate=(wx, wy, wz))

    # SphereLight for warm glow
    glow = UsdLux.SphereLight.Define(stage, f"{path}/Glow")
    glow.CreateRadiusAttr(emitter_radius * 1.5)
    glow.CreateIntensityAttr(12000 if visible else 0)
    glow.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.05))
    _set_xform(glow, translate=(0, 0, 3))

    # Strategy 1: Reference fire preset USD
    if fire_preset and os.path.isfile(fire_preset):
        fire_ref = f"{path}/FirePreset"
        _xform(stage, fire_ref)
        stage.GetPrimAtPath(fire_ref).GetReferences().AddReference(fire_preset)
        print(f"[ResQ-AI]   Flow preset: {os.path.basename(fire_preset)}")
    else:
        # Strategy 2: Emissive geometry fallback
        flame_outer = _mat(stage, "FlameOuter",
                           albedo=(1.0, 0.3, 0.0),
                           emissive=(1.0, 0.4, 0.0),
                           emissive_intensity=5000, roughness=1.0)
        flame_core = _mat(stage, "FlameCore",
                          albedo=(1.0, 0.9, 0.3),
                          emissive=(1.0, 0.9, 0.3),
                          emissive_intensity=8000, roughness=1.0)
        _cone(stage, f"{path}/Flame1", radius=emitter_radius,
              height=emitter_radius * 2.5, translate=(0, 0, emitter_radius * 1.2))
        _bind(stage, f"{path}/Flame1", flame_outer)
        _cone(stage, f"{path}/Core", radius=emitter_radius * 0.5,
              height=emitter_radius * 2.0, translate=(0, 0, emitter_radius))
        _bind(stage, f"{path}/Core", flame_core)

    # Smoke
    if smoke_preset and os.path.isfile(smoke_preset):
        smoke_ref = f"{path}/SmokePreset"
        _xform(stage, smoke_ref, translate=(0, 0, 2))
        stage.GetPrimAtPath(smoke_ref).GetReferences().AddReference(smoke_preset)

    if not visible:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()

    _label_recursive(stage, path, "fire")


def _build_fire_system(stage, buildings_info, tree_positions,
                       wind_direction) -> List[Dict]:
    """Create vegetation fires + building fire emitters.

    Returns list of vegetation fire dicts for FireSpreadManager.
    """
    root = "/World/Fires"
    _xform(stage, root)

    fire_preset = _find_fire_preset()
    smoke_preset = _find_smoke_preset()
    if fire_preset:
        print(f"[ResQ-AI] Fire preset: {os.path.basename(fire_preset)}")
    if smoke_preset:
        print(f"[ResQ-AI] Smoke preset: {os.path.basename(smoke_preset)}")

    # Normalise wind
    wn = Gf.Vec3f(*wind_direction)
    wlen = wn.GetLength()
    if wlen > 1e-6:
        wn /= wlen

    # ── Vegetation fires (windward-biased) ────────────────────────────
    veg_root = f"{root}/Vegetation"
    _xform(stage, veg_root)

    scored_trees = []
    for tp in tree_positions:
        tlen = tp.GetLength()
        if tlen < 1e-6:
            continue
        alignment = Gf.Dot(tp.GetNormalized(), wn)
        scored_trees.append((alignment, tp))
    scored_trees.sort(key=lambda x: -x[0])

    num_veg = min(12, len(scored_trees))
    veg_fires = []
    for i in range(num_veg):
        _, tp = scored_trees[i]
        vf_path = f"{veg_root}/VF{i:02d}"
        active = i < 3  # first 3 start burning
        _spawn_flow_fire(stage, vf_path,
                         float(tp[0]), float(tp[1]), 0.5,
                         fire_preset, smoke_preset,
                         emitter_radius=1.2, visible=active)
        veg_fires.append({
            "path": vf_path, "active": active,
            "pos": Gf.Vec3f(float(tp[0]), float(tp[1]), 0.5),
        })

    print(f"[ResQ-AI] {num_veg} vegetation fires "
          f"({sum(1 for v in veg_fires if v['active'])} initially active)")

    # ── Building fires (all initially invisible) ─────────────────────
    bldg_root = f"{root}/Buildings"
    _xform(stage, bldg_root)

    bldg_list = list(buildings_info.values()) if isinstance(buildings_info, dict) \
                else buildings_info
    for b in bldg_list:
        bf_path = f"{bldg_root}/BF_{b['name']}"
        fz = b["h"] + 2.0
        _spawn_flow_fire(stage, bf_path,
                         b["x"], b["y"], fz,
                         fire_preset, smoke_preset,
                         emitter_radius=2.0, visible=False)

    print(f"[ResQ-AI] {len(bldg_list)} building fire emitters (all invisible)")
    return veg_fires


# ---------------------------------------------------------------------------
# Fire Spread Manager — wind-driven spread + flee movement
# ---------------------------------------------------------------------------

class FireSpreadManager:
    """Handles fire spreading between vegetation and buildings,
    and triggers victim fleeing when buildings catch fire."""

    def __init__(self, stage, buildings_info, *,
                 spread_radius=35.0, wind_direction=(1, 0, 0),
                 wind_speed=5.0, vegetation_fires=None,
                 tree_positions=None):
        self._stage = stage
        self._buildings = list(buildings_info.values()) \
            if isinstance(buildings_info, dict) else list(buildings_info)
        self._spread_radius = spread_radius
        self._step = 0
        self._check_interval = 60  # physics steps between spread checks

        # Wind
        wn = Gf.Vec3f(*wind_direction)
        wlen = wn.GetLength()
        self._wind = wn / wlen if wlen > 1e-6 else Gf.Vec3f(1, 0, 0)
        self._wind_speed = wind_speed

        # Vegetation fires
        self._veg_fires = vegetation_fires or []
        self._tree_positions = tree_positions or []

        # Tracking
        self._burning_buildings = set()
        self._flee_lerps = []  # (prim_path, start, end, step, total)

    def _wind_factor(self, from_pos, to_pos):
        """0..1 how downwind `to` is from `from`."""
        d = to_pos - from_pos
        dlen = d.GetLength()
        if dlen < 1e-6:
            return 0.5
        return 0.5 + 0.5 * Gf.Dot(d / dlen, self._wind)

    def ignite_vegetation_fires(self):
        """Start the initially-active vegetation fires."""
        count = 0
        for vf in self._veg_fires:
            if vf["active"]:
                prim = self._stage.GetPrimAtPath(vf["path"])
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeVisible()
                count += 1
        print(f"[ResQ-AI] 🔥 Ignited {count} initial vegetation fires.")

    def _ignite_building(self, b):
        """Set a building on fire and trigger victim flee."""
        name = b["name"]
        if name in self._burning_buildings:
            return
        self._burning_buildings.add(name)
        b["is_burning"] = True

        # Make building fire visible
        bf_path = f"/World/Fires/Buildings/BF_{name}"
        prim = self._stage.GetPrimAtPath(bf_path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()

        print(f"[ResQ-AI] 🔥 Building {name} caught fire!")

        # Trigger flee for victims in this building
        self._trigger_flee(b)

    def _trigger_flee(self, b):
        """Start lerp-based flee movement for victims in this building."""
        ped_paths = b.get("ped_paths", [])
        if not ped_paths:
            return

        bx, by = b["x"], b["y"]
        for pp in ped_paths:
            prim = self._stage.GetPrimAtPath(pp)
            if not prim.IsValid():
                continue

            # Flee direction: away from building, slightly randomised
            angle = math.atan2(by, bx) + math.pi + random.uniform(-0.5, 0.5)
            flee_dist = 20 + random.uniform(0, 15)
            target_x = bx + flee_dist * math.cos(angle)
            target_y = by + flee_dist * math.sin(angle)

            start = Gf.Vec3d(bx + random.uniform(-2, 2),
                              by + random.uniform(-2, 2), 0)
            end = Gf.Vec3d(target_x, target_y, 0)
            self._flee_lerps.append((pp, start, end, 0, 120))

        print(f"[ResQ-AI] 🏃 {len(ped_paths)} victim(s) fleeing from {b['name']}")

    def _advance_flee_lerps(self):
        """Advance all active flee lerps by one step."""
        still_active = []
        for pp, start, end, step, total in self._flee_lerps:
            prim = self._stage.GetPrimAtPath(pp)
            if not prim.IsValid():
                continue
            t = min(1.0, (step + 1) / total)
            # Ease-out: t' = 1 - (1-t)^2
            t_ease = 1.0 - (1.0 - t) ** 2
            pos = start + (end - start) * t_ease

            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            if ops:
                ops[0].Set(pos)
            else:
                xf.AddTranslateOp().Set(pos)

            if step + 1 < total:
                still_active.append((pp, start, end, step + 1, total))
        self._flee_lerps = still_active

    def _check_spread(self):
        """Check fire spread: veg→veg, veg→building, building→building."""
        time_factor = min(1.0, self._step / 1800.0)

        # ── Veg → Veg spread ─────────────────────────────────────────
        newly_active = []
        for vf in self._veg_fires:
            if vf["active"]:
                continue
            for src in self._veg_fires:
                if not src["active"]:
                    continue
                d = (vf["pos"] - src["pos"]).GetLength()
                if d < 15.0:
                    wf = self._wind_factor(src["pos"], vf["pos"])
                    prob = time_factor * (1.0 - d / 15.0) * (0.3 + 1.7 * wf) * 0.03
                    if random.random() < prob:
                        newly_active.append(vf)
                        break
        for vf in newly_active:
            vf["active"] = True
            prim = self._stage.GetPrimAtPath(vf["path"])
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeVisible()

        # ── Veg → Building spread ────────────────────────────────────
        for b in self._buildings:
            if b.get("is_burning"):
                continue
            bp = Gf.Vec3f(b["x"], b["y"], 0)
            for vf in self._veg_fires:
                if not vf["active"]:
                    continue
                d = (bp - vf["pos"]).GetLength()
                if d < self._spread_radius:
                    wf = self._wind_factor(vf["pos"], bp)
                    prob = time_factor * (1.0 - d / self._spread_radius) * \
                           (0.3 + 1.7 * wf) * 0.02
                    if random.random() < prob:
                        self._ignite_building(b)
                        break

        # ── Building → Building spread ───────────────────────────────
        burning = [b for b in self._buildings if b.get("is_burning")]
        for b in self._buildings:
            if b.get("is_burning"):
                continue
            bp = Gf.Vec3f(b["x"], b["y"], 0)
            for src in burning:
                sp = Gf.Vec3f(src["x"], src["y"], 0)
                d = (bp - sp).GetLength()
                if d < self._spread_radius:
                    wf = self._wind_factor(sp, bp)
                    prob = time_factor * (1.0 - d / self._spread_radius) * \
                           (0.3 + 1.7 * wf) * 0.01
                    if random.random() < prob:
                        self._ignite_building(b)
                        break

    def on_physics_step(self, dt):
        """Called every physics step by World callback."""
        self._step += 1
        self._advance_flee_lerps()
        if self._step % self._check_interval == 0:
            self._check_spread()

    def print_status(self):
        """Print current simulation status."""
        n_burn = len(self._burning_buildings)
        n_bldg = len(self._buildings)
        n_veg = sum(1 for v in self._veg_fires if v["active"])
        n_veg_total = len(self._veg_fires)
        print(f"[ResQ-AI] step={self._step}  "
              f"buildings={n_burn}/{n_bldg}  "
              f"veg_fires={n_veg}/{n_veg_total}  "
              f"fleeing={len(self._flee_lerps)}")


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _build_physics(stage):
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    col_plane = UsdGeom.Xform.Define(stage, "/World/GroundCollider")
    _set_xform(col_plane, translate=(0, 0, 0))
    UsdPhysics.CollisionAPI.Apply(col_plane.GetPrim())
    _cube(stage, "/World/GroundCollider/Plane", size=1.0,
          translate=(0, 0, -0.5), scale=(500, 500, 1.0))
    UsdPhysics.CollisionAPI.Apply(
        stage.GetPrimAtPath("/World/GroundCollider/Plane"))

# ===========================================================================
# Main
# ===========================================================================

def generate_scene():
    """Build the full urban disaster scene.

    Returns (stage, buildings_info, tree_positions).
    """
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _xform(stage, "/World")

    print("[ResQ-AI] Building 3D terrain …")
    _build_terrain(stage)

    print("[ResQ-AI] Referencing CityEngine city …")
    city_ok = _build_city_reference(stage)

    if city_ok:
        print("[ResQ-AI] Discovering CityEngine buildings …")
        n = _discover_city_buildings(stage)
        if n == 0:
            city_ok = False

    if not city_ok:
        print("[ResQ-AI] Building fallback structures with OmniPBR …")
        _build_buildings_fallback(stage)

    # Collect building info as dicts for pedestrians/fire
    # Use the fallback spec list as canonical building data
    specs = [
        ("OfficeA",  30,  30, 16, 12, 32), ("TowerB",   58,  30, 10, 14, 55),
        ("AptA",     30,  58, 14, 14, 24), ("AptB",     58,  58, 12, 10, 18),
        ("OfficeC", -30,  30, 18, 14, 40), ("ShopA",   -58,  30, 10,  8, 10),
        ("AptC",    -30,  58, 14, 12, 26), ("OfficeD", -58,  58, 12, 12, 38),
        ("OfficeE",  30, -30, 14, 16, 30), ("ShopB",    58, -30,  8, 10, 10),
        ("AptD",     30, -58, 14, 14, 20), ("Warehouse",58, -58, 18, 18, 12),
        ("OfficeF", -30, -30, 16, 12, 34), ("AptE",    -58, -30, 12, 10, 16),
        ("ShopC",   -30, -58, 10, 10,  8), ("AptF",    -58, -58, 14, 14, 22),
    ]
    buildings_info = [
        {"name": name, "x": x, "y": y,
         "hsx": sx // 2, "hsy": sy // 2, "h": h,
         "is_burning": False, "ped_paths": []}
        for name, x, y, sx, sy, h in specs
    ]

    print("[ResQ-AI] Building roads …")
    _build_roads(stage, city_ok)

    print("[ResQ-AI] Placing vehicles …")
    _build_vehicles(stage)

    print("[ResQ-AI] Planting vegetation (PointInstancer) …")
    tree_positions = _build_vegetation(stage)

    print("[ResQ-AI] Adding street props …")
    _build_street_props(stage)

    print("[ResQ-AI] Creating disaster zone 1 — collapsed building …")
    _build_disaster_collapsed(stage)

    print("[ResQ-AI] Creating disaster zone 2 — vehicle fire …")
    _build_disaster_fire(stage)

    print("[ResQ-AI] Setting up HDRI lighting …")
    _build_lighting(stage)

    print("[ResQ-AI] Configuring physics …")
    _build_physics(stage)

    print("[ResQ-AI] Scene generation complete.")
    return stage, buildings_info, tree_positions


def main() -> None:
    print("=" * 72)
    print("  ResQ-AI — Urban Disaster Scene Generator + Simulation")
    print("=" * 72)
    wind_dir = tuple(_args.wind_direction)
    print(f"  Wind: ({wind_dir[0]:.1f}, {wind_dir[1]:.1f}, {wind_dir[2]:.1f})"
          f" at {_args.wind_speed:.1f} m/s")
    print(f"  Pedestrians: {_args.num_pedestrians}")
    print("-" * 72)

    # ── 1. Generate base scene ─────────────────────────────────────────
    stage, buildings_info, tree_positions = generate_scene()

    # ── 2. Export the static base scene ───────────────────────────────
    output_path = "resqai_urban_disaster.usda"
    stage.GetRootLayer().Export(output_path)
    print(f"[ResQ-AI] Stage exported to {output_path}")

    # ── 3. Spawn pedestrians ──────────────────────────────────────────
    print("\n[ResQ-AI] ── Spawning pedestrians ──")
    random.seed(2024)
    _build_pedestrians(stage, buildings_info, count=_args.num_pedestrians)

    # ── 4. Build fire system ──────────────────────────────────────────
    print("\n[ResQ-AI] ── Building fire system ──")
    veg_fires = _build_fire_system(stage, buildings_info,
                                    tree_positions, wind_dir)

    # ── 5. Fire spread manager ────────────────────────────────────────
    print("\n[ResQ-AI] ── Fire Spread Logic ──")
    fire_mgr = FireSpreadManager(
        stage, buildings_info,
        spread_radius=_args.fire_spread_radius,
        wind_direction=wind_dir,
        wind_speed=_args.wind_speed,
        vegetation_fires=veg_fires,
        tree_positions=tree_positions)

    fire_mgr.ignite_vegetation_fires()

    # ── 6. Simulation loop ────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)
    world.add_physics_callback("fire_spread", fire_mgr.on_physics_step)
    world.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print("\n[ResQ-AI] Simulation running. Press Ctrl-C or close window.\n")

    step = 0
    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            if step % 300 == 0:
                fire_mgr.print_status()

            if _args.max_steps and step >= _args.max_steps:
                print(f"[ResQ-AI] Reached --max-steps={_args.max_steps}.")
                break

    except KeyboardInterrupt:
        print("\n[ResQ-AI] Interrupted.")

    # ── Cleanup & report ──────────────────────────────────────────────
    timeline.stop()
    simulation_app.close()

    print("\n" + "=" * 72)
    print("  ResQ-AI — Final Fire Report")
    print("=" * 72)
    for b in buildings_info:
        status = "🔥 BURNING" if b.get("is_burning") else "✅ Safe"
        victims = len(b.get("ped_paths", []))
        print(f"  {b['name']:12s}  {status}  "
              f"({victims} victim{'s' if victims != 1 else ''})")

    total_burning = sum(1 for b in buildings_info if b.get("is_burning"))
    active_veg = sum(1 for v in veg_fires if v["active"])
    print(f"\n  Buildings: {total_burning}/{len(buildings_info)} on fire")
    print(f"  Vegetation: {active_veg}/{len(veg_fires)} active")
    print(f"  Wind: ({wind_dir[0]:.1f}, {wind_dir[1]:.1f}, "
          f"{wind_dir[2]:.1f}) at {_args.wind_speed:.1f} m/s")
    print(f"  Simulation: {step} physics steps")
    print("=" * 72)


if __name__ == "__main__":
    main()
