#!/usr/bin/env python3
"""Generate a 3D urban disaster environment for the ResQ-AI simulation pipeline.

Pure OpenUSD scene authoring via ``pxr``.  Every geometric prim keeps its
correct type (Cube, Cylinder, Sphere, Cone) so the renderer can draw it.
Materials use OmniPBR for metallic/roughness rendering when available, falling
back to UsdPreviewSurface.  An HDRI dome light provides realistic outdoor
illumination and a distant light acts as a sun.

Semantic class mapping
----------------------
fire meshes        -> 'fire'
pedestrian assets  -> 'person'
car / vehicle      -> 'vehicle'
building / rubble  -> 'building'
trees / parks      -> 'vegetation'
roads / ground     -> 'terrain'

Usage (from the Isaac Sim python environment)::

    /isaac-sim/python.sh sim_bridge/generate_urban_scene.py          # GUI mode
    /isaac-sim/python.sh sim_bridge/generate_urban_scene.py --headless
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import TYPE_CHECKING

from isaacsim import SimulationApp

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI urban disaster scene generator")
    parser.add_argument("--headless", action="store_true", help="Run without a viewport window")
    return parser.parse_args()

_args = _parse_args()

simulation_app = SimulationApp({"headless": _args.headless})

import omni.usd
from omni.isaac.core import World
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

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
        return
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
# Transform helper — sets ops on an *existing* Xformable without redefining
# ---------------------------------------------------------------------------

def _set_xform(xformable: UsdGeom.Xformable,
               translate: tuple = (0, 0, 0),
               rotate_deg: tuple = (0, 0, 0),
               scale: tuple = (1, 1, 1)):
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_deg))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))

# ---------------------------------------------------------------------------
# Prim creation — geometry + transform in a single call
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
# OmniPBR material (metallic/roughness) with UsdPreviewSurface fallback
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
# Scene sections
# ---------------------------------------------------------------------------

def _build_terrain(stage):
    _xform(stage, "/World/Terrain")
    _cube(stage, "/World/Terrain/Ground", size=1.0,
          translate=(0, 0, -0.05), scale=(250, 250, 0.1))
    m = _mat(stage, "TerrainMat", albedo=(0.22, 0.21, 0.18), roughness=0.95)
    _bind(stage, "/World/Terrain/Ground", m)

    _cube(stage, "/World/Terrain/Sidewalk_N", size=1.0,
          translate=(0, 8, 0.03), scale=(250, 4, 0.06))
    _cube(stage, "/World/Terrain/Sidewalk_S", size=1.0,
          translate=(0, -8, 0.03), scale=(250, 4, 0.06))
    _cube(stage, "/World/Terrain/Sidewalk_E", size=1.0,
          translate=(8, 0, 0.03), scale=(4, 250, 0.06))
    _cube(stage, "/World/Terrain/Sidewalk_W", size=1.0,
          translate=(-8, 0, 0.03), scale=(4, 250, 0.06))
    sw = _mat(stage, "SidewalkMat", albedo=(0.45, 0.44, 0.42), roughness=0.85)
    for s in ["Sidewalk_N", "Sidewalk_S", "Sidewalk_E", "Sidewalk_W"]:
        _bind(stage, f"/World/Terrain/{s}", sw)

    _label_recursive(stage, "/World/Terrain", "terrain")


def _build_roads(stage):
    root = "/World/Roads"
    _xform(stage, root)
    asphalt = _mat(stage, "AsphaltMat", albedo=(0.08, 0.08, 0.08), roughness=0.9)

    _cube(stage, f"{root}/EW", size=1.0, translate=(0, 0, 0.01), scale=(250, 14, 0.06))
    _bind(stage, f"{root}/EW", asphalt)
    _cube(stage, f"{root}/NS", size=1.0, translate=(0, 0, 0.01), scale=(14, 250, 0.06))
    _bind(stage, f"{root}/NS", asphalt)

    marking = _mat(stage, "MarkingMat", albedo=(0.95, 0.85, 0.15), roughness=0.6)
    for i, off in enumerate(range(-110, 111, 12)):
        _cube(stage, f"{root}/MarkEW_{i}", size=1.0,
              translate=(off, 0, 0.05), scale=(5, 0.25, 0.02))
        _bind(stage, f"{root}/MarkEW_{i}", marking)
    for i, off in enumerate(range(-110, 111, 12)):
        _cube(stage, f"{root}/MarkNS_{i}", size=1.0,
              translate=(0, off, 0.05), scale=(0.25, 5, 0.02))
        _bind(stage, f"{root}/MarkNS_{i}", marking)

    _label_recursive(stage, root, "terrain")


def _build_buildings(stage):
    root = "/World/Buildings"
    _xform(stage, root)

    concrete = _mat(stage, "ConcreteMat", albedo=(0.52, 0.50, 0.47), roughness=0.85)
    brick    = _mat(stage, "BrickMat",    albedo=(0.55, 0.25, 0.18), roughness=0.8)
    glass    = _mat(stage, "GlassMat",    albedo=(0.35, 0.55, 0.70), roughness=0.1, metallic=0.6, opacity=0.45)
    dark_win = _mat(stage, "DarkWinMat",  albedo=(0.12, 0.18, 0.25), roughness=0.15, metallic=0.5, opacity=0.6)
    roof     = _mat(stage, "RoofMat",     albedo=(0.3, 0.28, 0.26), roughness=0.75)

    specs = [
        ("OfficeA",    30,  30, 16, 12, 32, concrete),
        ("TowerB",     58,  30, 10, 14, 55, concrete),
        ("AptA",       30,  58, 14, 14, 24, brick),
        ("AptB",       58,  58, 12, 10, 18, brick),
        ("OfficeC",   -30,  30, 18, 14, 40, concrete),
        ("ShopA",     -58,  30, 10,  8, 10, brick),
        ("AptC",      -30,  58, 14, 12, 26, brick),
        ("OfficeD",   -58,  58, 12, 12, 38, concrete),
        ("OfficeE",    30, -30, 14, 16, 30, concrete),
        ("ShopB",      58, -30,  8, 10, 10, brick),
        ("AptD",       30, -58, 14, 14, 20, brick),
        ("Warehouse",  58, -58, 18, 18, 12, concrete),
        ("OfficeF",   -30, -30, 16, 12, 34, concrete),
        ("AptE",      -58, -30, 12, 10, 16, brick),
        ("ShopC",     -30, -58, 10, 10,  8, brick),
        ("AptF",      -58, -58, 14, 14, 22, brick),
    ]

    for name, x, y, sx, sy, h, base_mat in specs:
        bp = f"{root}/{name}"
        _xform(stage, bp, translate=(x, y, 0))
        _cube(stage, f"{bp}/Body", size=1.0,
              translate=(0, 0, h / 2.0), scale=(sx, sy, h))
        _bind(stage, f"{bp}/Body", base_mat)

        _cube(stage, f"{bp}/Roof", size=1.0,
              translate=(0, 0, h + 0.15), scale=(sx + 0.6, sy + 0.6, 0.3))
        _bind(stage, f"{bp}/Roof", roof)

        if h > 14:
            # Window bands on two faces
            for fi, (dx, dy, wsx, wsy) in enumerate([
                (sx / 2 + 0.06, 0, 0.12, sy * 0.85),
                (0, sy / 2 + 0.06, sx * 0.85, 0.12),
            ]):
                floors = int(h / 4)
                for fl in range(floors):
                    fz = 3 + fl * (h - 4) / max(floors - 1, 1)
                    win_mat = glass if fl % 3 != 0 else dark_win
                    _cube(stage, f"{bp}/Win_{fi}_{fl}", size=1.0,
                          translate=(dx, dy, fz), scale=(wsx, wsy, 1.8))
                    _bind(stage, f"{bp}/Win_{fi}_{fl}", win_mat)

    _label_recursive(stage, root, "building")


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
    tire  = _mat(stage, "TireMat",  albedo=(0.03, 0.03, 0.03), roughness=0.95)
    chrome = _mat(stage, "ChromeMat", albedo=(0.7, 0.7, 0.72), roughness=0.08, metallic=1.0)
    wind  = _mat(stage, "WindshieldMat", albedo=(0.25, 0.4, 0.55), roughness=0.05, metallic=0.4, opacity=0.4)

    cars = [
        ("Car01",  22,   7.5, 0,   "Red"),    ("Car02",  38,   7.5, 0,   "Blue"),
        ("Car03",  54,  -7.5, 180, "White"),   ("Car04", -22,   7.5, 0,   "Yellow"),
        ("Car05", -42,  -7.5, 180, "Black"),   ("Car06",  7.5,  28,  90,  "Silver"),
        ("Car07", -7.5,  44,  270, "Blue"),    ("Car08",  7.5, -28,  90,  "White"),
        ("Car09", -7.5, -48,  270, "Yellow"),  ("Car10", -68,   7.5, 0,   "Black"),
        ("Car11",  74,  -7.5, 180, "Red"),     ("Car12",  7.5,  74,  90,  "Silver"),
    ]

    for vid, x, y, rz, color in cars:
        vp = f"{root}/{vid}"
        _xform(stage, vp, translate=(x, y, 0), rotate_deg=(0, 0, rz))

        _cube(stage, f"{vp}/Body", size=1.0,
              translate=(0, 0, 0.55), scale=(4.4, 1.85, 0.75))
        _bind(stage, f"{vp}/Body", paints[color])

        _cube(stage, f"{vp}/Cabin", size=1.0,
              translate=(0.2, 0, 1.15), scale=(2.4, 1.7, 0.65))
        _bind(stage, f"{vp}/Cabin", wind)

        _cube(stage, f"{vp}/Hood", size=1.0,
              translate=(-1.5, 0, 0.85), scale=(1.2, 1.75, 0.12))
        _bind(stage, f"{vp}/Hood", paints[color])

        _cube(stage, f"{vp}/Bumper_F", size=1.0,
              translate=(-2.25, 0, 0.35), scale=(0.2, 1.6, 0.3))
        _bind(stage, f"{vp}/Bumper_F", chrome)
        _cube(stage, f"{vp}/Bumper_R", size=1.0,
              translate=(2.25, 0, 0.35), scale=(0.2, 1.6, 0.3))
        _bind(stage, f"{vp}/Bumper_R", chrome)

        for wi, (wx, wy) in enumerate([(-1.3, 1.0), (-1.3, -1.0), (1.3, 1.0), (1.3, -1.0)]):
            _cylinder(stage, f"{vp}/Wheel_{wi}", radius=0.34, height=0.22, axis="Y",
                      translate=(wx, wy, 0.34))
            _bind(stage, f"{vp}/Wheel_{wi}", tire)
            _cylinder(stage, f"{vp}/Hub_{wi}", radius=0.18, height=0.24, axis="Y",
                      translate=(wx, wy, 0.34))
            _bind(stage, f"{vp}/Hub_{wi}", chrome)

    _label_recursive(stage, root, "vehicle")


def _build_pedestrians(stage):
    root = "/World/Pedestrians"
    _xform(stage, root)
    skin = _mat(stage, "SkinMat", albedo=(0.72, 0.55, 0.45), roughness=0.85)
    hair = _mat(stage, "HairMat", albedo=(0.12, 0.08, 0.06), roughness=0.9)
    shirts = [
        _mat(stage, "ShirtNavy",   albedo=(0.1, 0.15, 0.35), roughness=0.75),
        _mat(stage, "ShirtOlive",  albedo=(0.25, 0.3, 0.15), roughness=0.75),
        _mat(stage, "ShirtMaroon", albedo=(0.4, 0.1, 0.1),   roughness=0.75),
        _mat(stage, "ShirtTeal",   albedo=(0.1, 0.35, 0.35), roughness=0.75),
    ]
    pants = _mat(stage, "PantsMat", albedo=(0.15, 0.15, 0.18), roughness=0.8)
    shoe  = _mat(stage, "ShoeMat",  albedo=(0.08, 0.06, 0.05), roughness=0.9)

    peds = [
        ("Ped01", 15, 3),  ("Ped02", 24, -3), ("Ped03", -18, 4),
        ("Ped04", -38,-3), ("Ped05", 3, 18),  ("Ped06", -4, 38),
        ("Ped07", 3, -24), ("Ped08", -4, -42),("Ped09", 42, 42),
        ("Ped10", -44,40), ("Ped11", 40, -44),("Ped12", -42,-40),
    ]

    for idx, (name, x, y) in enumerate(peds):
        pp = f"{root}/{name}"
        _xform(stage, pp, translate=(x, y, 0))

        _cylinder(stage, f"{pp}/Torso", radius=0.2, height=0.65,
                  translate=(0, 0, 1.08))
        _bind(stage, f"{pp}/Torso", shirts[idx % len(shirts)])

        _sphere(stage, f"{pp}/Head", radius=0.15, translate=(0, 0, 1.6))
        _bind(stage, f"{pp}/Head", skin)

        _sphere(stage, f"{pp}/Hair", radius=0.16, translate=(0, 0, 1.68))
        _bind(stage, f"{pp}/Hair", hair)

        _cylinder(stage, f"{pp}/Hips", radius=0.18, height=0.3,
                  translate=(0, 0, 0.7))
        _bind(stage, f"{pp}/Hips", pants)

        for li, ly in enumerate([-0.1, 0.1]):
            _cylinder(stage, f"{pp}/Leg_{li}", radius=0.08, height=0.55,
                      translate=(0, ly, 0.28))
            _bind(stage, f"{pp}/Leg_{li}", pants)

            _cube(stage, f"{pp}/Shoe_{li}", size=1.0,
                  translate=(0.06, ly, 0.04), scale=(0.22, 0.1, 0.08))
            _bind(stage, f"{pp}/Shoe_{li}", shoe)

        for ai, ay in enumerate([-0.28, 0.28]):
            _cylinder(stage, f"{pp}/Arm_{ai}", radius=0.06, height=0.55,
                      translate=(0, ay, 1.0))
            _bind(stage, f"{pp}/Arm_{ai}", shirts[idx % len(shirts)])
            _sphere(stage, f"{pp}/Hand_{ai}", radius=0.06,
                    translate=(0, ay, 0.7))
            _bind(stage, f"{pp}/Hand_{ai}", skin)

    _label_recursive(stage, root, "person")


def _build_vegetation(stage):
    root = "/World/Vegetation"
    _xform(stage, root)
    bark   = _mat(stage, "BarkMat",   albedo=(0.28, 0.16, 0.08), roughness=0.92)
    canopy = _mat(stage, "CanopyMat", albedo=(0.12, 0.38, 0.1),  roughness=0.85)
    grass  = _mat(stage, "GrassMat",  albedo=(0.18, 0.42, 0.12), roughness=0.9)

    spots = [
        (18, 14),  (38, 14),  (58, 14),  (-18, 14), (-38, 14), (-58, 14),
        (14, 18),  (14, 38),  (14, 58),  (-14, 18), (-14, 38), (-14, 58),
        (18, -14), (38, -14), (58, -14), (-18,-14), (-38,-14), (-58,-14),
        (14, -18), (14, -38), (14, -58), (-14,-18), (-14,-38), (-14,-58),
    ]

    random.seed(7)
    for i, (tx, ty) in enumerate(spots):
        tp = f"{root}/Tree_{i:02d}"
        _xform(stage, tp, translate=(tx, ty, 0))
        trunk_h = random.uniform(2.5, 4.5)
        _cylinder(stage, f"{tp}/Trunk", radius=0.2, height=trunk_h,
                  translate=(0, 0, trunk_h / 2))
        _bind(stage, f"{tp}/Trunk", bark)
        cr = random.uniform(1.6, 2.8)
        _sphere(stage, f"{tp}/Canopy", radius=cr,
                translate=(0, 0, trunk_h + cr * 0.6))
        _bind(stage, f"{tp}/Canopy", canopy)

    parks = [(42, 44), (-42, 44), (42, -44), (-42, -44)]
    for i, (px, py) in enumerate(parks):
        _cube(stage, f"{root}/Park_{i}", size=1.0,
              translate=(px, py, 0.02), scale=(12, 12, 0.04))
        _bind(stage, f"{root}/Park_{i}", grass)

    _label_recursive(stage, root, "vegetation")


# ---------------------------------------------------------------------------
# Disaster zones
# ---------------------------------------------------------------------------

def _build_disaster_collapsed(stage):
    dz = "/World/DisasterZones/CollapsedBuilding"
    _xform(stage, "/World/DisasterZones")
    _xform(stage, dz, translate=(-30, -58, 0))

    rubble = _mat(stage, "RubbleMat", albedo=(0.42, 0.40, 0.36), roughness=0.92)
    rebar  = _mat(stage, "RebarMat",  albedo=(0.32, 0.20, 0.12), roughness=0.7, metallic=0.6)

    walls = [
        (-3, 0, 3.5, 8, 0.5, 7, 0),
        (3,  2, 2.5, 0.5, 6, 5, 0),
        (0, -3, 1.5, 6, 0.5, 3, 15),
    ]
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
    _cube(stage, f"{dz}/BurntCar/Body", size=1.0,
          translate=(0, 0, 0.55), scale=(4.4, 1.85, 0.75))
    _bind(stage, f"{dz}/BurntCar/Body", burnt)
    _cube(stage, f"{dz}/BurntCar/Cabin", size=1.0,
          translate=(0.2, 0, 1.15), scale=(2.2, 1.6, 0.5))
    _bind(stage, f"{dz}/BurntCar/Cabin", burnt)
    _label_recursive(stage, f"{dz}/BurntCar", "vehicle")

    fire_root = f"{dz}/Flames"
    _xform(stage, fire_root, translate=(0, 0, 1.0))
    flame_outer = _mat(stage, "FlameOuter", albedo=(1.0, 0.3, 0.0),
                       emissive=(1.0, 0.4, 0.0), emissive_intensity=5000.0, roughness=1.0)
    flame_core  = _mat(stage, "FlameCore", albedo=(1.0, 0.9, 0.3),
                       emissive=(1.0, 0.9, 0.3), emissive_intensity=8000.0, roughness=1.0)

    flames = [
        ("Core",  "cone", dict(radius=0.5, height=2.8), (0, 0, 1.4),     (1,1,1),        (0,0,0),   True),
        ("Left",  "cone", dict(radius=0.38, height=2.0), (-0.6, 0.3, 1.0), (1,1,1),       (0,0,12),  False),
        ("Right", "cone", dict(radius=0.38, height=2.0), (0.5, -0.2, 1.0), (1,1,1),       (0,0,-10), False),
        ("Back",  "cone", dict(radius=0.3, height=1.6),  (-0.9, 0, 0.8),  (1,1,1),        (0,0,18),  False),
        ("EmberA","sphere", dict(radius=0.22),           (0.3, 0.4, 2.6), (1,1,1),         (0,0,0),   True),
        ("EmberB","sphere", dict(radius=0.18),           (-0.4,-0.3, 2.8),(1,1,1),          (0,0,0),   False),
        ("EmberC","sphere", dict(radius=0.15),           (0.1, 0.1, 3.1), (1,1,1),         (0,0,0),   True),
        ("Glow",  "sphere", dict(radius=0.9),            (0, 0, 0.2),     (1.6,1.6,0.4),   (0,0,0),   True),
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
# Street furniture (crosswalk poles, barriers, fire hydrants, dumpsters)
# ---------------------------------------------------------------------------

def _build_street_props(stage):
    root = "/World/StreetProps"
    _xform(stage, root)
    pole_mat   = _mat(stage, "PoleMat",   albedo=(0.4, 0.4, 0.42), roughness=0.4, metallic=0.9)
    hydrant_m  = _mat(stage, "HydrantMat",albedo=(0.7, 0.15, 0.1), roughness=0.5, metallic=0.7)
    barrier_m  = _mat(stage, "BarrierMat",albedo=(0.9, 0.45, 0.0), roughness=0.7)
    dumpster_m = _mat(stage, "DumpsterMat",albedo=(0.15, 0.3, 0.15),roughness=0.65, metallic=0.5)

    # Street lights
    for i, (x, y) in enumerate([(20, 10), (50, 10), (-20, 10), (-50, 10),
                                 (20, -10), (50, -10), (-20, -10), (-50, -10),
                                 (10, 20), (10, 50), (-10, 20), (-10, 50)]):
        _cylinder(stage, f"{root}/Pole_{i}", radius=0.08, height=6.0,
                  translate=(x, y, 3.0))
        _bind(stage, f"{root}/Pole_{i}", pole_mat)
        _sphere(stage, f"{root}/Lamp_{i}", radius=0.25,
                translate=(x, y, 6.2))
        lamp_m = _mat(stage, f"LampGlow_{i}", albedo=(1.0, 0.95, 0.8),
                      emissive=(1.0, 0.95, 0.8), emissive_intensity=800.0)
        _bind(stage, f"{root}/Lamp_{i}", lamp_m)

    # Fire hydrants
    for i, (x, y) in enumerate([(12, 10), (-12, 10), (10, -12), (-10, -38)]):
        hp = f"{root}/Hydrant_{i}"
        _xform(stage, hp, translate=(x, y, 0))
        _cylinder(stage, f"{hp}/Base", radius=0.15, height=0.6, translate=(0, 0, 0.3))
        _bind(stage, f"{hp}/Base", hydrant_m)
        _sphere(stage, f"{hp}/Cap", radius=0.18, translate=(0, 0, 0.65))
        _bind(stage, f"{hp}/Cap", hydrant_m)

    # Concrete barriers near disaster zone
    for i in range(4):
        _cube(stage, f"{root}/Barrier_{i}", size=1.0,
              translate=(-22 + i * 5, -50, 0.4), scale=(1.5, 0.6, 0.8))
        _bind(stage, f"{root}/Barrier_{i}", barrier_m)

    # Dumpster
    _cube(stage, f"{root}/Dumpster", size=1.0,
          translate=(-14, -42, 0.6), scale=(2.5, 1.5, 1.2))
    _bind(stage, f"{root}/Dumpster", dumpster_m)

    _label_recursive(stage, root, "terrain")


# ---------------------------------------------------------------------------
# Lighting — HDRI dome + distant sun + area fill
# ---------------------------------------------------------------------------

def _build_lighting(stage):
    root = "/World/Lights"
    _xform(stage, root)

    dome = UsdLux.DomeLight.Define(stage, f"{root}/Sky")
    dome.CreateIntensityAttr(1200.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))
    try:
        dome.CreateTextureFileAttr(
            f"{ASSET_ROOT}/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr")
    except Exception:
        pass

    sun_xf = UsdGeom.Xform.Define(stage, f"{root}/Sun")
    sun_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 120))
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-50, 25, 0))
    sun = UsdLux.DistantLight.Define(stage, f"{root}/Sun/Light")
    sun.CreateIntensityAttr(4500.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.94, 0.82))

    fill = UsdLux.RectLight.Define(stage, f"{root}/FillLight")
    fill.CreateIntensityAttr(300.0)
    fill.CreateWidthAttr(80.0)
    fill.CreateHeightAttr(80.0)
    fill.CreateColorAttr(Gf.Vec3f(0.7, 0.8, 1.0))
    _set_xform(fill, translate=(0, 0, 60), rotate_deg=(-90, 0, 0))


# ---------------------------------------------------------------------------
# Physics ground plane (for simulation)
# ---------------------------------------------------------------------------

def _build_physics(stage):
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)

    col_plane = UsdGeom.Xform.Define(stage, "/World/GroundCollider")
    _set_xform(col_plane, translate=(0, 0, 0))
    col_prim = stage.GetPrimAtPath("/World/GroundCollider")
    UsdPhysics.CollisionAPI.Apply(col_prim)
    col_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("none")
    _cube(stage, "/World/GroundCollider/Plane", size=1.0,
          translate=(0, 0, -0.5), scale=(500, 500, 1.0))
    UsdPhysics.CollisionAPI.Apply(
        stage.GetPrimAtPath("/World/GroundCollider/Plane"))


# ===========================================================================
# Main
# ===========================================================================

def generate_scene() -> Usd.Stage:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _xform(stage, "/World")

    print("[ResQ-AI] Building terrain …")
    _build_terrain(stage)

    print("[ResQ-AI] Building roads …")
    _build_roads(stage)

    print("[ResQ-AI] Building structures …")
    _build_buildings(stage)

    print("[ResQ-AI] Placing vehicles …")
    _build_vehicles(stage)

    print("[ResQ-AI] Placing pedestrians …")
    _build_pedestrians(stage)

    print("[ResQ-AI] Planting vegetation …")
    _build_vegetation(stage)

    print("[ResQ-AI] Adding street props …")
    _build_street_props(stage)

    print("[ResQ-AI] Creating disaster zone 1 — collapsed building …")
    _build_disaster_collapsed(stage)

    print("[ResQ-AI] Creating disaster zone 2 — vehicle fire …")
    _build_disaster_fire(stage)

    print("[ResQ-AI] Setting up lighting …")
    _build_lighting(stage)

    print("[ResQ-AI] Configuring physics …")
    _build_physics(stage)

    print("[ResQ-AI] Scene generation complete.")
    return stage


def main() -> None:
    stage = generate_scene()

    output_path = "/tmp/resqai_urban_disaster.usda"
    stage.GetRootLayer().Export(output_path)
    print(f"[ResQ-AI] Stage exported to {output_path}")

    if not _args.headless:
        world = World()
        world.reset()
        print("[ResQ-AI] Running viewer — close the window to exit.")
        while simulation_app.is_running():
            world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
