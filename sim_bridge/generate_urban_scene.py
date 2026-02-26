#!/usr/bin/env python3
"""Generate a 3D urban disaster environment for the ResQ-AI simulation pipeline.

Creates a new USD stage and populates it with a city block containing roads,
buildings, vehicles, pedestrians, trees, and two disaster zones (collapsed
building + vehicle fire).  Every prim receives a semantic class label via
omni.replicator.core so downstream annotators can produce segmentation and
synthetic thermal data.

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
    /isaac-sim/python.sh sim_bridge/generate_urban_scene.py --headless  # headless
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap – must happen before any Omniverse imports
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI urban disaster scene generator")
    parser.add_argument("--headless", action="store_true", help="Run without a viewport window")
    return parser.parse_args()

_args = _parse_args()

simulation_app = SimulationApp({"headless": _args.headless})

# ---------------------------------------------------------------------------
# Omniverse / USD imports (available only after SimulationApp is created)
# ---------------------------------------------------------------------------
import omni.usd
from omni.isaac.core import World
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Semantic labelling helpers
# ---------------------------------------------------------------------------

def _apply_semantic_label(prim: Usd.Prim, label: str) -> None:
    """Attach a semantic class label to *prim* using the best available API.

    Tries the modern ``omni.replicator.core.functional`` path first (Isaac Sim
    >= 5.0 / ``UsdSemantics.LabelsAPI``), then falls back to the legacy
    ``Semantics.SemanticsAPI`` so the script works across Isaac Sim versions.
    """
    # Modern path (Isaac Sim >= 5.0)
    try:
        import omni.replicator.core.functional as rep_functional
        rep_functional.modify.semantics(prim, {"class": [label]}, mode="replace")
        return
    except (ModuleNotFoundError, ImportError, AttributeError):
        pass

    # Legacy path
    try:
        import Semantics
        instance_name = f"class_{label}"
        sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set(instance_name)
        sem.GetSemanticDataAttr().Set(label)
        return
    except Exception:
        pass

    print(f"[WARN] Could not apply semantic label '{label}' to {prim.GetPath()}")


def _apply_semantic_label_recursive(stage: Usd.Stage, root_path: str, label: str) -> None:
    """Apply *label* to the prim at *root_path* and every geometric descendant."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Xformable):
            _apply_semantic_label(prim, label)


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

_material_cache: dict[str, str] = {}


def _get_or_create_material(
    stage: Usd.Stage,
    name: str,
    diffuse: tuple[float, float, float],
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    opacity: float = 1.0,
) -> str:
    """Return the Sdf path of a UsdPreviewSurface material, creating it once."""
    if name in _material_cache:
        return _material_cache[name]

    mat_path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    _material_cache[name] = mat_path
    return mat_path


def _bind_material(stage: Usd.Stage, prim_path: str, mat_path: str) -> None:
    """Bind a material to a prim and all mesh descendants."""
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)


# ---------------------------------------------------------------------------
# Prim creation helpers
# ---------------------------------------------------------------------------

def _xform(stage: Usd.Stage, path: str, translate: tuple, rotate_deg: tuple = (0, 0, 0), scale: tuple = (1, 1, 1)):
    """Define an Xform prim with translate / rotate / scale ops."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    return xf


def _cube(stage: Usd.Stage, path: str, size: float = 1.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    return cube


def _cylinder(stage: Usd.Stage, path: str, radius: float = 0.5, height: float = 1.0):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    return cyl


def _sphere(stage: Usd.Stage, path: str, radius: float = 0.5):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    return sph


def _cone(stage: Usd.Stage, path: str, radius: float = 0.5, height: float = 1.0):
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    return cone


# ---------------------------------------------------------------------------
# Scene section builders
# ---------------------------------------------------------------------------

def _build_ground_plane(stage: Usd.Stage) -> None:
    """Large ground / terrain plane."""
    ground_path = "/World/Terrain/GroundPlane"
    _xform(stage, "/World/Terrain", translate=(0, 0, 0))
    _cube(stage, ground_path, size=1.0)
    _xform(stage, ground_path, translate=(0, 0, -0.05), scale=(200, 200, 0.1))

    mat = _get_or_create_material(stage, "TerrainMat", diffuse=(0.25, 0.25, 0.22))
    _bind_material(stage, ground_path, mat)
    _apply_semantic_label_recursive(stage, "/World/Terrain", "terrain")


def _build_roads(stage: Usd.Stage) -> None:
    """Create a crossroads layout: two perpendicular roads through the block."""
    road_root = "/World/Roads"
    _xform(stage, road_root, translate=(0, 0, 0))
    mat = _get_or_create_material(stage, "RoadMat", diffuse=(0.12, 0.12, 0.12))

    # East-west road
    ew = f"{road_root}/EastWest"
    _cube(stage, ew, size=1.0)
    _xform(stage, ew, translate=(0, 0, 0.01), scale=(200, 12, 0.08))
    _bind_material(stage, ew, mat)

    # North-south road
    ns = f"{road_root}/NorthSouth"
    _cube(stage, ns, size=1.0)
    _xform(stage, ns, translate=(0, 0, 0.01), scale=(12, 200, 0.08))
    _bind_material(stage, ns, mat)

    # Lane markings (simple yellow cubes)
    marking_mat = _get_or_create_material(stage, "MarkingMat", diffuse=(0.9, 0.8, 0.1))
    for i, y_off in enumerate(range(-90, 91, 10)):
        mp = f"{road_root}/MarkingEW_{i}"
        _cube(stage, mp, size=1.0)
        _xform(stage, mp, translate=(y_off, 0, 0.06), scale=(4, 0.3, 0.02))
        _bind_material(stage, mp, marking_mat)

    for i, x_off in enumerate(range(-90, 91, 10)):
        mp = f"{road_root}/MarkingNS_{i}"
        _cube(stage, mp, size=1.0)
        _xform(stage, mp, translate=(0, x_off, 0.06), scale=(0.3, 4, 0.02))
        _bind_material(stage, mp, marking_mat)

    _apply_semantic_label_recursive(stage, road_root, "terrain")


def _build_buildings(stage: Usd.Stage) -> None:
    """Place a variety of buildings in each quadrant of the city block."""
    bld_root = "/World/Buildings"
    _xform(stage, bld_root, translate=(0, 0, 0))
    mat_concrete = _get_or_create_material(stage, "ConcreteMat", diffuse=(0.55, 0.55, 0.50))
    mat_brick = _get_or_create_material(stage, "BrickMat", diffuse=(0.6, 0.3, 0.2))
    mat_glass = _get_or_create_material(stage, "GlassMat", diffuse=(0.4, 0.6, 0.75), opacity=0.6)

    buildings = [
        # (name, x, y, sx, sy, height, material)
        ("OfficeA",   30,  30, 16, 12, 30, mat_concrete),
        ("OfficeB",   55,  30, 10, 14, 45, mat_concrete),
        ("ApartmentA", 30, 55, 14, 14, 22, mat_brick),
        ("ApartmentB", 55, 55, 12, 10, 18, mat_brick),
        ("OfficeC",  -30,  30, 18, 14, 38, mat_concrete),
        ("ShopA",    -55,  30, 10,  8, 10, mat_brick),
        ("ApartmentC",-30, 55, 14, 12, 25, mat_brick),
        ("OfficeD",  -55, 55, 12, 12, 35, mat_concrete),
        ("OfficeE",   30, -30, 14, 16, 28, mat_concrete),
        ("ShopB",     55, -30,  8, 10, 10, mat_brick),
        ("ApartmentD", 30,-55, 14, 14, 20, mat_brick),
        ("WarehouseA", 55,-55, 18, 18, 12, mat_concrete),
        ("OfficeF",  -30, -30, 16, 12, 32, mat_concrete),
        ("ApartmentE",-55, -30, 12, 10, 15, mat_brick),
        ("ShopC",    -30, -55, 10, 10,  8, mat_brick),
        ("ApartmentF",-55, -55, 14, 14, 22, mat_brick),
    ]

    for name, x, y, sx, sy, h, mat in buildings:
        bp = f"{bld_root}/{name}"
        body_path = f"{bp}/Body"
        _xform(stage, bp, translate=(x, y, 0))
        _cube(stage, body_path, size=1.0)
        _xform(stage, body_path, translate=(0, 0, h / 2.0), scale=(sx, sy, h))
        _bind_material(stage, body_path, mat)

        # Window strip on taller buildings
        if h > 15:
            win_path = f"{bp}/Windows"
            _cube(stage, win_path, size=1.0)
            _xform(stage, win_path, translate=(sx / 2.0 + 0.05, 0, h / 2.0), scale=(0.1, sy * 0.8, h * 0.7))
            _bind_material(stage, win_path, mat_glass)

    _apply_semantic_label_recursive(stage, bld_root, "building")


def _build_vehicles(stage: Usd.Stage) -> None:
    """Place parked and street vehicles along the roads."""
    veh_root = "/World/Vehicles"
    _xform(stage, veh_root, translate=(0, 0, 0))
    colors = [
        ("CarRed",    (0.7, 0.1, 0.1)),
        ("CarBlue",   (0.1, 0.2, 0.7)),
        ("CarWhite",  (0.85, 0.85, 0.85)),
        ("CarYellow", (0.8, 0.7, 0.1)),
        ("CarBlack",  (0.08, 0.08, 0.08)),
    ]
    car_mats = {n: _get_or_create_material(stage, n, d) for n, d in colors}
    tire_mat = _get_or_create_material(stage, "TireMat", diffuse=(0.05, 0.05, 0.05))
    windshield_mat = _get_or_create_material(stage, "WindshieldMat", diffuse=(0.3, 0.5, 0.7), opacity=0.5)

    placements = [
        # (id, x, y, rotation_z, color_key)
        ("Car01",  20,  7.5, 0,   "CarRed"),
        ("Car02",  35,  7.5, 0,   "CarBlue"),
        ("Car03",  50, -7.5, 180, "CarWhite"),
        ("Car04", -20,  7.5, 0,   "CarYellow"),
        ("Car05", -40, -7.5, 180, "CarBlack"),
        ("Car06",  7.5, 25,  90,  "CarRed"),
        ("Car07", -7.5, 40,  270, "CarBlue"),
        ("Car08",  7.5,-25,  90,  "CarWhite"),
        ("Car09", -7.5,-45,  270, "CarYellow"),
        ("Car10", -65,  7.5, 0,   "CarBlack"),
        ("Car11",  70, -7.5, 180, "CarRed"),
        ("Car12",  7.5, 70,  90,  "CarBlue"),
    ]

    for vid, x, y, rz, color_key in placements:
        vp = f"{veh_root}/{vid}"
        _xform(stage, vp, translate=(x, y, 0), rotate_deg=(0, 0, rz))

        # Car body
        body = f"{vp}/Body"
        _cube(stage, body, size=1.0)
        _xform(stage, body, translate=(0, 0, 0.7), scale=(4.2, 1.8, 1.0))
        _bind_material(stage, body, car_mats[color_key])

        # Cabin / roof
        cabin = f"{vp}/Cabin"
        _cube(stage, cabin, size=1.0)
        _xform(stage, cabin, translate=(0.3, 0, 1.3), scale=(2.2, 1.6, 0.7))
        _bind_material(stage, cabin, windshield_mat)

        # Four wheels
        for wi, (wx, wy) in enumerate([(-1.2, 0.9), (-1.2, -0.9), (1.2, 0.9), (1.2, -0.9)]):
            wp = f"{vp}/Wheel_{wi}"
            _cylinder(stage, wp, radius=0.35, height=0.25)
            _xform(stage, wp, translate=(wx, wy, 0.35), rotate_deg=(90, 0, 0))
            _bind_material(stage, wp, tire_mat)

    _apply_semantic_label_recursive(stage, veh_root, "vehicle")


def _build_pedestrians(stage: Usd.Stage) -> None:
    """Place stylised pedestrian stand-ins (capsule body + sphere head)."""
    ped_root = "/World/Pedestrians"
    _xform(stage, ped_root, translate=(0, 0, 0))
    skin_mat = _get_or_create_material(stage, "SkinMat", diffuse=(0.8, 0.6, 0.5))
    shirt_mats = [
        _get_or_create_material(stage, "ShirtBlue", diffuse=(0.15, 0.25, 0.6)),
        _get_or_create_material(stage, "ShirtGreen", diffuse=(0.2, 0.5, 0.2)),
        _get_or_create_material(stage, "ShirtRed", diffuse=(0.6, 0.15, 0.15)),
        _get_or_create_material(stage, "ShirtOrange", diffuse=(0.8, 0.45, 0.1)),
    ]

    pedestrians = [
        # (name, x, y)
        ("Ped01", 15,  3), ("Ped02", 22, -3), ("Ped03", -18,  4),
        ("Ped04", -35, -3), ("Ped05",  3, 18), ("Ped06", -4, 35),
        ("Ped07",  3, -22), ("Ped08", -4, -40), ("Ped09", 40, 40),
        ("Ped10", -42, 38), ("Ped11", 38, -42), ("Ped12", -40, -38),
    ]

    for idx, (name, x, y) in enumerate(pedestrians):
        pp = f"{ped_root}/{name}"
        _xform(stage, pp, translate=(x, y, 0))

        # Torso (capsule approximated by cylinder + half-spheres)
        torso = f"{pp}/Torso"
        _cylinder(stage, torso, radius=0.22, height=0.9)
        _xform(stage, torso, translate=(0, 0, 1.0))
        _bind_material(stage, torso, shirt_mats[idx % len(shirt_mats)])

        # Head
        head = f"{pp}/Head"
        _sphere(stage, head, radius=0.16)
        _xform(stage, head, translate=(0, 0, 1.65))
        _bind_material(stage, head, skin_mat)

        # Legs (two thin cylinders)
        for li, ly in enumerate([-0.1, 0.1]):
            leg = f"{pp}/Leg_{li}"
            _cylinder(stage, leg, radius=0.08, height=0.55)
            _xform(stage, leg, translate=(0, ly, 0.28))
            _bind_material(stage, leg, shirt_mats[(idx + 2) % len(shirt_mats)])

    _apply_semantic_label_recursive(stage, ped_root, "person")


def _build_vegetation(stage: Usd.Stage) -> None:
    """Place trees and park patches along sidewalks."""
    veg_root = "/World/Vegetation"
    _xform(stage, veg_root, translate=(0, 0, 0))
    trunk_mat = _get_or_create_material(stage, "TrunkMat", diffuse=(0.35, 0.2, 0.1))
    canopy_mat = _get_or_create_material(stage, "CanopyMat", diffuse=(0.15, 0.45, 0.12))
    grass_mat = _get_or_create_material(stage, "GrassMat", diffuse=(0.2, 0.55, 0.15))

    tree_spots = [
        (18, 14), (35, 14), (52, 14), (-18, 14), (-35, 14), (-52, 14),
        (14, 18), (14, 35), (14, 52), (-14, 18), (-14, 35), (-14, 52),
        (18, -14), (35, -14), (52, -14), (-18, -14), (-35, -14), (-52, -14),
        (14, -18), (14, -35), (14, -52), (-14, -18), (-14, -35), (-14, -52),
    ]

    for i, (tx, ty) in enumerate(tree_spots):
        tp = f"{veg_root}/Tree_{i:02d}"
        _xform(stage, tp, translate=(tx, ty, 0))

        trunk = f"{tp}/Trunk"
        _cylinder(stage, trunk, radius=0.18, height=3.0)
        _xform(stage, trunk, translate=(0, 0, 1.5))
        _bind_material(stage, trunk, trunk_mat)

        canopy = f"{tp}/Canopy"
        _sphere(stage, canopy, radius=1.8)
        _xform(stage, canopy, translate=(0, 0, 4.2))
        _bind_material(stage, canopy, canopy_mat)

    # Small park patches
    park_locations = [(40, 42), (-40, 42), (40, -42), (-40, -42)]
    for i, (px, py) in enumerate(park_locations):
        pp = f"{veg_root}/Park_{i}"
        _cube(stage, pp, size=1.0)
        _xform(stage, pp, translate=(px, py, 0.02), scale=(10, 10, 0.04))
        _bind_material(stage, pp, grass_mat)

    _apply_semantic_label_recursive(stage, veg_root, "vegetation")


# ---------------------------------------------------------------------------
# Disaster zones
# ---------------------------------------------------------------------------

def _build_disaster_collapsed_building(stage: Usd.Stage) -> None:
    """Disaster zone 1 — a collapsed building with rubble debris."""
    dz = "/World/DisasterZones/CollapsedBuilding"
    _xform(stage, "/World/DisasterZones", translate=(0, 0, 0))
    _xform(stage, dz, translate=(-30, -55, 0))

    rubble_mat = _get_or_create_material(stage, "RubbleMat", diffuse=(0.45, 0.42, 0.38))
    rebar_mat = _get_or_create_material(stage, "RebarMat", diffuse=(0.35, 0.22, 0.15))

    # Partially standing wall segments
    for i, (dx, dy, dz_off, sx, sy, sz, rz) in enumerate([
        (-3, 0, 3.5, 8, 0.5, 7, 0),
        (3,  2, 2.5, 0.5, 6, 5, 0),
        (0, -3, 1.5, 6, 0.5, 3, 15),
    ]):
        wp = f"{dz}/Wall_{i}"
        _cube(stage, wp, size=1.0)
        _xform(stage, wp, translate=(dx, dy, dz_off), scale=(sx, sy, sz), rotate_deg=(0, 0, rz))
        _bind_material(stage, wp, rubble_mat)

    # Rubble / debris chunks (tilted cubes)
    import random
    random.seed(42)
    for i in range(20):
        rp = f"{dz}/Rubble_{i:02d}"
        s = random.uniform(0.3, 1.8)
        _cube(stage, rp, size=1.0)
        _xform(
            stage, rp,
            translate=(random.uniform(-6, 6), random.uniform(-6, 6), s / 2.0),
            scale=(s, s * random.uniform(0.5, 1.5), s * random.uniform(0.3, 1.0)),
            rotate_deg=(random.uniform(-25, 25), random.uniform(-25, 25), random.uniform(0, 360)),
        )
        _bind_material(stage, rp, rubble_mat if i % 3 else rebar_mat)

    # Fallen rebar cylinders
    for i in range(6):
        rb = f"{dz}/Rebar_{i}"
        _cylinder(stage, rb, radius=0.04, height=random.uniform(2.0, 5.0))
        _xform(
            stage, rb,
            translate=(random.uniform(-5, 5), random.uniform(-5, 5), 0.5),
            rotate_deg=(random.uniform(-40, 40), random.uniform(-40, 40), 0),
        )
        _bind_material(stage, rb, rebar_mat)

    _apply_semantic_label_recursive(stage, dz, "building")


def _build_disaster_vehicle_fire(stage: Usd.Stage) -> None:
    """Disaster zone 2 — a burning vehicle with fire/flame meshes."""
    dz = "/World/DisasterZones/VehicleFire"
    _xform(stage, dz, translate=(50, -7.5, 0))

    # Burnt-out car shell
    burn_mat = _get_or_create_material(stage, "BurntMetalMat", diffuse=(0.1, 0.08, 0.06))
    body = f"{dz}/BurntCar/Body"
    _xform(stage, f"{dz}/BurntCar", translate=(0, 0, 0))
    _cube(stage, body, size=1.0)
    _xform(stage, body, translate=(0, 0, 0.7), scale=(4.2, 1.8, 1.0))
    _bind_material(stage, body, burn_mat)

    cabin = f"{dz}/BurntCar/Cabin"
    _cube(stage, cabin, size=1.0)
    _xform(stage, cabin, translate=(0.3, 0, 1.3), scale=(2.0, 1.5, 0.5))
    _bind_material(stage, cabin, burn_mat)

    # Label the burnt car as vehicle
    _apply_semantic_label_recursive(stage, f"{dz}/BurntCar", "vehicle")

    # Fire / flames — cone + sphere clusters to represent flickering fire
    fire_root = f"{dz}/Flames"
    _xform(stage, fire_root, translate=(0, 0, 1.2))
    fire_mat = _get_or_create_material(
        stage, "FireMat",
        diffuse=(1.0, 0.35, 0.0),
        emissive=(4.0, 1.5, 0.1),
    )
    fire_core_mat = _get_or_create_material(
        stage, "FireCoreMat",
        diffuse=(1.0, 0.85, 0.2),
        emissive=(6.0, 3.0, 0.3),
    )

    flame_elements = [
        # (name, shape, params, translate, scale, rotate)
        ("Flame_Core",    "cone",   {"radius": 0.5, "height": 2.5}, (0, 0, 1.25),    (1, 1, 1),       (0, 0, 0)),
        ("Flame_Left",    "cone",   {"radius": 0.35, "height": 1.8}, (-0.6, 0.3, 0.9), (1, 1, 1),      (0, 0, 10)),
        ("Flame_Right",   "cone",   {"radius": 0.35, "height": 1.8}, (0.5, -0.2, 0.9), (1, 1, 1),      (0, 0, -8)),
        ("Flame_Back",    "cone",   {"radius": 0.3, "height": 1.5},  (-0.8, 0, 0.75),  (1, 1, 1),      (0, 0, 15)),
        ("Ember_A",       "sphere", {"radius": 0.25},                (0.3, 0.4, 2.4),  (1, 1, 1),      (0, 0, 0)),
        ("Ember_B",       "sphere", {"radius": 0.2},                 (-0.4, -0.3, 2.6),(1, 1, 1),      (0, 0, 0)),
        ("Ember_C",       "sphere", {"radius": 0.18},                (0.1, 0.1, 2.9),  (1, 1, 1),      (0, 0, 0)),
        ("Glow_Base",     "sphere", {"radius": 0.8},                 (0, 0, 0.3),      (1.5, 1.5, 0.5),(0, 0, 0)),
    ]

    for fname, shape, params, trans, scl, rot in flame_elements:
        fp = f"{fire_root}/{fname}"
        if shape == "cone":
            _cone(stage, fp, **params)
        else:
            _sphere(stage, fp, **params)
        _xform(stage, fp, translate=trans, scale=scl, rotate_deg=rot)
        is_core = "Core" in fname or "Glow" in fname
        _bind_material(stage, fp, fire_core_mat if is_core else fire_mat)

    # Smoke column above fire (semi-transparent dark spheres)
    smoke_mat = _get_or_create_material(stage, "SmokeMat", diffuse=(0.15, 0.15, 0.15), opacity=0.35)
    for i in range(5):
        sp = f"{fire_root}/Smoke_{i}"
        r = 0.6 + i * 0.3
        _sphere(stage, sp, radius=r)
        _xform(stage, sp, translate=(i * 0.15, i * 0.1, 3.0 + i * 1.2))
        _bind_material(stage, sp, smoke_mat)

    _apply_semantic_label_recursive(stage, fire_root, "fire")


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def _build_lighting(stage: Usd.Stage) -> None:
    """Add a distant (sun) light and a dome light for ambient fill."""
    light_root = "/World/Lights"
    _xform(stage, light_root, translate=(0, 0, 0))

    sun = UsdGeom.Xform.Define(stage, f"{light_root}/Sun")
    sun.AddTranslateOp().Set(Gf.Vec3d(0, 0, 100))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))
    sun_light = stage.DefinePrim(f"{light_root}/Sun/DistantLight", "DistantLight")
    sun_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)
    sun_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.53)
    sun_light.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.95, 0.85))

    dome = stage.DefinePrim(f"{light_root}/DomeLight", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(500.0)


# ===========================================================================
# Main
# ===========================================================================

def generate_scene() -> Usd.Stage:
    """Build the full urban disaster scene and return the USD stage."""
    # Create a fresh stage
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Set stage up-axis and meters-per-unit
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root Xform
    _xform(stage, "/World", translate=(0, 0, 0))

    print("[ResQ-AI] Building terrain …")
    _build_ground_plane(stage)

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

    print("[ResQ-AI] Creating disaster zone 1 — collapsed building …")
    _build_disaster_collapsed_building(stage)

    print("[ResQ-AI] Creating disaster zone 2 — vehicle fire …")
    _build_disaster_vehicle_fire(stage)

    print("[ResQ-AI] Setting up lighting …")
    _build_lighting(stage)

    print("[ResQ-AI] Scene generation complete.")
    return stage


def main() -> None:
    stage = generate_scene()

    # Optionally save the composed stage to disk
    output_path = "/tmp/resqai_urban_disaster.usda"
    stage.GetRootLayer().Export(output_path)
    print(f"[ResQ-AI] Stage exported to {output_path}")

    if not _args.headless:
        # Keep the app alive for interactive inspection
        world = World()
        world.reset()
        print("[ResQ-AI] Running viewer — close the window to exit.")
        while simulation_app.is_running():
            world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
