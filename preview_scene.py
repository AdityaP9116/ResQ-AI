#!/usr/bin/env python3
"""ResQ-AI Scene Preview — Standalone USD Generator

Generates a self-contained .usda scene with the full orchestrator layout
(terrain, buildings, forest, characters, fire markers, lighting) using
only ``pip install usd-core``.  No Isaac Sim required.

Usage:
    python3 preview_scene.py                     # default output
    python3 preview_scene.py -o my_scene.usda    # custom output name
    python3 preview_scene.py --wind 1.0 0.5 0.0  # wind from northeast

View with:
    usdview resqai_preview.usda
    # or drag into Blender (File → Import → USD)
    # or open in NVIDIA Omniverse USD Composer
"""

import argparse
import math
import os
import random
import sys
from typing import List, Tuple, Optional, Dict, Any

# ── USD core imports (from pip install usd-core) ─────────────────────────────
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
except ImportError:
    print("ERROR: 'usd-core' is not installed.")
    print("       Run:  pip install usd-core")
    sys.exit(1)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Generate a ResQ-AI USD preview")
    p.add_argument("-o", "--output", default="resqai_preview.usda",
                   help="Output file name (default: resqai_preview.usda)")
    p.add_argument("--num-trees", type=int, default=120)
    p.add_argument("--num-pedestrians", type=int, default=8)
    p.add_argument("--wind", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   help="Wind direction vector X Y Z")
    p.add_argument("--wind-speed", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  USD Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _xform(stage, path, translate=None, rotate_z=None, scale=None):
    """Create an Xform prim and optionally set translate/rotate/scale."""
    xf = UsdGeom.Xform.Define(stage, path)
    if translate:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z:
        xf.AddRotateZOp().Set(rotate_z)
    if scale:
        xf.AddScaleOp().Set(Gf.Vec3d(*scale))
    return xf


def _cube(stage, path, size=1.0, translate=None, scale=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    if translate:
        cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if scale:
        cube.AddScaleOp().Set(Gf.Vec3d(*scale))
    return cube


def _cylinder(stage, path, radius=0.5, height=1.0, translate=None):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    if translate:
        cyl.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return cyl


def _sphere(stage, path, radius=0.5, translate=None, scale=None):
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    if translate:
        sph.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if scale:
        sph.AddScaleOp().Set(Gf.Vec3d(*scale))
    return sph


def _cone(stage, path, radius=1.0, height=2.0, translate=None):
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    if translate:
        cone.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return cone


def _mat(stage, name, albedo=(0.5, 0.5, 0.5), roughness=0.5,
         metallic=0.0, opacity=1.0,
         emissive=None, emissive_intensity=0.0):
    """Create a UsdPreviewSurface material under /World/Looks."""
    mat_path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*albedo))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*emissive))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat_path


def _bind(stage, prim_path, mat_path):
    """Bind a material to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    if prim.IsValid() and mat:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Building data
# ═══════════════════════════════════════════════════════════════════════════════

class BuildingInfo:
    def __init__(self, name, world_x, world_y, half_sx, half_sy, height, color):
        self.name = name
        self.world_x = world_x
        self.world_y = world_y
        self.half_sx = half_sx
        self.half_sy = half_sy
        self.height = height
        self.color = color
        self.pedestrian_paths = []
        self.is_burning = False


# Predefined building layout matching the base USDA scene
BUILDINGS = [
    BuildingInfo("OfficeA",    25,  25,  8,  6, 28, (0.60, 0.62, 0.65)),
    BuildingInfo("OfficeB",   -25,  25,  7,  7, 22, (0.55, 0.58, 0.62)),
    BuildingInfo("Hospital",   25, -25, 10,  8, 18, (0.90, 0.92, 0.95)),
    BuildingInfo("School",    -25, -25,  9,  7, 14, (0.75, 0.70, 0.60)),
    BuildingInfo("Apartments", 0,   35,  6, 10, 32, (0.50, 0.45, 0.42)),
    BuildingInfo("Tower",      0,  -35,  5,  5, 45, (0.38, 0.42, 0.48)),
    BuildingInfo("Mall",       35,   0, 12,  8, 12, (0.82, 0.78, 0.72)),
    BuildingInfo("Warehouse", -35,   0, 11,  9, 10, (0.65, 0.60, 0.55)),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene generators
# ═══════════════════════════════════════════════════════════════════════════════

def create_terrain(stage):
    """Ground plane + sidewalks."""
    print("  • Terrain")
    ground_mat = _mat(stage, "GroundMat", albedo=(0.18, 0.28, 0.12),
                      roughness=0.95)
    _cube(stage, "/World/Terrain/Ground", size=1.0,
          translate=(0, 0, -0.5), scale=(200, 200, 1))
    _bind(stage, "/World/Terrain/Ground", ground_mat)

    road_mat = _mat(stage, "RoadMat", albedo=(0.15, 0.15, 0.17),
                    roughness=0.7)
    # Cross-shaped road
    _cube(stage, "/World/Roads/RoadNS", size=1.0,
          translate=(0, 0, 0.02), scale=(8, 200, 0.04))
    _bind(stage, "/World/Roads/RoadNS", road_mat)
    _cube(stage, "/World/Roads/RoadEW", size=1.0,
          translate=(0, 0, 0.02), scale=(200, 8, 0.04))
    _bind(stage, "/World/Roads/RoadEW", road_mat)

    sidewalk_mat = _mat(stage, "SidewalkMat", albedo=(0.45, 0.43, 0.40),
                        roughness=0.8)
    for i, (dx, dy, sx, sy) in enumerate([
        (0, 80, 200, 3), (0, -80, 200, 3),
        (80, 0, 3, 200), (-80, 0, 3, 200),
    ]):
        _cube(stage, f"/World/Terrain/Walk_{i}", size=1.0,
              translate=(dx, dy, 0.05), scale=(sx, sy, 0.1))
        _bind(stage, f"/World/Terrain/Walk_{i}", sidewalk_mat)


def create_buildings(stage, buildings):
    """Procedural buildings with windows."""
    print("  • Buildings")
    bldg_root = "/World/Buildings"
    _xform(stage, bldg_root)

    glass_mat = _mat(stage, "GlassMat", albedo=(0.12, 0.18, 0.28),
                     roughness=0.08, metallic=0.7, opacity=0.5)
    dark_win = _mat(stage, "DarkWinMat", albedo=(0.06, 0.08, 0.12),
                    roughness=0.1, metallic=0.5, opacity=0.6)

    for b in buildings:
        bp = f"{bldg_root}/{b.name}"
        _xform(stage, bp)

        # Building body
        body_mat = _mat(stage, f"Bldg_{b.name}_Mat", albedo=b.color,
                        roughness=0.6)
        _cube(stage, f"{bp}/Body", size=1.0,
              translate=(b.world_x, b.world_y, b.height / 2),
              scale=(b.half_sx * 2, b.half_sy * 2, b.height))
        _bind(stage, f"{bp}/Body", body_mat)

        # Roof cap (slightly lighter)
        rc = tuple(min(1.0, c + 0.1) for c in b.color)
        roof_mat = _mat(stage, f"Bldg_{b.name}_Roof", albedo=rc,
                        roughness=0.5)
        _cube(stage, f"{bp}/Roof", size=1.0,
              translate=(b.world_x, b.world_y, b.height + 0.15),
              scale=(b.half_sx * 2 + 0.4, b.half_sy * 2 + 0.4, 0.3))
        _bind(stage, f"{bp}/Roof", roof_mat)

        # Windows on all 4 faces
        floor_h = 4.0
        num_floors = max(1, int(b.height / floor_h))
        faces = [
            ("FXP", b.half_sx + 0.08, 0, b.half_sy * 0.85, 0.08),
            ("FXN", -(b.half_sx + 0.08), 0, b.half_sy * 0.85, 0.08),
            ("FYP", 0, b.half_sy + 0.08, 0.08, b.half_sx * 0.85),
            ("FYN", 0, -(b.half_sy + 0.08), 0.08, b.half_sx * 0.85),
        ]
        for fname, fx, fy, wsx, wsy in faces:
            for fl in range(num_floors):
                fz = 3.0 + fl * max(1, (b.height - 4.0)) / max(num_floors, 1)
                wm = glass_mat if fl % 3 != 0 else dark_win
                wp = f"{bp}/Win_{fname}_F{fl}"
                _cube(stage, wp, size=1.0,
                      translate=(b.world_x + fx, b.world_y + fy, fz),
                      scale=(wsx, wsy, 1.8))
                _bind(stage, wp, wm)


def create_forest(stage, buildings, num_trees=120, seed=42):
    """4-prototype forest with clustering — returns positions."""
    print(f"  • Forest ({num_trees} trees, 4 types)")
    random.seed(seed)

    _xform(stage, "/World/Forest")
    proto_root = "/World/Forest/Prototypes"
    _xform(stage, proto_root)

    # Materials
    bark = _mat(stage, "Bark", albedo=(0.28, 0.16, 0.08), roughness=0.92)
    bark_lt = _mat(stage, "BarkLight", albedo=(0.55, 0.45, 0.35), roughness=0.9)
    pine_m = _mat(stage, "Pine", albedo=(0.06, 0.22, 0.06), roughness=0.88)
    oak_m = _mat(stage, "Oak", albedo=(0.12, 0.38, 0.10), roughness=0.85)
    bush_m = _mat(stage, "Bush", albedo=(0.15, 0.30, 0.08), roughness=0.8)
    birch_m = _mat(stage, "Birch", albedo=(0.20, 0.42, 0.15), roughness=0.82)

    inner_r, outer_r = 15.0, 65.0

    def _inside_bldg(x, y):
        for b in buildings:
            if abs(x - b.world_x) < b.half_sx + 3 and \
               abs(y - b.world_y) < b.half_sy + 3:
                return True
        return False

    def _near_road(x, y):
        w = 5.0
        return abs(x) < w or abs(y) < w

    # Cluster centres
    clusters = []
    for _ in range(num_trees // 8):
        a = random.uniform(0, 2 * math.pi)
        r = random.uniform(inner_r + 5, outer_r - 5)
        cx, cy = r * math.cos(a), r * math.sin(a)
        if not _inside_bldg(cx, cy) and not _near_road(cx, cy):
            clusters.append((cx, cy))

    tree_positions = []
    attempts = 0
    tree_id = 0

    while len(tree_positions) < num_trees and attempts < num_trees * 15:
        attempts += 1

        if clusters and random.random() < 0.6:
            ccx, ccy = random.choice(clusters)
            x = ccx + random.gauss(0, 4.0)
            y = ccy + random.gauss(0, 4.0)
        else:
            ang = random.uniform(0, 2 * math.pi)
            rad = random.uniform(inner_r, outer_r)
            x, y = rad * math.cos(ang), rad * math.sin(ang)

        if _inside_bldg(x, y):
            continue
        if _near_road(x, y) and random.random() < 0.7:
            continue

        # Pick type
        rv = random.random()
        if rv < 0.35:
            kind, s = "Pine", random.uniform(0.8, 1.3)
        elif rv < 0.60:
            kind, s = "Oak", random.uniform(0.7, 1.2)
        elif rv < 0.85:
            kind, s = "Bush", random.uniform(0.6, 1.5)
        else:
            kind, s = "Birch", random.uniform(0.9, 1.4)

        tp = f"/World/Forest/T{tree_id:03d}"
        _xform(stage, tp, translate=(x, y, 0), scale=(s, s, s))

        if kind == "Pine":
            _cylinder(stage, f"{tp}/Trunk", 0.15, 4.5, translate=(0, 0, 2.25))
            _bind(stage, f"{tp}/Trunk", bark)
            _cone(stage, f"{tp}/Can", 2.0, 5.0, translate=(0, 0, 6.5))
            _bind(stage, f"{tp}/Can", pine_m)
        elif kind == "Oak":
            _cylinder(stage, f"{tp}/Trunk", 0.25, 3.0, translate=(0, 0, 1.5))
            _bind(stage, f"{tp}/Trunk", bark)
            _sphere(stage, f"{tp}/Can", 3.0, translate=(0, 0, 5.0))
            _bind(stage, f"{tp}/Can", oak_m)
        elif kind == "Bush":
            _sphere(stage, f"{tp}/Body", 1.2, translate=(0, 0, 0.8),
                    scale=(1.3, 1.3, 0.8))
            _bind(stage, f"{tp}/Body", bush_m)
        else:  # Birch
            _cylinder(stage, f"{tp}/Trunk", 0.10, 5.5, translate=(0, 0, 2.75))
            _bind(stage, f"{tp}/Trunk", bark_lt)
            _sphere(stage, f"{tp}/Can", 1.8, translate=(0, 0, 6.5),
                    scale=(1.0, 1.0, 1.3))
            _bind(stage, f"{tp}/Can", birch_m)

        tree_positions.append(Gf.Vec3f(x, y, 0))
        tree_id += 1

    print(f"    Placed {len(tree_positions)} trees")
    return tree_positions


def create_characters(stage, buildings, count=8, seed=42):
    """Geometry characters placed inside buildings."""
    print(f"  • Characters ({count} victims)")
    random.seed(seed + 100)

    _xform(stage, "/World/Victims")
    skin = _mat(stage, "Skin", albedo=(0.72, 0.55, 0.45), roughness=0.85)
    shirts = [
        _mat(stage, "ShirtA", albedo=(0.1, 0.15, 0.35), roughness=0.75),
        _mat(stage, "ShirtB", albedo=(0.6, 0.15, 0.1), roughness=0.75),
        _mat(stage, "ShirtC", albedo=(0.15, 0.4, 0.15), roughness=0.75),
    ]
    pants = _mat(stage, "Pants", albedo=(0.15, 0.15, 0.18), roughness=0.8)

    vi = 0
    for bldg in buildings:
        per_bldg = max(1, count // len(buildings))
        for j in range(per_bldg):
            if vi >= count:
                break
            path = f"/World/Victims/V{vi:02d}"
            shirt = shirts[vi % len(shirts)]

            # Position inside building
            ox = bldg.world_x + random.uniform(-bldg.half_sx + 1.5,
                                                bldg.half_sx - 1.5)
            oy = bldg.world_y + random.uniform(-bldg.half_sy + 1.5,
                                                bldg.half_sy - 1.5)
            oz = 0.0

            _xform(stage, path, translate=(ox, oy, oz))

            # Torso
            _cylinder(stage, f"{path}/Torso", 0.22, 0.50,
                      translate=(0, 0, 1.15))
            _bind(stage, f"{path}/Torso", shirt)

            # Head
            _sphere(stage, f"{path}/Head", 0.12, translate=(0, 0, 1.63))
            _bind(stage, f"{path}/Head", skin)

            # Neck
            _cylinder(stage, f"{path}/Neck", 0.06, 0.12,
                      translate=(0, 0, 1.47))
            _bind(stage, f"{path}/Neck", skin)

            # Hips
            _cylinder(stage, f"{path}/Hips", 0.20, 0.15,
                      translate=(0, 0, 0.85))
            _bind(stage, f"{path}/Hips", pants)

            # Legs
            for li, ly in enumerate([-0.10, 0.10]):
                _cylinder(stage, f"{path}/Leg{li}", 0.08, 0.55,
                          translate=(0, ly, 0.33))
                _bind(stage, f"{path}/Leg{li}", pants)

            # Arms
            for ai, ay in enumerate([-0.30, 0.30]):
                _cylinder(stage, f"{path}/Arm{ai}", 0.06, 0.55,
                          translate=(0, ay, 1.05))
                _bind(stage, f"{path}/Arm{ai}", shirt)

            bldg.pedestrian_paths.append(path)
            vi += 1


def create_fire_markers(stage, buildings, tree_positions, wind_dir, seed=42):
    """Static fire markers — emissive geometry cones + lights.

    Creates vegetation fires (ground level) and building fire positions.
    """
    print("  • Fire markers (vegetation + building)")
    random.seed(seed + 200)

    _xform(stage, "/World/Fires")
    flame_mat = _mat(stage, "Flame", albedo=(1.0, 0.3, 0.0),
                     emissive=(1.0, 0.4, 0.0), emissive_intensity=5000,
                     roughness=1.0)
    core_mat = _mat(stage, "FlameCore", albedo=(1.0, 0.9, 0.3),
                    emissive=(1.0, 0.9, 0.3), emissive_intensity=8000,
                    roughness=1.0)

    # Score trees by windward position
    wind_norm = Gf.Vec3f(*wind_dir)
    wlen = wind_norm.GetLength()
    if wlen > 1e-6:
        wind_norm /= wlen

    scored = []
    for tp in tree_positions:
        tlen = tp.GetLength()
        if tlen < 1e-6:
            continue
        alignment = Gf.Dot(tp.GetNormalized(), wind_norm)
        scored.append((alignment, tp))
    scored.sort(key=lambda x: -x[0])

    # Vegetation fires (first 12 windward trees, first 3 visible)
    _xform(stage, "/World/Fires/VegFires")
    num_vf = min(12, len(scored))
    veg_fires = []
    for i in range(num_vf):
        _, tp = scored[i]
        vf_path = f"/World/Fires/VegFires/F{i:02d}"
        _xform(stage, vf_path, translate=(float(tp[0]), float(tp[1]), 0.5))

        # Fire cones
        _cone(stage, f"{vf_path}/Flame1", 1.2, 3.5, translate=(0, 0, 1.8))
        _bind(stage, f"{vf_path}/Flame1", flame_mat)
        _cone(stage, f"{vf_path}/Core", 0.6, 2.5, translate=(0, 0, 1.5))
        _bind(stage, f"{vf_path}/Core", core_mat)

        # Fire glow light
        light = UsdLux.SphereLight.Define(stage, f"{vf_path}/Glow")
        light.CreateRadiusAttr(1.5)
        light.CreateIntensityAttr(8000)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.05))
        light.AddTranslateOp().Set(Gf.Vec3d(0, 0, 4))

        active = i < 3
        veg_fires.append({"path": vf_path, "active": active,
                          "pos": Gf.Vec3f(float(tp[0]), float(tp[1]), 0.5)})

        # Hide inactive fires
        if not active:
            UsdGeom.Imageable(stage.GetPrimAtPath(vf_path)).MakeInvisible()

    print(f"    {num_vf} vegetation fires ({sum(1 for v in veg_fires if v['active'])} active)")

    # Building fires (initially not burning, but show markers for reference)
    for b in buildings:
        fp = f"/World/Fires/B_{b.name}"
        fz = b.height + 2.0
        _xform(stage, fp, translate=(b.world_x, b.world_y, fz))
        _cone(stage, f"{fp}/Flame1", 1.5, 4.0, translate=(0, 0, 2))
        _bind(stage, f"{fp}/Flame1", flame_mat)
        _sphere(stage, f"{fp}/Core", 0.8, translate=(0, 0, 2.5))
        _bind(stage, f"{fp}/Core", core_mat)

        light = UsdLux.SphereLight.Define(stage, f"{fp}/Glow")
        light.CreateRadiusAttr(2.0)
        light.CreateIntensityAttr(15000)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.05))
        light.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))

        # All building fires start invisible
        UsdGeom.Imageable(stage.GetPrimAtPath(fp)).MakeInvisible()

    return veg_fires


def create_lighting(stage):
    """Sky dome, distant sun, ambient fill."""
    print("  • Lighting")
    _xform(stage, "/World/Lights")

    # Sun
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    sun.CreateAngleAttr(0.53)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-55, 30, 0))

    # Sky dome
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    dome.CreateIntensityAttr(500)
    dome.CreateColorAttr(Gf.Vec3f(0.6, 0.72, 0.88))

    # Ambient fill
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/Fill")
    fill.CreateRadiusAttr(0.5)
    fill.CreateIntensityAttr(2000)
    fill.CreateColorAttr(Gf.Vec3f(0.9, 0.85, 0.8))
    fill.AddTranslateOp().Set(Gf.Vec3d(0, 0, 80))


def create_wind_indicator(stage, wind_dir, wind_speed):
    """Visual wind direction arrow on the ground."""
    print(f"  • Wind arrow: ({wind_dir[0]:.1f}, {wind_dir[1]:.1f}, "
          f"{wind_dir[2]:.1f}) at {wind_speed:.0f} m/s")
    _xform(stage, "/World/WindArrow")

    # Arrow shaft (cylinder along wind direction)
    wind_norm = Gf.Vec3f(*wind_dir)
    wlen = wind_norm.GetLength()
    if wlen > 1e-6:
        wind_norm /= wlen

    arrow_mat = _mat(stage, "ArrowMat", albedo=(0.2, 0.6, 1.0),
                     emissive=(0.2, 0.6, 1.0), emissive_intensity=1000,
                     roughness=1.0)

    # Place arrow at origin, pointing in wind direction
    angle_z = math.degrees(math.atan2(float(wind_norm[1]),
                                       float(wind_norm[0])))

    arrow_len = 8.0 + wind_speed * 0.5
    _xform(stage, "/World/WindArrow/Arrow", translate=(0, 0, 0.3),
           rotate_z=angle_z)
    _cylinder(stage, "/World/WindArrow/Arrow/Shaft", 0.3, arrow_len,
              translate=(arrow_len / 2, 0, 0))
    _bind(stage, "/World/WindArrow/Arrow/Shaft", arrow_mat)
    _cone(stage, "/World/WindArrow/Arrow/Head", 0.8, 2.0,
          translate=(arrow_len + 1, 0, 0))
    _bind(stage, "/World/WindArrow/Arrow/Head", arrow_mat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_path = os.path.join(_SCRIPT_DIR, args.output)

    print("=" * 60)
    print("  ResQ-AI — Standalone Scene Preview Generator")
    print("=" * 60)
    print(f"  Output: {out_path}")
    print(f"  Trees:  {args.num_trees}  |  Victims: {args.num_pedestrians}")
    print(f"  Wind:   ({args.wind[0]:.1f}, {args.wind[1]:.1f}, "
          f"{args.wind[2]:.1f}) at {args.wind_speed:.0f} m/s")
    print("-" * 60)

    # Create a new USD stage
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _xform(stage, "/World")
    _xform(stage, "/World/Looks")
    _xform(stage, "/World/Terrain")
    _xform(stage, "/World/Roads")

    buildings = list(BUILDINGS)

    create_terrain(stage)
    create_buildings(stage, buildings)
    tree_pos = create_forest(stage, buildings, args.num_trees, args.seed)
    create_characters(stage, buildings, args.num_pedestrians, args.seed)
    create_fire_markers(stage, buildings, tree_pos, args.wind, args.seed)
    create_lighting(stage)
    create_wind_indicator(stage, args.wind, args.wind_speed)

    stage.GetRootLayer().Save()

    print("-" * 60)
    print(f"✅ Scene saved to: {out_path}")
    print(f"   ({os.path.getsize(out_path) / 1024:.0f} KB)")
    print()
    print("To view:")
    print(f"  usdview {args.output}")
    print(f"  # or open in Blender: File → Import → USD")
    print(f"  # or open in NVIDIA Omniverse USD Composer")
    print("=" * 60)


if __name__ == "__main__":
    main()
