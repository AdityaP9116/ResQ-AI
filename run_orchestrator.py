#!/usr/bin/env python3
"""ResQ-AI Orchestrator — Procedural urban disaster simulation for drone SAR.

This standalone script drives an NVIDIA Isaac Sim 5.1 simulation end-to-end:

1. Loads the pre-authored ``resqai_urban_disaster.usda`` stage.
2. Discovers every building prim by checking semantic labels
   (``SemanticsLabelsAPI:class == "building"``) with a prim-path fallback.
3. Procedurally generates a dense forest ring around the city perimeter using
   ``UsdGeom.PointInstancer``.
4. Spawns a fire emitter inside each building (initially invisible) via
   ``omni.flow`` with an emissive-geometry fallback.
5. Places 15-30 pedestrian agents **inside** buildings (the drone's search
   targets) with wandering navigation waypoints.
6. Runs a physics-step callback that spreads fire between buildings based on
   distance, toggles fire visibility, and triggers indoor occupants to flee
   once their building ignites.

Usage (from the Isaac Sim Python environment)::

    /isaac-sim/python.sh run_orchestrator.py                # GUI mode
    /isaac-sim/python.sh run_orchestrator.py --headless      # headless mode

Important constraints satisfied:
  • No building fracture / collapse logic.
  • Standard OpenUSD + Isaac Sim core APIs only.
  • No deprecated ``omni.particle.system``.
  • ``omni.flow`` for fire/smoke; ``omni.anim.people`` + ``omni.anim.navigation``
    for pedestrians.
"""

from __future__ import annotations

import argparse
import math
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Isaac Sim bootstrap — MUST precede all Omniverse / pxr imports
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI disaster orchestrator")
    parser.add_argument("--headless", action="store_true",
                        help="Run without a viewport window")
    parser.add_argument("--scene", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "resqai_urban_disaster.usda"),
                        help="Path to the base USDA scene file")
    parser.add_argument("--num-pedestrians", type=int, default=20,
                        help="Number of indoor victims to spawn (15-30)")
    parser.add_argument("--fire-spread-radius", type=float, default=35.0,
                        help="Distance threshold (metres) for fire spreading "
                             "(buildings are ~28 m apart in the default scene)")
    parser.add_argument("--wind-direction", type=float, nargs=3,
                        default=[1.0, 0.0, 0.0],
                        help="Wind direction vector (X Y Z), default: 1 0 0")
    parser.add_argument("--wind-speed", type=float, default=5.0,
                        help="Wind speed in m/s (affects fire spread bias)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N physics steps (0 = unlimited)")
    return parser.parse_args()

_args = _parse_args()

# Clamp pedestrian count to the requested 15-30 range
_args.num_pedestrians = max(15, min(30, _args.num_pedestrians))

from isaacsim import SimulationApp  # noqa: E402

# Only create SimulationApp when run as standalone script. When imported (e.g. by
# headless_e2e_test via orchestrator.orchestrator_bridge), the caller has already
# created SimulationApp — creating a second one causes an access violation.
if __name__ == "__main__":
    simulation_app = SimulationApp({"headless": _args.headless})
else:
    simulation_app = None

# ---------------------------------------------------------------------------
# Now safe to import Omniverse / pxr modules
# ---------------------------------------------------------------------------
import omni.usd                                        # noqa: E402
import omni.timeline                                   # noqa: E402
from omni.isaac.core import World                      # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

# Try to import omni.flow for fire/smoke FX
try:
    import omni.flow as _flow_module                    # noqa: E402
    HAS_FLOW = True
except (ImportError, ModuleNotFoundError):
    HAS_FLOW = False

# Try to import omni.anim for people and navigation
try:
    import omni.anim.people as _anim_people             # noqa: E402
    HAS_ANIM_PEOPLE = True
except (ImportError, ModuleNotFoundError):
    HAS_ANIM_PEOPLE = False

try:
    import omni.anim.navigation as _anim_nav            # noqa: E402
    HAS_ANIM_NAV = True
except (ImportError, ModuleNotFoundError):
    HAS_ANIM_NAV = False

print(f"[ResQ-AI] Extensions: omni.flow={'YES' if HAS_FLOW else 'NO'}  "
      f"omni.anim.people={'YES' if HAS_ANIM_PEOPLE else 'NO'}  "
      f"omni.anim.nav={'YES' if HAS_ANIM_NAV else 'NO'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BuildingInfo:
    """Metadata about a single building discovered on the stage."""
    prim_path: str          # e.g. "/World/Buildings/OfficeA"
    name: str               # e.g. "OfficeA"
    world_x: float          # centre X in world space
    world_y: float          # centre Y in world space
    half_sx: float          # half-width  along X (footprint)
    half_sy: float          # half-width  along Y (footprint)
    height: float           # full building height
    is_burning: bool = False
    fire_prim_path: str = ""
    pedestrian_paths: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Helper utilities (mirroring generate_urban_scene.py patterns)
# ═══════════════════════════════════════════════════════════════════════════════

def _set_xform(xformable: UsdGeom.Xformable,
               translate: Tuple = (0, 0, 0),
               rotate_deg: Tuple = (0, 0, 0),
               scale: Tuple = (1, 1, 1)) -> None:
    """Add translate / rotateXYZ / scale ops to an existing Xformable."""
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_deg))
    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _xform(stage: Usd.Stage, path: str, **kw) -> UsdGeom.Xform:
    xf = UsdGeom.Xform.Define(stage, path)
    _set_xform(xf, **kw)
    return xf


def _cube(stage: Usd.Stage, path: str, size: float = 1.0, **kw):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(size)
    _set_xform(c, **kw)
    return c


def _cone(stage: Usd.Stage, path: str, radius: float = 0.5,
          height: float = 1.0, axis: str = "Z", **kw):
    c = UsdGeom.Cone.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    _set_xform(c, **kw)
    return c


def _sphere(stage: Usd.Stage, path: str, radius: float = 0.5, **kw):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    _set_xform(s, **kw)
    return s


def _cylinder(stage: Usd.Stage, path: str, radius: float = 0.5,
              height: float = 1.0, axis: str = "Z", **kw):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    _set_xform(c, **kw)
    return c


# ── Material helpers ──────────────────────────────────────────────────────────

_mat_cache: Dict[str, str] = {}


def _mat(stage: Usd.Stage, name: str, *,
         albedo: Tuple = (0.5, 0.5, 0.5),
         roughness: float = 0.5,
         metallic: float = 0.0,
         emissive: Tuple = (0.0, 0.0, 0.0),
         emissive_intensity: float = 0.0,
         opacity: float = 1.0) -> str:
    """Create (or re-use) an OmniPBR material and return its prim path."""
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
        "info:implementationSource", Sdf.ValueTypeNames.Token, True
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset, True
    ).Set(mdl_file)
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, True
    ).Set(mdl_name)

    shader.CreateInput("diffuse_color_constant",
                        Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*albedo))
    shader.CreateInput("reflection_roughness_constant",
                        Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant",
                        Sdf.ValueTypeNames.Float).Set(metallic)

    if use_opacity:
        shader.CreateInput("enable_opacity",
                            Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("opacity_constant",
                            Sdf.ValueTypeNames.Float).Set(opacity)

    if emissive_intensity > 0:
        shader.CreateInput("enable_emission",
                            Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color",
                            Sdf.ValueTypeNames.Color3f).Set(
                                Gf.Vec3f(*emissive))
        shader.CreateInput("emissive_intensity",
                            Sdf.ValueTypeNames.Float).Set(emissive_intensity)

    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out")

    _mat_cache[name] = mat_path
    return mat_path


def _bind(stage: Usd.Stage, prim_path: str, mat_path: str) -> None:
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)


# ── Semantic-label helpers ────────────────────────────────────────────────────

def _apply_semantic_label(prim: Usd.Prim, label: str) -> None:
    """Apply a semantic class label using Replicator, with Semantics fallback."""
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


def _prim_has_semantic_label(prim: Usd.Prim, target_label: str) -> bool:
    """Check if *prim* carries a ``SemanticsLabelsAPI:class`` whose token
    array contains *target_label*, OR uses the Replicator-style attribute
    ``semantic:Semantics:params:semanticData``."""
    # Method 1: inline SemanticsLabelsAPI authored in the USDA
    attr = prim.GetAttribute("semantics:labels:class")
    if attr and attr.IsValid():
        val = attr.Get()
        if val and target_label in val:
            return True

    # Method 2: Replicator / Semantics schema attrs
    for attr_name in ["semantic:Semantics:params:semanticData",
                      "semantics:Semantics:params:semanticData"]:
        a = prim.GetAttribute(attr_name)
        if a and a.IsValid():
            v = a.Get()
            if v and target_label.lower() in str(v).lower():
                return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TASK 1 — Discover buildings (target acquisition)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_buildings(stage: Usd.Stage) -> List[BuildingInfo]:
    """Traverse the USD stage and return metadata for every building.

    Strategy:
      1. First try scanning **all** prims for the semantic label "building".
      2. Fall back to inspecting the well-known ``/World/Buildings/*`` path
         (structure authored by ``generate_urban_scene.py``).

    For each building Xform we look for a ``Body`` child cube whose
    ``xformOp:scale`` gives (sx, sy, h) and whose parent Xform's
    ``xformOp:translate`` gives the world (X, Y) position.
    """
    buildings: List[BuildingInfo] = []
    found_paths: set = set()

    # --- Pass 1: semantic-label scan across the whole stage ----------------
    for prim in stage.Traverse():
        if _prim_has_semantic_label(prim, "building"):
            # Walk up to the Xform parent that represents the building root
            candidate = prim
            while candidate and candidate.GetPath().pathString != "/":
                parent = candidate.GetParent()
                if parent and parent.IsA(UsdGeom.Xform):
                    # If the parent is /World/Buildings we found the building
                    if parent.GetPath().pathString.startswith(
                            "/World/Buildings/"):
                        candidate = parent
                        break
                    elif parent.GetPath().pathString == "/World/Buildings":
                        break
                candidate = parent

            path_str = candidate.GetPath().pathString
            if path_str not in found_paths and path_str != "/World/Buildings":
                info = _extract_building_info(stage, candidate)
                if info:
                    buildings.append(info)
                    found_paths.add(path_str)

    # --- Pass 2: prim-path fallback for /World/Buildings/* -----------------
    bldg_root = stage.GetPrimAtPath("/World/Buildings")
    if bldg_root and bldg_root.IsValid():
        for child in bldg_root.GetChildren():
            cp = child.GetPath().pathString
            if cp not in found_paths and child.IsA(UsdGeom.Xformable):
                info = _extract_building_info(stage, child)
                if info:
                    buildings.append(info)
                    found_paths.add(cp)

    print(f"[ResQ-AI] Discovered {len(buildings)} buildings:")
    for b in buildings:
        print(f"          • {b.name:12s}  pos=({b.world_x:6.1f}, "
              f"{b.world_y:6.1f})  footprint="
              f"{b.half_sx*2:.0f}×{b.half_sy*2:.0f}  h={b.height:.0f}m")

    return buildings


def _extract_building_info(stage: Usd.Stage,
                           prim: Usd.Prim) -> Optional[BuildingInfo]:
    """Extract world position and size from a building Xform prim.

    Expects the structure authored by ``generate_urban_scene.py``::

        /World/Buildings/<Name>          <- Xform  (translate = world X, Y)
            /Body                        <- Cube   (scale = sx, sy, height)
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None

    # World-space transform of the building root Xform
    world_mtx = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_pos = world_mtx.ExtractTranslation()

    # Look for the Body child to get dimensions
    body_path = prim.GetPath().AppendChild("Body")
    body_prim = stage.GetPrimAtPath(body_path)

    sx, sy, h = 10.0, 10.0, 15.0  # sensible defaults

    if body_prim and body_prim.IsValid():
        body_xf = UsdGeom.Xformable(body_prim)
        if body_xf:
            # Read xformOp:scale for dimensions
            for op in body_xf.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:scale":
                    sv = op.Get()
                    if sv is not None:
                        sx, sy, h = float(sv[0]), float(sv[1]), float(sv[2])
                        break

    return BuildingInfo(
        prim_path=prim.GetPath().pathString,
        name=prim.GetName(),
        world_x=float(world_pos[0]),
        world_y=float(world_pos[1]),
        half_sx=sx / 2.0,
        half_sy=sy / 2.0,
        height=h,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TASK 1b — Procedural forest generation (PointInstancer)
# ═══════════════════════════════════════════════════════════════════════════════

# Path to a generic tree USD on the Omniverse Nucleus.  Replace with your own
# asset if available.  The instancer prototype is a locally-built tree.
GENERIC_TREE_USD = (
    "omniverse://localhost/NVIDIA/Assets/Vegetation/"
    "Trees/Pinus/NewSpruceTree.usd"
)


def generate_forest(stage: Usd.Stage,
                    buildings: List[BuildingInfo],
                    num_trees: int = 120,
                    seed: int = 42) -> List[Gf.Vec3f]:
    """Place trees within the city using 4 prototype types with natural
    clustering.  Returns a list of tree positions for fire-spread tracking.

    Prototypes: 0=Pine (tall conifer), 1=Oak (broad canopy), 2=Bush (low),
    3=Birch (thin/tall).
    """
    random.seed(seed)
    inner_r, outer_r = 15.0, 65.0
    print(f"[ResQ-AI] Generating {num_trees} trees within city "
          f"(r={inner_r}-{outer_r} m, 4 prototypes) …")

    forest_root = "/World/Forest"
    _xform(stage, forest_root)

    # ── Materials ─────────────────────────────────────────────────────
    bark_mat = _mat(stage, "ForestBarkMat",
                    albedo=(0.28, 0.16, 0.08), roughness=0.92)
    bark_light = _mat(stage, "ForestBarkLightMat",
                      albedo=(0.55, 0.45, 0.35), roughness=0.90)
    pine_mat = _mat(stage, "ForestPineMat",
                    albedo=(0.06, 0.22, 0.06), roughness=0.88)
    oak_mat = _mat(stage, "ForestOakMat",
                   albedo=(0.12, 0.38, 0.10), roughness=0.85)
    bush_mat = _mat(stage, "ForestBushMat",
                    albedo=(0.15, 0.30, 0.08), roughness=0.80)
    birch_mat = _mat(stage, "ForestBirchMat",
                     albedo=(0.20, 0.42, 0.15), roughness=0.82)

    # ── 4 prototype trees ─────────────────────────────────────────────
    proto_root = f"{forest_root}/Prototypes"
    _xform(stage, proto_root)

    # Proto 0: Pine — tall conical conifer
    p_pine = f"{proto_root}/Pine"
    _xform(stage, p_pine)
    _cylinder(stage, f"{p_pine}/Trunk", radius=0.15, height=4.5,
              translate=(0, 0, 2.25))
    _bind(stage, f"{p_pine}/Trunk", bark_mat)
    _cone(stage, f"{p_pine}/Canopy", radius=2.0, height=5.0,
          translate=(0, 0, 6.5))
    _bind(stage, f"{p_pine}/Canopy", pine_mat)

    # Proto 1: Oak — broad spherical canopy
    p_oak = f"{proto_root}/Oak"
    _xform(stage, p_oak)
    _cylinder(stage, f"{p_oak}/Trunk", radius=0.25, height=3.0,
              translate=(0, 0, 1.5))
    _bind(stage, f"{p_oak}/Trunk", bark_mat)
    _sphere(stage, f"{p_oak}/Canopy", radius=3.0,
            translate=(0, 0, 5.0))
    _bind(stage, f"{p_oak}/Canopy", oak_mat)

    # Proto 2: Bush — low dense shrub
    p_bush = f"{proto_root}/Bush"
    _xform(stage, p_bush)
    _sphere(stage, f"{p_bush}/Body", radius=1.2,
            translate=(0, 0, 0.8), scale=(1.3, 1.3, 0.8))
    _bind(stage, f"{p_bush}/Body", bush_mat)

    # Proto 3: Birch — thin tall tree
    p_birch = f"{proto_root}/Birch"
    _xform(stage, p_birch)
    _cylinder(stage, f"{p_birch}/Trunk", radius=0.10, height=5.5,
              translate=(0, 0, 2.75))
    _bind(stage, f"{p_birch}/Trunk", bark_light)
    _sphere(stage, f"{p_birch}/Canopy", radius=1.8,
            translate=(0, 0, 6.5), scale=(1.0, 1.0, 1.3))
    _bind(stage, f"{p_birch}/Canopy", birch_mat)

    # Mark prototypes as non-renderable
    for pp in [p_pine, p_oak, p_bush, p_birch]:
        proto_img = UsdGeom.Imageable(stage.GetPrimAtPath(pp))
        if proto_img:
            proto_img.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    # ── Configure the PointInstancer ──────────────────────────────────
    instancer_path = f"{forest_root}/TreeInstancer"
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    proto_paths = [Sdf.Path(p) for p in [p_pine, p_oak, p_bush, p_birch]]
    instancer.CreatePrototypesRel().SetTargets(proto_paths)

    # Build exclusion zones from buildings (with a 3 m buffer)
    def _inside_any_building(x: float, y: float) -> bool:
        for b in buildings:
            margin = 3.0
            if (abs(x - b.world_x) < b.half_sx + margin and
                    abs(y - b.world_y) < b.half_sy + margin):
                return True
        return False

    # Road corridors: clearings along cardinal axes (simulating streets)
    def _near_road(x: float, y: float) -> bool:
        road_w = 4.0
        return (abs(x) < road_w or abs(y) < road_w or
                abs(x - y) < road_w * 1.5 or abs(x + y) < road_w * 1.5)

    positions: List[Gf.Vec3f] = []
    orientations: List[Gf.Quath] = []
    scales: List[Gf.Vec3f] = []
    proto_indices: List[int] = []
    attempts = 0

    # Generate cluster centres for natural grouping
    num_clusters = num_trees // 8
    cluster_centres = []
    for _ in range(num_clusters):
        ca = random.uniform(0, 2 * math.pi)
        cr = random.uniform(inner_r + 5, outer_r - 5)
        cx, cy = cr * math.cos(ca), cr * math.sin(ca)
        if not _inside_any_building(cx, cy) and not _near_road(cx, cy):
            cluster_centres.append((cx, cy))

    while len(positions) < num_trees and attempts < num_trees * 15:
        attempts += 1

        # 60% clustered, 40% scattered
        if cluster_centres and random.random() < 0.6:
            ccx, ccy = random.choice(cluster_centres)
            x = ccx + random.gauss(0, 4.0)
            y = ccy + random.gauss(0, 4.0)
        else:
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(inner_r, outer_r)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

        if _inside_any_building(x, y):
            continue
        # Reduce density near roads (skip 70% of road candidates)
        if _near_road(x, y) and random.random() < 0.7:
            continue

        # Weighted prototype selection: 35% pine, 25% oak, 25% bush, 15% birch
        r = random.random()
        if r < 0.35:
            pidx = 0   # pine
            s = random.uniform(0.8, 1.3)
        elif r < 0.60:
            pidx = 1   # oak
            s = random.uniform(0.7, 1.2)
        elif r < 0.85:
            pidx = 2   # bush
            s = random.uniform(0.6, 1.5)
        else:
            pidx = 3   # birch
            s = random.uniform(0.9, 1.4)

        rot_z = random.uniform(0, 360)
        quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rot_z).GetQuat()
        qh = Gf.Quath(float(quat.GetReal()),
                       float(quat.GetImaginary()[0]),
                       float(quat.GetImaginary()[1]),
                       float(quat.GetImaginary()[2]))

        positions.append(Gf.Vec3f(x, y, 0.0))
        orientations.append(qh)
        scales.append(Gf.Vec3f(s, s, s))
        proto_indices.append(pidx)

    instancer.CreatePositionsAttr(positions)
    instancer.CreateOrientationsAttr(orientations)
    instancer.CreateScalesAttr(scales)
    instancer.CreateProtoIndicesAttr(proto_indices)

    _label_recursive(stage, forest_root, "vegetation")
    print(f"[ResQ-AI] Placed {len(positions)} trees (4 types) within the city.")

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TASK 2 — Procedural fire emitter setup
# ═══════════════════════════════════════════════════════════════════════════════

# Reference fire USD on the Nucleus (used when omni.flow is unavailable)
FIRE_ASSET_USD = "omniverse://localhost/NVIDIA/Assets/FX/Fire.usd"


def spawn_fire_emitters(stage: Usd.Stage,
                        buildings: List[BuildingInfo],
                        tree_positions: List[Gf.Vec3f],
                        wind_direction: Gf.Vec3f) -> List[Dict]:
    """Spawn TWO types of fire emitters:

    1. Vegetation fires (10-15 at ground level on windward side of forest)
       — start VISIBLE (wildfire origin)
    2. Building fires (one per building on rooftop)
       — start INVISIBLE (activated by spread logic)

    Returns a list of vegetation fire dicts for FireSpreadManager.
    """
    fires_root = "/World/Fires"
    _xform(stage, fires_root)

    fire_preset = _find_fire_preset()
    smoke_preset = _find_smoke_preset()
    if fire_preset:
        print(f"[ResQ-AI] Using fire preset: {os.path.basename(fire_preset)}")
    if smoke_preset:
        print(f"[ResQ-AI] Using smoke preset: {os.path.basename(smoke_preset)}")

    # Pre-create fire materials (emissive cones & spheres fallback)
    flame_outer = _mat(stage, "OrcFlameOuter",
                       albedo=(1.0, 0.3, 0.0),
                       emissive=(1.0, 0.4, 0.0),
                       emissive_intensity=5000.0, roughness=1.0)
    flame_core = _mat(stage, "OrcFlameCore",
                      albedo=(1.0, 0.9, 0.3),
                      emissive=(1.0, 0.9, 0.3),
                      emissive_intensity=8000.0, roughness=1.0)
    smoke_mat = _mat(stage, "OrcSmokeMat",
                     albedo=(0.12, 0.12, 0.12), roughness=1.0, opacity=0.3)

    # ── 1. VEGETATION FIRES ──────────────────────────────────────────────
    veg_fires: List[Dict] = []
    veg_root = f"{fires_root}/VegetationFires"
    _xform(stage, veg_root)

    # Pick 10-15 tree positions on the windward side
    wind_norm = Gf.Vec3f(wind_direction)
    wlen = wind_norm.GetLength()
    if wlen > 1e-6:
        wind_norm /= wlen

    # Score trees by how windward they are (dot product with wind)
    scored_trees = []
    for tp in tree_positions:
        tlen = tp.GetLength()
        if tlen < 1e-6:
            continue
        alignment = Gf.Dot(tp.GetNormalized(), wind_norm)
        scored_trees.append((alignment, tp))
    scored_trees.sort(key=lambda x: -x[0])  # most windward first

    num_veg_fires = min(random.randint(10, 15), len(scored_trees))
    for vi in range(num_veg_fires):
        _, tp = scored_trees[vi]
        vf_path = f"{veg_root}/vegfire_{vi:02d}"
        vf_z = 0.5  # ground level

        _spawn_flow_fire(stage, vf_path, float(tp[0]), float(tp[1]), vf_z,
                         1.5, 1.5, fire_preset, smoke_preset,
                         flame_core, flame_outer, smoke_mat,
                         emitter_radius=3.0, fuel=2.5, smoke_amount=1.2)

        # Vegetation fires start VISIBLE
        _label_recursive(stage, vf_path, "fire")

        veg_fires.append({
            "path": vf_path,
            "pos": Gf.Vec3f(float(tp[0]), float(tp[1]), vf_z),
            "active": vi < 3,  # first 2-3 ignite immediately
        })

        # Initially invisible unless auto-ignited
        fire_prim = stage.GetPrimAtPath(vf_path)
        if fire_prim.IsValid() and vi >= 3:
            UsdGeom.Imageable(fire_prim).MakeInvisible()

    print(f"[ResQ-AI] Spawned {num_veg_fires} vegetation fires "
          f"({sum(1 for v in veg_fires if v['active'])} initially burning).")

    # ── 2. BUILDING FIRES ────────────────────────────────────────────────
    for bldg in buildings:
        fire_path = f"{fires_root}/{bldg.name}_fire"
        fire_z = bldg.height + 2.0

        _spawn_flow_fire(stage, fire_path,
                         bldg.world_x, bldg.world_y, fire_z,
                         bldg.half_sx, bldg.half_sy,
                         fire_preset, smoke_preset,
                         flame_core, flame_outer, smoke_mat)

        # Make the fire invisible at start
        fire_prim = stage.GetPrimAtPath(fire_path)
        if fire_prim.IsValid():
            UsdGeom.Imageable(fire_prim).MakeInvisible()

        bldg.fire_prim_path = fire_path
        _label_recursive(stage, fire_path, "fire")

    print(f"[ResQ-AI] Spawned {len(buildings)} building fire emitters "
          f"(all invisible).")

    return veg_fires


def _spawn_flow_fire(stage: Usd.Stage, path: str,
                     wx: float, wy: float, wz: float,
                     half_sx: float, half_sy: float,
                     fire_preset: Optional[str] = None,
                     smoke_preset: Optional[str] = None,
                     flame_core_mat: str = "",
                     flame_outer_mat: str = "",
                     smoke_mat: str = "",
                     emitter_radius: Optional[float] = None,
                     fuel: float = 1.8,
                     smoke_amount: float = 0.7) -> None:
    """Create a fire emitter via preset USD reference, manual Flow, or
    emissive geometry fallback.  Includes a SphereLight for glow."""
    _xform(stage, path, translate=(wx, wy, wz))

    # ── SphereLight for fire glow ─────────────────────────────────────
    fire_light = UsdLux.SphereLight.Define(stage, f"{path}/FireLight")
    fire_light.CreateRadiusAttr(2.0)
    fire_light.CreateIntensityAttr(15000.0)
    fire_light.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.05))
    _set_xform(fire_light, translate=(0, 0, 4.0))

    em_radius = emitter_radius or min(half_sx, half_sy) * 0.4

    # Strategy 1: Reference fire preset USD
    if fire_preset and os.path.isfile(fire_preset):
        fire_ref_path = f"{path}/FirePreset"
        fire_ref_prim = stage.DefinePrim(fire_ref_path)
        fire_ref_prim.GetReferences().AddReference(fire_preset)
        if smoke_preset and os.path.isfile(smoke_preset):
            smoke_ref_path = f"{path}/SmokePreset"
            smoke_ref_prim = stage.DefinePrim(smoke_ref_path)
            smoke_ref_prim.GetReferences().AddReference(smoke_preset)
        return

    # Strategy 2: Manual omni.flow hierarchy
    if HAS_FLOW:
        sim_path = f"{path}/flowSimulate"
        sim_prim = stage.DefinePrim(sim_path, "FlowSimulate")
        sim_prim.CreateAttribute("buoyancyPerTemp",
                                 Sdf.ValueTypeNames.Float).Set(4.0)
        sim_prim.CreateAttribute("burnPerTemp",
                                 Sdf.ValueTypeNames.Float).Set(0.15)
        sim_prim.CreateAttribute("coolingRate",
                                 Sdf.ValueTypeNames.Float).Set(2.5)
        sim_prim.CreateAttribute("gravity",
                                 Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(0.0, 0.0, -3.0))

        emitter_path = f"{path}/flowEmitter"
        emitter_prim = stage.DefinePrim(emitter_path, "FlowEmitterSphere")
        emitter_prim.CreateAttribute("radius",
                                     Sdf.ValueTypeNames.Float).Set(
            float(em_radius))
        emitter_prim.CreateAttribute("fuel",
                                     Sdf.ValueTypeNames.Float).Set(fuel)
        emitter_prim.CreateAttribute("temperature",
                                     Sdf.ValueTypeNames.Float).Set(1500.0)
        emitter_prim.CreateAttribute("smoke",
                                     Sdf.ValueTypeNames.Float).Set(
            smoke_amount)
        emitter_prim.CreateAttribute("velocity",
                                     Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(0.0, 0.0, 2.5))

        render_path = f"{path}/flowRender"
        render_prim = stage.DefinePrim(render_path, "FlowRender")
        render_prim.CreateAttribute("colorMapResolution",
                                    Sdf.ValueTypeNames.Int).Set(64)
        render_prim.CreateAttribute("shadowFactor",
                                    Sdf.ValueTypeNames.Float).Set(1.0)
        return

    # Strategy 3: Emissive geometry fallback
    _spawn_geometry_fire(stage, path, 0, 0, 0,
                         flame_core_mat, flame_outer_mat, smoke_mat)


def _spawn_geometry_fire(stage: Usd.Stage, path: str,
                         wx: float, wy: float, wz: float,
                         core_mat: str, outer_mat: str,
                         smoke_mat: str) -> None:
    """Create a LARGE cluster of emissive cones and spheres simulating fire.

    Scaled up significantly so it's clearly visible from the drone's altitude.
    Includes a point light for realistic illumination of surroundings.
    """
    _xform(stage, path, translate=(wx, wy, wz))

    # === Point light to illuminate surroundings with fire glow ===
    fire_light = UsdLux.SphereLight.Define(stage, f"{path}/FireLight")
    fire_light.CreateRadiusAttr(2.0)
    fire_light.CreateIntensityAttr(15000.0)
    fire_light.CreateColorAttr(Gf.Vec3f(1.0, 0.45, 0.05))
    _set_xform(fire_light, translate=(0, 0, 4.0))

    # === Core flame (large central cone) ===
    _cone(stage, f"{path}/Core", radius=1.5, height=8.0,
          translate=(0, 0, 4.0))
    _bind(stage, f"{path}/Core", core_mat)

    # === Surrounding flame cones ===
    offsets = [
        (-1.5,  0.8, 3.0, 1.0, 6.0,  15),
        ( 1.3, -0.6, 3.0, 0.9, 5.5, -12),
        ( 0.4,  1.4, 2.5, 0.8, 5.0,   8),
        (-0.8, -1.2, 2.5, 0.7, 4.5, -18),
        ( 1.8,  0.3, 2.0, 0.6, 4.0,  22),
        (-1.0,  1.6, 2.0, 0.5, 3.5, -10),
    ]
    for i, (dx, dy, dz, r, h, rot) in enumerate(offsets):
        mat = core_mat if i % 2 == 0 else outer_mat
        _cone(stage, f"{path}/Flame_{i}", radius=r, height=h,
              translate=(dx, dy, dz), rotate_deg=(0, 0, rot))
        _bind(stage, f"{path}/Flame_{i}", mat)

    # === Hot ember spheres ===
    for i in range(6):
        ang = i * 60
        r = 0.4 + random.uniform(0, 0.3)
        ex = 1.5 * math.cos(math.radians(ang))
        ey = 1.5 * math.sin(math.radians(ang))
        _sphere(stage, f"{path}/Ember_{i}", radius=r,
                translate=(ex, ey, 6.0 + i * 0.5))
        _bind(stage, f"{path}/Ember_{i}", core_mat)

    # === Glow disc at base ===
    _sphere(stage, f"{path}/Glow", radius=2.5,
            translate=(0, 0, 0.3), scale=(3.0, 3.0, 0.5))
    _bind(stage, f"{path}/Glow", core_mat)

    # === Smoke column ===
    for i in range(8):
        r = 0.8 + i * 0.5
        _sphere(stage, f"{path}/Smoke_{i}", radius=r,
                translate=(i * 0.3 - 0.5, i * 0.2, 7.0 + i * 2.0))
        _bind(stage, f"{path}/Smoke_{i}", smoke_mat)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TASK 3 — Indoor pedestrian spawning
# ═══════════════════════════════════════════════════════════════════════════════

# ── Character USD assets (local paths from Rigged Characters Asset Pack) ──
_CHARACTER_USDS = [
    os.path.join(_SCRIPT_DIR, "assets", "Characters", "Assets", "Characters",
                 "Reallusion", "Worker", "Worker.usd"),
    os.path.join(_SCRIPT_DIR, "assets", "Characters", "Assets", "Characters",
                 "Reallusion", "ActorCore", "Business_F_0002", "Actor",
                 "business-f-0002", "business-f-0002.usd"),
    os.path.join(_SCRIPT_DIR, "assets", "Characters", "Assets", "Characters",
                 "Reallusion", "ActorCore", "Uniform_F_0001", "Actor",
                 "uniform_f_0001", "uniform_f_0001.usd"),
    os.path.join(_SCRIPT_DIR, "assets", "Characters", "Assets", "Characters",
                 "Reallusion", "ActorCore", "Uniform_M_0001", "Actor",
                 "uniform_m_0001", "uniform_m_0001.usd"),
    os.path.join(_SCRIPT_DIR, "assets", "Characters", "Assets", "Characters",
                 "Reallusion", "ActorCore", "Party_M_0001", "Actor",
                 "party-m-0001", "party-m-0001.usd"),
]

# ── Fire Flow preset paths (from Extensions Sample Asset Pack) ──
_FLOW_SAMPLES_DIR = os.path.join(
    _SCRIPT_DIR, "assets", "Particles", "Assets", "Extensions",
    "Samples", "Flow", "samples")


def _find_fire_preset() -> Optional[str]:
    """Find a fire preset USD in the Particles pack Flow samples."""
    if not os.path.isdir(_FLOW_SAMPLES_DIR):
        return None
    # Prefer Fire.usda, then WispyFire.usda
    for candidate in ["Fire.usda", "WispyFire.usda"]:
        path = os.path.join(_FLOW_SAMPLES_DIR, candidate)
        if os.path.isfile(path):
            return path
    for fname in os.listdir(_FLOW_SAMPLES_DIR):
        if "fire" in fname.lower() and fname.endswith((".usd", ".usdc", ".usda")):
            return os.path.join(_FLOW_SAMPLES_DIR, fname)
    return None


def _find_smoke_preset() -> Optional[str]:
    """Find a smoke preset USD in the Particles pack Flow samples."""
    if not os.path.isdir(_FLOW_SAMPLES_DIR):
        return None
    for candidate in ["DenseSmoke.usda", "DarkSmoke.usda"]:
        path = os.path.join(_FLOW_SAMPLES_DIR, candidate)
        if os.path.isfile(path):
            return path
    for fname in os.listdir(_FLOW_SAMPLES_DIR):
        if "smoke" in fname.lower() and fname.endswith((".usd", ".usdc", ".usda")):
            return os.path.join(_FLOW_SAMPLES_DIR, fname)
    return None


def spawn_indoor_pedestrians(stage: Usd.Stage,
                             buildings: List[BuildingInfo],
                             count: int = 20) -> None:
    """Spawn *count* pedestrian agents inside the discovered buildings.

    Pedestrians are distributed across buildings so that every building gets
    at least one victim (if there are enough).  Spawn coordinates are the
    building's world X/Y centre with random offsets kept inside the footprint,
    and a Z height corresponding to different floors.

    If ``omni.anim.people`` is available the agents are proper animated
    characters.  Otherwise we place simple geometric stand-ins (cylinder
    torso + sphere head) following the same approach as
    ``generate_urban_scene.py`` and label them ``"person"``.

    Indoor navigation waypoints are generated for each pedestrian so they
    wander within the building bounds.  When ``omni.anim.navigation`` is
    available these feed into the navmesh system; otherwise they are stored as
    custom USD attributes for inspection.
    """
    count = max(15, min(30, count))
    ped_root = "/World/IndoorVictims"
    _xform(stage, ped_root)

    # Ensure at least one ped per building, distribute the rest randomly
    assignments: List[int] = []  # building index for each pedestrian
    num_bldgs = len(buildings)
    if num_bldgs == 0:
        print("[ResQ-AI] WARNING: No buildings found — cannot spawn "
              "indoor pedestrians.")
        return

    # Round-robin first pass (one per building)
    for i in range(min(count, num_bldgs)):
        assignments.append(i)
    # Distribute remaining randomly
    for _ in range(count - len(assignments)):
        assignments.append(random.randint(0, num_bldgs - 1))
    random.shuffle(assignments)

    # Materials for geometric fallback
    skin_mat = _mat(stage, "VictimSkinMat",
                    albedo=(0.72, 0.55, 0.45), roughness=0.85)
    shirt_mats = [
        _mat(stage, "VictimShirtA",
             albedo=(0.1, 0.15, 0.35), roughness=0.75),
        _mat(stage, "VictimShirtB",
             albedo=(0.4, 0.1, 0.1), roughness=0.75),
        _mat(stage, "VictimShirtC",
             albedo=(0.1, 0.35, 0.35), roughness=0.75),
        _mat(stage, "VictimShirtD",
             albedo=(0.25, 0.3, 0.15), roughness=0.75),
    ]
    pants_mat = _mat(stage, "VictimPantsMat",
                     albedo=(0.15, 0.15, 0.18), roughness=0.8)

    floor_height = 4.0  # metres per floor (matches building window spacing)

    for idx, bldg_idx in enumerate(assignments):
        bldg = buildings[bldg_idx]
        ped_name = f"Victim_{idx:02d}"
        ped_path = f"{ped_root}/{ped_name}"

        # ── Compute indoor spawn position ─────────────────────────────────
        offset_x = random.uniform(-bldg.half_sx * 0.7, bldg.half_sx * 0.7)
        offset_y = random.uniform(-bldg.half_sy * 0.7, bldg.half_sy * 0.7)
        spawn_x = bldg.world_x + offset_x
        spawn_y = bldg.world_y + offset_y

        # Pick a random floor (Z height)
        max_floors = max(1, int(bldg.height / floor_height))
        floor = random.randint(0, max_floors - 1)
        spawn_z = floor * floor_height + 0.05

        if HAS_ANIM_PEOPLE:
            _spawn_anim_person(stage, ped_path, spawn_x, spawn_y, spawn_z,
                               idx, skin_mat,
                               shirt_mats[idx % len(shirt_mats)], pants_mat)
        else:
            _spawn_geometry_person(stage, ped_path,
                                   spawn_x, spawn_y, spawn_z,
                                   skin_mat,
                                   shirt_mats[idx % len(shirt_mats)],
                                   pants_mat)

        # ── Indoor navigation waypoints (wander loop) ─────────────────────
        waypoints = _generate_indoor_waypoints(bldg, spawn_z, num_points=4)
        _assign_nav_waypoints(stage, ped_path, waypoints, bldg)

        # Register this pedestrian on the building
        bldg.pedestrian_paths.append(ped_path)

    total = len(assignments)
    mode_label = 'omni.anim.people' if HAS_ANIM_PEOPLE else 'geometry (static)'
    print(f"[ResQ-AI] Spawned {total} indoor victims across "
          f"{num_bldgs} buildings ({mode_label}).")
    if not HAS_ANIM_PEOPLE:
        print("[ResQ-AI] NOTE: Pedestrians are static geometric stand-ins.")
        print("          To enable walking/fleeing animations, ensure the ")
        print("          'omni.anim.people' extension is enabled in Isaac Sim.")
        print("          (Window → Extensions → search 'anim people' → Enable)")


def _spawn_anim_person(stage: Usd.Stage, path: str,
                       x: float, y: float, z: float, idx: int,
                       skin_mat: str, shirt_mat: str,
                       pants_mat: str) -> None:
    """Reference a character USD at the given position.

    Validates the file exists first. If missing, falls back to geometry.
    If omni.anim.people + omni.anim.navigation are available, attempts to
    bake a navmesh and register the character for idle/patrol behaviour.
    """
    char_usd = _CHARACTER_USDS[idx % len(_CHARACTER_USDS)]

    # ── Validate file exists ──────────────────────────────────────────
    if not os.path.isfile(char_usd):
        print(f"[ResQ-AI] WARNING: Character file not found: {char_usd}")
        print(f"          Falling back to geometry person for {path}")
        _spawn_geometry_person(stage, path, x, y, z,
                               skin_mat, shirt_mat, pants_mat)
        return

    print(f"[ResQ-AI] Loading character: {os.path.basename(char_usd)} → {path}")
    xf = _xform(stage, path, translate=(x, y, z))

    # Add as a USD reference
    prim = stage.GetPrimAtPath(path)
    prim.GetReferences().AddReference(char_usd)
    _apply_semantic_label(prim, "person")

    # ── Try to configure omni.anim.people behaviour ───────────────────
    if HAS_ANIM_PEOPLE and HAS_ANIM_NAV:
        try:
            # Bake a navmesh on the ground plane for navigation
            nav_mgr = _anim_nav.get_navigation_manager()
            if nav_mgr is not None:
                nav_mgr.bake_navmesh()

            # Register with the people extension
            people_mgr = _anim_people.get_people_manager()
            if people_mgr is not None:
                people_mgr.register_character(path)
                # Set initial behaviour to idle
                people_mgr.set_character_behavior(path, "idle")
                print(f"[ResQ-AI]   Registered {path} with anim.people (idle)")
        except Exception as exc:
            print(f"[ResQ-AI]   WARNING: anim.people setup failed for "
                  f"{path}: {exc}")


def _spawn_geometry_person(stage: Usd.Stage, path: str,
                           x: float, y: float, z: float,
                           skin_mat: str, shirt_mat: str,
                           pants_mat: str) -> None:
    """Build a geometric person with improved proportions (~1.75 m tall).

    Creates a full-body silhouette visible from drone altitude:
    head (r=0.12), neck, torso (r=0.22), hips, arms, and legs.
    """
    _xform(stage, path, translate=(x, y, z))

    # Torso: radius 0.22, height 0.50, centred at z=1.15
    _cylinder(stage, f"{path}/Torso", radius=0.22, height=0.50,
              translate=(0, 0, 1.15))
    _bind(stage, f"{path}/Torso", shirt_mat)

    # Head: radius 0.12, at z=1.63
    _sphere(stage, f"{path}/Head", radius=0.12, translate=(0, 0, 1.63))
    _bind(stage, f"{path}/Head", skin_mat)

    # Neck: small cylinder connecting head and torso
    _cylinder(stage, f"{path}/Neck", radius=0.05, height=0.10,
              translate=(0, 0, 1.46))
    _bind(stage, f"{path}/Neck", skin_mat)

    # Hips: radius 0.20, height 0.25, at z=0.78
    _cylinder(stage, f"{path}/Hips", radius=0.20, height=0.25,
              translate=(0, 0, 0.78))
    _bind(stage, f"{path}/Hips", pants_mat)

    # Legs: radius 0.08, height 0.55, at z=0.33
    for li, ly in enumerate([-0.10, 0.10]):
        _cylinder(stage, f"{path}/Leg_{li}", radius=0.08, height=0.55,
                  translate=(0, ly, 0.33))
        _bind(stage, f"{path}/Leg_{li}", pants_mat)

    # Feet: small flattened spheres
    for fi, fy in enumerate([-0.10, 0.10]):
        _sphere(stage, f"{path}/Foot_{fi}", radius=0.06,
                translate=(0.04, fy, 0.06), scale=(1.5, 1.0, 0.6))
        _bind(stage, f"{path}/Foot_{fi}", pants_mat)

    # Arms: radius 0.06, height 0.55, at z=1.05
    for ai, ay in enumerate([-0.30, 0.30]):
        _cylinder(stage, f"{path}/Arm_{ai}", radius=0.06, height=0.55,
                  translate=(0, ay, 1.05))
        _bind(stage, f"{path}/Arm_{ai}", shirt_mat)
        # Hands
        _sphere(stage, f"{path}/Hand_{ai}", radius=0.055,
                translate=(0, ay, 0.74))
        _bind(stage, f"{path}/Hand_{ai}", skin_mat)

    _label_recursive(stage, path, "person")


def _generate_indoor_waypoints(bldg: BuildingInfo,
                               floor_z: float,
                               num_points: int = 4) -> List[Gf.Vec3d]:
    """Generate a tight rectangular wander loop inside a building floor."""
    margin = 1.5  # stay away from walls
    cx, cy = bldg.world_x, bldg.world_y
    hx = max(margin, bldg.half_sx - margin)
    hy = max(margin, bldg.half_sy - margin)

    # Rectangular patrol: four corners with some jitter
    corners = [
        (cx - hx + random.uniform(0, 1), cy - hy + random.uniform(0, 1)),
        (cx + hx - random.uniform(0, 1), cy - hy + random.uniform(0, 1)),
        (cx + hx - random.uniform(0, 1), cy + hy - random.uniform(0, 1)),
        (cx - hx + random.uniform(0, 1), cy + hy - random.uniform(0, 1)),
    ]
    return [Gf.Vec3d(px, py, floor_z) for px, py in corners[:num_points]]


def _assign_nav_waypoints(stage: Usd.Stage, ped_path: str,
                          waypoints: List[Gf.Vec3d],
                          bldg: BuildingInfo) -> None:
    """Attach navigation waypoints to a pedestrian prim.

    If ``omni.anim.navigation`` is available, configure a navmesh command
    sequence.  Otherwise store waypoints as custom USD attributes so they can
    be consumed by downstream logic.
    """
    prim = stage.GetPrimAtPath(ped_path)
    if not prim.IsValid():
        return

    if HAS_ANIM_NAV:
        # omni.anim.navigation: set navigation target commands
        try:
            # Store as custom array attribute; the navigation extension
            # picks up waypoint targets from character behaviour scripts.
            wp_attr = prim.CreateAttribute(
                "resqai:nav:waypoints",
                Sdf.ValueTypeNames.Point3dArray)
            wp_attr.Set(waypoints)

            prim.CreateAttribute(
                "resqai:nav:loop", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute(
                "resqai:nav:buildingPath",
                Sdf.ValueTypeNames.String).Set(bldg.prim_path)
        except Exception as exc:
            print(f"[ResQ-AI] WARNING: Failed to set nav waypoints for "
                  f"{ped_path}: {exc}")
    else:
        # Fallback: store waypoints as custom attributes for manual lookup
        wp_attr = prim.CreateAttribute(
            "resqai:nav:waypoints", Sdf.ValueTypeNames.Point3dArray)
        wp_attr.Set(waypoints)
        prim.CreateAttribute(
            "resqai:nav:loop", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute(
            "resqai:nav:buildingPath",
            Sdf.ValueTypeNames.String).Set(bldg.prim_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TASK 4 — Fire spread logic (physics-step callback)
# ═══════════════════════════════════════════════════════════════════════════════

class FireSpreadManager:
    """Manages wind-driven fire spread across vegetation and buildings,
    with lerp-based pedestrian flee movement.

    Fire spreads via three paths:
      1. Vegetation → vegetation (through forest ring, wind-biased)
      2. Vegetation → building   (within 15m downwind, probability ramp)
      3. Building  → building    (distance + wind bias: 2x downwind, 0.3x upwind)
    """

    CHECK_INTERVAL: int = 60

    def __init__(self, stage: Usd.Stage, buildings: List[BuildingInfo],
                 spread_radius: float = 40.0,
                 wind_direction: Gf.Vec3f = Gf.Vec3f(1, 0, 0),
                 wind_speed: float = 5.0,
                 vegetation_fires: Optional[List[Dict]] = None,
                 tree_positions: Optional[List[Gf.Vec3f]] = None) -> None:
        self.stage = stage
        self.buildings = buildings
        self.spread_radius = spread_radius
        self.step_count = 0

        # Wind parameters
        self.wind_direction = Gf.Vec3f(wind_direction)
        wlen = self.wind_direction.GetLength()
        if wlen > 1e-6:
            self.wind_direction /= wlen
        self.wind_speed = wind_speed

        # Vegetation fire tracking
        self.vegetation_fires = vegetation_fires or []
        self.tree_positions = tree_positions or []

        # Lerp-based flee state: {ped_path: {start, end, step, total_steps}}
        self._flee_state: Dict[str, Dict] = {}

        # Pre-compute building centre positions
        self._positions = {
            b.name: (b.world_x, b.world_y) for b in buildings
        }

        print(f"[ResQ-AI] FireSpreadManager: spread_radius={spread_radius:.0f}m  "
              f"wind=({wind_direction[0]:.1f}, {wind_direction[1]:.1f}, "
              f"{wind_direction[2]:.1f})  speed={wind_speed:.1f} m/s")
        print(f"[ResQ-AI]   {len(self.vegetation_fires)} vegetation fire slots, "
              f"{len(self.tree_positions)} tree positions tracked")

    # ── Public API ────────────────────────────────────────────────────────

    def ignite_vegetation_fires(self) -> None:
        """Ignite the first 2-3 vegetation fires to start the wildfire."""
        count = 0
        for vf in self.vegetation_fires:
            if vf["active"]:
                count += 1
                fp = self.stage.GetPrimAtPath(vf["path"])
                if fp and fp.IsValid():
                    UsdGeom.Imageable(fp).MakeVisible()
        print(f"[ResQ-AI] 🔥 Ignited {count} initial vegetation fires.")

    def ignite_random_building(self) -> None:
        """Ignite one random building to kick off the simulation."""
        if not self.buildings:
            return
        victim = random.choice(self.buildings)
        self._ignite(victim)
        print(f"[ResQ-AI] 🔥 Initial ignition: {victim.name}")

    def on_physics_step(self, dt: float) -> None:
        """Called every physics step."""
        self.step_count += 1

        # Advance flee lerps every step
        self._advance_flee_lerps()

        if self.step_count % self.CHECK_INTERVAL != 0:
            return
        self._check_spread()

    def print_status(self) -> None:
        """Print current fire status with wind info."""
        burning = sum(1 for b in self.buildings if b.is_burning)
        active_veg = sum(1 for v in self.vegetation_fires if v["active"])
        print(f"[ResQ-AI] step={self.step_count}  "
              f"burning_buildings={burning}/{len(self.buildings)}  "
              f"active_veg_fires={active_veg}/{len(self.vegetation_fires)}  "
              f"wind=({self.wind_direction[0]:.1f}, "
              f"{self.wind_direction[1]:.1f})")

    # ── Internals ─────────────────────────────────────────────────────────

    def _ignite(self, bldg: BuildingInfo) -> None:
        """Set a building on fire: flip state, make fire prim visible,
        and trigger flee for its indoor occupants."""
        if bldg.is_burning:
            return
        bldg.is_burning = True

        fire_prim = self.stage.GetPrimAtPath(bldg.fire_prim_path)
        if fire_prim and fire_prim.IsValid():
            UsdGeom.Imageable(fire_prim).MakeVisible()

        self._trigger_flee(bldg)

    def _wind_factor(self, source_pos: Tuple[float, float],
                     target_pos: Tuple[float, float]) -> float:
        """Compute wind alignment factor (0..1) for spread direction."""
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return 0.5
        nx, ny = dx / length, dy / length
        alignment = nx * float(self.wind_direction[0]) + \
                    ny * float(self.wind_direction[1])
        return 0.5 + 0.5 * alignment  # map from [-1,1] to [0,1]

    def _check_spread(self) -> None:
        """Wind-driven fire spread across three paths."""
        time_factor = min(1.0, 0.01 + self.step_count / 10000.0)

        # ── A. Vegetation → Vegetation spread ──────────────────────────
        active_veg = [v for v in self.vegetation_fires if v["active"]]
        inactive_veg = [v for v in self.vegetation_fires if not v["active"]]

        for dst_vf in inactive_veg:
            dst_pos = (float(dst_vf["pos"][0]), float(dst_vf["pos"][1]))
            for src_vf in active_veg:
                src_pos = (float(src_vf["pos"][0]), float(src_vf["pos"][1]))
                dist = math.sqrt((src_pos[0] - dst_pos[0]) ** 2 +
                                 (src_pos[1] - dst_pos[1]) ** 2)
                wf = self._wind_factor(src_pos, dst_pos)
                proximity = (self.spread_radius / max(dist, 1.0)) ** 2
                p = min(0.8, time_factor * proximity * (0.3 + 1.7 * wf))
                if random.random() < p:
                    dst_vf["active"] = True
                    fp = self.stage.GetPrimAtPath(dst_vf["path"])
                    if fp and fp.IsValid():
                        UsdGeom.Imageable(fp).MakeVisible()
                    print(f"[ResQ-AI] 🔥 Vegetation fire spread → "
                          f"{dst_vf['path']} (d={dist:.0f}m, p={p:.2f})")
                    break

        # ── B. Vegetation → Building spread ────────────────────────────
        for bldg in self.buildings:
            if bldg.is_burning:
                continue
            bpos = (bldg.world_x, bldg.world_y)
            for src_vf in active_veg:
                spos = (float(src_vf["pos"][0]), float(src_vf["pos"][1]))
                dist = math.sqrt((spos[0] - bpos[0]) ** 2 +
                                 (spos[1] - bpos[1]) ** 2)
                if dist > 15.0:
                    continue  # too far for veg→building
                wf = self._wind_factor(spos, bpos)
                proximity = (15.0 / max(dist, 1.0)) ** 2
                p = min(0.8, time_factor * proximity * (0.3 + 1.7 * wf))
                if random.random() < p:
                    self._ignite(bldg)
                    print(f"[ResQ-AI] 🔥 Vegetation→Building: → {bldg.name} "
                          f"(d={dist:.1f}m, wf={wf:.2f})")
                    break

        # ── C. Building → Building spread ──────────────────────────────
        burning = [b for b in self.buildings if b.is_burning]
        not_burning = [b for b in self.buildings if not b.is_burning]

        for dst in not_burning:
            dst_pos = self._positions[dst.name]
            min_dist = float('inf')
            nearest_src = None
            for src in burning:
                src_pos = self._positions[src.name]
                dist = math.sqrt((src_pos[0] - dst_pos[0]) ** 2 +
                                 (src_pos[1] - dst_pos[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_src = src

            if nearest_src is None:
                continue

            src_pos = self._positions[nearest_src.name]
            wf = self._wind_factor(src_pos, dst_pos)
            proximity = (self.spread_radius / max(min_dist, 1.0)) ** 2
            p_spread = min(0.8, time_factor * proximity * (0.3 + 1.7 * wf))

            if random.random() < p_spread:
                self._ignite(dst)
                print(f"[ResQ-AI] 🔥 Building spread: {nearest_src.name} → "
                      f"{dst.name} (d={min_dist:.1f}m, wf={wf:.2f}, "
                      f"p={p_spread:.2f})")

    def _trigger_flee(self, bldg: BuildingInfo) -> None:
        """Initiate flee for all pedestrians in a burning building.

        If HAS_ANIM_PEOPLE: set navigation target.
        Otherwise: set up lerp state for smooth movement over 120 steps.
        """
        if not bldg.pedestrian_paths:
            return

        for idx, ped_path in enumerate(bldg.pedestrian_paths):
            ped_prim = self.stage.GetPrimAtPath(ped_path)
            if not ped_prim or not ped_prim.IsValid():
                continue

            # Compute flee destination
            dx = -bldg.world_x
            dy = -bldg.world_y
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                dx, dy = 1.0, 0.0
            else:
                dx /= length
                dy /= length

            flee_dist = bldg.half_sx + 3.0 + idx * 2.5 + random.uniform(0, 2)
            lateral = (idx - len(bldg.pedestrian_paths) / 2.0) * 2.0
            flee_x = bldg.world_x + dx * flee_dist + (-dy) * lateral
            flee_y = bldg.world_y + dy * flee_dist + dx * lateral
            flee_z = 0.05

            if HAS_ANIM_PEOPLE:
                # Set navigation target for animated characters
                try:
                    people_mgr = _anim_people.get_people_manager()
                    if people_mgr is not None:
                        people_mgr.set_navigation_target(
                            ped_path, (flee_x, flee_y, flee_z))
                except Exception:
                    pass
                # Also do an immediate move as fallback
                self._set_prim_translate(ped_prim, flee_x, flee_y, flee_z)
            else:
                # Get current position for lerp start
                xformable = UsdGeom.Xformable(ped_prim)
                start_pos = (bldg.world_x, bldg.world_y, 0.05)
                if xformable:
                    for op in xformable.GetOrderedXformOps():
                        if op.GetOpName() == "xformOp:translate":
                            v = op.Get()
                            if v:
                                start_pos = (float(v[0]), float(v[1]),
                                             float(v[2]))
                            break

                self._flee_state[ped_path] = {
                    "start": start_pos,
                    "end": (flee_x, flee_y, flee_z),
                    "step": 0,
                    "total": 120,
                }

            ped_prim.CreateAttribute(
                "resqai:state", Sdf.ValueTypeNames.String).Set("fleeing")

        print(f"[ResQ-AI] 🏃 {len(bldg.pedestrian_paths)} victim(s) in "
              f"{bldg.name} fleeing (visible to drone).")

    def _advance_flee_lerps(self) -> None:
        """Advance all active flee lerps by one step."""
        done_keys = []
        for ped_path, state in self._flee_state.items():
            state["step"] += 1
            t = min(1.0, state["step"] / state["total"])
            # Smooth ease-out
            t = 1.0 - (1.0 - t) ** 2

            sx, sy, sz = state["start"]
            ex, ey, ez = state["end"]
            cx = sx + (ex - sx) * t
            cy = sy + (ey - sy) * t
            cz = sz + (ez - sz) * t

            prim = self.stage.GetPrimAtPath(ped_path)
            if prim and prim.IsValid():
                self._set_prim_translate(prim, cx, cy, cz)

            if state["step"] >= state["total"]:
                done_keys.append(ped_path)

        for k in done_keys:
            del self._flee_state[k]

    @staticmethod
    def _set_prim_translate(prim: Any, x: float, y: float, z: float) -> None:
        """Set or update the xformOp:translate on a prim."""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3d(x, y, z))
                return
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))


# ═══════════════════════════════════════════════════════════════════════════════
# 7a. Character validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_characters(stage: Usd.Stage,
                        buildings: List[BuildingInfo]) -> None:
    """Check that character references actually loaded geometry.

    For each pedestrian, verify the prim exists and has child meshes.
    If a reference failed (no children), clear it and create a geometry
    fallback at the same position.
    """
    print("[ResQ-AI] ── Validating character references ──")

    # Scan character USD paths first
    for i, char_path in enumerate(_CHARACTER_USDS):
        exists = os.path.isfile(char_path)
        basename = os.path.basename(char_path)
        dirname = os.path.dirname(char_path)
        status = "✅ found" if exists else "❌ NOT FOUND"
        print(f"  [{i}] {basename}: {status}")
        if not exists:
            # Try to find any .usd/.usdc in the parent directory
            if os.path.isdir(dirname):
                found = [f for f in os.listdir(dirname)
                         if f.endswith((".usd", ".usdc", ".usda"))]
                if found:
                    print(f"       Available files in {dirname}: {found}")

    # Materials for geometry fallback
    skin_mat = _mat(stage, "VictimSkinMat",
                    albedo=(0.72, 0.55, 0.45), roughness=0.85)
    shirt_mat = _mat(stage, "VictimShirtA",
                     albedo=(0.1, 0.15, 0.35), roughness=0.75)
    pants_mat = _mat(stage, "VictimPantsMat",
                     albedo=(0.15, 0.15, 0.18), roughness=0.8)

    ok_count = 0
    fixed_count = 0
    for bldg in buildings:
        for ped_path in bldg.pedestrian_paths:
            prim = stage.GetPrimAtPath(ped_path)
            if not prim.IsValid():
                print(f"  WARNING: {ped_path} is not valid")
                continue
            children = list(prim.GetChildren())
            if len(children) == 0:
                print(f"  WARNING: {ped_path} has no children — "
                      f"reference may have failed. Falling back to geometry.")
                prim.GetReferences().ClearReferences()
                # Read current position
                xformable = UsdGeom.Xformable(prim)
                x, y, z = 0, 0, 0
                if xformable:
                    mtx = xformable.ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
                    pos = mtx.ExtractTranslation()
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                _spawn_geometry_person(stage, ped_path, x, y, z,
                                       skin_mat, shirt_mat, pants_mat)
                fixed_count += 1
            else:
                print(f"  OK: {ped_path} loaded with "
                      f"{len(children)} child prims")
                ok_count += 1

    print(f"[ResQ-AI] Character validation: {ok_count} OK, "
          f"{fixed_count} fixed with geometry fallback.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7b.  Add visible windows to buildings
# ═══════════════════════════════════════════════════════════════════════════════


def add_building_windows(stage: Usd.Stage,
                         buildings: List[BuildingInfo]) -> None:
    """Add dark window bands to all four faces of each building.

    The base USDA building Body is a solid cube with no visible windows.
    This function adds darker glass-like panels on each face at regular
    floor intervals so buildings look realistic from the drone camera.
    """
    glass_mat = _mat(stage, "OrcGlassMat",
                     albedo=(0.15, 0.22, 0.30), roughness=0.08,
                     metallic=0.7, opacity=0.55)
    dark_win = _mat(stage, "OrcDarkWinMat",
                    albedo=(0.08, 0.10, 0.14), roughness=0.1,
                    metallic=0.5, opacity=0.65)

    for bldg in buildings:
        if bldg.height < 6:
            continue  # skip very low buildings

        bldg_root = bldg.prim_path
        win_root = f"{bldg_root}/Windows"
        _xform(stage, win_root)

        floor_h = 4.0
        num_floors = max(1, int(bldg.height / floor_h))

        # Faces: (+X, -X, +Y, -Y)
        faces = [
            ("FaceXP", bldg.half_sx + 0.08, 0, bldg.half_sy * 0.85, 0.08),
            ("FaceXN", -(bldg.half_sx + 0.08), 0, bldg.half_sy * 0.85, 0.08),
            ("FaceYP", 0, bldg.half_sy + 0.08, 0.08, bldg.half_sx * 0.85),
            ("FaceYN", 0, -(bldg.half_sy + 0.08), 0.08, bldg.half_sx * 0.85),
        ]

        for face_name, dx, dy, wsx, wsy in faces:
            for fl in range(num_floors):
                fz = 3.0 + fl * (bldg.height - 4.0) / max(num_floors - 1, 1)
                win_mat = glass_mat if fl % 3 != 0 else dark_win
                win_path = f"{win_root}/{face_name}_F{fl}"
                _cube(stage, win_path, size=1.0,
                      translate=(dx, dy, fz),
                      scale=(wsx, wsy, 1.8))
                _bind(stage, win_path, win_mat)

    print(f"[ResQ-AI] Added window bands to {len(buildings)} buildings.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  ResQ-AI Orchestrator — Urban Disaster Simulation")
    print("=" * 72)

    # ── Load the base USDA scene ──────────────────────────────────────────
    scene_path = os.path.abspath(_args.scene)
    if not os.path.isfile(scene_path):
        print(f"[ResQ-AI] ERROR: Scene file not found: {scene_path}")
        simulation_app.close()
        sys.exit(1)

    print(f"[ResQ-AI] Loading base stage: {scene_path}")
    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        print("[ResQ-AI] ERROR: Failed to open stage.")
        simulation_app.close()
        sys.exit(1)

    # ── Create Isaac Sim world ────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # ── Task 1: Discover buildings ────────────────────────────────────────
    print("\n[ResQ-AI] ── Task 1: Building Discovery ──")
    buildings = discover_buildings(stage)
    if not buildings:
        print("[ResQ-AI] WARNING: No buildings found on stage.  "
              "Proceeding with limited functionality.")

    # ── Task 1b: Add visible windows to buildings ─────────────────────────
    print("\n[ResQ-AI] ── Task 1b: Adding Windows ──")
    add_building_windows(stage, buildings)

    # ── Task 1c: Generate city trees ──────────────────────────────────────
    print("\n[ResQ-AI] ── Task 1c: City Trees ──")
    tree_positions = generate_forest(stage, buildings, num_trees=120)

    # ── Task 2: Spawn fire emitters ───────────────────────────────────────
    print("\n[ResQ-AI] ── Task 2: Fire Emitter Setup ──")
    wind_dir = Gf.Vec3f(*_args.wind_direction)
    veg_fires = spawn_fire_emitters(stage, buildings, tree_positions, wind_dir)

    # ── Task 3: Spawn indoor pedestrians ──────────────────────────────────
    print("\n[ResQ-AI] ── Task 3: Indoor Victim Placement ──")
    spawn_indoor_pedestrians(stage, buildings,
                             count=_args.num_pedestrians)

    # ── Task 3b: Validate character loading ───────────────────────────────
    print("\n[ResQ-AI] ── Task 3b: Character Validation ──")
    validate_characters(stage, buildings)

    # ── Task 4: Fire spread manager ───────────────────────────────────────
    print("\n[ResQ-AI] ── Task 4: Fire Spread Logic ──")
    fire_mgr = FireSpreadManager(
        stage, buildings,
        spread_radius=_args.fire_spread_radius,
        wind_direction=wind_dir,
        wind_speed=_args.wind_speed,
        vegetation_fires=veg_fires,
        tree_positions=tree_positions)

    # Ignite initial vegetation fires (wildfire origin)
    fire_mgr.ignite_vegetation_fires()

    # Register a physics-step callback for fire spreading
    world.add_physics_callback("fire_spread", fire_mgr.on_physics_step)

    # ── Run the simulation loop ───────────────────────────────────────────
    print("\n[ResQ-AI] ── Simulation Loop ──")
    world.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print("[ResQ-AI] Simulation running.  Press Ctrl-C or close the "
          "window to exit.\n")

    step = 0
    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            # Periodic status print (every 300 steps ≈ 5 seconds at 60 Hz)
            if step % 300 == 0:
                fire_mgr.print_status()

            if _args.max_steps and step >= _args.max_steps:
                print(f"[ResQ-AI] Reached --max-steps={_args.max_steps}, "
                      f"stopping.")
                break

    except KeyboardInterrupt:
        print("\n[ResQ-AI] Interrupted by user.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    timeline.stop()
    simulation_app.close()

    # Print final fire report
    print("\n" + "=" * 72)
    print("  ResQ-AI — Final Fire Report")
    print("=" * 72)
    for b in buildings:
        status = "🔥 BURNING" if b.is_burning else "✅ Safe"
        victims = len(b.pedestrian_paths)
        print(f"  {b.name:12s}  {status}  "
              f"({victims} victim{'s' if victims != 1 else ''})")

    total_burning = sum(1 for b in buildings if b.is_burning)
    active_veg = sum(1 for v in veg_fires if v["active"])
    print(f"\n  Total: {total_burning}/{len(buildings)} buildings on fire.")
    print(f"  Vegetation fires: {active_veg}/{len(veg_fires)} active.")
    print(f"  Wind: ({_args.wind_direction[0]:.1f}, "
          f"{_args.wind_direction[1]:.1f}, "
          f"{_args.wind_direction[2]:.1f}) at {_args.wind_speed:.1f} m/s")
    print(f"  Simulation ended after {step} physics steps.")
    print("=" * 72)


if __name__ == "__main__":
    main()
