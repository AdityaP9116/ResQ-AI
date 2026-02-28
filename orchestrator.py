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

    /isaac-sim/python.sh orchestrator.py                # GUI mode
    /isaac-sim/python.sh orchestrator.py --headless      # headless mode

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
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N physics steps (0 = unlimited)")
    return parser.parse_args()

_args = _parse_args()

# Clamp pedestrian count to the requested 15-30 range
_args.num_pedestrians = max(15, min(30, _args.num_pedestrians))

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": _args.headless})

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
                    seed: int = 42) -> None:
    """Place trees WITHIN the city, scattered along streets and between
    buildings.  Trees are placed in a 15-65 m band from world origin,
    avoiding building footprints.

    Parameters
    ----------
    stage : Usd.Stage
        The active USD stage.
    buildings : list[BuildingInfo]
        Discovered buildings — used to avoid overlapping.
    num_trees : int
        Total number of tree point-instances.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    inner_r, outer_r = 15.0, 65.0
    print(f"[ResQ-AI] Generating {num_trees} trees within city "
          f"(r={inner_r}-{outer_r} m, avoiding buildings) …")

    forest_root = "/World/Forest"
    _xform(stage, forest_root)

    # ── Build a local prototype tree ────────────────────────────────────
    proto_root = f"{forest_root}/Prototypes"
    _xform(stage, proto_root)

    bark_mat = _mat(stage, "ForestBarkMat",
                    albedo=(0.28, 0.16, 0.08), roughness=0.92)
    canopy_mat = _mat(stage, "ForestCanopyMat",
                      albedo=(0.10, 0.35, 0.08), roughness=0.85)

    proto_tree = f"{proto_root}/Tree"
    _xform(stage, proto_tree)
    trunk_h = 3.5
    _cylinder(stage, f"{proto_tree}/Trunk", radius=0.18, height=trunk_h,
              translate=(0, 0, trunk_h / 2.0))
    _bind(stage, f"{proto_tree}/Trunk", bark_mat)
    _sphere(stage, f"{proto_tree}/Canopy", radius=2.2,
            translate=(0, 0, trunk_h + 1.3))
    _bind(stage, f"{proto_tree}/Canopy", canopy_mat)

    # Mark prototype as non-renderable (PointInstancer still clones it)
    proto_img = UsdGeom.Imageable(stage.GetPrimAtPath(proto_tree))
    if proto_img:
        proto_img.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

    # ── Configure the PointInstancer ──────────────────────────────────
    instancer_path = f"{forest_root}/TreeInstancer"
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    instancer.CreatePrototypesRel().SetTargets([Sdf.Path(proto_tree)])

    # Build exclusion zones from buildings (with a 3 m buffer)
    def _inside_any_building(x: float, y: float) -> bool:
        for b in buildings:
            margin = 3.0
            if (abs(x - b.world_x) < b.half_sx + margin and
                    abs(y - b.world_y) < b.half_sy + margin):
                return True
        return False

    positions: List[Gf.Vec3f] = []
    orientations: List[Gf.Quath] = []
    scales: List[Gf.Vec3f] = []
    proto_indices: List[int] = []
    attempts = 0

    while len(positions) < num_trees and attempts < num_trees * 10:
        attempts += 1
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(inner_r, outer_r)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        if _inside_any_building(x, y):
            continue

        s = random.uniform(0.7, 1.4)
        rot_z = random.uniform(0, 360)
        quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), rot_z).GetQuat()
        qh = Gf.Quath(float(quat.GetReal()),
                       float(quat.GetImaginary()[0]),
                       float(quat.GetImaginary()[1]),
                       float(quat.GetImaginary()[2]))

        positions.append(Gf.Vec3f(x, y, 0.0))
        orientations.append(qh)
        scales.append(Gf.Vec3f(s, s, s))
        proto_indices.append(0)

    instancer.CreatePositionsAttr(positions)
    instancer.CreateOrientationsAttr(orientations)
    instancer.CreateScalesAttr(scales)
    instancer.CreateProtoIndicesAttr(proto_indices)

    _label_recursive(stage, forest_root, "vegetation")
    print(f"[ResQ-AI] Placed {len(positions)} trees within the city.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TASK 2 — Procedural fire emitter setup
# ═══════════════════════════════════════════════════════════════════════════════

# Reference fire USD on the Nucleus (used when omni.flow is unavailable)
FIRE_ASSET_USD = "omniverse://localhost/NVIDIA/Assets/FX/Fire.usd"


def spawn_fire_emitters(stage: Usd.Stage,
                        buildings: List[BuildingInfo]) -> None:
    """Spawn one fire emitter per building, initially invisible.

    Strategy:
      • If ``omni.flow`` is available, create a Flow emitter prim with
        temperature / fuel parameters configured for a building fire.
      • Otherwise, fall back to an emissive-geometry fire (cone + sphere
        cluster) identical to the pattern in ``generate_urban_scene.py``.

    All fire prims start with ``visibility = "invisible"`` so they can be
    toggled on when the fire-spread logic ignites a building.
    """
    fires_root = "/World/Fires"
    _xform(stage, fires_root)

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

    for bldg in buildings:
        fire_path = f"{fires_root}/{bldg.name}_fire"
        # Place fire ON TOP of the building so it's visible from the drone.
        # +2 m above rooftop so it pokes above the roof slab.
        fire_z = bldg.height + 2.0

        if HAS_FLOW:
            # ── omni.flow emitter ─────────────────────────────────────────
            _spawn_flow_fire(stage, fire_path,
                             bldg.world_x, bldg.world_y, fire_z,
                             bldg.half_sx, bldg.half_sy)
        else:
            # ── Fallback: emissive geometry fire ──────────────────────────
            _spawn_geometry_fire(stage, fire_path,
                                 bldg.world_x, bldg.world_y, fire_z,
                                 flame_core, flame_outer, smoke_mat)

        # Make the fire invisible at start
        fire_prim = stage.GetPrimAtPath(fire_path)
        if fire_prim.IsValid():
            UsdGeom.Imageable(fire_prim).MakeInvisible()

        bldg.fire_prim_path = fire_path

        # Apply semantic label
        _label_recursive(stage, fire_path, "fire")

    print(f"[ResQ-AI] Spawned {len(buildings)} fire emitters "
          f"({'omni.flow' if HAS_FLOW else 'emissive geometry'} mode, "
          f"all invisible).")


def _spawn_flow_fire(stage: Usd.Stage, path: str,
                     wx: float, wy: float, wz: float,
                     half_sx: float, half_sy: float) -> None:
    """Create an ``omni.flow`` fire emitter at world position (wx, wy, wz).

    The emitter is sized proportionally to the building footprint so that
    larger buildings get larger fires.

    This creates the standard Flow USD prim hierarchy::

        /path            <- Xform (positioned at building roof)
          /flowSimulate  <- OmniFlow typed prim (sim params)
          /flowEmitter   <- FlowEmitterSphere typed prim
          /flowRender    <- OmniFlow render prim
    """
    _xform(stage, path, translate=(wx, wy, wz))

    # ── Flow Simulate prim ────────────────────────────────────────────
    sim_path = f"{path}/flowSimulate"
    sim_prim = stage.DefinePrim(sim_path, "FlowSimulate")
    sim_prim.CreateAttribute("buoyancyPerTemp", Sdf.ValueTypeNames.Float).Set(4.0)
    sim_prim.CreateAttribute("burnPerTemp", Sdf.ValueTypeNames.Float).Set(0.15)
    sim_prim.CreateAttribute("coolingRate", Sdf.ValueTypeNames.Float).Set(2.5)
    sim_prim.CreateAttribute("gravity", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(0.0, 0.0, -3.0))

    # ── Flow Emitter Sphere ───────────────────────────────────────────
    emitter_path = f"{path}/flowEmitter"
    emitter_prim = stage.DefinePrim(emitter_path, "FlowEmitterSphere")

    em_radius = min(half_sx, half_sy) * 0.4
    emitter_prim.CreateAttribute("radius", Sdf.ValueTypeNames.Float).Set(
        float(em_radius))
    emitter_prim.CreateAttribute("fuel", Sdf.ValueTypeNames.Float).Set(1.8)
    emitter_prim.CreateAttribute("temperature", Sdf.ValueTypeNames.Float).Set(
        1500.0)
    emitter_prim.CreateAttribute("smoke", Sdf.ValueTypeNames.Float).Set(0.7)
    emitter_prim.CreateAttribute("velocity", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3f(0.0, 0.0, 2.5))

    # ── Flow Render prim ──────────────────────────────────────────────
    render_path = f"{path}/flowRender"
    render_prim = stage.DefinePrim(render_path, "FlowRender")
    render_prim.CreateAttribute("colorMapResolution", Sdf.ValueTypeNames.Int).Set(64)
    render_prim.CreateAttribute("shadowFactor", Sdf.ValueTypeNames.Float).Set(1.0)


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

# Character USD assets on the Nucleus (for omni.anim.people references)
_CHARACTER_USDS = [
    "omniverse://localhost/NVIDIA/Assets/Characters/Reallusion/"
    "Worker/Worker.usd",
    "omniverse://localhost/NVIDIA/Assets/Characters/Reallusion/"
    "BusinessMan/BusinessMan.usd",
    "omniverse://localhost/NVIDIA/Assets/Characters/Reallusion/"
    "Casual_Female/Casual_Female.usd",
]


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
        # Random offset within 70 % of the footprint
        offset_x = random.uniform(-bldg.half_sx * 0.7, bldg.half_sx * 0.7)
        offset_y = random.uniform(-bldg.half_sy * 0.7, bldg.half_sy * 0.7)
        spawn_x = bldg.world_x + offset_x
        spawn_y = bldg.world_y + offset_y

        # Pick a random floor (Z height)
        max_floors = max(1, int(bldg.height / floor_height))
        floor = random.randint(0, max_floors - 1)
        spawn_z = floor * floor_height + 0.05  # slightly above floor slab

        if HAS_ANIM_PEOPLE:
            # ── omni.anim.people character ────────────────────────────────
            _spawn_anim_person(stage, ped_path, spawn_x, spawn_y, spawn_z,
                               idx)
        else:
            # ── Geometric fallback (cylinder + sphere) ────────────────────
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
                       x: float, y: float, z: float, idx: int) -> None:
    """Reference an omni.anim.people character USD at the given position."""
    char_usd = _CHARACTER_USDS[idx % len(_CHARACTER_USDS)]
    xf = _xform(stage, path, translate=(x, y, z))

    # Add as a USD reference so the character loads from the Nucleus
    prim = stage.GetPrimAtPath(path)
    prim.GetReferences().AddReference(char_usd)

    _apply_semantic_label(prim, "person")


def _spawn_geometry_person(stage: Usd.Stage, path: str,
                           x: float, y: float, z: float,
                           skin_mat: str, shirt_mat: str,
                           pants_mat: str) -> None:
    """Build a simple geometric person (cylinder torso + sphere head)."""
    _xform(stage, path, translate=(x, y, z))

    _cylinder(stage, f"{path}/Torso", radius=0.2, height=0.65,
              translate=(0, 0, 1.08))
    _bind(stage, f"{path}/Torso", shirt_mat)

    _sphere(stage, f"{path}/Head", radius=0.15, translate=(0, 0, 1.6))
    _bind(stage, f"{path}/Head", skin_mat)

    _cylinder(stage, f"{path}/Hips", radius=0.18, height=0.3,
              translate=(0, 0, 0.7))
    _bind(stage, f"{path}/Hips", pants_mat)

    for li, ly in enumerate([-0.1, 0.1]):
        _cylinder(stage, f"{path}/Leg_{li}", radius=0.08, height=0.55,
                  translate=(0, ly, 0.28))
        _bind(stage, f"{path}/Leg_{li}", pants_mat)

    for ai, ay in enumerate([-0.28, 0.28]):
        _cylinder(stage, f"{path}/Arm_{ai}", radius=0.06, height=0.55,
                  translate=(0, ay, 1.0))
        _bind(stage, f"{path}/Arm_{ai}", shirt_mat)
        _sphere(stage, f"{path}/Hand_{ai}", radius=0.06,
                translate=(0, ay, 0.7))
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
    """Manages fire state, distance-based spreading, and pedestrian flee logic.

    Attributes
    ----------
    buildings : list[BuildingInfo]
        All discovered buildings (state is mutated in-place).
    spread_radius : float
        Distance threshold (metres) for fire to jump between buildings.
    stage : Usd.Stage
        Active USD stage.
    step_count : int
        Counts physics steps for timing / throttling logic.
    """

    # How often (in physics steps) we check for fire spreading.
    # Checking every step is wasteful; every 60 steps ≈ 1 second at 60 Hz.
    CHECK_INTERVAL: int = 60

    def __init__(self, stage: Usd.Stage, buildings: List[BuildingInfo],
                 spread_radius: float = 40.0) -> None:
        self.stage = stage
        self.buildings = buildings
        self.spread_radius = spread_radius
        self.step_count = 0
        self._first_check = True  # debug flag for first spread check

        # Pre-compute building centre positions for fast distance checks
        self._positions = {
            b.name: (b.world_x, b.world_y)
            for b in buildings
        }

        # ── Debug: print all building positions and nearest distances ──
        print(f"[ResQ-AI] FireSpreadManager: spread_radius = {spread_radius:.1f} m")
        print(f"[ResQ-AI] Building positions:")
        for b in buildings:
            print(f"          {b.name:12s}  ({b.world_x:7.1f}, {b.world_y:7.1f})")

        # Show closest neighbour for each building
        for b in buildings:
            min_dist = float('inf')
            closest = "none"
            for other in buildings:
                if other.name == b.name:
                    continue
                d = math.sqrt((b.world_x - other.world_x)**2 +
                              (b.world_y - other.world_y)**2)
                if d < min_dist:
                    min_dist = d
                    closest = other.name
            within = "✅ within" if min_dist <= spread_radius else "❌ outside"
            print(f"          {b.name:12s} → nearest: {closest:12s} "
                  f"d={min_dist:.1f} m  {within} {spread_radius:.0f} m radius")

    # ── Public API ────────────────────────────────────────────────────────

    def ignite_random_building(self) -> None:
        """Ignite one random building to kick off the simulation."""
        if not self.buildings:
            return
        victim = random.choice(self.buildings)
        self._ignite(victim)
        print(f"[ResQ-AI] 🔥 Initial ignition: {victim.name}")

    def on_physics_step(self, dt: float) -> None:
        """Called every physics step.  Checks for fire spreading at a
        throttled interval to avoid excessive computation."""
        self.step_count += 1
        if self.step_count % self.CHECK_INTERVAL != 0:
            return
        self._check_spread()

    # ── Internals ─────────────────────────────────────────────────────────

    def _ignite(self, bldg: BuildingInfo) -> None:
        """Set a building on fire: flip state, make fire prim visible,
        and trigger flee for its indoor occupants."""
        if bldg.is_burning:
            return
        bldg.is_burning = True

        # Make fire visible
        fire_prim = self.stage.GetPrimAtPath(bldg.fire_prim_path)
        if fire_prim and fire_prim.IsValid():
            UsdGeom.Imageable(fire_prim).MakeVisible()

        # Trigger flee for pedestrians inside this building
        self._trigger_flee(bldg)

    def _check_spread(self) -> None:
        """Time-based probabilistic fire spread.

        Instead of a hard distance threshold, every building has a
        probability of catching fire each check that increases with:
          1. Proximity to any burning building (inverse-square)
          2. Time elapsed since the simulation started

        This ensures fire GRADUALLY spreads everywhere, with nearby
        buildings catching fire sooner.
        """
        burning = [b for b in self.buildings if b.is_burning]
        not_burning = [b for b in self.buildings if not b.is_burning]

        if not not_burning or not burning:
            return

        # Time factor: increases spread probability over time
        # At step 0 → factor=0.01, at step 3600 (≈60s) → factor ~0.35
        time_factor = min(1.0, 0.01 + self.step_count / 10000.0)

        for dst in not_burning:
            dx, dy = self._positions[dst.name]

            # Find minimum distance to any burning building
            min_dist = float('inf')
            nearest_src = None
            for src in burning:
                sx, sy = self._positions[src.name]
                dist = math.sqrt((sx - dx) ** 2 + (sy - dy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_src = src

            # Probability: higher for closer buildings, grows with time
            # P = time_factor * (spread_radius / max(dist, 1))^2
            proximity = (self.spread_radius / max(min_dist, 1.0)) ** 2
            p_spread = min(0.8, time_factor * proximity)

            if random.random() < p_spread:
                self._ignite(dst)
                print(f"[ResQ-AI] 🔥 Fire spread: {nearest_src.name} → "
                      f"{dst.name} (d={min_dist:.1f} m, p={p_spread:.2f})")

    def _trigger_flee(self, bldg: BuildingInfo) -> None:
        """PHYSICALLY MOVE pedestrians out of the building so the drone
        camera can detect them.  For static geometry pedestrians,
        navigation waypoints have no effect — we *must* relocate the
        prim's xformOp:translate to place them on the street."""
        if not bldg.pedestrian_paths:
            return

        for idx, ped_path in enumerate(bldg.pedestrian_paths):
            ped_prim = self.stage.GetPrimAtPath(ped_path)
            if not ped_prim or not ped_prim.IsValid():
                continue

            # Compute a flee destination outside the building:
            # Move outward along the direction from building centre to world
            # origin, staggered so multiple victims don't stack on each other.
            dx = -bldg.world_x
            dy = -bldg.world_y
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                dx, dy = 1.0, 0.0
            else:
                dx /= length
                dy /= length

            # Each victim exits at building edge + 3-10 m into the street,
            # with a lateral offset so they don't all overlap.
            flee_dist = bldg.half_sx + 3.0 + idx * 2.5 + random.uniform(0, 2)
            lateral = (idx - len(bldg.pedestrian_paths) / 2.0) * 2.0

            flee_x = bldg.world_x + dx * flee_dist + (-dy) * lateral
            flee_y = bldg.world_y + dy * flee_dist + dx * lateral
            flee_z = 0.05  # ground level

            # ── Actually relocate the prim ─────────────────────────────
            xformable = UsdGeom.Xformable(ped_prim)
            if xformable:
                # Find the existing translate op and overwrite it
                found = False
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:translate":
                        op.Set(Gf.Vec3d(flee_x, flee_y, flee_z))
                        found = True
                        break
                if not found:
                    # No existing translate op — add one
                    xformable.AddTranslateOp().Set(
                        Gf.Vec3d(flee_x, flee_y, flee_z))

            # Mark as fleeing
            ped_prim.CreateAttribute(
                "resqai:state", Sdf.ValueTypeNames.String).Set("fleeing")

        print(f"[ResQ-AI] 🏃 {len(bldg.pedestrian_paths)} victim(s) in "
              f"{bldg.name} moved outside (visible to drone).")


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
    generate_forest(stage, buildings, num_trees=120)

    # ── Task 2: Spawn fire emitters ───────────────────────────────────────
    print("\n[ResQ-AI] ── Task 2: Fire Emitter Setup ──")
    spawn_fire_emitters(stage, buildings)

    # ── Task 3: Spawn indoor pedestrians ──────────────────────────────────
    print("\n[ResQ-AI] ── Task 3: Indoor Victim Placement ──")
    spawn_indoor_pedestrians(stage, buildings,
                             count=_args.num_pedestrians)

    # ── Task 4: Fire spread manager ───────────────────────────────────────
    print("\n[ResQ-AI] ── Task 4: Fire Spread Logic ──")
    fire_mgr = FireSpreadManager(stage, buildings,
                                 spread_radius=_args.fire_spread_radius)

    # Ignite one random building at the start
    fire_mgr.ignite_random_building()

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
                burning = sum(1 for b in buildings if b.is_burning)
                print(f"[ResQ-AI] step={step}  "
                      f"burning={burning}/{len(buildings)}")

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
    print(f"\n  Total: {total_burning}/{len(buildings)} buildings on fire.")
    print(f"  Simulation ended after {step} physics steps.")
    print("=" * 72)


if __name__ == "__main__":
    main()
