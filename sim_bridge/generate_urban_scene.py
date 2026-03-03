"""
ResQ-AI Urban Disaster Scene Generator v4 - FINAL FIX
======================================================
All assets have metersPerUnit=0.01 (centimeters). ALL need scale=0.01.

Measured bounding boxes (in cm, raw stage units):
  Brownstone02: 665 x 3307 x 1613  -> 6.7m x 33.1m x 16.1m (narrow rowhouse strip)
  Brownstone03: 587 x 3309 x 1847  -> 5.9m x 33.1m x 18.5m (narrow rowhouse strip)
  StreetLamp:   378 x 57 x 1337    -> 3.8m x 0.6m x 13.4m
  Douglas_Fir:  302 x 277 x 603    -> 3.0m x 2.8m x 6.0m
  Worker:       196 x 48 x 194     -> 2.0m x 0.5m x 1.9m

The brownstones are NARROW ROWHOUSE STRIPS (~7m x 33m in real meters).
We place 2-3 per block edge, side by side, to form city blocks.

Run via: C:\\isaacsim\\python.bat generate_urban_scene.py
Or paste into Isaac Sim Script Editor.
"""

import os
import math
import random

import importlib
pxr_usd = importlib.import_module("pxr.Usd")
pxr_usdgeom = importlib.import_module("pxr.UsdGeom")
pxr_usdlux = importlib.import_module("pxr.UsdLux")
pxr_sdf = importlib.import_module("pxr.Sdf")
pxr_gf = importlib.import_module("pxr.Gf")
pxr_vt = importlib.import_module("pxr.Vt")

Usd = pxr_usd
UsdGeom = pxr_usdgeom
UsdLux = pxr_usdlux
Sdf = pxr_sdf
Gf = pxr_gf
Vt = pxr_vt

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ASSETS_ROOT = "C:/Users/anshu/College/CosmosCookoff/ResQ-AI/assets"
OUTPUT_PATH = "C:/Users/anshu/College/CosmosCookoff/ResQ-AI/resqai_urban_disaster.usda"
BROWNSTONE_BASE = ASSETS_ROOT + "/Architecture/Demos/AEC/BrownstoneDemo"

# ALL assets are in centimeters (mpu=0.01). Scale everything by 0.01.
S = 0.01

# Brownstone real-world sizes after scaling (meters):
# Brownstone02: ~6.7m wide (X) x 33.1m long (Y) x 16.1m tall (Z)
# Brownstone03: ~5.9m wide (X) x 33.1m long (Y) x 18.5m tall (Z)
# Both have min near origin in raw coords.
#
# Brownstone02 raw: min(0, 0, -253) max(665, 3307, 1360)
#   center_x_cm = 332.7, center_y_cm = 1653.3
# Brownstone03 raw: min(-1197, -1727, -241) max(-610, 1582, 1605)
#   center_x_cm = -903.3, center_y_cm = -72.5

# Building row dimensions in meters
BROW_WIDTH = 7.0    # X extent of a single brownstone row
BROW_LENGTH = 33.0  # Y extent (the long side)

# City layout in meters
ROAD_W = 12.0
SIDEWALK_W = 3.0
# A block holds 2 brownstone rows side-by-side (facing each other across a courtyard)
# plus some spacing
BLOCK_W = 80.0   # X direction (holds ~4-5 building rows + gaps)
BLOCK_H = 45.0   # Y direction (slightly longer than one brownstone)
NUM_COLS = 3
NUM_ROWS = 2
GAP = ROAD_W + 2.0 * SIDEWALK_W  # total gap between blocks

PARK_BLOCKS = [(1, 0)]

# Colors
CLR_ROAD = (0.08, 0.08, 0.08)
CLR_SIDEWALK = (0.55, 0.53, 0.50)
CLR_GRASS = (0.15, 0.35, 0.12)
CLR_BLOCK = (0.40, 0.38, 0.35)

# Heights
GROUND_Z = 0.0
ROAD_Z = 0.005
SIDEWALK_Z = 0.12
BLOCK_Z = 0.15

# =============================================================================
# ASSET PATHS
# =============================================================================

BUILDING_ASSETS = [
    BROWNSTONE_BASE + "/Assets/Revit_Brownstone02/Revit_Brownstone02_Exterior.usd",
    BROWNSTONE_BASE + "/Assets/Revit_Brownstone03/Revit_Brownstone03_Exterior.usd",
]

# Raw centers in cm (to offset so geometry centers on our placement point)
BUILDING_CENTERS_CM = [
    (332.7, 1653.3),    # Brownstone02
    (-903.3, -72.5),    # Brownstone03
]

# Trees (removed American_Beech - broken texture)
TREE_ASSETS = [
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/Douglas_Fir.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/Hawthorn.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/Honey_Locust.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/Largetooth_Aspen.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/Scarlet_Oak_fall.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Trees/White_Ash.usd",
]

# Shrubs (removed Acacia - broken texture)
SHRUB_ASSETS = [
    BROWNSTONE_BASE + "/Assets/Vegetation/Shrub/Forsythia.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Shrub/Hibiscus.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Shrub/Hydrangea.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Shrub/Lupin.usd",
    BROWNSTONE_BASE + "/Assets/Vegetation/Shrub/Sweet_Mock_Orange.usd",
]

PROP_PATHS = {
    "StreetLamp":   BROWNSTONE_BASE + "/Props/StreetLamp/StreetLamp.usd",
    "StreetLight":  BROWNSTONE_BASE + "/Props/StreetLight01/StreetLight01.usd",
    "Hydrant":      BROWNSTONE_BASE + "/Props/Hydrant/Hydrant.usd",
    "ParkBench":    BROWNSTONE_BASE + "/Props/ParkBench01/ParkBench01.usd",
    "TrashCan":     BROWNSTONE_BASE + "/Props/TrashCan/TrashCan.usd",
    "OutdoorTrash": BROWNSTONE_BASE + "/Props/OutdoorTrash01/OutdoorTrash01.usd",
    "Mailbox01":    BROWNSTONE_BASE + "/Props/Mailbox/MailboxType01.usd",
    "Mailbox02":    BROWNSTONE_BASE + "/Props/Mailbox/MailboxType02.usd",
    "Shelter":      BROWNSTONE_BASE + "/Props/MobilaShelter/MobilaShelter.usd",
}

# Characters (only Worker + Debra - ActorCore has broken Default_Material.mdl)
CHARACTER_ASSETS = [
    ASSETS_ROOT + "/Characters/Assets/Characters/Reallusion/Worker/Worker.usd",
    ASSETS_ROOT + "/Characters/Assets/Characters/Reallusion/Debra/Debra.usd",
]

FLOW_FIRE = ASSETS_ROOT + "/Particles/Assets/Extensions/Samples/Flow/presets/Fire/Fire.usda"
FLOW_SMOKE = ASSETS_ROOT + "/Particles/Assets/Extensions/Samples/Flow/presets/Smoke/DarkSmoke.usda"

# =============================================================================
# LAYOUT HELPERS
# =============================================================================

def total_size():
    tw = NUM_COLS * BLOCK_W + (NUM_COLS - 1) * GAP
    th = NUM_ROWS * BLOCK_H + (NUM_ROWS - 1) * GAP
    return tw, th

def block_center(col, row):
    tw, th = total_size()
    cx = -tw / 2.0 + col * (BLOCK_W + GAP) + BLOCK_W / 2.0
    cy = -th / 2.0 + row * (BLOCK_H + GAP) + BLOCK_H / 2.0
    return cx, cy

def block_bounds(col, row):
    cx, cy = block_center(col, row)
    return (cx - BLOCK_W / 2, cy - BLOCK_H / 2,
            cx + BLOCK_W / 2, cy + BLOCK_H / 2)

# =============================================================================
# MESH / REF HELPERS
# =============================================================================

def quad_mesh(stage, path, cx, cy, w, d, z, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    hw = w / 2.0
    hd = d / 2.0
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(cx - hw, cy - hd, z),
        Gf.Vec3f(cx + hw, cy - hd, z),
        Gf.Vec3f(cx + hw, cy + hd, z),
        Gf.Vec3f(cx - hw, cy + hd, z),
    ]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(color[0], color[1], color[2])]))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(cx - hw, cy - hd, z),
        Gf.Vec3f(cx + hw, cy + hd, z),
    ]))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    return mesh


def place_ref(stage, prim_path, asset_path, tx, ty, tz,
              rot_z=0.0, scale=0.01, ref_prim_path="",
              offset_cx_cm=0.0, offset_cy_cm=0.0):
    """Place a USD reference with proper centering.

    offset_cx_cm / offset_cy_cm: the asset's center in raw cm coords.
    We subtract these (scaled) from the translate so the asset's visual
    center lands at (tx, ty, tz).
    """
    xf = UsdGeom.Xform.Define(stage, prim_path)
    prim = xf.GetPrim()

    if ref_prim_path:
        prim.GetReferences().AddReference(
            assetPath=asset_path,
            primPath=Sdf.Path(ref_prim_path))
    else:
        prim.GetReferences().AddReference(asset_path)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    # Compute final position: subtract the asset's own center offset
    final_x = tx - offset_cx_cm * scale
    final_y = ty - offset_cy_cm * scale

    xformable.AddTranslateOp(
        opSuffix="city",
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(final_x, final_y, tz))

    if rot_z != 0.0:
        xformable.AddRotateZOp(
            opSuffix="city",
            precision=UsdGeom.XformOp.PrecisionDouble
        ).Set(float(rot_z))

    xformable.AddScaleOp(
        opSuffix="city",
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(scale, scale, scale))

    return prim


def place_simple(stage, prim_path, asset_path, tx, ty, tz,
                 rot_z=0.0, scale=0.01, scale_rand=0.0):
    """Place a reference without center offset (for props, trees, chars)."""
    xf = UsdGeom.Xform.Define(stage, prim_path)
    prim = xf.GetPrim()
    prim.GetReferences().AddReference(asset_path)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    xformable.AddTranslateOp(
        opSuffix="city",
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(tx, ty, tz))

    if rot_z != 0.0:
        xformable.AddRotateZOp(
            opSuffix="city",
            precision=UsdGeom.XformOp.PrecisionDouble
        ).Set(float(rot_z))

    s = scale
    if scale_rand > 0:
        s = s * random.uniform(1.0 - scale_rand, 1.0 + scale_rand)

    xformable.AddScaleOp(
        opSuffix="city",
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(s, s, s))

    return prim


# =============================================================================
# BUILD FUNCTIONS
# =============================================================================

def build_ground(stage):
    tw, th = total_size()
    margin = 30.0
    UsdGeom.Xform.Define(stage, "/World/Ground")

    # Dark asphalt base
    quad_mesh(stage, "/World/Ground/Base",
              0, 0, tw + margin * 2, th + margin * 2, GROUND_Z, CLR_ROAD)

    UsdGeom.Xform.Define(stage, "/World/Ground/Roads")
    ri = 0

    # N-S roads between columns
    col = 0
    while col < NUM_COLS - 1:
        xn, yn, xx, yx = block_bounds(col, 0)
        lx, ly, lxx, lyx = block_bounds(col + 1, 0)
        road_cx = (xx + lx) / 2.0
        quad_mesh(stage, "/World/Ground/Roads/NS_" + str(ri),
                  road_cx, 0, ROAD_W, th + margin, ROAD_Z, CLR_ROAD)
        # center line
        quad_mesh(stage, "/World/Ground/Roads/CL_" + str(ri),
                  road_cx, 0, 0.15, th + margin, ROAD_Z + 0.001,
                  (0.8, 0.7, 0.1))
        ri = ri + 1
        col = col + 1

    # E-W roads between rows
    row = 0
    while row < NUM_ROWS - 1:
        xn, yn, xx, yx = block_bounds(0, row)
        xn2, yn2, xx2, yx2 = block_bounds(0, row + 1)
        road_cy = (yx + yn2) / 2.0
        quad_mesh(stage, "/World/Ground/Roads/EW_" + str(ri),
                  0, road_cy, tw + margin, ROAD_W, ROAD_Z, CLR_ROAD)
        quad_mesh(stage, "/World/Ground/Roads/CL_" + str(ri),
                  0, road_cy, tw + margin, 0.15, ROAD_Z + 0.001,
                  (0.8, 0.7, 0.1))
        ri = ri + 1
        row = row + 1

    # Sidewalks
    UsdGeom.Xform.Define(stage, "/World/Ground/Sidewalks")
    si = 0
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            bxn, byn, bxx, byx = block_bounds(col, row)
            bcx, bcy = block_center(col, row)

            # South sidewalk
            quad_mesh(stage, "/World/Ground/Sidewalks/SW_" + str(si),
                      bcx, byn - SIDEWALK_W / 2.0,
                      BLOCK_W + SIDEWALK_W * 2, SIDEWALK_W,
                      SIDEWALK_Z, CLR_SIDEWALK)
            si = si + 1
            # North sidewalk
            quad_mesh(stage, "/World/Ground/Sidewalks/SW_" + str(si),
                      bcx, byx + SIDEWALK_W / 2.0,
                      BLOCK_W + SIDEWALK_W * 2, SIDEWALK_W,
                      SIDEWALK_Z, CLR_SIDEWALK)
            si = si + 1
            # West sidewalk
            quad_mesh(stage, "/World/Ground/Sidewalks/SW_" + str(si),
                      bxn - SIDEWALK_W / 2.0, bcy,
                      SIDEWALK_W, BLOCK_H,
                      SIDEWALK_Z, CLR_SIDEWALK)
            si = si + 1
            # East sidewalk
            quad_mesh(stage, "/World/Ground/Sidewalks/SW_" + str(si),
                      bxx + SIDEWALK_W / 2.0, bcy,
                      SIDEWALK_W, BLOCK_H,
                      SIDEWALK_Z, CLR_SIDEWALK)
            si = si + 1
            row = row + 1
        col = col + 1

    # Block interiors
    UsdGeom.Xform.Define(stage, "/World/Ground/Blocks")
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            cx, cy = block_center(col, row)
            is_park = (col, row) in PARK_BLOCKS
            if is_park:
                clr = CLR_GRASS
            else:
                clr = CLR_BLOCK
            quad_mesh(stage, "/World/Ground/Blocks/B_" + str(col) + "_" + str(row),
                      cx, cy, BLOCK_W, BLOCK_H, BLOCK_Z, clr)
            row = row + 1
        col = col + 1

    print("  Ground: " + str(int(tw)) + "m x " + str(int(th)) + "m")


def build_buildings(stage):
    """Place brownstone rowhouse strips along block edges.

    Each brownstone is ~7m wide x 33m long (after 0.01 scale).
    We place 3-4 rows per block along the north and south edges,
    with the long axis (Y in raw coords) running along the block edge.

    Brownstone02 raw center: (332.7, 1653.3)
    Brownstone03 raw center: (-903.3, -72.5)
    """
    UsdGeom.Xform.Define(stage, "/World/Buildings")
    idx = 0

    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            if (col, row) in PARK_BLOCKS:
                row = row + 1
                continue

            bxn, byn, bxx, byx = block_bounds(col, row)
            bcx, bcy = block_center(col, row)

            # Place building rows along both long edges (south and north)
            # Each row: ~7m wide, ~33m long
            # We rotate 90 degrees so the long axis runs along X (block width)
            # After 90 deg rotation: width becomes 33m (along X), depth becomes 7m (along Y)

            num_rows_per_edge = 2  # 2 rows deep on each side
            building_depth = BROW_WIDTH  # 7m per row after rotation
            courtyard = BLOCK_H - 4 * building_depth  # open space in middle

            edge_idx = 0
            while edge_idx < 2:
                r = 0
                while r < num_rows_per_edge:
                    # Which asset to use
                    asset_idx = (idx) % len(BUILDING_ASSETS)
                    asset = BUILDING_ASSETS[asset_idx]
                    ocx = BUILDING_CENTERS_CM[asset_idx][0]
                    ocy = BUILDING_CENTERS_CM[asset_idx][1]

                    # Y position: stack rows from the edge inward
                    if edge_idx == 0:
                        # South edge: start from byn, go inward
                        by = byn + 2.0 + building_depth / 2.0 + r * (building_depth + 1.0)
                        rot = 90.0
                    else:
                        # North edge: start from byx, go inward
                        by = byx - 2.0 - building_depth / 2.0 - r * (building_depth + 1.0)
                        rot = 270.0

                    # X position: center on block
                    bx = bcx

                    place_ref(stage,
                              "/World/Buildings/Bld_" + str(idx),
                              asset, bx, by, BLOCK_Z,
                              rot_z=rot, scale=S,
                              ref_prim_path="/World",
                              offset_cx_cm=ocx,
                              offset_cy_cm=ocy)
                    idx = idx + 1
                    r = r + 1
                edge_idx = edge_idx + 1

            row = row + 1
        col = col + 1

    print("  Buildings: " + str(idx) + " brownstone rows placed")


def build_vegetation(stage):
    UsdGeom.Xform.Define(stage, "/World/Vegetation")
    ti = 0
    si = 0

    # Sidewalk trees along N and S edges
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            bxn, byn, bxx, byx = block_bounds(col, row)

            for sw_y in [byn - SIDEWALK_W * 0.6, byx + SIDEWALK_W * 0.6]:
                x = bxn + 6.0
                while x < bxx - 6.0:
                    place_simple(stage,
                                 "/World/Vegetation/ST_" + str(ti),
                                 random.choice(TREE_ASSETS),
                                 x + random.uniform(-1, 1), sw_y, SIDEWALK_Z,
                                 rot_z=random.uniform(0, 360),
                                 scale=S, scale_rand=0.25)
                    ti = ti + 1
                    x = x + random.uniform(12, 18)
            row = row + 1
        col = col + 1

    # Park trees and shrubs
    for pcol, prow in PARK_BLOCKS:
        bxn, byn, bxx, byx = block_bounds(pcol, prow)
        pad = 5.0

        j = 0
        while j < random.randint(10, 16):
            place_simple(stage,
                         "/World/Vegetation/PT_" + str(ti),
                         random.choice(TREE_ASSETS),
                         random.uniform(bxn + pad, bxx - pad),
                         random.uniform(byn + pad, byx - pad),
                         BLOCK_Z,
                         rot_z=random.uniform(0, 360),
                         scale=S, scale_rand=0.3)
            ti = ti + 1
            j = j + 1

        j = 0
        while j < random.randint(15, 25):
            place_simple(stage,
                         "/World/Vegetation/PS_" + str(si),
                         random.choice(SHRUB_ASSETS),
                         random.uniform(bxn + pad, bxx - pad),
                         random.uniform(byn + pad, byx - pad),
                         BLOCK_Z,
                         rot_z=random.uniform(0, 360),
                         scale=S, scale_rand=0.3)
            si = si + 1
            j = j + 1

    print("  Vegetation: " + str(ti) + " trees, " + str(si) + " shrubs")


def build_props(stage):
    UsdGeom.Xform.Define(stage, "/World/StreetProps")
    pi = 0

    # Street lamps every ~18m on sidewalks
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            bxn, byn, bxx, byx = block_bounds(col, row)

            for sw_y in [byn - SIDEWALK_W * 0.7, byx + SIDEWALK_W * 0.7]:
                x = bxn + 8.0
                while x < bxx - 8.0:
                    place_simple(stage,
                                 "/World/StreetProps/Lamp_" + str(pi),
                                 PROP_PATHS["StreetLamp"],
                                 x, sw_y, SIDEWALK_Z,
                                 rot_z=0, scale=S)
                    pi = pi + 1
                    x = x + 18.0
            row = row + 1
        col = col + 1

    # Scatter per block
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            bxn, byn, bxx, byx = block_bounds(col, row)

            scatter = [
                ("Hydrant", random.randint(1, 2)),
                ("ParkBench", random.randint(2, 4)),
                ("TrashCan", random.randint(1, 3)),
                ("Mailbox01", 1),
                ("OutdoorTrash", random.randint(1, 2)),
            ]

            for prop_name, count in scatter:
                j = 0
                while j < count:
                    side = random.choice(["S", "N", "W", "E"])
                    if side == "S":
                        px = random.uniform(bxn + 3, bxx - 3)
                        py = byn - random.uniform(0.5, SIDEWALK_W - 0.3)
                    elif side == "N":
                        px = random.uniform(bxn + 3, bxx - 3)
                        py = byx + random.uniform(0.5, SIDEWALK_W - 0.3)
                    elif side == "W":
                        px = bxn - random.uniform(0.5, SIDEWALK_W - 0.3)
                        py = random.uniform(byn + 3, byx - 3)
                    else:
                        px = bxx + random.uniform(0.5, SIDEWALK_W - 0.3)
                        py = random.uniform(byn + 3, byx - 3)

                    place_simple(stage,
                                 "/World/StreetProps/" + prop_name + "_" + str(pi),
                                 PROP_PATHS[prop_name],
                                 px, py, SIDEWALK_Z,
                                 rot_z=random.uniform(0, 360),
                                 scale=S)
                    pi = pi + 1
                    j = j + 1
            row = row + 1
        col = col + 1

    print("  Props: " + str(pi) + " placed")


def build_civilians(stage):
    UsdGeom.Xform.Define(stage, "/World/Civilians")
    ci = 0

    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            bxn, byn, bxx, byx = block_bounds(col, row)

            j = 0
            while j < 5:
                side = random.choice(["S", "N"])
                px = random.uniform(bxn + 5, bxx - 5)
                if side == "S":
                    py = byn - random.uniform(0.5, SIDEWALK_W - 0.3)
                else:
                    py = byx + random.uniform(0.5, SIDEWALK_W - 0.3)

                asset = random.choice(CHARACTER_ASSETS)

                # Workers are Z-up already (up=Z from diagnostic)
                # No rot_x needed! They are already oriented correctly.
                prim = place_simple(stage,
                                    "/World/Civilians/Civ_" + str(ci).zfill(3),
                                    asset,
                                    px, py, SIDEWALK_Z,
                                    rot_z=random.uniform(0, 360),
                                    scale=S)

                prim.CreateAttribute("resqai:is_civilian",
                                     Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute("resqai:panic_state",
                                     Sdf.ValueTypeNames.String).Set("idle")
                prim.CreateAttribute("resqai:health",
                                     Sdf.ValueTypeNames.Float).Set(100.0)
                prim.CreateAttribute("resqai:civilian_id",
                                     Sdf.ValueTypeNames.Int).Set(ci)
                ci = ci + 1
                j = j + 1
            row = row + 1
        col = col + 1

    print("  Civilians: " + str(ci) + " placed")


def build_fire_zones(stage):
    UsdGeom.Xform.Define(stage, "/World/FireZones")

    # Place fire zones near building blocks
    building_blocks = []
    col = 0
    while col < NUM_COLS:
        row = 0
        while row < NUM_ROWS:
            if (col, row) not in PARK_BLOCKS:
                building_blocks.append((col, row))
            row = row + 1
        col = col + 1

    idx = 0
    while idx < min(5, len(building_blocks)):
        bc, br = building_blocks[idx % len(building_blocks)]
        cx, cy = block_center(bc, br)
        ox = random.uniform(-BLOCK_W / 4, BLOCK_W / 4)
        oy = random.choice([-1, 1]) * BLOCK_H / 3

        base = "/World/FireZones/FZ_" + str(idx)
        xf = UsdGeom.Xform.Define(stage, base)
        prim = xf.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(cx + ox, cy + oy, BLOCK_Z))

        prim.CreateAttribute("resqai:fire_active",
                             Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("resqai:fire_radius",
                             Sdf.ValueTypeNames.Float).Set(random.uniform(3, 8))
        prim.CreateAttribute("resqai:fire_intensity",
                             Sdf.ValueTypeNames.Float).Set(random.uniform(0.5, 1.0))
        prim.CreateAttribute("resqai:fire_spread_rate",
                             Sdf.ValueTypeNames.Float).Set(0.5)

        fe = UsdGeom.Xform.Define(stage, base + "/FlowFireEmitter")
        fe.GetPrim().CreateAttribute("resqai:emitter_type",
                                     Sdf.ValueTypeNames.String).Set("fire")
        fe.GetPrim().CreateAttribute("resqai:flow_preset_path",
                                     Sdf.ValueTypeNames.String).Set(FLOW_FIRE)

        se = UsdGeom.Xform.Define(stage, base + "/FlowSmokeEmitter")
        se.GetPrim().CreateAttribute("resqai:emitter_type",
                                     Sdf.ValueTypeNames.String).Set("smoke")
        se.GetPrim().CreateAttribute("resqai:flow_preset_path",
                                     Sdf.ValueTypeNames.String).Set(FLOW_SMOKE)

        idx = idx + 1

    print("  Fire zones: " + str(idx) + " created")


def build_drone_ops(stage):
    tw, th = total_size()
    UsdGeom.Xform.Define(stage, "/World/DroneOps")

    sp = UsdGeom.Xform.Define(stage, "/World/DroneOps/SpawnPoint")
    UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 50))
    sp.GetPrim().CreateAttribute("resqai:is_spawn",
                                 Sdf.ValueTypeNames.Bool).Set(True)

    UsdGeom.Xform.Define(stage, "/World/DroneOps/Waypoints")
    wps = [
        (0, 0, 50),
        (tw / 3, th / 3, 40),
        (tw / 3, -th / 3, 35),
        (-tw / 3, -th / 3, 35),
        (-tw / 3, th / 3, 40),
        (0, 0, 25),
    ]
    i = 0
    while i < len(wps):
        pos = wps[i]
        wp = UsdGeom.Xform.Define(stage, "/World/DroneOps/Waypoints/WP_" + str(i).zfill(3))
        UsdGeom.Xformable(wp.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(pos[0], pos[1], pos[2]))
        wp.GetPrim().CreateAttribute("resqai:waypoint_index",
                                     Sdf.ValueTypeNames.Int).Set(i)
        wp.GetPrim().CreateAttribute("resqai:hover_duration",
                                     Sdf.ValueTypeNames.Float).Set(3.0)
        i = i + 1

    fb = UsdGeom.Xform.Define(stage, "/World/DroneOps/FlightBoundary")
    fb.GetPrim().CreateAttribute("resqai:min_bound",
        Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(-tw / 2 - 30, -th / 2 - 30, 1))
    fb.GetPrim().CreateAttribute("resqai:max_bound",
        Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(tw / 2 + 30, th / 2 + 30, 100))

    print("  DroneOps: spawn + " + str(len(wps)) + " waypoints")


def build_lighting(stage):
    UsdGeom.Xform.Define(stage, "/World/Lighting")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/DomeLight")
    dome.CreateIntensityAttr(1500)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.92, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/SunLight")
    sun.CreateIntensityAttr(6000)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 30, 0))

    print("  Lighting: dome + sun")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 50)
    print("ResQ-AI City Generator v4")
    print("=" * 50)

    tw, th = total_size()
    print("City: " + str(NUM_COLS) + "x" + str(NUM_ROWS) + " blocks")
    print("Block: " + str(int(BLOCK_W)) + "m x " + str(int(BLOCK_H)) + "m")
    print("Total: " + str(int(tw)) + "m x " + str(int(th)) + "m")
    print("All assets scaled by " + str(S) + " (cm to m)")
    print("")

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(240)
    stage.SetTimeCodesPerSecond(24)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    print("[1/8] Ground...")
    build_ground(stage)

    print("[2/8] Buildings...")
    build_buildings(stage)

    print("[3/8] Vegetation...")
    build_vegetation(stage)

    print("[4/8] Props...")
    build_props(stage)

    print("[5/8] Civilians...")
    build_civilians(stage)

    print("[6/8] Fire zones...")
    build_fire_zones(stage)

    print("[7/8] Drone ops...")
    build_drone_ops(stage)

    print("[8/8] Lighting...")
    build_lighting(stage)

    stage.GetRootLayer().Save()

    print("")
    print("SAVED: " + OUTPUT_PATH)
    print("")
    print("Open in Isaac Sim, press F to frame all, and zoom out to see the city.")


if __name__ == "__main__":
    main()
