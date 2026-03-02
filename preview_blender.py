#!/usr/bin/env python3
"""ResQ-AI — Blender Scene Preview

Run from Blender's command line:
    /Applications/Blender.app/Contents/MacOS/Blender --python preview_blender.py

Or from Blender's Scripting tab: paste and run.
"""

import bpy
import math
import random
from mathutils import Vector, Euler

# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

WIND_DIR = (1.0, 0.5, 0.0)
WIND_SPEED = 8.0
NUM_TREES = 120
NUM_VICTIMS = 8
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_mat(name, color, emission=None, emission_strength=0, alpha=1.0,
             roughness=0.5, metallic=0.0):
    """Create a Principled BSDF material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = 'BLEND' if hasattr(mat, 'blend_method') else None
    return mat


def add_cube(name, location, scale, mat, collection=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    if collection:
        move_to_collection(obj, collection)
    return obj


def add_cylinder(name, location, radius, height, mat, collection=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=height,
                                         location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    if collection:
        move_to_collection(obj, collection)
    return obj


def add_sphere(name, location, radius, mat, collection=None, scale=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location,
                                          segments=16, ring_count=12)
    obj = bpy.context.active_object
    obj.name = name
    if scale:
        obj.scale = scale
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    if collection:
        move_to_collection(obj, collection)
    return obj


def add_cone(name, location, radius, height, mat, collection=None):
    bpy.ops.mesh.primitive_cone_add(radius1=radius, radius2=0,
                                     depth=height, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    if collection:
        move_to_collection(obj, collection)
    return obj


def add_light(name, location, light_type, color, energy, radius=1.0):
    if light_type == 'SUN':
        bpy.ops.object.light_add(type='SUN', location=location)
    elif light_type == 'POINT':
        bpy.ops.object.light_add(type='POINT', location=location)
    elif light_type == 'AREA':
        bpy.ops.object.light_add(type='AREA', location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.color = color
    obj.data.energy = energy
    if hasattr(obj.data, 'shadow_soft_size'):
        obj.data.shadow_soft_size = radius
    return obj


def get_or_create_collection(name, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent:
        parent.children.link(col)
    else:
        bpy.context.scene.collection.children.link(col)
    return col


def move_to_collection(obj, collection):
    for col in obj.users_collection:
        col.objects.unlink(obj)
    collection.objects.link(obj)


# ═══════════════════════════════════════════════════════════════════════════════
#  Building data
# ═══════════════════════════════════════════════════════════════════════════════

BUILDINGS = [
    {"name": "OfficeA",    "x": 25,  "y": 25,  "hsx": 8,  "hsy": 6,
     "h": 28, "color": (0.60, 0.62, 0.65)},
    {"name": "OfficeB",    "x": -25, "y": 25,  "hsx": 7,  "hsy": 7,
     "h": 22, "color": (0.55, 0.58, 0.62)},
    {"name": "Hospital",   "x": 25,  "y": -25, "hsx": 10, "hsy": 8,
     "h": 18, "color": (0.90, 0.92, 0.95)},
    {"name": "School",     "x": -25, "y": -25, "hsx": 9,  "hsy": 7,
     "h": 14, "color": (0.75, 0.70, 0.60)},
    {"name": "Apartments", "x": 0,   "y": 35,  "hsx": 6,  "hsy": 10,
     "h": 32, "color": (0.50, 0.45, 0.42)},
    {"name": "Tower",      "x": 0,   "y": -35, "hsx": 5,  "hsy": 5,
     "h": 45, "color": (0.38, 0.42, 0.48)},
    {"name": "Mall",       "x": 35,  "y": 0,   "hsx": 12, "hsy": 8,
     "h": 12, "color": (0.82, 0.78, 0.72)},
    {"name": "Warehouse",  "x": -35, "y": 0,   "hsx": 11, "hsy": 9,
     "h": 10, "color": (0.65, 0.60, 0.55)},
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Scene generators
# ═══════════════════════════════════════════════════════════════════════════════

def clear_scene():
    """Remove everything from the default scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)
    # Remove orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def create_terrain():
    print("  • Terrain & Roads")
    col = get_or_create_collection("Terrain")

    ground_mat = make_mat("Ground", (0.18, 0.30, 0.12), roughness=0.95)
    add_cube("Ground", (0, 0, -0.5), (200, 200, 1), ground_mat, col)

    road_mat = make_mat("Road", (0.15, 0.15, 0.17), roughness=0.7)
    add_cube("RoadNS", (0, 0, 0.02), (8, 200, 0.04), road_mat, col)
    add_cube("RoadEW", (0, 0, 0.02), (200, 8, 0.04), road_mat, col)

    # Diagonal roads
    diag = add_cube("RoadDiag1", (0, 0, 0.02), (6, 200, 0.04), road_mat, col)
    diag.rotation_euler = (0, 0, math.radians(45))
    diag2 = add_cube("RoadDiag2", (0, 0, 0.02), (6, 200, 0.04), road_mat, col)
    diag2.rotation_euler = (0, 0, math.radians(-45))

    sidewalk_mat = make_mat("Sidewalk", (0.45, 0.43, 0.40), roughness=0.8)
    for i, (dx, dy, sx, sy) in enumerate([
        (0, 80, 200, 3), (0, -80, 200, 3),
        (80, 0, 3, 200), (-80, 0, 3, 200),
    ]):
        add_cube(f"Walk_{i}", (dx, dy, 0.05), (sx, sy, 0.1), sidewalk_mat, col)


def create_buildings():
    print("  • Buildings (8)")
    col = get_or_create_collection("Buildings")

    glass_mat = make_mat("Glass", (0.12, 0.18, 0.28), roughness=0.08,
                         metallic=0.7, alpha=0.5)
    dark_win = make_mat("DarkWindow", (0.06, 0.08, 0.12), roughness=0.1,
                        metallic=0.5, alpha=0.6)

    for b in BUILDINGS:
        # Building body
        body_mat = make_mat(f"Bldg_{b['name']}", b["color"], roughness=0.6)
        add_cube(f"{b['name']}_Body",
                 (b["x"], b["y"], b["h"] / 2),
                 (b["hsx"] * 2, b["hsy"] * 2, b["h"]),
                 body_mat, col)

        # Roof
        rc = tuple(min(1.0, c + 0.12) for c in b["color"])
        roof_mat = make_mat(f"Roof_{b['name']}", rc, roughness=0.45)
        add_cube(f"{b['name']}_Roof",
                 (b["x"], b["y"], b["h"] + 0.2),
                 (b["hsx"] * 2 + 0.5, b["hsy"] * 2 + 0.5, 0.4),
                 roof_mat, col)

        # Windows (every floor, all 4 faces)
        floor_h = 4.0
        num_floors = max(1, int(b["h"] / floor_h))
        faces = [
            (b["hsx"] + 0.08, 0, b["hsy"] * 0.85, 0.08),
            (-(b["hsx"] + 0.08), 0, b["hsy"] * 0.85, 0.08),
            (0, b["hsy"] + 0.08, 0.08, b["hsx"] * 0.85),
            (0, -(b["hsy"] + 0.08), 0.08, b["hsx"] * 0.85),
        ]
        for fi, (fx, fy, wsx, wsy) in enumerate(faces):
            for fl in range(num_floors):
                fz = 3.0 + fl * max(1, (b["h"] - 4)) / max(num_floors, 1)
                wm = glass_mat if fl % 3 != 0 else dark_win
                add_cube(f"{b['name']}_W{fi}F{fl}",
                         (b["x"] + fx, b["y"] + fy, fz),
                         (wsx, wsy, 1.8), wm, col)

    return BUILDINGS


def create_forest(buildings):
    print(f"  • Forest ({NUM_TREES} trees)")
    random.seed(SEED)
    col = get_or_create_collection("Forest")

    bark = make_mat("Bark", (0.28, 0.16, 0.08), roughness=0.92)
    bark_lt = make_mat("BarkLight", (0.55, 0.45, 0.35), roughness=0.9)
    pine_m = make_mat("PineMat", (0.06, 0.25, 0.06), roughness=0.88)
    oak_m = make_mat("OakMat", (0.12, 0.40, 0.10), roughness=0.85)
    bush_m = make_mat("BushMat", (0.15, 0.32, 0.08), roughness=0.8)
    birch_m = make_mat("BirchMat", (0.20, 0.44, 0.15), roughness=0.82)

    inner_r, outer_r = 15.0, 65.0

    def inside_bldg(x, y):
        for b in buildings:
            if abs(x - b["x"]) < b["hsx"] + 3 and \
               abs(y - b["y"]) < b["hsy"] + 3:
                return True
        return False

    def near_road(x, y):
        w = 5.0
        return abs(x) < w or abs(y) < w

    # Cluster centres
    clusters = []
    for _ in range(NUM_TREES // 8):
        a = random.uniform(0, 2 * math.pi)
        r = random.uniform(inner_r + 5, outer_r - 5)
        cx, cy = r * math.cos(a), r * math.sin(a)
        if not inside_bldg(cx, cy) and not near_road(cx, cy):
            clusters.append((cx, cy))

    tree_positions = []
    attempts = 0
    tid = 0

    while len(tree_positions) < NUM_TREES and attempts < NUM_TREES * 15:
        attempts += 1

        if clusters and random.random() < 0.6:
            ccx, ccy = random.choice(clusters)
            x = ccx + random.gauss(0, 4.0)
            y = ccy + random.gauss(0, 4.0)
        else:
            ang = random.uniform(0, 2 * math.pi)
            rad = random.uniform(inner_r, outer_r)
            x, y = rad * math.cos(ang), rad * math.sin(ang)

        if inside_bldg(x, y):
            continue
        if near_road(x, y) and random.random() < 0.7:
            continue

        rv = random.random()
        if rv < 0.35:
            kind, s = "Pine", random.uniform(0.8, 1.3)
        elif rv < 0.60:
            kind, s = "Oak", random.uniform(0.7, 1.2)
        elif rv < 0.85:
            kind, s = "Bush", random.uniform(0.6, 1.5)
        else:
            kind, s = "Birch", random.uniform(0.9, 1.4)

        if kind == "Pine":
            trunk = add_cylinder(f"T{tid}_trunk", (x, y, 2.25 * s),
                                  0.15 * s, 4.5 * s, bark, col)
            canopy = add_cone(f"T{tid}_canopy", (x, y, 6.5 * s),
                               2.0 * s, 5.0 * s, pine_m, col)
        elif kind == "Oak":
            trunk = add_cylinder(f"T{tid}_trunk", (x, y, 1.5 * s),
                                  0.25 * s, 3.0 * s, bark, col)
            canopy = add_sphere(f"T{tid}_canopy", (x, y, 5.0 * s),
                                 3.0 * s, oak_m, col)
        elif kind == "Bush":
            canopy = add_sphere(f"T{tid}_bush", (x, y, 0.8 * s),
                                 1.2 * s, bush_m, col,
                                 scale=(1.3 * s, 1.3 * s, 0.8 * s))
        else:  # Birch
            trunk = add_cylinder(f"T{tid}_trunk", (x, y, 2.75 * s),
                                  0.10 * s, 5.5 * s, bark_lt, col)
            canopy = add_sphere(f"T{tid}_canopy", (x, y, 6.5 * s),
                                 1.8 * s, birch_m, col,
                                 scale=(1.0 * s, 1.0 * s, 1.3 * s))

        tree_positions.append((x, y, 0))
        tid += 1

    print(f"    Placed {len(tree_positions)} trees")
    return tree_positions


def create_characters(buildings):
    print(f"  • Characters ({NUM_VICTIMS} victims)")
    random.seed(SEED + 100)
    col = get_or_create_collection("Victims")

    skin = make_mat("Skin", (0.72, 0.55, 0.45), roughness=0.85)
    shirts = [
        make_mat("ShirtBlue", (0.1, 0.15, 0.45), roughness=0.75),
        make_mat("ShirtRed", (0.6, 0.15, 0.1), roughness=0.75),
        make_mat("ShirtGreen", (0.15, 0.4, 0.15), roughness=0.75),
    ]
    pants = make_mat("Pants", (0.15, 0.15, 0.20), roughness=0.8)

    vi = 0
    for b in buildings:
        per_b = max(1, NUM_VICTIMS // len(buildings))
        for j in range(per_b):
            if vi >= NUM_VICTIMS:
                break
            shirt = shirts[vi % len(shirts)]
            ox = b["x"] + random.uniform(-b["hsx"] + 1.5, b["hsx"] - 1.5)
            oy = b["y"] + random.uniform(-b["hsy"] + 1.5, b["hsy"] - 1.5)

            # Torso
            add_cylinder(f"V{vi:02d}_torso", (ox, oy, 1.15), 0.22, 0.50,
                         shirt, col)
            # Head
            add_sphere(f"V{vi:02d}_head", (ox, oy, 1.63), 0.12, skin, col)
            # Legs
            for li, ly in enumerate([-0.10, 0.10]):
                add_cylinder(f"V{vi:02d}_leg{li}", (ox, oy + ly, 0.33),
                             0.08, 0.55, pants, col)
            # Arms
            for ai, ay in enumerate([-0.30, 0.30]):
                add_cylinder(f"V{vi:02d}_arm{ai}", (ox, oy + ay, 1.05),
                             0.06, 0.55, shirt, col)
            vi += 1


def create_fires(buildings, tree_positions):
    print("  • Fire markers")
    random.seed(SEED + 200)
    col = get_or_create_collection("Fires")

    flame_mat = make_mat("Flame", (1.0, 0.3, 0.0), roughness=1.0,
                         emission=(1.0, 0.4, 0.0), emission_strength=50)
    core_mat = make_mat("FlameCore", (1.0, 0.9, 0.3), roughness=1.0,
                        emission=(1.0, 0.9, 0.3), emission_strength=80)

    # Wind scoring
    wn = Vector(WIND_DIR).normalized()
    scored = []
    for tp in tree_positions:
        tv = Vector(tp)
        if tv.length > 0.01:
            alignment = tv.normalized().dot(wn)
            scored.append((alignment, tp))
    scored.sort(key=lambda x: -x[0])

    # Vegetation fires (first 12, first 3 active/visible)
    num_vf = min(12, len(scored))
    for i in range(num_vf):
        _, tp = scored[i]
        x, y = tp[0], tp[1]

        # Fire cone
        add_cone(f"VegFire{i:02d}_flame", (x, y, 2.3), 1.2, 3.5,
                 flame_mat, col)
        add_cone(f"VegFire{i:02d}_core", (x, y, 2.0), 0.6, 2.5,
                 core_mat, col)

        # Glow light
        light = add_light(f"VegFire{i:02d}_glow", (x, y, 4),
                          'POINT', (1.0, 0.45, 0.05), 5000, radius=3)
        move_to_collection(light, col)

        # Hide inactive fires (only first 3 visible)
        if i >= 3:
            for suffix in ["_flame", "_core", "_glow"]:
                obj = bpy.data.objects.get(f"VegFire{i:02d}{suffix}")
                if obj:
                    obj.hide_viewport = True
                    obj.hide_render = True

    print(f"    {num_vf} vegetation fires (3 active)")

    # Building fires (all hidden)
    for b in buildings:
        x, y = b["x"], b["y"]
        fz = b["h"] + 2.0
        f1 = add_cone(f"BldgFire_{b['name']}_flame", (x, y, fz + 2),
                      1.5, 4.0, flame_mat, col)
        f2 = add_sphere(f"BldgFire_{b['name']}_core", (x, y, fz + 2.5),
                        0.8, core_mat, col)
        f1.hide_viewport = True
        f1.hide_render = True
        f2.hide_viewport = True
        f2.hide_render = True


def create_lighting():
    print("  • Lighting")
    # Sun
    sun = add_light("Sun", (0, 0, 50), 'SUN', (1.0, 0.95, 0.85), 5)
    sun.rotation_euler = Euler((math.radians(-55), math.radians(0),
                                math.radians(30)))
    # Fill light
    add_light("Fill", (0, 0, 80), 'POINT', (0.9, 0.85, 0.8), 50000,
              radius=50)

    # World background (sky blue)
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.45, 0.60, 0.82, 1.0)
    bg.inputs["Strength"].default_value = 0.8


def create_wind_arrow():
    print(f"  • Wind arrow → ({WIND_DIR[0]:.1f}, {WIND_DIR[1]:.1f})")
    col = get_or_create_collection("WindArrow")

    arrow_mat = make_mat("WindArrow", (0.2, 0.6, 1.0), roughness=1.0,
                         emission=(0.2, 0.6, 1.0), emission_strength=10)

    angle_z = math.atan2(WIND_DIR[1], WIND_DIR[0])
    arrow_len = 8.0 + WIND_SPEED * 0.5

    # Shaft
    shaft = add_cylinder("WindShaft", (0, 0, 0.5), 0.3, arrow_len,
                          arrow_mat, col)
    shaft.rotation_euler = (0, math.radians(90), angle_z)
    shaft.location = (math.cos(angle_z) * arrow_len / 2,
                      math.sin(angle_z) * arrow_len / 2, 0.5)

    # Arrowhead
    head = add_cone("WindHead", (0, 0, 0.5), 0.8, 2.5, arrow_mat, col)
    head.rotation_euler = (0, math.radians(90), angle_z)
    head.location = (math.cos(angle_z) * (arrow_len + 1.2),
                     math.sin(angle_z) * (arrow_len + 1.2), 0.5)


def setup_camera():
    """Position a camera for an overview shot."""
    bpy.ops.object.camera_add(location=(80, -80, 65))
    cam = bpy.context.active_object
    cam.name = "Overview"
    cam.rotation_euler = Euler((math.radians(55), 0, math.radians(45)))
    cam.data.lens = 28
    bpy.context.scene.camera = cam


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ResQ-AI — Blender Scene Preview")
    print("=" * 60)

    clear_scene()

    create_terrain()
    buildings = create_buildings()
    tree_pos = create_forest(buildings)
    create_characters(buildings)
    create_fires(buildings, tree_pos)
    create_lighting()
    create_wind_arrow()
    setup_camera()

    # Set viewport shading to Material Preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
            # Frame all
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = bpy.context.copy()
                    override['area'] = area
                    override['region'] = region
                    with bpy.context.temp_override(**override):
                        bpy.ops.view3d.view_all()
                    break
            break

    total = len(bpy.data.objects)
    print("-" * 60)
    print(f"  ✅ Scene created: {total} objects")
    print(f"     8 buildings • {NUM_TREES} trees • {NUM_VICTIMS} victims")
    print(f"     Wind: ({WIND_DIR[0]:.1f}, {WIND_DIR[1]:.1f}) "
          f"at {WIND_SPEED:.0f} m/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
