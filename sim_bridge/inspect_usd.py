"""
Inspect USD files to extract stage metadata: metersPerUnit, upAxis, root prim, extents.
Requires: pip install usd-core  (or run inside Isaac Sim's Python environment)
"""
import sys
import os

try:
    from pxr import Usd, UsdGeom, Gf
except ImportError:
    print("ERROR: pxr (OpenUSD) not available. Install with: pip install usd-core")
    print("Or run this script inside Isaac Sim's Python environment.")
    sys.exit(1)


def inspect_usd(filepath):
    """Open a USD file and print stage metadata + root prim info."""
    print(f"\n{'='*80}")
    print(f"FILE: {filepath}")
    print(f"{'='*80}")

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found!")
        return

    stage = Usd.Stage.Open(filepath)
    if not stage:
        print(f"  ERROR: Could not open stage!")
        return

    # --- Stage-level metadata ---
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)

    print(f"\n  metersPerUnit : {meters_per_unit}")
    if meters_per_unit == 1.0:
        print(f"                  -> Stage units are METERS")
    elif abs(meters_per_unit - 0.01) < 0.001:
        print(f"                  -> Stage units are CENTIMETERS")
    elif abs(meters_per_unit - 0.001) < 0.0001:
        print(f"                  -> Stage units are MILLIMETERS")
    elif abs(meters_per_unit - 0.0254) < 0.001:
        print(f"                  -> Stage units are INCHES")
    elif abs(meters_per_unit - 0.3048) < 0.01:
        print(f"                  -> Stage units are FEET")
    else:
        print(f"                  -> Custom unit scale")

    print(f"  upAxis        : {up_axis}")

    # --- Default prim ---
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        print(f"  defaultPrim   : {default_prim.GetPath()}")
    else:
        print(f"  defaultPrim   : (none set)")

    # --- Root prim ---
    root_prims = [p for p in stage.GetPseudoRoot().GetChildren()]
    print(f"  Root prims    : {[str(p.GetPath()) for p in root_prims]}")

    # --- Compute bounding box ---
    print(f"\n  --- Bounding Box (world space) ---")
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )

        # Try default prim first, then pseudo-root
        target_prim = default_prim if default_prim else stage.GetPseudoRoot()
        bbox = bbox_cache.ComputeWorldBound(target_prim)
        bbox_range = bbox.ComputeAlignedRange()

        if not bbox_range.IsEmpty():
            min_pt = bbox_range.GetMin()
            max_pt = bbox_range.GetMax()
            size = bbox_range.GetSize()

            print(f"  Target prim   : {target_prim.GetPath()}")
            print(f"  Min (xyz)     : ({min_pt[0]:.2f}, {min_pt[1]:.2f}, {min_pt[2]:.2f})")
            print(f"  Max (xyz)     : ({max_pt[0]:.2f}, {max_pt[1]:.2f}, {max_pt[2]:.2f})")
            print(f"  Size (xyz)    : ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")

            # Convert to meters
            size_m = Gf.Vec3d(size[0] * meters_per_unit,
                             size[1] * meters_per_unit,
                             size[2] * meters_per_unit)
            print(f"\n  Size in METERS: ({size_m[0]:.2f}, {size_m[1]:.2f}, {size_m[2]:.2f})")
            print(f"  Approx dimensions: {size_m[0]:.1f}m x {size_m[1]:.1f}m x {size_m[2]:.1f}m")
        else:
            print(f"  (empty bounding box - may contain only references/xforms)")
    except Exception as e:
        print(f"  ERROR computing bbox: {e}")

    # --- List top-level children (max 20) ---
    if default_prim:
        children = list(default_prim.GetChildren())
        print(f"\n  --- Top-level children of {default_prim.GetPath()} ({len(children)} total) ---")
        for child in children[:20]:
            print(f"    {child.GetPath()}  [{child.GetTypeName()}]")
        if len(children) > 20:
            print(f"    ... and {len(children) - 20} more")

    print()


# === Files to inspect ===
BASE = r"C:\Users\anshu\College\CosmosCookoff\ResQ-AI\assets"

files = [
    # Building
    os.path.join(BASE, "Architecture", "Demos", "AEC", "BrownstoneDemo", "Assets",
                 "Revit_Brownstone01", "Revit_Brownstone01_Exterior.usd"),
    os.path.join(BASE, "Architecture", "Demos", "AEC", "BrownstoneDemo", "Assets",
                 "Brownstone01.usd"),

    # Character
    os.path.join(BASE, "Characters", "Assets", "Characters", "Reallusion",
                 "Worker", "Worker.usd"),

    # Prop
    os.path.join(BASE, "Architecture", "Demos", "AEC", "BrownstoneDemo", "Props",
                 "Hydrant", "Hydrant.usd"),
    os.path.join(BASE, "Architecture", "Demos", "AEC", "BrownstoneDemo", "Props",
                 "StreetLamp", "StreetLamp.usd"),
]

for f in files:
    inspect_usd(f)
