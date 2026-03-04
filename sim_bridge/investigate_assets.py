"""
Investigate brownstone + character USD assets — writes results to file.
Run via: C:\\isaacsim\\isaac-sim.bat --exec --headless "path/to/this.py"
"""
from pxr import Usd, UsdGeom, Gf
import os, sys

ASSETS = "C:/Users/anshu/College/CosmosCookoff/ResQ-AI/assets"
OUT = "C:/Users/anshu/College/CosmosCookoff/ResQ-AI/sim_bridge/investigation_results.txt"

FILES = {
    "Brownstone02": ASSETS + "/Architecture/Demos/AEC/BrownstoneDemo/Assets/Revit_Brownstone02/Revit_Brownstone02_Exterior.usd",
    "Brownstone03": ASSETS + "/Architecture/Demos/AEC/BrownstoneDemo/Assets/Revit_Brownstone03/Revit_Brownstone03_Exterior.usd",
    "Worker":       ASSETS + "/Characters/Assets/Characters/Reallusion/Worker/Worker.usd",
    "Debra":        ASSETS + "/Characters/Assets/Characters/Reallusion/Debra/Debra.usd",
}

lines = []
def log(msg=""):
    lines.append(msg)
    print(msg)

def investigate(name, path):
    log()
    log("=" * 60)
    log("FILE: " + name)
    log("PATH: " + path)
    log("=" * 60)

    stg = Usd.Stage.Open(path)
    if not stg:
        log("  FAILED to open stage!")
        return

    mpu = UsdGeom.GetStageMetersPerUnit(stg)
    up = UsdGeom.GetStageUpAxis(stg)
    dp = stg.GetDefaultPrim()
    log("  metersPerUnit: " + str(mpu))
    log("  upAxis: " + str(up))
    log("  defaultPrim: " + (str(dp.GetPath()) if dp else "NONE"))

    root = stg.GetPseudoRoot()
    log("")
    log("  Root children:")
    for child in root.GetChildren():
        log("    " + str(child.GetPath()) + " (" + child.GetTypeName() + ")")
        for gc in child.GetChildren():
            log("      " + str(gc.GetPath()) + " (" + gc.GetTypeName() + ")")
            # One more level for brownstones
            if name.startswith("Brownstone"):
                for ggc in gc.GetChildren():
                    log("        " + str(ggc.GetPath()) + " (" + ggc.GetTypeName() + ")")

    # BBox
    targets = ["/World"]
    if dp:
        targets.append(str(dp.GetPath()))
    for prim_path in set(targets):
        prim = stg.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        try:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            bbox = cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
            wx = (mx[0] - mn[0]) * mpu
            wy = (mx[1] - mn[1]) * mpu
            wz = (mx[2] - mn[2]) * mpu
            log("")
            log("  BBox of " + prim_path + ":")
            log("    raw min: ({:.2f}, {:.2f}, {:.2f})".format(mn[0], mn[1], mn[2]))
            log("    raw max: ({:.2f}, {:.2f}, {:.2f})".format(mx[0], mx[1], mx[2]))
            log("    size (stage units): {:.2f} x {:.2f} x {:.2f}".format(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]))
            log("    size (meters): {:.2f} x {:.2f} x {:.2f}".format(wx, wy, wz))
            log("    center (stage units): ({:.2f}, {:.2f}, {:.2f})".format(
                (mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2))
        except Exception as e:
            log("  BBox ERROR for " + prim_path + ": " + str(e))

for name, path in FILES.items():
    try:
        investigate(name, path)
    except Exception as e:
        log("ERROR investigating " + name + ": " + str(e))

log("")
log("Done.")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
log("Results written to: " + OUT)
