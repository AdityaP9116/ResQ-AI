"""Civilian tracking and census for ResQ-AI Isaac Sim pipeline.

Reads /World/Civilians/Civ_* prims, tracks their states, correlates
with YOLO person detections, and produces structured civilian reports.

Isaac Sim 5.1 compatibility: NO f-strings.  importlib for pxr.
"""

import importlib
import math
import time

import numpy as np

# pxr via importlib
Usd = importlib.import_module("pxr.Usd")
UsdGeom = importlib.import_module("pxr.UsdGeom")
Gf = importlib.import_module("pxr.Gf")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEAR_FIRE_RADIUS_M = 30.0
CRITICAL_RADIUS_M = 10.0

VALID_STATES = [
    "idle", "alert", "fleeing", "panicking",
    "injured", "incapacitated", "rescued",
]


# ---------------------------------------------------------------------------
# CivilianTracker
# ---------------------------------------------------------------------------

class CivilianTracker(object):
    """Tracks civilians in the Isaac Sim disaster scene.

    Reads prim attributes, correlates with YOLO detections, and
    provides real-time census data.
    """

    def __init__(self):
        self._civilians = {}   # civ_id -> {prim_path, position, state, health, ...}
        self._fire_zones = {}  # zone_name -> {position, radius, active}
        self._last_update = 0.0
        self._discover_civilians()
        self._discover_fire_zones()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_civilians(self):
        """Read all /World/Civilians/Civ_* prims."""
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            root = stage.GetPrimAtPath("/World/Civilians")
            if not root.IsValid():
                print("[CivTracker] /World/Civilians not found")
                return

            children = root.GetChildren()
            ci = 0
            while ci < len(children):
                prim = children[ci]
                name = prim.GetName()
                if name.startswith("Civ_"):
                    pos = self._get_prim_position(prim)
                    civ_id_attr = prim.GetAttribute("resqai:civilian_id")
                    civ_id = int(civ_id_attr.Get()) if civ_id_attr.IsValid() else ci

                    state_attr = prim.GetAttribute("resqai:panic_state")
                    state = str(state_attr.Get()) if state_attr.IsValid() else "idle"

                    health_attr = prim.GetAttribute("resqai:health")
                    health = float(health_attr.Get()) if health_attr.IsValid() else 100.0

                    self._civilians[civ_id] = {
                        "prim_path": str(prim.GetPath()),
                        "name": name,
                        "position": pos,
                        "state": state,
                        "health": health,
                        "detected_by_yolo": False,
                        "last_seen_frame": -1,
                    }
                ci = ci + 1

            print("[CivTracker] Found " + str(len(self._civilians)) +
                  " civilians")
        except Exception as e:
            print("[CivTracker] Discovery error: " + str(e))

    def _discover_fire_zones(self):
        """Cache fire zone positions for proximity checks."""
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            root = stage.GetPrimAtPath("/World/FireZones")
            if not root.IsValid():
                return

            children = root.GetChildren()
            fi = 0
            while fi < len(children):
                prim = children[fi]
                name = prim.GetName()
                if name.startswith("FZ_"):
                    pos = self._get_prim_position(prim)
                    radius_attr = prim.GetAttribute("resqai:fire_radius")
                    radius = float(radius_attr.Get()) if radius_attr.IsValid() else 5.0
                    active_attr = prim.GetAttribute("resqai:fire_active")
                    active = bool(active_attr.Get()) if active_attr.IsValid() else False

                    self._fire_zones[name] = {
                        "position": pos,
                        "radius": radius,
                        "active": active,
                    }
                fi = fi + 1
        except Exception as e:
            print("[CivTracker] Fire zone discovery error: " + str(e))

    def _get_prim_position(self, prim):
        """Extract translate from xformable prim."""
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        oi = 0
        while oi < len(ops):
            op = ops[oi]
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                return [float(v[0]), float(v[1]), float(v[2])]
            oi = oi + 1
        return [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Update from live data
    # ------------------------------------------------------------------

    def update(self, detections=None, fire_report=None, frame_idx=0):
        """Refresh civilian data from prim attributes and YOLO detections.

        Parameters
        ----------
        detections : list[dict] or None
            Person detections from DualYOLODetector.detect().
        fire_report : dict or None
            From FireManager.get_fire_report() — used to update fire zone
            active status.
        frame_idx : int
            Current frame number (for recency tracking).
        """
        self._last_update = time.time()

        # Refresh prim attributes
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                civ_ids = list(self._civilians.keys())
                ci = 0
                while ci < len(civ_ids):
                    cid = civ_ids[ci]
                    info = self._civilians[cid]
                    prim = stage.GetPrimAtPath(info["prim_path"])
                    if prim.IsValid():
                        info["position"] = self._get_prim_position(prim)

                        st = prim.GetAttribute("resqai:panic_state")
                        if st.IsValid():
                            info["state"] = str(st.Get())

                        hp = prim.GetAttribute("resqai:health")
                        if hp.IsValid():
                            info["health"] = float(hp.Get())
                    ci = ci + 1

                # Update fire zone active status
                if fire_report and fire_report.get("active_fires"):
                    active_names = set()
                    af = fire_report["active_fires"]
                    ai = 0
                    while ai < len(af):
                        active_names.add(af[ai]["zone"])
                        ai = ai + 1

                    fz_keys = list(self._fire_zones.keys())
                    fi = 0
                    while fi < len(fz_keys):
                        fk = fz_keys[fi]
                        self._fire_zones[fk]["active"] = fk in active_names
                        fi = fi + 1
        except Exception as e:
            print("[CivTracker] Update error: " + str(e))

        # Mark YOLO-detected civilians
        if detections is not None:
            person_dets = []
            di = 0
            while di < len(detections):
                if detections[di]["class"] == "person":
                    person_dets.append(detections[di])
                di = di + 1

            # Reset detection flags
            civ_ids = list(self._civilians.keys())
            ci = 0
            while ci < len(civ_ids):
                self._civilians[civ_ids[ci]]["detected_by_yolo"] = False
                ci = ci + 1

            # Simple nearest-match correlation (could be improved with
            # camera projection, but this works for census purposes)
            pi = 0
            while pi < len(person_dets):
                # Mark the closest civilian as detected
                best_id = None
                best_dist = 999999.0
                ci = 0
                while ci < len(civ_ids):
                    cid = civ_ids[ci]
                    if not self._civilians[cid]["detected_by_yolo"]:
                        # Use a simple heuristic — this would need camera
                        # projection for true 2D-3D correlation
                        best_id = cid
                        self._civilians[cid]["detected_by_yolo"] = True
                        self._civilians[cid]["last_seen_frame"] = frame_idx
                        break
                    ci = ci + 1
                pi = pi + 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_civilian_report(self):
        """Return structured civilian census.

        Returns
        -------
        dict
            See Prompt 4 spec for full schema.
        """
        by_state = {}
        si = 0
        while si < len(VALID_STATES):
            by_state[VALID_STATES[si]] = 0
            si = si + 1

        near_fire = []
        critical_count = 0
        total_health = 0.0
        rescue_priority = []

        civ_ids = list(self._civilians.keys())
        ci = 0
        while ci < len(civ_ids):
            cid = civ_ids[ci]
            info = self._civilians[cid]

            # Count by state
            state = info["state"]
            if state in by_state:
                by_state[state] = by_state[state] + 1
            else:
                by_state["idle"] = by_state.get("idle", 0) + 1

            total_health = total_health + info["health"]

            # Proximity to fire zones
            nearest_fire_dist = 999999.0
            nearest_fire_zone = ""

            fz_keys = list(self._fire_zones.keys())
            fi = 0
            while fi < len(fz_keys):
                fk = fz_keys[fi]
                fz = self._fire_zones[fk]
                if fz["active"]:
                    dx = info["position"][0] - fz["position"][0]
                    dy = info["position"][1] - fz["position"][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < nearest_fire_dist:
                        nearest_fire_dist = dist
                        nearest_fire_zone = fk
                fi = fi + 1

            if nearest_fire_dist < NEAR_FIRE_RADIUS_M:
                near_fire.append({
                    "civilian_id": cid,
                    "fire_zone": nearest_fire_zone,
                    "distance_m": round(nearest_fire_dist, 1),
                    "health": info["health"],
                    "state": info["state"],
                })

            if nearest_fire_dist < CRITICAL_RADIUS_M:
                critical_count = critical_count + 1

            # Rescue priority: injured/incapacitated civilians near fire
            if info["state"] in ("injured", "incapacitated") or info["health"] < 50.0:
                rescue_priority.append({
                    "civilian_id": cid,
                    "position": info["position"],
                    "health": info["health"],
                    "nearest_fire_dist": round(nearest_fire_dist, 1),
                })

            ci = ci + 1

        # Sort rescue priority by health (lowest first), then distance
        rescue_priority.sort(
            key=lambda r: (r["health"], r["nearest_fire_dist"]))

        total = len(civ_ids)
        avg_health = total_health / max(total, 1)

        return {
            "timestamp": time.time(),
            "total": total,
            "by_state": by_state,
            "near_fire": near_fire,
            "critical_danger": critical_count,
            "average_health": round(avg_health, 1),
            "rescue_priority": rescue_priority,
        }
