"""Fire simulation manager for ResQ-AI Isaac Sim pipeline.

Discovers /World/FireZones/FZ_* prims, creates OmniFlow emitters by
referencing Flow .usda presets, simulates fire spread, and provides
structured fire reports.

Runs inside Isaac Sim 5.1 Script Editor.
Isaac Sim 5.1 compatibility: NO f-strings.  importlib for pxr.
"""

import importlib
import math
import time
import os

# pxr imports via importlib (Isaac Sim 5.1 requirement)
Usd = importlib.import_module("pxr.Usd")
UsdGeom = importlib.import_module("pxr.UsdGeom")
Sdf = importlib.import_module("pxr.Sdf")
Gf = importlib.import_module("pxr.Gf")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IGNITE_DELAY_S = 2.0          # seconds before first fire ignites
SPREAD_CHECK_INTERVAL_S = 2.0 # seconds between spread checks
SPREAD_RANGE_FACTOR = 100.0   # metres * intensity = spread reach (zones are 63-97m apart)
INTENSITY_GROWTH = 0.15       # per tick
MAX_INTENSITY = 1.5
WIND_DIRECTION = [1.0, 0.3]   # default wind (normalised internally)
WIND_INFLUENCE = 0.5          # how much wind biases spread probability
INITIAL_FIRE_ZONE = 1         # FZ_1 ignites first


# ---------------------------------------------------------------------------
# FireManager
# ---------------------------------------------------------------------------

class FireManager(object):
    """Manages fire zone lifecycle inside the Isaac Sim stage.

    Usage (Script Editor)::

        import importlib
        mod = importlib.import_module("sim_bridge.fire_system")
        mgr = mod.FireManager()
        mgr.start()
        # later ...
        print(mgr.get_fire_report())
        mgr.stop()
    """

    def __init__(self):
        self._zones = {}        # zone_name -> {...attrs...}
        self._active = {}       # zone_name -> {start_time, intensity, ...}
        self._start_time = None
        self._sub = None        # timeline subscription
        self._last_spread_check = 0.0
        self._started = False
        self._on_ignite_cb = None  # optional callback(zone_name, zone_index) called on ignition

        # Normalise wind
        wx, wy = WIND_DIRECTION
        wlen = math.sqrt(wx * wx + wy * wy) or 1.0
        self._wind = [wx / wlen, wy / wlen]

        self._discover_zones()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_zones(self):
        """Read all /World/FireZones/FZ_* prims from the current stage."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[FireManager] WARNING: no stage open")
            return

        root = stage.GetPrimAtPath("/World/FireZones")
        if not root.IsValid():
            print("[FireManager] WARNING: /World/FireZones not found")
            return

        children = root.GetChildren()
        idx = 0
        while idx < len(children):
            prim = children[idx]
            name = prim.GetName()
            if name.startswith("FZ_"):
                xf = UsdGeom.Xformable(prim)
                pos = [0.0, 0.0, 0.0]
                ops = xf.GetOrderedXformOps()
                oi = 0
                while oi < len(ops):
                    op = ops[oi]
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        v = op.Get()
                        pos = [float(v[0]), float(v[1]), float(v[2])]
                    oi = oi + 1

                radius_attr = prim.GetAttribute("resqai:fire_radius")
                intensity_attr = prim.GetAttribute("resqai:fire_intensity")
                spread_attr = prim.GetAttribute("resqai:fire_spread_rate")

                # Read flow preset paths from child prims
                fire_preset = ""
                smoke_preset = ""
                fire_child = stage.GetPrimAtPath(
                    str(prim.GetPath()) + "/FlowFireEmitter")
                if fire_child.IsValid():
                    fp = fire_child.GetAttribute("resqai:flow_preset_path")
                    if fp.IsValid():
                        fire_preset = str(fp.Get())
                smoke_child = stage.GetPrimAtPath(
                    str(prim.GetPath()) + "/FlowSmokeEmitter")
                if smoke_child.IsValid():
                    sp = smoke_child.GetAttribute("resqai:flow_preset_path")
                    if sp.IsValid():
                        smoke_preset = str(sp.Get())

                self._zones[name] = {
                    "prim_path": str(prim.GetPath()),
                    "position": pos,
                    "radius": float(radius_attr.Get()) if radius_attr.IsValid() else 5.0,
                    "initial_intensity": float(intensity_attr.Get()) if intensity_attr.IsValid() else 0.7,
                    "spread_rate": float(spread_attr.Get()) if spread_attr.IsValid() else 0.5,
                    "fire_preset": fire_preset,
                    "smoke_preset": smoke_preset,
                }
                print("[FireManager] Discovered zone " + name +
                      " at " + str(pos))
            idx = idx + 1

        print("[FireManager] Total zones: " + str(len(self._zones)))

    # ------------------------------------------------------------------
    # Ignition
    # ------------------------------------------------------------------

    def _ignite_zone(self, zone_name):
        """Activate a fire zone: set prim attribute and add Flow references."""
        if zone_name in self._active:
            return  # already burning

        info = self._zones.get(zone_name)
        if info is None:
            print("[FireManager] Unknown zone: " + zone_name)
            return

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        prim = stage.GetPrimAtPath(info["prim_path"])
        if prim.IsValid():
            active_attr = prim.GetAttribute("resqai:fire_active")
            if active_attr.IsValid():
                active_attr.Set(True)

        # Create Flow fire + smoke effect via omni.kit.commands
        self._add_flow_effect(stage, zone_name, info["prim_path"],
                              info["fire_preset"], info["smoke_preset"])

        self._active[zone_name] = {
            "start_time": time.time(),
            "intensity": info["initial_intensity"],
            "radius": info["radius"],
            "spread_direction": list(self._wind),
        }

        print("[FireManager] IGNITED " + zone_name)

        # Notify external callback (e.g. to create visual Flow fire)
        if self._on_ignite_cb is not None:
            try:
                # Extract zone index from name like "FZ_2"
                zone_idx = int(zone_name.split("_")[1])
                self._on_ignite_cb(zone_name, zone_idx)
            except Exception as _cb_err:
                print("[FireManager] on_ignite callback error: " + str(_cb_err))

    def _resolve_preset_url(self, preset_path):
        """Find the actual file for a Flow preset path.

        The USDA scene stores paths like:
            ./assets/Particles/Assets/Extensions/Samples/Flow/presets/Fire/Fire.usda
        but the files may be under a versioned subfolder (e.g. 104/).

        Returns the resolved absolute path, or empty string.
        """
        if not preset_path:
            return ""

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))

        # 1) Exact relative path
        candidate = os.path.normpath(os.path.join(project_root, preset_path))
        if os.path.isfile(candidate):
            return candidate

        # 2) With version folder inserted before "presets/"
        for version in ("104", "105", "107"):
            versioned = preset_path.replace(
                "Flow/presets/", "Flow/" + version + "/presets/")
            candidate = os.path.normpath(
                os.path.join(project_root, versioned))
            if os.path.isfile(candidate):
                return candidate

        # 3) Isaac Sim extscache
        isaacsim_root = os.environ.get("ISAAC_SIM_PATH", "")
        if not isaacsim_root:
            # Try to find from pip-installed isaacsim
            try:
                import isaacsim
                isaacsim_root = os.path.dirname(isaacsim.__file__)
            except ImportError:
                isaacsim_root = ""
        extscache = os.path.join(isaacsim_root, "extscache") if isaacsim_root else ""
        if os.path.isdir(extscache):
            basename = os.path.basename(preset_path)
            parent = os.path.basename(os.path.dirname(preset_path))
            dlist = os.listdir(extscache)
            di = 0
            while di < len(dlist):
                if dlist[di].startswith("omni.flowusd"):
                    candidate = os.path.join(
                        extscache, dlist[di],
                        "data", "presets", parent, basename)
                    if os.path.isfile(candidate):
                        return candidate
                di = di + 1

        return ""

    def _add_flow_effect(self, stage, zone_name, prim_path, fire_preset, smoke_preset):
        """Create a Flow fire effect at the given zone using omni.kit.commands.

        Uses the FlowCreatePresets command from the omni.flowusd extension,
        which properly creates FlowEmitterSphere, FlowSimulate, FlowOffscreen,
        and FlowRender prims and handles up-axis Y->Z conversion.

        Fire size is scaled proportionally to the zone radius so that
        smaller zones produce visibly smaller fires.
        """
        try:
            import omni.kit.commands
        except ImportError:
            print("[FireManager] omni.kit.commands not available")
            return

        # Determine visual scale from zone radius (baseline = 7.0m)
        zone_info = self._zones.get(zone_name, {})
        zone_radius = zone_info.get("radius", 5.0)
        _BASELINE_RADIUS = 7.0
        fire_scale = max(0.35, zone_radius / _BASELINE_RADIUS)  # clamp so tiny fires are still visible

        # Resolve fire preset URL
        fire_url = self._resolve_preset_url(fire_preset)
        if not fire_url:
            # Try the omni.flowusd built-in preset as fallback
            try:
                from omni.flowusd.scripts.commands import get_preset_url
                fire_url = get_preset_url("Fire")
            except Exception:
                pass

        if fire_url:
            try:
                omni.kit.commands.execute(
                    "FlowCreatePresets",
                    preset_name="Fire",
                    paths=[prim_path],
                    url=fire_url,
                    layer=-1,
                )
                print("[FireManager] Created Flow fire at " +
                      prim_path + " from " + fire_url +
                      " (scale=" + str(round(fire_scale, 2)) + ")")

                # Scale the Flow emitter sphere to vary fire size
                self._scale_flow_emitters(stage, prim_path, fire_scale)

            except Exception as e:
                print("[FireManager] FlowCreatePresets fire error: " + str(e))
        else:
            print("[FireManager] WARNING: fire preset not found for " +
                  zone_name)

        # Resolve smoke preset URL (optional)
        smoke_url = self._resolve_preset_url(smoke_preset)
        if smoke_url:
            try:
                omni.kit.commands.execute(
                    "FlowCreatePresets",
                    preset_name="Smoke",
                    paths=[prim_path],
                    url=smoke_url,
                    layer=-1,
                )
                print("[FireManager] Created Flow smoke at " + prim_path)
            except Exception as e:
                print("[FireManager] FlowCreatePresets smoke error: " + str(e))

    # ------------------------------------------------------------------
    # Flow emitter scaling  (visual size variation per zone)
    # ------------------------------------------------------------------

    def _scale_flow_emitters(self, stage, prim_path, scale_factor):
        """Scale FlowEmitterSphere radius and coupleRate under prim_path.

        This makes smaller fire zones produce visibly smaller fires.
        We scale:
          - FlowEmitterSphere radius attribute (controls particle spawn volume)
          - coupleRate (emission density — lower for smaller fires)
        """
        try:
            root_prim = stage.GetPrimAtPath(prim_path)
            if not root_prim.IsValid():
                return
            for child in root_prim.GetAllChildren():
                cname = child.GetName()
                # Scale emitter sphere radius
                if "Emitter" in cname or "emitter" in cname:
                    rad = child.GetAttribute("radius")
                    if rad and rad.IsValid():
                        old_val = float(rad.Get())
                        rad.Set(old_val * scale_factor)
                    cr = child.GetAttribute("coupleRate")
                    if cr and cr.IsValid():
                        old_cr = float(cr.Get())
                        cr.Set(old_cr * scale_factor)
                # Also check grandchildren (Flow creates nested prims)
                for gchild in child.GetAllChildren():
                    gcname = gchild.GetName()
                    if "Emitter" in gcname or "emitter" in gcname:
                        rad2 = gchild.GetAttribute("radius")
                        if rad2 and rad2.IsValid():
                            old_val2 = float(rad2.Get())
                            rad2.Set(old_val2 * scale_factor)
                        cr2 = gchild.GetAttribute("coupleRate")
                        if cr2 and cr2.IsValid():
                            old_cr2 = float(cr2.Get())
                            cr2.Set(old_cr2 * scale_factor)
            if abs(scale_factor - 1.0) > 0.05:
                print("[FireManager] Scaled emitters at " + prim_path +
                      " by " + str(round(scale_factor, 2)))
        except Exception as _se:
            print("[FireManager] Flow scale error at " + prim_path + ": " + str(_se))

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def start(self):
        """Subscribe to timeline ticks and begin the fire simulation."""
        if self._started:
            return

        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        self._sub = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            omni.timeline.TimelineEventType.CURRENT_TIME_TICKED,
            self._on_tick,
        )
        self._start_time = time.time()
        self._started = True
        print("[FireManager] Started — first fire in " +
              str(IGNITE_DELAY_S) + "s")

    def stop(self):
        """Unsubscribe from timeline and deactivate."""
        if self._sub is not None:
            self._sub = None
        self._started = False
        print("[FireManager] Stopped")

    def _on_tick(self, event):
        """Called every simulation tick."""
        if not self._started:
            return

        now = time.time()
        elapsed = now - self._start_time

        # --- Initial ignition after delay --------------------------------
        zone_key = "FZ_" + str(INITIAL_FIRE_ZONE)
        if elapsed >= IGNITE_DELAY_S and zone_key not in self._active:
            self._ignite_zone(zone_key)

        # --- Grow active fires -------------------------------------------
        keys = list(self._active.keys())
        ki = 0
        while ki < len(keys):
            k = keys[ki]
            info = self._active[k]
            if info["intensity"] < MAX_INTENSITY:
                info["intensity"] = min(info["intensity"] + INTENSITY_GROWTH * 0.016,
                                        MAX_INTENSITY)
                # Grow radius proportionally
                info["radius"] = self._zones[k]["radius"] * (
                    1.0 + 0.3 * (info["intensity"] - self._zones[k]["initial_intensity"]))
            ki = ki + 1

        # --- Spread check ------------------------------------------------
        if now - self._last_spread_check >= SPREAD_CHECK_INTERVAL_S:
            self._last_spread_check = now
            self._check_spread()

    def _check_spread(self):
        """Check if any inactive zones should ignite based on proximity."""
        import random as _random

        active_keys = list(self._active.keys())
        all_keys = list(self._zones.keys())

        zi = 0
        while zi < len(all_keys):
            target = all_keys[zi]
            if target in self._active:
                zi = zi + 1
                continue

            target_pos = self._zones[target]["position"]

            ai = 0
            while ai < len(active_keys):
                src = active_keys[ai]
                src_pos = self._zones[src]["position"]
                src_intensity = self._active[src]["intensity"]

                dx = target_pos[0] - src_pos[0]
                dy = target_pos[1] - src_pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                reach = SPREAD_RANGE_FACTOR * src_intensity

                if dist < reach and dist > 0:
                    # Base probability inversely proportional to distance
                    prob = (1.0 - dist / reach) * self._zones[src]["spread_rate"]

                    # Wind bonus: if target is downwind, increase probability
                    dir_x = dx / dist
                    dir_y = dy / dist
                    dot = dir_x * self._wind[0] + dir_y * self._wind[1]
                    if dot > 0:
                        prob = prob + dot * WIND_INFLUENCE

                    prob = min(prob, 0.8)

                    if _random.random() < prob:
                        self._ignite_zone(target)
                        break

                ai = ai + 1
            zi = zi + 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_fire_report(self):
        """Return structured fire status report.

        Returns
        -------
        dict
            Keys: active_fires, total_area_burning_m2, spread_rate_m_per_min,
            estimated_containment_time_min
        """
        now = time.time()
        active_list = []
        total_area = 0.0

        keys = list(self._active.keys())
        ki = 0
        while ki < len(keys):
            zname = keys[ki]
            info = self._active[zname]
            zone = self._zones[zname]
            burning_s = now - info["start_time"]
            area = math.pi * info["radius"] * info["radius"]
            total_area = total_area + area

            active_list.append({
                "zone": zname,
                "position": zone["position"],
                "intensity": round(info["intensity"], 2),
                "radius": round(info["radius"], 2),
                "spread_direction": info["spread_direction"],
                "time_burning_seconds": round(burning_s, 1),
            })
            ki = ki + 1

        # Estimate spread rate (area growth per minute)
        spread_rate = 0.0
        if len(active_list) > 0:
            max_burn_time = max(
                [a["time_burning_seconds"] for a in active_list])
            if max_burn_time > 0:
                spread_rate = total_area / (max_burn_time / 60.0)

        # Rough containment estimate
        containment = 0.0
        if spread_rate > 0:
            # Assume containment takes 3x the current burn time
            containment = max_burn_time / 60.0 * 3.0

        return {
            "active_fires": active_list,
            "total_area_burning_m2": round(total_area, 1),
            "spread_rate_m_per_min": round(spread_rate, 2),
            "estimated_containment_time_min": round(containment, 1),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_fire_mgr = None


def get_fire_manager():
    """Return or create the global FireManager singleton."""
    global _fire_mgr
    if _fire_mgr is None:
        _fire_mgr = FireManager()
    return _fire_mgr
