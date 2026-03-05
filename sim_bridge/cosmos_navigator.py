"""Cosmos Reason 2 navigation brain for ResQ-AI.

Integrates with a Cosmos / OpenAI-compatible VLM endpoint to make
high-level navigation decisions for the disaster-response drone.
Falls back to a deterministic waypoint patrol when the VLM is unavailable.

Isaac Sim 5.1 compatibility: NO f-strings.  importlib for pxr.
"""

import importlib
import json
import math
import os
import time
import base64
import threading

import numpy as np

# pxr via importlib
Gf = importlib.import_module("pxr.Gf")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load .env values if available
_env_url = os.environ.get("RESQAI_VLLM_URL", "")
_env_model = os.environ.get("RESQAI_COSMOS_MODEL", "")

COSMOS_API_URL = _env_url or "http://localhost:8000/v1/chat/completions"
COSMOS_MODEL = _env_model or "nvidia/Cosmos-Reason1-7B"
DECISION_INTERVAL = 2.0   # seconds between Cosmos calls

# PD controller gains
KP = 2.0
KD = 0.5
MAX_SPEED = 15.0  # m/s


# ---------------------------------------------------------------------------
# CosmosNavigator
# ---------------------------------------------------------------------------

class CosmosNavigator(object):
    """VLM-driven drone navigation with PD waypoint following.

    Parameters
    ----------
    api_url : str
        OpenAI-compatible chat completions endpoint.
    model : str
        Model identifier string.
    decision_interval : float
        Minimum seconds between VLM queries.
    """

    def __init__(self, api_url=None, model=None, decision_interval=None):
        self._api_url = api_url or COSMOS_API_URL
        self._model = model or COSMOS_MODEL
        self._interval = decision_interval or DECISION_INTERVAL

        self._last_decision_time = 0.0
        self._current_waypoint = None
        self._prev_velocity_error = np.zeros(3)
        self._decision_log = []
        self._patrol_index = 0
        self._patrol_waypoints = []
        self._flight_bounds = None

        # Background VLM result
        self._pending_result = None
        self._pending_lock = threading.Lock()

        # Read patrol waypoints and flight boundary from stage
        self._load_scene_data()

        print("[CosmosNav] Initialised — endpoint: " + self._api_url)
        print("[CosmosNav] Model: " + self._model)

    # ------------------------------------------------------------------
    # Scene data
    # ------------------------------------------------------------------

    def _load_scene_data(self):
        """Read waypoints and flight boundary from the USD stage."""
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            # Waypoints
            wp_root = stage.GetPrimAtPath("/World/DroneOps/Waypoints")
            if wp_root.IsValid():
                children = wp_root.GetChildren()
                ci = 0
                while ci < len(children):
                    child = children[ci]
                    xf = importlib.import_module("pxr.UsdGeom").Xformable(child)
                    ops = xf.GetOrderedXformOps()
                    oi = 0
                    while oi < len(ops):
                        op = ops[oi]
                        if op.GetOpType() == importlib.import_module("pxr.UsdGeom").XformOp.TypeTranslate:
                            v = op.Get()
                            self._patrol_waypoints.append(
                                [float(v[0]), float(v[1]), float(v[2])])
                        oi = oi + 1
                    ci = ci + 1

            # Flight boundary
            fb = stage.GetPrimAtPath("/World/DroneOps/FlightBoundary")
            if fb.IsValid():
                mn = fb.GetAttribute("resqai:min_bound")
                mx = fb.GetAttribute("resqai:max_bound")
                if mn.IsValid() and mx.IsValid():
                    mn_v = mn.Get()
                    mx_v = mx.Get()
                    self._flight_bounds = {
                        "min": [float(mn_v[0]), float(mn_v[1]), float(mn_v[2])],
                        "max": [float(mx_v[0]), float(mx_v[1]), float(mx_v[2])],
                    }
        except Exception as e:
            print("[CosmosNav] Scene data load error: " + str(e))

        if not self._patrol_waypoints:
            # Default fallback patrol
            self._patrol_waypoints = [
                [0, 0, 50], [50, 30, 40], [50, -30, 35],
                [-50, -30, 35], [-50, 30, 40], [0, 0, 25],
            ]

        print("[CosmosNav] Patrol waypoints: " + str(len(self._patrol_waypoints)))

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------

    def get_decision(self, rgb_frame, detections, fire_report,
                     civilian_summary, drone_pos, battery_pct):
        """Query Cosmos (or fallback) and return a navigation decision.

        Parameters
        ----------
        rgb_frame : np.ndarray
            Current camera frame (H, W, 3) uint8 BGR.
        detections : list[dict]
            Output from DualYOLODetector.detect().
        fire_report : dict
            Output from FireManager.get_fire_report().
        civilian_summary : dict
            Output from CivilianTracker.get_civilian_report().
        drone_pos : list or np.ndarray
            [x, y, z] current drone position.
        battery_pct : float
            Battery level 0-100.

        Returns
        -------
        dict
            Keys: next_waypoint, priority, urgency_level, reasoning,
            recommended_actions
        """
        now = time.time()

        # Check if enough time has passed
        if now - self._last_decision_time < self._interval:
            # Return patrol waypoint
            return self._patrol_decision(drone_pos)

        self._last_decision_time = now

        # Check for pending async result
        with self._pending_lock:
            if self._pending_result is not None:
                result = self._pending_result
                self._pending_result = None
                self._current_waypoint = result.get("next_waypoint")
                self._decision_log.append({
                    "time": now,
                    "action": result.get("priority", "unknown"),
                    "reasoning": result.get("reasoning", ""),
                    "waypoint": result.get("next_waypoint", [0, 0, 50]),
                })
                return result

        # Fire off async VLM query
        self._query_cosmos_async(rgb_frame, detections, fire_report,
                                 civilian_summary, drone_pos, battery_pct)

        # Return patrol while waiting
        return self._patrol_decision(drone_pos)

    def _patrol_decision(self, drone_pos):
        """Return next patrol waypoint when Cosmos is unavailable."""
        if not self._patrol_waypoints:
            return {
                "next_waypoint": [0, 0, 50],
                "priority": "patrol",
                "urgency_level": "low",
                "reasoning": "No waypoints configured, hovering at origin",
                "recommended_actions": ["Configure waypoints"],
            }

        wp = self._patrol_waypoints[self._patrol_index]
        if drone_pos is not None:
            dist = math.sqrt(
                (wp[0] - drone_pos[0]) ** 2 +
                (wp[1] - drone_pos[1]) ** 2 +
                (wp[2] - drone_pos[2]) ** 2
            )
            if dist < 3.0:
                self._patrol_index = (self._patrol_index + 1) % len(
                    self._patrol_waypoints)
                wp = self._patrol_waypoints[self._patrol_index]

        return {
            "next_waypoint": wp,
            "priority": "patrol",
            "urgency_level": "low",
            "reasoning": "Patrolling waypoint " + str(self._patrol_index),
            "recommended_actions": [],
        }

    # ------------------------------------------------------------------
    # Cosmos VLM call
    # ------------------------------------------------------------------

    def _query_cosmos_async(self, rgb_frame, detections, fire_report,
                            civilian_summary, drone_pos, battery_pct):
        """Fire off Cosmos query in a background thread."""
        t = threading.Thread(
            target=self._do_cosmos_query,
            args=(rgb_frame, detections, fire_report,
                  civilian_summary, drone_pos, battery_pct),
            daemon=True,
        )
        t.start()

    def _do_cosmos_query(self, rgb_frame, detections, fire_report,
                         civilian_summary, drone_pos, battery_pct):
        """Blocking Cosmos VLM call (runs in background thread)."""
        try:
            import requests
        except ImportError:
            print("[CosmosNav] requests library not available")
            return

        try:
            # Encode image
            image_b64 = ""
            if rgb_frame is not None:
                try:
                    import cv2
                    _, buf = cv2.imencode(".jpg", rgb_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 60])
                    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                except Exception:
                    pass

            # Build prompt
            prompt_text = self._build_prompt(detections, fire_report,
                                             civilian_summary, drone_pos,
                                             battery_pct)

            # Build messages
            messages = [
                {"role": "system", "content": (
                    "You are an autonomous disaster response drone AI. "
                    "Analyze the scene and sensor data. Respond ONLY in "
                    "valid JSON with fields: next_waypoint (list of 3 floats), "
                    "priority (string), urgency_level (low/medium/high/critical), "
                    "reasoning (string), recommended_actions (list of strings)."
                )},
            ]

            user_content = []
            if image_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_b64,
                    },
                })
            user_content.append({"type": "text", "text": prompt_text})

            messages.append({"role": "user", "content": user_content})

            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }

            resp = requests.post(
                self._api_url,
                json=payload,
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data.get("choices", [{}])[0].get(
                    "message", {}).get("content", "")
                result = self._parse_result(text)
                if result is not None:
                    # Clamp waypoint to flight boundary
                    wp = result.get("next_waypoint", [0, 0, 50])
                    result["next_waypoint"] = self._clamp_waypoint(wp)
                    with self._pending_lock:
                        self._pending_result = result
            else:
                print("[CosmosNav] VLM returned status " + str(resp.status_code))

        except Exception as e:
            print("[CosmosNav] VLM query error: " + str(e))

    def _build_prompt(self, detections, fire_report, civilian_summary,
                      drone_pos, battery_pct):
        """Construct the structured text prompt for Cosmos."""
        # Count detections
        people_count = 0
        fire_count = 0
        di = 0
        while di < len(detections):
            d = detections[di]
            if d["class"] == "person":
                people_count = people_count + 1
            elif d["class"] == "fire":
                fire_count = fire_count + 1
            di = di + 1

        lines = []
        lines.append("Current detections: " + str(people_count) +
                      " people detected, " + str(fire_count) + " active fires.")

        # Fire zones
        if fire_report and fire_report.get("active_fires"):
            lines.append("Fire zones:")
            fi = 0
            fires = fire_report["active_fires"]
            while fi < len(fires):
                f = fires[fi]
                lines.append("  - " + str(f["zone"]) + " at " +
                             str(f["position"]) + " intensity=" +
                             str(f["intensity"]))
                fi = fi + 1

        # Civilian status
        if civilian_summary:
            by_state = civilian_summary.get("by_state", {})
            state_parts = []
            state_keys = list(by_state.keys())
            si = 0
            while si < len(state_keys):
                sk = state_keys[si]
                state_parts.append(str(by_state[sk]) + " " + str(sk))
                si = si + 1
            lines.append("Civilian status: " + ", ".join(state_parts))
            lines.append("Critical danger: " + str(
                civilian_summary.get("critical_danger", 0)))

        # Drone telemetry
        dp = drone_pos if drone_pos is not None else [0, 0, 50]
        lines.append("Drone position: (" + str(round(dp[0], 1)) + ", " +
                      str(round(dp[1], 1)) + ", " + str(round(dp[2], 1)) +
                      "), battery: " + str(round(battery_pct, 1)) + "%")

        lines.append("")
        lines.append("Based on the visual scene and data, determine:")
        lines.append("1. Which area needs immediate attention?")
        lines.append("2. Where should the drone fly next? Give waypoint as [x, y, z]")
        lines.append("3. Priority: rescue (injured civilians) or recon (map fire spread)?")
        lines.append("4. Should emergency services be alerted? What urgency level?")

        return "\n".join(lines)

    def _parse_result(self, text):
        """Parse JSON from Cosmos response text."""
        if not text:
            return None

        # Try direct JSON parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code fence
        start = text.find("```json")
        if start >= 0:
            start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end])
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try finding first { ... }
        brace_start = text.find("{")
        if brace_start >= 0:
            brace_end = text.rfind("}")
            if brace_end > brace_start:
                try:
                    return json.loads(text[brace_start:brace_end + 1])
                except (json.JSONDecodeError, ValueError):
                    pass

        print("[CosmosNav] Could not parse VLM response")
        return None

    # ------------------------------------------------------------------
    # PD Controller
    # ------------------------------------------------------------------

    def compute_velocity(self, current_pos, dt=0.016):
        """Compute velocity command using PD controller toward current waypoint.

        Parameters
        ----------
        current_pos : list or np.ndarray
            [x, y, z] current drone position.
        dt : float
            Timestep in seconds.

        Returns
        -------
        np.ndarray
            [vx, vy, vz] velocity command (m/s).
        """
        if self._current_waypoint is None:
            if self._patrol_waypoints:
                self._current_waypoint = self._patrol_waypoints[
                    self._patrol_index]
            else:
                return np.zeros(3)

        target = np.array(self._current_waypoint, dtype=np.float64)
        current = np.array(current_pos, dtype=np.float64)

        error = target - current
        d_error = (error - self._prev_velocity_error) / max(dt, 0.001)
        self._prev_velocity_error = error.copy()

        velocity = KP * error + KD * d_error

        # Clamp speed
        speed = np.linalg.norm(velocity)
        if speed > MAX_SPEED:
            velocity = velocity * (MAX_SPEED / speed)

        return velocity

    def _clamp_waypoint(self, wp):
        """Clamp a waypoint within flight boundaries."""
        if self._flight_bounds is None:
            return wp

        mn = self._flight_bounds["min"]
        mx = self._flight_bounds["max"]
        return [
            max(mn[0], min(mx[0], wp[0])),
            max(mn[1], min(mx[1], wp[1])),
            max(mn[2], min(mx[2], wp[2])),
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_decision_log(self):
        """Return list of all logged decisions."""
        return list(self._decision_log)

    def get_current_waypoint(self):
        """Return the current target waypoint or None."""
        return self._current_waypoint
