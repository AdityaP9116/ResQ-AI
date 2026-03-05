#!/usr/bin/env python3
"""Generate flight_data.json with a smooth drone path through known zone positions.

Uses the actual YOLO detection data to anchor the drone at the correct zone at
the correct video time, with smooth Catmull-Rom interpolation between waypoints.

Video mapping: 301 frames @ 15fps, video frame N = sim step N+10
Normalized time: t = (step - 10) / 300

Known zone visits from YOLO data:
  FZ_1: YOLO steps 156-192, peak at 167  →  t_enter=0.487, t_focus=0.523, t_leave=0.607
  FZ_2: YOLO steps 204-219, peak at 210  →  t_enter=0.647, t_focus=0.667, t_leave=0.697
  FZ_0: YOLO steps 282-310, peak at 285  →  t_enter=0.907, t_focus=0.917, t_leave=1.000
"""

import json
import math
import os
import numpy as np

STEP_START = 10
STEP_END = 310
SURVEY_ALT = 110.0

# Fire zone world positions
FIRE_ZONES = {
    "FZ_0": [-94.80, -16.50],
    "FZ_1": [-98.63, 46.50],
    "FZ_2": [-1.79, 46.50],
    "FZ_3": [87.55, -16.50],
    "FZ_4": [79.75, 46.50],
}

# The drone orbit center and starting position
ORBIT_CENTER = [0.0, 15.0]
ORBIT_RADIUS = 55.0
START_POS = [0.0, 15.0]  # x=0, y=15 (center of orbit)


def orbit_pos(angle):
    """Position on the orbit circle."""
    return [
        ORBIT_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
        ORBIT_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
    ]


def catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom spline interpolation between p1 and p2."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
    y = 0.5 * ((2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
    return [x, y]


def main():
    # ── Define waypoints with normalized time ──
    # The drone orbits during steps 10-~140 (t=0.0 to ~0.43), then
    # visits FZ_1, transits to FZ_2, transits to FZ_0.

    # Build orbit portion (t = 0 to 0.45)
    orbit_speed = 0.025  # rad per step
    orbit_steps = 130     # steps 10-140 in orbit
    orbit_waypoints = []
    for i in range(0, orbit_steps + 1, 5):  # every 5 steps
        angle = i * orbit_speed
        pos = orbit_pos(angle)
        t = i / 300  # normalized time
        orbit_waypoints.append({"t": round(t, 4), "x": round(pos[0], 2), "y": round(pos[1], 2)})

    # Zone visit waypoints (from YOLO data)
    zone_visits = [
        # Pre-approach: drone leaves orbit, heading toward FZ_1
        {"t": 0.433, "x": -60.0, "y": 46.0},  # step ~140, heading to FZ_1

        # FZ_1 approach and focus
        {"t": 0.487, "x": -95.0, "y": 46.5},   # step 156, entering FZ_1
        {"t": 0.523, "x": -98.63, "y": 46.50},  # step 167, FZ_1 peak
        {"t": 0.570, "x": -98.0, "y": 46.5},    # step 181, still at FZ_1
        {"t": 0.607, "x": -96.0, "y": 46.5},    # step 192, leaving FZ_1

        # Transit FZ_1 → FZ_2
        {"t": 0.630, "x": -60.0, "y": 46.5},    # step 199, transiting east
        {"t": 0.647, "x": -20.0, "y": 46.5},    # step 204, entering FZ_2

        # FZ_2 focus
        {"t": 0.667, "x": -1.79, "y": 46.50},   # step 210, FZ_2 peak
        {"t": 0.697, "x": -1.79, "y": 46.50},   # step 219, leaving FZ_2

        # Transit FZ_2 → FZ_0
        {"t": 0.730, "x": -10.0, "y": 35.0},    # step 229, heading southwest
        {"t": 0.790, "x": -40.0, "y": 10.0},    # step 247
        {"t": 0.850, "x": -70.0, "y": -8.0},    # step 265
        {"t": 0.907, "x": -90.0, "y": -15.0},   # step 282, entering FZ_0

        # FZ_0 focus
        {"t": 0.917, "x": -94.80, "y": -16.50},  # step 285, FZ_0 peak
        {"t": 0.960, "x": -94.80, "y": -16.50},  # step 298, still at FZ_0
        {"t": 1.000, "x": -94.80, "y": -16.50},  # step 310, end
    ]

    # Merge orbit + zone waypoints
    all_waypoints = orbit_waypoints + zone_visits

    # Sort by time and deduplicate
    all_waypoints.sort(key=lambda w: w["t"])
    clean = [all_waypoints[0]]
    for w in all_waypoints[1:]:
        if w["t"] > clean[-1]["t"] + 0.001:
            clean.append(w)
    all_waypoints = clean

    # ── Interpolate to every step ──
    full_positions = []
    wp_idx = 0

    for step in range(STEP_START, STEP_END + 1):
        t = (step - STEP_START) / (STEP_END - STEP_START)

        # Find enclosing waypoints
        while wp_idx < len(all_waypoints) - 1 and all_waypoints[wp_idx + 1]["t"] < t:
            wp_idx += 1

        if wp_idx >= len(all_waypoints) - 1:
            # Past last waypoint
            pos = [all_waypoints[-1]["x"], all_waypoints[-1]["y"]]
        else:
            w0 = all_waypoints[max(0, wp_idx - 1)]
            w1 = all_waypoints[wp_idx]
            w2 = all_waypoints[min(wp_idx + 1, len(all_waypoints) - 1)]
            w3 = all_waypoints[min(wp_idx + 2, len(all_waypoints) - 1)]

            dt = w2["t"] - w1["t"]
            if dt < 0.0001:
                local_t = 0
            else:
                local_t = min(1.0, max(0.0, (t - w1["t"]) / dt))

            pos = catmull_rom(
                [w0["x"], w0["y"]],
                [w1["x"], w1["y"]],
                [w2["x"], w2["y"]],
                [w3["x"], w3["y"]],
                local_t,
            )

        full_positions.append({
            "t": round(t, 4),
            "x": round(pos[0], 2),
            "y": round(pos[1], 2),
        })

    # ── Zone timeline ──
    zone_timeline = [
        {
            "zone": "FZ_1",
            "enterStep": 156, "focusStep": 167, "leaveStep": 192,
            "enterT": 0.487, "focusT": 0.523, "leaveT": 0.607,
        },
        {
            "zone": "FZ_2",
            "enterStep": 204, "focusStep": 210, "leaveStep": 219,
            "enterT": 0.647, "focusT": 0.667, "leaveT": 0.697,
        },
        {
            "zone": "FZ_0",
            "enterStep": 282, "focusStep": 285, "leaveStep": 310,
            "enterT": 0.907, "focusT": 0.917, "leaveT": 1.000,
        },
    ]

    # ── Thin path for frontend (~100 points) ──
    key_steps = {STEP_START, STEP_END}
    for zt in zone_timeline:
        key_steps.update([zt["enterStep"], zt["focusStep"], zt["leaveStep"]])

    thin_path = []
    for i, fp in enumerate(full_positions):
        step = STEP_START + i
        if step in key_steps or i % 3 == 0:
            thin_path.append(fp)
    # Ensure endpoints
    if thin_path[-1]["t"] < 1.0:
        thin_path.append(full_positions[-1])

    result = {
        "zone_timeline": zone_timeline,
        "flight_path": thin_path,
        "full_positions": full_positions,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output_1", "flight_data.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved flight data to {out_path}")

    print(f"\nZone timeline:")
    for zt in zone_timeline:
        print(f"  {zt['zone']}: t={zt['enterT']:.3f} → {zt['focusT']:.3f} → {zt['leaveT']:.3f}")

    print(f"\nFlight path: {len(thin_path)} waypoints")
    for step in [10, 50, 100, 140, 156, 167, 192, 204, 210, 219, 250, 282, 285, 310]:
        idx = step - STEP_START
        if 0 <= idx < len(full_positions):
            p = full_positions[idx]
            print(f"  Step {step} (t={p['t']:.3f}): ({p['x']:8.2f}, {p['y']:8.2f})")


if __name__ == "__main__":
    main()
