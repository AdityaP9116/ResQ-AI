#!/usr/bin/env python3
"""Generate accurate flight_data.json by replaying the DroneController
with mode switches derived from the Cosmos reasoning log.

Known facts from the actual simulation:
- Cosmos Step 41:  state=focus,   target=FZ_1, distance=62m
- Cosmos Step 47:  state=focus,   target=FZ_1, distance=62m
- Cosmos Step 122: state=survey,  target=FZ_1, distance=82m
- Cosmos Step 128: state=survey,  target=FZ_1, distance=92m
- Cosmos Step 174: state=focus,   target=FZ_1, distance=5m   (AT FZ_1)
- Cosmos Step 254: state=survey,  target=FZ_2, distance=62m
- Cosmos Step 279: state=approach, target=FZ_2, distance=52m
- Cosmos Step 290: state=approach, target=FZ_2, distance=13m  (near FZ_2)

Fire detections:
- Steps 156-192: drone near FZ_1 (peak at step 167)
- Steps 282-310: drone near FZ_2 (peak at step 285)
"""

import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sim_bridge.drone_controller import DroneController

# Fire zone positions
FZ_1 = np.array([-98.63, 46.5, 110.0])
FZ_2 = np.array([-1.79, 46.5, 110.0])
FZ_0 = np.array([-94.80, -16.5, 110.0])

START_POS = [0.0, 15.0, 110.0]
ORBIT_CENTER = [0.0, 15.0]
ORBIT_RADIUS = 55.0
ORBIT_SPEED = 0.025

NUM_STEPS = 301  # steps 10..310
WARMUP = 10

def dist_2d(a, b):
    return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

def run_simulation():
    """Replay DroneController with mode switches matching the Cosmos log."""
    ctrl = DroneController(
        START_POS,
        max_vel=4.5,
        max_accel=0.6,
        drag=0.04,
        hover_drift=0.12,
        wind_strength=0.25,
        altitude=110.0,
    )
    ctrl.configure_orbit(ORBIT_CENTER, radius=ORBIT_RADIUS, speed=ORBIT_SPEED)
    ctrl.start_orbit()

    # We need to calibrate mode-switch steps to match Cosmos distances.
    # The sim state machine switches based on Cosmos VLM decisions.
    # From the log, the sequence is:
    #
    # Phase 1: Orbit → Focus on FZ_1 → Orbit (survey) → Focus on FZ_1 → Orbit
    # Phase 2: Orbit → Approach FZ_2
    #
    # Mode switch estimates (will be tuned):
    SWITCHES = [
        # (step, action, target, speed_factor)
        (10,  'orbit', None, 1.0),          # Start orbiting
        (35,  'waypoint', FZ_1, 1.0),       # Cosmos: focus on FZ_1 (at step 41: 62m)
        (90,  'orbit', None, 1.0),          # Back to survey after first pass
        (145, 'waypoint', FZ_1, 0.6),       # Focus on FZ_1 again (step 174: 5m)
        (195, 'orbit', None, 1.0),          # Leave FZ_1, resume survey
        (260, 'waypoint', FZ_2, 1.0),       # Approach FZ_2 (step 254 survey→approach)
    ]
    
    positions = []
    switch_idx = 0
    
    for step in range(WARMUP, WARMUP + NUM_STEPS):
        # Apply mode switches
        while switch_idx < len(SWITCHES) and step >= SWITCHES[switch_idx][0]:
            _, action, target, spd = SWITCHES[switch_idx]
            if action == 'orbit':
                ctrl.start_orbit()
                ctrl._target_speed_factor = spd
            elif action == 'waypoint':
                ctrl.go_to(target, speed_factor=spd)
            switch_idx += 1
        
        # Slow down near fire zones
        d_fz1 = dist_2d(ctrl.pos, FZ_1)
        d_fz2 = dist_2d(ctrl.pos, FZ_2)
        if ctrl._mode == 'waypoint':
            min_d = min(d_fz1, d_fz2)
            if min_d < 15:
                ctrl.slow_down(0.25)
            elif min_d < 30:
                ctrl.slow_down(0.4)
        
        pos = ctrl.step()
        positions.append(pos.copy())
    
    return positions

def validate_and_report(positions):
    """Check simulated distances against Cosmos log."""
    # Cosmos log constraints: (step, zone_pos, expected_distance)
    constraints = [
        (41, FZ_1, 62),
        (47, FZ_1, 62),
        (122, FZ_1, 82),
        (128, FZ_1, 92),
        (174, FZ_1, 5),
        (254, FZ_2, 62),
        (279, FZ_2, 52),
        (290, FZ_2, 13),
    ]
    
    print("\nValidation against Cosmos reasoning log:")
    print("-" * 60)
    total_err = 0
    for step, target, expected_dist in constraints:
        idx = step - WARMUP
        if 0 <= idx < len(positions):
            actual_dist = dist_2d(positions[idx], target)
            err = abs(actual_dist - expected_dist)
            total_err += err
            mark = "✓" if err < 15 else "✗"
            print(f"  Step {step}: expected={expected_dist}m, actual={actual_dist:.1f}m, err={err:.1f}m {mark}")
    print(f"\n  Mean error: {total_err / len(constraints):.1f}m")
    
    # Also check fire detection steps
    fire_steps = [167, 210, 285]
    print("\nFire detection step positions:")
    for step in fire_steps:
        idx = step - WARMUP
        if 0 <= idx < len(positions):
            pos = positions[idx]
            d1 = dist_2d(pos, FZ_1)
            d2 = dist_2d(pos, FZ_2)
            nearest = "FZ_1" if d1 < d2 else "FZ_2"
            print(f"  Step {step}: pos=({pos[0]:.1f}, {pos[1]:.1f}) dist_FZ_1={d1:.1f}m dist_FZ_2={d2:.1f}m nearest={nearest}")

def build_flight_data(positions):
    """Build flight_data.json from simulated positions."""
    # Zone timeline based on YOLO fire detections:
    # FZ_1: steps 156-192 (drone is actually near FZ_1)
    # FZ_2 intermediate: steps 204-219 (fire visible during transit)  
    # FZ_0 label (actually near FZ_2): steps 282-310
    
    # Verify actual proximity at these steps
    fz1_peak_idx = 167 - WARMUP
    fz0_peak_idx = 285 - WARMUP  # labeled FZ_0 but actually near FZ_2
    
    fz1_dist = dist_2d(positions[fz1_peak_idx], FZ_1) if fz1_peak_idx < len(positions) else 999
    fz2_at_204 = dist_2d(positions[204 - WARMUP], FZ_2) if (204 - WARMUP) < len(positions) else 999
    fz2_at_285 = dist_2d(positions[fz0_peak_idx], FZ_2) if fz0_peak_idx < len(positions) else 999
    
    print(f"\nZone proximity check:")
    print(f"  Step 167 (FZ_1 peak): {fz1_dist:.1f}m from FZ_1")
    print(f"  Step 204 (FZ_2 transit): {fz2_at_204:.1f}m from FZ_2")
    print(f"  Step 285 ('FZ_0' peak): {fz2_at_285:.1f}m from FZ_2")
    
    # Zone timeline: Keep the pipeline's zone labels (FZ_1, FZ_2, FZ_0)
    # but ensure the drone POSITIONS match reality
    zone_timeline = [
        {
            "zone": "FZ_1",
            "enterStep": 156, "focusStep": 167, "leaveStep": 192,
            "enterT": (156 - WARMUP) / 300, "focusT": (167 - WARMUP) / 300,
            "leaveT": (192 - WARMUP) / 300,
        },
        {
            "zone": "FZ_2",
            "enterStep": 204, "focusStep": 210, "leaveStep": 219,
            "enterT": (204 - WARMUP) / 300, "focusT": (210 - WARMUP) / 300,
            "leaveT": (219 - WARMUP) / 300,
        },
        {
            "zone": "FZ_0",
            "enterStep": 282, "focusStep": 285, "leaveStep": 310,
            "enterT": (282 - WARMUP) / 300, "focusT": (285 - WARMUP) / 300,
            "leaveT": (310 - WARMUP) / 300,
        },
    ]
    
    # Flight path: sample every 3rd step for compactness
    flight_path = []
    for i in range(0, len(positions), 3):
        t = i / 300
        flight_path.append({
            "t": round(t, 4),
            "x": round(float(positions[i][0]), 2),
            "y": round(float(positions[i][1]), 2),
        })
    # Ensure t=1.0 endpoint
    if flight_path[-1]["t"] < 1.0:
        flight_path.append({
            "t": 1.0,
            "x": round(float(positions[-1][0]), 2),
            "y": round(float(positions[-1][1]), 2),
        })
    
    # Full positions (every step)
    full_positions = []
    for i, pos in enumerate(positions):
        t = i / 300
        full_positions.append({
            "t": round(t, 4),
            "x": round(float(pos[0]), 2),
            "y": round(float(pos[1]), 2),
        })
    
    return {
        "zone_timeline": zone_timeline,
        "flight_path": flight_path,
        "full_positions": full_positions,
    }


def iterative_tune():
    """Try different switch timings and find the best fit."""
    
    best_err = float('inf')
    best_switches = None
    best_positions = None
    
    # Grid search over key switch parameters
    for first_focus_step in [30, 33, 35, 37, 40]:
        for first_orbit_return in [80, 85, 90, 95, 100]:
            for second_focus_step in [135, 140, 145, 150, 155]:
                for orbit_after_fz1 in [190, 195, 200, 205]:
                    for approach_fz2 in [255, 258, 260, 263, 265]:
                        
                        ctrl = DroneController(
                            START_POS,
                            max_vel=4.5, max_accel=0.6, drag=0.04,
                            hover_drift=0.12, wind_strength=0.25, altitude=110.0,
                        )
                        ctrl.configure_orbit(ORBIT_CENTER, radius=ORBIT_RADIUS, speed=ORBIT_SPEED)
                        ctrl.start_orbit()
                        
                        switches = [
                            (10,  'orbit', None, 1.0),
                            (first_focus_step, 'waypoint', FZ_1, 1.0),
                            (first_orbit_return, 'orbit', None, 1.0),
                            (second_focus_step, 'waypoint', FZ_1, 0.6),
                            (orbit_after_fz1, 'orbit', None, 1.0),
                            (approach_fz2, 'waypoint', FZ_2, 1.0),
                        ]
                        
                        positions = []
                        switch_idx = 0
                        
                        for step in range(WARMUP, WARMUP + NUM_STEPS):
                            while switch_idx < len(switches) and step >= switches[switch_idx][0]:
                                _, action, target, spd = switches[switch_idx]
                                if action == 'orbit':
                                    ctrl.start_orbit()
                                    ctrl._target_speed_factor = spd
                                elif action == 'waypoint':
                                    ctrl.go_to(target, speed_factor=spd)
                                switch_idx += 1
                            
                            # Slow near targets
                            if ctrl._mode == 'waypoint':
                                d = np.linalg.norm(ctrl.target[:2] - ctrl.pos[:2])
                                if d < 10:
                                    ctrl.slow_down(0.2)
                                elif d < 25:
                                    ctrl.slow_down(0.35)
                            
                            pos = ctrl.step()
                            positions.append(pos.copy())
                        
                        # Score against Cosmos constraints
                        constraints = [
                            (41, FZ_1, 62, 1.0),   # weight
                            (47, FZ_1, 62, 0.5),
                            (122, FZ_1, 82, 0.5),
                            (128, FZ_1, 92, 0.5),
                            (174, FZ_1, 5, 3.0),    # critical: at FZ_1
                            (254, FZ_2, 62, 0.5),
                            (279, FZ_2, 52, 0.5),
                            (290, FZ_2, 13, 3.0),   # critical: near FZ_2
                        ]
                        
                        total_err = 0
                        for step, target, expected, weight in constraints:
                            idx = step - WARMUP
                            if 0 <= idx < len(positions):
                                actual = dist_2d(positions[idx], target)
                                total_err += weight * abs(actual - expected)
                        
                        if total_err < best_err:
                            best_err = total_err
                            best_switches = switches
                            best_positions = positions
    
    print(f"Best error: {best_err:.1f}")
    print(f"Best switches:")
    for s in best_switches:
        print(f"  Step {s[0]}: {s[1]} → {s[2][:2] if s[2] is not None else 'orbit'}")
    
    return best_positions


if __name__ == "__main__":
    print("=== Tuning mode switch timings ===")
    positions = iterative_tune()
    
    validate_and_report(positions)
    
    data = build_flight_data(positions)
    
    # Save
    out_path = os.path.join("debug_output_1", "flight_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {out_path}")
    print(f"  Zone timeline: {len(data['zone_timeline'])} zones")
    print(f"  Flight path: {len(data['flight_path'])} waypoints")
    print(f"  Full positions: {len(data['full_positions'])} entries")
    
    # Print key moments
    for zt in data['zone_timeline']:
        step = zt['focusStep']
        idx = step - WARMUP
        pos = positions[idx]
        print(f"  {zt['zone']}: step {step} (t={zt['focusT']:.3f}) → ({pos[0]:.1f}, {pos[1]:.1f})")
