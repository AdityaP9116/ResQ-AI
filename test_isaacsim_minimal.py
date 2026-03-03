"""Test: exact same as headless_e2e_test.py but from root directory."""
from __future__ import annotations
import os
import sys
import time

print("[TEST] Step 1: stdlib imports OK")

from isaacsim import SimulationApp
print("[TEST] Step 2: isaacsim imported OK")

simulation_app = SimulationApp({"headless": True})
print("[TEST] Step 3: SimulationApp created OK!")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
print("[TEST] Step 4: dotenv OK")

import numpy as np
print("[TEST] Step 5: numpy OK")

import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from pxr import Gf, UsdGeom
print("[TEST] Step 6: omni/pxr OK")

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
print("[TEST] Step 7: pegasus OK")

from sim_bridge.generate_urban_scene import main as generate_scene
print("[TEST] Step 8: generate_urban_scene OK")

from sim_bridge.spawn_drone import spawn_resqai_drone
print("[TEST] Step 9: spawn_drone OK")

from sim_bridge.thermal_sim import generate_synthetic_thermal
from sim_bridge.projection_utils import make_intrinsics_from_fov
print("[TEST] Step 10: thermal/projection OK")

from orchestrator.orchestrator_bridge import OrchestratorBridge
print("[TEST] Step 11: orchestrator_bridge OK — ALL IMPORTS PASSED!")

simulation_app.close()
print("[TEST] Done!")
