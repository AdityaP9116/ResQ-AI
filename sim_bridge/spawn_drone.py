#!/usr/bin/env python3
"""Spawn a Pegasus Multirotor (Iris) rigged with three Replicator camera sensors and IMU streaming.

Sensors mounted on the drone body at a 45-degree downward pitch:
    1. RGB Camera          — 640×480, 30 Hz
    2. Semantic Segmentation Camera — 640×480, 30 Hz (uses Replicator annotator)
    3. Depth Camera        — 640×480, 30 Hz (Distance-to-Image-Plane, critical for
                             Cosmos Reason 2 agent 3D waypoint computation)

IMU (accelerometer + gyroscope) data is streamed at 250 Hz via a lightweight
in-process backend so the RL flight-stabilization policy can consume it every
physics step without ROS overhead.

Usage::

    # Standalone (spawns into an empty world)
    /isaac-sim/python.sh sim_bridge/spawn_drone.py

    # Headless
    /isaac-sim/python.sh sim_bridge/spawn_drone.py --headless

    # As a library — call from another script that already has a running stage
    from sim_bridge.spawn_drone import spawn_resqai_drone
    drone, imu_backend = spawn_resqai_drone()
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must precede all Omniverse imports)
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI drone spawner")
    parser.add_argument("--headless", action="store_true", help="Run without a viewport window")
    parser.add_argument(
        "--stage-usd",
        type=str,
        default=None,
        help="Optional path to a .usd/.usda stage to load before spawning the drone "
             "(e.g. /tmp/resqai_urban_disaster.usda from generate_urban_scene.py)",
    )
    return parser.parse_args()


# Only create SimulationApp when run as standalone script. When imported (e.g. by
# headless_e2e_test.py), the caller has already created SimulationApp — creating
# a second one causes an access violation during extension shutdown.
if __name__ == "__main__":
    _args = _parse_args()
    simulation_app = SimulationApp({"headless": _args.headless})
else:
    _args = None
    simulation_app = None

# ---------------------------------------------------------------------------
# Omniverse / Pegasus imports (only valid after SimulationApp is created)
# ---------------------------------------------------------------------------
import carb
import omni.timeline
import omni.usd
import omni.replicator.core as rep
from omni.isaac.core.world import World
from pxr import Gf, Sdf, Usd, UsdGeom

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.graphical_sensors.graphical_sensor import GraphicalSensor
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS
from pegasus.simulator.logic.sensors.sensor import Sensor
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state import State


# ═══════════════════════════════════════════════════════════════════════════
# Custom Graphical Sensors — Semantic Segmentation & Depth cameras
# ═══════════════════════════════════════════════════════════════════════════

class ReplicatorCamera(GraphicalSensor):
    """A camera that exposes the underlying ``isaacsim.sensors.camera.Camera``
    and registers one or more Replicator annotators against its render product.

    Subclasses override ``_register_annotators`` to attach the annotators they
    need (semantic segmentation, depth, etc.).
    """

    def __init__(
        self,
        camera_name: str,
        *,
        resolution: tuple[int, int] = (640, 480),
        frequency: float = 30.0,
        position: np.ndarray | None = None,
        orientation_euler_deg: np.ndarray | None = None,
        clipping_range: tuple[float, float] = (0.1, 200.0),
    ):
        super().__init__(sensor_type=self._sensor_type_name(), update_rate=frequency)

        self._camera_name = camera_name
        self._resolution = resolution
        self._frequency = frequency
        self._position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self._orientation = orientation_euler_deg if orientation_euler_deg is not None else np.array([0.0, 0.0, 180.0])
        self._clipping_range = clipping_range

        self._stage_prim_path: str = ""
        self._camera = None
        self._annotators: dict[str, rep.annotators.Annotator] = {}
        self._state: dict = {}
        self._ready = False

        # Skip a few warm-up frames before reading data
        self._warmup_counter = 0
        self._warmup_frames = 5   # reduced from 60 – RTX warmup already done externally

    # Subclasses must implement:
    def _sensor_type_name(self) -> str:
        return "ReplicatorCamera"

    def _register_annotators(self, render_product_path: str) -> dict[str, rep.annotators.Annotator]:
        """Return {name: annotator} dict.  Called once after camera init."""
        return {}

    # --- Lifecycle ----------------------------------------------------------

    def initialize(self, vehicle):
        super().initialize(vehicle)
        from isaacsim.sensors.camera.camera import Camera
        from omni.usd import get_stage_next_free_path

        self._stage_prim_path = get_stage_next_free_path(
            PegasusInterface().world.stage,
            self._vehicle.prim_path + "/body/" + self._camera_name,
            False,
        )
        self._camera_name = self._stage_prim_path.rpartition("/")[-1]

        self._camera = Camera(
            prim_path=self._stage_prim_path,
            frequency=self._frequency,
            resolution=self._resolution,
        )
        self._camera.set_local_pose(
            self._position,
            Rotation.from_euler("ZYX", self._orientation, degrees=True).as_quat(),
        )

    def start(self):
        self._camera.initialize()
        self._camera.set_resolution(self._resolution)
        self._camera.set_clipping_range(*self._clipping_range)
        self._camera.set_frequency(self._frequency)

        rp = self._camera._render_product_path
        self._annotators = self._register_annotators(rp)
        self._ready = True

    def stop(self):
        self._ready = False

    @property
    def state(self):
        return self._state

    def update(self, state: State, dt: float):
        self._warmup_counter += 1
        if self._warmup_counter < self._warmup_frames or not self._ready:
            return None

        self._state = {
            "camera_name": self._camera_name,
            "stage_prim_path": self._stage_prim_path,
            "height": self._resolution[1],
            "width": self._resolution[0],
            "frequency": self._frequency,
            "camera": self._camera,
        }

        for name, annotator in self._annotators.items():
            try:
                self._state[name] = annotator.get_data()
            except Exception:
                self._state[name] = None

        return self._state


class RGBCamera(ReplicatorCamera):
    """Replicator-based RGB camera.
    
    Produces standard RGB images using the replicator annotator pipeline.
    """

    def _sensor_type_name(self) -> str:
        return "RGBCamera"

    def _register_annotators(self, render_product_path: str):
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product_path])
        return {"rgba": rgb}


class SemanticSegmentationCamera(ReplicatorCamera):
    """Replicator-based semantic segmentation camera.

    Produces per-pixel class IDs that correspond to the semantic labels applied
    to scene prims (fire, person, vehicle, building, vegetation, terrain).
    """

    def _sensor_type_name(self) -> str:
        return "SemanticSegmentationCamera"

    def _register_annotators(self, render_product_path: str):
        seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
        seg.attach([render_product_path])
        return {"semantic_segmentation": seg}


class DepthCamera(ReplicatorCamera):
    """Replicator-based depth camera (distance-to-image-plane).

    This is the format needed by the Cosmos Reason 2 agent for computing
    metric 3D waypoints from monocular depth.
    """

    def _sensor_type_name(self) -> str:
        return "DepthCamera"

    def _register_annotators(self, render_product_path: str):
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach([render_product_path])
        return {"depth": depth}

    def start(self):
        super().start()
        self._camera.add_distance_to_image_plane_to_frame()


# ═══════════════════════════════════════════════════════════════════════════
# IMU Streaming Backend — feeds accelerometer + gyroscope to RL policy
# ═══════════════════════════════════════════════════════════════════════════

class IMUStreamBackend(Backend):
    """Lightweight in-process backend that captures IMU readings every physics
    step without requiring ROS or MAVLink.

    The RL flight-stabilization policy reads from ``latest_imu`` at whatever
    rate it needs.
    """

    def __init__(self):
        super().__init__(config=None)

        self.latest_imu: dict = {
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "angular_velocity": np.array([0.0, 0.0, 0.0]),
            "linear_acceleration": np.array([0.0, 0.0, 0.0]),
        }
        self.latest_state: dict = {
            "position": np.zeros(3),
            "attitude": np.array([0.0, 0.0, 0.0, 1.0]),
            "linear_velocity": np.zeros(3),
            "linear_body_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "linear_acceleration": np.zeros(3),
        }
        self._input_ref = [0.0, 0.0, 0.0, 0.0]

    # --- Backend interface --------------------------------------------------

    def update_sensor(self, sensor_type: str, data):
        if sensor_type == "IMU" and data is not None:
            self.latest_imu = {
                "orientation": np.array(data["orientation"], dtype=np.float64),
                "angular_velocity": np.array(data["angular_velocity"], dtype=np.float64),
                "linear_acceleration": np.array(data["linear_acceleration"], dtype=np.float64),
            }

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state: State):
        self.latest_state = {
            "position": state.position.copy(),
            "attitude": state.attitude.copy(),
            "linear_velocity": state.linear_velocity.copy(),
            "linear_body_velocity": state.linear_body_velocity.copy(),
            "angular_velocity": state.angular_velocity.copy(),
            "linear_acceleration": state.linear_acceleration.copy(),
        }

    def input_reference(self):
        return self._input_ref

    def set_rotor_velocities(self, velocities: list[float]):
        """Called by the RL policy to command rotor angular velocities [rad/s]."""
        self._input_ref = list(velocities)

    def update(self, dt: float):
        pass

    def start(self):
        self._input_ref = [0.0, 0.0, 0.0, 0.0]

    def stop(self):
        self._input_ref = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self._input_ref = [0.0, 0.0, 0.0, 0.0]

    # --- Convenience accessors for the RL policy ----------------------------

    @property
    def accelerometer(self) -> np.ndarray:
        """Latest accelerometer reading [ax, ay, az] in body FRD frame (m/s^2)."""
        return self.latest_imu["linear_acceleration"]

    @property
    def gyroscope(self) -> np.ndarray:
        """Latest gyroscope reading [p, q, r] in body FRD frame (rad/s)."""
        return self.latest_imu["angular_velocity"]

    @property
    def orientation(self) -> np.ndarray:
        """Latest orientation quaternion [qx, qy, qz, qw] (FRD-NED)."""
        return self.latest_imu["orientation"]


# ═══════════════════════════════════════════════════════════════════════════
# Drone Spawning
# ═══════════════════════════════════════════════════════════════════════════

# Iris drone body forward (body +X) aligns with world +Y when yaw=0
# (confirmed by camera world-pose diagnostic).
#
# Camera orientation in ZYX Euler (applied as Rz×Ry×Rx in body frame).
# [90, 0, -110] gives 70° forward-and-down along the drone's forward
# direction (world +Y), with camera "up" having a +Z world component.
#
# Derivation with R_B (yaw=0): bodyX=worldY, bodyY=worldX, bodyZ=world-Z
#   The Rx angle = -90 - (tilt_from_nadir).
#   70° from horizontal = 20° from nadir → Rx = -90 - 20 = -110
#   optical axis = -Z_cam ≈ (0, 0.342, -0.940) = 70° forward-down ✓
#   (Runtime code in headless_e2e_test.py overrides this precisely.)
_SENSOR_ORIENTATION = np.array([90.0, 0.0, -110.0])

# Mount point: at body CG (no offset to avoid FRD sign confusion)
_SENSOR_POSITION = np.array([0.0, 0.0, 0.0])


def spawn_resqai_drone(
    stage_prefix: str = "/World/ResQDrone",
    init_pos: list[float] | None = None,
    init_yaw_deg: float = 0.0,
) -> tuple[Multirotor, IMUStreamBackend]:
    """Spawn an Iris multirotor configured for the ResQ-AI pipeline.

    Args:
        stage_prefix: USD prim path where the drone is placed.
        init_pos: Initial [x, y, z] in ENU metres.  Defaults to [0, 0, 2].
        init_yaw_deg: Initial heading in degrees.

    Returns:
        (drone, imu_backend) where *imu_backend* provides real-time
        accelerometer / gyroscope readings for the RL policy.
    """
    if init_pos is None:
        init_pos = [0.0, 0.0, 2.0]

    init_quat = Rotation.from_euler("XYZ", [0.0, 0.0, init_yaw_deg], degrees=True).as_quat()

    # ---- Configure sensors ------------------------------------------------

    imu = IMU(config={"update_rate": 250.0})

    rgb_cam = RGBCamera(
        "rgb_cam",
        resolution=(640, 480),
        frequency=30.0,
        position=_SENSOR_POSITION,
        orientation_euler_deg=_SENSOR_ORIENTATION,
        clipping_range=(0.1, 200.0),
    )

    seg_cam = SemanticSegmentationCamera(
        "seg_cam",
        resolution=(640, 480),
        frequency=30.0,
        position=_SENSOR_POSITION,
        orientation_euler_deg=_SENSOR_ORIENTATION,
    )

    depth_cam = DepthCamera(
        "depth_cam",
        resolution=(640, 480),
        frequency=30.0,
        position=_SENSOR_POSITION,
        orientation_euler_deg=_SENSOR_ORIENTATION,
    )

    # ---- Configure backend ------------------------------------------------

    imu_backend = IMUStreamBackend()

    # ---- Assemble Multirotor config ---------------------------------------

    config = MultirotorConfig()
    config.sensors = [Barometer(), imu, Magnetometer(), GPS()]
    config.graphical_sensors = [rgb_cam, seg_cam, depth_cam]
    config.backends = [imu_backend]

    drone = Multirotor(
        stage_prefix,
        ROBOTS["Iris"],
        0,
        init_pos,
        init_quat,
        config=config,
    )

    print(f"[ResQ-AI] Drone spawned at {stage_prefix}")
    print(f"          Position : {init_pos}")
    print(f"          Sensors  : RGB (640×480), Semantic Seg (640×480), Depth (640×480)")
    print(f"          IMU      : 250 Hz (accel + gyro) → IMUStreamBackend")
    print(f"          Camera orientation ZYX: {list(_SENSOR_ORIENTATION)} (70° forward-down)")

    return drone, imu_backend


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    timeline = omni.timeline.get_timeline_interface()

    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    # Optionally load a pre-built stage (e.g. the urban disaster scene)
    if _args.stage_usd:
        print(f"[ResQ-AI] Loading stage from {_args.stage_usd}")
        omni.usd.get_context().open_stage(_args.stage_usd)
    else:
        world.scene.add_default_ground_plane()

    # Spawn the drone
    drone, imu_backend = spawn_resqai_drone(
        stage_prefix="/World/ResQDrone",
        init_pos=[0.0, 0.0, 2.0],
    )

    world.reset()

    # Print IMU data periodically as a sanity check
    _step = [0]

    def _imu_logger(dt: float):
        _step[0] += 1
        if _step[0] % 500 == 0:
            acc = imu_backend.accelerometer
            gyr = imu_backend.gyroscope
            ori = imu_backend.orientation
            pos = imu_backend.latest_state["position"]
            print(
                f"[IMU @step {_step[0]}]  "
                f"accel=[{acc[0]:+7.3f}, {acc[1]:+7.3f}, {acc[2]:+7.3f}]  "
                f"gyro=[{gyr[0]:+7.4f}, {gyr[1]:+7.4f}, {gyr[2]:+7.4f}]  "
                f"pos=[{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]"
            )

    world.add_physics_callback("resqai_imu_logger", _imu_logger)

    # Run the simulation
    timeline.play()
    print("[ResQ-AI] Simulation running — press Ctrl+C or close the window to exit.")

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        pass

    carb.log_warn("ResQ-AI drone simulation shutting down.")
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
