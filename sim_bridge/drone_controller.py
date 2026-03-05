"""MPC-inspired smooth dynamics controller for ResQ-AI drone.

Replaces the blocky constant-speed linear interpolation with a velocity /
acceleration-based system that produces natural, cinematic drone flight.

Features
--------
* Smooth acceleration & deceleration — no abrupt stops
* Velocity-proportional-to-distance braking near waypoints
* Wind perturbation (multi-frequency sinusoidal + gusts) for realism
* Speed-factor modulation — slow down over fires, speed up in transit
* Hover drift — small oscillations when hovering (drones never stop perfectly)
* Smooth yaw tracking toward velocity direction
* Orbit mode — circular survey with tangential velocity (seamless transitions)
* Altitude hold — maintains target altitude with gentle oscillation
"""

from __future__ import annotations

import math
import numpy as np


class DroneController:
    """Simulates smooth drone flight dynamics with MPC-style control.

    State vector: position [x, y, z], velocity [vx, vy, vz], yaw (rad).

    Two modes of operation:
    * **orbit** — smooth circular survey around a center point
    * **waypoint** — fly toward a target with smooth accel / decel

    Transitions between modes are seamless because the velocity state is
    preserved — the drone arcs into the new trajectory instead of snapping.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        start_pos: list | np.ndarray,
        *,
        max_vel: float = 4.5,           # m / step
        max_accel: float = 0.6,         # m / step²
        drag: float = 0.04,             # velocity damping per step
        hover_drift: float = 0.12,      # amplitude of hover oscillation
        wind_strength: float = 0.25,    # base wind amplitude
        altitude: float = 110.0,        # target altitude (m)
    ) -> None:
        self.pos = np.array(start_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0  # radians, 0 = world +X

        # Limits
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.drag = drag
        self.hover_drift = hover_drift
        self.wind_strength = wind_strength
        self.altitude = altitude

        # Target
        self.target = self.pos.copy()

        # Speed factor (1.0 = normal, < 1.0 = slow over fire)
        self._speed_factor = 1.0
        self._target_speed_factor = 1.0

        # Orbit state
        self._orbit_angle = 0.0
        self._orbit_center = np.array(start_pos[:2], dtype=np.float64)
        self._orbit_radius = 55.0
        self._orbit_speed = 0.025  # rad / step
        self._mode = "orbit"

        # Wind state — three uncorrelated sinusoidal components + gusts
        rng = np.random.default_rng(42)
        self._wind_phase = rng.uniform(0, 2 * math.pi, size=3)
        self._wind_freq = np.array([0.023, 0.017, 0.009])
        self._gust_phase = rng.uniform(0, 2 * math.pi)
        self._gust_freq = 0.005

        self._step_count = 0

        # Diagnostics
        self.at_target = False

    # ------------------------------------------------------------------
    # Public API — mode transitions
    # ------------------------------------------------------------------

    def configure_orbit(
        self,
        center: list | np.ndarray,
        radius: float = 55.0,
        speed: float = 0.025,
    ) -> None:
        """Configure orbit parameters (can be called before or after
        ``start_orbit``)."""
        self._orbit_center = np.array(center[:2], dtype=np.float64)
        self._orbit_radius = radius
        self._orbit_speed = speed

    def start_orbit(self) -> None:
        """Switch to orbit mode.  Current velocity is preserved so the
        transition is a smooth arc, not a snap."""
        self._mode = "orbit"
        # Sync orbit angle with current XY position
        dx = self.pos[0] - self._orbit_center[0]
        dy = self.pos[1] - self._orbit_center[1]
        self._orbit_angle = math.atan2(dy, dx)

    def go_to(
        self,
        target: list | np.ndarray,
        speed_factor: float = 1.0,
    ) -> None:
        """Switch to waypoint mode — fly toward *target* at *speed_factor*
        of max velocity.  Current velocity is preserved for a smooth arc."""
        self.target = np.array(target, dtype=np.float64)
        self.target[2] = self.altitude  # enforce altitude hold
        self._target_speed_factor = speed_factor
        self._mode = "waypoint"
        self.at_target = False

    def slow_down(self, factor: float = 0.35) -> None:
        """Reduce speed factor (e.g. when hovering over fire)."""
        self._target_speed_factor = factor

    def resume_speed(self) -> None:
        """Resume normal speed."""
        self._target_speed_factor = 1.0

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(self) -> np.ndarray:
        """Advance one simulation step.  Returns the new position [x, y, z].

        This method:
        1. Computes desired velocity based on current mode
        2. Applies acceleration limits (smooth response)
        3. Adds wind + drag
        4. Integrates position
        5. Updates yaw
        """
        self._step_count += 1
        t = self._step_count

        # ---- Smooth speed factor transition ----
        blend = 0.06
        self._speed_factor += blend * (self._target_speed_factor - self._speed_factor)

        # ---- Compute desired velocity based on mode ----
        if self._mode == "orbit":
            desired_vel = self._orbit_step(t)
        else:
            desired_vel = self._waypoint_step(t)

        # ---- Acceleration limiting (smooth response) ----
        vel_error = desired_vel - self.vel
        accel_mag = np.linalg.norm(vel_error)
        if accel_mag > self.max_accel:
            vel_error = vel_error / accel_mag * self.max_accel
        self.vel += vel_error

        # ---- Wind perturbation (reduced when hovering) ----
        wind = self._compute_wind(t) * max(self._speed_factor, 0.3)
        self.vel += wind

        # ---- Drag (velocity damping — stronger when hovering for stability) ----
        effective_drag = self.drag + (1.0 - self._speed_factor) * 0.06
        self.vel *= (1.0 - effective_drag)

        # ---- Velocity clamp ----
        speed = np.linalg.norm(self.vel)
        max_spd = self.max_vel * max(self._speed_factor, 0.15)
        if speed > max_spd:
            self.vel = self.vel / speed * max_spd

        # ---- Altitude hold (stiffer spring when hovering) ----
        alt_error = self.altitude - self.pos[2]
        alt_gain = 0.08 + (1.0 - self._speed_factor) * 0.12
        self.vel[2] += alt_error * alt_gain

        # ---- Integrate position ----
        self.pos += self.vel

        # ---- Smooth yaw tracking ----
        lateral_speed = np.linalg.norm(self.vel[:2])
        if lateral_speed > 0.15:
            target_yaw = math.atan2(self.vel[1], self.vel[0])
            yaw_diff = target_yaw - self.yaw
            # Normalize to [-π, π]
            yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
            self.yaw += yaw_diff * 0.08  # smooth yaw response

        # ---- at_target check (for state machine) ----
        if self._mode == "waypoint":
            dist = np.linalg.norm(self.target[:2] - self.pos[:2])
            self.at_target = (dist < 5.0) and (speed < 0.8)
        else:
            self.at_target = False

        return self.pos.copy()

    # ------------------------------------------------------------------
    # Internal — orbit dynamics
    # ------------------------------------------------------------------

    def _orbit_step(self, t: int) -> np.ndarray:
        """Compute desired velocity to track a circular orbit."""
        # Advance orbit angle
        self._orbit_angle += self._orbit_speed * self._speed_factor

        # Desired position on the orbit circle
        orbit_pos = np.array([
            self._orbit_center[0] + self._orbit_radius * math.cos(self._orbit_angle),
            self._orbit_center[1] + self._orbit_radius * math.sin(self._orbit_angle),
            self.altitude,
        ])

        # PD-like tracking: spring toward orbit position + damping
        # The spring constant must be gentle enough for smooth arcs
        pos_error = orbit_pos - self.pos
        desired_vel = pos_error * 0.18 + self.vel * 0.05

        # Add slight hover drift (makes the orbit feel less mechanical)
        desired_vel += self._hover_perturbation(t) * 0.3

        return desired_vel

    # ------------------------------------------------------------------
    # Internal — waypoint dynamics
    # ------------------------------------------------------------------

    def _waypoint_step(self, t: int) -> np.ndarray:
        """Compute desired velocity to approach a waypoint with smooth
        deceleration."""
        diff = self.target - self.pos
        dist = np.linalg.norm(diff[:2])  # XY distance

        if dist > 1.5:
            # Desired speed: ramp down proportionally near target
            # This creates a smooth exponential-like deceleration
            brake_dist = self.max_vel * self._speed_factor * 6.0
            desired_speed = min(
                self.max_vel * self._speed_factor,
                (dist / brake_dist) * self.max_vel * self._speed_factor,
            )
            desired_speed = max(desired_speed, 0.15)  # min creep speed

            direction = diff / (np.linalg.norm(diff) + 1e-8)
            desired_vel = direction * desired_speed
        else:
            # At target — very gentle hover (scaled down for stability)
            desired_vel = self._hover_perturbation(t) * 0.3

        return desired_vel

    # ------------------------------------------------------------------
    # Internal — wind and hover
    # ------------------------------------------------------------------

    def _compute_wind(self, t: int) -> np.ndarray:
        """Multi-frequency sinusoidal wind with occasional gusts."""
        # Base wind: three uncorrelated sinusoidal components
        base = np.array([
            math.sin(self._wind_freq[0] * t + self._wind_phase[0]),
            math.sin(self._wind_freq[1] * t + self._wind_phase[1]),
            math.sin(self._wind_freq[2] * t + self._wind_phase[2]) * 0.3,
        ]) * self.wind_strength * 0.008

        # Occasional gusts (every ~120 steps for ~20 steps)
        gust_cycle = (t % 120)
        if gust_cycle < 20:
            gust_intensity = 0.5 * math.sin(math.pi * gust_cycle / 20.0)
            gust_dir = np.array([
                math.cos(self._gust_phase + t * self._gust_freq),
                math.sin(self._gust_phase + t * self._gust_freq),
                0.0,
            ])
            base += gust_dir * gust_intensity * self.wind_strength * 0.015

        return base

    def _hover_perturbation(self, t: int) -> np.ndarray:
        """Small oscillation when hovering — drones never hold perfectly still."""
        # Scale drift down when speed_factor is low (focused hovering)
        drift_scale = self.hover_drift * max(self._speed_factor, 0.25)
        return np.array([
            math.sin(0.07 * t) * drift_scale,
            math.cos(0.053 * t + 1.0) * drift_scale,
            math.sin(0.031 * t + 2.0) * drift_scale * 0.3,
        ])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        """Current speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.vel))

    @property
    def speed_factor(self) -> float:
        return self._speed_factor

    @property
    def mode(self) -> str:
        return self._mode

    def __repr__(self) -> str:
        return (
            f"DroneController(pos=[{self.pos[0]:.1f}, {self.pos[1]:.1f}, {self.pos[2]:.1f}], "
            f"vel={self.speed:.2f} m/s, mode={self._mode}, yaw={math.degrees(self.yaw):.0f}°)"
        )
