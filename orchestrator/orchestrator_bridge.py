"""Orchestrator bridge — connects the simulation sensor pipeline to the ResQ-AI
inference stack (YOLO detection → 3-D projection → Cosmos Reason 2 prompt).

This module extracts the core logic from the original ``orchestrator/main.py``
into a class that can be driven by live numpy arrays from Isaac Sim instead of
a video file.

Usage from the simulation loop::

    from orchestrator.orchestrator_bridge import OrchestratorBridge

    bridge = OrchestratorBridge(yolo_weights="path/to/best.pt")

    result = bridge.process_frame(
        rgb_frame=rgb,            # (H, W, 3) uint8 BGR
        thermal_frame=thermal,    # (H, W)   uint8 grayscale | None
        depth_map=depth,          # (H, W)   float metres
        camera_intrinsics=K,      # 3×3 numpy
        camera_world_pose=pose,   # (pos, quat_wxyz) or 4×4
        frame_idx=step,
        drone_position=pos,
    )
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# Hazard tracker lives next to this file
sys.path.insert(0, os.path.dirname(__file__))
from logic_gates import HazardTracker

# Projection utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim_bridge.projection_utils import batch_pixel_to_3d_world


# ──────────────────────────────────────────────────────────────────────────
# Cosmos Reason 2 JSON prompt builder
# ──────────────────────────────────────────────────────────────────────────

def build_cosmos_prompt(
    hazards_3d: list[dict],
    drone_position: np.ndarray | list | None = None,
    frame_idx: int = 0,
) -> str:
    """Package detected hazards + 3-D coordinates into a JSON prompt for
    Cosmos Reason 2 so it can output a navigational waypoint.

    The prompt follows a structured schema that Cosmos can parse:

    .. code-block:: json

        {
            "task": "navigation_waypoint",
            "drone_state": { "position": [x, y, z] },
            "observations": [
                {
                    "class": "fire",
                    "pixel": [320, 280],
                    "world_xyz": [12.5, -3.2, 0.8],
                    "confidence": 0.92
                }
            ],
            "instruction": "..."
        }

    Args:
        hazards_3d: List of dicts, each with keys ``class_name``,
            ``bbox_centre``, ``world_xyz`` (may be None), ``confidence``.
        drone_position: Current drone [x, y, z] in world frame.
        frame_idx: Current simulation step for logging.

    Returns:
        A JSON string ready to send to Cosmos Reason 2.
    """
    observations = []
    for h in hazards_3d:
        obs: dict[str, Any] = {
            "class": h["class_name"],
            "pixel": list(h["bbox_centre"]),
        }
        if h.get("world_xyz") is not None:
            obs["world_xyz"] = [round(float(v), 3) for v in h["world_xyz"]]
        if h.get("confidence") is not None:
            obs["confidence"] = round(float(h["confidence"]), 3)
        observations.append(obs)

    drone_pos = [0.0, 0.0, 0.0]
    if drone_position is not None:
        drone_pos = [round(float(v), 3) for v in np.asarray(drone_position).ravel()[:3]]

    prompt = {
        "task": "navigation_waypoint",
        "frame": frame_idx,
        "drone_state": {
            "position": drone_pos,
        },
        "observations": observations,
        "instruction": (
            "You are a search-and-rescue navigation agent.  Given the drone's "
            "current position and the 3-D world coordinates of detected hazards, "
            "output the next waypoint [x, y, z] the drone should fly to in order "
            "to investigate the highest-priority hazard while maintaining a safe "
            "stand-off distance.  Respond with JSON: "
            '{\"waypoint\": [x, y, z], \"reasoning\": \"...\"}.'
        ),
    }

    return json.dumps(prompt, indent=2)


# ──────────────────────────────────────────────────────────────────────────
# VLM / Cosmos communication (async)
# ──────────────────────────────────────────────────────────────────────────

_vlm_responses: dict[int, dict] = {}
_vlm_requested: set[int] = set()
_vlm_lock = threading.Lock()


def _query_vlm_async(vlm_url: str, hazard_id: int, image_crop: np.ndarray, cosmos_prompt: str) -> None:
    """Send a hazard crop + Cosmos prompt to the VLM server in a background thread."""
    try:
        _, buffer = cv2.imencode(".jpg", image_crop)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "image_base64": img_b64,
            "context": cosmos_prompt,
        }
        resp = requests.post(vlm_url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            with _vlm_lock:
                _vlm_responses[hazard_id] = data
            print(f"[Cosmos] Hazard {hazard_id} → {data.get('advice', '(no advice)')[:120]}")
        else:
            print(f"[Cosmos] Hazard {hazard_id}: server returned {resp.status_code}")
    except Exception as exc:
        print(f"[Cosmos] Hazard {hazard_id}: request failed — {exc}")


# ──────────────────────────────────────────────────────────────────────────
# OrchestratorBridge
# ──────────────────────────────────────────────────────────────────────────

class OrchestratorBridge:
    """Stateful bridge that runs YOLO detection on live frames, projects
    bounding-box centres to 3-D, and produces Cosmos Reason 2 prompts.

    This replaces the video-file loop from the original ``orchestrator/main.py``
    with a ``process_frame()`` method that accepts numpy arrays.
    """

    def __init__(
        self,
        yolo_weights: str | None = None,
        vlm_url: str = "http://localhost:8000/analyze",
        iou_threshold: float = 0.5,
        debounce_frames: int = 5,
    ):
        self.vlm_url = vlm_url

        # ---- YOLO model ---------------------------------------------------
        if yolo_weights and os.path.exists(yolo_weights):
            self._yolo_path = yolo_weights
        else:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "Phase1_SituationalAwareness", "best.pt"
            )
            self._yolo_path = default_path if os.path.exists(default_path) else "yolov8n.pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Orchestrator] Loading YOLO from {self._yolo_path}  (device={self.device})")
        self.yolo = YOLO(self._yolo_path)

        # ---- Hazard tracker (logic gates) ---------------------------------
        self.tracker = HazardTracker(
            iou_threshold=iou_threshold,
            debounce_frames=debounce_frames,
        )

        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray | None,
        depth_map: np.ndarray | None,
        camera_intrinsics: np.ndarray | None,
        camera_world_pose: tuple | np.ndarray | None,
        frame_idx: int = 0,
        drone_position: np.ndarray | None = None,
    ) -> dict | None:
        """Run the full orchestrator pipeline on a single frame.

        Args:
            rgb_frame: ``(H, W, 3)`` uint8 BGR image.
            thermal_frame: ``(H, W)`` uint8 thermal image, or ``None``.
            depth_map: ``(H, W)`` float depth in metres, or ``None``.
            camera_intrinsics: 3×3 intrinsic matrix ``K``.
            camera_world_pose: ``(pos, quat_wxyz)`` or 4×4 matrix.
            frame_idx: Current simulation step.
            drone_position: Drone [x, y, z] in world frame.

        Returns:
            A dict with keys ``hazards`` (list of hazard dicts) and
            ``cosmos_prompt`` (JSON string), or ``None`` if nothing detected.
        """
        self._frame_count += 1

        # ---- Phase 1: YOLO detection on RGB (or thermal) ------------------
        inference_input = rgb_frame
        if thermal_frame is not None and thermal_frame.size > 0:
            inference_input = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)

        results = self.yolo.predict(rgb_frame, device=self.device, verbose=False)

        detected_boxes: list[list[float]] = []
        class_ids: list[int] = []
        confidences: list[float] = []

        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy().tolist()
                detected_boxes.append(b)
                class_ids.append(int(box.cls[0].item()))
                confidences.append(float(box.conf[0].item()))

        if not detected_boxes:
            return None

        # ---- Logic gates (temporal debounce + IoU tracking) ---------------
        active_hazards = self.tracker.update(detected_boxes, class_ids)
        if not active_hazards:
            return None

        # ---- Compute bounding-box centres ---------------------------------
        centres: list[tuple[float, float]] = []
        for h in active_hazards:
            b = h["box"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            centres.append((cx, cy))

        # ---- 3-D projection via depth map ---------------------------------
        world_coords: list[np.ndarray | None] = [None] * len(centres)
        if depth_map is not None and camera_intrinsics is not None and camera_world_pose is not None:
            world_coords = batch_pixel_to_3d_world(
                centres, depth_map, camera_intrinsics, camera_world_pose,
            )

        # ---- Build per-hazard output + Cosmos prompt input ----------------
        hazards_3d: list[dict] = []
        for idx, hazard in enumerate(active_hazards):
            cid = hazard["class_id"]
            class_name = self.yolo.names.get(cid, f"class_{cid}")

            entry = {
                "hazard_id": hazard["id"],
                "class_name": class_name,
                "class_id": cid,
                "bbox": hazard["box"],
                "bbox_centre": list(centres[idx]),
                "world_xyz": world_coords[idx].tolist() if world_coords[idx] is not None else None,
                "confidence": confidences[idx] if idx < len(confidences) else None,
                "frame_idx": frame_idx,
            }

            if drone_position is not None:
                entry["drone_position"] = np.asarray(drone_position).tolist()

            hazards_3d.append(entry)

        # ---- Build Cosmos Reason 2 JSON prompt ----------------------------
        cosmos_prompt = build_cosmos_prompt(hazards_3d, drone_position, frame_idx)

        # ---- Fire async VLM request for new hazards -----------------------
        for idx, hazard in enumerate(active_hazards):
            hid = hazard["id"]
            if hid in _vlm_requested:
                continue
            _vlm_requested.add(hid)

            b = hazard["box"]
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(rgb_frame.shape[1], int(b[2])), min(rgb_frame.shape[0], int(b[3]))
            crop = rgb_frame[y1:y2, x1:x2]
            if crop.size > 0:
                threading.Thread(
                    target=_query_vlm_async,
                    args=(self.vlm_url, hid, crop.copy(), cosmos_prompt),
                    daemon=True,
                ).start()

        # ---- Attach any VLM responses that have arrived -------------------
        with _vlm_lock:
            for entry in hazards_3d:
                resp = _vlm_responses.get(entry["hazard_id"])
                if resp:
                    entry["vlm_analysis"] = resp.get("advice", "")

        return {
            "hazards": hazards_3d,
            "cosmos_prompt": cosmos_prompt,
        }
