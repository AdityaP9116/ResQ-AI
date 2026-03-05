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
import importlib.util
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
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _THIS_DIR)
from logic_gates import HazardTracker

# Projection utility
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sim_bridge.projection_utils import batch_pixel_to_3d_world

# HuggingFace weight auto-download
# NOTE: Isaac Sim ships its own `utils` package which shadows our project's
# `utils` module.  Load model_downloader.py directly by file path to avoid
# the namespace collision.
_md_path = os.path.join(_PROJECT_ROOT, "utils", "model_downloader.py")
_md_spec = importlib.util.spec_from_file_location("resqai_model_downloader", _md_path)
_md_mod = importlib.util.module_from_spec(_md_spec)
_md_spec.loader.exec_module(_md_mod)
get_phase1_weights = _md_mod.get_phase1_weights
get_phase2_weights = _md_mod.get_phase2_weights


# ──────────────────────────────────────────────────────────────────────────
# Cosmos Reason 2 JSON prompt builder
# ──────────────────────────────────────────────────────────────────────────

def build_cosmos_prompt(
    hazards_3d: list[dict],
    drone_position: np.ndarray | list | None = None,
    frame_idx: int = 0,
    fire_report: dict | None = None,
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
            "fire_situation": { ... },
            "instruction": "..."
        }

    Args:
        hazards_3d: List of dicts, each with keys ``class_name``,
            ``bbox_centre``, ``world_xyz`` (may be None), ``confidence``.
        drone_position: Current drone [x, y, z] in world frame.
        frame_idx: Current simulation step for logging.
        fire_report: Optional fire situation report from FireManager.

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
        "fire_situation": fire_report if fire_report else {"active_fires": [], "total_area_burning_m2": 0},
        "instruction": (
            f"You are a search-and-rescue navigation agent controlling an autonomous drone "
            f"at bird's eye altitude ({drone_pos[2]:.0f}m). "
            "YOLO object detection has identified the hazard locations in the observations list. "
            "The fire_situation report shows active fire zones with burn areas and positions. "
            "Your PRIMARY task: decide which area the drone should fly to next. "
            "PRIORITY RULES (strict order): "
            "(1) CRITICAL: areas with BOTH active fire AND people nearby — investigate FIRST. "
            "(2) HIGH: areas with the largest fire concentration (most detections, biggest burn area). "
            "(3) MEDIUM: areas with people but no immediate fire — mark for rescue. "
            "(4) LOW: already-investigated areas or single small hazards. "
            "ANALYSIS STEPS: "
            "(a) Group observations by proximity — which area has the most fire + people? "
            "(b) Check fire_situation active_fires for each zone's burn area and spread direction. "
            "(c) Choose the highest-priority area and provide waypoint [x, y, z] ABOVE it at current altitude. "
            "(d) Explain why this area is top priority (fire count, people count, burn area). "
            "IMPORTANT: The drone stays at high altitude for bird's eye view. Do NOT lower altitude. "
            "Respond with JSON: "
            '{\"target_waypoint\": [x, y, z], \"reasoning\": \"priority analysis with fire+people counts...\", '
            '\"decision\": \"investigate|continue\", \"status\": \"critical|nominal\", '
            '\"advice\": \"...\", \"ground_crew_actions\": \"...\"}.'
        ),
    }

    return json.dumps(prompt, indent=2)


# ──────────────────────────────────────────────────────────────────────────
# VLM / Cosmos communication (async)
# ──────────────────────────────────────────────────────────────────────────

_vlm_responses: dict[int, dict] = {}
_vlm_requested: set[int] = set()
_vlm_lock = threading.Lock()
# Latest waypoint + reasoning from any VLM response (for flight controller)
_latest_vlm_waypoint: list[float] | None = None
_latest_vlm_reasoning: str = ""
# Module-level debug dir (set by OrchestratorBridge.__init__ so _query_vlm_async can use it)
_debug_dir: str | None = None


def _query_vlm_async(vlm_url: str, hazard_id: int, image_crop: np.ndarray, cosmos_prompt: str) -> None:
    """Send a hazard crop + Cosmos prompt to the VLM server in a background thread."""
    global _latest_vlm_waypoint, _latest_vlm_reasoning
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
                wp = data.get("target_waypoint")
                if wp is not None and isinstance(wp, (list, tuple)) and len(wp) >= 3:
                    _latest_vlm_waypoint = [float(wp[0]), float(wp[1]), float(wp[2])]
                    _latest_vlm_reasoning = data.get("reasoning", "")
            # ---- DEBUG: save raw VLM response ---------------------------------
            if _debug_dir:
                vlm_path = os.path.join(_debug_dir, "frames", f"vlm_response_hazard_{hazard_id:04d}.json")
                try:
                    with open(vlm_path, "w") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass
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
        seg_weights: str | None = None,
        vlm_url: str = "http://localhost:8000/analyze",
        iou_threshold: float = 0.5,
        debounce_frames: int = 5,
        debug_dir: str | None = None,
    ):
        global _debug_dir
        self.vlm_url = vlm_url

        # ---- Debug output dir ---------------------------------------------
        self._debug_dir = debug_dir
        _debug_dir = debug_dir  # expose to module-level for _query_vlm_async
        if debug_dir:
            os.makedirs(os.path.join(debug_dir, "frames"), exist_ok=True)
            print(f"[Orchestrator] Debug mode ON → {debug_dir}/frames/")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Phase 1: YOLO detection model --------------------------------
        if yolo_weights and os.path.exists(yolo_weights):
            self._yolo_path = yolo_weights
        else:
            self._yolo_path = get_phase1_weights()

        print(f"[Orchestrator] Loading Phase 1 YOLO (detect) from {self._yolo_path}  (device={self.device})")
        self.yolo = YOLO(self._yolo_path)

        # ---- Phase 2: YOLO segmentation model -----------------------------
        if seg_weights and os.path.exists(seg_weights):
            self._seg_path = seg_weights
        else:
            self._seg_path = get_phase2_weights()

        self.yolo_seg = None
        if os.path.isfile(self._seg_path):
            print(f"[Orchestrator] Loading Phase 2 YOLO (segment) from {self._seg_path}")
            self.yolo_seg = YOLO(self._seg_path, task="segment")
        else:
            print(f"[Orchestrator] Phase 2 weights not found at {self._seg_path} — segmentation disabled")

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
        fire_report: dict | None = None,
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
        _fidx = f"{frame_idx:06d}"  # zero-padded for filenames

        # ---- Phase 1: YOLO detection on RGB (or thermal) ------------------
        inference_input = rgb_frame
        if thermal_frame is not None and thermal_frame.size > 0:
            inference_input = cv2.cvtColor(thermal_frame, cv2.COLOR_GRAY2BGR)

        # ---- DEBUG: save RGB fed into YOLO --------------------------------
        if self._debug_dir:
            cv2.imwrite(os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_rgb.jpg"), rgb_frame)
            if thermal_frame is not None and thermal_frame.size > 0:
                cv2.imwrite(os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_thermal.jpg"), thermal_frame)

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

        # ---- DEBUG: save YOLO raw outputs ---------------------------------
        if self._debug_dir:
            class_names = [self.yolo.names.get(cid, f"class_{cid}") for cid in class_ids]
            yolo_log = {
                "frame_idx": frame_idx,
                "num_detections": len(detected_boxes),
                "boxes": detected_boxes,
                "class_ids": class_ids,
                "class_names": class_names,
                "confidences": confidences,
            }
            with open(os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_yolo.json"), "w") as f:
                json.dump(yolo_log, f, indent=2)

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
        cosmos_prompt = build_cosmos_prompt(hazards_3d, drone_position, frame_idx,
                                              fire_report=fire_report)

        # ---- DEBUG: save Cosmos prompt ------------------------------------
        if self._debug_dir:
            prompt_path = os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_cosmos_prompt.json")
            with open(prompt_path, "w") as f:
                f.write(cosmos_prompt)

        # ---- Phase 2: Segmentation ----------------------------------------
        seg_result = None
        if self.yolo_seg is not None:
            try:
                seg_results = self.yolo_seg.predict(
                    rgb_frame, device=self.device, verbose=False, retina_masks=True,
                )
                seg_classes: list[dict] = []
                for sr in seg_results:
                    if sr.masks is not None:
                        for i, mask in enumerate(sr.masks.data):
                            mask_np = mask.cpu().numpy().astype(np.uint8)
                            cid = int(sr.boxes.cls[i].item()) if sr.boxes is not None else -1
                            cls_name = self.yolo_seg.names.get(cid, f"class_{cid}")
                            pixel_area = int(mask_np.sum())
                            seg_classes.append({
                                "class_id": cid,
                                "class_name": cls_name,
                                "pixel_area": pixel_area,
                                "confidence": float(sr.boxes.conf[i].item()) if sr.boxes is not None else 0.0,
                            })

                seg_result = {
                    "num_masks": len(seg_classes),
                    "classes": seg_classes,
                }

                # DEBUG: save segmentation overlay
                if self._debug_dir and seg_results:
                    seg_overlay = seg_results[0].plot()
                    cv2.imwrite(
                        os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_seg.jpg"),
                        seg_overlay,
                    )
                    seg_json_path = os.path.join(self._debug_dir, "frames", f"frame_{_fidx}_seg.json")
                    with open(seg_json_path, "w") as f:
                        json.dump(seg_result, f, indent=2)

            except Exception as exc:
                print(f"[Orchestrator] Phase 2 segmentation error: {exc}")

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
            "segmentation": seg_result,
            "target_waypoint": _latest_vlm_waypoint,
            "reasoning": _latest_vlm_reasoning,
        }
