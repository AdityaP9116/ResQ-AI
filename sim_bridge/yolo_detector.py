"""Dual-model YOLO detector for ResQ-AI Isaac Sim pipeline.

Combines two YOLO models:
  1. AIDER-trained model  -> fire detection (class index 1)
  2. Pretrained YOLOv8n   -> COCO person detection (class index 0)

Only "person" and "fire" detections are returned.  Fire detections can
optionally be cross-validated against thermal hotspots.

Isaac Sim 5.1 compatibility: NO f-strings.  Use str() + concatenation.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# AIDER dataset class mapping (from data_colab.yaml)
AIDER_CLASSES = {
    0: "collapsed_building",
    1: "fire",
    2: "flooded_areas",
    3: "traffic_incident",
}

# Only keep fire from the AIDER model
AIDER_KEEP_CLASSES = {1}  # fire

# COCO class we care about
COCO_PERSON_CLASS = 0

# Confidence thresholds
FIRE_CONF_THRESHOLD = 0.4
PERSON_CONF_THRESHOLD = 0.3

# Default weight paths (relative to project root)
DEFAULT_FIRE_WEIGHTS = os.path.join(
    _PROJECT_ROOT, "Phase1_SituationalAwareness", "best.pt"
)
DEFAULT_PERSON_WEIGHTS = "yolov8n.pt"  # auto-downloaded by ultralytics


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DualYOLODetector(object):
    """Loads two YOLO models and exposes a unified detect() method.

    Parameters
    ----------
    fire_weights : str or None
        Path to AIDER-trained weights (.pt or .engine).  Falls back to
        ``Phase1_SituationalAwareness/best.pt``.
    person_weights : str or None
        Path to a COCO-pretrained YOLOv8 model.  Defaults to ``yolov8n.pt``
        which ultralytics will auto-download the first time.
    fire_conf : float
        Minimum confidence for fire detections.
    person_conf : float
        Minimum confidence for person detections.
    device : str or int
        Inference device (``"cpu"``, ``0``, etc.).
    """

    def __init__(
        self,
        fire_weights=None,
        person_weights=None,
        fire_conf=FIRE_CONF_THRESHOLD,
        person_conf=PERSON_CONF_THRESHOLD,
        device=0,
    ):
        from ultralytics import YOLO

        self._fire_conf = fire_conf
        self._person_conf = person_conf

        # --- Fire model (AIDER) ------------------------------------------
        fw = fire_weights or DEFAULT_FIRE_WEIGHTS
        if not os.path.isfile(fw):
            # Try .engine variant
            engine = fw.rsplit(".", 1)[0] + ".engine"
            if os.path.isfile(engine):
                fw = engine
        print("[YOLODetector] Loading fire model: " + str(fw))
        self._fire_model = YOLO(fw, task="detect")

        # --- Person model (COCO pretrained) ------------------------------
        pw = person_weights or DEFAULT_PERSON_WEIGHTS
        print("[YOLODetector] Loading person model: " + str(pw))
        self._person_model = YOLO(pw, task="detect")

        self._device = device
        print("[YOLODetector] Initialised (device=" + str(device) + ")")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, rgb_frame, thermal_hotspots=None):
        """Run both models on *rgb_frame* and return unified detections.

        Parameters
        ----------
        rgb_frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image from the drone camera.
        thermal_hotspots : list[dict] or None
            Output of ``ThermalProcessor.process()`` — used to cross-validate
            fire detections.  If ``None``, fire detections are NOT confirmed.

        Returns
        -------
        list[dict]
            Each dict has keys:
              class       – "person" or "fire"
              confidence  – float
              bbox        – [x1, y1, x2, y2]
              center      – [cx, cy]
              confirmed   – bool
              thermal_intensity – float or None
        """
        detections = []

        # --- Fire detections (AIDER model) --------------------------------
        fire_results = self._fire_model.predict(
            rgb_frame,
            device=self._device,
            verbose=False,
            conf=self._fire_conf,
        )
        for result in fire_results:
            boxes = result.boxes
            if boxes is None:
                continue
            i = 0
            while i < len(boxes):
                cls_id = int(boxes.cls[i].item())
                if cls_id in AIDER_KEEP_CLASSES:
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bbox = [x1, y1, x2, y2]

                    # Cross-validate with thermal
                    confirmed = False
                    thermal_val = None
                    if thermal_hotspots is not None:
                        overlap = _check_thermal_overlap(bbox, thermal_hotspots)
                        if overlap is not None:
                            confirmed = True
                            thermal_val = overlap

                    detections.append({
                        "class": "fire",
                        "confidence": conf,
                        "bbox": bbox,
                        "center": [cx, cy],
                        "confirmed": confirmed,
                        "thermal_intensity": thermal_val,
                    })
                i = i + 1

        # --- Person detections (COCO model) -------------------------------
        person_results = self._person_model.predict(
            rgb_frame,
            device=self._device,
            verbose=False,
            conf=self._person_conf,
        )
        for result in person_results:
            boxes = result.boxes
            if boxes is None:
                continue
            i = 0
            while i < len(boxes):
                cls_id = int(boxes.cls[i].item())
                if cls_id == COCO_PERSON_CLASS:
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    detections.append({
                        "class": "person",
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "confirmed": True,  # person needs only YOLO
                        "thermal_intensity": None,
                    })
                i = i + 1

        return detections

    def detect_fire_only(self, rgb_frame):
        """Convenience: return only fire detections (no person model run)."""
        results = self._fire_model.predict(
            rgb_frame,
            device=self._device,
            verbose=False,
            conf=self._fire_conf,
        )
        dets = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            i = 0
            while i < len(boxes):
                cls_id = int(boxes.cls[i].item())
                if cls_id in AIDER_KEEP_CLASSES:
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    dets.append({
                        "class": "fire",
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                        "confirmed": False,
                        "thermal_intensity": None,
                    })
                i = i + 1
        return dets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_thermal_overlap(bbox, hotspots):
    """Return thermal intensity if *bbox* overlaps any hotspot, else None.

    Overlap is determined by checking whether the hotspot centre falls
    within the enlarged bounding box (20% padding on each side).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * 0.2
    pad_y = h * 0.2
    ex1 = x1 - pad_x
    ey1 = y1 - pad_y
    ex2 = x2 + pad_x
    ey2 = y2 + pad_y

    best_intensity = None
    idx = 0
    while idx < len(hotspots):
        hs = hotspots[idx]
        hcx, hcy = hs["center"]
        if ex1 <= hcx <= ex2 and ey1 <= hcy <= ey2:
            intensity = hs.get("temperature_estimate", 0.0)
            if best_intensity is None or intensity > best_intensity:
                best_intensity = intensity
        idx = idx + 1

    return best_intensity
