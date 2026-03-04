"""Synthetic thermal-infrared image generation from Replicator semantic segmentation.

Converts per-pixel semantic class IDs produced by the ``SemanticSegmentationCamera``
(via ``omni.replicator.core``) into a single-channel grayscale intensity image that
approximates a long-wave infrared (LWIR, 8-14 μm) thermal camera.

Thermal intensity mapping (uint8)::

    fire        -> 255   (white-hot)
    person      -> 200
    vehicle     -> 160
    building    -> 120
    vegetation  ->  60
    terrain     ->  50
    unlabeled   ->  20   (cold / sky / unknown)

Gaussian noise is added to simulate real IR sensor read noise and
non-uniformity correction residuals.

Usage::

    from sim_bridge.thermal_sim import generate_synthetic_thermal

    # --- Option A: raw Replicator annotator dict (from SemanticSegmentationCamera) ---
    seg_output = seg_cam.state["semantic_segmentation"]   # dict from annotator.get_data()
    thermal = generate_synthetic_thermal(seg_output)

    # --- Option B: plain numpy class-ID array (e.g. from an offline pipeline) ---
    import numpy as np
    ids = np.zeros((480, 640), dtype=np.int32)
    thermal = generate_synthetic_thermal(ids)
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Thermal intensity map
# ──────────────────────────────────────────────────────────────────────────

THERMAL_INTENSITY: dict[str, int] = {
    "fire":       255,
    "person":     200,
    "vehicle":    160,
    "building":   120,
    "vegetation":  60,
    "terrain":     50,
}

UNLABELED_INTENSITY: int = 20

# Default sensor-noise parameters (tuned to look realistic on 640×480 output)
_DEFAULT_NOISE_SIGMA: float = 4.0


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def generate_thermal_from_rgb(
    rgb_bgr: np.ndarray,
    *,
    noise_sigma: float = _DEFAULT_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic thermal image from an RGB (BGR-ordered) frame.

    Uses colour-temperature heuristics when semantic segmentation is
    unavailable or produces only ``unlabeled`` pixels:

    - **Red/orange** (fire)       → 230-255  (white-hot)
    - **Dark** (buildings/rubble) → 120-160  (solar absorption / warm)
    - **Green** (vegetation)      → 55-70    (cool)
    - **Blue** (water/flood)      → 25-40    (cold)
    - **Bright/grey** (concrete)  → 85-100   (ambient)
    - **Other / medium**          → 90-110   (baseline)

    Returns:
        ``uint8`` array of shape **(H, W)** with thermal intensities.
    """
    import cv2

    if rng is None:
        rng = np.random.default_rng()

    if rgb_bgr.ndim == 3 and rgb_bgr.shape[2] == 4:
        rgb_bgr = rgb_bgr[:, :, :3]

    hsv = cv2.cvtColor(rgb_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Start with a baseline proportional to brightness
    thermal = (v.astype(np.float32) * 0.35 + 70).clip(60, 130)

    # ---- Fire: red/orange hues, high saturation, high value ----
    fire_mask = ((h < 18) | (h > 160)) & (s > 70) & (v > 80)
    thermal[fire_mask] = 240 + rng.uniform(-10, 15, size=fire_mask.sum())

    # ---- Vegetation: green hues ----
    green_mask = (h > 28) & (h < 85) & (s > 35) & (v > 30)
    thermal[green_mask] = 60 + (v[green_mask].astype(np.float32) - 80) * 0.1

    # ---- Water / flood: blue hues ----
    blue_mask = (h > 85) & (h < 135) & (s > 30)
    thermal[blue_mask] = 30 + (v[blue_mask].astype(np.float32) * 0.05)

    # ---- Dark surfaces (buildings, rubble, asphalt) – solar absorption ----
    dark_mask = (v < 90) & (s < 70) & ~green_mask & ~blue_mask & ~fire_mask
    thermal[dark_mask] = 130 + (90 - v[dark_mask].astype(np.float32)) * 0.4

    # ---- Bright / light concrete / sidewalk ----
    bright_mask = (v > 190) & (s < 35) & ~fire_mask
    thermal[bright_mask] = 90

    # ---- Add sensor noise ----
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, thermal.shape)
        thermal = thermal + noise

    return np.clip(thermal, 0, 255).astype(np.uint8)


def generate_synthetic_thermal(
    semantic_input: np.ndarray | dict,
    *,
    noise_sigma: float = _DEFAULT_NOISE_SIGMA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert a semantic segmentation array into a synthetic thermal image.

    Args:
        semantic_input: Either

            * A **dict** returned by ``annotator.get_data()`` with keys
              ``"data"`` (int32 array, H×W or H×W×1) and ``"info"``
              containing ``"idToLabels"`` — the mapping from integer class ID
              to label metadata.  This is the native output format of the
              Replicator ``semantic_segmentation`` annotator.

            * A plain **numpy int array** (H×W) where each element is already
              a class ID.  In this mode, a default ``id → label`` mapping is
              built from the six ResQ-AI classes (IDs 1-6) with 0 = unlabeled.

        noise_sigma: Standard deviation of additive Gaussian noise (in
            intensity units, 0-255).  Set to 0 for a perfectly clean output.
        rng: Optional ``numpy.random.Generator`` for reproducible noise.

    Returns:
        A ``uint8`` numpy array of shape **(H, W)** with thermal intensities.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- Unpack Replicator annotator output if needed ---------------------
    if isinstance(semantic_input, dict):
        class_ids, id_to_label = _unpack_replicator_output(semantic_input)
    else:
        class_ids = np.asarray(semantic_input, dtype=np.int32)
        id_to_label = None

    if class_ids.ndim == 3:
        class_ids = class_ids[..., 0]

    if class_ids.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (H, W) or 3-D (H, W, 1) class-ID array, "
            f"got shape {class_ids.shape}"
        )

    # ---- Build the look-up table ------------------------------------------
    lut = _build_lut(id_to_label)

    # ---- Map class IDs → thermal intensity --------------------------------
    thermal = lut[np.clip(class_ids, 0, len(lut) - 1)]

    # ---- Add Gaussian sensor noise ----------------------------------------
    if noise_sigma > 0:
        noise = rng.normal(loc=0.0, scale=noise_sigma, size=thermal.shape)
        thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255)

    return thermal.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _unpack_replicator_output(output: dict) -> tuple[np.ndarray, dict[int, str]]:
    """Extract the class-ID array and id→label mapping from Replicator output.

    The annotator returns::

        {
            "data": np.ndarray(int32, H×W),
            "info": {
                "idToLabels": {
                    "0": {"class": "BACKGROUND"},
                    "1": {"class": "fire"},
                    ...
                }
            }
        }
    """
    data = output.get("data")
    if data is None:
        raise KeyError("Semantic segmentation output missing 'data' key")
    class_ids = np.asarray(data, dtype=np.int32)

    info = output.get("info", {})
    raw_map = info.get("idToLabels", {})

    id_to_label: dict[int, str] = {}
    for id_str, meta in raw_map.items():
        try:
            cid = int(id_str)
        except (ValueError, TypeError):
            continue
        if isinstance(meta, dict):
            label = meta.get("class", "")
        else:
            label = str(meta)
        id_to_label[cid] = label.lower().strip()

    return class_ids, id_to_label


_DEFAULT_ID_TO_LABEL: dict[int, str] = {
    0: "unlabeled",
    1: "fire",
    2: "person",
    3: "vehicle",
    4: "building",
    5: "vegetation",
    6: "terrain",
}


def _build_lut(id_to_label: dict[int, str] | None) -> np.ndarray:
    """Build a 1-D uint8 look-up table:  ``thermal_value = lut[class_id]``."""
    if id_to_label is None or len(id_to_label) == 0:
        id_to_label = _DEFAULT_ID_TO_LABEL

    max_id = max(id_to_label.keys()) if id_to_label else 0
    max_id = max(max_id, 256)
    lut = np.full(max_id + 1, UNLABELED_INTENSITY, dtype=np.uint8)

    for cid, raw_label in id_to_label.items():
        label = raw_label.lower().strip()
        matched = False
        for key, intensity in THERMAL_INTENSITY.items():
            if key in label:
                lut[cid] = intensity
                matched = True
                break
        if not matched:
            lut[cid] = UNLABELED_INTENSITY

    return lut


# ──────────────────────────────────────────────────────────────────────────
# Quick self-test / demo
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    H, W = 480, 640
    rng = np.random.default_rng(42)

    class_ids = np.zeros((H, W), dtype=np.int32)

    # Paint horizontal bands for each class
    band_h = H // 7
    class_ids[0 * band_h : 1 * band_h, :] = 1   # fire
    class_ids[1 * band_h : 2 * band_h, :] = 2   # person
    class_ids[2 * band_h : 3 * band_h, :] = 3   # vehicle
    class_ids[3 * band_h : 4 * band_h, :] = 4   # building
    class_ids[4 * band_h : 5 * band_h, :] = 5   # vegetation
    class_ids[5 * band_h : 6 * band_h, :] = 6   # terrain
    # remainder stays 0 → unlabeled

    thermal = generate_synthetic_thermal(class_ids, noise_sigma=4.0, rng=rng)

    print(f"Thermal image shape : {thermal.shape}")
    print(f"Thermal dtype       : {thermal.dtype}")
    print(f"Min / Max intensity : {thermal.min()} / {thermal.max()}")

    # Per-band mean intensity (should approximate the mapping values)
    labels = ["fire", "person", "vehicle", "building", "vegetation", "terrain", "unlabeled"]
    for i, label in enumerate(labels):
        band = thermal[i * band_h : (i + 1) * band_h, :]
        print(f"  {label:>12s}  mean={band.mean():.1f}  (expected {list(THERMAL_INTENSITY.values()) + [UNLABELED_INTENSITY]}[{i}])")

    # Also test the Replicator-dict path
    fake_rep_output = {
        "data": class_ids,
        "info": {
            "idToLabels": {
                "0": {"class": "BACKGROUND"},
                "1": {"class": "fire"},
                "2": {"class": "person"},
                "3": {"class": "vehicle"},
                "4": {"class": "building"},
                "5": {"class": "vegetation"},
                "6": {"class": "terrain"},
            }
        },
    }
    thermal2 = generate_synthetic_thermal(fake_rep_output, noise_sigma=0.0)
    assert thermal2.shape == (H, W)
    assert thermal2.dtype == np.uint8
    assert thermal2[0, 0] == 255  # fire band, no noise
    print("\nReplicator-dict path: OK")
    print("All checks passed.")
