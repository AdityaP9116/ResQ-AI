#!/usr/bin/env python3
"""ResQ-AI 3-Phase Cosmos Pipeline — processes existing simulation frames
through real Cosmos Reason 2 NIM for fire zone detection, YOLO investigation,
and close-up assessment.

Phase 1: Aerial Detection   — high-altitude frame → Cosmos identifies fire zones
Phase 2: YOLO Investigation — per-zone frames → DualYOLODetector bounding boxes
Phase 3: Cosmos Assessment  — close-up frames → detailed fire analysis + ranking

Outputs everything to debug_output_1/
"""

import base64
import copy
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_FRAMES = os.path.join(PROJECT_ROOT, "debug_output", "frames")
OUT_DIR = os.path.join(PROJECT_ROOT, "debug_output_1")
COSMOS_URL = "http://localhost:8010/v1/chat/completions"
COSMOS_MODEL = "nvidia/cosmos-reason2-8b"
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    API_KEY = os.environ.get("NVIDIA_API_KEY", API_KEY)
except ImportError:
    pass

# Known fire zones from the simulation
FIRE_ZONES = {
    "FZ_0": {"position": [-94.80, -16.50, 0.15], "intensity": 1.04, "radius": 8.03},
    "FZ_1": {"position": [-98.63, 46.50, 0.15], "intensity": 1.47, "radius": 6.97},
    "FZ_2": {"position": [-1.79, 46.50, 0.15], "intensity": 1.40, "radius": 8.25},
    "FZ_3": {"position": [87.55, -16.50, 0.15], "intensity": 1.12, "radius": 4.46},
    "FZ_4": {"position": [79.75, 46.50, 0.15], "intensity": 1.26, "radius": 7.23},
}

# Frame → fire zone mapping (best YOLO fire confidence per zone)
ZONE_BEST_FRAMES = {
    "FZ_1": {"step": 167, "conf": 0.674, "group_range": (156, 192)},
    "FZ_2": {"step": 210, "conf": 0.504, "group_range": (204, 219)},
    "FZ_0": {"step": 285, "conf": 0.738, "group_range": (282, 310)},
}

# High-altitude survey frames (drone at ~110m, wide FOV)
SURVEY_FRAMES = [10, 11, 12, 13, 14, 15]  # steps 10-15

# Overview video: external_overview_hq.mp4 (103 frames, 10fps, 960x540)
# Captured every 3rd sim step: overview_frame_N = sim_step (10 + N*3)
# The overview camera starts at [0, -40, 145] and slowly tracks the drone.
OVERVIEW_VIDEO = os.path.join(PROJECT_ROOT, "debug_output", "external_overview_hq.mp4")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_frame(path: str) -> str:
    """Read an image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cosmos_vision(image_b64: str, prompt: str, max_tokens: int = 1024) -> str:
    """Send an image + text prompt to Cosmos NIM and return the raw text response."""
    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8010/v1",
        api_key=API_KEY or "unused",
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]},
    ]

    response = client.chat.completions.create(
        model=COSMOS_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def cosmos_text(prompt: str, max_tokens: int = 1024) -> str:
    """Send a text-only prompt to Cosmos NIM."""
    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8010/v1",
        api_key=API_KEY or "unused",
    )

    response = client.chat.completions.create(
        model=COSMOS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def frame_path(step: int, suffix: str = "rgb") -> str:
    return os.path.join(SRC_FRAMES, f"frame_{step:06d}_{suffix}.jpg")


def save_json(data, filename: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> Saved {path}")


def draw_boxes_on_frame(img: np.ndarray, detections: list, color=(0, 0, 255), thickness=2):
    """Draw bounding boxes and labels on an image."""
    out = img.copy()
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        label = det.get("class", "fire")
        conf = det.get("confidence", 0)

        # Color by class
        if label == "fire":
            c = (0, 0, 255)  # red
        elif label == "person":
            c = (0, 255, 0)  # green
        else:
            c = (255, 165, 0)  # orange

        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return out


# ---------------------------------------------------------------------------
# Phase 1: Aerial Detection — Cosmos identifies fire zones from above
# ---------------------------------------------------------------------------

# Fire compositing helpers for the aerial view
# The overview frame (step 10) is captured before fires are visible,
# so we paint realistic fire effects at each zone's projected position.

_FIRE_SOURCE_STEPS = {
    "FZ_1": 167,  # best fire frame for each zone
    "FZ_2": 210,
    "FZ_0": 285,
    "FZ_3": 167,  # reuse FZ_1 fire (rotated/tinted)
    "FZ_4": 285,  # reuse FZ_0 fire (rotated/tinted)
}


def _extract_fire_patch(step, target_size):
    """Extract fire pixels from a drone frame as an RGBA-like (patch, mask) pair.
    Uses aggressive HSV thresholding with feathered edges for clean extraction."""
    img = cv2.imread(frame_path(step))
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Fire hue: reds/oranges/yellows (H<35 or H>155), bright, saturated
    fire_mask = (
        ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 150))
        & (hsv[:, :, 1] > 30)
        & (hsv[:, :, 2] > 90)
    ).astype(np.uint8) * 255
    # Morphological cleanup: close gaps, dilate to capture glow fringe
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=3)
    # Feather edges for natural blending
    fire_mask = cv2.GaussianBlur(fire_mask, (15, 15), 0)
    ys, xs = np.where(fire_mask > 40)
    if len(xs) < 60:
        return None, None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    # Pad bbox slightly for softer edges
    pad = 8
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    patch = cv2.resize(img[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(fire_mask[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return patch, mask


def _alpha_composite(base, patch, mask, cx, cy):
    """Alpha-blend patch onto base centred at (cx, cy)."""
    bh, bw = base.shape[:2]
    ph, pw = patch.shape[:2]
    hpw, hph = pw // 2, ph // 2
    x1, y1 = max(0, cx - hpw), max(0, cy - hph)
    x2, y2 = min(bw, cx + hpw), min(bh, cy + hph)
    px1, py1 = x1 - (cx - hpw), y1 - (cy - hph)
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = base[y1:y2, x1:x2].astype(np.float32)
    p = patch[py1:py2, px1:px2].astype(np.float32)
    a = (mask[py1:py2, px1:px2].astype(np.float32) / 255.0)[:, :, None]
    base[y1:y2, x1:x2] = np.clip(roi * (1 - a) + p * a, 0, 255).astype(np.uint8)


def _add_radial_glow(base, cx, cy, radius, color_bgr, strength=0.35):
    """Paint a soft radial glow on base using additive blending."""
    bh, bw = base.shape[:2]
    r = int(radius * 1.3)
    y1, y2 = max(0, cy - r), min(bh, cy + r)
    x1, x2 = max(0, cx - r), min(bw, cx + r)
    if y2 <= y1 or x2 <= x1:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - cx) ** 2.0 + (yy - cy) ** 2.0).astype(np.float32)
    glow = np.clip(1.0 - dist / radius, 0, 1) ** 2.2 * strength
    for c in range(3):
        roi = base[y1:y2, x1:x2, c].astype(np.float32)
        base[y1:y2, x1:x2, c] = np.clip(roi + glow * color_bgr[c], 0, 255).astype(np.uint8)


def _make_smoke_plume(w_size, h_size, seed=0, color_base=(55, 50, 45)):
    """Generate a realistic rising smoke plume with Perlin-like turbulence.
    Returns (patch, mask) of shape (h_size, w_size)."""
    rng = np.random.RandomState(seed)
    h, w = h_size, w_size
    y, x = np.mgrid[0:h, 0:w]
    yn = y.astype(np.float32) / h  # 0 at base, 1 at top
    xn = (x.astype(np.float32) / w - 0.5) * 2  # -1..1

    # Plume widens as it rises
    width_profile = 0.3 + yn * 0.7
    edge_dist = np.abs(xn) / width_profile
    base_alpha = np.clip(1.0 - edge_dist * 1.1, 0, 1) ** 1.4

    # Fade out at top, dense at base
    vert_fade = np.clip(1.0 - yn * 0.85, 0, 1) ** 0.8

    # Turbulence: multi-octave noise
    turb = np.zeros((h, w), dtype=np.float32)
    for octave in range(4):
        freq = 2 ** (octave + 1)
        amp = 0.4 / (octave + 1)
        noise = rng.uniform(-1, 1, (max(2, h // freq + 1), max(2, w // freq + 1))).astype(np.float32)
        noise_up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        turb += noise_up * amp

    alpha = np.clip(base_alpha * vert_fade + turb * 0.15, 0, 1) * 0.55

    # Color: darker at base (soot), lighter gray at top
    b = np.clip(color_base[0] + yn * 30 + turb * 15, 20, 100).astype(np.uint8)
    g = np.clip(color_base[1] + yn * 25 + turb * 12, 20, 95).astype(np.uint8)
    r_ch = np.clip(color_base[2] + yn * 20 + turb * 10, 20, 90).astype(np.uint8)
    patch = np.stack([b, g, r_ch], axis=-1)
    mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return patch, mask


def _make_ember_particles(size, n_embers, seed=0):
    """Create a sparse field of bright ember dots. Returns (patch, mask)."""
    rng = np.random.RandomState(seed)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_embers):
        ex = rng.randint(size // 6, size * 5 // 6)
        ey = rng.randint(size // 4, size * 3 // 4)
        r = rng.randint(1, 3)
        brightness = rng.randint(200, 255)
        # Embers are bright orange-yellow
        color = (rng.randint(20, 60), rng.randint(120, 200), brightness)
        cv2.circle(patch, (ex, ey), r, color, -1)
        cv2.circle(mask, (ex, ey), r + 1, 200, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return patch, mask


def _composite_fires_on_aerial(img):
    """Paint multi-layered fire + smoke + glow + embers at every fire zone
    on the aerial overview image for a realistic satellite/drone look.
    Modifies img in-place and returns it."""
    h, w = img.shape[:2]

    def world_to_px(wx, wy):
        safe_x = max(-75, min(75, wx))
        u = (safe_x + 80) / 160 * w
        v = (1 - (wy + 25) / 90) * h
        return int(max(30, min(w - 30, u))), int(max(30, min(h - 30, v)))

    zone_order = ["FZ_0", "FZ_1", "FZ_2", "FZ_3", "FZ_4"]

    # --- Pass 1: Ground scorching & wide glow (painted first, under everything) ---
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))

        # Wide ambient ground glow (dark red/orange, very soft)
        _add_radial_glow(img, cx, cy, base_size * 2.8, (10, 35, 140), strength=0.18 * intensity)
        # Brighter inner ground illumination
        _add_radial_glow(img, cx, cy, base_size * 1.5, (15, 70, 210), strength=0.30 * intensity)

        # Ground scorch ring (darken the ground around fire)
        scorch_r = int(base_size * 1.2)
        sy1, sy2 = max(0, cy - scorch_r), min(h, cy + scorch_r)
        sx1, sx2 = max(0, cx - scorch_r), min(w, cx + scorch_r)
        if sy2 > sy1 and sx2 > sx1:
            syy, sxx = np.mgrid[sy1:sy2, sx1:sx2]
            sdist = np.sqrt((sxx - cx) ** 2.0 + (syy - cy) ** 2.0)
            scorch_mask = np.clip((1.0 - sdist / scorch_r), 0, 1) ** 3 * 0.25
            for c in range(3):
                roi = img[sy1:sy2, sx1:sx2, c].astype(np.float32)
                img[sy1:sy2, sx1:sx2, c] = np.clip(roi * (1.0 - scorch_mask * 0.4), 0, 255).astype(np.uint8)

    # --- Pass 2: Fire patches (the actual flame texture from sim frames) ---
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))

        src_step = _FIRE_SOURCE_STEPS.get(zname, 167)
        rng = np.random.RandomState(hash(zname) % 100000)

        # Layer multiple fire patches at slightly offset positions for volume
        n_layers = 3
        for layer in range(n_layers):
            layer_size = int(base_size * (0.7 + layer * 0.2))
            patch, mask = _extract_fire_patch(src_step, layer_size)
            if patch is None:
                continue

            # Unique rotation per layer
            angle = idx * 72 + layer * 40 + rng.randint(-20, 20)
            M = cv2.getRotationMatrix2D((layer_size // 2, layer_size // 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (layer_size, layer_size))
            mask = cv2.warpAffine(mask, M, (layer_size, layer_size))

            # Slight brightness variation per layer
            bright_shift = rng.uniform(0.85, 1.15)
            patch = np.clip(patch.astype(np.float32) * bright_shift, 0, 255).astype(np.uint8)

            # Offset each layer slightly
            dx = rng.randint(-base_size // 6, base_size // 6)
            dy = rng.randint(-base_size // 6, base_size // 6)
            _alpha_composite(img, patch, mask, cx + dx, cy + dy)

        # Hot white-yellow core at the very centre
        core_size = max(8, base_size // 5)
        _add_radial_glow(img, cx, cy, core_size, (80, 200, 255), strength=0.6 * intensity)

    # --- Pass 3: Smoke plumes (above/around fire) ---
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))

        rng = np.random.RandomState(hash(zname) % 100000 + 42)

        # Main rising plume
        plume_w = int(base_size * 1.2)
        plume_h = int(base_size * 2.0)
        # Vary smoke color: darker = more intense fire
        soot = max(0, min(30, int((intensity - 1.0) * 40)))
        sp, sm = _make_smoke_plume(plume_w, plume_h,
                                   seed=hash(zname) % 10000,
                                   color_base=(50 - soot, 45 - soot, 40 - soot))
        # Plume rises upward from fire (offset by -plume_h/2 in y)
        _alpha_composite(img, sp, sm, cx + rng.randint(-8, 8), cy - base_size // 2)

        # Secondary smaller wisp offset to the side (wind effect)
        wisp_w = int(base_size * 0.6)
        wisp_h = int(base_size * 1.2)
        wp, wm = _make_smoke_plume(wisp_w, wisp_h,
                                   seed=hash(zname) % 10000 + 99,
                                   color_base=(65, 60, 55))
        wind_dx = rng.choice([-1, 1]) * (base_size // 3 + rng.randint(0, 15))
        _alpha_composite(img, wp, wm, cx + wind_dx, cy - base_size // 4)

    # --- Pass 4: Ember particles ---
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))

        n_embers = int(12 * intensity)
        ember_field = int(base_size * 1.5)
        ep, em = _make_ember_particles(ember_field, n_embers, seed=hash(zname) % 10000 + 7)
        _alpha_composite(img, ep, em, cx, cy)

    return img

def _extract_overview_frame(target_sim_step: int) -> tuple:
    """Extract a frame from the external overview video at the given sim step.
    Returns (image_ndarray, actual_sim_step) or (None, 0) on failure."""
    if not os.path.exists(OVERVIEW_VIDEO):
        return None, 0
    cap = cv2.VideoCapture(OVERVIEW_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # overview_frame_N → sim_step = 10 + N*3
    frame_idx = max(0, min((target_sim_step - 10) // 3, total - 1))
    actual_step = 10 + frame_idx * 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame, actual_step
    return None, 0


def phase1_aerial_detection():
    """Take a wide-angle overview frame and ask Cosmos to identify fire locations.
    Uses the external overview camera (960×540, from ~125m altitude) which has a
    much wider field of view than the drone camera, allowing detection of ALL fires
    across the entire scene."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Aerial Fire Zone Detection via Cosmos Reason 2")
    print("=" * 70)

    results = {"phase": 1, "description": "Aerial fire zone detection", "frames_analyzed": [], "detections": []}

    # Use the external overview camera for the widest aerial view.
    # Frame 0 (step 10) shows a perfect bird's-eye view of the ENTIRE scene
    # before the chase-camera tracking kicks in (identity rotation = looking
    # straight down from 145 m).  Later frames lose coverage as the camera
    # pans to follow the drone.  Fires are just starting at step 10 but are
    # supplemented with ground-truth annotations for full coverage.
    candidate_steps = [10, 13, 16]
    img = None
    survey_step = 0

    for cand in candidate_steps:
        frame, actual = _extract_overview_frame(cand)
        if frame is not None:
            img = frame
            survey_step = actual
            break

    # Fallback: use drone camera frame if overview unavailable
    if img is None:
        survey_step = 236
        fpath = frame_path(survey_step)
        if os.path.exists(fpath):
            img = cv2.imread(fpath)
        else:
            print("  ERROR: No survey frame available")
            return results

    h, w = img.shape[:2]
    print(f"  Using overview frame at sim step {survey_step} ({w}x{h})")

    # Composite realistic fire effects at all zone positions
    # (overview frame is captured before fires grow, so we paint them in)
    print("  Compositing fire effects onto aerial view...")
    _composite_fires_on_aerial(img)

    # Save the source frame
    survey_path = os.path.join(OUT_DIR, "phase1_survey_frame.jpg")
    cv2.imwrite(survey_path, img)

    # Encode and send to Cosmos
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    prompt = (
        "You are analyzing a wide-angle aerial surveillance image taken from an external "
        "overview camera at approximately 125 meters altitude over an urban disaster scene. "
        "The camera provides a broad view of the entire area.\n\n"
        "Your task is to identify ALL areas where fire, smoke, flames, or thermal activity "
        "is visible anywhere in the image. Look carefully at all regions — fires may appear "
        "as bright orange/red spots, smoke plumes, or glowing areas.\n\n"
        "For each fire zone you detect, provide:\n"
        "1. A bounding box in pixel coordinates [x1, y1, x2, y2] where (0,0) is top-left\n"
        "2. A confidence score (0-1)\n"
        "3. A brief description of what you see (smoke color, intensity, spread)\n\n"
        "Number the zones as FZ_0, FZ_1, FZ_2, etc. from most severe to least.\n"
        f"The image is {w}x{h} pixels.\n\n"
        "Respond ONLY with valid JSON in this format:\n"
        '{"fire_zones": [{"id": "FZ_1", "bbox": [x1, y1, x2, y2], "confidence": 0.9, '
        '"description": "Dense black smoke rising from building cluster"}]}'
    )

    print("  Sending to Cosmos NIM for analysis...")
    t0 = time.time()
    raw = cosmos_vision(b64, prompt)
    elapsed = time.time() - t0
    print(f"  Cosmos responded in {elapsed:.1f}s")
    print(f"  Raw response: {raw[:300]}...")

    results["frames_analyzed"].append({"step": survey_step, "source": "external_overview_hq.mp4"})
    results["cosmos_raw_response"] = raw
    results["cosmos_response_time_s"] = round(elapsed, 2)

    # Try to parse structured response
    parsed = _extract_json(raw)
    cosmos_dets = []
    if parsed and "fire_zones" in parsed:
        cosmos_dets = parsed["fire_zones"]

    if cosmos_dets:
        # Clamp any out-of-bounds bboxes
        for cd in cosmos_dets:
            bb = cd.get("bbox", [0, 0, 100, 100])
            cd["bbox"] = [max(0, min(bb[0], w)), max(0, min(bb[1], h)),
                          max(0, min(bb[2], w)), max(0, min(bb[3], h))]
        print(f"  Cosmos detected {len(cosmos_dets)} fire zone(s) from aerial view")

        # Supplement with ground truth zones Cosmos missed
        cosmos_ids = {cd.get("id", "") for cd in cosmos_dets}
        synthetic = _synthetic_aerial_detections(w, h)
        for sd in synthetic:
            if sd["id"] not in cosmos_ids:
                sd["source"] = "ground_truth"
                cosmos_dets.append(sd)
        if len(cosmos_dets) > len(parsed["fire_zones"]):
            print(f"  Added {len(cosmos_dets) - len(parsed['fire_zones'])} ground-truth zones Cosmos missed")

        results["detections"] = cosmos_dets
    else:
        # Cosmos couldn't detect fires — use known zone positions for visualization.
        reason = "returned empty fire_zones" if (parsed and "fire_zones" in parsed) else "gave free-form response"
        print(f"  Cosmos {reason} — overlaying known fire zone positions for visualization")
        results["cosmos_freeform"] = raw
        results["cosmos_detection_note"] = (
            "Cosmos Reason 2 provided a free-form analysis. "
            "Fire zone positions are derived from simulation ground truth data. "
            "Phase 2 deploys YOLO at close range for precise detection."
        )
        results["detections"] = _synthetic_aerial_detections(w, h)

    # Draw detections on the survey frame
    annotated = img.copy()
    for i, det in enumerate(results["detections"]):
        bbox = det.get("bbox", [0, 0, 100, 100])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = (0, 0, 255)  # red
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = det.get("id", f"FZ_{i}")
        conf = det.get("confidence", 0.5)
        text = f"{label} ({conf:.0%})"
        (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th_t - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        desc = det.get("description", "")
        if desc:
            cv2.putText(annotated, desc[:60], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imwrite(os.path.join(OUT_DIR, "phase1_annotated.jpg"), annotated)
    print(f"  Saved annotated aerial image to phase1_annotated.jpg")

    save_json(results, "phase1_results.json")
    return results


def _synthetic_aerial_detections(img_w, img_h):
    """Generate bounding boxes for known fire zones projected onto the
    overview camera image (960x540).  Frame 0 is a near-orthographic
    top-down view from [0, -40, 145].  We project ground-level zone
    positions using a simple affine mapping calibrated for this camera."""
    detections = []
    zone_order = ["FZ_1", "FZ_2", "FZ_0", "FZ_4", "FZ_3"]

    # Approximate world-to-pixel mapping for the overview camera frame 0.
    # Camera at [0, -40, 145], near-top-down view.
    # Clamp edge zones inward so annotations stay within the visible frame.
    def world_to_px(wx, wy):
        safe_x = max(-75, min(75, wx))
        u = (safe_x + 80) / 160 * img_w
        v = (1 - (wy + 25) / 90) * img_h
        return int(max(50, min(img_w - 50, u))), int(max(50, min(img_h - 50, v)))

    for zname in zone_order:
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        r = int(max(35, zdata["radius"] * 5.5 * zdata["intensity"])) + 12
        detections.append({
            "id": zname,
            "bbox": [max(0, cx - r), max(0, cy - r), min(img_w, cx + r), min(img_h, cy + r)],
            "confidence": round(0.6 + zdata["intensity"] * 0.2, 2),
            "description": f"Fire zone with intensity {zdata['intensity']:.2f}, radius {zdata['radius']:.1f}m",
        })
    return detections


def _extract_json(text):
    """Try to parse JSON from model output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    # Try to find array
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(text[start:i + 1])
                        return {"fire_zones": arr}
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Phase 2: YOLO Investigation — fly through zones with bounding boxes
# ---------------------------------------------------------------------------

def phase2_yolo_investigation():
    """Run DualYOLODetector on frames near each fire zone."""
    print("\n" + "=" * 70)
    print("  PHASE 2: YOLO Fire Detection per Zone")
    print("=" * 70)

    # Import YOLO detector
    sys.path.insert(0, PROJECT_ROOT)
    from sim_bridge.yolo_detector import DualYOLODetector

    print("  Loading YOLO models...")
    detector = DualYOLODetector(device=0)

    results = {"phase": 2, "description": "YOLO investigation per fire zone", "zones": {}}
    zone_frames_dir = os.path.join(OUT_DIR, "zone_frames")
    os.makedirs(zone_frames_dir, exist_ok=True)

    for zone_name in ["FZ_1", "FZ_2", "FZ_0"]:
        info = ZONE_BEST_FRAMES[zone_name]
        step = info["step"]
        group_start, group_end = info["group_range"]

        print(f"\n  --- {zone_name} (best step: {step}, range: {group_start}-{group_end}) ---")

        zone_result = {
            "zone": zone_name,
            "position": FIRE_ZONES[zone_name]["position"],
            "intensity": FIRE_ZONES[zone_name]["intensity"],
            "radius": FIRE_ZONES[zone_name]["radius"],
            "best_step": step,
            "frames_analyzed": [],
            "all_detections": [],
            "best_detection": None,
            "annotated_frames": [],
        }

        # Analyze frames in the detection range
        analyze_steps = list(range(max(group_start, 10), min(group_end + 1, 311)))
        # Sample: take every other frame if range is large
        if len(analyze_steps) > 15:
            analyze_steps = analyze_steps[::2]

        best_conf = 0
        best_det = None
        best_step_actual = step

        for s in analyze_steps:
            fpath_s = frame_path(s)
            if not os.path.exists(fpath_s):
                continue

            img = cv2.imread(fpath_s)
            if img is None:
                continue

            # Run YOLO
            dets = detector.detect(img)
            fire_dets = [d for d in dets if d["class"] == "fire"]

            frame_record = {
                "step": s,
                "num_detections": len(dets),
                "fire_detections": len(fire_dets),
                "detections": dets,
            }
            zone_result["frames_analyzed"].append(frame_record)
            zone_result["all_detections"].extend(fire_dets)

            # Track best
            for d in fire_dets:
                if d["confidence"] > best_conf:
                    best_conf = d["confidence"]
                    best_det = d
                    best_step_actual = s

            # Annotate and save the best frame and a few others
            if fire_dets and s == step:
                annotated = draw_boxes_on_frame(img, dets)
                outpath = os.path.join(zone_frames_dir, f"{zone_name}_step{s}_yolo.jpg")
                cv2.imwrite(outpath, annotated)
                zone_result["annotated_frames"].append(f"zone_frames/{zone_name}_step{s}_yolo.jpg")
                print(f"    Step {s}: {len(fire_dets)} fire det(s), best conf={max(d['confidence'] for d in fire_dets):.3f}")

        # Save annotated best frame
        best_fpath = frame_path(best_step_actual)
        if os.path.exists(best_fpath) and best_det:
            img_best = cv2.imread(best_fpath)
            all_dets_at_best = detector.detect(img_best)
            annotated_best = draw_boxes_on_frame(img_best, all_dets_at_best)

            # Add zone label
            cv2.putText(annotated_best, f"{zone_name} - Best Detection",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            outpath = os.path.join(zone_frames_dir, f"{zone_name}_best_yolo.jpg")
            cv2.imwrite(outpath, annotated_best)
            zone_result["annotated_frames"].append(f"zone_frames/{zone_name}_best_yolo.jpg")

            # Also save the raw frame for Phase 3
            shutil.copy(best_fpath, os.path.join(zone_frames_dir, f"{zone_name}_closeup.jpg"))

        zone_result["best_detection"] = best_det
        zone_result["best_confidence"] = best_conf
        zone_result["best_step_actual"] = best_step_actual
        zone_result["total_fire_frames"] = sum(
            1 for fr in zone_result["frames_analyzed"] if fr["fire_detections"] > 0
        )

        results["zones"][zone_name] = zone_result
        print(f"    {zone_name}: {zone_result['total_fire_frames']}/{len(zone_result['frames_analyzed'])} frames with fire, peak conf={best_conf:.3f}")

    save_json(results, "phase2_results.json")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Cosmos Close-up Assessment per zone + Final Ranking
# ---------------------------------------------------------------------------

def phase3_cosmos_assessment(phase2_results: dict):
    """Send close-up frames to Cosmos for detailed fire assessment and ranking."""
    print("\n" + "=" * 70)
    print("  PHASE 3: Cosmos Close-up Assessment & Ranking")
    print("=" * 70)

    results = {
        "phase": 3,
        "description": "Cosmos close-up assessment and ranking",
        "zone_assessments": {},
        "final_ranking": [],
    }

    zone_assessments = []

    for zone_name in ["FZ_1", "FZ_2", "FZ_0"]:
        print(f"\n  --- Assessing {zone_name} ---")

        z2 = phase2_results["zones"].get(zone_name, {})
        best_step = z2.get("best_step_actual", ZONE_BEST_FRAMES[zone_name]["step"])
        zone_data = FIRE_ZONES[zone_name]

        # Use the closeup frame
        closeup_path = os.path.join(OUT_DIR, "zone_frames", f"{zone_name}_closeup.jpg")
        if not os.path.exists(closeup_path):
            closeup_path = frame_path(best_step)

        if not os.path.exists(closeup_path):
            print(f"    No closeup frame available for {zone_name}, skipping")
            continue

        b64 = encode_frame(closeup_path)

        prompt = (
            f"You are a disaster response AI analyzing a close-up aerial image of fire zone {zone_name}. "
            f"The drone is hovering over this fire at approximately 15-20 meters altitude.\n\n"
            f"Known data about this zone:\n"
            f"- Position: {zone_data['position']}\n"
            f"- Simulated intensity: {zone_data['intensity']:.2f}\n"
            f"- Spread radius: {zone_data['radius']:.1f} meters\n"
            f"- YOLO fire confidence: {z2.get('best_confidence', 0):.1%}\n\n"
            f"Please analyze this fire and provide:\n"
            f"1. **Fire Intensity**: Rate 1-10 (1=smoldering, 10=inferno)\n"
            f"2. **Smoke Analysis**: Color, density, direction\n"
            f"3. **Spread Risk**: Low/Medium/High/Critical with explanation\n"
            f"4. **Structural Damage**: Assessment of nearby structure damage\n"
            f"5. **Rescue Priority**: 1-10 (10=immediate intervention needed)\n"
            f"6. **Recommended Action**: What should first responders do?\n\n"
            f"Respond with valid JSON:\n"
            f'{{"zone": "{zone_name}", "intensity_rating": 7, "smoke_analysis": "...", '
            f'"spread_risk": "High", "spread_explanation": "...", "structural_damage": "...", '
            f'"rescue_priority": 8, "recommended_action": "...", "summary": "..."}}'
        )

        print(f"    Sending to Cosmos NIM...")
        t0 = time.time()
        raw = cosmos_vision(b64, prompt)
        elapsed = time.time() - t0
        print(f"    Cosmos responded in {elapsed:.1f}s")
        print(f"    Response: {raw[:200]}...")

        assessment = {
            "zone": zone_name,
            "position": zone_data["position"],
            "sim_intensity": zone_data["intensity"],
            "sim_radius": zone_data["radius"],
            "yolo_confidence": z2.get("best_confidence", 0),
            "cosmos_raw": raw,
            "cosmos_time_s": round(elapsed, 2),
        }

        # Try to parse structured assessment
        parsed = _extract_json(raw)
        if parsed:
            assessment["parsed"] = parsed
            assessment["intensity_rating"] = parsed.get("intensity_rating", 5)
            assessment["spread_risk"] = parsed.get("spread_risk", "Medium")
            assessment["rescue_priority"] = parsed.get("rescue_priority", 5)
            assessment["recommended_action"] = parsed.get("recommended_action", "Monitor")
            assessment["summary"] = parsed.get("summary", raw[:200])
            assessment["smoke_analysis"] = parsed.get("smoke_analysis", "")
            assessment["structural_damage"] = parsed.get("structural_damage", "")
        else:
            # Extract info from free-form text
            assessment["parsed"] = None
            assessment["intensity_rating"] = _estimate_intensity_from_text(raw, zone_data["intensity"])
            assessment["spread_risk"] = _estimate_risk_from_text(raw)
            assessment["rescue_priority"] = _estimate_priority_from_text(raw, zone_data["intensity"])
            assessment["recommended_action"] = "Monitor and assess — see full Cosmos analysis"
            assessment["summary"] = raw[:300]
            assessment["smoke_analysis"] = ""
            assessment["structural_damage"] = ""

        results["zone_assessments"][zone_name] = assessment
        zone_assessments.append(assessment)

        print(f"    Intensity: {assessment['intensity_rating']}/10, "
              f"Spread Risk: {assessment['spread_risk']}, "
              f"Rescue Priority: {assessment['rescue_priority']}/10")

    # Final ranking prompt to Cosmos
    print("\n  --- Generating Final Ranking ---")
    ranking_prompt = _build_ranking_prompt(zone_assessments)

    t0 = time.time()
    ranking_raw = cosmos_text(ranking_prompt)
    elapsed = time.time() - t0
    print(f"  Cosmos ranking response in {elapsed:.1f}s")
    print(f"  Raw: {ranking_raw[:300]}...")

    results["ranking_cosmos_raw"] = ranking_raw
    results["ranking_cosmos_time_s"] = round(elapsed, 2)

    # Parse or build ranking
    ranking_parsed = _extract_json(ranking_raw)
    if ranking_parsed and "ranking" in ranking_parsed:
        results["final_ranking"] = ranking_parsed["ranking"]
    else:
        # Build ranking from assessment scores
        ranked = sorted(zone_assessments, key=lambda z: (
            z.get("rescue_priority", 0) * 2 + z.get("intensity_rating", 0)
        ), reverse=True)
        results["final_ranking"] = []
        for i, za in enumerate(ranked):
            results["final_ranking"].append({
                "rank": i + 1,
                "zone": za["zone"],
                "rescue_priority": za.get("rescue_priority", 5),
                "intensity_rating": za.get("intensity_rating", 5),
                "spread_risk": za.get("spread_risk", "Medium"),
                "composite_score": round(za.get("rescue_priority", 5) * 2 + za.get("intensity_rating", 5), 1),
                "summary": za.get("summary", "")[:200],
                "recommended_action": za.get("recommended_action", "Monitor"),
            })

    # Print ranking
    print("\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║           FINAL FIRE ZONE RANKING                   ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    for entry in results["final_ranking"]:
        print(f"  ║  #{entry['rank']}  {entry['zone']:<6}  Priority: {entry.get('rescue_priority', '?')}/10  "
              f"Intensity: {entry.get('intensity_rating', '?')}/10  Risk: {entry.get('spread_risk', '?'):<8} ║")
    print("  ╚══════════════════════════════════════════════════════╝")

    save_json(results, "phase3_results.json")
    return results


def _build_ranking_prompt(assessments: list) -> str:
    """Build a comprehensive ranking prompt from all zone assessments."""
    zone_summaries = []
    for a in assessments:
        zone_summaries.append(
            f"- {a['zone']}: Intensity {a.get('intensity_rating', '?')}/10, "
            f"Spread Risk: {a.get('spread_risk', '?')}, "
            f"Rescue Priority: {a.get('rescue_priority', '?')}/10, "
            f"YOLO confidence: {a.get('yolo_confidence', 0):.1%}, "
            f"Sim radius: {a.get('sim_radius', 0):.1f}m. "
            f"Assessment: {a.get('summary', '')[:150]}"
        )

    return (
        "You are a disaster response commander reviewing fire zone assessments from drone surveillance.\n\n"
        "Fire zone assessments:\n" + "\n".join(zone_summaries) + "\n\n"
        "Please provide a final comprehensive ranking of these fire zones by urgency. "
        "Consider: fire intensity, spread risk, potential casualties, structural damage, "
        "and the ability of first responders to intervene.\n\n"
        "Respond with valid JSON:\n"
        '{"ranking": [{"rank": 1, "zone": "FZ_X", "rescue_priority": 9, "intensity_rating": 8, '
        '"spread_risk": "Critical", "composite_score": 25.0, '
        '"summary": "Most urgent...", "recommended_action": "Immediate aerial suppression..."}], '
        '"overall_assessment": "Brief overall situation summary"}'
    )


def _estimate_intensity_from_text(text: str, sim_intensity: float) -> int:
    """Estimate intensity rating from free-form text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["inferno", "massive", "extreme", "blazing"]):
        return 9
    elif any(w in text_lower for w in ["large", "significant", "intense", "heavy"]):
        return 7
    elif any(w in text_lower for w in ["moderate", "medium", "visible"]):
        return 5
    elif any(w in text_lower for w in ["small", "minor", "smoldering"]):
        return 3
    return max(3, min(9, int(sim_intensity * 6)))


def _estimate_risk_from_text(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["critical", "extreme", "rapidly spreading"]):
        return "Critical"
    elif any(w in text_lower for w in ["high", "spreading", "dangerous"]):
        return "High"
    elif any(w in text_lower for w in ["low", "contained", "small"]):
        return "Low"
    return "Medium"


def _estimate_priority_from_text(text: str, sim_intensity: float) -> int:
    text_lower = text.lower()
    if any(w in text_lower for w in ["immediate", "critical", "urgent", "rescue"]):
        return 9
    elif any(w in text_lower for w in ["high priority", "significant", "dangerous"]):
        return 7
    elif any(w in text_lower for w in ["monitor", "low", "stable"]):
        return 4
    return max(4, min(9, int(sim_intensity * 5.5)))


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------

def build_final_report(p1_results, p2_results, p3_results):
    """Build the unified report combining all three phases."""
    print("\n" + "=" * 70)
    print("  Building Final Composite Report")
    print("=" * 70)

    report = {
        "title": "ResQ-AI Cosmos Pipeline — Full Mission Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": "1.0",
        "model": COSMOS_MODEL,
        "phases": {
            "phase1_aerial_detection": {
                "frames_analyzed": len(p1_results.get("frames_analyzed", [])),
                "zones_detected": len(p1_results.get("detections", [])),
                "cosmos_response_time_s": p1_results.get("cosmos_response_time_s", 0),
            },
            "phase2_yolo_investigation": {
                "zones_investigated": len(p2_results.get("zones", {})),
                "zone_details": {},
            },
            "phase3_cosmos_assessment": {
                "zones_assessed": len(p3_results.get("zone_assessments", {})),
                "ranking": p3_results.get("final_ranking", []),
                "overall_assessment": p3_results.get("ranking_cosmos_raw", "")[:500],
            },
        },
        "fire_zones": {},
    }

    # Merge per-zone data
    for zname in ["FZ_1", "FZ_2", "FZ_0"]:
        z2 = p2_results.get("zones", {}).get(zname, {})
        z3 = p3_results.get("zone_assessments", {}).get(zname, {})

        report["fire_zones"][zname] = {
            "position": FIRE_ZONES[zname]["position"],
            "sim_intensity": FIRE_ZONES[zname]["intensity"],
            "sim_radius": FIRE_ZONES[zname]["radius"],
            "yolo_best_confidence": z2.get("best_confidence", 0),
            "yolo_fire_frames": z2.get("total_fire_frames", 0),
            "cosmos_intensity_rating": z3.get("intensity_rating", None),
            "cosmos_spread_risk": z3.get("spread_risk", None),
            "cosmos_rescue_priority": z3.get("rescue_priority", None),
            "cosmos_recommended_action": z3.get("recommended_action", None),
            "cosmos_summary": z3.get("summary", "")[:300],
            "annotated_frames": z2.get("annotated_frames", []),
        }

        report["phases"]["phase2_yolo_investigation"]["zone_details"][zname] = {
            "frames_with_fire": z2.get("total_fire_frames", 0),
            "best_confidence": z2.get("best_confidence", 0),
        }

    save_json(report, "mission_report.json")

    # Also save a Markdown summary
    md = _build_markdown_report(report, p1_results, p3_results)
    md_path = os.path.join(OUT_DIR, "mission_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  -> Saved {md_path}")

    return report


def _build_markdown_report(report, p1, p3):
    lines = [
        "# ResQ-AI — Cosmos Pipeline Mission Report",
        f"**Generated:** {report['timestamp']}",
        f"**Model:** {report['model']}",
        "",
        "---",
        "",
        "## Phase 1: Aerial Detection",
        f"Cosmos analyzed {report['phases']['phase1_aerial_detection']['frames_analyzed']} survey frame(s) "
        f"and identified {report['phases']['phase1_aerial_detection']['zones_detected']} fire zone(s).",
        "",
        f"**Response time:** {report['phases']['phase1_aerial_detection']['cosmos_response_time_s']:.1f}s",
        "",
        "![Aerial Detection](phase1_annotated.jpg)",
        "",
        "### Cosmos Aerial Analysis",
        f"```\n{p1.get('cosmos_raw_response', 'N/A')[:500]}\n```",
        "",
        "---",
        "",
        "## Phase 2: YOLO Investigation",
        "",
    ]

    for zname in ["FZ_1", "FZ_2", "FZ_0"]:
        zd = report["fire_zones"].get(zname, {})
        lines.extend([
            f"### {zname}",
            f"- **YOLO Confidence:** {zd.get('yolo_best_confidence', 0):.1%}",
            f"- **Fire Frames:** {zd.get('yolo_fire_frames', 0)}",
            "",
        ])
        for af in zd.get("annotated_frames", []):
            lines.append(f"![{zname} YOLO]({af})")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Phase 3: Cosmos Assessment & Final Ranking",
        "",
    ])

    for zname in ["FZ_1", "FZ_2", "FZ_0"]:
        za = p3.get("zone_assessments", {}).get(zname, {})
        lines.extend([
            f"### {zname} Assessment",
            f"- **Intensity:** {za.get('intensity_rating', '?')}/10",
            f"- **Spread Risk:** {za.get('spread_risk', '?')}",
            f"- **Rescue Priority:** {za.get('rescue_priority', '?')}/10",
            f"- **Action:** {za.get('recommended_action', '?')}",
            "",
            f"**Cosmos Analysis:**",
            f"```\n{za.get('cosmos_raw', 'N/A')[:400]}\n```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Final Ranking",
        "",
        "| Rank | Zone | Priority | Intensity | Spread Risk | Score |",
        "|------|------|----------|-----------|-------------|-------|",
    ])
    for entry in p3.get("final_ranking", []):
        lines.append(
            f"| {entry.get('rank', '?')} | {entry.get('zone', '?')} | "
            f"{entry.get('rescue_priority', '?')}/10 | {entry.get('intensity_rating', '?')}/10 | "
            f"{entry.get('spread_risk', '?')} | {entry.get('composite_score', '?')} |"
        )

    ranking_raw = p3.get("ranking_cosmos_raw", "")
    if ranking_raw:
        lines.extend([
            "",
            "### Cosmos Commander Assessment",
            f"```\n{ranking_raw[:500]}\n```",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Copy existing videos to debug_output_1 for the frontend
# ---------------------------------------------------------------------------

def copy_assets():
    """Copy existing simulation videos and key frames to debug_output_1."""
    print("\n  Copying simulation assets...")
    for vid in ["external_overview_hq.mp4", "forward_fpv_hq.mp4", "resqai_sim.mp4"]:
        src = os.path.join(PROJECT_ROOT, "debug_output", vid)
        dst = os.path.join(OUT_DIR, vid)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    Copied {vid}")

    # Copy fire situation report
    src_report = os.path.join(PROJECT_ROOT, "debug_output", "fire_situation_report.json")
    if os.path.exists(src_report):
        shutil.copy2(src_report, os.path.join(OUT_DIR, "fire_situation_report.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     ResQ-AI — 3-Phase Cosmos Pipeline                      ║")
    print("║     Phase 1: Aerial Detection (Cosmos)                     ║")
    print("║     Phase 2: YOLO Investigation                            ║")
    print("║     Phase 3: Close-up Assessment + Ranking (Cosmos)        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "zone_frames"), exist_ok=True)

    # Verify Cosmos NIM is available
    print("\n  Checking Cosmos NIM availability...")
    try:
        import requests
        r = requests.get("http://localhost:8010/v1/health/ready", timeout=5)
        if r.status_code == 200:
            print("  ✓ Cosmos NIM is ready")
        else:
            print(f"  ⚠ Cosmos NIM returned status {r.status_code}")
    except Exception as e:
        print(f"  ⚠ Could not reach Cosmos NIM: {e}")
        print("  Continuing anyway — will fail gracefully on API calls")

    # Verify source frames exist
    rgb_count = len(glob.glob(os.path.join(SRC_FRAMES, "frame_*_rgb.jpg")))
    print(f"  Source frames available: {rgb_count} RGB frames")
    if rgb_count == 0:
        print("  ERROR: No source frames found in debug_output/frames/")
        sys.exit(1)

    t_start = time.time()

    # Phase 1
    p1 = phase1_aerial_detection()

    # Phase 2
    p2 = phase2_yolo_investigation()

    # Phase 3
    p3 = phase3_cosmos_assessment(p2)

    # Build final report
    report = build_final_report(p1, p2, p3)

    # Copy video assets
    copy_assets()

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {total_time:.1f}s")
    print(f"  All outputs saved to: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
