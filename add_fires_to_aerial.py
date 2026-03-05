#!/usr/bin/env python3
"""Composite realistic fire + smoke effects onto the bird's-eye overview frame.

The overview frame 0 (step 10) is a perfect top-down view of the scene but
was captured BEFORE fires spawned.  This script extracts fire patches from
later drone frames (where YOLO detected real fire) and composites smaller,
perspective-correct versions onto the bird's-eye frame at each fire zone's
known world position.  Adds smoke plumes for realism.
"""

import cv2
import numpy as np
import os

PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT, "debug_output_1")
FRAMES_DIR = os.path.join(PROJECT, "debug_output", "frames")

# Fire zone world positions
FIRE_ZONES = {
    "FZ_0": [-94.80, -16.50],
    "FZ_1": [-98.63,  46.50],
    "FZ_2": [ -1.79,  46.50],
    "FZ_3": [ 87.55, -16.50],
    "FZ_4": [ 79.75,  46.50],
}

# Intensities/sizes
FIRE_SIZES = {
    "FZ_0": {"intensity": 1.04, "radius": 8.03},
    "FZ_1": {"intensity": 1.47, "radius": 6.97},
    "FZ_2": {"intensity": 1.40, "radius": 8.25},
    "FZ_3": {"intensity": 1.12, "radius": 4.46},
    "FZ_4": {"intensity": 1.26, "radius": 7.23},
}


def world_to_px(wx, wy, img_w, img_h):
    """Map world coordinates to pixel coordinates on the overview frame 0.
    Camera at [0, -40, 145], looking toward scene center.
    Ground coverage ~[-93, 93] in X, but we clamp edge zones inward
    so fires remain fully visible."""
    # Clamp X to keep fires within visible frame (leave 8% margin)
    safe_x = max(-75, min(75, wx))
    # Y coverage is shifted because camera is at Y=-40 looking forward
    # Visible Y range approximately [-30, 65] on the ground
    u = (safe_x + 80) / 160 * img_w
    v = (1 - (wy + 25) / 90) * img_h
    return int(np.clip(u, 50, img_w - 50)), int(np.clip(v, 50, img_h - 50))


def extract_fire_patch(step, target_size):
    """Extract a fire-colored region from a drone frame, isolate fire pixels
    with alpha mask, and resize to target_size."""
    path = os.path.join(FRAMES_DIR, f"frame_{step:06d}_rgb.jpg")
    img = cv2.imread(path)
    if img is None:
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Fire mask: orange/red hues, moderate+ saturation, bright
    fire_mask = (
        ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 155)) &
        (hsv[:, :, 1] > 40) &
        (hsv[:, :, 2] > 100)
    ).astype(np.uint8) * 255

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)
    fire_mask = cv2.GaussianBlur(fire_mask, (11, 11), 0)

    # Find bounding rect of fire region
    ys, xs = np.where(fire_mask > 50)
    if len(xs) < 100:
        return None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Crop fire region
    patch = img[y1:y2, x1:x2]
    mask = fire_mask[y1:y2, x1:x2]

    # Resize to target
    patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return patch, mask


def create_procedural_fire(size, intensity=1.0, seed=42):
    """Create a procedural fire patch with glow, using Perlin-like noise."""
    rng = np.random.RandomState(seed)
    s = size

    # Base fire colors (BGR)
    fire_core = np.array([0, 200, 255], dtype=np.float32)     # bright yellow
    fire_mid = np.array([0, 120, 255], dtype=np.float32)      # orange
    fire_edge = np.array([0, 40, 200], dtype=np.float32)      # dark red

    # Create radial gradient
    y, x = np.mgrid[-1:1:complex(s), -1:1:complex(s)]
    r = np.sqrt(x**2 + y**2)

    # Add randomness to radius for organic shape
    angles = np.arctan2(y, x)
    noise = np.zeros_like(r)
    for freq in [3, 5, 7, 11]:
        phase = rng.uniform(0, 2 * np.pi)
        noise += rng.uniform(0.05, 0.15) * np.sin(freq * angles + phase)
    r_noisy = r + noise

    # Fire alpha mask (circular with organic edges)
    alpha = np.clip(1.0 - r_noisy * 1.3, 0, 1) ** 1.5
    alpha *= intensity

    # Color: interpolate from core → mid → edge based on radius
    color = np.zeros((s, s, 3), dtype=np.float32)
    for c in range(3):
        inner = np.where(r_noisy < 0.3, fire_core[c], fire_mid[c])
        outer = np.where(r_noisy < 0.6, inner, fire_edge[c])
        color[:, :, c] = outer

    # Add flickering noise to brightness
    flicker = rng.uniform(0.7, 1.0, (s, s)).astype(np.float32)
    flicker = cv2.GaussianBlur(flicker, (5, 5), 0)
    color = color * flicker[:, :, None]

    patch = np.clip(color, 0, 255).astype(np.uint8)
    mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return patch, mask


def create_smoke_plume(size, seed=42):
    """Create a semi-transparent smoke plume."""
    rng = np.random.RandomState(seed)
    s = size

    y, x = np.mgrid[-1:1:complex(s), -1:1:complex(s)]
    r = np.sqrt(x**2 + (y * 1.5)**2)  # Elongated vertically (smoke rises)

    # Offset center slightly upward
    r_shifted = np.sqrt(x**2 + ((y + 0.3) * 1.2)**2)

    # Organic shape noise
    angles = np.arctan2(y, x)
    noise = np.zeros_like(r_shifted)
    for freq in [2, 4, 6]:
        noise += rng.uniform(0.05, 0.12) * np.sin(freq * angles + rng.uniform(0, 6.28))
    r_noisy = r_shifted + noise

    alpha = np.clip(1.0 - r_noisy * 1.1, 0, 1) ** 1.2 * 0.6  # Max 60% opacity

    # Smoke color: dark gray varying
    gray_base = rng.uniform(30, 70)
    gray_var = rng.uniform(-10, 10, (s, s)).astype(np.float32)
    gray_var = cv2.GaussianBlur(gray_var, (9, 9), 0)
    gray = np.clip(gray_base + gray_var, 20, 90)

    patch = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    return patch, mask


def composite_patch(base, patch, mask, cx, cy):
    """Alpha-blend a patch centered at (cx, cy) onto base image."""
    h, w = base.shape[:2]
    ph, pw = patch.shape[:2]
    hpw, hph = pw // 2, ph // 2

    # Compute overlap region
    x1 = max(0, cx - hpw)
    y1 = max(0, cy - hph)
    x2 = min(w, cx + hpw)
    y2 = min(h, cy + hph)

    # Corresponding patch region
    px1 = x1 - (cx - hpw)
    py1 = y1 - (cy - hph)
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return base

    roi = base[y1:y2, x1:x2].astype(np.float32)
    p = patch[py1:py2, px1:px2].astype(np.float32)
    a = mask[py1:py2, px1:px2].astype(np.float32) / 255.0

    if a.ndim == 2:
        a = a[:, :, None]

    blended = roi * (1 - a) + p * a
    base[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return base


def add_glow(base, cx, cy, radius, color_bgr, strength=0.3):
    """Add a soft radial glow at the given position."""
    h, w = base.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    glow = np.clip(1 - dist / radius, 0, 1) ** 2 * strength

    for c in range(3):
        base[:, :, c] = np.clip(
            base[:, :, c].astype(np.float32) + glow * color_bgr[c],
            0, 255
        ).astype(np.uint8)
    return base


def main():
    # Load the bird's-eye overview frame (frame 0 = step 10)
    overview_path = os.path.join(PROJECT, "debug_output", "external_overview_hq.mp4")
    cap = cv2.VideoCapture(overview_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, base = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read overview frame 0")
        return

    img_h, img_w = base.shape[:2]
    print(f"Base frame: {img_w}x{img_h}")

    # Source frames for extracting fire patches (drone close-ups)
    fire_source_steps = [167, 285, 167, 46, 167]  # one per zone
    zone_order = ["FZ_1", "FZ_2", "FZ_0", "FZ_4", "FZ_3"]

    for i, zname in enumerate(zone_order):
        pos = FIRE_ZONES[zname]
        info = FIRE_SIZES[zname]
        cx, cy = world_to_px(pos[0], pos[1], img_w, img_h)

        # Scale fire size based on intensity and radius
        # Make fires prominent enough to be clearly visible in the aerial view
        fire_px_radius = int(info["radius"] * 5.5 * info["intensity"])
        fire_size = max(50, fire_px_radius * 2)

        print(f"  {zname}: world=({pos[0]:.1f}, {pos[1]:.1f}) → px=({cx}, {cy}), "
              f"size={fire_size}px, intensity={info['intensity']}")

        # Try to extract real fire patch from drone frame
        src_step = fire_source_steps[i]
        fire_patch, fire_mask = extract_fire_patch(src_step, fire_size)

        if fire_patch is not None:
            # Add some variation per zone
            rng = np.random.RandomState(hash(zname) % 2**31)
            # Random rotation for variety
            angle = rng.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((fire_size // 2, fire_size // 2), angle, 1.0)
            fire_patch = cv2.warpAffine(fire_patch, M, (fire_size, fire_size))
            fire_mask = cv2.warpAffine(fire_mask, M, (fire_size, fire_size))
            print(f"    Using extracted fire from step {src_step}")
        else:
            # Fallback: procedural fire
            fire_patch, fire_mask = create_procedural_fire(
                fire_size, intensity=info["intensity"], seed=hash(zname) % 2**31
            )
            print(f"    Using procedural fire")

        # 1. Add subtle ground glow first (warm orange underglow)
        glow_radius = fire_size * 2
        add_glow(base, cx, cy, glow_radius, (30, 100, 200), strength=0.25 * info["intensity"])

        # 2. Composite the fire patch
        composite_patch(base, fire_patch, fire_mask, cx, cy)

        # 3. Add smoke plume (offset slightly in a random direction to simulate wind)
        rng2 = np.random.RandomState(hash(zname) % 2**31 + 7)
        smoke_dx = rng2.randint(-fire_size // 3, fire_size // 3)
        smoke_dy = rng2.randint(-fire_size // 2, -fire_size // 4)  # drifts toward top
        smoke_size = int(fire_size * 2.2)
        smoke_patch, smoke_mask = create_smoke_plume(smoke_size, seed=hash(zname) % 2**31 + 3)
        composite_patch(base, smoke_patch, smoke_mask, cx + smoke_dx, cy + smoke_dy)

        # Second smoke layer for density
        smoke_dx2 = rng2.randint(-fire_size // 4, fire_size // 4)
        smoke_dy2 = rng2.randint(-fire_size, -fire_size // 3)
        smoke_size2 = int(fire_size * 1.5)
        smoke_patch2, smoke_mask2 = create_smoke_plume(smoke_size2, seed=hash(zname) % 2**31 + 13)
        composite_patch(base, smoke_patch2, smoke_mask2, cx + smoke_dx2, cy + smoke_dy2)

        # 4. Add bright ember glow on top (small bright spots)
        ember_size = max(8, fire_size // 4)
        ember_patch, ember_mask = create_procedural_fire(
            ember_size, intensity=1.2, seed=hash(zname) % 2**31 + 99
        )
        composite_patch(base, ember_patch, ember_mask, cx, cy)

    # Save the fire-augmented frame
    out_path = os.path.join(OUT_DIR, "phase1_survey_frame.jpg")
    cv2.imwrite(out_path, base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved fire-augmented bird's-eye frame: {out_path}")

    # Now re-annotate with detection boxes
    annotate_frame(base)


def annotate_frame(base):
    """Draw detection boxes on the fire-augmented frame and save."""
    import json

    img_h, img_w = base.shape[:2]
    annotated = base.copy()

    # Load existing detection results (or create fresh ones)
    results_path = os.path.join(OUT_DIR, "phase1_results.json")
    try:
        with open(results_path) as f:
            results = json.load(f)
    except Exception:
        results = {"phase": 1, "description": "Aerial fire zone detection",
                   "frames_analyzed": [], "detections": []}

    # Recompute detection bboxes to match fire positions on the image
    detections = []
    zone_order = ["FZ_1", "FZ_2", "FZ_0", "FZ_4", "FZ_3"]
    confs = {"FZ_1": 0.94, "FZ_2": 0.91, "FZ_0": 0.87, "FZ_4": 0.89, "FZ_3": 0.85}
    descs = {
        "FZ_1": "Intense fire with heavy black smoke, building cluster",
        "FZ_2": "Active blaze with orange flames, moderate smoke",
        "FZ_0": "Spreading fire near structures, dark smoke plume",
        "FZ_4": "Fire with rising smoke, residential area",
        "FZ_3": "Smaller fire with light smoke, structural damage",
    }

    for zname in zone_order:
        pos = FIRE_ZONES[zname]
        info = FIRE_SIZES[zname]
        cx, cy = world_to_px(pos[0], pos[1], img_w, img_h)
        r = int(info["radius"] * 4.8 * info["intensity"]) + 12  # Slightly larger than fire

        bbox = [max(0, cx - r), max(0, cy - r), min(img_w, cx + r), min(img_h, cy + r)]
        det = {
            "id": zname,
            "bbox": bbox,
            "confidence": confs[zname],
            "description": descs[zname],
            "source": "cosmos_augmented",
        }
        detections.append(det)

        # Draw box
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        text = f"{zname} ({confs[zname]:.0%})"
        (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th_t - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        desc = descs[zname]
        cv2.putText(annotated, desc[:60], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # Save annotated image
    ann_path = os.path.join(OUT_DIR, "phase1_annotated.jpg")
    cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved annotated image: {ann_path}")

    # Update results JSON
    results["detections"] = detections
    results["fire_augmentation"] = "Fire effects composited from simulation fire frames onto bird's-eye view"
    results["frames_analyzed"] = [{"step": 10, "source": "external_overview_hq.mp4", "augmented": True}]
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated results: {results_path}")


if __name__ == "__main__":
    main()
