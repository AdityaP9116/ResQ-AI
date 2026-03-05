#!/usr/bin/env python3
"""Standalone script to composite realistic fires onto the aerial overview
frame. Runs just the fire compositing from run_cosmos_pipeline.py without
needing to re-run the full pipeline."""

import cv2
import json
import numpy as np
import os

PROJECT = os.path.dirname(os.path.abspath(__file__))
SRC_FRAMES = os.path.join(PROJECT, "debug_output", "frames")
OUT_DIR = os.path.join(PROJECT, "debug_output_1")

FIRE_ZONES = {
    "FZ_0": {"position": [-94.80, -16.50, 0.15], "intensity": 1.04, "radius": 8.03},
    "FZ_1": {"position": [-98.63, 46.50, 0.15], "intensity": 1.47, "radius": 6.97},
    "FZ_2": {"position": [-1.79, 46.50, 0.15], "intensity": 1.40, "radius": 8.25},
    "FZ_3": {"position": [87.55, -16.50, 0.15], "intensity": 1.12, "radius": 4.46},
    "FZ_4": {"position": [79.75, 46.50, 0.15], "intensity": 1.26, "radius": 7.23},
}

FIRE_SOURCE_STEPS = {"FZ_1": 167, "FZ_2": 210, "FZ_0": 285, "FZ_3": 167, "FZ_4": 285}


def frame_path(step):
    return os.path.join(SRC_FRAMES, f"frame_{step:06d}_rgb.jpg")


def extract_fire_patch(step, target_size):
    img = cv2.imread(frame_path(step))
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fire_mask = (
        ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 150))
        & (hsv[:, :, 1] > 30)
        & (hsv[:, :, 2] > 90)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=3)
    fire_mask = cv2.GaussianBlur(fire_mask, (15, 15), 0)
    ys, xs = np.where(fire_mask > 40)
    if len(xs) < 60:
        return None, None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 8
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    patch = cv2.resize(img[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(fire_mask[y1:y2, x1:x2], (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return patch, mask


def alpha_composite(base, patch, mask, cx, cy):
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


def add_radial_glow(base, cx, cy, radius, color_bgr, strength=0.35):
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


def make_smoke_plume(w_size, h_size, seed=0, color_base=(55, 50, 45)):
    rng = np.random.RandomState(seed)
    h, w = h_size, w_size
    y, x = np.mgrid[0:h, 0:w]
    yn = y.astype(np.float32) / h
    xn = (x.astype(np.float32) / w - 0.5) * 2

    width_profile = 0.3 + yn * 0.7
    edge_dist = np.abs(xn) / width_profile
    base_alpha = np.clip(1.0 - edge_dist * 1.1, 0, 1) ** 1.4
    vert_fade = np.clip(1.0 - yn * 0.85, 0, 1) ** 0.8

    turb = np.zeros((h, w), dtype=np.float32)
    for octave in range(4):
        freq = 2 ** (octave + 1)
        amp = 0.4 / (octave + 1)
        noise = rng.uniform(-1, 1, (max(2, h // freq + 1), max(2, w // freq + 1))).astype(np.float32)
        noise_up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        turb += noise_up * amp

    alpha = np.clip(base_alpha * vert_fade + turb * 0.15, 0, 1) * 0.55

    b = np.clip(color_base[0] + yn * 30 + turb * 15, 20, 100).astype(np.uint8)
    g = np.clip(color_base[1] + yn * 25 + turb * 12, 20, 95).astype(np.uint8)
    r_ch = np.clip(color_base[2] + yn * 20 + turb * 10, 20, 90).astype(np.uint8)
    patch = np.stack([b, g, r_ch], axis=-1)
    mask = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return patch, mask


def make_ember_particles(size, n_embers, seed=0):
    rng = np.random.RandomState(seed)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n_embers):
        ex = rng.randint(size // 6, size * 5 // 6)
        ey = rng.randint(size // 4, size * 3 // 4)
        r = rng.randint(1, 3)
        brightness = rng.randint(200, 255)
        color = (rng.randint(20, 60), rng.randint(120, 200), brightness)
        cv2.circle(patch, (ex, ey), r, color, -1)
        cv2.circle(mask, (ex, ey), r + 1, 200, -1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return patch, mask


def composite_fires_on_aerial(img):
    h, w = img.shape[:2]

    def world_to_px(wx, wy):
        safe_x = max(-75, min(75, wx))
        u = (safe_x + 80) / 160 * w
        v = (1 - (wy + 25) / 90) * h
        return int(max(30, min(w - 30, u))), int(max(30, min(h - 30, v)))

    zone_order = ["FZ_0", "FZ_1", "FZ_2", "FZ_3", "FZ_4"]

    # Pass 1: Ground scorching & wide glow
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))

        add_radial_glow(img, cx, cy, base_size * 2.8, (10, 35, 140), strength=0.18 * intensity)
        add_radial_glow(img, cx, cy, base_size * 1.5, (15, 70, 210), strength=0.30 * intensity)

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

    # Pass 2: Fire patches (multi-layered)
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))
        src_step = FIRE_SOURCE_STEPS.get(zname, 167)
        rng = np.random.RandomState(hash(zname) % 100000)

        for layer in range(3):
            layer_size = int(base_size * (0.7 + layer * 0.2))
            patch, mask = extract_fire_patch(src_step, layer_size)
            if patch is None:
                continue
            angle = idx * 72 + layer * 40 + rng.randint(-20, 20)
            M = cv2.getRotationMatrix2D((layer_size // 2, layer_size // 2), angle, 1.0)
            patch = cv2.warpAffine(patch, M, (layer_size, layer_size))
            mask = cv2.warpAffine(mask, M, (layer_size, layer_size))
            bright_shift = rng.uniform(0.85, 1.15)
            patch = np.clip(patch.astype(np.float32) * bright_shift, 0, 255).astype(np.uint8)
            dx = rng.randint(-base_size // 6, base_size // 6)
            dy = rng.randint(-base_size // 6, base_size // 6)
            alpha_composite(img, patch, mask, cx + dx, cy + dy)

        core_size = max(8, base_size // 5)
        add_radial_glow(img, cx, cy, core_size, (80, 200, 255), strength=0.6 * intensity)

    # Pass 3: Smoke plumes
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))
        rng = np.random.RandomState(hash(zname) % 100000 + 42)

        plume_w = int(base_size * 1.2)
        plume_h = int(base_size * 2.0)
        soot = max(0, min(30, int((intensity - 1.0) * 40)))
        sp, sm = make_smoke_plume(plume_w, plume_h, seed=hash(zname) % 10000,
                                  color_base=(50 - soot, 45 - soot, 40 - soot))
        alpha_composite(img, sp, sm, cx + rng.randint(-8, 8), cy - base_size // 2)

        wisp_w = int(base_size * 0.6)
        wisp_h = int(base_size * 1.2)
        wp, wm = make_smoke_plume(wisp_w, wisp_h, seed=hash(zname) % 10000 + 99,
                                  color_base=(65, 60, 55))
        wind_dx = rng.choice([-1, 1]) * (base_size // 3 + rng.randint(0, 15))
        alpha_composite(img, wp, wm, cx + wind_dx, cy - base_size // 4)

    # Pass 4: Embers
    for idx, zname in enumerate(zone_order):
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        intensity = zdata["intensity"]
        radius = zdata["radius"]
        base_size = int(max(55, min(130, radius * 6 * intensity)))
        n_embers = int(12 * intensity)
        ember_field = int(base_size * 1.5)
        ep, em = make_ember_particles(ember_field, n_embers, seed=hash(zname) % 10000 + 7)
        alpha_composite(img, ep, em, cx, cy)

    return img


def main():
    # Extract clean overview frame
    cap = cv2.VideoCapture(os.path.join(PROJECT, "debug_output", "external_overview_hq.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, img = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read overview frame")
        return
    print(f"Clean overview frame: {img.shape[1]}x{img.shape[0]}")

    # Composite fires
    composite_fires_on_aerial(img)

    # Save survey frame
    cv2.imwrite(os.path.join(OUT_DIR, "phase1_survey_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("Saved phase1_survey_frame.jpg")

    # Regenerate annotated version with detection boxes
    p1_path = os.path.join(OUT_DIR, "phase1_results.json")
    if os.path.exists(p1_path):
        p1 = json.load(open(p1_path))
        annotated = img.copy()
        h, w = annotated.shape[:2]
        for det in p1.get("detections", []):
            bbox = det.get("bbox", [0, 0, 100, 100])
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = det.get("id", "FZ_?")
            conf = det.get("confidence", 0.5)
            text = f"{label} ({conf:.0%})"
            (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th_t - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(OUT_DIR, "phase1_annotated.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("Saved phase1_annotated.jpg")

    # Stats
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fire_mask = (
        ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 160))
        & (hsv[:, :, 1] > 50)
        & (hsv[:, :, 2] > 120)
    )
    print(f"Fire pixel coverage: {fire_mask.sum() / fire_mask.size * 100:.2f}%")


if __name__ == "__main__":
    main()
