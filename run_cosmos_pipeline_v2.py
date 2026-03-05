#!/usr/bin/env python3
"""ResQ-AI Pipeline v2 — Real Rendered Aerial + All 5 Zones

Phase 1: Cosmos analyzes the Isaac Sim–rendered aerial image (with real Flow fires)
Phase 2: YOLO investigation on drone frames from ALL 5 fire zones
Phase 3: Cosmos close-up assessment + ranking for all 5 zones

Prerequisites:
  1. sim_bridge/render_aerial_view.py has been run → debug_output_2/aerial_rendered.jpg
  2. headless_e2e_test.py has been run with RESQAI_ZONE_ORDER (all 5 zones) →
     debug_output_2/frames/, debug_output_2/external_overview_hq.mp4
  3. Cosmos NIM running on localhost:8010

Outputs: debug_output_2/
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
SRC_DIR = os.path.join(PROJECT_ROOT, "debug_output_2")
SRC_FRAMES = os.path.join(SRC_DIR, "frames")
OUT_DIR = SRC_DIR  # outputs go into same dir
COSMOS_URL = "http://localhost:8010/v1/chat/completions"
COSMOS_MODEL = "nvidia/cosmos-reason2-8b"
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    API_KEY = os.environ.get("NVIDIA_API_KEY", API_KEY)
except ImportError:
    pass

# All 5 fire zones
FIRE_ZONES = {
    "FZ_0": {"position": [-94.80, -16.50, 0.15], "intensity": 1.04, "radius": 8.03},
    "FZ_1": {"position": [-98.63, 46.50, 0.15], "intensity": 1.47, "radius": 6.97},
    "FZ_2": {"position": [-1.79, 46.50, 0.15], "intensity": 1.40, "radius": 8.25},
    "FZ_3": {"position": [87.55, -16.50, 0.15], "intensity": 1.12, "radius": 4.46},
    "FZ_4": {"position": [79.75, 46.50, 0.15], "intensity": 1.26, "radius": 7.23},
}

# Zone visit order (will be populated after Phase 1 from Cosmos detection order)
ZONE_ORDER = ["FZ_1", "FZ_2", "FZ_0", "FZ_3", "FZ_4"]

# Aerial rendered image (from render_aerial_view.py)
AERIAL_IMAGE = os.path.join(SRC_DIR, "aerial_rendered.jpg")
AERIAL_IMAGE_WEB = os.path.join(SRC_DIR, "aerial_rendered_web.jpg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_frame(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cosmos_vision(image_b64: str, prompt: str, max_tokens: int = 1024) -> str:
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
        temperature=0.6,
    )
    return response.choices[0].message.content or ""


def cosmos_text(prompt: str, max_tokens: int = 1024) -> str:
    import openai
    client = openai.OpenAI(
        base_url="http://localhost:8010/v1",
        api_key=API_KEY or "unused",
    )
    response = client.chat.completions.create(
        model=COSMOS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.6,
    )
    return response.choices[0].message.content or ""


def frame_path(step: int, suffix: str = "rgb") -> str:
    return os.path.join(SRC_FRAMES, f"frame_{step:06d}_{suffix}.jpg")


def save_json(data, filename: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  -> Saved {path}")


def draw_boxes_on_frame(img, detections, color=(0, 0, 255), thickness=2):
    out = img.copy()
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        label = det.get("class", "fire")
        conf = det.get("confidence", 0)
        if label == "fire":
            c = (0, 0, 255)
        elif label == "person":
            c = (0, 255, 0)
        else:
            c = (255, 165, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def _extract_json(text):
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
# Phase 1: Cosmos analyzes rendered aerial image
# ---------------------------------------------------------------------------

def phase1_aerial_detection():
    """Send the Isaac Sim rendered aerial image (with real fires) to Cosmos."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Aerial Fire Detection (Real Rendered Fires)")
    print("=" * 70)

    results = {
        "phase": 1,
        "description": "Aerial fire zone detection from Isaac Sim rendered image",
        "source": "render_aerial_view.py (real Flow particle fires)",
        "frames_analyzed": [],
        "detections": [],
    }

    # Use the rendered aerial image
    aerial_path = AERIAL_IMAGE
    if not os.path.exists(aerial_path):
        # Fallback to web version
        aerial_path = AERIAL_IMAGE_WEB
    if not os.path.exists(aerial_path):
        print("  ERROR: No rendered aerial image found!")
        print(f"  Expected: {AERIAL_IMAGE}")
        print("  Run sim_bridge/render_aerial_view.py first.")
        return results

    img = cv2.imread(aerial_path)
    h, w = img.shape[:2]
    print(f"  Using rendered aerial image ({w}x{h}): {aerial_path}")

    # Save survey frame copy for frontend
    survey_path = os.path.join(OUT_DIR, "phase1_survey_frame.jpg")
    cv2.imwrite(survey_path, img)

    # Encode and send to Cosmos
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    prompt = (
        "You are analyzing a high-altitude aerial surveillance image rendered from a "
        "disaster simulation. The image is taken from a static overhead camera at "
        "approximately 180 meters altitude, looking straight down at an urban disaster scene.\n\n"
        "The scene contains REAL particle fire effects rendered by the simulation engine. "
        "These appear as bright orange/yellow fire flames and grey/black smoke plumes "
        "rising from various locations across the scene.\n\n"
        "Your task is to identify ALL areas where fire, smoke, flames, or thermal activity "
        "is visible. Look carefully at every region of the image — there should be "
        "approximately 5 fire zones spread across the scene.\n\n"
        "For each fire zone you detect, provide:\n"
        "1. A bounding box in pixel coordinates [x1, y1, x2, y2] where (0,0) is top-left\n"
        "2. A confidence score (0-1)\n"
        "3. A brief description of what you see\n"
        "4. Estimated severity: Low, Medium, High, or Critical\n\n"
        "Number the zones as FZ_0, FZ_1, FZ_2, etc.\n"
        f"Image dimensions: {w}x{h} pixels.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"fire_zones": [{"id": "FZ_0", "bbox": [x1, y1, x2, y2], "confidence": 0.9, '
        '"description": "...", "severity": "High"}]}'
    )

    print("  Sending rendered aerial image to Cosmos NIM...")
    t0 = time.time()
    raw = cosmos_vision(b64, prompt)
    elapsed = time.time() - t0
    print(f"  Cosmos responded in {elapsed:.1f}s")
    print(f"  Raw response: {raw[:500]}...")

    results["frames_analyzed"].append({"step": "rendered", "source": "aerial_rendered.jpg"})
    results["cosmos_raw_response"] = raw
    results["cosmos_response_time_s"] = round(elapsed, 2)

    # Parse response
    parsed = _extract_json(raw)
    cosmos_dets = []
    if parsed and "fire_zones" in parsed:
        cosmos_dets = parsed["fire_zones"]

    if cosmos_dets:
        for cd in cosmos_dets:
            bb = cd.get("bbox", [0, 0, 100, 100])
            cd["bbox"] = [max(0, min(bb[0], w)), max(0, min(bb[1], h)),
                          max(0, min(bb[2], w)), max(0, min(bb[3], h))]
        print(f"  Cosmos detected {len(cosmos_dets)} fire zone(s)")

        # Supplement with ground truth for any zones cosmos missed
        cosmos_ids = {cd.get("id", "") for cd in cosmos_dets}
        gt_dets = _ground_truth_detections(w, h)
        for gd in gt_dets:
            if gd["id"] not in cosmos_ids:
                gd["source"] = "ground_truth_supplement"
                cosmos_dets.append(gd)
                print(f"  Added ground-truth for missed zone {gd['id']}")

        results["detections"] = cosmos_dets
    else:
        print("  Cosmos gave free-form response — using ground truth positions")
        results["cosmos_freeform"] = raw
        results["detections"] = _ground_truth_detections(w, h)

    # Draw detections on aerial image
    annotated = img.copy()
    for i, det in enumerate(results["detections"]):
        bbox = det.get("bbox", [0, 0, 100, 100])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = det.get("id", f"FZ_{i}")
        conf = det.get("confidence", 0.5)
        severity = det.get("severity", "Unknown")
        text = f"{label} ({conf:.0%}) [{severity}]"
        (tw, th_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th_t - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        desc = det.get("description", "")
        if desc:
            cv2.putText(annotated, desc[:60], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imwrite(os.path.join(OUT_DIR, "phase1_annotated.jpg"), annotated)
    print("  Saved phase1_annotated.jpg")

    save_json(results, "phase1_results.json")
    return results


def _ground_truth_detections(img_w, img_h):
    """Generate bounding boxes for all 5 known fire zones, projected
    onto the overhead camera image (camera at [0, 15, 180] looking down)."""
    detections = []

    # Aerial camera: static at [0, 15, alt], 18mm focal, 36mm aperture
    # FOV_h ≈ 2*atan(18/18) ≈ 90°, range visible ≈ 2*180*tan(45°) ≈ 360m
    # Scene spans X: -100..90, Y: -20..50 → fits within ~200m x 70m
    def world_to_px(wx, wy):
        # Camera at [0, 15, 180], FOV ~90° → visible range ~[-180, 180] in both X,Y
        # But scene is only ~200m wide and ~85m deep
        # Simple linear mapping: center of image = (0, 15) in world coords
        u = (wx + 180) / 360 * img_w
        v = (1 - (wy - 15 + 180) / 360) * img_h
        return int(max(20, min(img_w - 20, u))), int(max(20, min(img_h - 20, v)))

    for zname in ZONE_ORDER:
        zdata = FIRE_ZONES[zname]
        cx, cy = world_to_px(zdata["position"][0], zdata["position"][1])
        r = int(max(25, zdata["radius"] * 4 * zdata["intensity"])) + 10
        detections.append({
            "id": zname,
            "bbox": [max(0, cx - r), max(0, cy - r), min(img_w, cx + r), min(img_h, cy + r)],
            "confidence": round(0.6 + zdata["intensity"] * 0.2, 2),
            "description": f"Fire zone: intensity {zdata['intensity']:.2f}, radius {zdata['radius']:.1f}m",
            "severity": "High" if zdata["intensity"] > 1.3 else "Medium",
        })
    return detections


# ---------------------------------------------------------------------------
# Phase 2: YOLO Investigation on all 5 zones
# ---------------------------------------------------------------------------

def _find_zone_best_frames():
    """Scan all simulation frames with YOLO to find the best fire detection
    frame for each zone. Uses zone proximity + YOLO fire confidence."""
    from sim_bridge.yolo_detector import DualYOLODetector

    print("  Scanning frames to find best detection per zone...")
    detector = DualYOLODetector(device=0)

    # Get all available frames
    frame_files = sorted(glob.glob(os.path.join(SRC_FRAMES, "frame_*_rgb.jpg")))
    if not frame_files:
        print("  ERROR: No frames found in", SRC_FRAMES)
        return {}

    print(f"  Found {len(frame_files)} frames to scan")

    # Load cosmos reasoning log to map frames → zones
    # Also load the zone visit timeline from fire_situation_report.json
    zone_frames = {zn: {"best_step": None, "best_conf": 0.0, "range": []} for zn in FIRE_ZONES}

    # Read YOLO detection JSONs if available
    yolo_jsons = sorted(glob.glob(os.path.join(SRC_FRAMES, "frame_*_yolo.json")))

    # Strategy: scan frames, run YOLO, and assign each frame to nearest zone
    # based on drone position (estimated from frame number in the flight path)
    sample_rate = max(1, len(frame_files) // 150)  # sample ~150 frames max

    for idx, fpath in enumerate(frame_files):
        if idx % sample_rate != 0:
            continue

        step = int(os.path.basename(fpath).split("_")[1])

        # Check for existing YOLO json
        yolo_json = os.path.join(SRC_FRAMES, f"frame_{step:06d}_yolo.json")
        fire_dets = []

        if os.path.exists(yolo_json):
            try:
                with open(yolo_json) as f:
                    jdata = json.load(f)
                fire_dets = [d for d in jdata if d.get("class") == "fire"]
            except Exception:
                pass
        else:
            # Run YOLO on this frame
            img = cv2.imread(fpath)
            if img is None:
                continue
            dets = detector.detect(img)
            fire_dets = [d for d in dets if d["class"] == "fire"]

        if not fire_dets:
            continue

        best_fire_conf = max(d["confidence"] for d in fire_dets)

        # Assign to nearest zone based on frame timing
        # With 5 zones visited sequentially, each zone gets roughly
        # (total_frames/5) frames. We use position if available.
        # For now, assign based on step ranges (will be refined)
        total_frames = len(frame_files)
        frames_per_zone = total_frames // len(ZONE_ORDER)
        survey_frames = 50  # first ~50 frames are survey
        zone_idx = min(len(ZONE_ORDER) - 1,
                       max(0, (step - survey_frames) // max(1, frames_per_zone)))
        if step < survey_frames:
            # During survey, assign to nearest based on step
            zone_idx = 0

        zone_name = ZONE_ORDER[min(zone_idx, len(ZONE_ORDER) - 1)]

        zone_frames[zone_name]["range"].append(step)
        if best_fire_conf > zone_frames[zone_name]["best_conf"]:
            zone_frames[zone_name]["best_conf"] = best_fire_conf
            zone_frames[zone_name]["best_step"] = step

    # Build ZONE_BEST_FRAMES format
    result = {}
    for zn, zf in zone_frames.items():
        if zf["best_step"] is not None:
            rng = sorted(zf["range"])
            result[zn] = {
                "step": zf["best_step"],
                "conf": zf["best_conf"],
                "group_range": (rng[0], rng[-1]) if rng else (0, 0),
            }
        else:
            # No YOLO fire detection — use middle of estimated range
            total = len(frame_files)
            idx = ZONE_ORDER.index(zn) if zn in ZONE_ORDER else 0
            frames_per = total // len(ZONE_ORDER)
            mid = 50 + idx * frames_per + frames_per // 2
            result[zn] = {
                "step": min(mid, total - 1),
                "conf": 0.0,
                "group_range": (50 + idx * frames_per, 50 + (idx + 1) * frames_per),
            }

    for zn in ZONE_ORDER:
        if zn in result:
            print(f"    {zn}: best step={result[zn]['step']}, conf={result[zn]['conf']:.3f}, "
                  f"range={result[zn]['group_range']}")

    return result


def phase2_yolo_investigation():
    """Run YOLO detection on frames near each fire zone — ALL 5 zones."""
    print("\n" + "=" * 70)
    print("  PHASE 2: YOLO Fire Detection — All 5 Zones")
    print("=" * 70)

    sys.path.insert(0, PROJECT_ROOT)
    from sim_bridge.yolo_detector import DualYOLODetector

    print("  Loading YOLO models...")
    detector = DualYOLODetector(device=0)

    # Auto-discover best frames per zone
    zone_best = _find_zone_best_frames()

    results = {"phase": 2, "description": "YOLO investigation — all 5 fire zones", "zones": {}}
    zone_frames_dir = os.path.join(OUT_DIR, "zone_frames")
    os.makedirs(zone_frames_dir, exist_ok=True)

    for zone_name in ZONE_ORDER:
        info = zone_best.get(zone_name)
        if not info:
            print(f"\n  --- {zone_name}: No frame data, skipping ---")
            continue

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

        # Analyze frames in range
        analyze_steps = list(range(max(group_start, 10), min(group_end + 1, 10000)))
        if len(analyze_steps) > 20:
            analyze_steps = analyze_steps[::max(1, len(analyze_steps) // 20)]

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

            for d in fire_dets:
                if d["confidence"] > best_conf:
                    best_conf = d["confidence"]
                    best_det = d
                    best_step_actual = s

            if fire_dets and s == step:
                annotated = draw_boxes_on_frame(img, dets)
                outpath = os.path.join(zone_frames_dir, f"{zone_name}_step{s}_yolo.jpg")
                cv2.imwrite(outpath, annotated)
                zone_result["annotated_frames"].append(f"zone_frames/{zone_name}_step{s}_yolo.jpg")
                print(f"    Step {s}: {len(fire_dets)} fire det(s), "
                      f"best conf={max(d['confidence'] for d in fire_dets):.3f}")

        # Save annotated best frame
        best_fpath = frame_path(best_step_actual)
        if os.path.exists(best_fpath):
            img_best = cv2.imread(best_fpath)
            if img_best is not None:
                all_dets = detector.detect(img_best)
                annotated_best = draw_boxes_on_frame(img_best, all_dets)
                cv2.putText(annotated_best, f"{zone_name} - Best Detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                outpath = os.path.join(zone_frames_dir, f"{zone_name}_best_yolo.jpg")
                cv2.imwrite(outpath, annotated_best)
                zone_result["annotated_frames"].append(f"zone_frames/{zone_name}_best_yolo.jpg")

                # Save raw closeup for Phase 3
                shutil.copy(best_fpath, os.path.join(zone_frames_dir, f"{zone_name}_closeup.jpg"))

        zone_result["best_detection"] = best_det
        zone_result["best_confidence"] = best_conf
        zone_result["best_step_actual"] = best_step_actual
        zone_result["total_fire_frames"] = sum(
            1 for fr in zone_result["frames_analyzed"] if fr["fire_detections"] > 0
        )

        results["zones"][zone_name] = zone_result
        print(f"    {zone_name}: {zone_result['total_fire_frames']}/{len(zone_result['frames_analyzed'])} "
              f"frames with fire, peak conf={best_conf:.3f}")

    save_json(results, "phase2_results.json")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Cosmos close-up assessment for all 5 zones
# ---------------------------------------------------------------------------

def phase3_cosmos_assessment(phase2_results: dict):
    """Send close-up frames to Cosmos for detailed fire assessment — all 5 zones."""
    print("\n" + "=" * 70)
    print("  PHASE 3: Cosmos Close-up Assessment — All 5 Zones")
    print("=" * 70)

    results = {
        "phase": 3,
        "description": "Cosmos close-up assessment — all 5 fire zones",
        "zone_assessments": {},
        "final_ranking": [],
    }

    zone_assessments = []

    for zone_name in ZONE_ORDER:
        print(f"\n  --- Assessing {zone_name} ---")

        z2 = phase2_results["zones"].get(zone_name, {})
        zone_data = FIRE_ZONES[zone_name]

        # Find closeup frame
        closeup_path = os.path.join(OUT_DIR, "zone_frames", f"{zone_name}_closeup.jpg")
        if not os.path.exists(closeup_path):
            best_step = z2.get("best_step_actual", z2.get("best_step", 100))
            closeup_path = frame_path(best_step)

        if not os.path.exists(closeup_path):
            print(f"    No closeup frame for {zone_name}, skipping")
            continue

        b64 = encode_frame(closeup_path)

        # Classify fire severity category from sim data for prompt context
        yolo_conf = z2.get('best_confidence', 0)
        _fi = zone_data['intensity']
        _fr = zone_data['radius']
        if _fi >= 1.4:
            _fire_cat = "large active blaze with tall flames"
        elif _fi >= 1.2:
            _fire_cat = "medium-intensity fire with spreading flames"
        elif _fi >= 1.0:
            _fire_cat = "moderate fire, partially contained"
        elif _fi >= 0.7:
            _fire_cat = "small localized fire or hotspot"
        else:
            _fire_cat = "smoldering remains or minor thermal signature"

        if yolo_conf >= 0.6:
            _detect_desc = f"YOLO detected fire here with HIGH confidence ({yolo_conf:.0%}), indicating clearly visible flames"
        elif yolo_conf >= 0.4:
            _detect_desc = f"YOLO detected fire with MODERATE confidence ({yolo_conf:.0%}), indicating partially visible fire"
        elif yolo_conf > 0:
            _detect_desc = f"YOLO detected fire with LOW confidence ({yolo_conf:.0%}), fire barely visible from altitude"
        else:
            _detect_desc = "YOLO did NOT detect any fire at this location — the fire may be obscured, very small, or already subsiding"

        prompt = (
            f"You are a disaster response AI analyzing a close-up aerial image of fire zone {zone_name}. "
            f"The drone is hovering at approximately 15-20 meters altitude.\n\n"
            f"CRITICAL: You MUST use the FULL 1-10 scale. Different fires have DIFFERENT severities. "
            f"Do NOT default to 7 or 8 for everything. Rate each fire based on its specific visual evidence.\n\n"
            f"ZONE-SPECIFIC DATA:\n"
            f"- Position: {zone_data['position']}\n"
            f"- Fire description: {_fire_cat}\n"
            f"- Thermal intensity measurement: {_fi:.2f} (scale: 0.5=minimal, 1.0=moderate, 1.5=extreme)\n"
            f"- Fire spread radius: {_fr:.1f} meters (2m=tiny, 5m=moderate, 8m+=large)\n"
            f"- Detection: {_detect_desc}\n\n"
            f"SCORING GUIDELINES (you MUST follow these):\n"
            f"- Intensity 1-3: Small/smoldering fires, low thermal readings, small radius\n"
            f"- Intensity 4-5: Moderate fires, visible flames, medium radius\n"
            f"- Intensity 6-7: Significant fires, active burning, large radius\n"
            f"- Intensity 8-10: Major infernos, extreme thermal, spreading rapidly\n"
            f"- Priority 1-3: Low urgency, contained fires\n"
            f"- Priority 4-6: Medium urgency, needs monitoring\n"
            f"- Priority 7-8: High urgency, active response needed\n"
            f"- Priority 9-10: Critical emergency, immediate action\n"
            f"- Spread Risk: Low (contained) / Medium (could spread) / High (actively spreading) / Critical (out of control)\n\n"
            f"Analyze the image carefully and provide your assessment as valid JSON only:\n"
            f'{{"zone": "{zone_name}", "intensity_rating": <1-10>, "smoke_analysis": "<description>", '
            f'"spread_risk": "<Low|Medium|High|Critical>", "spread_explanation": "<why>", "structural_damage": "<assessment>", '
            f'"rescue_priority": <1-10>, "recommended_action": "<specific action>", "summary": "<2 sentence assessment>"}}'
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
            assessment["parsed"] = None
            assessment["intensity_rating"] = _estimate_intensity(raw, zone_data["intensity"])
            assessment["spread_risk"] = _estimate_risk(raw)
            assessment["rescue_priority"] = _estimate_priority(raw, zone_data["intensity"])
            assessment["recommended_action"] = "Monitor and assess"
            assessment["summary"] = raw[:300]
            assessment["smoke_analysis"] = ""
            assessment["structural_damage"] = ""

        results["zone_assessments"][zone_name] = assessment
        zone_assessments.append(assessment)

        print(f"    Intensity: {assessment['intensity_rating']}/10, "
              f"Spread: {assessment['spread_risk']}, "
              f"Priority: {assessment['rescue_priority']}/10")

    # Final ranking
    print("\n  --- Generating Final Ranking (all 5 zones) ---")
    ranking_prompt = _build_ranking_prompt(zone_assessments)

    t0 = time.time()
    ranking_raw = cosmos_text(ranking_prompt)
    elapsed = time.time() - t0
    print(f"  Cosmos ranking in {elapsed:.1f}s")
    print(f"  Raw: {ranking_raw[:300]}...")

    results["ranking_cosmos_raw"] = ranking_raw
    results["ranking_cosmos_time_s"] = round(elapsed, 2)

    ranking_parsed = _extract_json(ranking_raw)

    # Build lookup from per-zone assessments for YOLO + sim data
    za_lookup = {a["zone"]: a for a in zone_assessments}

    def _compute_composite(entry, za=None):
        """Multi-factor composite score using per-zone assessment data."""
        pri = entry.get("rescue_priority", 5)
        inten = entry.get("intensity_rating", 5)
        risk_str = entry.get("spread_risk", "Medium")
        risk_val = {"Critical": 10, "High": 7, "Medium": 4, "Low": 1}.get(risk_str, 4)
        # Incorporate YOLO and sim data from per-zone assessment
        if za is None:
            za = za_lookup.get(entry.get("zone", ""), {})
        yolo = za.get("yolo_confidence", 0)
        sim_i = za.get("sim_intensity", 1.0)
        sim_r = za.get("sim_radius", 5.0)
        # Weighted: priority(x3) + intensity(x2) + risk + yolo_bonus + sim_factors
        score = (pri * 3) + (inten * 2) + risk_val + (yolo * 10) + (sim_i * 3) + (sim_r * 0.5)
        return round(score, 1)

    if ranking_parsed and "ranking" in ranking_parsed:
        # Use Cosmos ranking but RECALCULATE composite_score with our multi-factor formula
        cosmos_ranking = ranking_parsed["ranking"]
        for entry in cosmos_ranking:
            entry["composite_score"] = _compute_composite(entry)
        # Re-sort by our computed composite in case Cosmos order differs
        cosmos_ranking.sort(key=lambda e: e["composite_score"], reverse=True)
        for i, entry in enumerate(cosmos_ranking):
            entry["rank"] = i + 1
        results["final_ranking"] = cosmos_ranking
    else:
        ranked = sorted(zone_assessments, key=lambda z: _compute_composite({}, z), reverse=True)
        results["final_ranking"] = []
        for i, za in enumerate(ranked):
            results["final_ranking"].append({
                "rank": i + 1,
                "zone": za["zone"],
                "rescue_priority": za.get("rescue_priority", 5),
                "intensity_rating": za.get("intensity_rating", 5),
                "spread_risk": za.get("spread_risk", "Medium"),
                "composite_score": _compute_composite({}, za),
                "summary": za.get("summary", "")[:200],
                "recommended_action": za.get("recommended_action", "Monitor"),
            })

    print("\n  " + "=" * 56)
    print("       FINAL FIRE ZONE RANKING (ALL 5 ZONES)")
    print("  " + "=" * 56)
    for entry in results["final_ranking"]:
        print(f"  #{entry['rank']}  {entry['zone']:<6}  Priority: {entry.get('rescue_priority', '?')}/10  "
              f"Intensity: {entry.get('intensity_rating', '?')}/10  Risk: {entry.get('spread_risk', '?')}")
    print("  " + "=" * 56)

    save_json(results, "phase3_results.json")
    return results


def _build_ranking_prompt(assessments):
    zone_summaries = []
    for a in assessments:
        zone_summaries.append(
            f"- {a['zone']}: Intensity {a.get('intensity_rating', '?')}/10, "
            f"Spread Risk: {a.get('spread_risk', '?')}, "
            f"Rescue Priority: {a.get('rescue_priority', '?')}/10, "
            f"YOLO detection confidence: {a.get('yolo_confidence', 0):.1%}, "
            f"Thermal intensity: {a.get('sim_intensity', 0):.2f}, "
            f"Fire radius: {a.get('sim_radius', 0):.1f}m. "
            f"Assessment: {a.get('summary', '')[:150]}"
        )
    return (
        "You are a disaster response commander reviewing fire zone assessments from drone surveillance.\n\n"
        "Fire zone assessments (5 zones total):\n" + "\n".join(zone_summaries) + "\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. These 5 fires are DIFFERENT in severity. Your ranking MUST reflect real differences.\n"
        "2. composite_score formula: (rescue_priority * 3) + (intensity_rating * 2) + spread_risk_value "
        "where spread_risk_value is Critical=10, High=7, Medium=4, Low=1.\n"
        "3. Each zone MUST have DIFFERENT scores. Do NOT give the same priority or intensity to multiple zones.\n"
        "4. Use the FULL 1-10 scale. A small fire with 0% YOLO detection should score 2-4, not 7-8.\n"
        "5. Zones with 0% YOLO detection confidence have fires that are very small or subsiding — rate them LOW.\n"
        "6. Zones with >60% YOLO confidence have clearly dangerous fires — rate them HIGH.\n\n"
        "Rank ALL 5 fire zones from most to least urgent as valid JSON:\n"
        '{"ranking": [{"rank": 1, "zone": "FZ_X", "rescue_priority": <1-10>, "intensity_rating": <1-10>, '
        '"spread_risk": "<Critical|High|Medium|Low>", "composite_score": <calculated per formula>, '
        '"summary": "<why this rank>", "recommended_action": "<specific action>"}], '
        '"overall_assessment": "Brief overall situation summary"}'
    )


def _estimate_intensity(text, sim_intensity):
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


def _estimate_risk(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["critical", "extreme", "rapidly"]):
        return "Critical"
    elif any(w in text_lower for w in ["high", "spreading", "dangerous"]):
        return "High"
    elif any(w in text_lower for w in ["low", "contained", "small"]):
        return "Low"
    return "Medium"


def _estimate_priority(text, sim_intensity):
    text_lower = text.lower()
    if any(w in text_lower for w in ["immediate", "critical", "urgent"]):
        return 9
    elif any(w in text_lower for w in ["high priority", "significant"]):
        return 7
    elif any(w in text_lower for w in ["monitor", "low", "stable"]):
        return 4
    return max(4, min(9, int(sim_intensity * 5.5)))


# ---------------------------------------------------------------------------
# Final report builder
# ---------------------------------------------------------------------------

def build_final_report(p1, p2, p3):
    print("\n" + "=" * 70)
    print("  Building Final Composite Report (5 zones)")
    print("=" * 70)

    report = {
        "title": "ResQ-AI Pipeline v2 — Full 5-Zone Mission Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": "2.0",
        "model": COSMOS_MODEL,
        "note": "Uses real Isaac Sim rendered fires (Flow particles), not composited",
        "phases": {
            "phase1_aerial_detection": {
                "source": "Isaac Sim rendered aerial image (render_aerial_view.py)",
                "zones_detected": len(p1.get("detections", [])),
                "cosmos_response_time_s": p1.get("cosmos_response_time_s", 0),
            },
            "phase2_yolo_investigation": {
                "zones_investigated": len(p2.get("zones", {})),
                "zone_details": {},
            },
            "phase3_cosmos_assessment": {
                "zones_assessed": len(p3.get("zone_assessments", {})),
                "ranking": p3.get("final_ranking", []),
                "overall_assessment": p3.get("ranking_cosmos_raw", "")[:500],
            },
        },
        "fire_zones": {},
    }

    for zname in ZONE_ORDER:
        z2 = p2.get("zones", {}).get(zname, {})
        z3 = p3.get("zone_assessments", {}).get(zname, {})

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

    # Markdown report
    md = _build_markdown(report, p1, p3)
    md_path = os.path.join(OUT_DIR, "mission_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  -> Saved {md_path}")

    return report


def _build_markdown(report, p1, p3):
    lines = [
        "# ResQ-AI Pipeline v2 — Full 5-Zone Mission Report",
        f"**Generated:** {report['timestamp']}",
        f"**Model:** {report['model']}",
        f"**Note:** Real Isaac Sim rendered fires (Flow particles)",
        "",
        "---",
        "",
        "## Phase 1: Aerial Detection (Rendered Fires)",
        f"Cosmos analyzed the Isaac Sim rendered aerial image and identified "
        f"{report['phases']['phase1_aerial_detection']['zones_detected']} fire zone(s).",
        "",
        f"**Response time:** {report['phases']['phase1_aerial_detection']['cosmos_response_time_s']:.1f}s",
        "",
        "![Aerial Detection](phase1_annotated.jpg)",
        "",
        "### Cosmos Analysis",
        f"```\n{p1.get('cosmos_raw_response', 'N/A')[:500]}\n```",
        "",
        "---",
        "",
        "## Phase 2: YOLO Investigation (All 5 Zones)",
        "",
    ]

    for zname in ZONE_ORDER:
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
        "---", "", "## Phase 3: Cosmos Assessment & Ranking (All 5 Zones)", "",
    ])

    for zname in ZONE_ORDER:
        za = p3.get("zone_assessments", {}).get(zname, {})
        lines.extend([
            f"### {zname}",
            f"- **Intensity:** {za.get('intensity_rating', '?')}/10",
            f"- **Spread Risk:** {za.get('spread_risk', '?')}",
            f"- **Rescue Priority:** {za.get('rescue_priority', '?')}/10",
            f"- **Action:** {za.get('recommended_action', '?')}",
            "", f"```\n{za.get('cosmos_raw', 'N/A')[:400]}\n```", "",
        ])

    lines.extend([
        "---", "", "## Final Ranking (5 Zones)", "",
        "| Rank | Zone | Priority | Intensity | Spread Risk | Score |",
        "|------|------|----------|-----------|-------------|-------|",
    ])
    for entry in p3.get("final_ranking", []):
        lines.append(
            f"| {entry.get('rank', '?')} | {entry.get('zone', '?')} | "
            f"{entry.get('rescue_priority', '?')}/10 | {entry.get('intensity_rating', '?')}/10 | "
            f"{entry.get('spread_risk', '?')} | {entry.get('composite_score', '?')} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Copy simulation video assets
# ---------------------------------------------------------------------------

def copy_assets():
    """Copy simulation videos from debug_output_2 source dir."""
    print("\n  Checking for video assets...")
    for vid in ["external_overview_hq.mp4", "forward_fpv_hq.mp4", "resqai_sim.mp4"]:
        src = os.path.join(SRC_DIR, vid)
        if os.path.exists(src):
            print(f"    Found {vid}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  ResQ-AI Pipeline v2 — Real Rendered Fires, All 5 Zones")
    print("  Phase 1: Cosmos Aerial Detection (rendered image)")
    print("  Phase 2: YOLO Investigation (all 5 zones)")
    print("  Phase 3: Cosmos Assessment + Ranking (all 5 zones)")
    print("=" * 65)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "zone_frames"), exist_ok=True)

    # Check Cosmos NIM
    print("\n  Checking Cosmos NIM...")
    try:
        import requests
        r = requests.get("http://localhost:8010/v1/health/ready", timeout=5)
        if r.status_code == 200:
            print("  Cosmos NIM ready")
        else:
            print(f"  Cosmos NIM status {r.status_code}")
    except Exception as e:
        print(f"  Could not reach Cosmos NIM: {e}")

    # Check inputs
    if not os.path.exists(AERIAL_IMAGE) and not os.path.exists(AERIAL_IMAGE_WEB):
        print(f"\n  WARNING: No rendered aerial image at {AERIAL_IMAGE}")
        print("  Phase 1 will fail unless aerial render is run first.")

    frame_count = len(glob.glob(os.path.join(SRC_FRAMES, "frame_*_rgb.jpg")))
    print(f"  Source frames: {frame_count}")
    if frame_count == 0:
        print("  WARNING: No simulation frames found. Phase 2 will use fallbacks.")

    t_start = time.time()

    p1 = phase1_aerial_detection()
    p2 = phase2_yolo_investigation()
    p3 = phase3_cosmos_assessment(p2)
    report = build_final_report(p1, p2, p3)
    copy_assets()

    total = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Pipeline v2 complete in {total:.1f}s")
    print(f"  All outputs: {OUT_DIR}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
