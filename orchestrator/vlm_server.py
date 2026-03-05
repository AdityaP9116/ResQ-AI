"""VLM server for ResQ-AI — routes requests to Cosmos Reason 2 or falls back to mock logic.

Supports three backends (selected via environment / CLI):

1. **NVIDIA NIM cloud** (``--backend nim``):
   Calls ``https://integrate.api.nvidia.com/v1/chat/completions`` using
   the OpenAI-compatible endpoint.  Requires ``NVIDIA_API_KEY`` env var.

2. **Local vLLM** (``--backend vllm``):
   Calls a local vLLM server at ``http://localhost:8001/v1/chat/completions``
   (start it with ``vllm serve nvidia/Cosmos-Reason2-2B --port 8001``).

3. **Mock** (``--backend mock``, default):
   Deterministic priority logic — no GPU required.  Good for testing the
   pipeline end-to-end without a real model.

Usage::

    # Mock (no API key needed)
    python orchestrator/vlm_server.py

    # NVIDIA NIM cloud
    export NVIDIA_API_KEY="nvapi-..."
    python orchestrator/vlm_server.py --backend nim

    # Local vLLM (start vLLM first in another terminal)
    python orchestrator/vlm_server.py --backend vllm --vllm-url http://localhost:8001
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from typing import Any

# Auto-load .env from project root (so NVIDIA_API_KEY etc. are available)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import logging
import sys

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("VLMServer")

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ResQ-AI VLM Server")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Globals set by CLI args or ENV at startup
# ---------------------------------------------------------------------------
_BACKEND: str = os.getenv("VLM_BACKEND", "mock")
_NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "").strip()
_VLLM_URL: str = "http://localhost:8001"
_COSMOS_MODEL: str = os.getenv("RESQAI_COSMOS_MODEL", "nvidia/cosmos-reason2-8b")

logger.info(f"Loaded NVIDIA_API_KEY: {_NVIDIA_API_KEY[:5]}***" if _NVIDIA_API_KEY else "Loaded NVIDIA_API_KEY: None/Empty")


# ---------------------------------------------------------------------------
# Hazard priority for mock backend
# ---------------------------------------------------------------------------
PRIORITY_MAP = {
    "fire": 4,
    "person": 3,
    "vehicle": 2,
    "building": 1,
}
OBSERVATION_ALTITUDE_OFFSET = 18.0


class AnalyzeRequest(BaseModel):
    image_base64: str
    context: str


# ═══════════════════════════════════════════════════════════════════════════
# Backend 1: Mock (no GPU)
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_class(raw: str) -> str:
    cls = raw.lower().strip()
    if "fire" in cls:
        return "fire"
    if cls in ("person", "pedestrian"):
        return "person"
    if cls in ("vehicle", "car", "truck"):
        return "vehicle"
    if cls in ("building", "rubble", "collapsed"):
        return "building"
    return "building"


def _mock_inference(context: str) -> dict[str, Any]:
    """Priority-based waypoint selection using fire density + people proximity.

    Scores each fire zone by: nearby fire detections, nearby people,
    burn area, and intensity. Areas with BOTH fire and people are CRITICAL.
    """
    try:
        ctx = json.loads(context)
    except json.JSONDecodeError:
        return {
            "status": "nominal",
            "advice": "Invalid context JSON.",
            "decision": "continue",
            "target_waypoint": None,
            "reasoning": "Could not parse Cosmos prompt.",
        }

    observations = ctx.get("observations", [])
    drone_state = ctx.get("drone_state", {})
    drone_pos = drone_state.get("position", [0, 0, 110])
    frame_idx = ctx.get("frame", 0)
    fire_situation = ctx.get("fire_situation", {})
    active_fires = fire_situation.get("active_fires", [])
    total_burn_area = fire_situation.get("total_area_burning_m2", 0)

    if not observations:
        return {
            "status": "nominal",
            "advice": "No hazards in view. Continuing bird's eye survey sweep.",
            "decision": "continue",
            "target_waypoint": None,
            "reasoning": (
                "No active threats detected in current camera field of view. "
                "Recommend continuing orbital survey to build full situational map. "
                f"Fire situation: {len(active_fires)} active zones, {total_burn_area:.0f}m² burning."
            ),
        }

    # ---- Classify all observations ----
    fires = []
    people = []
    others = []
    for obs in observations:
        cls_name = _normalize_class(obs.get("class") or "")
        world_xyz = obs.get("world_xyz")
        if world_xyz is None or len(world_xyz) < 3:
            continue
        entry = {"cls": cls_name, "pos": world_xyz, "conf": obs.get("confidence", 0), "obs": obs}
        if cls_name == "fire":
            fires.append(entry)
        elif cls_name == "person":
            people.append(entry)
        else:
            others.append(entry)

    # ---- Score each fire zone by fire density + people proximity ----
    zone_scores = []
    for af in active_fires:
        zpos = af.get("position", [0, 0, 0])
        burn_area = 3.14159 * af.get("radius", 5) ** 2
        intensity = af.get("intensity", 0.5)

        # Count YOLO fire detections near this zone (within 30m)
        nearby_fires = sum(1 for f in fires
                          if ((f["pos"][0] - zpos[0])**2 + (f["pos"][1] - zpos[1])**2) < 900)
        # Count people near this zone (within 40m)
        nearby_people = sum(1 for p in people
                           if ((p["pos"][0] - zpos[0])**2 + (p["pos"][1] - zpos[1])**2) < 1600)

        # Composite priority score: people near fire are CRITICAL
        score = (nearby_fires * 3.0 +
                 nearby_people * 5.0 +
                 burn_area * 0.01 +
                 intensity * 2.0)

        zone_scores.append({
            "zone": af.get("zone", "unknown"),
            "pos": zpos,
            "score": score,
            "nearby_fires": nearby_fires,
            "nearby_people": nearby_people,
            "burn_area": burn_area,
            "intensity": intensity,
        })

    # If no fire zones in report but YOLO sees fires, use YOLO detection positions
    if not zone_scores and fires:
        for f in fires:
            nearby_people = sum(1 for p in people
                               if ((p["pos"][0] - f["pos"][0])**2 + (p["pos"][1] - f["pos"][1])**2) < 1600)
            score = 3.0 + nearby_people * 5.0 + f["conf"] * 2.0
            zone_scores.append({
                "zone": "YOLO_detection",
                "pos": f["pos"],
                "score": score,
                "nearby_fires": 1,
                "nearby_people": nearby_people,
                "burn_area": 0,
                "intensity": f["conf"],
            })

    # Fall back to any observation (people or other hazards)
    if not zone_scores:
        best = people[0] if people else (others[0] if others else None)
        if best is None:
            return {
                "status": "nominal",
                "advice": "Detected objects but no valid 3D positions. Continue survey.",
                "decision": "continue",
                "target_waypoint": None,
                "reasoning": "Missing 3D coordinates for detected objects.",
            }
        hx, hy, hz = float(best["pos"][0]), float(best["pos"][1]), float(best["pos"][2])
        return {
            "status": "critical" if best["cls"] == "person" else "nominal",
            "advice": f"{best['cls'].upper()} detected at [{hx:.1f}, {hy:.1f}]. Moving to observe from above.",
            "decision": "investigate",
            "target_waypoint": [hx, hy, drone_pos[2]],
            "reasoning": f"{best['cls'].upper()} at [{hx:.1f}, {hy:.1f}, {hz:.1f}]. Priority target for investigation.",
        }

    # ---- Pick the highest-priority zone ----
    zone_scores.sort(key=lambda z: -z["score"])
    best_zone = zone_scores[0]
    bx, by = float(best_zone["pos"][0]), float(best_zone["pos"][1])
    target_alt = drone_pos[2]  # maintain current altitude
    target_waypoint = [bx, by, target_alt]

    dist = ((drone_pos[0] - bx)**2 + (drone_pos[1] - by)**2)**0.5

    # Build detailed priority reasoning
    priority_list = ", ".join(
        f"{z['zone']}(score={z['score']:.1f}, fires={z['nearby_fires']}, "
        f"people={z['nearby_people']}, area={z['burn_area']:.0f}m²)"
        for z in zone_scores[:3]
    )

    critical = best_zone["nearby_people"] > 0 and best_zone["nearby_fires"] > 0

    reasoning = (
        f"PRIORITY ANALYSIS: {len(zone_scores)} fire zones evaluated. "
        f"{'CRITICAL — PEOPLE NEAR FIRE: ' if critical else ''}"
        f"Zone {best_zone['zone']} is TOP PRIORITY "
        f"(score={best_zone['score']:.1f}: {best_zone['nearby_fires']} fire detections, "
        f"{best_zone['nearby_people']} people nearby, {best_zone['burn_area']:.0f}m² burning, "
        f"intensity={best_zone['intensity']:.2f}). "
        f"{'Civilians detected near active fire — rescue urgency ELEVATED. ' if critical else ''}"
        f"All zones ranked: [{priority_list}]. "
        f"DIRECTIVE: Fly to [{bx:.1f}, {by:.1f}, {target_alt:.0f}] — "
        f"maintain bird's eye altitude for overview. "
        f"Distance: {dist:.0f}m. Total area burning: {total_burn_area:.0f}m²."
    )

    advice = (
        f"{'CRITICAL: People near fire! ' if critical else ''}"
        f"Priority target: {best_zone['zone']} at [{bx:.1f}, {by:.1f}] — "
        f"{best_zone['nearby_fires']} fires, {best_zone['nearby_people']} people, "
        f"{best_zone['burn_area']:.0f}m² burning. "
        f"{'Deploy ground rescue immediately.' if critical else 'Monitor and assess.'}"
    )

    return {
        "status": "critical" if best_zone["nearby_fires"] > 0 else "nominal",
        "advice": advice,
        "decision": "investigate",
        "target_waypoint": target_waypoint,
        "reasoning": reasoning,
        "action": "ROUTE_OVERRIDE",
        "vector_x": target_waypoint[0],
        "vector_y": target_waypoint[1],
        "altitude_adjustment": 0,  # maintain altitude
        "fire_count": len(fires),
        "person_count": len(people),
        "distance_to_target": round(dist, 1),
        "priority_score": round(best_zone["score"], 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Backend 2 & 3: Real Cosmos Reason 2 (NIM cloud or local vLLM)
# ═══════════════════════════════════════════════════════════════════════════

def _cosmos_inference(image_b64: str, context: str) -> dict[str, Any]:
    """Call Cosmos Reason 2 via OpenAI-compatible chat completions endpoint."""
    import openai

    if _BACKEND == "nim":
        base_url = "https://integrate.api.nvidia.com/v1"
        api_key = _NVIDIA_API_KEY
    else:
        base_url = f"{_VLLM_URL}/v1"
        api_key = "unused"

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    system_prompt = (
        "You are a search-and-rescue navigation agent for an autonomous drone. "
        "Given the drone's current position and 3-D world coordinates of detected hazards, "
        "output the next waypoint [x, y, z] the drone should fly to in order to investigate "
        "the highest-priority hazard while maintaining a safe stand-off distance (15-20m above). "
        "Priority order: fire > person > vehicle > building. "
        "Respond ONLY with JSON: "
        '{"target_waypoint": [x, y, z], "reasoning": "...", "decision": "investigate|continue", '
        '"status": "critical|nominal", "advice": "..."}'
    )

    user_content: list[dict[str, Any]] = []
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    user_content.append({
        "type": "text",
        "text": f"Hazard context:\n{context}\n\nProvide your waypoint decision as JSON.",
    })

    try:
        response = client.chat.completions.create(
            model=_COSMOS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content or ""
        print(f"[Cosmos] Raw response: {raw_text[:200]}")

        # Try to extract JSON from the response
        result = _extract_json_from_response(raw_text)
        if result:
            # Ensure required keys exist
            result.setdefault("status", "critical")
            result.setdefault("advice", raw_text[:200])
            result.setdefault("decision", "investigate")
            result.setdefault("reasoning", "")
            result.setdefault("target_waypoint", None)
            return result

        # Couldn't parse JSON — fall back to mock with the raw text as advice
        print("[Cosmos] Could not parse JSON from response, falling back to mock logic")
        mock_result = _mock_inference(context)
        mock_result["advice"] = f"[Cosmos raw] {raw_text[:200]}. [Mock fallback] {mock_result['advice']}"
        return mock_result

    except Exception as exc:
        print(f"[Cosmos] API call failed: {exc}")
        print("[Cosmos] Falling back to mock logic")
        mock_result = _mock_inference(context)
        mock_result["advice"] = f"[Cosmos unavailable: {exc}] {mock_result['advice']}"
        return mock_result


def _extract_json_from_response(text: str) -> dict | None:
    """Try to parse JSON from model output, handling markdown code fences."""
    text = text.strip()

    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in text
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
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze_image(request: AnalyzeRequest):
    print(f"[VLM] Received request (backend={_BACKEND})")

    if _BACKEND == "mock":
        time.sleep(0.5)  # Simulate processing
        return _mock_inference(request.context)
    else:
        return _cosmos_inference(request.image_base64, request.context)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": _BACKEND}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResQ-AI VLM Server")
    parser.add_argument(
        "--backend",
        choices=["mock", "nim", "vllm"],
        default=os.environ.get("RESQAI_VLM_BACKEND", "mock"),
        help="Inference backend: mock (no GPU), nim (NVIDIA cloud), vllm (local)",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("RESQAI_VLLM_URL", "http://localhost:8001"),
        help="Base URL of local vLLM server (for --backend vllm)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RESQAI_COSMOS_MODEL", "nvidia/cosmos-reason2-8b"),
        help="Model name for chat completions",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_server_args()

    _BACKEND = args.backend
    _VLLM_URL = args.vllm_url
    _COSMOS_MODEL = args.model
    _NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

    if _BACKEND == "nim" and not _NVIDIA_API_KEY:
        print("=" * 60)
        print("ERROR: --backend nim requires NVIDIA_API_KEY environment variable.")
        print("")
        print("Get your API key from: https://build.nvidia.com/settings/api-keys")
        print("Then run:")
        print("  export NVIDIA_API_KEY='nvapi-...'")
        print("  python orchestrator/vlm_server.py --backend nim")
        print("=" * 60)
        raise SystemExit(1)

    print(f"[VLM Server] Backend : {_BACKEND}")
    if _BACKEND == "nim":
        print(f"[VLM Server] API Key : {_NVIDIA_API_KEY[:12]}...")
        print(f"[VLM Server] Model   : {_COSMOS_MODEL}")
    elif _BACKEND == "vllm":
        print(f"[VLM Server] vLLM URL: {_VLLM_URL}")
        print(f"[VLM Server] Model   : {_COSMOS_MODEL}")
    else:
        print("[VLM Server] Using deterministic mock logic (no GPU needed)")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
