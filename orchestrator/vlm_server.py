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

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ResQ-AI VLM Server")

# ---------------------------------------------------------------------------
# Globals set by CLI args at startup
# ---------------------------------------------------------------------------
_BACKEND: str = "mock"
_NVIDIA_API_KEY: str = ""
_VLLM_URL: str = "http://localhost:8001"
_COSMOS_MODEL: str = "nvidia/cosmos-reason2-2b"

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
    """Deterministic priority-based waypoint selection."""
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
    if not observations:
        return {
            "status": "nominal",
            "advice": "No hazards detected. Continue survey.",
            "decision": "continue",
            "target_waypoint": None,
            "reasoning": "No observations in context.",
        }

    candidates = []
    for obs in observations:
        cls_name = _normalize_class(obs.get("class") or "")
        priority = PRIORITY_MAP.get(cls_name, 0)
        world_xyz = obs.get("world_xyz")
        if world_xyz is None or len(world_xyz) < 3:
            continue
        candidates.append((cls_name, priority, world_xyz, obs))

    if not candidates:
        return {
            "status": "nominal",
            "advice": "Hazards detected but no valid 3D positions.",
            "decision": "continue",
            "target_waypoint": None,
            "reasoning": "Missing world_xyz coordinates.",
        }

    candidates.sort(key=lambda c: -c[1])
    best_cls, _, (hx, hy, hz), _ = candidates[0]
    hx, hy, hz = float(hx), float(hy), float(hz)
    wp_z = hz + OBSERVATION_ALTITUDE_OFFSET
    target_waypoint = [hx, hy, wp_z]

    if best_cls == "fire":
        reasoning = (
            f"Fire is highest priority — active and poses immediate danger. "
            f"Approaching [{hx:.1f}, {hy:.1f}, {wp_z:.1f}] from {OBSERVATION_ALTITUDE_OFFSET}m altitude."
        )
    else:
        reasoning = (
            f"{best_cls.capitalize()} requires investigation. "
            f"Approaching [{hx:.1f}, {hy:.1f}, {wp_z:.1f}] from {OBSERVATION_ALTITUDE_OFFSET}m altitude."
        )

    return {
        "status": "critical",
        "advice": f"Prioritizing {best_cls} at [{hx:.1f}, {hy:.1f}, {hz:.1f}]. {reasoning}",
        "decision": "investigate",
        "target_waypoint": target_waypoint,
        "reasoning": reasoning,
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
        default=os.environ.get("RESQAI_COSMOS_MODEL", "nvidia/cosmos-reason2-2b"),
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
