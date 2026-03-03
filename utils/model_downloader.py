"""ResQ-AI Model Weight Downloader.

Automatically downloads YOLO weights from HuggingFace Hub if they are not
already cached locally.  Both repos are public, so no token is required by
default.  Set ``HF_TOKEN`` in ``.env`` for private repos.

Usage::

    from utils.model_downloader import get_phase1_weights, get_phase2_weights

    phase1_path = get_phase1_weights()   # → "weights/phase1_best.pt"
    phase2_path = get_phase2_weights()   # → "weights/phase2_best.pt"
"""

from __future__ import annotations

import os
import shutil

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ── Configuration (overridable via .env) ─────────────────────────────────
_PHASE1_REPO = os.environ.get("RESQAI_PHASE1_REPO", "ResQAI/Yolo-Phase1")
_PHASE2_REPO = os.environ.get("RESQAI_PHASE2_REPO", "ResQAI/Yolo-Phase2")
_HF_TOKEN = os.environ.get("HF_TOKEN", None) or None  # None if empty string
_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "weights")


def ensure_weights(
    repo_id: str,
    filename: str = "best.pt",
    local_name: str | None = None,
) -> str:
    """Download a model file from HuggingFace Hub if not already cached.

    Args:
        repo_id: HuggingFace repo ID, e.g. ``"ResQAI/Yolo-Phase1"``.
        filename: Name of the file in the repo (default ``"best.pt"``).
        local_name: Name to save locally under ``weights/``.  If ``None``,
            uses *filename*.

    Returns:
        Absolute path to the local weight file.
    """
    os.makedirs(_WEIGHTS_DIR, exist_ok=True)
    local_name = local_name or filename
    local_path = os.path.join(_WEIGHTS_DIR, local_name)

    if os.path.isfile(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[ModelDL] {local_name} already cached ({size_mb:.1f}MB) → {local_path}")
        return local_path

    print(f"[ModelDL] Downloading {repo_id}/{filename} → {local_path} …")

    try:
        from huggingface_hub import hf_hub_download

        # Download to HF cache, then copy to our weights dir
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=_HF_TOKEN,
        )
        shutil.copy2(cached_path, local_path)

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"[ModelDL] ✅ Downloaded {local_name} ({size_mb:.1f}MB)")
        return local_path

    except ImportError:
        print("[ModelDL] ❌ huggingface_hub not installed. Install with:")
        print("         pip install huggingface_hub")
        print(f"[ModelDL] Falling back to local search …")
        return _fallback_local(local_name, filename)

    except Exception as exc:
        print(f"[ModelDL] ❌ Download failed: {exc}")
        print(f"[ModelDL] Falling back to local search …")
        return _fallback_local(local_name, filename)


def _fallback_local(local_name: str, filename: str) -> str:
    """Search common local paths for the weight file."""
    candidates = [
        os.path.join(_WEIGHTS_DIR, local_name),
        os.path.join(_PROJECT_ROOT, "Phase1_SituationalAwareness", filename),
        os.path.join(_PROJECT_ROOT, "Phase2_StructuralSegmentation", filename),
        os.path.join(_PROJECT_ROOT, filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"[ModelDL] Found local fallback: {path}")
            return path

    print(f"[ModelDL] ⚠️  No local weights found for {local_name}")
    return os.path.join(_WEIGHTS_DIR, local_name)  # return expected path anyway


# ── Convenience wrappers ─────────────────────────────────────────────────

def get_phase1_weights() -> str:
    """Get Phase 1 (detection) YOLO weights, downloading if needed."""
    return ensure_weights(_PHASE1_REPO, "best.pt", "phase1_best.pt")


def get_phase2_weights() -> str:
    """Get Phase 2 (segmentation) YOLO weights, downloading if needed."""
    return ensure_weights(_PHASE2_REPO, "best.pt", "phase2_best.pt")


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ResQ-AI Model Downloader — standalone test")
    print("=" * 60)
    p1 = get_phase1_weights()
    p2 = get_phase2_weights()
    print(f"\nPhase 1: {p1}  (exists={os.path.isfile(p1)})")
    print(f"Phase 2: {p2}  (exists={os.path.isfile(p2)})")
    print("Done.")
