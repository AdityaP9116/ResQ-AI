import os
from huggingface_hub import hf_hub_download, snapshot_download

def setup_brev_environment():
    print("🚀 Initializing ResQ-AI Brev.dev Environment Setup from Hugging Face 🚀\n")

    # 1. Download Phase 1 (Scout) Model
    print("Downloading Phase 1 Model...")
    phase1_dir = os.path.join(os.path.dirname(__file__), "Phase1_SituationalAwareness")
    os.makedirs(phase1_dir, exist_ok=True)
    hf_hub_download(
        repo_id="ResQAI/Yolo-Phase1",
        filename="best.pt",
        local_dir=phase1_dir
    )
    print("✅ Phase 1 Model ready.\n")

    # 2. Download Phase 2 (Navigator) Model
    print("Downloading Phase 2 Model...")
    phase2_dir = os.path.join(os.path.dirname(__file__), "Phase2_StructuralSegmentation")
    os.makedirs(phase2_dir, exist_ok=True)
    hf_hub_download(
        repo_id="ResQAI/Yolo-Phase2",
        filename="best.pt",
        local_dir=phase2_dir
    )
    print("✅ Phase 2 Model ready.\n")

    # 3. Download Phase 1 Dataset
    print("Downloading AIDER Dataset (Phase 1)...")
    dataset1_dir = os.path.join(os.path.dirname(__file__), "Datasets", "Phase 1 - AIDER-Disaster")
    os.makedirs(dataset1_dir, exist_ok=True)
    snapshot_download(
        repo_id="ResQAI/Aider",
        repo_type="dataset",
        local_dir=dataset1_dir
    )
    print("✅ AIDER Dataset ready.\n")

    # 4. Download Phase 2 Dataset
    print("Downloading RescueNet Dataset (Phase 2)... (This may take a while)")
    dataset2_dir = os.path.join(os.path.dirname(__file__), "Datasets", "Phase2_RescueNet")
    os.makedirs(dataset2_dir, exist_ok=True)
    snapshot_download(
        repo_id="ResQAI/RescueNet",
        repo_type="dataset",
        local_dir=dataset2_dir
    )
    print("✅ RescueNet Dataset ready.\n")
    
    print("🎉 All models and datasets successfully pulled from Hugging Face! The Brev machine is ready to run.")

if __name__ == "__main__":
    setup_brev_environment()
