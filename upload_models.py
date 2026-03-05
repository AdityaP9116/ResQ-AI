import sys
import os
from huggingface_hub import HfApi

def main():
    if len(sys.argv) != 2:
        print("Error: Missing Hugging Face Token.")
        print("Usage: python upload_models.py <YOUR_HUGGINGFACE_WRITE_TOKEN>")
        sys.exit(1)

    token = sys.argv[1]
    api = HfApi(token=token)

    # 1. Upload Phase 1 (Scout)
    phase1_path = "Phase1_SituationalAwareness/best.pt"
    phase1_repo = "ResQAI/Yolo-Phase1"
    
    if os.path.exists(phase1_path):
        print(f"Uploading Phase 1 (Scout) model to {phase1_repo}...")
        try:
            api.upload_file(
                path_or_fileobj=phase1_path,
                path_in_repo="best.pt",
                repo_id=phase1_repo,
                repo_type="model",
            )
            print("✅ Phase 1 upload successful!")
        except Exception as e:
            print(f"❌ Error uploading Phase 1: {e}")
    else:
        print(f"❌ Error: Could not find Phase 1 model at {phase1_path}")

    # 2. Upload Phase 2 (Navigator)
    phase2_path = "runs/content/runs/rescuenet_seg/weights/best.pt"
    phase2_repo = "ResQAI/Yolo-Phase2"

    if os.path.exists(phase2_path):
        print(f"\nUploading Phase 2 (Navigator) model to {phase2_repo}...")
        try:
            api.upload_file(
                path_or_fileobj=phase2_path,
                path_in_repo="best.pt",
                repo_id=phase2_repo,
                repo_type="model",
            )
            print("✅ Phase 2 upload successful!")
        except Exception as e:
            print(f"❌ Error uploading Phase 2: {e}")
    else:
        print(f"❌ Error: Could not find Phase 2 model at {phase2_path}")
        
    print("\nAll uploads completed.")

if __name__ == "__main__":
    main()
