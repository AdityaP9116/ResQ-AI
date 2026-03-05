import sys
import os
from huggingface_hub import HfApi

def main():
    if len(sys.argv) != 2:
        print("Error: Missing Hugging Face Token.")
        print("Usage: python upload_datasets.py <YOUR_HUGGINGFACE_WRITE_TOKEN>")
        sys.exit(1)

    token = sys.argv[1]
    api = HfApi(token=token)

    # 1. Upload Phase 1 (Scout) Dataset
    phase1_path = "Datasets/Phase 1 - AIDER-Disaster"
    phase1_repo = "ResQAI/Aider"
    
    if os.path.exists(phase1_path):
        print(f"Uploading Phase 1 (Aider) Dataset to {phase1_repo}...\nThis may take several minutes.")
        try:
            api.upload_folder(
                folder_path=phase1_path,
                repo_id=phase1_repo,
                repo_type="dataset",
            )
            print("✅ Phase 1 Dataset upload successful!")
        except Exception as e:
            print(f"❌ Error uploading Phase 1 Dataset: {e}")
    else:
        print(f"❌ Error: Could not find Phase 1 dataset at {phase1_path}")

    # 2. Upload Phase 2 (Navigator) Dataset
    phase2_path = "Datasets/Phase2_RescueNet"
    phase2_repo = "ResQAI/RescueNet"

    if os.path.exists(phase2_path):
        print(f"\nUploading Phase 2 (RescueNet) Dataset to {phase2_repo}...\nThis is a large file and will take some time.")
        try:
            api.upload_folder(
                folder_path=phase2_path,
                repo_id=phase2_repo,
                repo_type="dataset",
            )
            print("✅ Phase 2 Dataset upload successful!")
        except Exception as e:
            print(f"❌ Error uploading Phase 2 Dataset: {e}")
    else:
        print(f"❌ Error: Could not find Phase 2 dataset at {phase2_path}")
        
    print("\nAll dataset uploads completed.")

if __name__ == "__main__":
    main()
