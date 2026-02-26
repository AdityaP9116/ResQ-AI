import kaggle
import sys

# Configure dataset
dataset_name = "yaroslavchyrko/rescuenet"

print(f"Listing files in {dataset_name}...")
try:
    kaggle.api.authenticate()
    files = kaggle.api.dataset_list_files(dataset_name)
    
    print(f"Found {len(files.files)} files.")
    # Print first 20 files to get structure
    for i, f in enumerate(files.files):
        if i < 20:
            print(f.name)
        else:
            break
            
except Exception as e:
    print(f"Error listing files: {e}")
