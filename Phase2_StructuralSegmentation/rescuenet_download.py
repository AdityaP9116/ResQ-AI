import kaggle
import os

# Configure download path
dataset_name = "yaroslavchyrko/rescuenet"
destination_folder = "../Datasets/Phase2_RescueNet"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

print(f"Downloading {dataset_name} to {destination_folder}...")
try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
    print("Download complete!")
except Exception as e:
    print(f"Error downloading dataset: {e}")
