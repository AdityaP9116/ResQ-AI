from ultralytics import YOLO
import os
import glob

def main():
    # Model Path
    model_path = "best.engine"
    if not os.path.exists(model_path):
        if os.path.exists("best.pt"):
            print("Engine not found, using .pt")
            model_path = "best.pt"
        else:
            print("No model found!")
            return

    # Dataset Path (Relative to Phase1_SituationalAwareness)
    test_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/Phase 1 - AIDER-Disaster/test/images"))
    
    if not os.path.exists(test_images_path):
        print(f"Error: Test dataset directory not found at {test_images_path}")
        return

    print(f"Loading model: {model_path}")
    print(f"Processing images from: {test_images_path}")
    
    model = YOLO(model_path, task='detect')

    # Run inference on the whole directory
    # save=True will save images with bounding boxes
    # conf=0.25 is standard
    # device=0 for GPU
    results = model.predict(
        source=test_images_path,
        save=True,
        conf=0.25,
        iou=0.45,
        device=0,
        project='runs/detect',
        name='test_set_inference',
        exist_ok=True # Overwrite if exists, or just add to it. Actually exist_ok=True allows writing to same folder.
    )

    print(f"Processing complete. Results saved to runs/detect/test_set_inference")

if __name__ == "__main__":
    main()
