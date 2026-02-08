from ultralytics import YOLO
import glob
import os

def main():
    engine_path = "best.engine"
    if not os.path.exists(engine_path):
        print(f"Error: {engine_path} not found.")
        return

    print(f"Loading TensorRT engine: {engine_path}")
    # Load the exported TensorRT model
    # Note: Ultralytics automatically handles loading .engine files if tensorrt is installed
    model = YOLO(engine_path, task='detect')

    # Find a test image
    test_images = glob.glob(r"Datasets\Phase 1 - AIDER-Disaster\test\images\*.jpg")
    if not test_images:
        print("No test images found.")
        return
    
    test_image = test_images[0]
    print(f"Running inference on: {test_image}")

    # Run inference
    try:
        results = model.predict(test_image, device=0)
        print("Inference successful!")
        for r in results:
            print(f"Detected {len(r.boxes)} objects.")
            r.save(filename='inference_result.jpg')
            print("Result saved to inference_result.jpg")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
