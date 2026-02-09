from ultralytics import YOLO
import os

def main():
    # Path to the best trained model
    # model_path = r"ResQ-AI-Phase1\yolov8s_aider\weights\best.pt"
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train_yolo.py first.")
        return

    # Load the trained model
    model = YOLO(model_path)

    # Export the model to TensorRT format
    model.export(
        format='engine',
        device=0,
        half=True,  # FP16 quantization
        simplify=True,
        workspace=4 # Workspace size (GB)
    )
    
    print(f"Export complete. TensorRT engine should be in the same directory as {model_path}")

if __name__ == '__main__':
    main()
