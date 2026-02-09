from ultralytics import YOLO
import os
import sys
import traceback

def main():
    print("DEBUG: Starting main function...")
    try:
        # Load a model
        print("DEBUG: Loading model yolov8s.pt...")
        model = YOLO('yolov8s.pt')  # load a pretrained model

        # Use absolute path to data.yaml in the sibling Datasets directory
        # data_path = "test_config.yaml" 
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Datasets/Phase 1 - AIDER-Disaster/data.yaml"))
        print(f"DEBUG: Using data path: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            return

        print("DEBUG: Starting training on GPU...")
        results = model.train(
            data=data_path,
            epochs=50,
            imgsz=640,
            device=0, # Use GPU
            batch=16,
            project='ResQ-AI-Phase1',
            name='yolov8s_aider',
            exist_ok=True, # Fixed typo: exists_ok -> exist_ok
            verbose=True
        )
        print("DEBUG: Training completed successfully.")
        
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # Log to file to capture all output including C++ level errors if possible (though python redirection helps mostly only python level)
    # forcing unbuffered output
    sys.stdout.reconfigure(encoding='utf-8')
    
    with open("train_output.log", "w", encoding='utf-8') as f:
        sys.stdout = f
        sys.stderr = f
        print("DEBUG: Script started.")
        main()
        print("DEBUG: Script finished.")
