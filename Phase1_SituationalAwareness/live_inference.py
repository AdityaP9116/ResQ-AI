from ultralytics import YOLO
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ResQ-AI Phase 1: Live/Video Inference")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--model", type=str, default="best.engine", help="Path to model (best.engine or best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save the inference results to file")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback to .pt if engine not found (e.g. if user didn't run export)
        if os.path.exists("best.pt"):
            print(f"Warning: {model_path} not found. Falling back to best.pt")
            model_path = "best.pt"
        else:
            print(f"Error: Model not found at {model_path}")
            return

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path, task='detect')

    source = args.source
    # Convert "0" to integer 0 for webcam
    if source == "0":
        source = 0
    
    print(f"Starting inference on source: {source}")
    
    # Run inference with show=True to display window
    # stream=True ensures it's a generator (good for long videos) but for simple viewing show=True handles the loop mostly.
    # However, for custom control/overlay, we can iterate.
    
    # Run inference with stream=True
    # We disable show=True and handle display manually to ensure it fits the screen
    results = model.predict(source=source, show=False, conf=args.conf, save=args.save, stream=True, device=0)
    
    print("Press 'q' in the display window to exit.")
    
    for r in results:
        # Plot the results on the frame
        im_array = r.plot()
        
        # Resize for display if too large (e.g. > 720p height)
        height, width = im_array.shape[:2]
        max_height = 720
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            im_display = cv2.resize(im_array, (new_width, max_height))
        else:
            im_display = im_array

        cv2.imshow("ResQ-AI Live Inference", im_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
