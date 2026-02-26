from ultralytics import YOLO
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ResQ-AI Phase 2: Live Segmentation Inference")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to segmentation model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save the inference results to file")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Make sure to move 'best.pt' here!")
        return

    print(f"Loading segmentation model: {model_path}...")
    model = YOLO(model_path, task='segment')

    source = args.source
    if source == "0":
        source = 0
    
    print(f"Starting inference on source: {source}")
    
    # Run inference with stream=True
    results = model.predict(source=source, show=False, conf=args.conf, save=args.save, stream=True, device=0, retina_masks=True)
    
    print("Press 'q' in the display window to exit.")
    
    for r in results:
        # Plot segmentation masks
        im_array = r.plot()
        
        # Resize for display
        height, width = im_array.shape[:2]
        max_height = 720
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            im_display = cv2.resize(im_array, (new_width, max_height))
        else:
            im_display = im_array

        cv2.imshow("ResQ-AI Segmentation", im_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
