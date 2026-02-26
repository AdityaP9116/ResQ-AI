import cv2
import argparse
import sys
import os
import torch
import threading
import requests
import base64
import numpy as np
import json
import time
import subprocess
from ultralytics import YOLO
from ultralytics import YOLO
from logic_gates import HazardTracker
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="ResQ-AI Orchestrator")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., .mp4)")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/analyze", help="URL of the VLM server")
    return parser.parse_args()

# VLM Communication Globals
vlm_responses = {}
vlm_requested = set()
vlm_lock = threading.Lock()

def query_vlm_async(vlm_url, hazard_id, image_crop, context):
    """Sends a Base64 encoded image crop to the VLM server asynchronously."""
    print(f"[{hazard_id}] Sending to VLM for reasoning...")
    try:
        _, buffer = cv2.imencode('.jpg', image_crop)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "image_base64": img_str,
            "context": context
        }
        response = requests.post(vlm_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            with vlm_lock:
                vlm_responses[hazard_id] = data
            print(f"[{hazard_id}] VLM Response Received: {data['advice']}")
        else:
            print(f"[{hazard_id}] VLM Server Error: {response.status_code}")
    except Exception as e:
        print(f"[{hazard_id}] VLM Request Failed: {e}")

def main():
    args = parse_args()
    video_path = args.video

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        sys.exit(1)

    # Get standard video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Loaded: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} total frames.")
    
    frame_idx = 0
    
    # Create the main AR Dashboard window
    window_name = "ResQ-AI Operator Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # Default reasonable size
    
    # Load YOLOv8 for Phase 1 (Scout)
    print("Loading Phase 1 YOLOv8 model...")
    # Using small model for testing, will swap to custom weights later
    yolo_phase1_path = os.path.join(os.path.dirname(__file__), "..", "Phase1_SituationalAwareness", "best.pt")
    if not os.path.exists(yolo_phase1_path):
        print(f"Warning: {yolo_phase1_path} not found. Falling back to yolov8n.pt")
        yolo_phase1_path = "yolov8n.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    phase1_model = YOLO(yolo_phase1_path)

    # Load Phase 2 (Navigator) Segmentation Model
    print("Loading Phase 2 YOLOv8-Seg model...")
    yolo_phase2_path = os.path.join(os.path.dirname(__file__), "..", "Phase2_StructuralSegmentation", "best.pt")
    if not os.path.exists(yolo_phase2_path):
        print(f"Warning: {yolo_phase2_path} not found. Falling back to yolov8n-seg.pt")
        yolo_phase2_path = "yolov8n-seg.pt"
    phase2_model = YOLO(yolo_phase2_path)
    
    # Phase 2 Trigger Configuration
    phase2_cooldown_max = 150 # Run Phase 2 for X frames after hazard detection
    phase2_active_timer = 0

    # Flight Log & GPS Telemetry Simulator
    flight_report = []
    # Starting coordinates (e.g., Houston, TX staging area)
    current_lat = 29.7604
    current_lon = -95.3698
    hazards_dir = os.path.join(os.path.dirname(__file__), "hazards")
    os.makedirs(hazards_dir, exist_ok=True)

    # Initialize Logic Gates
    tracker = HazardTracker(iou_threshold=0.5, debounce_frames=5)
    
    print("Starting processing loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        # Scale frame to a maximum of 1280x720 while maintaining aspect ratio
        target_w, target_h = 1280, 720
        h, w = frame.shape[:2]
        
        # Calculate aspect ratios
        aspect = w / h
        if w > target_w or h > target_h:
            if target_w / aspect <= target_h:
                new_w = target_w
                new_h = int(new_w / aspect)
            else:
                new_h = target_h
                new_w = int(new_h * aspect)
            frame = cv2.resize(frame, (new_w, new_h))
            
        frame_idx += 1
        
        # Simulate Drone Movement (Increment GPS coordinates slightly every frame)
        current_lat += 0.00001
        current_lon += 0.00001
        
        # --- Phase 1: Scout ---
        # Run inference on the current frame
        results = phase1_model.predict(frame, device=device, verbose=False)
        frame_drawn = frame.copy()
        
        # Extract Phase 1 detections for the tracker
        detected_boxes = []
        class_ids = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get box coordinates in (left, top, right, bottom) format
                b = box.xyxy[0].cpu().numpy().tolist()
                c = int(box.cls[0].item())
                detected_boxes.append(b)
                class_ids.append(c)
                
                # Draw all raw detections in subtle yellow
                cv2.rectangle(frame_drawn, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 200), 1)
                
                # Add tiny label background based on text size
                label = f"{phase1_model.names[c]}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame_drawn, (int(b[0]), int(b[1])-20), (int(b[0])+w, int(b[1])), (0, 150, 150), -1)
                cv2.putText(frame_drawn, label, (int(b[0]), int(b[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Update Logic Gates
        active_hazards = tracker.update(detected_boxes, class_ids)
        
        # Draw fully active (debounced) hazards in RED
        for hazard in active_hazards:
            b = hazard['box']
            c = hazard['class_id']
            hid = hazard['id']
            cv2.rectangle(frame_drawn, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
            
            label = f"HAZARD {hid}: {phase1_model.names[c]}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_drawn, (int(b[0]), int(b[1])-25), (int(b[0])+w, int(b[1])), (0, 0, 255), -1)
            cv2.putText(frame_drawn, label, (int(b[0]), int(b[1])-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Log new hazards
            if hid not in vlm_requested:
                # We save a crop the first time it becomes active
                crop_path = os.path.join(hazards_dir, f"hazard_{hid}.jpg")
                x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(frame.shape[1], int(b[2])), min(frame.shape[0], int(b[3]))
                hazard_crop = frame[y1:y2, x1:x2]
                
                if hazard_crop.size > 0:
                     cv2.imwrite(crop_path, hazard_crop)
                     
                     # Add entry to Flight Report
                     flight_report.append({
                         "hazard_id": hid,
                         "class_name": phase1_model.names[c],
                         "frame_idx": frame_idx,
                         "latitude": current_lat,
                         "longitude": current_lon,
                         "image_path": crop_path,
                         "vlm_analysis": "Pending Phase 3 (Cross-Gate) Verification..."
                     })

        # Update Phase 2 Timer
        if len(active_hazards) > 0:
            phase2_active_timer = phase2_cooldown_max
            
        current_mode = "SCOUT (Phase 1)"
        
        # --- Phase 2: Navigator ---
        if phase2_active_timer > 0:
            current_mode = f"NAVIGATOR (Phase 2) - Cooldown: {phase2_active_timer}"
            phase2_active_timer -= 1
            
            # Run segmentation on the raw frame (not the drawn one)
            results_seg = phase2_model.predict(frame, device=device, verbose=False)
            
            # --- Phase 3: Commander (Cross-Phase Proximity Gate) ---
            if results_seg[0].masks is not None:
                masks = results_seg[0].masks.data.cpu().numpy()
                if masks.shape[0] > 0:
                    # Combine all segmentation masks into one binary image
                    combined_mask = np.any(masks, axis=0).astype(np.uint8)
                    combined_mask_resized = cv2.resize(combined_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    
                    for hazard in active_hazards:
                        hid = hazard['id']
                        if hid in vlm_requested:
                            continue # Already asked VLM about this specific hazard
                            
                        b = hazard['box']
                        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        
                        # Ensure bounding box is within frame boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_width, x2), min(frame_height, y2)
                        
                        # Check spatial overlap between YOLO-Detect bounding box and YOLO-Seg mask
                        overlap = np.sum(combined_mask_resized[y1:y2, x1:x2])
                        
                        if overlap > 0:
                            print(f"\n--- CROSS-PHASE GATE TRIGGERED ---")
                            print(f"Hazard {hid} overlaps with Segmentation Mask.")
                            vlm_requested.add(hid)
                            
                            # Crop the exact hazard from the raw frame
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                context_str = f"Hazard {hid} ({phase1_model.names[hazard['class_id']]})"
                                threading.Thread(target=query_vlm_async, args=(args.vlm_url, hid, crop.copy(), context_str)).start()
            
            # Overlay masks onto our already drawn frame without the seg bounding boxes
            frame_drawn = results_seg[0].plot(img=frame_drawn, labels=False, boxes=False)
            
        # Draw VLM Text on HUD & Update Flight Report
        with vlm_lock:
            for hazard in active_hazards:
                hid = hazard['id']
                if hid in vlm_responses:
                    b = hazard['box']
                    vlm_text = vlm_responses[hid]['advice']
                    
                    # Update JSON Flight Report
                    for entry in flight_report:
                        if entry["hazard_id"] == hid:
                            entry["vlm_analysis"] = vlm_text
                    
                    # Split long VLM text into multiple lines for readability
                    words = vlm_text.split()
                    lines = []
                    current_line = []
                    for word in words:
                        current_line.append(word)
                        if len(" ".join(current_line)) > 40: # Max ~40 chars per line
                            lines.append(" ".join(current_line))
                            current_line = []
                    if current_line:
                        lines.append(" ".join(current_line))
                        
                    y_offset = max(40, int(b[1]) - 50)
                    for i, line in enumerate(lines):
                        (w, h), _ = cv2.getTextSize(f"VLM: {line}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame_drawn, (int(b[0]), y_offset - h - 5), (int(b[0]) + w, y_offset + 5), (0, 0, 0), -1)
                        cv2.putText(frame_drawn, f"VLM: {line}", (int(b[0]), y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
                        y_offset += h + 10
                    
        # --- HUD Elements ---
        # Add a diagnostic overlay at the top left with black background for readability
        hud_text = f"Frame: {frame_idx}/{total_frames} | Mode: {current_mode} | Active Hazards: {len(active_hazards)}"
        (w, h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_drawn, (10, 10), (15 + w, 25 + h), (0, 0, 0), -1)
        cv2.putText(frame_drawn, hud_text, (15, 20 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(window_name, frame_drawn)
        
        # Wait for key press (1ms delay allows OpenCV to draw, adjust delay to match video FPS if needed for playback)
        # We use a 1ms delay to process as fast as possible for testing
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Processing manually aborted.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Video Stream Complete.")
    
    # Save the JSON Flight Report
    report_path = os.path.join(os.path.dirname(__file__), "Flight_Report.json")
    with open(report_path, 'w') as f:
        json.dump(flight_report, f, indent=4)
    print(f"[{len(flight_report)} Hazards Logged] Saved Flight Report to: {report_path}")
    
    # Trigger Map Generation
    print("Launching Folium Interactive Hazard Map...")
    map_script = os.path.join(os.path.dirname(__file__), "generate_map.py")
    subprocess.run([sys.executable, map_script, "--report", report_path])
    
    print("Orchestrator finished.")

if __name__ == "__main__":
    main()
