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

def query_vlm_async(vlm_url, hazard_id, image_crop, context, drone_params):
    """Sends a Base64 encoded image crop to the VLM server asynchronously."""
    print(f"[{hazard_id}] Sending to VLM for flight vector reasoning...")
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
                # Resume flight with new vectors
                drone_params["state"] = "FLYING"
                drone_params["action"] = data.get("action", "RESUME_PATH")
                drone_params["vector"] = f"X:{data.get('vector_x', 0)}, Y:{data.get('vector_y', 0)}, Alt:{data.get('altitude_adjustment', 0)}"
            print(f"[{hazard_id}] VLM Flight Command Received: {drone_params['action']}")
        else:
            print(f"[{hazard_id}] VLM Server Error: {response.status_code}")
            drone_params["state"] = "FLYING" # Auto-resume on error
    except Exception as e:
        print(f"[{hazard_id}] VLM Request Failed: {e}")
        drone_params["state"] = "FLYING" # Auto-resume on error

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

    # Phase 2 Trigger Configuration
    phase2_cooldown_max = 150 # Run Phase 2 for X frames after hazard detection
    phase2_active_timer = 0
    
    # Mathematical Depth Simulator Thresholds
    # A box taking up > 30% of the screen is an imminent collision risk
    collision_depth_threshold = 0.30 
    # A box taking up > 5% but < 30% is a distant structural danger worth inspecting
    inspection_depth_threshold = 0.05
    
    # Drone Agent State
    drone_params = {"state": "FLYING", "action": "WAYPOINT_FOLLOW", "vector": "N/A"}

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
        if drone_params["state"] == "FLYING":
            ret, current_raw_frame = cap.read()
            if not ret:
                print("End of video stream.")
                break
                
            # Scale frame to a maximum of 1280x720 while maintaining aspect ratio
            target_w, target_h = 1280, 720
            h, w = current_raw_frame.shape[:2]
            
            # Calculate aspect ratios
            aspect = w / h
            if w > target_w or h > target_h:
                if target_w / aspect <= target_h:
                    new_w = target_w
                    new_h = int(new_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(new_h * aspect)
                frame = cv2.resize(current_raw_frame, (new_w, new_h))
            else:
                frame = current_raw_frame.copy()
                
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
                        
                        # Calculate Mathematical Depth (Box Area vs Screen Area)
                        box_area = (x2 - x1) * (y2 - y1)
                        screen_area = frame_width * frame_height
                        depth_ratio = box_area / screen_area
                        
                        # Check spatial overlap between YOLO-Detect bounding box and YOLO-Seg mask
                        overlap = np.sum(combined_mask_resized[y1:y2, x1:x2])
                        
                        if overlap > 0:
                            vlm_prompt_context = None
                            hover_reason = None
                            
                            # Depth Triage System
                            if depth_ratio >= collision_depth_threshold:
                                hover_reason = "CRITICAL: Imminent Collision with Hazard"
                                vlm_prompt_context = f"Hazard {hid} ({phase1_model.names[hazard['class_id']]}) is physically blocking the drone's immediate flight path. Generate an emergency evasion vector."
                            elif depth_ratio >= inspection_depth_threshold:
                                hover_reason = "ASSESSMENT: Inspecting Distant Structural Integrity"
                                vlm_prompt_context = f"Hazard {hid} ({phase1_model.names[hazard['class_id']]}) located at mid-range. Should the drone alter its path to inspect closer, or continue its current search grid?"
                            else:
                                # Too far away, just log it but don't stop the drone
                                pass 
                            
                            if hover_reason is not None:
                                print(f"\n--- FLIGHT INTERRUPT ENGAGED ---")
                                print(f"Reason: {hover_reason} (Depth Ratio: {depth_ratio:.2f})")
                                vlm_requested.add(hid)
                                
                                # Lock the drone flight
                                drone_params["state"] = "HOVERING"
                                drone_params["action"] = hover_reason
                                
                                # Crop the exact hazard from the raw frame
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    threading.Thread(target=query_vlm_async, args=(args.vlm_url, hid, crop.copy(), vlm_prompt_context, drone_params)).start()
            
            # Overlay masks onto our already drawn frame without the seg bounding boxes
            frame_drawn = results_seg[0].plot(img=frame_drawn, labels=False, boxes=False)
            
        # Draw VLM Text on HUD & Update Flight Report
        with vlm_lock:
            for hazard in active_hazards:
                hid = hazard['id']
                if hid in vlm_responses:
                    b = hazard['box']
                    vlm_data = vlm_responses[hid]
                    
                    vlm_reasoning = vlm_data.get('reasoning', vlm_data.get('advice', ''))
                    vlm_action = vlm_data.get('action', '')
                    vlm_vx = vlm_data.get('vector_x', '')
                    vlm_vy = vlm_data.get('vector_y', '')
                    vlm_alt = vlm_data.get('altitude_adjustment', '')
                    
                    vlm_text = f"CMD: {vlm_action} [X:{vlm_vx}, Y:{vlm_vy}, ALT:{vlm_alt}] | Reason: {vlm_reasoning}"
                    
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
                        if len(" ".join(current_line)) > 55: # Max ~55 chars per line
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
        hud_text = f"Frame: {frame_idx}/{total_frames} | Mode: {current_mode} | Hazards: {len(active_hazards)}"
        hud_state = f"Drone: {drone_params['state']} | Action: {drone_params['action']} | Vector: {drone_params['vector']}"
        
        (w1, h1), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (w2, h2), _ = cv2.getTextSize(hud_state, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        w_max = max(w1, w2)
        
        cv2.rectangle(frame_drawn, (10, 10), (20 + w_max, 25 + h1 + h2 + 10), (0, 0, 0), -1)
        cv2.putText(frame_drawn, hud_text, (15, 20 + h1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        state_color = (0, 255, 255) if drone_params['state'] == 'HOVERING' else (255, 150, 0)
        cv2.putText(frame_drawn, hud_state, (15, 25 + h1 * 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
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
