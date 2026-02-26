import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area

class HazardTracker:
    def __init__(self, iou_threshold=0.5, debounce_frames=5):
        """
        Tracks bounding boxes across frames to apply temporal debouncing
        and spatial IoU filtering.
        """
        self.iou_threshold = iou_threshold
        self.debounce_frames = debounce_frames
        
        # Dictionary of tracked hazards. 
        # Key: unique ID, Value: dict containing box, class, frames_seen, last_seen
        self.tracked_hazards = {}
        self.next_id = 0
        self.current_frame = 0

    def update(self, detected_boxes, class_ids):
        """
        Update the tracker with new detections from the current frame.
        detected_boxes: list of [x1, y1, x2, y2]
        class_ids: list of class IDs corresponding to the boxes
        Returns: list of "active" hazards that have passed the debounce threshold
        """
        self.current_frame += 1
        active_hazards = []

        # Matches between new detections and existing tracks
        matched_new_indices = set()

        # Iterate over existing tracks
        for track_id, track_data in list(self.tracked_hazards.items()):
            best_iou = 0
            best_match_idx = -1

            for i, box in enumerate(detected_boxes):
                if i in matched_new_indices:
                    continue
                # Only compare same class to prevent morphing objects
                if track_data['class_id'] != class_ids[i]:
                    continue

                iou = calculate_iou(track_data['box'], box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_iou >= self.iou_threshold:
                # Update existing track
                self.tracked_hazards[track_id]['box'] = detected_boxes[best_match_idx]
                self.tracked_hazards[track_id]['frames_seen'] += 1
                self.tracked_hazards[track_id]['last_seen'] = self.current_frame
                matched_new_indices.add(best_match_idx)
            else:
                # Track lost. Allow a small grace period before deleting.
                if self.current_frame - track_data['last_seen'] > 3: 
                     del self.tracked_hazards[track_id]

        # Add unmatched new detections as new tracks
        for i, box in enumerate(detected_boxes):
            if i not in matched_new_indices:
                self.tracked_hazards[self.next_id] = {
                    'box': box,
                    'class_id': class_ids[i],
                    'frames_seen': 1,
                    'last_seen': self.current_frame
                }
                self.next_id += 1

        # Check which hazards are "active" (passed debounce threshold)
        for track_id, track_data in self.tracked_hazards.items():
            if track_data['frames_seen'] >= self.debounce_frames:
                active_hazards.append({
                    'id': track_id,
                    'box': track_data['box'],
                    'class_id': track_data['class_id']
                })

        return active_hazards
