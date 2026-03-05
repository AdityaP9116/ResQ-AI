"""Thermal camera processor for ResQ-AI.

Analyses thermal / infrared frames to identify heat hotspots that
indicate fire regions.  Complements YOLO visual fire detection with
thermal confirmation.

Isaac Sim 5.1 compatibility: NO f-strings.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pixel-intensity threshold (0-255) above which a pixel is considered "hot"
DEFAULT_HOT_THRESHOLD = 200

# Minimum blob area (in pixels) to count as a valid hotspot
MIN_BLOB_AREA = 50

# Temperature estimation range mapped from intensity [threshold..255]
TEMP_MIN_ESTIMATE = 100.0   # degrees C at threshold
TEMP_MAX_ESTIMATE = 1200.0  # degrees C at 255


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ThermalProcessor(object):
    """Identify hot regions in a thermal / IR camera frame.

    Parameters
    ----------
    hot_threshold : int
        Pixel intensity in [0, 255] above which a pixel is "hot".
    min_blob_area : int
        Minimum contiguous pixel count to be a valid hotspot.
    """

    def __init__(self, hot_threshold=DEFAULT_HOT_THRESHOLD,
                 min_blob_area=MIN_BLOB_AREA):
        self._threshold = hot_threshold
        self._min_area = min_blob_area

    def process(self, thermal_frame):
        """Detect hotspots in a single-channel thermal image.

        Parameters
        ----------
        thermal_frame : np.ndarray
            ``(H, W)`` uint8 grayscale thermal image.

        Returns
        -------
        list[dict]
            Each dict:
              center              – [cx, cy] pixel coords
              radius              – float (approximate)
              temperature_estimate – float (degrees C)
        """
        if thermal_frame is None:
            return []

        # Ensure 2-D
        if thermal_frame.ndim == 3:
            thermal_frame = thermal_frame[:, :, 0]

        # Threshold
        hot_mask = (thermal_frame >= self._threshold).astype(np.uint8)

        # Connected-component labelling (4-connectivity, pure numpy)
        labels = _label_components(hot_mask)
        if labels is None:
            return []

        max_label = int(labels.max())
        hotspots = []
        label_id = 1
        while label_id <= max_label:
            ys, xs = np.where(labels == label_id)
            area = len(ys)
            if area >= self._min_area:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                radius = math.sqrt(float(area) / math.pi)

                # Mean intensity in the blob -> temperature estimate
                blob_intensities = thermal_frame[ys, xs]
                mean_intensity = float(np.mean(blob_intensities))
                temp_est = _intensity_to_temp(mean_intensity, self._threshold)

                hotspots.append({
                    "center": [cx, cy],
                    "radius": radius,
                    "temperature_estimate": temp_est,
                })
            label_id = label_id + 1

        return hotspots

    @staticmethod
    def check_overlap(bbox, hotspots):
        """Check if a bounding box overlaps any thermal hotspot.

        Parameters
        ----------
        bbox : list
            ``[x1, y1, x2, y2]`` bounding box.
        hotspots : list[dict]
            Output from ``process()``.

        Returns
        -------
        float or None
            Temperature estimate of the strongest overlapping hotspot,
            or ``None`` if no overlap.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad_x = w * 0.2
        pad_y = h * 0.2
        ex1 = x1 - pad_x
        ey1 = y1 - pad_y
        ex2 = x2 + pad_x
        ey2 = y2 + pad_y

        best = None
        idx = 0
        while idx < len(hotspots):
            hs = hotspots[idx]
            hcx, hcy = hs["center"]
            if ex1 <= hcx <= ex2 and ey1 <= hcy <= ey2:
                t = hs.get("temperature_estimate", 0.0)
                if best is None or t > best:
                    best = t
            idx = idx + 1
        return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intensity_to_temp(intensity, threshold):
    """Map a pixel intensity in [threshold, 255] to an estimated temperature."""
    if intensity <= threshold:
        return TEMP_MIN_ESTIMATE
    ratio = (intensity - threshold) / max(255.0 - threshold, 1.0)
    return TEMP_MIN_ESTIMATE + ratio * (TEMP_MAX_ESTIMATE - TEMP_MIN_ESTIMATE)


def _label_components(binary_mask):
    """Simple 4-connected component labelling using numpy (no OpenCV needed).

    Parameters
    ----------
    binary_mask : np.ndarray
        ``(H, W)`` uint8 where 1 = foreground.

    Returns
    -------
    np.ndarray or None
        ``(H, W)`` int32 label image, or None if no foreground.
    """
    if binary_mask.max() == 0:
        return None

    try:
        # Prefer OpenCV if available (faster)
        import cv2
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4)
        return labels
    except ImportError:
        pass

    # Fallback: naive BFS labelling
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    visited = set()

    y = 0
    while y < h:
        x = 0
        while x < w:
            if binary_mask[y, x] == 1 and (y, x) not in visited:
                current_label = current_label + 1
                # BFS
                queue = [(y, x)]
                visited.add((y, x))
                qi = 0
                while qi < len(queue):
                    cy, cx = queue[qi]
                    labels[cy, cx] = current_label
                    # 4-connectivity neighbours
                    neighbours = []
                    if cy > 0:
                        neighbours.append((cy - 1, cx))
                    if cy < h - 1:
                        neighbours.append((cy + 1, cx))
                    if cx > 0:
                        neighbours.append((cy, cx - 1))
                    if cx < w - 1:
                        neighbours.append((cy, cx + 1))
                    ni = 0
                    while ni < len(neighbours):
                        ny, nx = neighbours[ni]
                        if binary_mask[ny, nx] == 1 and (ny, nx) not in visited:
                            visited.add((ny, nx))
                            queue.append((ny, nx))
                        ni = ni + 1
                    qi = qi + 1
            x = x + 1
        y = y + 1

    return labels
