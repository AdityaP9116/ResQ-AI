"""2-D pixel → 3-D world coordinate projection for the ResQ-AI pipeline.

When YOLO (or any 2-D detector) localises an object in the RGB / thermal
image, we need to lift the bounding-box centre into a metric 3-D world
coordinate so that Cosmos Reason 2 can plan waypoints.

The projection relies on:

1. **Depth map** — ``distance_to_image_plane`` from the Replicator depth
   annotator (the ``DepthCamera`` in ``spawn_drone.py``).
2. **Camera intrinsics** — the 3×3 matrix ``K`` (pinhole model).
3. **Camera world transform** — position + orientation of the camera in
   the world frame, supplied either as a 4×4 matrix or as
   ``(position, quaternion)`` from ``camera.get_world_pose()``.

Core function::

    world_xyz = pixel_to_3d_world(
        pixel_x, pixel_y,
        depth_map,
        camera_intrinsics,
        camera_world_transform,
    )

Batch helper::

    world_coords = batch_pixel_to_3d_world(
        detections,          # list of (pixel_x, pixel_y)
        depth_map,
        camera_intrinsics,
        camera_world_transform,
    )
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def pixel_to_3d_world(
    pixel_x: float,
    pixel_y: float,
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_world_transform: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Project a single pixel into a 3-D world coordinate.

    Args:
        pixel_x: Horizontal pixel coordinate (column), e.g. the centre-x of a
            YOLO bounding box.  Sub-pixel values are rounded to the nearest
            integer for depth look-up.
        pixel_y: Vertical pixel coordinate (row), e.g. the centre-y of a YOLO
            bounding box.
        depth_map: A 2-D ``float`` array of shape ``(H, W)`` (or ``(H, W, 1)``)
            containing *distance-to-image-plane* in metres, as produced by the
            Replicator ``distance_to_image_plane`` annotator.
        camera_intrinsics: The 3×3 pinhole intrinsic matrix::

                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

            Obtainable from ``MonocularCamera._intrinsics`` or
            ``camera.get_intrinsics_matrix()``.
        camera_world_transform: The camera's pose in the world frame, given as
            **either**:

            * A 4×4 homogeneous camera-to-world matrix (top-left 3×3 is the
              rotation, top-right 3×1 is the translation).
            * A ``(position, quaternion)`` tuple where *position* is ``[x, y, z]``
              and *quaternion* is scalar-first ``[w, x, y, z]`` — the format
              returned by ``camera.get_world_pose(camera_axes="world")``.
              In this convention: +X forward, +Z up.

    Returns:
        A 1-D numpy array ``[X, Y, Z]`` — the 3-D point in the world frame
        (metres, ENU / Isaac Sim convention).

    Raises:
        ValueError: If the pixel is outside the depth map or the depth is
            non-positive / infinite.
    """
    depth_map = _squeeze_depth(depth_map)
    H, W = depth_map.shape

    col = int(round(pixel_x))
    row = int(round(pixel_y))
    if not (0 <= col < W and 0 <= row < H):
        raise ValueError(
            f"Pixel ({pixel_x}, {pixel_y}) → index ({col}, {row}) is outside "
            f"the depth map of size ({W}×{H})."
        )

    depth = float(depth_map[row, col])
    if depth <= 0 or not np.isfinite(depth):
        raise ValueError(
            f"Invalid depth {depth:.4f} at pixel ({col}, {row}).  "
            "The pixel may be at the clipping boundary or occluded."
        )

    K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    cam_to_world = _resolve_world_transform(camera_world_transform)

    point_cam = _backproject_pixel(pixel_x, pixel_y, depth, K)
    point_world = _camera_to_world(point_cam, cam_to_world)

    return point_world


def batch_pixel_to_3d_world(
    detections: list[tuple[float, float]],
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_world_transform: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> list[np.ndarray | None]:
    """Project a batch of YOLO bounding-box centres into world coordinates.

    This is a convenience wrapper around :func:`pixel_to_3d_world`.

    Args:
        detections: A list of ``(pixel_x, pixel_y)`` tuples — typically the
            centre coordinates of each YOLO detection.
        depth_map: See :func:`pixel_to_3d_world`.
        camera_intrinsics: See :func:`pixel_to_3d_world`.
        camera_world_transform: See :func:`pixel_to_3d_world`.

    Returns:
        A list of the same length as *detections*.  Each element is either a
        ``[X, Y, Z]`` numpy array or ``None`` if projection failed for that
        detection (e.g. invalid depth).
    """
    results: list[np.ndarray | None] = []
    for px, py in detections:
        try:
            results.append(
                pixel_to_3d_world(px, py, depth_map, camera_intrinsics, camera_world_transform)
            )
        except ValueError:
            results.append(None)
    return results


def make_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Build a 3×3 pinhole intrinsic matrix from focal lengths and principal point.

    Convenience helper so callers don't have to construct the matrix manually.

    Args:
        fx: Focal length in pixels (horizontal).
        fy: Focal length in pixels (vertical).
        cx: Principal-point x (typically ``width / 2``).
        cy: Principal-point y (typically ``height / 2``).

    Returns:
        A 3×3 float64 intrinsic matrix ``K``.
    """
    return np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def make_intrinsics_from_fov(
    width: int,
    height: int,
    hfov_deg: float = 70.0,
) -> np.ndarray:
    """Derive a 3×3 intrinsic matrix from image size and horizontal FOV.

    Matches the default calculation in Pegasus ``MonocularCamera``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        hfov_deg: Horizontal field-of-view in degrees.

    Returns:
        A 3×3 float64 intrinsic matrix ``K``.
    """
    fx = 0.5 * width / np.tan(0.5 * np.radians(hfov_deg))
    fy = fx
    cx = 0.5 * width
    cy = 0.5 * height
    return make_intrinsics(fx, fy, cx, cy)


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────

def _squeeze_depth(depth_map: np.ndarray) -> np.ndarray:
    """Ensure *depth_map* is 2-D ``(H, W)``."""
    d = np.asarray(depth_map, dtype=np.float64)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]
    if d.ndim != 2:
        raise ValueError(f"depth_map must be (H, W) or (H, W, 1), got {depth_map.shape}")
    return d


def _backproject_pixel(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project a single pixel to 3-D in the **camera** frame.

    Uses the standard pinhole inverse-projection::

        P_cam = depth * K^{-1} @ [u, v, 1]^T

    The resulting point is in the camera's own coordinate system where
    +X is right, +Y is down, +Z is the optical axis (forward).
    """
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    K_inv = np.linalg.inv(K)
    direction = K_inv @ pixel_h
    return direction * depth


def _resolve_world_transform(
    transform: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Normalise *transform* into a 4×4 camera-to-world matrix.

    Accepts either:
    * A 4×4 matrix directly.
    * A (position, quaternion_wxyz) tuple as returned by
      ``camera.get_world_pose(camera_axes="world")``.
      In the "world" convention the camera axes are: +X forward, +Z up.
      We convert from that convention to the standard vision convention
      (+X right, +Y down, +Z forward) so the intrinsics back-projection
      and the world transform compose correctly.
    """
    if isinstance(transform, (list, tuple)) and len(transform) == 2:
        pos = np.asarray(transform[0], dtype=np.float64).ravel()
        quat_wxyz = np.asarray(transform[1], dtype=np.float64).ravel()

        # scipy uses scalar-last [x, y, z, w]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        R_world_cam_world_axes = Rotation.from_quat(quat_xyzw).as_matrix()

        # Isaac Sim "world" camera axes: +X forward, +Y left, +Z up
        # Vision / intrinsics convention:  +X right,  +Y down, +Z forward
        # The columns of this matrix map vision-convention unit vectors into
        # the Isaac "world" camera-axes vectors.
        #   vision +X (right)   → Isaac -Y (left→right flip)
        #   vision +Y (down)    → Isaac -Z (up→down flip)
        #   vision +Z (forward) → Isaac +X (forward)
        axes_correction = np.array([
            [ 0.0, 0.0, 1.0],   # vision Z → Isaac X
            [-1.0, 0.0, 0.0],   # vision X → Isaac -Y
            [ 0.0,-1.0, 0.0],   # vision Y → Isaac -Z
        ], dtype=np.float64)

        R = R_world_cam_world_axes @ axes_correction

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4×4 matrix or (pos, quat) tuple, got shape {T.shape}")
    return T


def _camera_to_world(point_cam: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """Transform a point from camera frame to world frame via the 4×4 matrix."""
    p_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0], dtype=np.float64)
    p_world = cam_to_world @ p_h
    return p_world[:3]


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic test: camera at (10, 20, 30), looking straight down (-Z world)
    # In the Isaac "world" camera convention (+X fwd, +Z up), looking down
    # means the camera's +X axis points toward -Z_world.

    W, H = 640, 480
    K = make_intrinsics_from_fov(W, H, hfov_deg=70.0)
    print(f"Intrinsics K:\n{K}\n")

    # Uniform depth plane at 15 m
    depth_map = np.full((H, W), 15.0, dtype=np.float64)

    # Camera-to-world as a 4×4: identity rotation → camera +Z (optical axis)
    # points along world +Z.  We place the camera at (10, 20, 30).
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 3] = [10.0, 20.0, 30.0]

    # Project the image centre — should land 15 m in front along +Z,
    # i.e. (10, 20, 45).
    cx, cy = W / 2.0, H / 2.0
    result = pixel_to_3d_world(cx, cy, depth_map, K, cam_to_world)
    print(f"Centre pixel  → world {result}")
    assert np.allclose(result, [10.0, 20.0, 45.0], atol=0.01), f"Expected [10, 20, 45], got {result}"

    # Project a pixel to the right of centre — should shift in +X
    result_right = pixel_to_3d_world(cx + 50, cy, depth_map, K, cam_to_world)
    print(f"Right pixel   → world {result_right}")
    assert result_right[0] > 10.0, "Expected X > 10 for a pixel right of centre"

    # Project a pixel below centre — should shift in +Y
    result_below = pixel_to_3d_world(cx, cy + 50, depth_map, K, cam_to_world)
    print(f"Below pixel   → world {result_below}")
    assert result_below[1] > 20.0, "Expected Y > 20 for a pixel below centre"

    # Batch test
    dets = [(cx, cy), (cx + 50, cy), (0, 0), (-10, -10)]
    batch = batch_pixel_to_3d_world(dets, depth_map, K, cam_to_world)
    print(f"\nBatch results ({len(dets)} detections):")
    for det, res in zip(dets, batch):
        status = f"{res}" if res is not None else "FAILED (out of bounds)"
        print(f"  {det} → {status}")
    assert batch[0] is not None
    assert batch[3] is None  # out of bounds

    # Test (position, quaternion) tuple input — identity orientation
    pos = np.array([10.0, 20.0, 30.0])
    # Identity quaternion w,x,y,z = 1,0,0,0 in "world" axes means
    # camera +X → world +X (forward), +Z → world +Z (up)
    # With the axes correction applied internally, the vision convention
    # +Z (forward) maps to world +X.
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    result_tuple = pixel_to_3d_world(cx, cy, depth_map, K, (pos, quat_wxyz))
    print(f"\nTuple input (identity quat, world axes) centre → {result_tuple}")

    print("\nAll projection tests passed.")
