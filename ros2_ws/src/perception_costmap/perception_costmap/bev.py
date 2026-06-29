"""
bev.py
------
Inverse-perspective mapping (IPM): warp a forward camera image (or a mask
derived from it) into the top-down metric costmap grid defined by a GridSpec.

Two ways to get the homography:

1. ``homography_from_points`` -- give 4 points in the image and their known
   ground positions (metres, robot frame). Works for any camera; calibrate
   once by clicking 4 ground points. This is the robust, recommended path.

2. ``homography_from_camera`` -- compute it analytically from camera
   intrinsics K + mounting (height, pitch). Convenient in CARLA where these
   are known exactly. Verify against a real frame before trusting it.

The homography maps image pixels -> grid cell coords (col, row), so the warped
output lines up cell-for-cell with the OccupancyGrid.
"""

import numpy as np
import cv2

from .occupancy import GridSpec


def _world_to_grid_affine(grid: GridSpec) -> np.ndarray:
    """3x3 affine mapping world ground (x, y, 1) -> grid (col, row, 1)."""
    s = 1.0 / grid.resolution
    return np.array([
        [s, 0.0, -grid.x_min * s],
        [0.0, s, -grid.y_min * s],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def homography_from_points(image_pts, world_pts, grid: GridSpec) -> np.ndarray:
    """
    image_pts: 4x2 pixel coords (u, v).
    world_pts: 4x2 ground coords (x forward, y left) in metres.
    Returns 3x3 homography mapping image pixels -> grid (col, row).
    """
    image_pts = np.asarray(image_pts, dtype=np.float32).reshape(4, 2)
    world_pts = np.asarray(world_pts, dtype=np.float64).reshape(4, 2)

    A = _world_to_grid_affine(grid)
    grid_pts = []
    for (x, y) in world_pts:
        v = A @ np.array([x, y, 1.0])
        grid_pts.append([v[0], v[1]])
    grid_pts = np.asarray(grid_pts, dtype=np.float32)

    return cv2.getPerspectiveTransform(image_pts, grid_pts)


def homography_from_camera(K, cam_height, pitch_deg, grid: GridSpec) -> np.ndarray:
    """
    Analytic ground-plane homography from intrinsics + mounting.

    K: 3x3 camera intrinsics. cam_height: metres above ground.
    pitch_deg: downward tilt of the optical axis (0 = looking at the horizon).

    World/robot frame: x forward, y left, z up. OpenCV camera frame: x right,
    y down, z forward. NOTE: verify the sign/þitch convention against your
    actual camera_info before relying on this in the field.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    theta = np.radians(pitch_deg)

    # Base world->camera rotation (no pitch): cam looks along +x world.
    R0 = np.array([
        [0.0, -1.0, 0.0],   # cam x (right) = -world y
        [0.0, 0.0, -1.0],   # cam y (down)  = -world z
        [1.0, 0.0, 0.0],    # cam z (fwd)   =  world x
    ], dtype=np.float64)
    # Pitch down about the camera's x (right) axis.
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ], dtype=np.float64)
    R = Rx @ R0

    C = np.array([0.0, 0.0, cam_height])      # camera centre in world
    t = -R @ C

    # Ground plane z=0: world->image homography uses columns r1, r2, t.
    H_world2img = K @ np.column_stack((R[:, 0], R[:, 1], t))
    H_img2world = np.linalg.inv(H_world2img)

    A = _world_to_grid_affine(grid)
    return A @ H_img2world


def warp_to_bev(image, H, grid: GridSpec, interp=cv2.INTER_NEAREST):
    """Warp an image/mask from camera view to the BEV grid (height, width)."""
    return cv2.warpPerspective(image, H, (grid.width, grid.height), flags=interp)


def bev_known_mask(H, image_shape, grid: GridSpec) -> np.ndarray:
    """
    The set of grid cells the camera actually sees (its ground footprint),
    found by warping an all-ones image. Used as the ``known_mask`` so cells
    outside the camera's view stay UNKNOWN instead of being marked off-road.
    """
    h, w = image_shape[:2]
    ones = np.full((h, w), 255, dtype=np.uint8)
    warped = warp_to_bev(ones, H, grid)
    return warped > 0
