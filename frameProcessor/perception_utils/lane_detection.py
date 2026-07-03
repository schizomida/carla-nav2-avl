import cv2
import numpy as np


def segment_lanes_ufldv2(lane_model, image, thickness=6, return_points=False):
    """
    Run UFLDv2 and return a lane mask.

    Args:
        lane_model: Loaded UFLDv2LaneDetector or any object with predict(image).
        image: OpenCV BGR image.
        thickness: Thickness of lane lines drawn into the mask.
        return_points: If True, also return raw lane point coordinates.

    Returns:
        If return_points is False:
            lane_mask_bgr
        If return_points is True:
            lane_mask_bgr, lane_points
    """
    lane_points = lane_model.predict(image)

    lane_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for lane in lane_points:
        if len(lane) < 2:
            continue

        sorted_points = sorted(lane, key=lambda point: (point[1], point[0]))
        points_array = np.array(sorted_points, dtype=np.int32)

        cv2.polylines(
            lane_mask,
            [points_array],
            isClosed=False,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    lane_mask_bgr = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    if return_points:
        return lane_mask_bgr, lane_points

    return lane_mask_bgr
