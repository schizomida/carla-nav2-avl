"""
segmentation.py
---------------
Drivable-road segmentation in image space. Returns a boolean mask where True
means "this pixel is road".

The classical HSV method is factored from Adam Castillo's
``perception/costmap.py`` (HSV threshold for low-saturation asphalt + keep the
largest connected blob). A learned segmenter (TwinLiteNet+) can be dropped in
behind the same ``segment_road`` interface without touching callers.
"""

import numpy as np
import cv2


def segment_road_hsv(img_bgr,
                     max_sat: int = 60,
                     val_lo: int = 40,
                     val_hi: int = 200) -> np.ndarray:
    """
    Classical road mask: asphalt is low-saturation, mid-brightness. Threshold
    in HSV, clean up, then keep only the largest connected blob (the road).
    Returns a boolean mask the size of the image.

    HSV ranges are lighting-dependent -- recalibrate for the real camera, or
    switch to the learned segmenter.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s < max_sat) & (v > val_lo) & (v < val_hi)).astype(np.uint8) * 255

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask > 0


def segment_road(img_bgr, method: str = "hsv", **kw) -> np.ndarray:
    """
    Dispatch to a road-segmentation backend. 'hsv' is the classical default;
    'twinlitenet' is reserved for the learned model (not wired in yet).
    """
    if method == "hsv":
        return segment_road_hsv(img_bgr, **kw)
    if method == "twinlitenet":
        raise NotImplementedError(
            "learned segmentation not wired in yet; see perception/twinLiteNetTest.py")
    raise ValueError(f"unknown segmentation method: {method!r}")
