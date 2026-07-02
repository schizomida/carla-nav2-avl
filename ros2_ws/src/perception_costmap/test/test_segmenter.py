import numpy as np
import pytest
from perception_costmap.segmentation import (
    create_segmenter, HsvSegmenter, letterbox)


def test_factory_hsv_returns_callable_mask():
    seg = create_segmenter("hsv", max_sat=60)
    img = np.full((40, 60, 3), (90, 90, 90), np.uint8)   # gray = "asphalt"
    mask = seg(img)
    assert mask.dtype == bool and mask.shape == (40, 60)
    assert mask.any()


def test_factory_unknown_method():
    with pytest.raises(ValueError):
        create_segmenter("segnet9000")


def test_letterbox_geometry():
    img = np.zeros((100, 200, 3), np.uint8)
    padded, ratio, (pl, pt) = letterbox(img, new_size=640)
    assert padded.shape[:2] == (640, 640)
    assert abs(ratio - 3.2) < 1e-6        # 640/200
    assert pl == 0 and pt == 160          # 100*3.2=320 tall -> 160 pad top
