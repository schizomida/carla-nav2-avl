import cv2
import numpy as np


def object_detection(image, model, confidence_threshold=0.4):
    """
    Run YOLO object detection and return bounding box data only.

    Args:
        image: OpenCV BGR image.
        model: Loaded Ultralytics YOLO model.
        confidence_threshold: Minimum confidence required to keep detection.

    Returns:
        List of dictionaries containing class_id, class_name, confidence, and box.
    """
    results = model(image, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence < confidence_threshold:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )

    return detections


def segment_objects(model, image, segmentation_type="instance"):
    """
    Run YOLO segmentation and return segmentation masks.

    Args:
        model: Loaded Ultralytics YOLO segmentation model.
        image: OpenCV BGR image.
        segmentation_type: "instance" or "semantic".

    Returns:
        If instance: list of binary BGR masks.
        If semantic: one combined binary BGR mask.
    """
    results = model(image, verbose=False)

    if len(results) == 0 or results[0].masks is None:
        if segmentation_type == "instance":
            return []
        return np.zeros_like(image)

    masks = results[0].masks.data.cpu().numpy()

    if segmentation_type == "instance":
        instance_masks = []
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            instance_masks.append(mask)
        return instance_masks

    if segmentation_type == "semantic":
        semantic_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            semantic_mask = cv2.bitwise_or(semantic_mask, mask)
        return cv2.cvtColor(semantic_mask, cv2.COLOR_GRAY2BGR)

    raise ValueError("segmentation_type must be 'instance' or 'semantic'.")
