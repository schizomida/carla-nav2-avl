import cv2
import numpy as np


def map_masks_and_boxes_to_image(
    image,
    masks=None,
    detections=None,
    mask_alpha=0.4,
    draw_class_names=True,
    mask_colors=None,
):
    """
    Overlay masks and bounding boxes onto an image.

    Args:
        image: Original OpenCV BGR image.
        masks: One mask or list of masks. Each mask can be single-channel
            binary or 3-channel BGR.
        detections: List of dictionaries from object_detection().
        mask_alpha: Transparency of mask overlays.
        draw_class_names: Whether to draw class labels above boxes.
        mask_colors: Optional list of BGR colors, one per mask.

    Returns:
        OpenCV BGR image with masks and bounding boxes drawn.
    """
    output_image = image.copy()

    if masks is None:
        masks = []

    if detections is None:
        detections = []

    if not isinstance(masks, list):
        masks = [masks]

    if mask_colors is None:
        mask_colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

    for mask_index, mask in enumerate(masks):
        if mask is None:
            continue

        mask = mask.astype(np.uint8)

        # 3-channel BGR mask, for example a colored semantic mask.
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            binary_mask = mask_gray > 0
            blended_image = cv2.addWeighted(output_image, 1.0, mask, mask_alpha, 0)
            output_image[binary_mask] = blended_image[binary_mask]
            continue

        # Single-channel binary mask.
        color = mask_colors[mask_index % len(mask_colors)]
        binary_mask = mask > 0
        colored_mask = np.zeros_like(output_image)
        colored_mask[binary_mask] = color
        blended_image = cv2.addWeighted(output_image, 1.0, colored_mask, mask_alpha, 0)
        output_image[binary_mask] = blended_image[binary_mask]

    for detection in detections:
        box = detection["box"]
        class_name = detection.get("class_name", "object")
        confidence = detection.get("confidence", 0.0)

        x1, y1, x2, y2 = [int(value) for value in box]
        box_color = (255, 255, 255)

        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)

        if draw_class_names:
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                output_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
                cv2.LINE_AA,
            )

    return output_image
