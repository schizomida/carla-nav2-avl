import cv2


def resize_image(image, width):
    """
    Resize an OpenCV image while preserving its aspect ratio.

    Args:
        image: OpenCV BGR image.
        width: Desired output width in pixels.

    Returns:
        Resized OpenCV image.
    """
    scale = width / image.shape[1]
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height))
