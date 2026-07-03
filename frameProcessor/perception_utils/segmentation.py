import cv2
import numpy as np
import torch
from PIL import Image


CITYSCAPES_CLASS_COLORS_BGR = {
    0: (255, 0, 0),  # road
    1: (232, 35, 244),  # sidewalk
    2: (70, 70, 70),  # building
    3: (156, 102, 102),  # wall
    4: (153, 153, 190),  # fence
    5: (153, 153, 153),  # pole
    6: (30, 170, 250),  # traffic light
    7: (0, 220, 220),  # traffic sign
    8: (35, 142, 107),  # vegetation
    9: (152, 251, 152),  # terrain
    10: (180, 130, 70),  # sky
    11: (60, 20, 220),  # person
    12: (0, 0, 255),  # rider
    13: (142, 0, 0),  # car
    14: (70, 0, 0),  # truck
    15: (100, 60, 0),  # bus
    16: (100, 80, 0),  # train
    17: (230, 0, 0),  # motorcycle
    18: (32, 11, 119),  # bicycle
}


def segment_semantic_classes(
    processor,
    model,
    image,
    class_ids=[0],
    class_colors_bgr=None,
):
    """
    Run SegFormer semantic segmentation and return a colored semantic mask.

    Args:
        processor: Loaded Hugging Face SegFormer image processor.
        model: Loaded Hugging Face SegFormer model.
        image: OpenCV BGR image.
        class_ids: Optional list of class IDs to show.
            If None, all known classes are shown.
            Example: [0, 1, 13] for road, sidewalk, car.
        class_colors_bgr: Optional dict mapping class IDs to BGR colors.

    Returns:
        3-channel OpenCV BGR image where selected semantic classes have color.
    """

    if class_colors_bgr is None:
        class_colors_bgr = CITYSCAPES_CLASS_COLORS_BGR

    if class_ids is None:
        class_ids = list(class_colors_bgr.keys())

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
        align_corners=False,
    )

    predicted_class_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    semantic_mask = np.zeros_like(image)

    for class_id in class_ids:
        if class_id not in class_colors_bgr:
            continue

        color = class_colors_bgr[class_id]
        semantic_mask[predicted_class_map == class_id] = color

    return semantic_mask


def segment_roads(processor, model, image, road_class_id=0):
    """
    Run SegFormer semantic segmentation and return only the road mask.

    Args:
        processor: Loaded Hugging Face SegFormer image processor.
        model: Loaded Hugging Face SegFormer model.
        image: OpenCV BGR image.
        road_class_id: Class id to treat as road. Cityscapes road is 0.

    Returns:
        3-channel BGR binary mask where road pixels are white.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
        align_corners=False,
    )

    predicted_class_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    road_mask = np.zeros_like(predicted_class_map, dtype=np.uint8)
    road_mask[predicted_class_map == road_class_id] = 255

    return cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)


def segment_drivable_area_and_lanes_twinlitenet(
    model,
    image,
    device="cpu",
    input_size=(640, 360),
):
    """
    Run TwinLiteNet+ and return drivable-area and lane-line masks.

    Args:
        model: Loaded TwinLiteNet+ model.
        image: OpenCV BGR image.
        device: "cpu" or "cuda".
        input_size: Model input size as (width, height).

    Returns:
        Tuple of 3-channel BGR binary masks:
            drivable_mask_bgr, lane_mask_bgr
    """
    original_height, original_width = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(image_rgb, input_size)
    image_float = resized_rgb.astype(np.float32) / 255.0
    image_chw = image_float.transpose(2, 0, 1)
    input_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)

    drivable_output = outputs[0][0]
    lane_output = outputs[1][0]

    def output_to_binary_mask(output):
        if len(output.shape) == 3 and output.shape[0] > 1:
            class_map = torch.argmax(output, dim=0)
            mask = class_map.cpu().numpy().astype(np.uint8)
            mask[mask > 0] = 255
        elif len(output.shape) == 3 and output.shape[0] == 1:
            probability = torch.sigmoid(output[0])
            mask = (probability > 0.5).cpu().numpy().astype(np.uint8) * 255
        else:
            probability = torch.sigmoid(output)
            mask = (probability > 0.5).cpu().numpy().astype(np.uint8) * 255
        return mask

    drivable_mask = output_to_binary_mask(drivable_output)
    lane_mask = output_to_binary_mask(lane_output)

    drivable_mask = cv2.resize(
        drivable_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    lane_mask = cv2.resize(
        lane_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    drivable_mask_bgr = cv2.cvtColor(drivable_mask, cv2.COLOR_GRAY2BGR)
    lane_mask_bgr = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    return drivable_mask_bgr, lane_mask_bgr
