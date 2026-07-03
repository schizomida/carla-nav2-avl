# =========================================================================================
# ====== This is a Test Program Showing the models running from laptop front camera =======
# =========================================================================================

import cv2

from ultralytics import YOLO

from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation

# to measure time to process frames
import time

from perception_utils import (
    resize_image,
    segment_roads,
    segment_semantic_classes,
    segment_lanes_ufldv2,
    object_detection,
    map_masks_and_boxes_to_image,
    load_lane_model,
)

# make the processing faster but with a cost in accuracy
PROCESS_EVERY_N_FRAMES = 1
RESIZE_WIDTH = 512
VIDEO_SOURCE = "videos/roadVideo.mp4" #you could easily change this to some other input like some camera

# --------------------------------------------------
# Load models ONCE before the loop
# --------------------------------------------------

yolo_model = YOLO("./models/yolo/yolov8n.pt")

road_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)
road_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
)
# road_model.eval()
# road_processor = None
# road_model = None

lane_model = load_lane_model()
print("All models loaded.")


def process_frame(frame):
    """
    Runs the perception models and returns a visualized image.
    """

    # Resize once so masks, boxes, and display all line up.
    image_resized = resize_image(frame, width=RESIZE_WIDTH)

    masks = []
    detections = []

    # -----------------------------
    # Road segmentation: SegFormer
    # -----------------------------
    if road_processor is not None and road_model is not None:
        road_mask = segment_roads(
            road_processor,
            road_model,
            image_resized,
        )

        masks.append(road_mask)

        ### testing other capabilities of the same model
        semantic_mask = segment_semantic_classes(
            road_processor,
            road_model,
            image_resized,
        )

        masks.append(semantic_mask)

    # -----------------------------
    # Lane detection: UFLDv2
    # -----------------------------
    if lane_model is not None:
        lane_mask, lane_points = segment_lanes_ufldv2(
            lane_model,
            image_resized,
            thickness=6,
            return_points=True,
        )

        masks.append(lane_mask)

        cv2.putText(
            image_resized,
            f"Lanes: {len(lane_points)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # -----------------------------
    # Object detection: YOLO
    # -----------------------------
    if yolo_model is not None:
        detections = object_detection(
            image_resized,
            yolo_model,
            confidence_threshold=0.4,
        )

    # -----------------------------
    # Overlay masks and detections
    # -----------------------------
    processed = map_masks_and_boxes_to_image(
        image=image_resized,
        masks=masks,
        detections=detections,
        mask_alpha=0.4,
        draw_class_names=True,
        mask_colors=[
            (0, 200, 0),  # road: green
            (0, 0, 255),  # lanes: red
        ],
    )

    return processed


def main():

    cap = cv2.VideoCapture("VIDEO_SOURCE", cv2.CAP_V4L2)  # from laptop camera

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frame_index = 0
    last_processed = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Only run the models every N frames
        if frame_index % PROCESS_EVERY_N_FRAMES == 0 or last_processed is None:
            start = time.perf_counter()
            last_processed = process_frame(frame)
            print(f"Frame processed in {(time.perf_counter() - start) * 1000:.2f} ms")
        processed = last_processed

        raw_display = cv2.resize(
            frame,
            (processed.shape[1], processed.shape[0]),
        )

        combined = cv2.hconcat([raw_display, processed])

        cv2.imshow("Raw | Processed", combined)

        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
