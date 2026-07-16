"""
Autonomous Vehicle Perception Demo
- Loads an image
- Detects and segments objects using YOLOv8
- Extracts object data points
- Projects detections into a simple bird's-eye-view map

Install:
    pip install ultralytics opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMAGE_PATH = "road_scene.jpg"
MODEL_PATH = "yolov8n-seg.pt"

src_points = np.float32([
    [580, 460],
    [700, 460],
    [1120, 720],
    [200, 720],
])

dst_points = np.float32([
    [300, 0],
    [500, 0],
    [500, 800],
    [300, 800],
])

BEV_WIDTH = 800
BEV_HEIGHT = 800

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image

def get_homography():
    return cv2.getPerspectiveTransform(src_points, dst_points)

def project_point_to_bev(point, homography):
    point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point_array, homography)
    return projected[0][0]

def create_bev_map(objects):
    bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    for obj in objects:
        x, y = map(int, obj["bev_point"])
        if 0 <= x < BEV_WIDTH and 0 <= y < BEV_HEIGHT:
            cv2.circle(bev, (x, y), 8, (255,255,255), -1)
            cv2.putText(bev, obj["class_name"], (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return bev

def process_image(image_path):
    model = YOLO(MODEL_PATH)
    image = load_image(image_path)
    H = get_homography()
    results = model(image)[0]
    annotated = image.copy()
    objects = []

    if results.masks is None:
        return image, np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8), []

    masks = results.masks.xy
    boxes = results.boxes

    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        ground = (int((x1+x2)/2), int(y2))
        bev = project_point_to_bev(ground, H)
        poly = masks[i].astype(np.int32)

        cv2.polylines(annotated,[poly],True,(0,255,0),2)
        cv2.rectangle(annotated,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        cv2.circle(annotated,ground,5,(0,0,255),-1)
        cv2.putText(annotated,f"{name} {conf:.2f}",(int(x1),int(y1)-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        objects.append({
            "class_id": cls,
            "class_name": name,
            "confidence": conf,
            "bbox": [float(x1),float(y1),float(x2),float(y2)],
            "ground_point_image": ground,
            "bev_point": bev.tolist(),
            "mask_polygon": poly.tolist()
        })

    return annotated, create_bev_map(objects), objects

if __name__ == "__main__":
    annotated, bev_map, objects = process_image(IMAGE_PATH)
    print(objects)
    cv2.imwrite("segmented_output.jpg", annotated)
    cv2.imwrite("bev_map.jpg", bev_map)

    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.show()
