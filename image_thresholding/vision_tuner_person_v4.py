import cv2
import os
import glob
import json
import numpy as np

# ---------- SETTINGS ----------
IMAGE_FOLDER = r"C:\Perception_Mentorship_AVL\submissions\day_3"
SAVE_FILE = "saved_thresholds_v4.json"

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

DEFAULT_SETTINGS = {
    "line_h_min": 0,
    "line_h_max": 179,
    "line_s_min": 0,
    "line_s_max": 80,
    "line_v_min": 170,
    "line_v_max": 255,

    "cone_h_min": 0,
    "cone_h_max": 7,
    "cone_s_min": 73,
    "cone_s_max": 255,
    "cone_v_min": 0,
    "cone_v_max": 123,

    "min_cone_area": 478,
    "cone_cleanup": 12,

    "show_labels": 1,
    "show_cones": 1,
    "show_people": 1,
    "show_center_line": 1,
}

def load_settings():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as file:
                loaded = json.load(file)
                settings = DEFAULT_SETTINGS.copy()
                settings.update(loaded)
                print("Loaded previous settings from", SAVE_FILE)
                return settings
        except:
            print("Could not load settings. Using defaults.")

    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SAVE_FILE, "w") as file:
        json.dump(settings, file, indent=4)

    print("Saved settings to", SAVE_FILE)

def nothing(x):
    pass

def draw_text_box(image, lines, x=15, y=15):
    box_width = 260
    line_height = 28
    padding = 10
    box_height = padding * 2 + line_height * len(lines)

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + box_width, y + box_height),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    current_y = y + padding + 20

    for text, color in lines:
        cv2.putText(
            image,
            text,
            (x + padding, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        current_y += line_height

def draw_clean_label(image, title, offset_text, x, y, color):
    label_y = max(y - 10, 20)

    cv2.putText(
        image,
        title,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    cv2.putText(
        image,
        offset_text,
        (x, label_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1
    )

# ---------- LOAD SETTINGS ----------
settings = load_settings()

# ---------- LOAD IMAGES ----------
image_paths = []

for ext in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

image_paths.sort()

if len(image_paths) == 0:
    print("No images found.")
    print("Check IMAGE_FOLDER:", IMAGE_FOLDER)
    exit()

current_index = 0
mode = "line"

# ---------- PERSON DETECTOR ----------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------- WINDOWS ----------
cv2.namedWindow("Detection View", cv2.WINDOW_NORMAL)
cv2.namedWindow("Line Controls", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cone Controls", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Detection View", 1450, 800)
cv2.resizeWindow("Line Controls", 520, 350)
cv2.resizeWindow("Cone Controls", 520, 450)

cv2.moveWindow("Detection View", 560, 40)
cv2.moveWindow("Line Controls", 20, 40)
cv2.moveWindow("Cone Controls", 20, 430)

# ---------- LINE TRACKBARS ----------
cv2.createTrackbar("H Min", "Line Controls", settings["line_h_min"], 179, nothing)
cv2.createTrackbar("H Max", "Line Controls", settings["line_h_max"], 179, nothing)
cv2.createTrackbar("S Min", "Line Controls", settings["line_s_min"], 255, nothing)
cv2.createTrackbar("S Max", "Line Controls", settings["line_s_max"], 255, nothing)
cv2.createTrackbar("V Min", "Line Controls", settings["line_v_min"], 255, nothing)
cv2.createTrackbar("V Max", "Line Controls", settings["line_v_max"], 255, nothing)

# ---------- CONE TRACKBARS ----------
cv2.createTrackbar("H Min", "Cone Controls", settings["cone_h_min"], 179, nothing)
cv2.createTrackbar("H Max", "Cone Controls", settings["cone_h_max"], 179, nothing)
cv2.createTrackbar("S Min", "Cone Controls", settings["cone_s_min"], 255, nothing)
cv2.createTrackbar("S Max", "Cone Controls", settings["cone_s_max"], 255, nothing)
cv2.createTrackbar("V Min", "Cone Controls", settings["cone_v_min"], 255, nothing)
cv2.createTrackbar("V Max", "Cone Controls", settings["cone_v_max"], 255, nothing)
cv2.createTrackbar("Min Area", "Cone Controls", settings["min_cone_area"], 10000, nothing)
cv2.createTrackbar("Cleanup", "Cone Controls", settings["cone_cleanup"], 25, nothing)

print("Controls:")
print("A / Left Arrow  = previous image")
print("D / Right Arrow = next image")
print("M               = switch LINE/CONE mask")
print("L               = show/hide labels")
print("C               = show/hide cones")
print("P               = show/hide people")
print("X               = show/hide center line")
print("S               = save settings")
print("Q / ESC         = quit")

while True:
    image_path = image_paths[current_index]
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        current_index = (current_index + 1) % len(image_paths)
        continue

    max_width = 720
    original_h, original_w = image.shape[:2]

    if original_w > max_width:
        scale = max_width / original_w
        image = cv2.resize(image, (int(original_w * scale), int(original_h * scale)))

    image_h, image_w = image.shape[:2]
    image_center_x = image_w // 2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---------- READ SLIDERS ----------
    settings["line_h_min"] = cv2.getTrackbarPos("H Min", "Line Controls")
    settings["line_h_max"] = cv2.getTrackbarPos("H Max", "Line Controls")
    settings["line_s_min"] = cv2.getTrackbarPos("S Min", "Line Controls")
    settings["line_s_max"] = cv2.getTrackbarPos("S Max", "Line Controls")
    settings["line_v_min"] = cv2.getTrackbarPos("V Min", "Line Controls")
    settings["line_v_max"] = cv2.getTrackbarPos("V Max", "Line Controls")

    settings["cone_h_min"] = cv2.getTrackbarPos("H Min", "Cone Controls")
    settings["cone_h_max"] = cv2.getTrackbarPos("H Max", "Cone Controls")
    settings["cone_s_min"] = cv2.getTrackbarPos("S Min", "Cone Controls")
    settings["cone_s_max"] = cv2.getTrackbarPos("S Max", "Cone Controls")
    settings["cone_v_min"] = cv2.getTrackbarPos("V Min", "Cone Controls")
    settings["cone_v_max"] = cv2.getTrackbarPos("V Max", "Cone Controls")
    settings["min_cone_area"] = cv2.getTrackbarPos("Min Area", "Cone Controls")
    settings["cone_cleanup"] = cv2.getTrackbarPos("Cleanup", "Cone Controls")

    cleanup_size = settings["cone_cleanup"]

    if cleanup_size < 1:
        cleanup_size = 1

    if cleanup_size % 2 == 0:
        cleanup_size += 1

    # ---------- LINE MASK ----------
    line_lower = np.array([
        settings["line_h_min"],
        settings["line_s_min"],
        settings["line_v_min"]
    ])

    line_upper = np.array([
        settings["line_h_max"],
        settings["line_s_max"],
        settings["line_v_max"]
    ])

    line_mask = cv2.inRange(hsv, line_lower, line_upper)
    line_pixels = cv2.countNonZero(line_mask)

    # ---------- CONE MASK ----------
    cone_lower = np.array([
        settings["cone_h_min"],
        settings["cone_s_min"],
        settings["cone_v_min"]
    ])

    cone_upper = np.array([
        settings["cone_h_max"],
        settings["cone_s_max"],
        settings["cone_v_max"]
    ])

    cone_mask = cv2.inRange(hsv, cone_lower, cone_upper)

    kernel = np.ones((cleanup_size, cleanup_size), np.uint8)

    cone_mask = cv2.morphologyEx(cone_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cone_mask = cv2.dilate(cone_mask, kernel, iterations=2)

    output = image.copy()

    # ---------- CENTER LINE ----------
    if settings["show_center_line"] == 1:
        cv2.line(
            output,
            (image_center_x, 0),
            (image_center_x, image_h),
            (255, 255, 0),
            1
        )

    # ---------- CONE DETECTION ----------
    contours, _ = cv2.findContours(
        cone_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cone_count = 0

    if settings["show_cones"] == 1:
        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= settings["min_cone_area"]:
                x, y, box_w, box_h = cv2.boundingRect(contour)

                cone_count += 1

                center_x = x + box_w // 2
                center_y = y + box_h // 2
                offset_x = center_x - image_center_x

                cv2.rectangle(
                    output,
                    (x, y),
                    (x + box_w, y + box_h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(output, (center_x, center_y), 5, (0, 255, 0), -1)

                if settings["show_labels"] == 1:
                    draw_clean_label(
                        output,
                        f"Cone {cone_count}",
                        f"Offset: {offset_x}",
                        x,
                        y,
                        (0, 255, 0)
                    )

    # ---------- IMPROVED PERSON DETECTION ----------
    person_count = 0

    if settings["show_people"] == 1:
        # Enlarge image so small/far people are easier to detect
        person_scale = 1.5
        person_image = cv2.resize(image, None, fx=person_scale, fy=person_scale)

        people, weights = hog.detectMultiScale(
            person_image,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.03
        )

        for (x, y, box_w, box_h) in people:
            # Convert coordinates back to original image size
            x = int(x / person_scale)
            y = int(y / person_scale)
            box_w = int(box_w / person_scale)
            box_h = int(box_h / person_scale)

            person_count += 1

            center_x = x + box_w // 2
            offset_x = center_x - image_center_x

            cv2.rectangle(
                output,
                (x, y),
                (x + box_w, y + box_h),
                (255, 0, 255),
                2
            )

            if settings["show_labels"] == 1:
                draw_clean_label(
                    output,
                    f"Person {person_count}",
                    f"Offset: {offset_x}",
                    x,
                    y,
                    (255, 0, 255)
                )

    # ---------- INFO BOX ----------
    if settings["show_labels"] == 1:
        filename = os.path.basename(image_path)

        info_lines = [
            (f"Image: {current_index + 1}/{len(image_paths)} {filename}", (255, 255, 255)),
            (f"Mode: {mode.upper()}", (0, 255, 255)),
            (f"Cones: {cone_count}", (0, 255, 0)),
            (f"People: {person_count}", (255, 0, 255)),
            (f"Line pixels: {line_pixels}", (0, 255, 255)),
        ]

        draw_text_box(output, info_lines)

    # ---------- MASK VIEW ----------
    if mode == "line":
        mask_to_show = line_mask
        mask_label = "LINE MASK"
    else:
        mask_to_show = cone_mask
        mask_label = "CONE MASK"

    mask_bgr = cv2.cvtColor(mask_to_show, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        mask_bgr,
        mask_label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    side_by_side = np.hstack((output, mask_bgr))

    cv2.imshow("Detection View", side_by_side)

    # ---------- KEYS ----------
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord("d") or key == 83:
        current_index = (current_index + 1) % len(image_paths)

    elif key == ord("a") or key == 81:
        current_index = (current_index - 1) % len(image_paths)

    elif key == ord("m"):
        if mode == "line":
            mode = "cone"
        else:
            mode = "line"

    elif key == ord("l"):
        settings["show_labels"] = 1 - settings["show_labels"]
        print("Show labels:", settings["show_labels"])

    elif key == ord("c"):
        settings["show_cones"] = 1 - settings["show_cones"]
        print("Show cones:", settings["show_cones"])

    elif key == ord("p"):
        settings["show_people"] = 1 - settings["show_people"]
        print("Show people:", settings["show_people"])

    elif key == ord("x"):
        settings["show_center_line"] = 1 - settings["show_center_line"]
        print("Show center line:", settings["show_center_line"])

    elif key == ord("s"):
        save_settings(settings)

cv2.destroyAllWindows()