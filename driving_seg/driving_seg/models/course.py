"""course: fine-tuned nano seg model for {cone, white_line} — the classes
no pretrained model covers. Weights come from tools/train_course.py; until
they exist this stage contributes nothing (base class warns once).
"""

from .base import SegModel

ID_TO_CLASS = {0: "cone", 1: "white_line"}


class CourseModel(SegModel):
    name = "course"

    def __init__(self, weights="models/course.pt", conf=0.30,
                 device=None, imgsz=640):
        super().__init__()
        self.weights = weights
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def _load(self):
        from ultralytics import YOLO
        return YOLO(self.weights)

    def _predict(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        res = self._backend.predict(bgr, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        out = {}
        if res.masks is None:
            return out
        masks = res.masks.data.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        for m, cid in zip(masks, clses):
            name = ID_TO_CLASS.get(int(cid))
            if name is None:
                continue
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            out[name] = out[name] | m if name in out else m
        return out
