from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    track_id: int
    x: float        # center x (pixels)
    y: float        # center y (pixels)
    w: float        # width (pixels)
    h: float        # height (pixels)
    conf: float


class Detector:
    """YOLOv8 wrapper that returns Detection objects with persistent track IDs."""

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.5):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._conf = conf
        self._iou = iou

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame,
            persist=True,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                track_id = int(box.id) if box.id is not None else -1
                x, y, w, h = (float(v) for v in box.xywh[0])
                conf = float(box.conf[0])
                detections.append(Detection(track_id=track_id, x=x, y=y, w=w, h=h, conf=conf))
        return detections
