import cv2
import numpy as np
from drone_vision.core.track_manager import Track
from drone_vision.core.motion_analyzer import MotionInfo
from drone_vision.core.alert_manager import Alert

_GREEN  = (0, 255, 0)
_BLUE   = (255, 0, 0)
_RED    = (0, 0, 255)
_YELLOW = (0, 255, 255)
_WHITE  = (255, 255, 255)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


def draw_bbox(frame: np.ndarray, track: Track, motion: MotionInfo | None = None):
    x, y, w, h = int(track.x), int(track.y), int(track.w), int(track.h)
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)

    label = f"ID:{track.track_id} Z:{track.z:.1f}m"
    if motion:
        label += f" V:{motion.speed_mps:.2f}m/s {motion.dir_label}"
    cv2.putText(frame, label, (x1, y1 - 8), _FONT, 0.55, _GREEN, 2)


def draw_trajectory(frame: np.ndarray, history: list[tuple[int, int]]):
    if len(history) < 2:
        return
    for i in range(1, len(history)):
        alpha = i / len(history)
        color = (0, int(200 * alpha), 0)
        cv2.line(frame, history[i - 1], history[i], color, 2)
    cv2.circle(frame, history[-1], 4, _BLUE, -1)


def draw_prediction(frame: np.ndarray, current: tuple[int, int], predicted: tuple[int, int]):
    cv2.arrowedLine(frame, current, predicted, _YELLOW, 2, tipLength=0.3)
    cv2.circle(frame, predicted, 6, _RED, -1)


def draw_alerts(frame: np.ndarray, alerts: list[Alert]):
    for i, alert in enumerate(alerts):
        y = 40 + i * 32
        cv2.putText(frame, alert.message, (15, y), _FONT, 0.75, _RED, 3)


def draw_hud(frame: np.ndarray, fps: float, track_count: int, stereo_active: bool):
    h, w = frame.shape[:2]
    bar = f"FPS:{fps:.1f}  Objects:{track_count}  Stereo:{'ON' if stereo_active else 'OFF'}"
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, bar, (8, h - 8), _FONT, 0.55, _WHITE, 1)
