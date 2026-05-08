import time
from dataclasses import dataclass

from .track_manager import Track
from .motion_analyzer import MotionInfo


@dataclass
class Alert:
    track_id: int
    message: str
    kind: str   # "detected" | "speed" | "proximity"


class AlertManager:
    """
    Fires alerts based on configurable thresholds with per-track cooldown.
    Config keys: speed_threshold (m/s), z_min (m), cooldown_seconds.
    """

    def __init__(self, config: dict):
        self._speed_threshold: float = config.get("speed_threshold", 0.5)
        self._z_min: float = config.get("z_min", 3.0)
        self._cooldown: float = config.get("cooldown_seconds", 2.0)
        self._last_alert: dict[str, float] = {}   # key: "tid:kind" → timestamp

    def check(self, tracks: list[Track], motions: dict[int, MotionInfo]) -> list[Alert]:
        alerts: list[Alert] = []
        now = time.time()

        for track in tracks:
            tid = track.track_id
            motion = motions.get(tid)

            # Trigger: object present
            if self._fire(tid, "detected", now):
                alerts.append(Alert(tid, f"ALERT: Object #{tid} Detected!", "detected"))

            if motion is None:
                continue

            # Trigger: speed
            if motion.speed_mps > self._speed_threshold:
                if self._fire(tid, "speed", now):
                    alerts.append(Alert(
                        tid,
                        f"ALERT: Object #{tid} fast ({motion.speed_mps:.1f} m/s)!",
                        "speed",
                    ))

            # Trigger: proximity
            if 0 < track.z < self._z_min:
                if self._fire(tid, "proximity", now):
                    alerts.append(Alert(
                        tid,
                        f"ALERT: Object #{tid} close ({track.z:.1f} m)!",
                        "proximity",
                    ))

        return alerts

    def _fire(self, tid: int, kind: str, now: float) -> bool:
        key = f"{tid}:{kind}"
        last = self._last_alert.get(key, 0.0)
        if now - last >= self._cooldown:
            self._last_alert[key] = now
            return True
        return False
