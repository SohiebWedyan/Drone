import math
from dataclasses import dataclass

from .track_manager import Track


@dataclass
class MotionInfo:
    speed_mps: float           # 3D speed in m/s (from Kalman velocity)
    dir_angle_deg: float       # 2D angle in image plane, degrees (0=right, 90=down)
    dir_label: str             # e.g. "Right+Down"
    x_next: int                # predicted pixel x
    y_next: int                # predicted pixel y
    z_next: float              # predicted depth (m)


class MotionAnalyzer:
    """
    Derives speed, direction, and next-position prediction from a Track's
    Kalman-smoothed velocity. Uses pixel velocities for 2D direction;
    uses depth velocity for z prediction.
    """

    def __init__(self, dt: float = 0.1):
        self._dt = dt   # seconds ahead for prediction

    def analyze(self, track: Track) -> MotionInfo:
        vx, vy, vz = track.vx, track.vy, track.vz

        # Speed: convert pixel velocity to approximate m/s using depth
        meter_per_pixel = 0.3 / track.h if track.h > 0 else 0.001
        speed_mps = math.sqrt(
            (vx * meter_per_pixel) ** 2 +
            (vy * meter_per_pixel) ** 2 +
            vz ** 2
        )

        dir_angle_deg = math.degrees(math.atan2(vy, vx))

        dir_label = self._angle_to_label(dir_angle_deg)

        x_next = int(track.x + vx * self._dt)
        y_next = int(track.y + vy * self._dt)
        z_next = track.z + vz * self._dt

        return MotionInfo(
            speed_mps=speed_mps,
            dir_angle_deg=dir_angle_deg,
            dir_label=dir_label,
            x_next=x_next,
            y_next=y_next,
            z_next=z_next,
        )

    @staticmethod
    def _angle_to_label(angle_deg: float) -> str:
        # Normalize to 0-360
        a = angle_deg % 360
        if 337.5 <= a or a < 22.5:
            return "Right"
        elif 22.5 <= a < 67.5:
            return "Right+Down"
        elif 67.5 <= a < 112.5:
            return "Down"
        elif 112.5 <= a < 157.5:
            return "Left+Down"
        elif 157.5 <= a < 202.5:
            return "Left"
        elif 202.5 <= a < 247.5:
            return "Left+Up"
        elif 247.5 <= a < 292.5:
            return "Up"
        else:
            return "Right+Up"
