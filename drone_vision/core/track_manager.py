from dataclasses import dataclass, field
from collections import deque
import time
import numpy as np

from .detector import Detection
from .kalman_tracker import KalmanTracker
from .stereo_depth import StereoDepth


@dataclass
class Track:
    track_id: int
    x: float
    y: float
    z: float
    vx: float   # pixels/s or m/s depending on coordinate space
    vy: float
    vz: float
    w: float    # bounding box width
    h: float    # bounding box height
    history: list = field(default_factory=list)   # [(x, y)] pixel positions


class TrackManager:
    """
    Maintains a KalmanTracker per object ID (sourced from YOLOv8 persistent tracking).
    Updates trackers each frame and expires stale ones.
    """

    def __init__(self, max_disappeared: int = 15, history_len: int = 30):
        self._trackers: dict[int, KalmanTracker] = {}
        self._histories: dict[int, deque] = {}
        self._last_dt: dict[int, float] = {}
        self._max_disappeared = max_disappeared
        self._history_len = history_len
        self._last_time = time.time()

    def update(
        self,
        detections: list[Detection],
        depth: StereoDepth,
        disparity: np.ndarray | None,
    ) -> list[Track]:
        now = time.time()
        dt = max(1e-4, now - self._last_time)
        self._last_time = now

        seen_ids: set[int] = set()

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue
            seen_ids.add(tid)

            # Resolve depth
            if disparity is not None:
                z = depth.depth_at(disparity, int(det.x), int(det.y))
                if z is None:
                    z = StereoDepth.depth_from_height(det.h)
            else:
                z = StereoDepth.depth_from_height(det.h)

            measurement = (det.x, det.y, z)

            if tid not in self._trackers:
                self._trackers[tid] = KalmanTracker(tid, measurement)
                self._histories[tid] = deque(maxlen=self._history_len)
            else:
                self._trackers[tid].set_dt(dt)
                self._trackers[tid].predict()

            self._trackers[tid].update(measurement)
            self._trackers[tid].last_seen = 0
            self._histories[tid].append((int(det.x), int(det.y)))

        # Age unseen trackers
        stale = []
        for tid, tracker in self._trackers.items():
            if tid not in seen_ids:
                tracker.last_seen += 1
                if tracker.last_seen > self._max_disappeared:
                    stale.append(tid)
        for tid in stale:
            del self._trackers[tid]
            del self._histories[tid]

        # Build Track list
        tracks: list[Track] = []
        for tid in seen_ids:
            if tid not in self._trackers:
                continue
            t = self._trackers[tid]
            x, y, z = t.position
            vx, vy, vz = t.velocity

            # Find matching detection for bbox size
            bbox_w, bbox_h = 0.0, 0.0
            for det in detections:
                if det.track_id == tid:
                    bbox_w, bbox_h = det.w, det.h
                    break

            tracks.append(Track(
                track_id=tid,
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                w=bbox_w, h=bbox_h,
                history=list(self._histories[tid]),
            ))

        return tracks
