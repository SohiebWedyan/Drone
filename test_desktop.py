"""
Drone Vision Phase 2 — Desktop Test Script
==========================================
Single-file, single-camera version for testing on a regular computer.
No stereo calibration required — uses height-ratio depth fallback.

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    python test_desktop.py                        # webcam 0, downloads yolov8n.pt
    python test_desktop.py --model best.pt        # your custom model
    python test_desktop.py --cam 1 --conf 0.35    # camera index 1
    python test_desktop.py --source video.mp4     # test on a video file

Controls:
    ESC / Q  — quit
    P        — pause / resume
    S        — save current frame as screenshot
"""

import argparse
import math
import time
import cv2
import numpy as np

# ─── Inline Kalman Tracker ────────────────────────────────────────────────────

class KalmanTracker:
    """6D constant-velocity Kalman filter: state = [x, y, z, vx, vy, vz]."""

    def __init__(self, track_id: int, x: float, y: float, z: float):
        self.track_id   = track_id
        self.last_seen  = 0
        self._kf        = cv2.KalmanFilter(6, 3)

        self._kf.transitionMatrix = np.array([
            [1,0,0,1,0,0],
            [0,1,0,0,1,0],
            [0,0,1,0,0,1],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float32)

        self._kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0],
        ], dtype=np.float32)

        self._kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self._kf.errorCovPost        = np.eye(6, dtype=np.float32)
        self._kf.statePost           = np.array(
            [[x],[y],[z],[0],[0],[0]], dtype=np.float32)

    def set_dt(self, dt: float):
        for i, j in ((0,3),(1,4),(2,5)):
            self._kf.transitionMatrix[i, j] = dt

    def predict(self):
        return self._kf.predict().flatten()

    def update(self, x: float, y: float, z: float):
        m = np.array([[x],[y],[z]], dtype=np.float32)
        return self._kf.correct(m).flatten()

    @property
    def pos(self):
        s = self._kf.statePost.flatten()
        return float(s[0]), float(s[1]), float(s[2])

    @property
    def vel(self):
        s = self._kf.statePost.flatten()
        return float(s[3]), float(s[4]), float(s[5])


# ─── Inline Alert Manager ─────────────────────────────────────────────────────

class AlertManager:
    def __init__(self, speed_thr=0.5, z_min=3.0, cooldown=2.0):
        self._speed_thr = speed_thr
        self._z_min     = z_min
        self._cooldown  = cooldown
        self._last: dict[str, float] = {}

    def check(self, tid, speed, z) -> list[str]:
        now    = time.time()
        alerts = []
        if self._fire(f"{tid}:det", now):
            alerts.append(f"Object #{tid} detected")
        if speed > self._speed_thr and self._fire(f"{tid}:spd", now):
            alerts.append(f"Object #{tid} fast  {speed:.1f} m/s")
        if 0 < z < self._z_min and self._fire(f"{tid}:prx", now):
            alerts.append(f"Object #{tid} close  {z:.1f} m")
        return alerts

    def _fire(self, key, now) -> bool:
        if now - self._last.get(key, 0) >= self._cooldown:
            self._last[key] = now
            return True
        return False


# ─── Inline Visualiser ────────────────────────────────────────────────────────

def draw_bbox(frame, x, y, w, h, tid, z, speed, direction):
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    label = f"ID:{tid}  Z:{z:.1f}m  V:{speed:.2f}m/s  {direction}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), (0, 220, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_trajectory(frame, history):
    if len(history) < 2:
        return
    for i in range(1, len(history)):
        alpha = i / len(history)
        color = (0, int(200 * alpha), 0)
        cv2.line(frame, history[i-1], history[i], color, 2)
    cv2.circle(frame, history[-1], 4, (255, 80, 0), -1)


def draw_prediction(frame, cx, cy, nx, ny):
    cv2.arrowedLine(frame, (int(cx), int(cy)), (int(nx), int(ny)),
                    (0, 220, 220), 2, tipLength=0.35)
    cv2.circle(frame, (int(nx), int(ny)), 7, (0, 0, 255), -1)


def draw_alerts(frame, alerts: list[str]):
    for i, msg in enumerate(alerts):
        y = 48 + i * 34
        cv2.putText(frame, f"! {msg}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 3)
        cv2.putText(frame, f"! {msg}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 1)


def draw_hud(frame, fps, count, paused):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
    status = f"FPS:{fps:.1f}  Objects:{count}  Mode:Single-Cam (fallback depth)"
    if paused:
        status = "[PAUSED]  " + status
    cv2.putText(frame, status, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def direction_label(vx, vy) -> str:
    angle = math.degrees(math.atan2(vy, vx)) % 360
    sectors = [
        (22.5,  "Right"),
        (67.5,  "Right+Down"),
        (112.5, "Down"),
        (157.5, "Left+Down"),
        (202.5, "Left"),
        (247.5, "Left+Up"),
        (292.5, "Up"),
        (337.5, "Right+Up"),
    ]
    for limit, label in sectors:
        if angle < limit:
            return label
    return "Right"


# ─── Rolling FPS ─────────────────────────────────────────────────────────────

class FPS:
    def __init__(self, n=30):
        self._t = []
        self._n = n

    def tick(self):
        self._t.append(time.time())
        if len(self._t) > self._n:
            self._t.pop(0)

    @property
    def value(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Drone Vision — Desktop Test")
    ap.add_argument("--model",  default="yolov8n.pt",
                    help="Path to YOLO model (default: yolov8n.pt — downloads automatically)")
    ap.add_argument("--source", default=None,
                    help="Video file path (default: webcam)")
    ap.add_argument("--cam",    type=int, default=0,
                    help="Camera index (default: 0)")
    ap.add_argument("--conf",   type=float, default=0.4,
                    help="Detection confidence threshold (default: 0.4)")
    ap.add_argument("--history",type=int,   default=30,
                    help="Trajectory history length (default: 30)")
    ap.add_argument("--max-gone", type=int, default=15,
                    help="Frames before expiring a lost track (default: 15)")
    args = ap.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # ── open source ───────────────────────────────────────────────────────────
    src = args.source if args.source else args.cam
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {src}")
        return

    print("Controls:  ESC/Q = quit  |  P = pause  |  S = screenshot")
    print("-" * 55)

    # ── state ─────────────────────────────────────────────────────────────────
    trackers:   dict[int, KalmanTracker] = {}
    histories:  dict[int, list]          = {}
    prev_time:  dict[int, float]         = {}
    alert_mgr   = AlertManager(speed_thr=0.5, z_min=3.0, cooldown=2.0)
    fps_counter = FPS()
    active_alerts: list[str] = []
    paused      = False
    screenshot  = 0

    REAL_H      = 0.30   # assumed object real height in meters
    FOCAL_LEN   = 800.0  # approximate focal length in pixels
    PRED_DT     = 0.10   # seconds ahead for prediction

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

        # ── YOLOv8 detect + track ─────────────────────────────────────────────
        results = model.track(frame, persist=True, conf=args.conf,
                               iou=0.5, verbose=False)

        now         = time.time()
        seen_ids    = set()
        frame_alerts = []

        for r in results:
            for box in r.boxes:
                tid = int(box.id) if box.id is not None else -1
                if tid < 0:
                    continue
                seen_ids.add(tid)
                x, y, w, h = (float(v) for v in box.xywh[0])

                # depth fallback
                z = (REAL_H * FOCAL_LEN) / h if h > 0 else 5.0

                # init or update Kalman tracker
                dt = max(1e-4, now - prev_time.get(tid, now))
                prev_time[tid] = now

                if tid not in trackers:
                    trackers[tid]  = KalmanTracker(tid, x, y, z)
                    histories[tid] = []
                else:
                    trackers[tid].set_dt(dt)
                    trackers[tid].predict()

                trackers[tid].update(x, y, z)

                kx, ky, kz = trackers[tid].pos
                vx, vy, vz = trackers[tid].vel

                # history for trajectory
                histories[tid].append((int(kx), int(ky)))
                if len(histories[tid]) > args.history:
                    histories[tid].pop(0)

                # motion
                meter_per_px = REAL_H / h if h > 0 else 0.001
                speed = math.sqrt(
                    (vx * meter_per_px)**2 +
                    (vy * meter_per_px)**2 +
                    vz**2
                )
                direction = direction_label(vx, vy)

                # prediction
                nx = kx + vx * PRED_DT
                ny = ky + vy * PRED_DT

                # draw
                draw_trajectory(frame, histories[tid])
                draw_bbox(frame, kx, ky, w, h, tid, kz, speed, direction)
                draw_prediction(frame, kx, ky, nx, ny)

                # alerts
                frame_alerts.extend(alert_mgr.check(tid, speed, kz))

                print(f"ID:{tid:3d}  Z={kz:.2f}m  V={speed:.2f}m/s  {direction}")

        # expire lost tracks
        gone = [tid for tid in trackers if tid not in seen_ids]
        for tid in gone:
            trackers[tid].last_seen += 1
            if trackers[tid].last_seen > args.max_gone:
                del trackers[tid]
                del histories[tid]
                prev_time.pop(tid, None)

        # keep alert list fresh (replace each frame so they auto-clear)
        active_alerts = frame_alerts

        draw_alerts(frame, active_alerts)
        fps_counter.tick()
        draw_hud(frame, fps_counter.value, len(seen_ids), paused)

        cv2.imshow("Drone Vision — Desktop Test  (ESC/Q=quit  P=pause  S=save)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key in (ord('s'), ord('S')):
            fname = f"screenshot_{screenshot:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
            screenshot += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
