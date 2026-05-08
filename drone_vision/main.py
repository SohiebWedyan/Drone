"""
Drone Vision Phase 2 — Main entry point.
Usage: python drone_vision/main.py [--config-dir path/to/config]
"""

import argparse
import os
import queue
import sys
import threading
import time

import cv2
import yaml

# Resolve paths relative to this file so the script works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Capture thread
# ─────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """
    Reads left+right frame pairs from two USB cameras and places them
    into a bounded queue. maxsize=2 ensures stale frames are dropped
    automatically, decoupling capture latency from processing latency.
    """

    def __init__(self, cam_cfg: dict, frame_queue: queue.Queue):
        super().__init__(daemon=True, name="CaptureThread")
        self._cfg = cam_cfg
        self._q = frame_queue
        self._stop_event = threading.Event()

    def run(self):
        cfg = self._cfg
        backend = cv2.CAP_V4L2 if cfg.get("backend") == "V4L2" else cv2.CAP_ANY
        cap_l = cv2.VideoCapture(cfg["left_index"], backend)
        cap_r = cv2.VideoCapture(cfg["right_index"], backend)
        for cap in (cap_l, cap_r):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            cap.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))

        if not cap_l.isOpened() or not cap_r.isOpened():
            print("ERROR [CaptureThread]: Cannot open cameras.")
            return

        while not self._stop_event.is_set():
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            if not ret_l or not ret_r:
                time.sleep(0.01)
                continue
            try:
                self._q.put_nowait((frame_l, frame_r, time.time()))
            except queue.Full:
                pass   # drop the oldest; consumer is slow

        cap_l.release()
        cap_r.release()

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────

def main(config_dir: str):
    cam_cfg = _load_yaml(os.path.join(config_dir, "cameras.yaml"))
    sys_cfg = _load_yaml(os.path.join(config_dir, "system.yaml"))

    # calib_path is relative to the package root; resolve against config_dir's parent
    _pkg_root = os.path.dirname(config_dir)
    calib_path = os.path.join(_pkg_root, cam_cfg["stereo"]["calib_path"])
    log_file = os.path.join(_HERE, "..", sys_cfg["performance"]["log_file"])

    from .core.stereo_depth import StereoDepth
    from .core.detector import Detector
    from .core.track_manager import TrackManager
    from .core.motion_analyzer import MotionAnalyzer
    from .core.alert_manager import AlertManager
    from .utils.visualizer import (
        draw_bbox, draw_trajectory, draw_prediction, draw_alerts, draw_hud
    )
    from .utils.fps_counter import FPSCounter
    from .utils.logger import get_logger

    logger = get_logger(log_file=log_file)
    logger.info("Starting Drone Vision Phase 2")

    depth_module = StereoDepth(calib_path if os.path.isfile(calib_path) else None)
    if depth_module.calibrated:
        logger.info("Stereo calibration loaded.")
    else:
        logger.warning("No calibration found — using height-ratio depth fallback.")

    model_path = os.path.join(_pkg_root, sys_cfg["model_path"])
    detector = Detector(
        model_path=model_path,
        conf=sys_cfg["confidence"],
        iou=sys_cfg["iou_threshold"],
    )
    track_mgr = TrackManager(
        max_disappeared=sys_cfg["max_disappeared_frames"],
        history_len=sys_cfg["track_history_len"],
    )
    motion_analyzer = MotionAnalyzer(dt=sys_cfg["prediction"]["dt"])
    alert_mgr = AlertManager(sys_cfg["alert"])
    fps_counter = FPSCounter()

    frame_queue: queue.Queue = queue.Queue(maxsize=2)
    capture_thread = CaptureThread(cam_cfg, frame_queue)
    capture_thread.start()
    logger.info("Capture thread started.")

    fps_warn = sys_cfg["performance"]["fps_warning_threshold"]

    try:
        while True:
            try:
                frame_l, frame_r, _ = frame_queue.get(timeout=2.0)
            except queue.Empty:
                logger.warning("No frames received for 2s — check cameras.")
                continue

            # Stereo rectify + disparity
            left_rect, right_rect = depth_module.rectify(frame_l, frame_r)
            disparity = depth_module.compute_disparity(left_rect, right_rect)

            # Detection on rectified left frame
            detections = detector.detect(left_rect)

            # Tracking
            tracks = track_mgr.update(detections, depth_module, disparity)

            # Motion analysis
            motions = {t.track_id: motion_analyzer.analyze(t) for t in tracks}

            # Alerts
            alerts = alert_mgr.check(tracks, motions)
            for alert in alerts:
                logger.info(alert.message)

            # Visualize on left frame
            display = left_rect.copy()
            for track in tracks:
                motion = motions.get(track.track_id)
                draw_trajectory(display, track.history)
                draw_bbox(display, track, motion)
                if motion:
                    draw_prediction(
                        display,
                        (int(track.x), int(track.y)),
                        (motion.x_next, motion.y_next),
                    )

            draw_alerts(display, alerts)
            fps_counter.tick()
            current_fps = fps_counter.fps
            if current_fps > 0 and current_fps < fps_warn:
                logger.warning(f"Low FPS: {current_fps:.1f}")
            draw_hud(display, current_fps, len(tracks), depth_module.calibrated)

            cv2.imshow("Drone Vision Phase 2", display)
            if cv2.waitKey(1) == 27:
                break

    finally:
        capture_thread.stop()
        capture_thread.join(timeout=3.0)
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Vision Phase 2")
    parser.add_argument(
        "--config-dir",
        default=os.path.join(_HERE, "config"),
        help="Path to config directory containing cameras.yaml and system.yaml",
    )
    args = parser.parse_args()
    main(config_dir=args.config_dir)
