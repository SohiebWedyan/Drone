"""
Capture left+right chessboard image pairs for stereo calibration.

Usage:
    python -m drone_vision.calibration.capture_pairs
    python -m drone_vision.calibration.capture_pairs --rows 7 --cols 5
    python -m drone_vision.calibration.capture_pairs --cam-l 0 --cam-r 2

Controls:
    SPACE  — save pair (only when both frames detect the board)
    F      — force-save without detection check (for manual inspection)
    ESC/Q  — quit

Tips if detection keeps failing:
    1. Run  python find_chessboard.py  first to auto-detect your board size
    2. Ensure even lighting — avoid glare and shadows on the board
    3. Hold the board 30–70 cm from the cameras
    4. Keep the ENTIRE board visible in both frames
    5. Print the board on paper — do NOT use a phone/tablet screen
"""

import argparse
import os
import sys
import yaml
import cv2
import numpy as np

IMAGES_DIR  = os.path.join(os.path.dirname(__file__), "images")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")

# Enhanced detection flags — much more reliable than defaults
_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def enhance(gray: np.ndarray) -> np.ndarray:
    """Equalise + blur to improve detection under tricky lighting."""
    return cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0)


def detect(gray: np.ndarray, rows: int, cols: int):
    """Try detection with enhanced image first, fall back to raw."""
    found, corners = cv2.findChessboardCorners(enhance(gray), (rows, cols), _FLAGS)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, (rows, cols), _FLAGS)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CRITERIA)
    return found, corners


def overlay_status(frame, label, found, pair_count, rows, cols):
    color = (0, 220, 0) if found else (0, 60, 220)
    icon  = "OK" if found else "--"
    cv2.putText(frame,
                f"{label}:{icon}  pairs:{pair_count}  board:{rows}x{cols}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if not found:
        cv2.putText(frame, "Board not detected",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)


def main():
    cfg  = load_config()
    ap   = argparse.ArgumentParser()
    ap.add_argument("--rows",  type=int, default=cfg["chessboard"]["rows"],
                    help=f"Inner corner rows (default from config: {cfg['chessboard']['rows']})")
    ap.add_argument("--cols",  type=int, default=cfg["chessboard"]["cols"],
                    help=f"Inner corner cols (default from config: {cfg['chessboard']['cols']})")
    ap.add_argument("--cam-l", type=int, default=cfg["left_index"],  help="Left camera index")
    ap.add_argument("--cam-r", type=int, default=cfg["right_index"], help="Right camera index")
    ap.add_argument("--width", type=int, default=cfg["width"])
    ap.add_argument("--height",type=int, default=cfg["height"])
    args = ap.parse_args()

    rows, cols = args.rows, args.cols

    backend = cv2.CAP_V4L2 if cfg.get("backend") == "V4L2" else cv2.CAP_ANY
    cap_l = cv2.VideoCapture(args.cam_l, backend)
    cap_r = cv2.VideoCapture(args.cam_r, backend)

    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap_l.isOpened():
        print(f"ERROR: Cannot open left camera (index {args.cam_l})")
        sys.exit(1)
    if not cap_r.isOpened():
        print(f"ERROR: Cannot open right camera (index {args.cam_r})")
        print("       If you only have one camera, run find_chessboard.py first,")
        print("       then connect the second camera before running this script.")
        sys.exit(1)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    pair_count = len([f for f in os.listdir(IMAGES_DIR) if f.startswith("left_")])

    print(f"Chessboard: {rows} x {cols} inner corners")
    print(f"  (Not matching? Run:  python find_chessboard.py  to auto-detect)")
    print(f"Saving to: {IMAGES_DIR}")
    print(f"SPACE=save  F=force-save  ESC/Q=quit")
    print("-" * 50)

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            print("ERROR: Frame capture failed.")
            break

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = detect(gray_l, rows, cols)
        found_r, corners_r = detect(gray_r, rows, cols)

        display_l = frame_l.copy()
        display_r = frame_r.copy()

        if found_l:
            cv2.drawChessboardCorners(display_l, (rows, cols), corners_l, True)
        if found_r:
            cv2.drawChessboardCorners(display_r, (rows, cols), corners_r, True)

        overlay_status(display_l, "LEFT",  found_l, pair_count, rows, cols)
        overlay_status(display_r, "RIGHT", found_r, pair_count, rows, cols)

        # Resize to fit screen if frames are large
        h, w = display_l.shape[:2]
        scale = min(1.0, 800 / w)
        if scale < 1.0:
            dw, dh = int(w * scale), int(h * scale)
            display_l = cv2.resize(display_l, (dw, dh))
            display_r = cv2.resize(display_r, (dw, dh))

        combined = cv2.hconcat([display_l, display_r])
        cv2.imshow("Stereo Calibration  (SPACE=save  F=force  ESC=quit)", combined)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break

        elif key == ord(' '):
            if found_l and found_r:
                cv2.imwrite(os.path.join(IMAGES_DIR, f"left_{pair_count:03d}.png"),  frame_l)
                cv2.imwrite(os.path.join(IMAGES_DIR, f"right_{pair_count:03d}.png"), frame_r)
                pair_count += 1
                print(f"[{pair_count:02d}] Saved pair")
            else:
                missing = []
                if not found_l: missing.append("LEFT")
                if not found_r: missing.append("RIGHT")
                print(f"      Not detected in: {', '.join(missing)} — skipped")
                print(f"      Tip: run  python find_chessboard.py  to check your board size")

        elif key in (ord('f'), ord('F')):
            # Force-save regardless of detection (useful for debugging)
            cv2.imwrite(os.path.join(IMAGES_DIR, f"left_{pair_count:03d}.png"),  frame_l)
            cv2.imwrite(os.path.join(IMAGES_DIR, f"right_{pair_count:03d}.png"), frame_r)
            pair_count += 1
            print(f"[{pair_count:02d}] Force-saved pair (no detection check)")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total pairs saved: {pair_count}")
    if pair_count < 15:
        print(f"  Need at least 15 pairs for good calibration ({15 - pair_count} more needed).")


if __name__ == "__main__":
    main()
