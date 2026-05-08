"""
Capture left+right chessboard image pairs for stereo calibration.
Usage: python calibration/capture_pairs.py
  SPACE → save current pair
  ESC   → quit
Target: 15-20 pairs with chessboard at varied angles and positions.
"""

import cv2
import os
import sys
import yaml

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    left_idx = cfg["left_index"]
    right_idx = cfg["right_index"]
    width = cfg["width"]
    height = cfg["height"]
    rows = cfg["chessboard"]["rows"]
    cols = cfg["chessboard"]["cols"]

    backend = cv2.CAP_V4L2 if cfg.get("backend") == "V4L2" else cv2.CAP_ANY
    cap_l = cv2.VideoCapture(left_idx, backend)
    cap_r = cv2.VideoCapture(right_idx, backend)
    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("ERROR: Could not open one or both cameras.")
        sys.exit(1)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    pair_count = len([f for f in os.listdir(IMAGES_DIR) if f.startswith("left_")]) // 1
    print(f"Saving pairs to: {IMAGES_DIR}")
    print(f"Chessboard pattern: {rows}x{cols} inner corners")
    print("SPACE = save pair | ESC = quit")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            print("ERROR: Frame capture failed.")
            break

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(gray_l, (rows, cols), None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, (rows, cols), None)

        display_l = frame_l.copy()
        display_r = frame_r.copy()
        if found_l:
            cv2.drawChessboardCorners(display_l, (rows, cols), corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display_r, (rows, cols), corners_r, found_r)

        status = f"Pairs: {pair_count} | "
        status += "L:OK " if found_l else "L:-- "
        status += "R:OK" if found_r else "R:--"
        cv2.putText(display_l, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = cv2.hconcat([display_l, display_r])
        cv2.imshow("Stereo Calibration Capture (SPACE=save, ESC=quit)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(" "):
            if found_l and found_r:
                cv2.imwrite(os.path.join(IMAGES_DIR, f"left_{pair_count:03d}.png"), frame_l)
                cv2.imwrite(os.path.join(IMAGES_DIR, f"right_{pair_count:03d}.png"), frame_r)
                pair_count += 1
                print(f"Saved pair {pair_count}")
            else:
                print("Chessboard not detected in both frames — skipping.")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"Done. Total pairs saved: {pair_count}")


if __name__ == "__main__":
    main()
