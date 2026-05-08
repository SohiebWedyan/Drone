"""
Compute stereo calibration from saved image pairs.
Usage: python calibration/stereo_calibrate.py
Reads images from calibration/images/, writes config/stereo_calib.npz.
"""

import cv2
import numpy as np
import os
import sys
import yaml
import glob

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "stereo_calib.npz")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    rows = cfg["chessboard"]["rows"]
    cols = cfg["chessboard"]["cols"]
    square_size = cfg["chessboard"]["square_size"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

    obj_points = []
    img_points_l = []
    img_points_r = []

    left_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "right_*.png")))

    if len(left_files) < 5:
        print(f"ERROR: Need at least 5 pairs, found {len(left_files)}. Run capture_pairs.py first.")
        sys.exit(1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None

    print(f"Processing {len(left_files)} image pairs...")
    valid = 0
    for lf, rf in zip(left_files, right_files):
        img_l = cv2.imread(lf)
        img_r = cv2.imread(rf)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]

        found_l, corners_l = cv2.findChessboardCorners(gray_l, (rows, cols), None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, (rows, cols), None)

        if found_l and found_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            valid += 1
            print(f"  OK: {os.path.basename(lf)}")
        else:
            print(f"  SKIP: {os.path.basename(lf)} (corners not found)")

    print(f"\nValid pairs: {valid}/{len(left_files)}")
    if valid < 5:
        print("ERROR: Not enough valid pairs for reliable calibration.")
        sys.exit(1)

    print("Calibrating left camera...")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, img_size, None, None)
    print(f"  Left RMS error: {ret_l:.4f}")

    print("Calibrating right camera...")
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, img_size, None, None)
    print(f"  Right RMS error: {ret_r:.4f}")

    print("Stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r,
        img_size, flags=flags,
        criteria=criteria
    )
    print(f"  Stereo RMS error: {ret_s:.4f}")

    print("Computing rectification maps...")
    R1, R2, P1, P2, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, img_size, cv2.CV_32FC1)

    np.savez(
        OUTPUT_PATH,
        K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        img_size=np.array(img_size),
        rms=np.array([ret_l, ret_r, ret_s])
    )
    print(f"\nCalibration saved to: {OUTPUT_PATH}")
    print(f"Focal length (fx): {K_l[0,0]:.2f} px")
    print(f"Baseline: {abs(T[0,0]):.4f} m")


if __name__ == "__main__":
    main()
