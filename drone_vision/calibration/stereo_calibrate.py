"""
Compute stereo calibration from saved image pairs.
Usage: python -m drone_vision.calibration.stereo_calibrate
Reads images from calibration/images/, writes config/stereo_calib.npz.
"""

import cv2
import numpy as np
import os
import sys
import yaml
import glob

IMAGES_DIR  = os.path.join(os.path.dirname(__file__), "images")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "stereo_calib.npz")

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)

PATTERNS = [
    (9, 6), (6, 9),
    (8, 6), (6, 8),
    (7, 6), (6, 7),
    (8, 5), (5, 8),
    (7, 5), (5, 7),
    (6, 5), (5, 6),
    (9, 7), (7, 9),
    (10, 7), (7, 10),
    (6, 4), (4, 6),
    (5, 4), (4, 5),
    (4, 3), (3, 4),
]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def enhance(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0)


def detect(gray: np.ndarray, rows: int, cols: int):
    """Try with enhanced image first, then raw, using improved flags."""
    for img in (enhance(gray), gray):
        found, corners = cv2.findChessboardCorners(img, (rows, cols), FLAGS)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners
    return False, None


def auto_detect_pattern(gray: np.ndarray):
    """Try all common patterns and return the first match."""
    for img in (enhance(gray), gray):
        for rows, cols in PATTERNS:
            found, corners = cv2.findChessboardCorners(img, (rows, cols), FLAGS)
            if found:
                return rows, cols
    return None, None


def main():
    cfg         = load_config()
    rows        = cfg["chessboard"]["rows"]
    cols        = cfg["chessboard"]["cols"]
    square_size = cfg["chessboard"]["square_size"]

    print(f"Config: {rows}×{cols} inner corners, square={square_size*100:.1f} cm")

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

    obj_points   = []
    img_points_l = []
    img_points_r = []

    left_files  = sorted(glob.glob(os.path.join(IMAGES_DIR, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "right_*.png")))

    if len(left_files) < 5:
        print(f"ERROR: Need at least 5 pairs, found {len(left_files)}.")
        print("       Run capture_pairs.py first.")
        sys.exit(1)

    print(f"Processing {len(left_files)} image pairs...\n")
    img_size = None
    valid    = 0

    for lf, rf in zip(left_files, right_files):
        img_l  = cv2.imread(lf)
        img_r  = cv2.imread(rf)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]

        found_l, corners_l = detect(gray_l, rows, cols)
        found_r, corners_r = detect(gray_r, rows, cols)

        # If config pattern fails, try auto-detection and warn
        if not found_l or not found_r:
            ar, ac = auto_detect_pattern(gray_l)
            if ar is not None and (ar != rows or ac != cols):
                print(f"  HINT: auto-detected pattern {ar}×{ac} — "
                      f"update cameras.yaml if this differs from {rows}×{cols}")

        if found_l and found_r:
            obj_points.append(objp)
            img_points_l.append(corners_l)
            img_points_r.append(corners_r)
            valid += 1
            print(f"  OK   {os.path.basename(lf)}")
        else:
            missing = []
            if not found_l: missing.append("LEFT")
            if not found_r: missing.append("RIGHT")
            print(f"  SKIP {os.path.basename(lf)}  —  not detected: {', '.join(missing)}")

    print(f"\nValid pairs: {valid}/{len(left_files)}")

    if valid < 5:
        print("ERROR: Not enough valid pairs. Re-capture with better lighting / flat board.")
        sys.exit(1)

    # ── Per-camera calibration ────────────────────────────────────────────────
    print("\nCalibrating left camera...")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        obj_points, img_points_l, img_size, None, None)
    print(f"  Left  RMS: {ret_l:.4f}  {'✅' if ret_l < 1.0 else '⚠️  > 1.0 — re-capture recommended'}")

    print("Calibrating right camera...")
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        obj_points, img_points_r, img_size, None, None)
    print(f"  Right RMS: {ret_r:.4f}  {'✅' if ret_r < 1.0 else '⚠️  > 1.0 — re-capture recommended'}")

    # ── Stereo calibration ────────────────────────────────────────────────────
    print("\nStereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, D_l, K_r, D_r,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    print(f"  Stereo RMS: {ret_s:.4f}  {'✅' if ret_s < 1.0 else '⚠️  > 1.0 — re-capture recommended'}")

    # ── Rectification ─────────────────────────────────────────────────────────
    print("\nComputing rectification maps...")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T, alpha=0)
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, img_size, cv2.CV_32FC1)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(
        OUTPUT_PATH,
        K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        img_size=np.array(img_size),
        rms=np.array([ret_l, ret_r, ret_s]),
    )

    baseline = abs(float(T[0, 0]))
    print(f"\n{'='*45}")
    print(f"Calibration saved → {OUTPUT_PATH}")
    print(f"Focal length (fx) : {K_l[0,0]:.2f} px")
    print(f"Baseline          : {baseline:.4f} m  ({baseline*100:.1f} cm)")
    print(f"{'='*45}")

    if ret_s > 1.0:
        print("\n⚠️  RMS > 1.0 — calibration may be inaccurate.")
        print("   Tips:")
        print("   • Make sure the board is completely flat (glued to cardboard)")
        print("   • Use good even lighting, no glare")
        print("   • Hold still when pressing SPACE")
        print("   • Vary the board angle and position in each shot")
    else:
        print("\n✅ Calibration looks good — run:  python3 run.py")


if __name__ == "__main__":
    main()
