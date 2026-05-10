"""
Compute stereo calibration from saved image pairs.
Usage: python -m drone_vision.calibration.stereo_calibrate
Reads images from calibration/images/, writes config/stereo_calib.npz.

Supported naming patterns (auto-detected):
  left_000.png / right_000.png
  L_000.png    / R_000.png
  cam0_000.png / cam1_000.png
  left000.png  / right000.png
  img_l_0.png  / img_r_0.png
  ... any pair where one name contains a left-keyword and the other right-keyword
"""

import cv2
import numpy as np
import os
import re
import sys
import yaml
import glob
from collections import defaultdict

IMAGES_DIR  = os.path.join(os.path.dirname(__file__), "images")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "stereo_calib.npz")

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)

PATTERNS = [
    (9, 6), (6, 9), (8, 6), (6, 8), (7, 6), (6, 7),
    (8, 5), (5, 8), (7, 5), (5, 7), (6, 5), (5, 6),
    (9, 7), (7, 9), (10, 7), (7, 10),
    (6, 4), (4, 6), (5, 4), (4, 5), (4, 3), (3, 4),
]

# Keywords that identify left / right images
LEFT_KEYS  = ("left", "l_", "_l", "cam0", "camera0", "img_l", "lft")
RIGHT_KEYS = ("right", "r_", "_r", "cam1", "camera1", "img_r", "rgt")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def enhance(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0)


def detect(gray: np.ndarray, rows: int, cols: int):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for img in (enhance(gray), gray):
        found, corners = cv2.findChessboardCorners(img, (rows, cols), FLAGS)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners
    return False, None


def auto_detect_pattern(gray: np.ndarray):
    for img in (enhance(gray), gray):
        for r, c in PATTERNS:
            found, _ = cv2.findChessboardCorners(img, (r, c), FLAGS)
            if found:
                return r, c
    return None, None


def _is_left(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in LEFT_KEYS)


def _is_right(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in RIGHT_KEYS)


def find_pairs(images_dir: str) -> list[tuple[str, str]]:
    """
    Scan the images directory and return sorted (left_path, right_path) pairs.
    Pairs are matched by extracting the numeric index from each filename.
    """
    all_imgs = sorted(
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    left_files  = [f for f in all_imgs if _is_left(os.path.basename(f))]
    right_files = [f for f in all_imgs if _is_right(os.path.basename(f))]

    if not left_files or not right_files:
        # Last resort: split all images in half (first half = left, second = right)
        mid = len(all_imgs) // 2
        left_files  = all_imgs[:mid]
        right_files = all_imgs[mid:]
        print("  Note: could not detect left/right keywords — splitting by order.")

    # Match by numeric index extracted from filename
    def get_index(path: str) -> str:
        nums = re.findall(r"\d+", os.path.basename(path))
        return nums[-1] if nums else os.path.basename(path)

    left_map  = {get_index(f): f for f in left_files}
    right_map = {get_index(f): f for f in right_files}

    common = sorted(set(left_map) & set(right_map))
    pairs  = [(left_map[k], right_map[k]) for k in common]

    # If no numeric overlap, pair by position
    if not pairs:
        pairs = list(zip(sorted(left_files), sorted(right_files)))

    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg         = load_config()
    rows        = cfg["chessboard"]["rows"]
    cols        = cfg["chessboard"]["cols"]
    square_size = cfg["chessboard"]["square_size"]

    print(f"Config: {rows}×{cols} inner corners, square = {square_size*100:.1f} cm")

    pairs = find_pairs(IMAGES_DIR)
    if len(pairs) < 5:
        print(f"ERROR: Need at least 5 image pairs, found {len(pairs)}.")
        print("       Run capture_pairs.py first.")
        sys.exit(1)

    print(f"Found {len(pairs)} image pairs in: {IMAGES_DIR}")
    print(f"Processing...\n")

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

    obj_points   = []
    img_points_l = []
    img_points_r = []
    img_size     = None
    valid        = 0

    for lf, rf in pairs:
        img_l  = cv2.imread(lf)
        img_r  = cv2.imread(rf)
        if img_l is None or img_r is None:
            print(f"  SKIP {os.path.basename(lf)} — could not read image")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray_l.shape[::-1]

        found_l, corners_l = detect(gray_l, rows, cols)
        found_r, corners_r = detect(gray_r, rows, cols)

        if not found_l or not found_r:
            # hint if a different pattern matches
            ar, ac = auto_detect_pattern(gray_l)
            hint = f" (auto-detected {ar}×{ac} — update cameras.yaml?)" if ar else ""
            missing = []
            if not found_l: missing.append("LEFT")
            if not found_r: missing.append("RIGHT")
            print(f"  SKIP {os.path.basename(lf):30s} — {', '.join(missing)} not detected{hint}")
            continue

        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)
        valid += 1
        print(f"  OK   {os.path.basename(lf):30s}   {os.path.basename(rf)}")

    print(f"\nValid pairs: {valid}/{len(pairs)}")
    if valid < 5:
        print("ERROR: Not enough valid pairs.")
        print("  • Make sure the board is flat and fully visible")
        print("  • Use good even lighting")
        print("  • Check rows/cols/square_size in cameras.yaml")
        sys.exit(1)

    # ── Per-camera calibration ────────────────────────────────────────────────
    print("\nCalibrating left camera...")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        obj_points, img_points_l, img_size, None, None)
    ok_l = "✅" if ret_l < 1.0 else "⚠️  > 1.0"
    print(f"  Left  RMS: {ret_l:.4f}  {ok_l}")

    print("Calibrating right camera...")
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        obj_points, img_points_r, img_size, None, None)
    ok_r = "✅" if ret_r < 1.0 else "⚠️  > 1.0"
    print(f"  Right RMS: {ret_r:.4f}  {ok_r}")

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
    ok_s = "✅" if ret_s < 1.0 else "⚠️  > 1.0"
    print(f"  Stereo RMS: {ret_s:.4f}  {ok_s}")

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
    print(f"\n{'='*50}")
    print(f"Calibration saved → {OUTPUT_PATH}")
    print(f"Focal length (fx) : {K_l[0,0]:.2f} px")
    print(f"Baseline          : {baseline:.4f} m  ({baseline*100:.1f} cm)")
    print(f"{'='*50}")

    if ret_s > 1.0:
        print("\n⚠️  RMS > 1.0 — tips to improve:")
        print("   • Board must be completely flat (glued to cardboard)")
        print("   • Even lighting, no glare on the board")
        print("   • Hold still when pressing SPACE")
        print("   • Vary angle and position in each shot")
        print("   • Verify square_size in cameras.yaml matches real measurement")
    else:
        print("\n✅ Calibration looks good — run:  python3 run.py")


if __name__ == "__main__":
    main()
