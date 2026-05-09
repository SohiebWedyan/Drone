"""
Chessboard Diagnostic Tool
==========================
Automatically detects the inner-corner dimensions of your chessboard
and tells you exactly what to put in cameras.yaml.

Usage:
    python find_chessboard.py              # uses webcam 0
    python find_chessboard.py --cam 1     # different camera
    python find_chessboard.py --image chessboard.jpg   # from an image file

Controls:
    ESC / Q  — quit
    S        — save current frame for manual inspection
"""

import argparse
import sys
import cv2
import numpy as np

# Every common inner-corner pattern to try (rows, cols)
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

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_FAST_CHECK
)


def try_detect(gray):
    """Try every pattern. Return (rows, cols, corners) for the first match."""
    for rows, cols in PATTERNS:
        found, corners = cv2.findChessboardCorners(gray, (rows, cols), FLAGS)
        if found:
            return rows, cols, corners
    return None, None, None


def enhance(gray):
    """Improve contrast for detection."""
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def run_camera(cam_idx):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {cam_idx}")
        sys.exit(1)

    print(f"Camera {cam_idx} opened. Trying {len(PATTERNS)} chessboard patterns…")
    print("Hold the chessboard in front of the camera.")
    print("Controls:  ESC/Q = quit  |  S = save frame")

    detected_pattern = None
    save_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = enhance(gray)

        rows, cols, corners = try_detect(enhanced)

        display = frame.copy()
        if rows is not None:
            detected_pattern = (rows, cols)
            cv2.drawChessboardCorners(display, (rows, cols), corners, True)
            msg = f"DETECTED: {rows} x {cols} inner corners"
            cv2.putText(display, msg, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(display, f"Put  rows: {rows}  cols: {cols}  in cameras.yaml",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            print(f"\n  Pattern found: rows={rows}, cols={cols}")
            print(f"  Update cameras.yaml:")
            print(f"    chessboard:")
            print(f"      rows: {rows}")
            print(f"      cols: {cols}")
        else:
            cv2.putText(display, "No chessboard detected — check lighting & distance",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(display, "Trying 20+ common patterns automatically",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 100, 255), 2)

        # Show enhanced grayscale in corner (useful for debugging)
        thumb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        h, w = display.shape[:2]
        th, tw = h // 4, w // 4
        thumb = cv2.resize(thumb, (tw, th))
        display[h - th:, w - tw:] = thumb
        cv2.putText(display, "enhanced", (w - tw + 4, h - th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        cv2.imshow("Chessboard Finder  (ESC=quit  S=save)", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            fname = f"debug_frame_{save_idx:03d}.png"
            cv2.imwrite(fname, frame)
            cv2.imwrite(f"debug_enhanced_{save_idx:03d}.png", enhanced)
            print(f"Saved {fname} and debug_enhanced_{save_idx:03d}.png")
            save_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if detected_pattern:
        r, c = detected_pattern
        print(f"\nResult — add this to cameras.yaml:")
        print(f"  chessboard:")
        print(f"    rows: {r}")
        print(f"    cols: {c}")
        print(f"    square_size: 0.025   # measure your square in meters and update this")
    else:
        print("\nNo pattern detected. Tips:")
        print("  1. Use a flat, printed chessboard (not digital screen)")
        print("  2. Ensure good, even lighting — avoid glare")
        print("  3. Hold the board 30–70 cm from the camera")
        print("  4. Make sure the ENTIRE board is visible in the frame")
        print("  5. Save a frame with S and inspect it")


def run_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"ERROR: Cannot read image: {path}")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = enhance(gray)

    print(f"Testing image: {path}")
    rows, cols, corners = try_detect(enhanced)
    if rows is None:
        rows, cols, corners = try_detect(gray)  # try without enhancement

    display = frame.copy()
    if rows is not None:
        cv2.drawChessboardCorners(display, (rows, cols), corners, True)
        print(f"\nPattern found: rows={rows}, cols={cols}")
        print(f"Update cameras.yaml:")
        print(f"  chessboard:")
        print(f"    rows: {rows}")
        print(f"    cols: {cols}")
    else:
        print("No chessboard pattern detected in image.")
        print("Patterns tried:", PATTERNS)

    cv2.imshow("Result (any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Auto-detect chessboard dimensions")
    ap.add_argument("--cam",   type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--image", default=None,        help="Test a saved image instead of camera")
    args = ap.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_camera(args.cam)


if __name__ == "__main__":
    main()
