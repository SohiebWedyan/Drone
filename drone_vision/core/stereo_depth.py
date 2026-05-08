import cv2
import numpy as np
import os


class StereoDepth:
    """
    Loads stereo calibration and computes per-pixel depth from a rectified pair.
    Falls back to object-height depth estimation when calibration is unavailable.
    """

    # SGBM parameters tuned for 640x480 indoor/outdoor balloon tracking
    _SGBM_MIN_DISP = 0
    _SGBM_NUM_DISP = 96     # must be divisible by 16
    _SGBM_BLOCK_SIZE = 7
    _SGBM_P1 = 8 * 3 * 7 ** 2
    _SGBM_P2 = 32 * 3 * 7 ** 2

    def __init__(self, calib_path: str | None = None):
        self.calibrated = False
        self.map1_l = self.map2_l = None
        self.map1_r = self.map2_r = None
        self.baseline = 0.06
        self.fx = 800.0
        self._sgbm = None

        if calib_path and os.path.isfile(calib_path):
            self._load(calib_path)

    def _load(self, path: str):
        data = np.load(path)
        self.map1_l = data["map1_l"]
        self.map2_l = data["map2_l"]
        self.map1_r = data["map1_r"]
        self.map2_r = data["map2_r"]
        T = data["T"]
        K_l = data["K_l"]
        self.baseline = float(abs(T[0, 0]))
        self.fx = float(K_l[0, 0])
        self._sgbm = cv2.StereoSGBM_create(
            minDisparity=self._SGBM_MIN_DISP,
            numDisparities=self._SGBM_NUM_DISP,
            blockSize=self._SGBM_BLOCK_SIZE,
            P1=self._SGBM_P1,
            P2=self._SGBM_P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.calibrated = True

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply undistort+rectify maps. Returns original frames if not calibrated."""
        if not self.calibrated:
            return left, right
        left_rect = cv2.remap(left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray | None:
        """Returns float32 disparity map (pixels). None if not calibrated."""
        if not self.calibrated:
            return None
        gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp = self._sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
        # Mask invalid disparities
        disp[disp <= 0] = np.nan
        return disp

    def depth_at(self, disparity: np.ndarray, x: int, y: int) -> float | None:
        """
        Return depth in meters at image coordinate (x, y) using Z = (baseline * fx) / d.
        Returns None if disparity is invalid at that point.
        """
        if disparity is None:
            return None
        h, w = disparity.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        # Sample a small window to reduce noise
        r = 3
        patch = disparity[max(0, y-r):y+r+1, max(0, x-r):x+r+1]
        valid = patch[~np.isnan(patch)]
        if len(valid) == 0:
            return None
        d = float(np.median(valid))
        if d <= 0:
            return None
        return (self.baseline * self.fx) / d

    @staticmethod
    def depth_from_height(bbox_h_px: float, real_height_m: float = 0.3, focal_length_px: float = 800.0) -> float:
        """Fallback single-camera depth estimate from known object height."""
        if bbox_h_px <= 0:
            return 0.0
        return (real_height_m * focal_length_px) / bbox_h_px
