import cv2
import numpy as np


class KalmanTracker:
    """
    6D constant-velocity Kalman filter: state = [x, y, z, vx, vy, vz].
    Measurement: [x, y, z] (3D position).
    """

    def __init__(self, track_id: int, initial_pos: tuple[float, float, float]):
        self.track_id = track_id
        self._kf = cv2.KalmanFilter(6, 3)

        dt = 1.0  # updated each frame via set_dt()

        # Transition: constant-velocity model
        self._kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=np.float32)

        # Measurement: observe x, y, z
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        self._kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)

        x, y, z = initial_pos
        self._kf.statePost = np.array([[x], [y], [z], [0], [0], [0]], dtype=np.float32)

        self.last_seen = 0   # frames since last detection

    def set_dt(self, dt: float):
        """Update transition matrix time step."""
        self._kf.transitionMatrix[0, 3] = dt
        self._kf.transitionMatrix[1, 4] = dt
        self._kf.transitionMatrix[2, 5] = dt

    def predict(self) -> np.ndarray:
        """Return predicted state [x, y, z, vx, vy, vz]."""
        return self._kf.predict().flatten()

    def update(self, measurement: tuple[float, float, float]) -> np.ndarray:
        """Correct with observed [x, y, z]. Returns corrected state."""
        meas = np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32)
        return self._kf.correct(meas).flatten()

    @property
    def state(self) -> np.ndarray:
        return self._kf.statePost.flatten()

    @property
    def position(self) -> tuple[float, float, float]:
        s = self.state
        return float(s[0]), float(s[1]), float(s[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        s = self.state
        return float(s[3]), float(s[4]), float(s[5])
