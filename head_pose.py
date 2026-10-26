"""
3D Head Pose Estimation Engine using solvePnP and Euler Angles.
Determines candidate's head orientation (Yaw, Pitch, Roll) and head direction.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


class HeadPoseEstimator:
    """
    Computes 3D head rotation angles (Yaw, Pitch, Roll) using facial keypoints.
    """

    # Generic 3D facial model points (in mm)
    MODEL_POINTS_3D = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye corner
        [225.0, 170.0, -135.0],   # Right eye corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0]   # Right mouth corner
    ], dtype=np.float64)

    def __init__(self, yaw_thresh: float = 22.0, pitch_thresh: float = 18.0) -> None:
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh

    def estimate_pose(
        self,
        image_points_2d: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Solves Perspective-n-Point to calculate Yaw, Pitch, Roll.

        image_points_2d must be an array of shape (6, 2):
        [nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner]
        """
        h, w = frame_shape[:2]

        # Camera intrinsic matrix approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "direction": "FORWARD",
                "is_deviated": False,
                "rvec": None,
                "tvec": None
            }

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose projection matrix to Euler angles
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        # Normalize angles
        if pitch > 180:
            pitch -= 360
        if yaw > 180:
            yaw -= 360
        if roll > 180:
            roll -= 360

        # Determine directional deviation
        direction = "FORWARD"
        is_deviated = False

        if yaw < -self.yaw_thresh:
            direction = "LOOKING RIGHT"
            is_deviated = True
        elif yaw > self.yaw_thresh:
            direction = "LOOKING LEFT"
            is_deviated = True
        elif pitch < -self.pitch_thresh:
            direction = "LOOKING DOWN (POSSIBLE CHEAT SHEET)"
            is_deviated = True
        elif pitch > self.pitch_thresh + 10.0:
            direction = "LOOKING UP"
            is_deviated = True

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "direction": direction,
            "is_deviated": is_deviated,
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix
        }

    def draw_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        nose_pt: Tuple[int, int]
    ) -> np.ndarray:
        """
        Projects 3D Cartesian coordinates onto the face to show head gaze vector.
        """
        if rvec is None or tvec is None or camera_matrix is None:
            return frame

        axis_length = 80.0
        axis_points_3d = np.array([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        proj_pts, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)

        p_nose = (int(nose_pt[0]), int(nose_pt[1]))
        p_x = (int(proj_pts[0][0][0]), int(proj_pts[0][0][1]))
        p_y = (int(proj_pts[1][0][0]), int(proj_pts[1][0][1]))
        p_z = (int(proj_pts[2][0][0]), int(proj_pts[2][0][1]))

        # X-axis (Pitch) - Red
        cv2.line(frame, p_nose, p_x, (0, 0, 255), 2)
        # Y-axis (Yaw) - Green
        cv2.line(frame, p_nose, p_y, (0, 255, 0), 2)
        # Z-axis (Roll / Forward vector) - Cyan
        cv2.line(frame, p_nose, p_z, (255, 255, 0), 2)

        return frame
